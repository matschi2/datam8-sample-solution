"""
Module to prepare payload to be rendered by DataM8.

* each payload function is processed in a separate thread
* each payload instance of a function is rendered in a async manor
* a payloads need to implement the IPayload protocol which just requires two functions
    - get_data() -> object // object available as `data` in template
    - get_output_path() -> Path // path where the rendered template gets saved to
* payloads can be split across multiple files and import via their filename/path within the
    `__modules` directory
* the cache can be used to carry over generated value or similar across payloads
    - it would technically also be possible to create a dummy payload that returns an empty list
        and simply adds some values to the cache

Some notes/explanations to lean into the way the payloads/templates are rendered and make debugging
simpler.

* payloads only "gath" entities to be rendered and do some slight initialization
* payloads contain references to the model, wrapper and any additional entity to allow for further
    lookups
* most logic is implemented on the payload itself as a function or property, so that it gets
    executed within the asynchronous call
* a search locator used in `get_many` and `get_many_where` needs to end on with a "/", otherwise
    DataM8 will look for an entity with that exact locator
* use match statements for structural pattern matching to avoid inreadable chains of if-elif-else
    blocks
* prefer returning basic types or objects of classes defined in the payload itself, to narrow
    potential errors to the payload itself and not errors thrown by DataM8 itself or be affected
    by changes within DataM8 or its model
* retrieving an entity by its locator via `get()` is faster than using `get_where()` with a filter
* all functions return an EntityWrapper instance, with an entity attribute containing the content of
    the corresponding json file

"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from datam8 import logging
from datam8.generate import BasePayload, IPayload, register_payload
from datam8.model import EntityWrapper, Model
from datam8.utils.cache import Cache
from datam8_model.attribute import Attribute, HistoryType
from datam8_model.data_type import DataType, DataTypeDefinition
from datam8_model.folder import Folder
from datam8_model.model import ExternalModelSource, ModelEntity
from datam8_model.zone import Zone

logger = logging.getLogger(__name__)

TARGET = "databricks"


def create_full_table_name(wrapper: EntityWrapper[ModelEntity]) -> str:
    # create copy of folders list without the zone
    name_parts = [f for f in wrapper.locator.folders][1:]
    name_parts.append(wrapper.entity.name)
    return "_".join(name_parts)


def create_resource_slug_from_name(name: str) -> str:
    resource_slug = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").lower()
    if len(resource_slug) == 0:
        return "default"
    return resource_slug


@register_payload("ddl_notebook.jinja2")
def ddl_notebooks(model: Model, cache: Cache) -> Sequence[DdlPayload]:
    wrappers = model.modelEntities.get_many("010-Stage/")
    wrappers.extend(model.modelEntities.get_many("020-Core/"))
    wrappers.extend(model.modelEntities.get_many("030-Curated/"))

    return [DdlPayload(wrapper, model) for wrapper in wrappers]


@register_payload("ddl_notebook.jinja2")
def ddl_raw_notebooks(model: Model, cache: Cache) -> Sequence[DdlPayload]:
    "Render stage entities again but using their sources for raw"
    wrappers = model.modelEntities.get_man("010-Stage/")

    return [
        # create one payload per source
        DdlRawPayload(wrapper, model, source)
        for wrapper in wrappers
        for source in wrapper.entity.sources
        if type(source) is ExternalModelSource
    ]


@register_payload("schema.yml.jinja2")
def dab_schemas(model: Model, cache: Cache) -> Sequence[IPayload]:
    """Emit Databricks bundle schema definitions for every configured zone."""
    payload = [
        BasePayload(
            data=[
                {
                    "resource_key": create_resource_slug_from_name(zone.entity.name),
                    "name": zone.entity.name,
                    "comment": zone.entity.displayName,
                }
                for zone in model.zones.values()
            ],
            output_path=Path("schemas", "schemas.yml"),
        )
    ]
    return payload


@register_payload("clusters.yml.jinja2")
def dab_cluster(model: Model, cache: Cache) -> Sequence[IPayload]:
    clusters = [wrapper.entity for wrapper in model.propertyValues.get_many("cluster/")]

    if len(clusters) == 0:
        return []

    payloads = [
        BasePayload(
            data=[
                {
                    "name": cluster.name,
                    "display_name": cluster.displayName or cluster.name,
                    "node_type": getattr(cluster, "node_type", "Standard_D4ds_v5"),
                    "num_workers": getattr(cluster, "num_workers", None),
                    "workload_type": getattr(cluster, "workload_type", "job"),
                    "spark_version": getattr(cluster, "spark_version", "16.4.x-scala2.12"),
                    "autotermination_minutes": getattr(cluster, "autotermination_minutes", 60),
                    "data_security_mode": getattr(
                        cluster, "data_security_mode", "DATA_SECURITY_MODE_DEDICATED"
                    ),
                    "runtime_engine": getattr(cluster, "runtime_engine", "STANDARD"),
                    "variable_name": f"cluster_{create_resource_slug_from_name(cluster.name)}",
                    "is_default": cluster.default or False,
                    "custom_tags": {},
                }
                for cluster in clusters
            ],
            output_path=Path("clusters.yml"),
        )
    ]
    return payloads


class DdlColumn:
    """
    Payload class for column definitions in ddl notebooks
    """

    def __init__(self, attr: Attribute, model: Model) -> None:
        self.attribute: Attribute = attr
        self.model: Model = model

    @property
    def type_definition(self) -> DataTypeDefinition:
        return self.model.dataTypes.get(self.attribute.dataType.type).entity

    @property
    def target_type(self) -> str:
        target_type = self.type_definition.targets[TARGET]
        char_length = self.attribute.dataType.charLen

        match [
            self.attribute.dataType.precision,
            self.attribute.dataType.scale,
            self.attribute.dataType.charLen,
        ]:
            case [int() as precision, None, None]:
                target_type += f"({precision})"
            case [int() as precision, int() as scale, None]:
                target_type += f"({precision}, {scale})"
            case [None, None, int() as char_length]:
                target_type += f"({char_length})"
            case [None, None, None]:  # valid case
                pass
            case _:  # invalid cases
                logger.warning("Invalid combination of precision, scale and charLen")

        return target_type

    @property
    def spark_data_type_expression(self) -> str:
        return f"{self.target_type}".upper()

    @property
    def spark_nullable(self) -> str:
        if self.attribute.dataType.nullable:
            return " NULL"

        return " NOT NULL"

    @property
    def spark_comment(self) -> str:
        if self.attribute.description is None:
            return ""

        return f" COMMENT '{self.attribute.description}'"

    @property
    def is_surrogate_key(self) -> bool:
        if self.attribute.properties is None:
            return False

        for pv in self.attribute.properties:
            if pv.property == "attribute_type" and pv.value == "SK":
                return True

        return False

    @property
    def spark_identity(self) -> str:
        if self.is_surrogate_key:
            return " GENERATED ALWAYS AS IDENTITY"

        return ""


class DdlRawColumn(DdlColumn):
    """
    Raw column definitions are mostly the same as their "normal" counterpart, except for the type as
    Create a fictive DataTypeDefinition which simply mirrors the type of the model which will be
    the source data type.
    """

    @property
    def type_definition(self) -> DataTypeDefinition:
        """
        In case of raw columns (source) assume the data type in the attribute is already the correct one
        """
        return DataTypeDefinition(
            name=self.attribute.dataType.type,
            targets={
                TARGET: self.attribute.dataType.type,
            },
        )


class DdlPayload(BasePayload):
    imports: list[str] = ["DataType", "StructField", "StructType"]

    def __init__(self, wrapper: EntityWrapper[ModelEntity], model: Model) -> None:
        self.model: Model = model
        self.wrapper: EntityWrapper[ModelEntity] = wrapper
        self.locator = wrapper.locator
        self.entity: ModelEntity = wrapper.entity
        self.zone: EntityWrapper[Zone] = model.get_zone_for_entity(wrapper)
        self.first_folder: EntityWrapper[Folder] = model.folders.get(
            "/".join(self.locator.folders[0:2])
        )

    @property
    def zone_name(self) -> str:
        return self.zone.entity.name

    def get_data(self) -> object:
        return self

    def get_output_path(self) -> Path:
        return Path(
            "ddl", *self.locator.folders, f"{self.entity.name or self.locator.entityName}.py"
        )

    @property
    def data_product(self) -> str | None:
        return self.wrapper.locator.folders[1]

    @property
    def data_module(self) -> str | None:
        return self.wrapper.locator.folders[2]

    @property
    def full_table_name(self) -> str:
        return create_full_table_name(self.wrapper)

    @property
    def table_comment(self) -> str:
        return self.entity.description or ""

    @property
    def columns(self) -> Sequence[DdlColumn]:
        return [DdlColumn(attr, self.model) for attr in self.entity.attributes]

    @property
    def has_scd2_history(self) -> bool:
        return any([attr.history == HistoryType.SCD2 for attr in self.entity.attributes])

    @property
    def partitions(self) -> list[str]:
        return [attribute.name for attribute in self.entity.attributes if attribute.isBusinessKey]

    @property
    def table_properties(self) -> dict[str, Any]:
        # TODO: get table properties from tags?
        table_properties_: dict[str, Any] = {}

        for pv in self.wrapper.properties.values():
            match [pv.property, pv.name]:
                # column mapping mode
                case ["column_mapping_mode" as prop, str() as name]:
                    table_properties_["delta.columnMapping.mode"] = name

                # type widening
                case ["enable_type_widening" as prop, "true"]:
                    table_properties_["delta.enableTypeWidening"] = True
                case ["enable_type_widening" as prop, "true"]:
                    table_properties_["delta.enableTypeWidening"] = False

                # data retention
                case ["data_retention" as prop, str() as name]:
                    interval_value = name.replace("_", " ")

                    if not interval_value.lower().startswith("interval"):
                        interval_value = f"interval {interval_value}"

                    table_properties_["delta.logRetentionDuration"] = interval_value
                    table_properties_["delta.deletedFileRetentionDuration"] = interval_value

                # invalid combinations
                case [
                    "enable_type_widening" | "column_mapping_mode" | "enable_type_widening" as prop,
                    _ as rest,
                ]:
                    raise Exception(f"Invalid value for {prop}: {rest}")

                case _:
                    # ignore other properties
                    pass

        return table_properties_

    @property
    def column_tags(self) -> list[tuple[str, dict[str, str]]]:
        tags_ = [
            (
                attr.name,
                {ref.property: ref.value for ref in attr.properties or []},
            )
            for attr in self.entity.attributes
            if len(attr.properties or []) > 0
        ]
        return tags_

    @property
    def refactored_columns(self) -> list[dict[str, str | Sequence[str]]]:
        refactors = [
            {
                "name": attr.name,
                "refactorNames": attr.refactorNames,
            }
            for attr in self.entity.attributes
            if attr.refactorNames is not None and len(attr.refactorNames) > 0
        ]
        return refactors

    @property
    def foreign_keys(self) -> list[DdlForeignKey]:
        constraints: list[DdlForeignKey] = []

        for rel in self.entity.relationships:
            remote_entity = self.model.modelEntities.get_by_id(rel.targetLocation)
            remote_table_name = create_full_table_name(remote_entity)
            remote_zone = self.model.get_zone_for_entity(remote_entity).entity.name
            constraints.append(
                DdlForeignKey(
                    table=self.full_table_name,
                    columns=[a.sourceName for a in rel.attributes],
                    remote_columns=[a.targetName for a in rel.attributes],
                    remote_table=remote_table_name,
                    remote_zone=remote_zone,
                )
            )

        return constraints

    @property
    def primary_key_attributes(self) -> list[Attribute]:
        return [attr for attr in self.entity.attributes if attr.isBusinessKey]


class DdlRawPayload(DdlPayload):
    """
    Describes a payload for a single stage-source combination, which are defined during
    initialization.
    """

    partitions = ["__Year", "__Month", "__Day", "__InsertTImestampUTC"]

    def __init__(
        self, wrapper: EntityWrapper[ModelEntity], model: Model, source: ExternalModelSource
    ) -> None:
        super().__init__(wrapper, model)

        self.source = source
        self.zone: EntityWrapper[Zone] = model.zones["raw"]
        self.base_columns: list[DdlRawColumn] = [
            DdlRawColumn(
                Attribute(
                    ordinalNumber=1000,
                    name=col,
                    dataType=DataType(type=type_, nullable=False),
                    dateAdded=datetime.now(UTC),
                    attributeType="",
                ),
                self.model,
            )
            for col, type_ in [
                ("__Year", "short"),
                ("__Month", "short"),
                ("__Day", "short"),
                ("__InsertTImestampUTC", "datetime"),
            ]
        ]

    def get_output_path(self) -> Path:
        return Path(
            "ddl",
            "raw",
            *self.locator.folders[1:],
            f"{self.entity.name or self.locator.entityName}.py",
        )

    @property
    def columns(self) -> list[DdlRawColumn]:
        assert self.source.mapping, "External source should have source mappings"
        data_source = self.model.dataSources.get(self.source.dataSource).entity
        data_source_type = self.model.dataSourceTypes.get(data_source.type).entity

        data_type_mappings = {
            dtm.sourceType: dtm.targetType for dtm in data_source_type.dataTypeMapping or []
        }
        data_type_mappings.update(
            {dtm.sourceType: dtm.targetType for dtm in data_source.dataTypeMapping or {}}
        )

        column_types: dict[str, DataType] = {}

        for sc in self.source.mapping:
            if sc.sourceDataType is not None and sc.sourceDataType.type in data_type_mappings:
                column_types[sc.sourceName] = sc.sourceDataType
                column_types[sc.sourceName].type = data_type_mappings[sc.sourceDataType.type]
            elif sc.sourceDataType is not None:
                column_types[sc.sourceName] = sc.sourceDataType
            else:
                raise Exception(f"Could not get match source type of {sc} in {self.entity.name}")

        cols_ = [
            DdlRawColumn(
                Attribute(
                    ordinalNumber=1,  # not used in template
                    attributeType="",  # not used in template
                    name=source_column_mapping.sourceName,
                    dataType=column_types[source_column_mapping.sourceName],
                    dateAdded=datetime.now(UTC),
                ),
                self.model,
            )
            for source_column_mapping in self.source.mapping
        ]

        return self.base_columns + cols_


@dataclasses.dataclass
class DdlForeignKey:
    table: str
    columns: list[str]
    remote_columns: list[str]
    remote_table: str
    remote_zone: str

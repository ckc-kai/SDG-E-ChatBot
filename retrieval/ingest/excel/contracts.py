"""Reviewed ingest and query contracts for the cleaned SDG&E QDR CSVs.

The contract is the single place that decides what a cleaned column means.
Loading is deliberately strict: a column that no bucket claims is reported as
unknown and is not persisted, so a regenerated cleaned file cannot silently
introduce an unreviewed (possibly sensitive) field into retrieval.
"""

from __future__ import annotations

import functools
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONTRACT_PATH = (
    Path(__file__).resolve().parents[3] / "config" / "excel_contracts.yaml"
)
CONTRACT_PATH_ENV = "SDGE_EXCEL_CONTRACTS_PATH"

FAMILIES = {"long_metric", "wide_risk", "entity"}
TABLE_KEY_RE = re.compile(r"^table_(\d{2})$")
FILE_TABLE_RE = re.compile(r"^sdge_table0*(\d+)", re.IGNORECASE)

_LEADING_ORDINAL_RE = re.compile(r"^\s*\d+[.)]\s*")
_NON_SLUG_RE = re.compile(r"[^a-z0-9]+")


class ContractError(RuntimeError):
    """Raised when a contract is malformed or a file violates its contract."""


def semantic_key(label: str) -> str:
    """Normalize a human metric label into a stable semantic key.

    Leading ordinals such as ``"1. Overhead circuit miles"`` are dropped so the
    key stays stable when Energy Safety renumbers a metric list.
    """
    text = _LEADING_ORDINAL_RE.sub("", (label or "").strip()).lower()
    slug = _NON_SLUG_RE.sub("_", text).strip("_")
    return slug[:80] or "unspecified"


@dataclass(frozen=True)
class TableContract:
    table_number: int
    key: str
    family: str
    title: str
    description: str
    identity_column: str | None
    identity_synthesize: tuple[str, ...]
    series_id: str | None
    source_metric_number: str | None
    semantic_fields: tuple[str, ...]
    semantic_category: str | None
    semantic_definition: str | None
    semantic_purpose: str | None
    semantic_assumptions: str | None
    value_column: str | None
    unit_column: str | None
    melt_columns: tuple[str, ...]
    period: dict[str, str]
    promoted: tuple[str, ...]
    json_dimensions: tuple[str, ...]
    annotations: tuple[str, ...]
    provenance: tuple[str, ...]
    entity: dict[str, str]
    searchable_text: tuple[str, ...]
    excluded: tuple[str, ...]
    ignored: tuple[str, ...]
    dedupe: dict[str, Any] = field(default_factory=dict)
    utility_column: str | None = "utility_id"

    @property
    def claimed_columns(self) -> set[str]:
        """Every column the contract knows about, in any bucket."""
        claimed: set[str] = set()
        if self.identity_column:
            claimed.add(self.identity_column)
        claimed.update(self.identity_synthesize)
        for single in (
            self.series_id,
            self.source_metric_number,
            self.value_column,
            self.unit_column,
            self.utility_column,
            self.semantic_category,
            self.semantic_definition,
            self.semantic_purpose,
            self.semantic_assumptions,
        ):
            if single:
                claimed.add(single)
        claimed.update(self.semantic_fields)
        claimed.update(self.melt_columns)
        claimed.update(self.period.values())
        claimed.update(self.promoted)
        claimed.update(self.json_dimensions)
        claimed.update(self.annotations)
        claimed.update(self.provenance)
        claimed.update(v for v in self.entity.values() if v)
        claimed.update(self.searchable_text)
        claimed.update(self.excluded)
        claimed.update(self.ignored)
        return claimed

    @property
    def required_columns(self) -> set[str]:
        """Columns whose absence invalidates the revision."""
        required: set[str] = set()
        if self.identity_column:
            required.add(self.identity_column)
        required.update(self.identity_synthesize)
        if self.family == "long_metric":
            if self.value_column:
                required.add(self.value_column)
            required.update(self.semantic_fields)
        elif self.family == "wide_risk":
            required.update(self.melt_columns)
        elif self.family == "entity":
            for role in ("key", "title"):
                if self.entity.get(role):
                    required.add(self.entity[role])
        year = self.period.get("year")
        if year:
            required.add(year)
        return required


@dataclass(frozen=True)
class ContractSet:
    version: str
    tables: dict[int, TableContract]
    measure_labels: dict[str, str]

    def for_table(self, table_number: int) -> TableContract:
        try:
            return self.tables[table_number]
        except KeyError as exc:
            raise ContractError(
                f"No reviewed contract for table {table_number}. Add one to "
                "config/excel_contracts.yaml before ingesting this file."
            ) from exc

    def measure_label(self, measure_name: str) -> str:
        return self.measure_labels.get(measure_name, measure_name.replace("_", " "))


def table_number_from_filename(name: str) -> int:
    match = FILE_TABLE_RE.match(Path(name).name)
    if not match:
        raise ContractError(
            f"Filename does not begin with 'sdge_table<number>': {name}"
        )
    return int(match.group(1))


def _tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(str(item) for item in value)


def _build_table(key: str, raw: dict[str, Any], defaults: dict[str, Any]) -> TableContract:
    match = TABLE_KEY_RE.match(key)
    if not match:
        raise ContractError(f"Contract key must look like 'table_07', got {key!r}")
    family = raw.get("family")
    if family not in FAMILIES:
        raise ContractError(f"{key}: family must be one of {sorted(FAMILIES)}")

    identity = raw.get("identity") or {}
    if not identity.get("column") and not identity.get("synthesize"):
        raise ContractError(f"{key}: identity needs 'column' or 'synthesize'")

    semantic = raw.get("semantic_group") or {}
    measure = raw.get("measure") or {}
    melt = (raw.get("melt") or {}).get("measure_columns")

    if family == "long_metric":
        if not measure.get("value"):
            raise ContractError(f"{key}: long_metric requires measure.value")
        if not semantic.get("fields"):
            raise ContractError(f"{key}: long_metric requires semantic_group.fields")
    if family == "wide_risk" and not melt:
        raise ContractError(f"{key}: wide_risk requires melt.measure_columns")
    if family == "entity" and not (raw.get("entity") or {}).get("type"):
        raise ContractError(f"{key}: entity requires entity.type")

    contract = TableContract(
        table_number=int(match.group(1)),
        key=key,
        family=family,
        title=raw.get("title", key),
        description=" ".join((raw.get("description") or "").split()),
        identity_column=identity.get("column"),
        identity_synthesize=_tuple(identity.get("synthesize")),
        series_id=raw.get("series_id"),
        source_metric_number=raw.get("source_metric_number"),
        semantic_fields=_tuple(semantic.get("fields")),
        semantic_category=semantic.get("category"),
        semantic_definition=semantic.get("definition"),
        semantic_purpose=semantic.get("purpose"),
        semantic_assumptions=semantic.get("assumptions"),
        value_column=measure.get("value"),
        unit_column=measure.get("unit"),
        melt_columns=_tuple(melt),
        period={k: v for k, v in (raw.get("period") or {}).items() if v},
        promoted=_tuple(raw.get("promoted")),
        json_dimensions=_tuple(raw.get("json")),
        annotations=_tuple(raw.get("annotations", defaults.get("annotations"))),
        provenance=_tuple(raw.get("provenance")),
        entity=dict(raw.get("entity") or {}),
        searchable_text=_tuple(raw.get("searchable_text")),
        excluded=_tuple(raw.get("excluded")),
        ignored=_tuple(raw.get("ignored")),
        dedupe=dict(raw.get("dedupe") or {}),
        utility_column=raw.get("utility", defaults.get("utility", "utility_id")),
    )

    # Goal 7 is only meaningful if an excluded column cannot also be claimed by
    # a bucket that persists it. Catch that contradiction at load time.
    persisted = contract.claimed_columns - set(contract.excluded) - set(contract.ignored)
    leaked = sorted(set(contract.excluded) & persisted)
    if leaked:
        raise ContractError(
            f"{key}: columns are marked excluded but also claimed by a "
            f"persisted bucket: {', '.join(leaked)}"
        )
    return contract


@functools.lru_cache(maxsize=1)
def load_contracts() -> ContractSet:
    path = Path(os.environ.get(CONTRACT_PATH_ENV, DEFAULT_CONTRACT_PATH))
    try:
        with path.open() as fh:
            raw = yaml.safe_load(fh) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ContractError(f"Cannot load Excel contracts from {path}") from exc

    defaults = raw.get("defaults") or {}
    tables: dict[int, TableContract] = {}
    for key, table_raw in (raw.get("tables") or {}).items():
        contract = _build_table(key, table_raw, defaults)
        if contract.table_number in tables:
            raise ContractError(f"Duplicate contract for table {contract.table_number}")
        tables[contract.table_number] = contract

    if not tables:
        raise ContractError(f"No table contracts found in {path}")

    return ContractSet(
        version=raw.get("version", "unversioned"),
        tables=tables,
        measure_labels=dict(raw.get("measure_labels") or {}),
    )


@dataclass(frozen=True)
class ColumnInventory:
    """Result of checking one cleaned file against its contract."""

    present: tuple[str, ...]
    unknown: tuple[str, ...]
    missing_required: tuple[str, ...]
    missing_optional: tuple[str, ...]
    excluded_present: tuple[str, ...]

    def as_dict(self) -> dict[str, list[str]]:
        return {
            "present": list(self.present),
            "unknown": list(self.unknown),
            "missing_required": list(self.missing_required),
            "missing_optional": list(self.missing_optional),
            "excluded_present": list(self.excluded_present),
        }


def inventory_columns(
    contract: TableContract, headers: list[str]
) -> ColumnInventory:
    present = [h.strip() for h in headers]
    present_set = set(present)
    claimed = contract.claimed_columns
    required = contract.required_columns
    return ColumnInventory(
        present=tuple(present),
        unknown=tuple(sorted(present_set - claimed)),
        missing_required=tuple(sorted(required - present_set)),
        missing_optional=tuple(sorted((claimed - required) - present_set)),
        excluded_present=tuple(
            sorted(present_set & set(contract.excluded))
        ),
    )

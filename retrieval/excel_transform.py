"""Transform cleaned CSV rows into exact facts and entity records.

Numbers are parsed with :class:`decimal.Decimal` and stored in PostgreSQL
``numeric`` so monetary values, distances, and risk scores aggregate without
binary floating-point drift. The cleaned string is always retained in
``value_raw`` so a displayed number can be audited against the source file.
"""

from __future__ import annotations

import csv
import hashlib
import logging
import re
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterator

from retrieval.excel_contracts import ContractSet, TableContract, semantic_key

logger = logging.getLogger(__name__)

# Cleaned Table 13 carries very wide comment/provenance columns.
csv.field_size_limit(10**9)

_NUMBER_CLEAN_RE = re.compile(r"[,$\s]")
_DATE_FORMATS = ("%Y-%m-%d", "%m/%d/%Y", "%Y/%m/%d")


@dataclass(frozen=True)
class Fact:
    table_number: int
    record_id: str
    source_metric_number: str | None
    series_id: str | None
    semantic_metric_key: str
    metric_name: str
    measure_name: str
    utility_id: str | None
    reporting_year: int | None
    reporting_quarter: int | None
    source_vintage_year: int | None
    year_basis: str | None
    period_end_date: date | None
    hftd_tier: str | None
    line_type: str | None
    dimensions: dict[str, Any]
    unit: str | None
    value_numeric: Decimal | None
    value_raw: str | None
    value_text: str | None
    comments: str | None
    blank_meaning: str | None
    provenance: dict[str, Any]


@dataclass(frozen=True)
class Record:
    table_number: int
    record_id: str
    entity_key: str | None
    entity_type: str
    title: str | None
    utility_id: str | None
    reporting_year: int | None
    reporting_quarter: int | None
    hftd_tier: str | None
    line_type: str | None
    date_start: date | None
    date_due: date | None
    date_end: date | None
    status: str | None
    attributes: dict[str, Any]
    searchable_text: str | None
    provenance: dict[str, Any]


def clean(value: str | None) -> str | None:
    """Normalize whitespace; return None only for a genuinely empty cell.

    Literal ``NA`` / ``N/A`` strings are deliberately preserved. The cleaning
    scripts already applied their own reviewed NA normalization (recorded in
    statuses such as ``normalized_na_to_null``); anything they left in place is
    meaningful content. In particular ``blank_meaning = 'N/A'`` is the recorded
    reason a value is blank and must survive ingestion.
    """
    if value is None:
        return None
    text = " ".join(value.replace("\xa0", " ").split())
    return text or None


def parse_decimal(value: str | None) -> Decimal | None:
    text = clean(value)
    if text is None:
        return None
    candidate = _NUMBER_CLEAN_RE.sub("", text)
    if candidate.endswith("%"):
        candidate = candidate[:-1]
    if candidate in {"", "-", "+", "."}:
        return None
    try:
        return Decimal(candidate)
    except InvalidOperation:
        return None


def parse_int(value: str | None) -> int | None:
    number = parse_decimal(value)
    if number is None:
        return None
    try:
        return int(number)
    except (ValueError, OverflowError):
        return None


def parse_date(value: str | None) -> date | None:
    text = clean(value)
    if text is None:
        return None
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def _subset(row: dict[str, str], columns: tuple[str, ...]) -> dict[str, Any]:
    """Collect the non-empty declared columns into a JSON payload."""
    out: dict[str, Any] = {}
    for column in columns:
        value = clean(row.get(column))
        if value is not None:
            out[column] = value
    return out


def _identity(contract: TableContract, row: dict[str, str], ordinal: int) -> str:
    if contract.identity_column:
        value = clean(row.get(contract.identity_column))
        if value:
            return value
    if contract.identity_synthesize:
        parts = [clean(row.get(c)) or "" for c in contract.identity_synthesize]
        digest = hashlib.sha1("|".join(parts).encode()).hexdigest()[:16]
        return f"T{contract.table_number}R-{digest}"
    return f"T{contract.table_number}R-ROW{ordinal}"


def _semantic(contract: TableContract, row: dict[str, str]) -> tuple[str, str]:
    labels = [clean(row.get(c)) for c in contract.semantic_fields]
    label = " / ".join(part for part in labels if part) or "Unspecified"
    return semantic_key(label), label


def _period(contract: TableContract, row: dict[str, str]) -> dict[str, Any]:
    period = contract.period
    return {
        "reporting_year": parse_int(row.get(period.get("year", ""))),
        "reporting_quarter": parse_int(row.get(period.get("quarter", ""))),
        "source_vintage_year": parse_int(row.get(period.get("source_vintage", ""))),
        "year_basis": clean(row.get(period.get("basis", ""))),
        "period_end_date": parse_date(row.get(period.get("period_end_date", ""))),
    }


def _promoted(contract: TableContract, row: dict[str, str]) -> dict[str, Any]:
    promoted = set(contract.promoted)
    return {
        "hftd_tier": clean(row.get("hftd_tier")) if "hftd_tier" in promoted else None,
        "line_type": clean(row.get("line_type")) if "line_type" in promoted else None,
    }


def _annotations(contract: TableContract, row: dict[str, str]) -> tuple[str | None, str | None]:
    comments = None
    blank_meaning = None
    for column in contract.annotations:
        value = clean(row.get(column))
        if column == "blank_meaning":
            blank_meaning = value
        elif column == "comments":
            comments = value
    return comments, blank_meaning


def _semantic_context(contract: TableContract, row: dict[str, str]) -> dict[str, Any]:
    """Carry the concept's category and written definition onto the fact.

    These columns describe the metric concept rather than locating the source
    cell, so they belong with the dimensions the card builder reads. Without
    this they would be claimed by the contract but never persisted, and the
    concept cards would silently lose their definitions.
    """
    context: dict[str, Any] = {}
    for column in (
        contract.semantic_category,
        contract.semantic_definition,
        contract.semantic_purpose,
        contract.semantic_assumptions,
    ):
        if column:
            value = clean(row.get(column))
            if value is not None:
                context[column] = value
    return context


def _long_metric_fact(
    contract: TableContract, row: dict[str, str], ordinal: int
) -> Fact:
    key, label = _semantic(contract, row)
    raw_value = clean(row.get(contract.value_column or ""))
    number = parse_decimal(raw_value)
    comments, blank_meaning = _annotations(contract, row)
    promoted = _promoted(contract, row)
    return Fact(
        table_number=contract.table_number,
        record_id=_identity(contract, row, ordinal),
        source_metric_number=clean(row.get(contract.source_metric_number or "")),
        series_id=clean(row.get(contract.series_id or "")),
        semantic_metric_key=key,
        metric_name=label,
        measure_name="actual_value",
        utility_id=clean(row.get(contract.utility_column or "")),
        **_period(contract, row),
        hftd_tier=promoted["hftd_tier"],
        line_type=promoted["line_type"],
        dimensions={
            **_subset(row, contract.json_dimensions),
            **_semantic_context(contract, row),
        },
        unit=clean(row.get(contract.unit_column or "")),
        value_numeric=number,
        value_raw=raw_value,
        # A non-numeric but present value is kept verbatim rather than dropped.
        value_text=raw_value if number is None and raw_value is not None else None,
        comments=comments,
        blank_meaning=blank_meaning,
        provenance=_subset(row, contract.provenance),
    )


def _wide_risk_facts(
    contracts: ContractSet,
    contract: TableContract,
    row: dict[str, str],
    ordinal: int,
) -> Iterator[Fact]:
    """Melt one wide risk row into one fact per risk measure.

    Null measures are emitted, not skipped: the plan requires that a missing
    risk component stays visible with its ``blank_meaning`` rather than
    disappearing from the fact table.
    """
    record_id = _identity(contract, row, ordinal)
    period = _period(contract, row)
    promoted = _promoted(contract, row)
    comments, blank_meaning = _annotations(contract, row)
    dimensions = _subset(row, contract.json_dimensions)
    provenance = _subset(row, contract.provenance)
    utility = clean(row.get(contract.utility_column or ""))
    source_metric = clean(row.get(contract.source_metric_number or ""))

    for measure in contract.melt_columns:
        raw_value = clean(row.get(measure))
        number = parse_decimal(raw_value)
        yield Fact(
            table_number=contract.table_number,
            record_id=record_id,
            source_metric_number=source_metric,
            series_id=None,
            semantic_metric_key=measure,
            metric_name=contracts.measure_label(measure),
            measure_name=measure,
            utility_id=utility,
            **period,
            hftd_tier=promoted["hftd_tier"],
            line_type=promoted["line_type"],
            dimensions=dimensions,
            unit=None,
            value_numeric=number,
            value_raw=raw_value,
            value_text=raw_value if number is None and raw_value is not None else None,
            comments=comments,
            blank_meaning=blank_meaning,
            provenance=provenance,
        )


def _entity_record(
    contract: TableContract, row: dict[str, str], ordinal: int
) -> Record:
    entity = contract.entity
    period = _period(contract, row)
    promoted = _promoted(contract, row)
    text_parts = [clean(row.get(c)) for c in contract.searchable_text]
    searchable = " ".join(part for part in text_parts if part) or None
    attributes = _subset(row, contract.json_dimensions)
    comments, blank_meaning = _annotations(contract, row)
    if comments:
        attributes.setdefault("comments", comments)
    if blank_meaning:
        attributes.setdefault("blank_meaning", blank_meaning)
    return Record(
        table_number=contract.table_number,
        record_id=_identity(contract, row, ordinal),
        entity_key=clean(row.get(entity.get("key", ""))),
        entity_type=entity["type"],
        title=clean(row.get(entity.get("title", ""))),
        utility_id=clean(row.get(contract.utility_column or "")),
        reporting_year=period["reporting_year"],
        reporting_quarter=period["reporting_quarter"],
        hftd_tier=promoted["hftd_tier"],
        line_type=promoted["line_type"],
        date_start=parse_date(row.get(entity.get("date_start", ""))),
        date_due=parse_date(row.get(entity.get("date_due", ""))),
        date_end=parse_date(row.get(entity.get("date_end", ""))),
        status=clean(row.get(entity.get("status", ""))),
        attributes=attributes,
        searchable_text=searchable,
        provenance=_subset(row, contract.provenance),
    )


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Read a cleaned CSV, tolerating the UTF-8 BOM the cleaners write."""
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        headers = [h.strip() for h in (reader.fieldnames or [])]
        rows = [dict(row) for row in reader]
    return headers, rows


def transform(
    contracts: ContractSet,
    contract: TableContract,
    rows: list[dict[str, str]],
) -> tuple[list[Fact], list[Record]]:
    facts: list[Fact] = []
    records: list[Record] = []
    for ordinal, row in enumerate(rows, start=1):
        if contract.family == "long_metric":
            facts.append(_long_metric_fact(contract, row, ordinal))
        elif contract.family == "wide_risk":
            facts.extend(_wide_risk_facts(contracts, contract, row, ordinal))
        else:
            records.append(_entity_record(contract, row, ordinal))
    return facts, records

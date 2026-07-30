"""Deterministic semantic-card builder for the Excel corpus.

Cards are retrieval aids, never numeric sources of truth. Their text is
generated from reviewed contract metadata plus profiled vocabulary from the
staged revision; no LLM authors card content. Every number an answer states
must still come from ``excel_facts`` / ``excel_records``.
"""

from __future__ import annotations

import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Iterable, Sequence

from retrieval.excel_contracts import ContractSet, TableContract
from retrieval.excel_transform import Fact, Record

logger = logging.getLogger(__name__)

CARD_BUILDER_VERSION = "excel-card-v1"
BREADCRUMB_ROOT = "SDG&E QDR Tables"

# Guardrails from the plan: aim below 500 cards, hard-fail above 600.
CARD_COUNT_TARGET = 500
CARD_COUNT_HARD_CAP = 600

# Vocabulary lists are truncated so a card stays well inside the embedding
# window and keeps its distinctive terms near the front.
MAX_VOCAB_VALUES = 12
MAX_CONCEPTS_LISTED = 40
MAX_TEXT_CHARS = 1200


@dataclass(frozen=True)
class Card:
    table_number: int
    card_type: str
    caption: str
    breadcrumb: str
    content: str
    structured_data: dict[str, Any]


_LEADING_ORDINAL_RE = re.compile(r"^\s*\d+[.)]\s*")


def display_label(label: str) -> str:
    """Drop the source list ordinal from a metric label for display.

    Cleaned labels such as ``"1. Overhead circuit miles"`` carry Energy
    Safety's list numbering. Keeping it in the caption would put a digit where
    the distinctive term should be, weakening both the weighted lexical index
    and the embedding. The unmodified label stays in ``excel_facts``.
    """
    return _LEADING_ORDINAL_RE.sub("", (label or "").strip()) or label


def _year_axis_lines(facts: Sequence[Fact]) -> list[str]:
    """Spell out the vintage-vs-reporting year mapping for tables 14/15."""
    triples = sorted(
        {
            (f.source_vintage_year, f.year_basis, f.reporting_year)
            for f in facts
            if f.source_vintage_year or f.reporting_year
        }
    )
    if not triples:
        return []
    lines = ["Year axes (submission vintage -> reporting year):"]
    for vintage, basis, reporting in triples:
        note = " (forward-looking)" if vintage and reporting and reporting != vintage else ""
        lines.append(f"- {vintage} submission -> describes {reporting}{note}; basis {basis}")
    return lines


def _fmt(value: Decimal | None) -> str:
    if value is None:
        return "—"
    text = format(value.normalize(), "f")
    return text


def _quarter_span(items: Sequence[Fact] | Sequence[Record]) -> str:
    periods = sorted(
        {
            (item.reporting_year, item.reporting_quarter)
            for item in items
            if item.reporting_year is not None
        }
    )
    if not periods:
        return "not specified"
    first, last = periods[0], periods[-1]

    def label(pair: tuple[int, int | None]) -> str:
        year, quarter = pair
        return f"{year} Q{quarter}" if quarter else str(year)

    return label(first) if first == last else f"{label(first)} through {label(last)}"


def _vocabulary(values: Iterable[str | None]) -> list[str]:
    counter = Counter(v for v in values if v)
    return [value for value, _ in counter.most_common(MAX_VOCAB_VALUES)]


def _dimension_vocabulary(facts: Sequence[Fact]) -> dict[str, list[str]]:
    collected: dict[str, list[str]] = defaultdict(list)
    for fact in facts:
        if fact.hftd_tier:
            collected["HFTD tier"].append(fact.hftd_tier)
        if fact.line_type:
            collected["line type"].append(fact.line_type)
        for key, value in fact.dimensions.items():
            if isinstance(value, str):
                collected[key.replace("_", " ")].append(value)
    return {
        name: _vocabulary(values)
        for name, values in collected.items()
        if len(set(values)) > 1 or name in {"HFTD tier", "line type"}
    }


def _filters_block(vocabulary: dict[str, list[str]], has_quarter: bool) -> list[str]:
    lines = ["Available filters:"]
    lines.append(
        "- reporting year and quarter" if has_quarter else "- reporting year"
    )
    for name, values in sorted(vocabulary.items()):
        if not values:
            continue
        lines.append(f"- {name}: {', '.join(values)}")
    return lines


def _truncate(text: str, limit: int = MAX_TEXT_CHARS) -> str:
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 1].rsplit(" ", 1)[0] + "…"


def _structured(
    contract: TableContract,
    card_type: str,
    *,
    semantic_metric_key: str | None = None,
    units: Sequence[str] = (),
    filters: Sequence[str] = (),
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "card_version": CARD_BUILDER_VERSION,
        "card_type": card_type,
        "table_number": contract.table_number,
        "source_key": f"sdge_table{contract.table_number:02d}",
        "family": contract.family,
        "allowed_filters": sorted(set(filters)),
        "units": sorted({u for u in units if u}),
    }
    if semantic_metric_key:
        payload["semantic_metric_key"] = semantic_metric_key
    if contract.family == "wide_risk":
        # Never let a consumer assume a single year axis on tables 14/15.
        payload["year_axes"] = ["reporting_year", "source_vintage_year"]
        payload["requires_year_basis"] = True
    if contract.dedupe:
        payload["default_dedupe"] = contract.dedupe
    if extra:
        payload.update(extra)
    return payload


def _filter_names(contract: TableContract, facts: Sequence[Fact]) -> list[str]:
    names = list(contract.promoted)
    if contract.period.get("year"):
        names.append("reporting_year")
    if contract.period.get("quarter"):
        names.append("reporting_quarter")
    if contract.family == "wide_risk":
        names += ["source_vintage_year", "year_basis"]
    keys = {key for fact in facts for key in fact.dimensions}
    return names + sorted(keys)


# --------------------------------------------------------------------------
# Card constructors
# --------------------------------------------------------------------------


def _overview_card(
    contract: TableContract,
    facts: Sequence[Fact],
    records: Sequence[Record],
    concepts: Sequence[tuple[str, str]],
) -> Card:
    items: Sequence[Any] = facts or records
    lines = [
        f"Table {contract.table_number} — {contract.title}",
        "",
        contract.description,
        "",
    ]
    if concepts:
        listed = [display_label(label) for _, label in concepts[:MAX_CONCEPTS_LISTED]]
        lines.append(f"Metrics reported in this table ({len(concepts)}):")
        lines += [f"- {label}" for label in listed]
        if len(concepts) > len(listed):
            lines.append(f"- …and {len(concepts) - len(listed)} more")
        lines.append("")
    if facts:
        lines += _filters_block(
            _dimension_vocabulary(facts), bool(contract.period.get("quarter"))
        )
        lines.append("")
    lines.append(f"Available periods: {_quarter_span(items)}.")
    if facts:
        lines.append(f"Stored facts: {len(facts)}.")
    if records:
        lines.append(f"Stored records: {len(records)}.")
    if contract.family == "wide_risk":
        lines.append(
            "This table separates the submission vintage year from the "
            "reporting year the risk value describes; a year question must "
            "state which axis it means."
        )
    lines.append(f"Source: SDG&E quarterly data report, Table {contract.table_number}.")

    return Card(
        table_number=contract.table_number,
        card_type="table_overview",
        caption=f"Table {contract.table_number} — {contract.title}",
        breadcrumb=f"{BREADCRUMB_ROOT} > Table {contract.table_number}",
        content="\n".join(lines),
        structured_data=_structured(
            contract,
            "table_overview",
            units=[f.unit for f in facts if f.unit],
            filters=_filter_names(contract, facts),
            extra={"concept_count": len(concepts)},
        ),
    )


def _concept_card(
    contract: TableContract,
    key: str,
    label: str,
    facts: Sequence[Fact],
    definitions: dict[str, dict[str, str]],
) -> Card:
    units = _vocabulary(f.unit for f in facts)
    vocabulary = _dimension_vocabulary(facts)
    reported = [f for f in facts if f.value_numeric is not None]
    label = display_label(label)

    lines = [f"Table {contract.table_number} — {label}", ""]

    detail = definitions.get(key, {})
    if detail.get("definition") and detail["definition"].strip() != label.strip():
        lines += [f"Definition: {_truncate(detail['definition'])}", ""]
    if detail.get("purpose"):
        lines += [f"Purpose: {_truncate(detail['purpose'])}", ""]

    if units:
        lines.append(f"Unit: {', '.join(units)}.")
    if contract.semantic_category:
        categories = _vocabulary(
            f.dimensions.get(contract.semantic_category) for f in facts
        )
        if categories:
            lines.append(f"Metric group: {', '.join(categories)}.")
    lines.append("")

    lines += _filters_block(vocabulary, bool(contract.period.get("quarter")))
    lines.append("")
    if contract.family == "wide_risk":
        lines += _year_axis_lines(facts)
    else:
        lines.append(f"Available periods: {_quarter_span(facts)}.")
    lines.append(
        f"Stored facts: {len(facts)} ({len(reported)} with a reported value)."
    )
    if reported:
        values = [f.value_numeric for f in reported]
        lines.append(
            f"Reported value range: {_fmt(min(values))} to {_fmt(max(values))}."
        )
    else:
        reasons = _vocabulary(f.blank_meaning for f in facts)
        if reasons:
            lines.append(f"No values reported. Recorded reason: {reasons[0]}.")
    lines.append(
        f"Source: SDG&E quarterly data report, Table {contract.table_number}."
    )

    return Card(
        table_number=contract.table_number,
        card_type="concept_metric",
        caption=f"Table {contract.table_number} — {label}",
        breadcrumb=f"{BREADCRUMB_ROOT} > Table {contract.table_number} > {label}",
        content="\n".join(lines),
        structured_data=_structured(
            contract,
            "concept_metric",
            semantic_metric_key=key,
            units=units,
            filters=_filter_names(contract, facts),
            extra={"fact_count": len(facts), "reported_count": len(reported)},
        ),
    )


def _activity_card(contract: TableContract, record: Record) -> Card:
    attrs = record.attributes
    title = record.title or record.entity_key or record.record_id
    lines = [f"WMP activity — {title}", ""]

    for field_name, prefix in (
        ("activity_objective", "Objective"),
        ("wmp_category", "WMP category"),
        ("wmp_initiative", "WMP initiative"),
        ("risk_target_reduction", "Risk targeted"),
    ):
        value = attrs.get(field_name)
        if value:
            lines.append(f"{prefix}: {_truncate(str(value), 400)}")

    target = attrs.get("annual_quant_target")
    units = attrs.get("quant_target_units")
    if target:
        lines.append(
            f"Annual quantitative target: {target}" + (f" {units}" if units else "")
        )
    if attrs.get("annual_qual_target"):
        lines.append(
            f"Annual qualitative target: {_truncate(str(attrs['annual_qual_target']), 400)}"
        )

    progress = [
        (quarter, attrs.get(f"quant_actual_progress_q1{suffix}"))
        for quarter, suffix in (("Q1", ""), ("Q2", "_2"), ("Q3", "_3"), ("Q4", "_4"))
    ]
    reported = [f"{q} {v}" for q, v in progress if v]
    if reported:
        lines.append(f"Cumulative quantitative progress: {', '.join(reported)}")

    if record.status:
        lines.append(f"Status: {record.status}")
    if attrs.get("corrective_actions_if_delayed"):
        lines.append(
            "Corrective action: "
            + _truncate(str(attrs["corrective_actions_if_delayed"]), 400)
        )

    period = (
        f"{record.reporting_year} Q{record.reporting_quarter}"
        if record.reporting_quarter
        else str(record.reporting_year)
    )
    lines.append("")
    lines.append(f"Reporting period: {period}.")
    if record.entity_key:
        lines.append(f"Activity tracking ID: {record.entity_key}.")
    lines.append(f"Source: SDG&E quarterly data report, Table {contract.table_number}.")

    return Card(
        table_number=contract.table_number,
        card_type="activity",
        caption=f"WMP activity — {title}",
        breadcrumb=f"{BREADCRUMB_ROOT} > Table {contract.table_number} > {title}",
        content="\n".join(lines),
        structured_data=_structured(
            contract,
            "activity",
            filters=["reporting_year", "reporting_quarter", "entity_key", "status"],
            extra={
                "entity_type": record.entity_type,
                "entity_key": record.entity_key,
                "record_id": record.record_id,
            },
        ),
    )


def _work_order_cards(
    contract: TableContract, records: Sequence[Record]
) -> list[Card]:
    """Routing cards for Table 13, grouped by equipment type.

    Individual work orders are never embedded; a known work-order number is
    answered by exact database lookup instead.
    """
    deduped = [
        record
        for record in records
        if str(record.attributes.get("exact_duplicate_index", "1")) == "1"
    ]
    by_equipment: dict[str, list[Record]] = defaultdict(list)
    for record in deduped:
        by_equipment[str(record.attributes.get("equipment_type") or "Unspecified")].append(
            record
        )

    ranked = sorted(by_equipment.items(), key=lambda kv: -len(kv[1]))
    top = ranked[:MAX_CONCEPTS_LISTED]

    cards: list[Card] = []
    for equipment, group in top:
        priorities = Counter(r.status for r in group if r.status)
        tiers = Counter(r.hftd_tier for r in group if r.hftd_tier)
        lines = [
            f"Table {contract.table_number} — open work orders for {equipment}",
            "",
            f"Open repair work orders recorded for equipment type {equipment}.",
            "",
            f"Distinct work-order rows (excluding exact duplicates): {len(group)}.",
        ]
        if priorities:
            lines.append(
                "GO 95 Rule 18 priority: "
                + ", ".join(f"{name} ({count})" for name, count in priorities.most_common())
            )
        if tiers:
            lines.append(
                "HFTD tier: "
                + ", ".join(f"{name} ({count})" for name, count in tiers.most_common())
            )
        lines.append(f"Available periods: {_quarter_span(group)}.")
        lines.append(
            "Individual work orders are looked up directly in the database by "
            "work order number; they are not embedded."
        )
        lines.append(
            f"Source: SDG&E quarterly data report, Table {contract.table_number}."
        )
        cards.append(
            Card(
                table_number=contract.table_number,
                card_type="work_order_summary",
                caption=f"Table {contract.table_number} — open work orders for {equipment}",
                breadcrumb=(
                    f"{BREADCRUMB_ROOT} > Table {contract.table_number} > {equipment}"
                ),
                content="\n".join(lines),
                structured_data=_structured(
                    contract,
                    "work_order_summary",
                    filters=[
                        "equipment_type",
                        "hftd_tier",
                        "line_type",
                        "reporting_year",
                        "reporting_quarter",
                        "status",
                    ],
                    extra={
                        "equipment_type": equipment,
                        "deduplicated_record_count": len(group),
                    },
                ),
            )
        )
    return cards


def _definitions(contract: TableContract, facts: Sequence[Fact]) -> dict[str, dict[str, str]]:
    """Collect per-concept definitions when the contract declares them.

    Only Table 3 carries definitions, and the plan forbids joining them into
    other tables without a reviewed crosswalk, so this stays table-local.
    """
    if not contract.semantic_definition:
        return {}
    out: dict[str, dict[str, str]] = {}
    for fact in facts:
        entry = out.setdefault(fact.semantic_metric_key, {})
        for role, column in (
            ("definition", contract.semantic_definition),
            ("purpose", contract.semantic_purpose),
            ("assumptions", contract.semantic_assumptions),
        ):
            if column and not entry.get(role):
                value = fact.dimensions.get(column)
                if value:
                    entry[role] = str(value)
    return out


def build_cards(
    contracts: ContractSet,
    contract: TableContract,
    facts: Sequence[Fact],
    records: Sequence[Record],
    *,
    definition_lookup: dict[str, dict[str, str]] | None = None,
) -> list[Card]:
    """Build the full card set for one table."""
    by_concept: dict[str, list[Fact]] = defaultdict(list)
    labels: dict[str, str] = {}
    for fact in facts:
        by_concept[fact.semantic_metric_key].append(fact)
        labels.setdefault(fact.semantic_metric_key, fact.metric_name)

    concepts = sorted(labels.items(), key=lambda kv: kv[1])
    definitions = definition_lookup or _definitions(contract, facts)

    cards = [_overview_card(contract, facts, records, concepts)]

    for key, label in concepts:
        cards.append(_concept_card(contract, key, label, by_concept[key], definitions))

    if contract.family == "entity":
        entity_type = contract.entity.get("type")
        if entity_type == "wmp_activity":
            cards += [_activity_card(contract, record) for record in records]
        elif entity_type == "work_order":
            cards += _work_order_cards(contract, records)

    return cards


def fit_cards_to_model(cards: Sequence[Card], tokenizer, max_tokens: int) -> list[Card]:
    """Shorten any card whose contextual form would exceed the model window.

    ``contextual_embedding_text_for_model`` raises rather than truncating, so an
    oversized card would abort ingestion. Trailing detail lines are dropped
    until the card fits; the heading and definition lines are kept.
    """
    fitted: list[Card] = []
    for card in cards:
        lines = card.content.split("\n")
        # Reserve room for the "Document:"/"Section:"/"Chunk:" context prefix.
        budget = max_tokens - 48
        while lines:
            candidate = "\n".join(lines)
            length = len(
                tokenizer.encode(
                    candidate,
                    add_special_tokens=True,
                    truncation=True,
                    max_length=max_tokens + 1,
                )
            )
            if length <= budget or len(lines) <= 4:
                break
            lines = lines[:-1]
        content = "\n".join(lines)
        if content != card.content:
            logger.debug("Shortened oversized card: %s", card.caption)
            fitted.append(
                Card(
                    table_number=card.table_number,
                    card_type=card.card_type,
                    caption=card.caption,
                    breadcrumb=card.breadcrumb,
                    content=content,
                    structured_data={**card.structured_data, "shortened": True},
                )
            )
        else:
            fitted.append(card)
    return fitted

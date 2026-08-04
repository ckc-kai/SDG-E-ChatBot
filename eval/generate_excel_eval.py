"""Generate deterministic, execution-verified Excel evaluation suites.

The release suite measures behavior supported by the current deterministic
Excel channel. The capability suite records harder target behavior without
turning those known gaps into a release failure. The cross-corpus suite uses
facts present in both Excel and PDF evidence to test source-aware routing.

    uv run python -m eval.generate_excel_eval
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
from collections import Counter
from dataclasses import asdict
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable

from retrieval.ingest.excel.contracts import load_contracts
from retrieval.query.excel.query import (
    FACTS,
    RECORDS,
    ExcelQueryPlan,
    Filter,
    PlanError,
    execute_plan,
)
from retrieval.utils import connect_db

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "excel-eval-v2"
DATASET_VERSION = "2026-07-30"
SAFE_YEARS = (2024, 2025)
RELEASE_TABLE_QUOTAS = {
    1: 5,
    2: 7,
    3: 7,
    4: 7,
    5: 7,
    6: 7,
    7: 7,
    8: 7,
    9: 7,
    10: 7,
    11: 7,
    13: 5,
}
RELEASE_TYPE_QUOTAS = {
    "metric_lookup": 49,
    "scoped_aggregate": 21,
    "entity_count": 10,
}
RELEASE_DIFFICULTY_QUOTAS = {"simple": 24, "medium": 40, "hard": 16}
CHALLENGE_TYPE_QUOTAS = {
    "period_trend": 4,
    "group_comparison": 4,
    "ranking": 4,
    "activity_attribute": 4,
    "missing_value": 4,
    "year_scope_clarification": 4,
}

_STOPWORDS = {
    "a", "an", "and", "at", "by", "did", "for", "from", "how", "in", "of",
    "on", "or", "report", "reported", "sdg", "sdge", "the", "to", "value",
    "was", "were", "what", "which", "with",
}


def _clean(text: str) -> str:
    return " ".join(str(text).split())


def _normalise(text: str) -> str:
    text = text.lower().replace("sdg&e", "sdge")
    return re.sub(r"[^a-z0-9.%-]+", " ", text).strip()


def _display_label(text: str) -> str:
    """Turn source labels into natural question text without changing meaning."""
    label = re.sub(r"^\d+\.\s*", "", _clean(text))
    label = re.sub(r"\(s\)", "s", label, flags=re.I)
    label = re.sub(
        r"net addition \(or removal\)\s*-\s*",
        "net change in ",
        label,
        flags=re.I,
    )
    label = re.sub(
        r"^number of overhead circuit miles upgraded$",
        "overhead circuit miles upgraded",
        label,
        flags=re.I,
    )
    label = re.sub(r"\b95 percentile\b", "95th percentile", label, flags=re.I)
    label = re.sub(
        r"\bcold end cotter keys\b", "cold-end cotter keys", label, flags=re.I
    )
    label = re.sub(r"\bother\s*\(specify\)\b", "other equipment", label, flags=re.I)
    label = re.sub(r"\bconductor\s*-\s*veg\b", "vegetation-related conductor", label, flags=re.I)
    for acronym in (
        "PSPS", "HFTD", "FPI", "CALFIRE", "OPI", "SAIDI", "SAIFI", "SCADA", "WMP"
    ):
        label = re.sub(rf"\b{acronym}\b", acronym, label, flags=re.I)
    return label


def _lower_initial(text: str) -> str:
    label = _display_label(text)
    return label[:1].lower() + label[1:] if label else label


def _stable_key(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _json_value(value: Any) -> Any:
    if isinstance(value, Decimal):
        return format(value, "f")
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    return value


def _plan_payload(plan: ExcelQueryPlan) -> dict[str, Any]:
    payload = asdict(plan)
    payload["filters"] = [asdict(flt) for flt in plan.filters]
    return payload


def _numeric_variants(value: Decimal) -> set[str]:
    """Render a value as defensible exact/source-precision PDF search forms."""
    exact = format(value, "f")
    trimmed = exact.rstrip("0").rstrip(".") if "." in exact else exact
    variants = {exact, trimmed}
    if value == value.to_integral_value():
        variants.add(f"{int(value):,}")
        variants.add(str(int(value)))
    else:
        for places in (1, 2, 3):
            rendered = f"{value:,.{places}f}"
            variants.add(rendered)
            variants.add(rendered.replace(",", ""))
    return {item for item in variants if item}


def _concept_tokens(concept: str) -> list[str]:
    tokens = [
        token for token in re.findall(r"[a-z0-9]+", _normalise(concept))
        if len(token) >= 3 and token not in _STOPWORDS and not token.isdigit()
    ]
    return list(dict.fromkeys(tokens))


class PdfOverlapChecker:
    """Find answer evidence only when value, concept, and period co-occur."""

    def __init__(self, conn) -> None:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT c.id, d.filename, c.content_type, c.page_start,
                       coalesce(c.caption, ''), c.content
                FROM chunks c
                JOIN documents d ON d.id = c.document_id
                WHERE c.content_type <> 'excel_card'
                ORDER BY c.id
                """
            )
            self.chunks = [
                {
                    "chunk_id": row[0],
                    "source_pdf": row[1],
                    "content_type": row[2],
                    "page_start_db": row[3],
                    "caption": row[4],
                    "content": _clean(row[5] or ""),
                    "normalised": _normalise(f"{row[4]} {row[5] or ''}"),
                }
                for row in cur.fetchall()
            ]

    def find_evidence(
        self,
        *,
        question: str,
        concept: str,
        value: Decimal,
    ) -> dict[str, Any] | None:
        variants = {_normalise(item).replace(",", "") for item in _numeric_variants(value)}
        tokens = _concept_tokens(concept)
        years = set(re.findall(r"\b20[2-3][0-9]\b", question))
        quarters = {q.lower() for q in re.findall(r"\bQ[1-4]\b", question, re.I)}
        required_tokens = min(2, len(tokens))

        for chunk in self.chunks:
            text = chunk["normalised"].replace(",", "")
            if not any(
                re.search(rf"(?<![\d.]){re.escape(variant)}(?![\d.])", text)
                for variant in variants
            ):
                continue
            if years and not years.intersection(text.split()):
                continue
            if quarters and not any(quarter in text for quarter in quarters):
                continue
            if sum(token in text for token in tokens) < required_tokens:
                continue
            excerpt = chunk["content"][:500]
            return {
                "chunk_id": chunk["chunk_id"],
                "source_pdf": chunk["source_pdf"],
                "content_type": chunk["content_type"],
                "page_start_db": chunk["page_start_db"],
                "content_excerpt": excerpt,
            }
        return None


def _distinct(conn, sql: str, params: tuple = ()) -> list[Any]:
    with conn.cursor() as cur:
        cur.execute(sql, params)
        return [row[0] for row in cur.fetchall() if row[0] is not None]


def _concepts(conn, table: int, limit: int = 40) -> list[tuple[str, str]]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT f.semantic_metric_key, min(f.metric_name)
            FROM excel_facts f
            JOIN excel_revisions r ON r.id=f.revision_id AND r.state='active'
            WHERE f.table_number=%s AND f.value_numeric IS NOT NULL
            GROUP BY 1 ORDER BY count(*) DESC, 1 LIMIT %s
            """,
            (table, limit),
        )
        return [(row[0], _clean(row[1])) for row in cur.fetchall()]


def _promoted_values(conn, table: int, column: str) -> list[str]:
    return _distinct(
        conn,
        f"""
        SELECT DISTINCT f.{column}
        FROM excel_facts f
        JOIN excel_revisions r ON r.id=f.revision_id AND r.state='active'
        WHERE f.table_number=%s ORDER BY 1
        """,
        (table,),
    )


def _dimension_values(conn, table: int, key: str) -> list[str]:
    return _distinct(
        conn,
        """
        SELECT DISTINCT f.dimensions ->> %s
        FROM excel_facts f
        JOIN excel_revisions r ON r.id=f.revision_id AND r.state='active'
        WHERE f.table_number=%s ORDER BY 1
        """,
        (key, table),
    )


def _candidate(
    question: str,
    question_type: str,
    plan: ExcelQueryPlan,
    table: int,
    semantic_key: str | None,
    concept: str,
    **extra: Any,
) -> dict[str, Any]:
    return {
        "question": _clean(question),
        "question_type": question_type,
        "plan": plan,
        "table_number": table,
        "semantic_metric_key": semantic_key,
        "concept": concept,
        **extra,
    }


def build_release_candidates(conn) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []

    # Table 1: supported entity counts, not unsupported target attributes.
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT rec.attributes ->> 'activity_classification',
                   rec.reporting_year, rec.reporting_quarter
            FROM excel_records rec
            JOIN excel_revisions r ON r.id=rec.revision_id AND r.state='active'
            WHERE rec.table_number=1
              AND rec.reporting_year=ANY(%s)
              AND rec.attributes ->> 'activity_classification' IS NOT NULL
            GROUP BY 1,2,3 ORDER BY 2 DESC,1
            """,
            (list(SAFE_YEARS),),
        )
        activity_scopes = cur.fetchall()
    activity_templates = (
        "How many {classification} activities did SDG&E list in its Q{quarter} {year} update?",
        "For {year} Q{quarter}, how many WMP activities were classified as {classification}s?",
        "What was the number of {classification} activities in SDG&E's {year} Q{quarter} workbook?",
    )
    for index, (classification, year, quarter) in enumerate(activity_scopes):
        question = activity_templates[index % len(activity_templates)].format(
            classification=classification.lower(), year=year, quarter=quarter
        )
        candidates.append(
            _candidate(
                question,
                "entity_count",
                ExcelQueryPlan(
                    table_number=1,
                    source=RECORDS,
                    aggregate="count",
                    filters=(
                        Filter("reporting_year", value=year),
                        Filter("reporting_quarter", value=quarter),
                        Filter(
                            "activity_classification",
                            value=classification,
                            json_key=True,
                        ),
                    ),
                ),
                1,
                None,
                f"{classification} WMP activities",
            )
        )
    for year in SAFE_YEARS:
        candidates.append(
            _candidate(
                f"How many WMP activities did SDG&E include in its {year} Q4 "
                "activity update?",
                "entity_count",
                ExcelQueryPlan(
                    table_number=1,
                    source=RECORDS,
                    aggregate="count",
                    filters=(
                        Filter("reporting_year", value=year),
                        Filter("reporting_quarter", value=4),
                    ),
                ),
                1,
                None,
                "WMP activities",
            )
        )

    # Tables 2, 3, 4 and 10: direct quarterly operational metrics.
    lookup_templates = (
        "What was SDG&E's {label} in {year} Q{quarter}?",
        "For {year} Q{quarter}, what value did SDG&E report for {label}?",
        "What value for {label} was reported in SDG&E's Q{quarter} {year} update?",
        "In Q{quarter} {year}, what did SDG&E report as {label}?",
    )
    for table in (2, 3, 4, 10):
        for concept_index, (key, name) in enumerate(_concepts(conn, table, 24)):
            label = _display_label(name)
            for year in SAFE_YEARS:
                for quarter in (2, 4):
                    template = lookup_templates[
                        (concept_index + year + quarter) % len(lookup_templates)
                    ]
                    candidates.append(
                        _candidate(
                            template.format(
                                label=label, year=year, quarter=quarter
                            ),
                            "metric_lookup",
                            ExcelQueryPlan(
                                table_number=table,
                                semantic_metric_key=key,
                                filters=(
                                    Filter("reporting_year", value=year),
                                    Filter("reporting_quarter", value=quarter),
                                ),
                                aggregate="sum",
                            ),
                            table,
                            key,
                            label,
                        )
                    )

    # Tables 5 and 6: scoped aggregates by risk/ignition driver.
    for table, driver_key, noun in (
        (5, "risk_event_driver", "risk events"),
        (6, "ignition_driver", "ignitions"),
    ):
        drivers = _dimension_values(conn, table, driver_key)
        for key, name in _concepts(conn, table, 6):
            for driver in drivers[:24]:
                driver_label = (
                    "other causes"
                    if driver.lower() == "other (other)"
                    else _display_label(driver)
                )
                for year in SAFE_YEARS:
                    candidates.append(
                        _candidate(
                            f"In {year}, how many {noun} did SDG&E attribute "
                            f"to {_lower_initial(driver_label)}?",
                            "scoped_aggregate",
                            ExcelQueryPlan(
                                table_number=table,
                                semantic_metric_key=key,
                                filters=(
                                    Filter("reporting_year", value=year),
                                    Filter(driver_key, value=driver, json_key=True),
                                ),
                                aggregate="sum",
                            ),
                            table,
                            key,
                            f"{driver_label} {noun}",
                        )
                    )

    # Tables 7-9: system composition and change, explicitly scoped.
    for table in (7, 8, 9):
        tiers = _promoted_values(conn, table, "hftd_tier")
        lines = _promoted_values(conn, table, "line_type")
        for key, name in _concepts(conn, table, 16):
            label = _display_label(name)
            for year in SAFE_YEARS:
                for quarter in (2, 4):
                    for tier in tiers:
                        for line in lines[:2]:
                            candidates.append(
                                _candidate(
                                    f"For {line.lower()} lines in {tier}, what "
                                    f"{_lower_initial(label)} did SDG&E report in "
                                    f"{year} Q{quarter}?",
                                    "metric_lookup",
                                    ExcelQueryPlan(
                                        table_number=table,
                                        semantic_metric_key=key,
                                        filters=(
                                            Filter("reporting_year", value=year),
                                            Filter("reporting_quarter", value=quarter),
                                            Filter("hftd_tier", value=tier),
                                            Filter("line_type", value=line),
                                        ),
                                        aggregate="sum",
                                    ),
                                    table,
                                    key,
                                    label,
                                )
                            )

    # Table 11: Territory and HFTD are alternative scopes, never additive.
    expense_types = _dimension_values(conn, 11, "expense_type")
    for key, name in _concepts(conn, 11, 30):
        for expense in expense_types:
            for year in SAFE_YEARS:
                for tier in ("Territory", "HFTD"):
                    scope = "territory-wide" if tier == "Territory" else "within the HFTD"
                    candidates.append(
                        _candidate(
                            f"How much {expense} did SDG&E report for {name} "
                            f"{scope} in {year}?",
                            "scoped_aggregate",
                            ExcelQueryPlan(
                                table_number=11,
                                semantic_metric_key=key,
                                filters=(
                                    Filter("reporting_year", value=year),
                                    Filter(
                                        "expense_type",
                                        value=expense,
                                        json_key=True,
                                    ),
                                    Filter("hftd_tier", value=tier),
                                ),
                                aggregate="sum",
                            ),
                            11,
                            key,
                            f"{expense} {name}",
                        )
                    )

    # Table 13: open-work-order entity counts.
    equipment = _distinct(
        conn,
        """
        SELECT rec.attributes ->> 'equipment_type'
        FROM excel_records rec
        JOIN excel_revisions r ON r.id=rec.revision_id AND r.state='active'
        WHERE rec.table_number=13
          AND rec.attributes ->> 'equipment_type' IS NOT NULL
        GROUP BY 1 ORDER BY count(*) DESC,1 LIMIT 30
        """,
    )
    count_templates = (
        "How many open {item} work orders did SDG&E report in {year} Q{quarter}?",
        "At the end of Q{quarter} {year}, how many {item} work orders remained open?",
        "What was the open-work-order count for {item} in SDG&E's {year} Q{quarter} update?",
    )
    for item_index, item in enumerate(equipment):
        for year in SAFE_YEARS:
            for quarter in (2, 4):
                template = count_templates[
                    (item_index + year + quarter) % len(count_templates)
                ]
                candidates.append(
                    _candidate(
                        template.format(
                            item=_lower_initial(item),
                            year=year,
                            quarter=quarter,
                        ),
                        "entity_count",
                        ExcelQueryPlan(
                            table_number=13,
                            source=RECORDS,
                            aggregate="count",
                            filters=(
                                Filter("reporting_year", value=year),
                                Filter("reporting_quarter", value=quarter),
                                Filter(
                                    "equipment_type",
                                    value=item,
                                    json_key=True,
                                ),
                            ),
                        ),
                        13,
                        None,
                        f"{_display_label(item)} open work orders",
                    )
                )

    return candidates


def _activity_rows(conn) -> list[tuple]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT rec.entity_key, rec.title, rec.reporting_year,
                   rec.attributes ->> 'annual_quant_target',
                   rec.attributes ->> 'quant_target_units',
                   rec.provenance
            FROM excel_records rec
            JOIN excel_revisions r ON r.id=rec.revision_id AND r.state='active'
            WHERE rec.table_number=1
              AND rec.reporting_year=ANY(%s)
              AND rec.attributes ->> 'annual_quant_target' IS NOT NULL
              AND rec.title IS NOT NULL
            ORDER BY rec.reporting_year, rec.entity_key
            """,
            (list(SAFE_YEARS),),
        )
        return cur.fetchall()


def build_challenge_candidates(conn) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []

    # Trends: return an ordered quarter/value series as the gold result.
    for table in (2, 3, 4, 10):
        for key, name in _concepts(conn, table, 4):
            candidates.append(
                _candidate(
                    f"How did SDG&E's {_lower_initial(name)} change "
                    "across the quarters of 2025?",
                    "period_trend",
                    ExcelQueryPlan(
                        table_number=table,
                        semantic_metric_key=key,
                        filters=(Filter("reporting_year", value=2025),),
                        group_by=("reporting_quarter",),
                        aggregate="sum",
                        limit=8,
                    ),
                    table,
                    key,
                    name,
                    answer_kind="rows",
                )
            )

    # Group comparisons across HFTD tiers.
    for table in (7, 9):
        for key, name in _concepts(conn, table, 8):
            candidates.append(
                _candidate(
                    f"In 2025 Q4, did SDG&E report more "
                    f"{_lower_initial(name)} in "
                    "HFTD Tier 2 or HFTD Tier 3?",
                    "group_comparison",
                    ExcelQueryPlan(
                        table_number=table,
                        semantic_metric_key=key,
                        filters=(
                            Filter("reporting_year", value=2025),
                            Filter("reporting_quarter", value=4),
                            Filter("hftd_tier", operator="ne", value="Non-HFTD"),
                        ),
                        group_by=("hftd_tier",),
                        operation="rank",
                        aggregate="sum",
                        limit=3,
                    ),
                    table,
                    key,
                    name,
                    answer_kind="rows",
                )
            )

    # Top circuit segments from Table 15, with submission vintage explicit.
    for key, name in _concepts(conn, 15, 8):
        candidates.append(
            _candidate(
                f"In SDG&E's 2025 submission, which three circuit segments had "
                f"the highest {_lower_initial(name)}?",
                "ranking",
                ExcelQueryPlan(
                    table_number=15,
                    semantic_metric_key=key,
                    filters=(Filter("source_vintage_year", value=2025),),
                    group_by=("record_id",),
                    operation="rank",
                    aggregate="max",
                    limit=3,
                ),
                15,
                key,
                name,
                answer_kind="rows",
            )
        )

    # Reviewed record attributes; the expected target is not a filter.
    for entity_key, title, year, target, units, _ in _activity_rows(conn):
        candidates.append(
            _candidate(
                f"What annual quantitative target did SDG&E set for "
                f"{_display_label(title)} "
                f"({entity_key}) in {year}?",
                "activity_attribute",
                ExcelQueryPlan(
                    table_number=1,
                    source=RECORDS,
                    operation="select",
                    group_by=("title",),
                    select_json_keys=("annual_quant_target",),
                    filters=(
                        Filter("reporting_year", value=year),
                        Filter("entity_key", value=entity_key),
                    ),
                    limit=2,
                ),
                1,
                None,
                title,
                answer_kind="attribute",
                expected_override=target,
                unit_override=units,
            )
        )

    # Realistic not-yet-reported questions. Corpus drift protection forces an
    # explicit regeneration when a future quarterly file eventually arrives.
    for table in (2, 3, 4, 10):
        key, name = _concepts(conn, table, 1)[0]
        plan = ExcelQueryPlan(
            table_number=table,
            semantic_metric_key=key,
            filters=(
                Filter("reporting_year", value=2026),
                Filter("reporting_quarter", value=1),
            ),
            aggregate="sum",
        )
        candidates.append(
            _candidate(
                f"Has SDG&E reported {_lower_initial(name)} for 2026 Q1 yet?",
                "missing_value",
                plan,
                table,
                key,
                name,
                answer_kind="empty",
                expected_override="Not reported",
                expected_channel_behavior="decline",
            )
        )

    # Bare reporting years on Tables 14/15 must request clarification.
    for table in (14, 15):
        for key, name in _concepts(conn, table, 4):
            candidates.append(
                _candidate(
                    f"What {_lower_initial(name)} did SDG&E report for 2025?",
                    "year_scope_clarification",
                    ExcelQueryPlan(
                        table_number=table,
                        semantic_metric_key=key,
                        filters=(Filter("reporting_year", value=2025),),
                        aggregate="sum",
                    ),
                    table,
                    key,
                    name,
                    answer_kind="clarification",
                    expected_channel_behavior="clarify",
                )
            )

    return candidates


def _answer_tolerance(value: Any, kind: str) -> str:
    if kind != "numeric":
        return "0"
    numeric = Decimal(str(value))
    return "0" if numeric == numeric.to_integral_value() else "1e-9"


def verify_candidate(
    conn,
    candidate: dict[str, Any],
    *,
    source_scope: str,
    checker: PdfOverlapChecker | None = None,
) -> dict[str, Any] | None:
    plan: ExcelQueryPlan = candidate["plan"]
    kind = candidate.get("answer_kind", "numeric")
    behavior = candidate.get("expected_channel_behavior", "answer")
    resolved_scope = "excel_only" if source_scope == "challenge" else source_scope
    try:
        result = execute_plan(plan, conn)
    except PlanError as exc:
        if kind != "clarification":
            return None
        return {
            "schema_version": SCHEMA_VERSION,
            "dataset_version": DATASET_VERSION,
            "question": candidate["question"],
            "question_type": candidate["question_type"],
            "difficulty": "hard",
            "table_number": candidate["table_number"],
            "semantic_metric_key": candidate["semantic_metric_key"],
            "source_scope": resolved_scope,
            "expected_channel_behavior": behavior,
            "expected_answer": str(exc),
            "expected_answer_kind": "clarification",
            "unit": None,
            "answer_tolerance": "0",
            "contributing_rows": 0,
            "provenance": [],
            "pdf_evidence": [],
            "preferred_lane": None,
            "plan": _plan_payload(plan),
        }

    if kind == "empty":
        if result.contributing_facts or (
            result.rows and result.rows[0][-1] is not None
        ):
            return None
        expected: Any = candidate["expected_override"]
    elif not result.rows:
        return None
    elif kind == "rows":
        expected = [[_json_value(cell) for cell in row] for row in result.rows]
        if len(expected) < 2:
            return None
    elif kind == "attribute":
        expected = candidate.get("expected_override", result.rows[0][-1])
        if len(result.rows) != 1 or str(result.rows[0][-1]) != str(expected):
            return None
    else:
        expected = candidate.get("expected_override", result.rows[0][-1])
        if expected is None:
            return None
    if (
        source_scope == "excel_only"
        and behavior == "answer"
        and plan.source == RECORDS
        and result.contributing_facts == 0
    ):
        return None

    pdf_evidence: list[dict[str, Any]] = []
    if checker is not None and kind in {"numeric", "attribute"}:
        try:
            overlap = checker.find_evidence(
                question=candidate["question"],
                concept=candidate["concept"],
                value=Decimal(str(expected)),
            )
        except Exception:
            overlap = None
        if overlap:
            pdf_evidence.append(overlap)
            if source_scope == "excel_only":
                return None
            if source_scope == "challenge":
                resolved_scope = "cross_corpus"

    unit = candidate.get("unit_override", result.unit)
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset_version": DATASET_VERSION,
        "question": candidate["question"],
        "question_type": candidate["question_type"],
        "difficulty": candidate.get("difficulty", "medium"),
        "table_number": candidate["table_number"],
        "semantic_metric_key": candidate["semantic_metric_key"],
        "source_scope": resolved_scope,
        "expected_channel_behavior": behavior,
        "expected_answer": _json_value(expected),
        "expected_answer_kind": kind,
        "unit": unit,
        "answer_tolerance": _answer_tolerance(expected, kind),
        "contributing_rows": result.contributing_facts,
        "provenance": result.provenance,
        "pdf_evidence": pdf_evidence,
        "preferred_lane": "excel" if resolved_scope == "excel_only" else None,
        "excel_only_reason": (
            "No PDF chunk contains the answer with matching concept and period"
            if resolved_scope == "excel_only"
            else None
        ),
        "plan": _plan_payload(plan),
    }


def _diverse_take(rows: Iterable[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: _stable_key(row["question"]))
    selected: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    for row in ordered:
        key = row.get("semantic_metric_key") or _normalise(row["question_type"])
        if key in seen_keys:
            continue
        selected.append(row)
        seen_keys.add(key)
        if len(selected) == count:
            return selected
    for row in ordered:
        if row in selected:
            continue
        selected.append(row)
        if len(selected) == count:
            return selected
    return selected


def select_release(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for table, quota in RELEASE_TABLE_QUOTAS.items():
        pool = [row for row in rows if row["table_number"] == table]
        chosen = _diverse_take(pool, quota)
        if len(chosen) != quota:
            raise RuntimeError(
                f"table {table} has {len(chosen)} verified candidates; needs {quota}"
            )
        selected.extend(chosen)

    # Difficulty is based on the tested reasoning shape, with exact benchmark
    # quotas: basic metric-period lookups, scoped filters, and multi-dimension
    # queries that are most vulnerable to binding/routing errors.
    simple_pool = [
        row for row in selected if row["table_number"] in {2, 3, 4, 10}
    ]
    for row in sorted(simple_pool, key=lambda item: _stable_key(item["question"]))[:24]:
        row["difficulty"] = "simple"
    hard_pool = [row for row in selected if row["table_number"] == 11]
    hard_pool.extend(
        sorted(
            [row for row in selected if row["table_number"] in {7, 8, 9}],
            key=lambda item: _stable_key(item["question"]),
        )[:9]
    )
    hard_ids = {id(row) for row in hard_pool}
    simple_ids = {id(row) for row in selected if row["difficulty"] == "simple"}
    for row in selected:
        if id(row) in hard_ids:
            row["difficulty"] = "hard"
        elif id(row) not in simple_ids:
            row["difficulty"] = "medium"

    selected.sort(key=lambda row: (row["table_number"], _stable_key(row["question"])))
    for index, row in enumerate(selected, 1):
        row["id"] = f"excel_eval_v2_{index:04d}"
    return selected


def select_challenge(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for question_type, quota in CHALLENGE_TYPE_QUOTAS.items():
        pool = [row for row in rows if row["question_type"] == question_type]
        chosen = _diverse_take(pool, quota)
        if len(chosen) != quota:
            raise RuntimeError(
                f"challenge {question_type} has {len(chosen)} candidates; needs {quota}"
            )
        selected.extend(chosen)
    for row in selected:
        row["difficulty"] = (
            "medium" if row["question_type"] in {"activity_attribute", "missing_value"}
            else "hard"
        )
    selected.sort(key=lambda row: (row["question_type"], _stable_key(row["question"])))
    for index, row in enumerate(selected, 1):
        row["id"] = f"excel_challenge_{index:04d}"
    return selected


def build_cross_corpus(
    conn,
    checker: PdfOverlapChecker,
    fact_count: int = 10,
) -> list[dict[str, Any]]:
    facts: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for entity_key, title, year, target, units, _ in _activity_rows(conn):
        if Decimal(str(target)) == 0:
            continue
        evidence = checker.find_evidence(
            question=f"{title} {year}",
            concept=title,
            value=Decimal(str(target)),
        )
        if not evidence:
            continue
        candidate = _candidate(
            f"What annual quantitative target did SDG&E set for {title} "
            f"({entity_key}) in {year}?",
            "cross_corpus_target",
            ExcelQueryPlan(
                table_number=1,
                source=RECORDS,
                operation="select",
                group_by=("title",),
                select_json_keys=("annual_quant_target",),
                filters=(
                    Filter("reporting_year", value=year),
                    Filter("entity_key", value=entity_key),
                ),
                limit=2,
            ),
            1,
            None,
            title,
            answer_kind="attribute",
            expected_override=target,
            unit_override=units,
        )
        verified = verify_candidate(
            conn, candidate, source_scope="cross_corpus", checker=None
        )
        if verified:
            facts.append((verified, evidence))
        if len(facts) == fact_count:
            break
    if len(facts) != fact_count:
        raise RuntimeError(
            f"found {len(facts)} verified PDF/Excel facts; needs {fact_count}"
        )

    rows: list[dict[str, Any]] = []
    for base, evidence in facts:
        title = base["provenance"][0].get("wmp_activity_name") if base["provenance"] else None
        concept = _display_label(
            title or base["question"].split(" for ", 1)[-1].rsplit(" (", 1)[0]
        )
        year_match = re.search(r"\b20[2-3][0-9]\b", base["question"])
        year = year_match.group(0) if year_match else "the reported year"
        for source, phrasing, behavior in (
            (
                "pdf",
                f"According to SDG&E's WMP filing, what was the {year} annual "
                f"target for {concept}?",
                "decline",
            ),
            (
                "excel",
                f"In SDG&E's cleaned quarterly activity workbook, what {year} "
                f"annual target is listed for {concept}?",
                "answer",
            ),
        ):
            row = {
                **base,
                "question": _clean(phrasing),
                "difficulty": "medium",
                "source_scope": "cross_corpus",
                "expected_channel_behavior": behavior,
                "preferred_lane": source,
                "pdf_evidence": [evidence],
                "excel_only_reason": None,
            }
            rows.append(row)
    rows.sort(key=lambda row: (_stable_key(row["question"]), row["preferred_lane"]))
    for index, row in enumerate(rows, 1):
        row["id"] = f"excel_cross_{index:04d}"
    return rows


def _active_revisions(conn) -> list[dict[str, Any]]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT s.table_number, s.source_key, r.id, r.source_file,
                   r.source_hash, r.contract_version, r.ingest_signature
            FROM excel_sources s
            JOIN excel_revisions r ON r.id=s.active_revision_id
            ORDER BY s.table_number
            """
        )
        return [
            {
                "table_number": row[0],
                "source_key": row[1],
                "revision_id": row[2],
                "source_file": row[3],
                "source_hash": row[4],
                "contract_version": row[5],
                "ingest_signature": row[6],
            }
            for row in cur.fetchall()
        ]


def _validate_rows(
    rows: list[dict[str, Any]],
    *,
    suite: str,
    expected_size: int,
) -> None:
    if len(rows) != expected_size:
        raise RuntimeError(f"{suite}: expected {expected_size} rows, got {len(rows)}")
    ids = [row["id"] for row in rows]
    questions = [_normalise(row["question"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError(f"{suite}: duplicate IDs")
    if len(questions) != len(set(questions)):
        raise RuntimeError(f"{suite}: duplicate normalized questions")
    for row in rows:
        if not row["question"].endswith("?"):
            raise RuntimeError(f"{row['id']}: question is not natural punctuation")
        if row["expected_channel_behavior"] == "answer" and not row["provenance"]:
            raise RuntimeError(f"{row['id']}: answer has no Excel provenance")
        if row["table_number"] == 11:
            tier_filters = [
                flt for flt in row["plan"]["filters"] if flt["field"] == "hftd_tier"
            ]
            if len(tier_filters) != 1:
                raise RuntimeError(f"{row['id']}: Table 11 needs exactly one scope")
        for flt in row["plan"]["filters"]:
            if flt["field"] == "annual_quant_target":
                raise RuntimeError(f"{row['id']}: expected target leaked into filters")
    if suite == "release":
        if Counter(row["table_number"] for row in rows) != Counter(
            RELEASE_TABLE_QUOTAS
        ):
            raise RuntimeError("release: table quotas do not match")
        if Counter(row["question_type"] for row in rows) != Counter(
            RELEASE_TYPE_QUOTAS
        ):
            raise RuntimeError("release: question-type quotas do not match")
        if Counter(row["difficulty"] for row in rows) != Counter(
            RELEASE_DIFFICULTY_QUOTAS
        ):
            raise RuntimeError("release: difficulty quotas do not match")
        if any(row["pdf_evidence"] for row in rows):
            raise RuntimeError("release: cross-corpus evidence leaked into suite")
    if suite == "challenge":
        if Counter(row["question_type"] for row in rows) != Counter(
            CHALLENGE_TYPE_QUOTAS
        ):
            raise RuntimeError("challenge: type quotas do not match")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "".join(
        json.dumps(row, sort_keys=True, ensure_ascii=False, default=str) + "\n"
        for row in rows
    )
    path.write_text(content)
    return hashlib.sha256(content.encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("eval/excel"))
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Backward-compatible override for the release-suite path.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    load_contracts()

    release_path = args.out or args.out_dir / "evaluation_excel.jsonl"
    out_dir = release_path.parent
    challenge_path = out_dir / "evaluation_excel_challenge.jsonl"
    cross_path = out_dir / "evaluation_excel_cross_corpus.jsonl"
    manifest_path = out_dir / "manifest.json"

    conn = connect_db()
    try:
        checker = PdfOverlapChecker(conn)
        logger.info("Building and verifying release candidates")
        verified_release = [
            verified
            for candidate in build_release_candidates(conn)
            if (
                verified := verify_candidate(
                    conn,
                    candidate,
                    source_scope="excel_only",
                    checker=checker,
                )
            )
        ]
        release = select_release(verified_release)

        logger.info("Building and verifying capability challenges")
        verified_challenge = [
            verified
            for candidate in build_challenge_candidates(conn)
            if (
                verified := verify_candidate(
                    conn,
                    candidate,
                    source_scope="challenge",
                    checker=checker,
                )
            )
        ]
        challenge = select_challenge(verified_challenge)

        logger.info("Building verified PDF/Excel overlap pairs")
        cross = build_cross_corpus(conn, checker)

        _validate_rows(release, suite="release", expected_size=80)
        _validate_rows(challenge, suite="challenge", expected_size=24)
        _validate_rows(cross, suite="cross", expected_size=20)

        hashes = {
            release_path.name: _write_jsonl(release_path, release),
            challenge_path.name: _write_jsonl(challenge_path, challenge),
            cross_path.name: _write_jsonl(cross_path, cross),
        }
        revisions = _active_revisions(conn)
        corpus_signature = hashlib.sha256(
            "\n".join(
                f"{row['table_number']}:{row['source_hash']}" for row in revisions
            ).encode()
        ).hexdigest()
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "dataset_version": DATASET_VERSION,
            "corpus_signature": corpus_signature,
            "active_revisions": revisions,
            "suites": {
                release_path.name: {"rows": 80, "sha256": hashes[release_path.name]},
                challenge_path.name: {
                    "rows": 24,
                    "sha256": hashes[challenge_path.name],
                },
                cross_path.name: {"rows": 20, "sha256": hashes[cross_path.name]},
            },
        }
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
    finally:
        conn.close()

    print(f"Wrote release suite:   {release_path} (80)")
    print(f"Wrote challenge suite: {challenge_path} (24)")
    print(f"Wrote cross suite:     {cross_path} (20)")
    print(f"Wrote corpus manifest: {manifest_path}")


if __name__ == "__main__":
    main()

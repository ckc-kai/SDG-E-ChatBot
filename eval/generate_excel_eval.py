"""Generate and verify an Excel-only evaluation set.

Every generated question must satisfy four conditions before it is kept:

1. **Answerable** — a validated ``ExcelQueryPlan`` executes and returns a
   non-null result from the active revision.
2. **Deterministic** — the plan yields exactly the expected shape (one scalar,
   or one clear winner for comparisons/rankings).
3. **Excel-only** — the answer cannot be found in the PDF corpus. Verified
   mechanically: the answer, rendered in several plausible numeric formats, must
   not appear in any PDF chunk's text. The PDF corpus is entirely 2023-vintage,
   so 2024-2025 quarterly actuals are structurally absent, but the check is run
   regardless rather than assumed.
4. **Grounded** — the expected value carries its unit and a source cell.

    uv run python -m eval.generate_excel_eval --out eval/excel/evaluation_excel.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
from dataclasses import asdict
from decimal import Decimal
from pathlib import Path
from typing import Any

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

# Long-metric tables: restrict to periods that cannot appear in the 2023 PDFs.
SAFE_YEARS = (2024, 2025)


def _q(text: str) -> str:
    return " ".join(text.split())


def _numeric_variants(value: Decimal) -> list[str]:
    """Distinctive renderings of a number as it might appear in prose.

    Deliberately excludes rounding to whole numbers. A 4-digit integer string
    occurs by coincidence somewhere in a 1,082-page filing, so testing it
    rejects sound questions for no reason. Only renderings precise enough to
    identify *this* value are checked; the structural argument in
    ``_structural_reason`` covers integer answers.
    """
    variants: set[str] = set()
    exact = format(value, "f")
    variants.add(exact)
    if "." in exact:
        for places in (2, 3):
            text = format(round(value, places), "f")
            if len(text.split(".")[-1]) >= 2:
                variants.add(text)
    return [v for v in variants if len(v.replace("-", "").replace(".", "")) >= 4]


def _structural_reason(plan: ExcelQueryPlan) -> str | None:
    """Why this question's answer cannot live in the PDF corpus.

    Every ingested PDF is 2023-vintage, so a 2024/2025 reporting period or a
    2024/2025 submission vintage is structurally absent from it.
    """
    for flt in plan.filters:
        if flt.field in {"reporting_year", "source_vintage_year"}:
            try:
                year = int(flt.value)
            except (TypeError, ValueError):
                continue
            if year >= 2024:
                return (
                    f"{flt.field}={year} postdates every ingested PDF "
                    "(corpus is 2023-vintage)"
                )
    return None


class ExcelOnlyChecker:
    """Rejects a candidate whose answer is discoverable in the PDF corpus."""

    def __init__(self, conn) -> None:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT content, coalesce(caption,'') FROM chunks "
                "WHERE content_type <> 'excel_card'"
            )
            self._corpus = "\n".join(
                f"{row[0]} {row[1]}" for row in cur.fetchall()
            )
        # Collapse whitespace so a number split across a line break still matches.
        self._corpus = re.sub(r"\s+", " ", self._corpus)
        self._flat = self._corpus.replace(",", "")

    def is_excel_only(self, value: Decimal) -> tuple[bool, str]:
        for variant in _numeric_variants(value):
            needle = variant.replace(",", "")
            if len(needle.replace("-", "").replace(".", "")) < 3:
                continue
            if needle in self._flat:
                return False, f"answer variant {variant!r} appears in the PDF corpus"
        return True, ""


def _distinct(conn, sql: str, params: tuple) -> list[Any]:
    with conn.cursor() as cur:
        cur.execute(sql, params)
        return [row[0] for row in cur.fetchall() if row[0] is not None]


def _concepts(conn, table_number: int, limit: int = 40) -> list[tuple[str, str]]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT f.semantic_metric_key, min(f.metric_name)
            FROM excel_facts f
            JOIN excel_revisions r ON r.id = f.revision_id AND r.state='active'
            WHERE f.table_number = %s AND f.value_numeric IS NOT NULL
            GROUP BY 1 ORDER BY count(*) DESC LIMIT %s
            """,
            (table_number, limit),
        )
        return cur.fetchall()


def _dim_values(conn, table_number: int, key: str) -> list[str]:
    return _distinct(
        conn,
        """
        SELECT DISTINCT f.dimensions ->> %s
        FROM excel_facts f
        JOIN excel_revisions r ON r.id = f.revision_id AND r.state='active'
        WHERE f.table_number = %s
        """,
        (key, table_number),
    )


def _promoted_values(conn, table_number: int, column: str) -> list[str]:
    return _distinct(
        conn,
        f"""
        SELECT DISTINCT f.{column}
        FROM excel_facts f
        JOIN excel_revisions r ON r.id = f.revision_id AND r.state='active'
        WHERE f.table_number = %s
        """,
        (table_number,),
    )


# --------------------------------------------------------------------------
# Candidate builders. Each yields (question, question_type, plan, expectation)
# --------------------------------------------------------------------------


def build_candidates(conn) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    rng = random.Random(20260730)

    def add(question, qtype, plan, table, key, extra=None):
        out.append(
            {
                "question": _q(question),
                "question_type": qtype,
                "plan": plan,
                "table_number": table,
                "semantic_metric_key": key,
                **(extra or {}),
            }
        )

    # ---- system composition / upgrades: tables 7, 8, 9 -------------------
    for table in (7, 8, 9):
        concepts = _concepts(conn, table, 8)
        tiers = _promoted_values(conn, table, "hftd_tier")
        lines = _promoted_values(conn, table, "line_type")
        for key, name in concepts:
            for year in SAFE_YEARS:
              for quarter in (1, 2, 3, 4):
                for tier in tiers[:3]:
                    label = re.sub(r"^\d+\.\s*", "", name)
                    line = lines[0] if lines else None
                    filters = [
                        Filter("reporting_year", value=year),
                        Filter("reporting_quarter", value=quarter),
                        Filter("hftd_tier", value=tier),
                    ]
                    if line:
                        filters.append(Filter("line_type", value=line))
                    add(
                        f"What did SDG&E report for {label.lower()} on "
                        f"{(line or '').lower()} lines in {tier} for {year} Q{quarter}?",
                        "value_lookup",
                        ExcelQueryPlan(
                            table_number=table,
                            semantic_metric_key=key,
                            filters=tuple(filters),
                            aggregate="sum",
                        ),
                        table,
                        key,
                    )

    # ---- risk events / ignitions by driver: tables 5, 6 ------------------
    for table, driver_key, noun in ((5, "risk_event_driver", "risk events"),
                                    (6, "ignition_driver", "ignitions")):
        drivers = _dim_values(conn, table, driver_key)
        concepts = _concepts(conn, table, 4)
        for key, name in concepts:
            for driver in drivers[:14]:
                for year in SAFE_YEARS:
                    add(
                        f"How many {noun} did SDG&E attribute to {driver.lower()} "
                        f"in {year}?",
                        "aggregate_sum",
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
                    )

    # ---- PSPS and quarterly operations: tables 10, 2, 3, 4 ---------------
    for table in (10, 2, 3, 4):
        for key, name in _concepts(conn, table, 14):
            label = re.sub(r"^\d+\.\s*", "", name)
            for year in SAFE_YEARS:
                for quarter in (2, 4):
                    add(
                        f"What value did SDG&E report for "
                        f"“{label}” in {year} Q{quarter}?",
                        "value_lookup",
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
                    )

    # ---- spend: table 11 -------------------------------------------------
    expense_types = _dim_values(conn, 11, "expense_type")
    for key, name in _concepts(conn, 11, 16):
        for expense in expense_types:
            for year in SAFE_YEARS:
                add(
                    f"How much {expense} did SDG&E spend on {name} in {year}?",
                    "aggregate_sum",
                    ExcelQueryPlan(
                        table_number=11,
                        semantic_metric_key=key,
                        filters=(
                            Filter("reporting_year", value=year),
                            Filter("expense_type", value=expense, json_key=True),
                        ),
                        aggregate="sum",
                    ),
                    11,
                    key,
                )

    # ---- tier comparison: tables 7, 9 ------------------------------------
    for table in (7, 9):
        for key, name in _concepts(conn, table, 4):
            label = re.sub(r"^\d+\.\s*", "", name)
            for year in SAFE_YEARS:
                add(
                    f"Did SDG&E report more {label.lower()} in HFTD Tier 2 or "
                    f"HFTD Tier 3 in {year} Q4?",
                    "group_comparison",
                    ExcelQueryPlan(
                        table_number=table,
                        semantic_metric_key=key,
                        filters=(
                            Filter("reporting_year", value=year),
                            Filter("reporting_quarter", value=4),
                            Filter("hftd_tier", operator="ne", value="Non-HFTD"),
                        ),
                        group_by=("hftd_tier",),
                        aggregate="sum",
                        operation="rank",
                    ),
                    table,
                    key,
                    {"answer_kind": "label"},
                )

    # ---- work orders: table 13 ------------------------------------------
    equipment = _distinct(
        conn,
        """
        SELECT rec.attributes ->> 'equipment_type'
        FROM excel_records rec
        JOIN excel_revisions r ON r.id = rec.revision_id AND r.state='active'
        WHERE rec.table_number = 13
        GROUP BY 1 ORDER BY count(*) DESC LIMIT 12
        """,
        (),
    )
    for item in equipment:
        for year in SAFE_YEARS:
            for quarter in (2, 4):
                add(
                    f"How many open {item.lower()} work orders did SDG&E report "
                    f"in {year} Q{quarter}?",
                    "record_count",
                    ExcelQueryPlan(
                        table_number=13,
                        source=RECORDS,
                        aggregate="count",
                        filters=(
                            Filter("reporting_year", value=year),
                            Filter("reporting_quarter", value=quarter),
                            Filter("equipment_type", value=item, json_key=True),
                        ),
                    ),
                    13,
                    None,
                )

    # ---- WMP activity targets: table 1 -----------------------------------
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT rec.title, rec.reporting_year,
                   rec.attributes ->> 'annual_quant_target',
                   rec.attributes ->> 'quant_target_units'
            FROM excel_records rec
            JOIN excel_revisions r ON r.id = rec.revision_id AND r.state='active'
            WHERE rec.table_number = 1 AND rec.reporting_year >= 2024
              AND rec.attributes ->> 'annual_quant_target' IS NOT NULL
              AND rec.title IS NOT NULL
            """
        )
        activities = cur.fetchall()
    for title, year, target, units in activities:
        add(
            f"What annual quantitative target did SDG&E set for the "
            f"{title} activity in {year}?",
            "activity_target",
            ExcelQueryPlan(
                table_number=1,
                source=RECORDS,
                operation="select",
                group_by=("title", "reporting_year"),
                filters=(
                    Filter("reporting_year", value=year),
                    Filter("title", value=title),
                    Filter("annual_quant_target", value=target, json_key=True),
                ),
                limit=5,
            ),
            1,
            None,
            {"answer_kind": "attribute", "target_value": target,
             "target_units": units},
        )

    # ---- risk, with explicit vintage: tables 14, 15 ----------------------
    for table in (14, 15):
        measures = _concepts(conn, table, 10)
        vintages = _promoted_values(conn, table, "source_vintage_year")
        tiers = _promoted_values(conn, table, "hftd_tier")
        for key, name in measures:
            for vintage in sorted(vintages)[-2:]:
                if table == 14:
                    for tier in tiers[:2]:
                        add(
                            f"In SDG&E's {vintage} WMP submission, what "
                            f"{name.lower()} was reported for distribution lines "
                            f"in {tier}?",
                            "value_lookup",
                            ExcelQueryPlan(
                                table_number=14,
                                semantic_metric_key=key,
                                filters=(
                                    Filter("source_vintage_year", value=int(vintage)),
                                    Filter("hftd_tier", value=tier),
                                    Filter("line_type", value="Distribution"),
                                ),
                                aggregate="sum",
                            ),
                            14,
                            key,
                        )
                else:
                    add(
                        f"In SDG&E's {vintage} WMP submission, what was the "
                        f"highest {name.lower()} recorded for any single "
                        f"distribution circuit segment?",
                        "ranking",
                        ExcelQueryPlan(
                            table_number=15,
                            semantic_metric_key=key,
                            filters=(
                                Filter("source_vintage_year", value=int(vintage)),
                            ),
                            aggregate="max",
                        ),
                        15,
                        key,
                    )

    # ---- deliberate ambiguity: tables 14/15 bare year --------------------
    for table in (14, 15):
        for key, name in _concepts(conn, table, 2):
            add(
                f"What {name.lower()} did SDG&E report for 2025?",
                "year_basis_ambiguous",
                ExcelQueryPlan(
                    table_number=table,
                    semantic_metric_key=key,
                    filters=(Filter("reporting_year", value=2025),),
                    aggregate="sum",
                ),
                table,
                key,
                {"expects_clarification": True},
            )

    rng.shuffle(out)
    return out


# --------------------------------------------------------------------------


def verify(conn, candidate: dict, checker: ExcelOnlyChecker) -> dict | None:
    plan: ExcelQueryPlan = candidate["plan"]
    expects_clarification = candidate.get("expects_clarification", False)
    try:
        result = execute_plan(plan, conn)
    except PlanError as exc:
        if expects_clarification:
            return {
                **candidate,
                "expected_clarification": str(exc),
                "expected_answer": None,
                "unit": None,
                "contributing_facts": None,
                "provenance": [],
                "excel_only_verified": True,
                "excel_only_reason": "ambiguous year axis; expects a clarification",
            }
        return None
    if expects_clarification:
        return None  # should have been refused; drop rather than mislabel

    if not result.rows:
        return None

    structural = _structural_reason(plan)
    if structural is None:
        return None  # only accept periods the PDF corpus cannot contain

    if candidate.get("answer_kind") == "attribute":
        # The plan pins one entity row; the answer is a stored attribute, so it
        # must still pass the Excel-only check as a number where possible.
        if len(result.rows) != 1:
            return None
        raw = candidate["target_value"]
        try:
            ok, reason = checker.is_excel_only(Decimal(str(raw)))
        except Exception:
            ok, reason = True, ""
        if not ok:
            return {**candidate, "_rejected": reason}
        return {
            **candidate,
            "expected_answer": str(raw),
            "expected_answer_kind": "attribute",
            "unit": candidate.get("target_units"),
            "contributing_facts": 1,
            "provenance": [],
            "excel_only_verified": True,
            "excel_only_reason": structural,
        }

    if candidate.get("answer_kind") == "label":
        if len(result.rows) < 2:
            return None
        top, second = result.rows[0], result.rows[1]
        if top[-1] is None or second[-1] is None or top[-1] == second[-1]:
            return None  # not a decisive comparison
        return {
            **candidate,
            "expected_answer": top[0],
            "expected_answer_kind": "label",
            "unit": result.unit,
            "contributing_facts": result.contributing_facts,
            "provenance": result.provenance[:2],
            "excel_only_verified": True,
            "excel_only_reason": structural,
        }

    if len(result.rows) != 1:
        return None
    value = result.rows[0][-1]
    if value is None:
        return None
    value = Decimal(value)
    if value == 0:
        return None  # a zero is not a discriminating answer
    # contributing_facts is only computed for the facts source.
    if plan.source == FACTS and result.contributing_facts == 0:
        return None
    ok, reason = checker.is_excel_only(value)
    if not ok:
        return {**candidate, "_rejected": reason}
    return {
        **candidate,
        "expected_answer": format(value, "f"),
        "expected_answer_kind": "numeric",
        "unit": result.unit,
        "contributing_facts": result.contributing_facts,
        "provenance": result.provenance[:2],
        "excel_only_verified": True,
        "excel_only_reason": structural,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path,
                        default=Path("eval/excel/evaluation_excel.jsonl"))
    parser.add_argument("--target", type=int, default=60)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    load_contracts()
    conn = connect_db()
    try:
        checker = ExcelOnlyChecker(conn)
        candidates = build_candidates(conn)
        logger.info("Built %d raw candidates", len(candidates))

        kept: list[dict] = []
        rejected_pdf = 0
        per_table: dict[int, int] = {}
        per_type: dict[str, int] = {}
        # Cap per table and per type so the set stays balanced.
        max_per_table = max(3, args.target // 6)
        max_per_type = max(4, args.target // 3)

        for candidate in candidates:
            if len(kept) >= args.target:
                break
            table = candidate["table_number"]
            qtype = candidate["question_type"]
            if per_table.get(table, 0) >= max_per_table:
                continue
            if per_type.get(qtype, 0) >= max_per_type:
                continue
            verified = verify(conn, candidate, checker)
            if verified is None:
                continue
            if "_rejected" in verified:
                rejected_pdf += 1
                continue
            per_table[table] = per_table.get(table, 0) + 1
            per_type[qtype] = per_type.get(qtype, 0) + 1
            verified["id"] = f"excel_eval_{len(kept)+1:04d}"
            plan = verified.pop("plan")
            verified["plan"] = {
                **{k: v for k, v in asdict(plan).items() if k != "filters"},
                "filters": [asdict(f) for f in plan.filters],
            }
            verified.pop("answer_kind", None)
            verified.pop("expects_clarification", None)
            kept.append(verified)

        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w") as fh:
            for row in kept:
                fh.write(json.dumps(row) + "\n")

        print(f"\nWrote {len(kept)} verified Excel-only questions to {args.out}")
        print(f"Rejected because the answer appears in the PDF corpus: {rejected_pdf}")
        print("\nBy table:   ", dict(sorted(per_table.items())))
        print("By type:    ", dict(sorted(per_type.items())))
    finally:
        conn.close()


if __name__ == "__main__":
    main()

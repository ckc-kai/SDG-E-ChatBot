"""Post-ingest self-check for the Excel corpus.

Re-runs the acceptance checks from the ingestion plan against whatever is
currently in the database and prints a PASS/FAIL line per check. Intended to be
run after every ingest, including after new cleaned files arrive.

    uv run python -m eval.verify_excel_ingest
"""

from __future__ import annotations

import csv
from decimal import Decimal, getcontext
from pathlib import Path

from retrieval.ingest.excel.contracts import load_contracts
from retrieval.utils import connect_db

# PostgreSQL numeric keeps more significant digits than Decimal's 28-digit
# default context. Raise it so this check compares against an equally exact
# reference instead of a rounded one.
getcontext().prec = 80
csv.field_size_limit(10**9)

INPUT_DIR = Path("excel_cleaning/cleaned_csv_rag_ready")
SENSITIVE_MARKERS = ("@sdge.com", "person_in_charge")


class Checker:
    def __init__(self) -> None:
        self.failures = 0

    def check(self, name: str, ok: bool, detail: str = "") -> None:
        status = "PASS" if ok else "FAIL"
        if not ok:
            self.failures += 1
        suffix = f"  {detail}" if detail else ""
        print(f"[{status}] {name}{suffix}")


def _csv_totals(contracts) -> tuple[int, int, int]:
    """Expected source rows, facts, and records straight from the CSVs."""
    rows = facts = records = 0
    for path in sorted(INPUT_DIR.glob("*.csv")):
        table = int(path.name.split("table")[1][:2])
        contract = contracts.for_table(table)
        with path.open(encoding="utf-8-sig", newline="") as fh:
            n = sum(1 for _ in csv.DictReader(fh))
        rows += n
        if contract.family == "long_metric":
            facts += n
        elif contract.family == "wide_risk":
            facts += n * len(contract.melt_columns)
        else:
            records += n
    return rows, facts, records


def _csv_sum(table: int, columns: list[str]) -> tuple[Decimal, int]:
    path = INPUT_DIR / f"sdge_table{table:02d}_rag_ready.csv"
    total, count = Decimal(0), 0
    with path.open(encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            for column in columns:
                raw = (row.get(column) or "").strip()
                if not raw:
                    continue
                try:
                    total += Decimal(raw.replace(",", "").replace("$", ""))
                    count += 1
                except Exception:
                    pass
    return total, count


def main() -> None:
    contracts = load_contracts()
    checker = Checker()
    conn = connect_db()
    want_rows, want_facts, want_records = _csv_totals(contracts)

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT
              (SELECT count(*) FROM excel_facts f
                 JOIN excel_revisions r ON r.id=f.revision_id AND r.state='active'),
              (SELECT count(*) FROM excel_records rec
                 JOIN excel_revisions r ON r.id=rec.revision_id AND r.state='active'),
              (SELECT count(*) FROM chunks WHERE content_type='excel_card'),
              (SELECT count(*) FROM excel_sources),
              (SELECT count(*) FROM excel_revisions WHERE state='active')
            """
        )
        facts, records, cards, sources, active = cur.fetchone()

        checker.check(
            "fact count reconciles to CSVs",
            facts == want_facts,
            f"db={facts} expected={want_facts}",
        )
        checker.check(
            "record count reconciles to CSVs",
            records == want_records,
            f"db={records} expected={want_records}",
        )
        checker.check(
            "one active revision per source",
            sources == active,
            f"sources={sources} active={active}",
        )
        checker.check("cards present", cards > 0, f"cards={cards}")

        cur.execute(
            "SELECT count(*) FROM chunks c LEFT JOIN excel_revisions r "
            "ON r.id=c.excel_revision_id "
            "WHERE c.content_type='excel_card' AND coalesce(r.state,'') <> 'active'"
        )
        checker.check(
            "every card points at an active revision", cur.fetchone()[0] == 0
        )

        cur.execute(
            "SELECT count(*) FROM excel_facts WHERE value_numeric IS NOT NULL "
            "AND value_raw IS NOT NULL AND value_numeric <> value_raw::numeric"
        )
        checker.check("numeric values round-trip exactly", cur.fetchone()[0] == 0)

        # Sensitive Table 1 fields must not exist in any retrieval surface.
        leaks = 0
        for marker in SENSITIVE_MARKERS:
            cur.execute(
                """
                SELECT
                  (SELECT count(*) FROM excel_records
                     WHERE attributes::text ILIKE %s
                        OR coalesce(searchable_text,'') ILIKE %s
                        OR provenance::text ILIKE %s)
                + (SELECT count(*) FROM chunks WHERE content_type='excel_card'
                     AND (content ILIKE %s OR coalesce(caption,'') ILIKE %s
                          OR structured_data::text ILIKE %s))
                """,
                tuple([f"%{marker}%"] * 6),
            )
            leaks += cur.fetchone()[0]
        checker.check("no personal data in facts, records, or cards", leaks == 0)

        cur.execute(
            "SELECT coalesce(sum(array_length(unknown_columns,1)),0) "
            "FROM excel_revisions WHERE state='active'"
        )
        unknown = cur.fetchone()[0]
        checker.check(
            "every source column is claimed by a contract",
            unknown == 0,
            f"unknown={unknown}",
        )

        # Aggregates must match an independent exact computation.
        targets: list[tuple[int, list[str]]] = []
        for table_number, contract in sorted(contracts.tables.items()):
            if contract.family == "long_metric":
                targets.append((table_number, [contract.value_column or ""]))
            elif contract.family == "wide_risk":
                targets.append((table_number, list(contract.melt_columns)))
        mismatches = []
        for table_number, columns in targets:
            want_sum, want_count = _csv_sum(table_number, columns)
            cur.execute(
                """
                SELECT coalesce(sum(f.value_numeric),0), count(f.value_numeric)
                FROM excel_facts f
                JOIN excel_revisions r ON r.id=f.revision_id AND r.state='active'
                WHERE f.table_number = %s
                """,
                (table_number,),
            )
            got_sum, got_count = cur.fetchone()
            if Decimal(got_sum) != want_sum or got_count != want_count:
                mismatches.append(str(table_number))
        checker.check(
            "aggregates match independent Decimal sums",
            not mismatches,
            f"mismatched tables: {', '.join(mismatches)}" if mismatches else "",
        )

        # Figure links (Phase 3) must not leave stale embeddings behind.
        cur.execute(
            "SELECT count(*) FROM chunks WHERE content_type='figure' "
            "AND content LIKE '%Describing context:%' "
            "AND content_hash <> encode(sha256(content::bytea), 'hex')"
        )
        stale = cur.fetchone()[0]
        checker.check(
            "enriched figures were re-embedded (content_hash matches content)",
            stale == 0,
            f"stale={stale}",
        )

        cur.execute("SELECT count(*) FROM chunks WHERE content_type <> 'excel_card'")
        pdf_chunks = cur.fetchone()[0]
        print(f"\nPDF chunks present: {pdf_chunks}")
        print(f"Excel cards: {cards}   facts: {facts}   records: {records}")

    conn.close()
    if checker.failures:
        raise SystemExit(f"\n{checker.failures} check(s) FAILED")
    print("\nAll Excel ingest checks passed.")


if __name__ == "__main__":
    main()

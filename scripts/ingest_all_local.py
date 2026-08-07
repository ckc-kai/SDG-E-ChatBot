"""Ingest the complete local corpus with one command.

Sources:
* every PDF and XLSX/XLSM below ``files/``;
* every reviewed ``sdge_tableNN_rag_ready.csv`` committed under
  ``excel_cleaning/cleaned_csv_rag_ready``.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from retrieval.ingest.excel.contracts import load_contracts
from retrieval.ingest.excel.ingest import ingest_file
from retrieval.ingest.excel.schema import migrate
from retrieval.ingest.excel.workbook import ingest_directory
from retrieval.ingest.pdf.ingest import ingest_pdf
from retrieval.object_storage import get_object_storage
from retrieval.setup_db import setup_database
from retrieval.utils import connect_db, get_embedding_model


logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--files-dir", type=Path, default=Path("files"))
    parser.add_argument(
        "--reviewed-excel-dir",
        type=Path,
        default=Path("excel_cleaning/cleaned_csv_rag_ready"),
    )
    parser.add_argument("--force-excel", action="store_true")
    parser.add_argument(
        "--skip-workbook",
        action="append",
        default=[],
        metavar="FILENAME",
        help=(
            "Skip a workbook filename during generic ingestion. Repeat for "
            "multiple files; useful when a large workbook needs a dedicated adapter."
        ),
    )
    parser.add_argument(
        "--pdf-mode",
        choices=("narrative", "structured"),
        default="narrative",
        help=(
            "PDF extraction mode. 'narrative' is the safe local default; "
            "'structured' also runs Docling table/figure extraction."
        ),
    )
    parser.add_argument(
        "--skip-existing-pdfs",
        action="store_true",
        help="Skip PDF filenames already present in the database.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    pdfs = sorted(args.files_dir.rglob("*.pdf"))
    skipped_workbooks = set(args.skip_workbook)
    workbooks = sorted(
        path
        for path in [*args.files_dir.rglob("*.xlsx"), *args.files_dir.rglob("*.xlsm")]
        if path.name not in skipped_workbooks
    )
    csvs = sorted(args.reviewed_excel_dir.glob("*.csv"))
    if not pdfs and not workbooks and not csvs:
        raise SystemExit("No PDF, workbook, or reviewed CSV sources were found")

    setup_database()
    model = get_embedding_model()
    storage = get_object_storage()
    contracts = load_contracts()
    conn = connect_db()
    failures: list[str] = []
    try:
        applied = migrate(conn)
        if applied:
            logger.info("Applied migrations: %s", ", ".join(applied))

        for path in pdfs:
            try:
                ingest_pdf(
                    path,
                    conn,
                    model,
                    storage=storage,
                    structured_enabled=args.pdf_mode == "structured",
                    skip_existing=args.skip_existing_pdfs,
                )
            except Exception as exc:
                conn.rollback()
                logger.exception("PDF ingest failed: %s", path.name)
                failures.append(f"{path.name}: {exc}")

        workbook_results = ingest_directory(
            args.files_dir,
            conn,
            model,
            force=args.force_excel,
            exclude_names=skipped_workbooks,
        )
        failures.extend(
            f"{name}: workbook ingest failed"
            for name, status in workbook_results.items()
            if status == "failed"
        )

        for path in csvs:
            try:
                report = ingest_file(
                    path,
                    conn,
                    model,
                    contracts,
                    force=args.force_excel,
                )
                logger.info(
                    "Reviewed Excel %s: %s (%d cards)",
                    path.name,
                    report.status,
                    report.cards,
                )
            except Exception as exc:
                conn.rollback()
                logger.exception("Reviewed Excel ingest failed: %s", path.name)
                failures.append(f"{path.name}: {exc}")
    finally:
        conn.close()

    print(
        f"Sources discovered: {len(pdfs)} PDF, {len(workbooks)} workbook, "
        f"{len(csvs)} reviewed CSV"
    )
    if failures:
        print("Failures:")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(f"{len(failures)} source file(s) failed to ingest")
    print("All discovered sources ingested successfully.")


if __name__ == "__main__":
    main()

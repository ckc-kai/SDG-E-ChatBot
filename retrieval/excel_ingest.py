"""Revision-safe ingest of the cleaned SDG&E QDR CSVs.

Each cleaned file maps to one logical source (``sdge_table07``). Every changed
file or ingestion recipe produces a new immutable revision. Facts, records, and
semantic cards are loaded against a ``staging`` revision and promoted in one
short transaction, so a retrieved card can never point at a different active
fact revision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from psycopg2.extras import Json, execute_values
from tqdm import tqdm

from retrieval.contextual_embeddings import (
    CONTEXTUAL_EMBEDDING_RECIPE,
    contextual_embedding_text_for_model,
)
from retrieval.excel_cards import (
    CARD_BUILDER_VERSION,
    CARD_COUNT_HARD_CAP,
    CARD_COUNT_TARGET,
    Card,
    build_cards,
    fit_cards_to_model,
)
from retrieval.excel_contracts import (
    ContractError,
    ContractSet,
    TableContract,
    inventory_columns,
    load_contracts,
    table_number_from_filename,
)
from retrieval.excel_schema import migrate
from retrieval.excel_transform import (
    Fact,
    Record,
    read_rows,
    transform,
)
from retrieval.failure_log import get_failure_logger
from retrieval.utils import connect_db, embedding_config, get_embedding_model

logger = logging.getLogger(__name__)
log_failure = get_failure_logger("excel_ingest")

TRANSFORM_VERSION = "excel-transform-v1"
CARD_CONTENT_TYPE = "excel_card"
CARD_EXTRACTOR = "excel-card-v1"
DEFAULT_INPUT_DIR = Path("excel_cleaning/cleaned_csv_rag_ready")


@dataclass
class LoadReport:
    file_name: str
    table_number: int
    status: str
    source_rows: int = 0
    facts: int = 0
    records: int = 0
    cards: int = 0
    unknown_columns: tuple[str, ...] = ()
    revision_id: int | None = None
    detail: str = ""


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ingest_signature(contracts: ContractSet) -> str:
    """Hash every setting that changes stored rows, cards, or vectors."""
    config = embedding_config()
    payload = {
        "recipe": "excel-ingest-v1",
        "transform": TRANSFORM_VERSION,
        "card_builder": CARD_BUILDER_VERSION,
        "contract_version": contracts.version,
        "embedding_provider": config.get("provider", "sentence_transformers"),
        "embedding_model": config["name"],
        "embedding_dimensions": config["dimensions"],
        "contextual_recipe": CONTEXTUAL_EMBEDDING_RECIPE,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def expected_counts(
    contract: TableContract, source_rows: int
) -> tuple[int, int]:
    """Row counts the transform must produce, used for reconciliation."""
    if contract.family == "long_metric":
        return source_rows, 0
    if contract.family == "wide_risk":
        return source_rows * len(contract.melt_columns), 0
    return 0, source_rows


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------


def _ensure_source(cur, contract: TableContract, source_key: str) -> int:
    cur.execute(
        """
        INSERT INTO excel_sources (source_key, table_number, title)
        VALUES (%s, %s, %s)
        ON CONFLICT (source_key)
        DO UPDATE SET title = EXCLUDED.title
        RETURNING id
        """,
        (source_key, contract.table_number, contract.title),
    )
    return cur.fetchone()[0]


def _active_revision(cur, source_id: int) -> tuple[int, str, str] | None:
    cur.execute(
        """
        SELECT id, source_hash, ingest_signature
        FROM excel_revisions
        WHERE source_id = %s AND state = 'active'
        """,
        (source_id,),
    )
    row = cur.fetchone()
    return (row[0], row[1], row[2]) if row else None


def _insert_staging_revision(
    cur,
    source_id: int,
    path: Path,
    source_hash: str,
    signature: str,
    contracts: ContractSet,
    inventory,
    source_rows: int,
) -> int:
    cur.execute(
        """
        INSERT INTO excel_revisions (
            source_id, source_file, source_hash, ingest_signature,
            contract_version, state, column_inventory, unknown_columns,
            source_row_count, source_uri
        )
        VALUES (%s, %s, %s, %s, %s, 'staging', %s, %s, %s, %s)
        RETURNING id
        """,
        (
            source_id,
            path.name,
            source_hash,
            signature,
            contracts.version,
            Json(inventory.as_dict()),
            list(inventory.unknown),
            source_rows,
            str(path),
        ),
    )
    return cur.fetchone()[0]


_FACT_COLUMNS = (
    "revision_id, table_number, record_id, source_metric_number, series_id, "
    "semantic_metric_key, metric_name, measure_name, utility_id, "
    "reporting_year, reporting_quarter, source_vintage_year, year_basis, "
    "period_end_date, hftd_tier, line_type, dimensions, unit, "
    "value_numeric, value_raw, value_text, comments, blank_meaning, provenance"
)


def _fact_row(revision_id: int, fact: Fact) -> tuple:
    return (
        revision_id,
        fact.table_number,
        fact.record_id,
        fact.source_metric_number,
        fact.series_id,
        fact.semantic_metric_key,
        fact.metric_name,
        fact.measure_name,
        fact.utility_id,
        fact.reporting_year,
        fact.reporting_quarter,
        fact.source_vintage_year,
        fact.year_basis,
        fact.period_end_date,
        fact.hftd_tier,
        fact.line_type,
        Json(fact.dimensions),
        fact.unit,
        fact.value_numeric,
        fact.value_raw,
        fact.value_text,
        fact.comments,
        fact.blank_meaning,
        Json(fact.provenance),
    )


_RECORD_COLUMNS = (
    "revision_id, table_number, record_id, entity_key, entity_type, title, "
    "utility_id, reporting_year, reporting_quarter, hftd_tier, line_type, "
    "date_start, date_due, date_end, status, attributes, searchable_text, "
    "provenance"
)


def _record_row(revision_id: int, record: Record) -> tuple:
    return (
        revision_id,
        record.table_number,
        record.record_id,
        record.entity_key,
        record.entity_type,
        record.title,
        record.utility_id,
        record.reporting_year,
        record.reporting_quarter,
        record.hftd_tier,
        record.line_type,
        record.date_start,
        record.date_due,
        record.date_end,
        record.status,
        Json(record.attributes),
        record.searchable_text,
        Json(record.provenance),
    )


def _bulk_load(cur, revision_id: int, facts: Sequence[Fact], records: Sequence[Record]) -> None:
    if facts:
        execute_values(
            cur,
            f"INSERT INTO excel_facts ({_FACT_COLUMNS}) VALUES %s",
            [_fact_row(revision_id, fact) for fact in facts],
            page_size=1000,
        )
    if records:
        execute_values(
            cur,
            f"INSERT INTO excel_records ({_RECORD_COLUMNS}) VALUES %s",
            [_record_row(revision_id, record) for record in records],
            page_size=1000,
        )


def _ensure_document(
    cur, source_key: str, source_hash: str, signature: str, card_count: int
) -> int:
    cur.execute(
        """
        INSERT INTO documents (
            filename, page_count, content_hash, ingest_signature, chunk_counts
        )
        VALUES (%s, 0, %s, %s, %s)
        ON CONFLICT (filename)
        DO UPDATE SET content_hash = EXCLUDED.content_hash,
                      ingest_signature = EXCLUDED.ingest_signature,
                      chunk_counts = EXCLUDED.chunk_counts,
                      ingested_at = now()
        RETURNING id
        """,
        (
            f"{source_key}.csv",
            source_hash,
            signature,
            Json({CARD_CONTENT_TYPE: card_count}),
        ),
    )
    return cur.fetchone()[0]


def _replace_cards(
    cur,
    document_id: int,
    revision_id: int,
    cards: Sequence[Card],
    raw_embeddings: Sequence[Sequence[float]],
    contextual_embeddings: Sequence[Sequence[float]],
    source_key: str,
) -> None:
    cur.execute(
        "DELETE FROM chunks WHERE document_id = %s AND content_type = %s",
        (document_id, CARD_CONTENT_TYPE),
    )
    if not cards:
        return
    config = embedding_config()
    rows = [
        (
            document_id,
            source_key,
            card.breadcrumb,
            None,
            0,
            0,
            index,
            CARD_CONTENT_TYPE,
            card.content,
            None,
            card.caption,
            Json(card.structured_data),
            None,
            None,
            len(card.content.split()),
            hashlib.sha256(card.content.encode()).hexdigest(),
            config.get("provider", "sentence_transformers"),
            config["name"],
            list(raw),
            config["name"],
            CONTEXTUAL_EMBEDDING_RECIPE,
            list(contextual),
            CARD_EXTRACTOR,
            revision_id,
        )
        for index, (card, raw, contextual) in enumerate(
            zip(cards, raw_embeddings, contextual_embeddings, strict=True)
        )
    ]
    execute_values(
        cur,
        """
        INSERT INTO chunks (
            document_id, sub_document, breadcrumb, section_number,
            page_start, page_end, chunk_index, content_type,
            content, retrieval_hint, caption, structured_data,
            object_key, media_type, token_count, content_hash,
            embedding_provider, embedding_model, embedding,
            contextual_embedding_model, contextual_embedding_recipe,
            contextual_embedding, extractor, excel_revision_id
        ) VALUES %s
        """,
        rows,
    )


# --------------------------------------------------------------------------
# Embedding
# --------------------------------------------------------------------------


def embed_cards(model, cards: Sequence[Card], source_key: str) -> tuple[list, list]:
    if not cards:
        return [], []
    raw = model.encode(
        [card.content for card in cards],
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    contextual_inputs = [
        contextual_embedding_text_for_model(
            f"{source_key}.csv",
            card.breadcrumb,
            card.content,
            model.tokenizer,
            model.max_seq_length,
        )
        for card in cards
    ]
    contextual = model.encode(
        contextual_inputs, normalize_embeddings=True, show_progress_bar=False
    )
    return [v.tolist() for v in raw], [v.tolist() for v in contextual]


# --------------------------------------------------------------------------
# Per-file ingest
# --------------------------------------------------------------------------


def ingest_file(
    path: Path,
    conn,
    model,
    contracts: ContractSet,
    *,
    dry_run: bool = False,
    force: bool = False,
) -> LoadReport:
    table_number = table_number_from_filename(path.name)
    contract = contracts.for_table(table_number)
    source_key = f"sdge_table{table_number:02d}"
    source_hash = file_hash(path)
    signature = ingest_signature(contracts)

    headers, rows = read_rows(path)
    inventory = inventory_columns(contract, headers)

    if inventory.missing_required:
        raise ContractError(
            f"{path.name}: missing required column(s): "
            + ", ".join(inventory.missing_required)
        )
    if inventory.excluded_present:
        logger.info(
            "%s: %d sensitive column(s) present and deliberately not loaded: %s",
            path.name,
            len(inventory.excluded_present),
            ", ".join(inventory.excluded_present),
        )
    if inventory.unknown:
        logger.warning(
            "%s: %d unreviewed column(s) recorded in diagnostics but NOT loaded: %s",
            path.name,
            len(inventory.unknown),
            ", ".join(inventory.unknown),
        )

    facts, records = transform(contracts, contract, rows)
    want_facts, want_records = expected_counts(contract, len(rows))
    if (len(facts), len(records)) != (want_facts, want_records):
        raise ContractError(
            f"{path.name}: reconciliation failed. Expected {want_facts} facts / "
            f"{want_records} records from {len(rows)} source rows, got "
            f"{len(facts)} / {len(records)}."
        )

    cards = fit_cards_to_model(
        build_cards(contracts, contract, facts, records),
        model.tokenizer,
        model.max_seq_length,
    )
    if len(cards) > CARD_COUNT_HARD_CAP:
        raise ContractError(
            f"{path.name}: {len(cards)} cards exceeds the hard cap of "
            f"{CARD_COUNT_HARD_CAP}. Tighten the semantic grouping."
        )

    if dry_run:
        return LoadReport(
            file_name=path.name,
            table_number=table_number,
            status="dry-run",
            source_rows=len(rows),
            facts=len(facts),
            records=len(records),
            cards=len(cards),
            unknown_columns=inventory.unknown,
        )

    with conn.cursor() as cur:
        source_id = _ensure_source(cur, contract, source_key)
        active = _active_revision(cur, source_id)
    conn.commit()

    if active and active[1:] == (source_hash, signature) and not force:
        # Hash equality alone is not enough: cards live in `chunks` and can be
        # removed independently of the revision row (schema reset, manual
        # cleanup). Rebuild when the active revision has lost its card set,
        # otherwise the corpus would stay silently unretrievable.
        with conn.cursor() as cur:
            cur.execute(
                "SELECT count(*) FROM chunks WHERE excel_revision_id = %s",
                (active[0],),
            )
            live_cards = cur.fetchone()[0]
        if live_cards == len(cards):
            return LoadReport(
                file_name=path.name,
                table_number=table_number,
                status="unchanged",
                source_rows=len(rows),
                revision_id=active[0],
                detail="source hash and ingest signature unchanged",
            )
        logger.warning(
            "%s: active revision %d has %d cards but %d are expected; rebuilding.",
            path.name,
            active[0],
            live_cards,
            len(cards),
        )

    # Staging load and embedding happen outside the activation transaction so
    # activation itself stays short.
    with conn.cursor() as cur:
        revision_id = _insert_staging_revision(
            cur, source_id, path, source_hash, signature, contracts, inventory, len(rows)
        )
        _bulk_load(cur, revision_id, facts, records)
        cur.execute(
            """
            UPDATE excel_revisions
            SET fact_count = %s, record_count = %s, card_count = %s
            WHERE id = %s
            """,
            (len(facts), len(records), len(cards), revision_id),
        )
    conn.commit()

    try:
        raw_embeddings, contextual_embeddings = embed_cards(model, cards, source_key)
        with conn.cursor() as cur:
            document_id = _ensure_document(
                cur, source_key, source_hash, signature, len(cards)
            )
            _replace_cards(
                cur,
                document_id,
                revision_id,
                cards,
                raw_embeddings,
                contextual_embeddings,
                source_key,
            )
            if active:
                cur.execute(
                    "UPDATE excel_revisions SET state = 'superseded' WHERE id = %s",
                    (active[0],),
                )
            cur.execute(
                """
                UPDATE excel_revisions
                SET state = 'active', activated_at = now()
                WHERE id = %s
                """,
                (revision_id,),
            )
            cur.execute(
                "UPDATE excel_sources SET active_revision_id = %s WHERE id = %s",
                (revision_id, source_id),
            )
        conn.commit()
    except Exception as exc:
        conn.rollback()
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE excel_revisions
                SET state = 'failed', error_summary = %s
                WHERE id = %s
                """,
                (str(exc)[:2000], revision_id),
            )
        conn.commit()
        raise

    return LoadReport(
        file_name=path.name,
        table_number=table_number,
        status="loaded",
        source_rows=len(rows),
        facts=len(facts),
        records=len(records),
        cards=len(cards),
        unknown_columns=inventory.unknown,
        revision_id=revision_id,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load cleaned SDG&E CSVs into facts, records, and cards."
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        nargs="?",
        default=DEFAULT_INPUT_DIR,
        help=f"Directory of cleaned CSVs (default: {DEFAULT_INPUT_DIR}).",
    )
    parser.add_argument(
        "--only", type=str, default=None, help="Substring filter on the filename."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate, transform, and report without writing to the database.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Create a new revision even when the source and recipe are unchanged.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    contracts = load_contracts()
    paths = sorted(args.input_dir.glob("*.csv"))
    if args.only:
        paths = [p for p in paths if args.only.lower() in p.name.lower()]
    if not paths:
        raise SystemExit(f"No CSV files matched in {args.input_dir}")

    model = get_embedding_model()
    conn = None
    if not args.dry_run:
        conn = connect_db()
        applied = migrate(conn)
        if applied:
            logger.info("Applied migrations: %s", ", ".join(applied))

    reports: list[LoadReport] = []
    try:
        for path in tqdm(paths, desc="Ingesting cleaned CSVs", unit="file"):
            try:
                reports.append(
                    ingest_file(
                        path, conn, model, contracts,
                        dry_run=args.dry_run, force=args.force,
                    )
                )
            except Exception as exc:
                if conn is not None:
                    conn.rollback()
                log_failure("excel_ingest", path.name, exc)
                logger.error("Failed to ingest %s: %s", path.name, exc)
                reports.append(
                    LoadReport(
                        file_name=path.name,
                        table_number=-1,
                        status="failed",
                        detail=str(exc),
                    )
                )
    finally:
        if conn is not None:
            conn.close()

    print("\n" + "=" * 78)
    print(f"{'file':34s} {'status':10s} {'rows':>7s} {'facts':>7s} {'recs':>7s} {'cards':>6s}")
    print("-" * 78)
    for report in reports:
        print(
            f"{report.file_name:34s} {report.status:10s} {report.source_rows:7d} "
            f"{report.facts:7d} {report.records:7d} {report.cards:6d}"
        )
    print("-" * 78)
    totals = (
        sum(r.source_rows for r in reports),
        sum(r.facts for r in reports),
        sum(r.records for r in reports),
        sum(r.cards for r in reports),
    )
    print(f"{'TOTAL':34s} {'':10s} {totals[0]:7d} {totals[1]:7d} {totals[2]:7d} {totals[3]:6d}")
    if totals[3] > CARD_COUNT_TARGET:
        print(
            f"\nNote: {totals[3]} cards exceeds the {CARD_COUNT_TARGET} target "
            f"(hard cap {CARD_COUNT_HARD_CAP})."
        )
    failed = [r for r in reports if r.status == "failed"]
    if failed:
        raise SystemExit(f"{len(failed)} file(s) failed to ingest")


if __name__ == "__main__":
    main()

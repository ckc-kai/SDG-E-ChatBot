"""Generic local XLSX ingestion into the shared retrieval index.

The reviewed ``sdge_tableNN`` CSV pipeline remains the authoritative path for
the quarterly-data-report tables.  This module covers additional workbooks in
the local ``files/`` corpus without pretending they follow those contracts.
Worksheet tables are rendered as Markdown with repeated headers and their exact
cell grids/ranges are stored in ``structured_data``. Formula text and cached
workbook values are both preserved when available. Irregular sheets still use
generated column labels, so no non-empty cell is discarded.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import xml.sax
from xml.sax.handler import ContentHandler
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Sequence

from openpyxl import load_workbook
from openpyxl.styles.numbers import BUILTIN_FORMATS, is_date_format
from openpyxl.utils.cell import coordinate_to_tuple, get_column_letter
from openpyxl.utils.datetime import from_excel
from psycopg2.extras import Json, execute_values
from tqdm import tqdm

from retrieval.contextual_embeddings import (
    CONTEXTUAL_EMBEDDING_RECIPE,
    contextual_embedding_text_for_model,
)
from retrieval.ingest.excel.schema import migrate
from retrieval.setup_db import setup_database
from retrieval.utils import connect_db, embedding_config, get_embedding_model


logger = logging.getLogger(__name__)
EXTRACTOR = "openpyxl-table-aware-v2"
INGEST_RECIPE = "generic-xlsx-tables-v2"
HEADER_SCAN_ROWS = 20


@dataclass(frozen=True)
class WorkbookChunk:
    sheet: str
    row_start: int
    row_end: int
    content: str
    token_count: int
    header_row_start: int | None
    header_row_end: int | None
    column_start: int
    column_end: int
    grid: tuple[tuple[str, ...], ...]
    formula_grid: tuple[tuple[str | None, ...], ...]

    @property
    def cell_range(self) -> str:
        first_row = min(
            self.row_start,
            self.header_row_start if self.header_row_start is not None else self.row_start,
        )
        return (
            f"{get_column_letter(self.column_start)}{first_row}:"
            f"{get_column_letter(self.column_end)}{self.row_end}"
        )


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _cell_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return " ".join(str(value).split())


def _markdown_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def _markdown_row(values: Sequence[str]) -> str:
    return "| " + " | ".join(_markdown_cell(value) for value in values) + " |"


def _unique_headers(values: Sequence[object]) -> tuple[str, ...]:
    """Return readable, unique labels, filling blank cells with Excel columns."""
    seen: dict[str, int] = {}
    labels: list[str] = []
    for index, value in enumerate(values, start=1):
        base = _cell_text(value) or f"Column {get_column_letter(index)}"
        seen[base] = seen.get(base, 0) + 1
        labels.append(base if seen[base] == 1 else f"{base} ({seen[base]})")
    return tuple(labels)


def _header_row(rows: Sequence[tuple[int, Sequence[object]]]) -> int | None:
    """Choose a conservative table header from the first populated rows.

    A candidate needs at least two textual labels. The row with the most labels
    wins; ties prefer the earliest row so ordinary tables are not mistaken for
    later all-text data rows.
    """
    candidates: list[tuple[int, int, int]] = []
    for row_number, values in rows[:HEADER_SCAN_ROWS]:
        populated = [value for value in values if _cell_text(value)]
        textual = [
            value
            for value in populated
            if isinstance(value, str) and not value.startswith("=")
        ]
        if len(populated) >= 2 and len(textual) >= 2:
            candidates.append((len(textual), len(populated), -row_number))
    winner = max(candidates, default=(0, 0, 0))
    return -winner[2] or None


def _display_and_formulas(
    formula_values: Sequence[object], cached_values: Sequence[object]
) -> tuple[tuple[str, ...], tuple[str | None, ...]]:
    display: list[str] = []
    formulas: list[str | None] = []
    for formula_value, cached_value in zip(formula_values, cached_values, strict=True):
        is_formula = isinstance(formula_value, str) and formula_value.startswith("=")
        formulas.append(formula_value if is_formula else None)
        chosen = cached_value if is_formula and cached_value is not None else formula_value
        display.append(_cell_text(chosen))
    return tuple(display), tuple(formulas)


def _token_count(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=True))


class _SheetMetadataHandler(ContentHandler):
    """Constant-memory scan of meaningful cells and formulas in sheet XML."""

    def __init__(self, workbook) -> None:
        super().__init__()
        self.workbook = workbook
        self.populated_rows: set[int] = set()
        self.populated_columns: set[int] = set()
        self.formulas: dict[str, str] = {}
        self.values: dict[str, object] = {}
        self.coordinate: str | None = None
        self.cell_type: str | None = None
        self.style_id = 0
        self.has_content = False
        self.in_formula = False
        self.in_value = False
        self.in_inline_text = False
        self.formula_parts: list[str] = []
        self.value_parts: list[str] = []

    def startElement(self, name, attrs) -> None:  # noqa: N802 - SAX API
        if name == "c":
            self.coordinate = attrs.get("r")
            self.cell_type = attrs.get("t")
            self.style_id = int(attrs.get("s", "0"))
            self.has_content = False
            self.formula_parts = []
            self.value_parts = []
        elif self.coordinate and name == "f":
            self.in_formula = True
            self.has_content = True
        elif self.coordinate and name == "v":
            self.in_value = True
            self.has_content = True
        elif self.coordinate and name == "t" and self.cell_type == "inlineStr":
            self.in_inline_text = True
            self.has_content = True

    def characters(self, content: str) -> None:
        if self.in_formula:
            self.formula_parts.append(content)
        if self.in_value or self.in_inline_text:
            self.value_parts.append(content)

    def endElement(self, name: str) -> None:  # noqa: N802 - SAX API
        if name == "f":
            self.in_formula = False
        elif name == "v":
            self.in_value = False
        elif name == "t":
            self.in_inline_text = False
        elif name == "c":
            if self.has_content and self.coordinate:
                row, column = coordinate_to_tuple(self.coordinate)
                self.populated_rows.add(row)
                self.populated_columns.add(column)
                formula = "".join(self.formula_parts).strip()
                if formula:
                    self.formulas[self.coordinate] = f"={formula}"
                raw = "".join(self.value_parts)
                if raw != "":
                    self.values[self.coordinate] = self._value(raw)
            self.coordinate = None
            self.formula_parts = []
            self.value_parts = []

    def _value(self, raw: str) -> object:
        if self.cell_type == "s":
            try:
                return self.workbook.shared_strings[int(raw)]
            except (IndexError, ValueError):
                return raw
        if self.cell_type == "b":
            return raw == "1"
        if self.cell_type in {"str", "inlineStr", "e"}:
            return raw
        try:
            number = float(raw)
        except ValueError:
            return raw
        try:
            style = self.workbook._cell_styles[self.style_id]
            number_format = BUILTIN_FORMATS.get(style.numFmtId)
            if number_format is None and style.numFmtId >= 164:
                number_format = self.workbook._number_formats[style.numFmtId - 164]
            if number_format and is_date_format(number_format):
                return from_excel(number, self.workbook.epoch)
        except (IndexError, TypeError, ValueError):
            pass
        return int(number) if number.is_integer() else number


def _sheet_metadata(
    workbook, worksheet
) -> tuple[int, int, dict[str, str], dict[str, object]]:
    handler = _SheetMetadataHandler(workbook)
    parser = xml.sax.make_parser()
    parser.setContentHandler(handler)
    with workbook._archive.open(worksheet._worksheet_path) as source:
        parser.parse(source)
    return (
        _main_region_end(handler.populated_rows),
        _main_region_end(handler.populated_columns),
        handler.formulas,
        handler.values,
    )


def _main_region_end(indices: set[int], *, gap_limit: int = 10_000) -> int:
    """Ignore isolated populated cells separated from the main table by a huge gap."""
    if not indices:
        return 0
    ordered = sorted(indices)
    end = ordered[0]
    for value in ordered[1:]:
        if value - end > gap_limit:
            break
        end = value
    return end


def _bounded_text_parts(tokenizer, prefix: str, text: str, max_tokens: int) -> list[str]:
    """Split even an unbroken formula/string by model tokens, not whitespace."""
    prefix_tokens = _token_count(tokenizer, prefix)
    allowance = max(1, max_tokens - prefix_tokens - 2)
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) <= allowance:
        return [text]
    if not hasattr(tokenizer, "decode"):
        words = text.split()
        return [
            " ".join(words[index : index + allowance])
            for index in range(0, len(words), allowance)
        ]
    return [
        tokenizer.decode(
            token_ids[index : index + allowance], skip_special_tokens=True
        )
        for index in range(0, len(token_ids), allowance)
    ]


def build_workbook_chunks(path: Path, tokenizer, max_tokens: int) -> list[WorkbookChunk]:
    """Read every non-empty sheet cell into row-aware, table-like chunks."""
    # Read cached/display values through openpyxl and formulas directly from the
    # same XLSX XML archive. Opening data_only=True and data_only=False copies at
    # once caused excessive memory use on large regulatory workbooks.
    workbook = load_workbook(path, read_only=True, data_only=True)
    chunks: list[WorkbookChunk] = []
    try:
        for worksheet in workbook.worksheets:
            max_row, max_column, formulas_by_cell, cached_by_cell = _sheet_metadata(
                workbook, worksheet
            )
            if max_row == 0 or max_column == 0:
                continue
            populated: list[
                tuple[int, tuple[str, ...], tuple[str | None, ...]]
            ] = []
            raw_for_header: list[tuple[int, Sequence[object]]] = []
            for row_number in range(1, max_row + 1):
                cached_row = tuple(
                    cached_by_cell.get(f"{get_column_letter(column)}{row_number}")
                    for column in range(1, max_column + 1)
                )
                formula_row = tuple(
                    formulas_by_cell.get(
                        f"{get_column_letter(column)}{row_number}", cached_value
                    )
                    for column, cached_value in enumerate(cached_row, start=1)
                )
                display, formulas = _display_and_formulas(formula_row, cached_row)
                if not any(display):
                    continue
                populated.append((row_number, display, formulas))
                raw_for_header.append((row_number, formula_row))

            if not populated:
                continue

            detected_header_row = _header_row(raw_for_header)
            detected = next(
                (row for row in populated if row[0] == detected_header_row), None
            )
            headers = (
                _unique_headers(detected[1])
                if detected
                else _unique_headers([None] * max_column)
            )
            prefix = (
                f"Workbook: {path.name}\nSheet: {worksheet.title}\n"
                f"Header row: {detected_header_row or 'generated'}\n"
            )
            separator = tuple("---" for _ in headers)
            pending: list[tuple[int, tuple[str, ...], tuple[str | None, ...]]] = []

            def flush() -> None:
                if not pending:
                    return
                grid = (headers, *(row[1] for row in pending))
                formula_grid = (
                    tuple(None for _ in headers),
                    *(row[2] for row in pending),
                )
                content = prefix + "\n".join(
                    [_markdown_row(headers), _markdown_row(separator),
                     *(_markdown_row(row[1]) for row in pending)]
                )
                chunks.append(
                    WorkbookChunk(
                        sheet=worksheet.title,
                        row_start=pending[0][0],
                        row_end=pending[-1][0],
                        content=content,
                        token_count=_token_count(tokenizer, content),
                        header_row_start=detected_header_row,
                        header_row_end=detected_header_row,
                        column_start=1,
                        column_end=max_column,
                        grid=grid,
                        formula_grid=formula_grid,
                    )
                )
                pending.clear()

            for row_number, display, formulas in populated:
                if detected_header_row is not None and row_number == detected_header_row:
                    continue
                candidate = prefix + "\n".join(
                    [_markdown_row(headers), _markdown_row(separator),
                     *( _markdown_row(item[1]) for item in pending),
                     _markdown_row(display)]
                )
                if pending and _token_count(tokenizer, candidate) > max_tokens:
                    flush()
                    candidate = prefix + "\n".join(
                        [_markdown_row(headers), _markdown_row(separator),
                         _markdown_row(display)]
                    )
                if _token_count(tokenizer, candidate) > max_tokens:
                    # Preserve a pathological wide row in bounded text pieces.
                    flush()
                    row_text = f"Source row {row_number}: {_markdown_row(display)}"
                    for part in _bounded_text_parts(
                        tokenizer, prefix, row_text, max_tokens
                    ):
                        content = prefix + part
                        chunks.append(
                            WorkbookChunk(
                                worksheet.title,
                                row_number,
                                row_number,
                                content,
                                _token_count(tokenizer, content),
                                detected_header_row,
                                detected_header_row,
                                1,
                                max_column,
                                (headers, display),
                                (tuple(None for _ in headers), formulas),
                            )
                        )
                else:
                    pending.append((row_number, display, formulas))
            flush()
    finally:
        workbook.close()
    return chunks


def _embed(model, path: Path, chunks: Sequence[WorkbookChunk]) -> tuple[list, list]:
    if not chunks:
        return [], []
    raw = model.encode(
        [chunk.content for chunk in chunks],
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    contextual_inputs = [
        contextual_embedding_text_for_model(
            path.name,
            f"{path.name} > {chunk.sheet} > rows {chunk.row_start}-{chunk.row_end}",
            chunk.content,
            model.tokenizer,
            model.max_seq_length,
        )
        for chunk in chunks
    ]
    contextual = model.encode(
        contextual_inputs,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return [value.tolist() for value in raw], [value.tolist() for value in contextual]


def ingest_workbook(path: Path, conn, model, *, force: bool = False) -> str:
    source_hash = _file_hash(path)
    signature = hashlib.sha256(
        f"{INGEST_RECIPE}|{embedding_config()}".encode()
    ).hexdigest()
    with conn.cursor() as cursor:
        cursor.execute(
            "SELECT id, content_hash, ingest_signature FROM documents WHERE filename = %s",
            (path.name,),
        )
        existing = cursor.fetchone()
    if existing and existing[1:] == (source_hash, signature) and not force:
        logger.info("Skipping unchanged workbook %s", path.name)
        return "unchanged"

    max_tokens = int(embedding_config()["max_tokens_per_chunk"])
    chunks = build_workbook_chunks(path, model.tokenizer, max_tokens)
    raw_embeddings, contextual_embeddings = _embed(model, path, chunks)
    config = embedding_config()
    with conn.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO documents (
                filename, page_count, content_hash, ingest_signature, chunk_counts
            ) VALUES (%s, 0, %s, %s, %s)
            ON CONFLICT (filename) DO UPDATE
            SET content_hash = EXCLUDED.content_hash,
                ingest_signature = EXCLUDED.ingest_signature,
                chunk_counts = EXCLUDED.chunk_counts,
                ingested_at = now()
            RETURNING id
            """,
            (path.name, source_hash, signature, Json({"excel_card": len(chunks)})),
        )
        document_id = cursor.fetchone()[0]
        cursor.execute("DELETE FROM chunks WHERE document_id = %s", (document_id,))
        rows = []
        for index, (chunk, raw, contextual) in enumerate(
            zip(chunks, raw_embeddings, contextual_embeddings, strict=True)
        ):
            breadcrumb = f"{path.name} > {chunk.sheet} > {chunk.cell_range}"
            rows.append(
                (
                    document_id,
                    chunk.sheet,
                    breadcrumb,
                    None,
                    0,
                    0,
                    index,
                    "excel_card",
                    chunk.content,
                    None,
                    f"{chunk.sheet}, {chunk.cell_range}",
                    Json(
                        {
                            "kind": "workbook_table",
                            "source_file": path.name,
                            "sheet": chunk.sheet,
                            "cell_range": chunk.cell_range,
                            "header_row_start": chunk.header_row_start,
                            "header_row_end": chunk.header_row_end,
                            "row_start": chunk.row_start,
                            "row_end": chunk.row_end,
                            "column_start": chunk.column_start,
                            "column_end": chunk.column_end,
                            "grid": [list(row) for row in chunk.grid],
                            "formula_grid": [
                                list(row) for row in chunk.formula_grid
                            ],
                        }
                    ),
                    None,
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    chunk.token_count,
                    hashlib.sha256(chunk.content.encode()).hexdigest(),
                    config.get("provider", "sentence_transformers"),
                    config["name"],
                    list(raw),
                    config["name"],
                    CONTEXTUAL_EMBEDDING_RECIPE,
                    list(contextual),
                    EXTRACTOR,
                )
            )
        if rows:
            execute_values(
                cursor,
                """
                INSERT INTO chunks (
                    document_id, sub_document, breadcrumb, section_number,
                    page_start, page_end, chunk_index, content_type,
                    content, retrieval_hint, caption, structured_data,
                    object_key, media_type, token_count, content_hash,
                    embedding_provider, embedding_model, embedding,
                    contextual_embedding_model, contextual_embedding_recipe,
                    contextual_embedding, extractor
                ) VALUES %s
                """,
                rows,
            )
    conn.commit()
    logger.info("Ingested %s: %d workbook chunks", path.name, len(chunks))
    return "loaded"


def ingest_directory(
    directory: Path,
    conn,
    model,
    *,
    force: bool = False,
    exclude_names: set[str] | None = None,
) -> dict[str, str]:
    excluded = exclude_names or set()
    paths = sorted(
        path
        for path in [*directory.rglob("*.xlsx"), *directory.rglob("*.xlsm")]
        if path.name not in excluded
    )
    results: dict[str, str] = {}
    for path in tqdm(paths, desc="Ingesting workbooks", unit="file"):
        try:
            results[path.name] = ingest_workbook(path, conn, model, force=force)
        except Exception:
            conn.rollback()
            logger.exception("Failed to ingest workbook %s", path.name)
            results[path.name] = "failed"
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path, nargs="?", default=Path("files"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    setup_database()
    conn = connect_db()
    try:
        migrate(conn)
        results = ingest_directory(
            args.input_dir, conn, get_embedding_model(), force=args.force
        )
    finally:
        conn.close()
    if not results:
        raise SystemExit(f"No XLSX/XLSM files matched in {args.input_dir}")
    failed = [name for name, status in results.items() if status == "failed"]
    if failed:
        raise SystemExit(f"Workbook ingestion failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()

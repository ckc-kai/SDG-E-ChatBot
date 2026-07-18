"""Ingest SDG&E Wildfire Mitigation Plan PDFs into Postgres/pgvector.

Chunking strategy: Hierachy
------------------
1. Hierarchy source: each PDF's native bookmark outline (pypdf `.outline`) is
   used as the section tree instead of inferring headings from text/font
   heuristics. 
   Regulatory WMP filings already carry accurate, deeply nested bookmarks (e.g. "8.1.7.6 Aging report").

2. Multi-document detection: WMP filings are frequently merged PDFs (a main
   body plus several appendices, some of which themselves wrap third-party
   sub-reports). Any bookmark whose title ends in ".pdf" marks the start of a
   new logical sub-document; its `breadcrumb` and `sub_document` scope reset
   there instead of inheriting the parent tree's path. This prevents a nested
   third-party report (e.g. a Kinectrics test report inside "Appendix D")
   from being mislabeled under SDG&E's own section numbering.

3. Leaf-only extraction: only outline nodes with no children ("leaf
   sections") have their page range's text extracted and chunked. Parent
   headings exist purely to build the breadcrumb path, not as their own
   content blocks.

4. Token-based sub-chunking: each leaf section's text is split using the
   embedding model's own tokenizer into MAX_TOKENS_PER_CHUNK windows with
   TOKEN_OVERLAP overlap, so chunk size matches the embedding model's
   effective context regardless of how long a section is.

5. Edge-case guards found across this corpus:
   - Bookmarks with an unresolvable destination page number inherit the
     previous bookmark's page instead of crashing.
   - Zero/negative-length leaf sections (repeated merge-divider bookmarks,
     e.g. duplicate "Appendix D" entries between embedded sub-reports) are
     dropped.
   - TODO: Leaf sections whose extracted text is near-empty (MIN_CHUNK_CHARS) are
     skipped rather than stored as noise (covers figure/map-only pages).
   - PDFs with no outline at all fall back to a single whole-document
     section (no breadcrumb) split purely by token count.

6. TODO: Tables and figures are intentionally out of scope for now: their content
   is extracted as flattened plain text through the same leaf-section path,
   which will scramble tabular structure. Every chunk carries a
   `content_type` field (currently always "narrative") so a future
   table/figure-aware extractor can feed the same `chunks` table with
   `content_type="table"` / `"figure"` rows without a schema change.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import pypdf
from psycopg2.extras import execute_values
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from retrieval.failure_log import get_failure_logger
from retrieval.utils import connect_db, get_embedding_model, load_config

logger = logging.getLogger(__name__)
log_failure = get_failure_logger("ingest")

_embedding_config = load_config()["local"]["embedding"]
EMBEDDING_MODEL_NAME = _embedding_config["model"]
MAX_TOKENS_PER_CHUNK = _embedding_config["max_tokens_per_chunk"]
TOKEN_OVERLAP = _embedding_config["token_overlap"]
MIN_CHUNK_CHARS = _embedding_config["minimum_chunk_chars"]
DEFAULT_CONTENT_TYPE = "narrative"

_SECTION_NUMBER_RE = re.compile(r"^(\d+(?:\.\d+)*)\.?\s")


@dataclass(frozen=True)
class SectionNode:
    title: str
    depth: int
    page_start: int | None
    breadcrumb: str
    sub_document: str | None
    section_number: str | None
    children: tuple["SectionNode", ...] = ()


@dataclass(frozen=True)
class LeafSection:
    breadcrumb: str
    sub_document: str | None
    section_number: str | None
    page_start: int
    page_end: int


@dataclass(frozen=True)
class Chunk:
    source_pdf: str
    sub_document: str | None
    breadcrumb: str
    section_number: str | None
    page_start: int
    page_end: int
    chunk_index: int
    content_type: str
    content: str
    token_count: int


def _extract_section_number(title: str) -> str | None:
    match = _SECTION_NUMBER_RE.match(title)
    return match.group(1) if match else None


def _is_document_boundary(title: str) -> bool:
    return title.lower().endswith(".pdf")


def _resolve_page_number(reader: pypdf.PdfReader, destination: object) -> int | None:
    try:
        return reader.get_destination_page_number(destination)
    except Exception:
        logger.debug("Could not resolve page number for bookmark %r", getattr(destination, "title", destination))
        return None


def _parse_outline(
    items: list,
    reader: pypdf.PdfReader,
    depth: int,
    parent_breadcrumb: str,
    parent_sub_document: str | None,
) -> list[SectionNode]:
    """Recursively convert a pypdf outline segment into a SectionNode tree."""
    nodes: list[SectionNode] = []
    index = 0
    while index < len(items):
        item = items[index]
        if isinstance(item, list):
            nodes.extend(_parse_outline(item, reader, depth, parent_breadcrumb, parent_sub_document))
            index += 1
            continue

        title = str(item.title).strip()
        page = _resolve_page_number(reader, item)
        is_boundary = _is_document_boundary(title)
        sub_document = title if is_boundary else parent_sub_document
        breadcrumb = title if is_boundary or not parent_breadcrumb else f"{parent_breadcrumb} > {title}"

        children: list[SectionNode] = []
        if index + 1 < len(items) and isinstance(items[index + 1], list):
            children = _parse_outline(items[index + 1], reader, depth + 1, breadcrumb, sub_document)
            index += 2
        else:
            index += 1

        nodes.append(
            SectionNode(
                title=title,
                depth=depth,
                page_start=page,
                breadcrumb=breadcrumb,
                sub_document=sub_document,
                section_number=_extract_section_number(title),
                children=tuple(children),
            )
        )
    return nodes


def _fill_missing_pages(nodes: tuple[SectionNode, ...], carry: int) -> tuple[tuple[SectionNode, ...], int]:
    """Replace unresolvable page numbers with the previous bookmark's page."""
    resolved: list[SectionNode] = []
    for node in nodes:
        page_start = node.page_start if node.page_start is not None else carry
        carry = page_start
        children, carry = _fill_missing_pages(node.children, carry)
        resolved.append(replace(node, page_start=page_start, children=children))
    return tuple(resolved), carry


def _flatten(nodes: tuple[SectionNode, ...]) -> list[SectionNode]:
    flat: list[SectionNode] = []
    for node in nodes:
        flat.append(node)
        flat.extend(_flatten(node.children))
    return flat


def _leaf_sections(flat: list[SectionNode], doc_page_count: int) -> list[LeafSection]:
    leaves: list[LeafSection] = []
    for i, node in enumerate(flat):
        if node.children:
            continue

        page_end = doc_page_count
        for later in flat[i + 1 :]:
            if later.depth <= node.depth:
                page_end = later.page_start
                break

        if page_end <= node.page_start:
            continue

        leaves.append(
            LeafSection(
                breadcrumb=node.breadcrumb,
                sub_document=node.sub_document,
                section_number=node.section_number,
                page_start=node.page_start,
                page_end=page_end,
            )
        )
    return leaves


def build_leaf_sections(pdf_path: Path, reader: pypdf.PdfReader) -> list[LeafSection]:
    if not reader.outline:
        return [
            LeafSection(
                breadcrumb=pdf_path.name,
                sub_document=None,
                section_number=None,
                page_start=0,
                page_end=len(reader.pages),
            )
        ]

    tree = tuple(_parse_outline(reader.outline, reader, depth=0, parent_breadcrumb="", parent_sub_document=None))
    tree, _ = _fill_missing_pages(tree, carry=0)
    return _leaf_sections(_flatten(tree), len(reader.pages))


def extract_section_text(reader: pypdf.PdfReader, page_start: int, page_end: int) -> str:
    page_end = min(page_end, len(reader.pages))
    texts = [reader.pages[i].extract_text() or "" for i in range(page_start, page_end)]
    return "\n".join(texts).strip()


def chunk_text(text: str, tokenizer, max_tokens: int, overlap: int) -> list[str]:
    if not text:
        return []

    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) <= max_tokens:
        return [text]

    stride = max_tokens - overlap
    pieces: list[str] = []
    for start in range(0, len(token_ids), stride):
        window = token_ids[start : start + max_tokens]
        if not window:
            break
        piece = tokenizer.decode(window, skip_special_tokens=True).strip()
        if piece:
            pieces.append(piece)
        if start + max_tokens >= len(token_ids):
            break
    return pieces


def build_chunks(pdf_path: Path, reader: pypdf.PdfReader, tokenizer) -> list[Chunk]:
    chunks: list[Chunk] = []
    for section in build_leaf_sections(pdf_path, reader):
        text = extract_section_text(reader, section.page_start, section.page_end)
        if len(text) < MIN_CHUNK_CHARS:
            logger.debug("Skipping near-empty section %r in %s", section.breadcrumb, pdf_path.name)
            continue

        for index, piece in enumerate(chunk_text(text, tokenizer, MAX_TOKENS_PER_CHUNK, TOKEN_OVERLAP)):
            chunks.append(
                Chunk(
                    source_pdf=pdf_path.name,
                    sub_document=section.sub_document,
                    breadcrumb=section.breadcrumb,
                    section_number=section.section_number,
                    page_start=section.page_start,
                    page_end=section.page_end,
                    chunk_index=index,
                    content_type=DEFAULT_CONTENT_TYPE,
                    content=piece,
                    token_count=len(tokenizer.encode(piece, add_special_tokens=False)),
                )
            )
    return chunks


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _existing_document(cur, filename: str) -> tuple[int, str] | None:
    cur.execute("SELECT id, content_hash FROM documents WHERE filename = %s", (filename,))
    row = cur.fetchone()
    return (row[0], row[1]) if row else None


def upsert_document(cur, filename: str, page_count: int, content_hash: str) -> int:
    cur.execute(
        """
        INSERT INTO documents (filename, page_count, content_hash)
        VALUES (%s, %s, %s)
        ON CONFLICT (filename)
        DO UPDATE SET page_count = EXCLUDED.page_count,
                      content_hash = EXCLUDED.content_hash,
                      ingested_at = now()
        RETURNING id
        """,
        (filename, page_count, content_hash),
    )
    return cur.fetchone()[0]


def replace_chunks(cur, document_id: int, chunks: Sequence[Chunk], embeddings: Sequence[Sequence[float]]) -> None:
    cur.execute("DELETE FROM chunks WHERE document_id = %s", (document_id,))
    if not chunks:
        return

    rows = [
        (
            document_id,
            chunk.sub_document,
            chunk.breadcrumb,
            chunk.section_number,
            chunk.page_start,
            chunk.page_end,
            chunk.chunk_index,
            chunk.content_type,
            chunk.content,
            chunk.token_count,
            EMBEDDING_MODEL_NAME,
            list(embedding),
        )
        for chunk, embedding in zip(chunks, embeddings)
    ]
    execute_values(
        cur,
        """
        INSERT INTO chunks (
            document_id, sub_document, breadcrumb, section_number,
            page_start, page_end, chunk_index, content_type,
            content, token_count, embedding_model, embedding
        ) VALUES %s
        """,
        rows,
    )


def embed_chunks(model: SentenceTransformer, chunks: Sequence[Chunk]) -> list[list[float]]:
    if not chunks:
        return []
    embeddings = model.encode(
        [chunk.content for chunk in chunks],
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    return [embedding.tolist() for embedding in embeddings]


def ingest_pdf(pdf_path: Path, conn, model: SentenceTransformer) -> None:
    content_hash = hash_file(pdf_path)
    with conn.cursor() as cur:
        existing = _existing_document(cur, pdf_path.name)
    if existing and existing[1] == content_hash:
        logger.info("Skipping %s (unchanged)", pdf_path.name)
        return

    reader = pypdf.PdfReader(pdf_path)
    chunks = build_chunks(pdf_path, reader, model.tokenizer)
    embeddings = embed_chunks(model, chunks)

    with conn.cursor() as cur:
        document_id = upsert_document(cur, pdf_path.name, len(reader.pages), content_hash)
        replace_chunks(cur, document_id, chunks, embeddings)
    conn.commit()
    logger.info("Ingested %s: %d chunks", pdf_path.name, len(chunks))


def main() -> None:
    parser = argparse.ArgumentParser(description="Chunk and embed WMP PDFs into Postgres/pgvector.")
    parser.add_argument(
        "pdf_dir",
        type=Path,
        nargs="?",
        default=Path("resources/wmp/pdf"),
        help="Directory containing PDF files to ingest (default: resources/wmp/pdf).",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    model = get_embedding_model()
    conn = connect_db()
    try:
        pdf_paths = sorted(args.pdf_dir.glob("*.pdf"))
        for pdf_path in tqdm(pdf_paths, desc="Ingesting PDFs", unit="file"):
            try:
                ingest_pdf(pdf_path, conn, model)
            except Exception as exc:
                conn.rollback()
                log_failure("ingest_pdf", pdf_path.name, exc)
                logger.warning("Failed to ingest %s: %s", pdf_path.name, exc)
    finally:
        conn.close()


if __name__ == "__main__":
    main()

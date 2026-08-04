"""Unified PDF ingest for narrative text, tables, and figures.

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
   - Heading-anchor trim: a bookmark only resolves to a PAGE, not a position
     within it. When a page holds the tail of the previous section followed by
     this section's heading, naive whole-page extraction would glue that
     leftover prose onto this section's first chunk. `extract_section_text`
     locates this section's own bookmark title on its first page (a literal,
     whitespace/case/punctuation-tolerant text search, not a semantic match)
     and trims everything before it. If the title can't be confidently located
     (too short/generic, or not found), the page is left untouched.

6. Structured extraction: the same command invokes the local Docling extractor
   for tables and figures. All content types are persisted atomically into one
   ``chunks`` table and receive raw plus contextual embeddings.

Generated figure descriptions are isolated in ``retrieval_hint``. They can
expand candidate recall when enabled in YAML, but are excluded from reranking
and answer evidence by default.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath

import pypdf
from psycopg2.extras import Json, execute_values
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from retrieval.contextual_embeddings import (
    CONTEXTUAL_EMBEDDING_RECIPE,
    contextual_embedding_text_for_model,
)
from retrieval.failure_log import get_failure_logger
from retrieval.object_storage import get_object_storage
from retrieval.setup_db import setup_database
from retrieval.utils import (
    connect_db,
    embedding_config,
    get_embedding_model,
    load_config,
)

logger = logging.getLogger(__name__)
log_failure = get_failure_logger("ingest")

_config = load_config()
_embedding_config = embedding_config()
EMBEDDING_PROVIDER = _embedding_config.get("provider", "sentence_transformers")
EMBEDDING_MODEL_NAME = _embedding_config["name"]
MAX_TOKENS_PER_CHUNK = _embedding_config["max_tokens_per_chunk"]
TOKEN_OVERLAP = _embedding_config["token_overlap"]
MIN_CHUNK_CHARS = _embedding_config["minimum_chunk_chars"]
DEFAULT_CONTENT_TYPE = "narrative"
_figure_description_config = (
    _config.get("extraction", {}).get("structured", {}).get("figure_description", {})
)
HINT_IN_CANDIDATE_RETRIEVAL = bool(
    _figure_description_config.get("candidate_retrieval", True)
)

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
    title: str  # this leaf's own heading text (not the full breadcrumb path), used to anchor extraction
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
    retrieval_hint: str | None = None
    caption: str | None = None
    structured_data: dict | None = None
    object_key: str | None = None
    media_type: str | None = None
    extractor: str = "pypdf-outline-v1"


def _extract_section_number(title: str) -> str | None:
    match = _SECTION_NUMBER_RE.match(title)
    return match.group(1) if match else None


def _is_document_boundary(title: str) -> bool:
    return title.lower().endswith(".pdf")


def _resolve_page_number(reader: pypdf.PdfReader, destination: object) -> int | None:
    try:
        return reader.get_destination_page_number(destination)
    except Exception:
        logger.debug(
            "Could not resolve page number for bookmark %r",
            getattr(destination, "title", destination),
        )
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
            nodes.extend(
                _parse_outline(
                    item, reader, depth, parent_breadcrumb, parent_sub_document
                )
            )
            index += 1
            continue

        title = str(item.title).strip()
        page = _resolve_page_number(reader, item)
        is_boundary = _is_document_boundary(title)
        sub_document = title if is_boundary else parent_sub_document
        breadcrumb = (
            title
            if is_boundary or not parent_breadcrumb
            else f"{parent_breadcrumb} > {title}"
        )

        children: list[SectionNode] = []
        if index + 1 < len(items) and isinstance(items[index + 1], list):
            children = _parse_outline(
                items[index + 1], reader, depth + 1, breadcrumb, sub_document
            )
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


def _fill_missing_pages(
    nodes: tuple[SectionNode, ...], carry: int
) -> tuple[tuple[SectionNode, ...], int]:
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
                title=node.title,
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
                title="",  # no bookmark title to anchor on; extraction stays whole-page
                breadcrumb=pdf_path.name,
                sub_document=None,
                section_number=None,
                page_start=0,
                page_end=len(reader.pages),
            )
        ]

    tree = tuple(
        _parse_outline(
            reader.outline,
            reader,
            depth=0,
            parent_breadcrumb="",
            parent_sub_document=None,
        )
    )
    tree, _ = _fill_missing_pages(tree, carry=0)
    return _leaf_sections(_flatten(tree), len(reader.pages))


MIN_HEADING_MATCH_CHARS = (
    8  # titles shorter/more generic than this are too risky to anchor on
)

_PUNCT_NORMALIZE = str.maketrans(
    {
        "‘": "'",
        "’": "'",
        "“": '"',
        "”": '"',
        "–": "-",
        "—": "-",
    }
)


def _normalize_punct(text: str) -> str:
    """1:1 character substitution (curly quotes/dashes -> ASCII) that preserves length,
    so a match offset found on the normalized text is still valid on the original."""
    return text.translate(_PUNCT_NORMALIZE)


def _heading_pattern(text: str) -> re.Pattern[str]:
    """Whitespace-tolerant, case-insensitive regex for a literal heading string."""
    words = text.split()
    return re.compile(r"\s+".join(re.escape(w) for w in words), re.IGNORECASE)


def _trim_to_heading(page_text: str, title: str) -> str:
    """Drop any text before this leaf section's own heading on its first page.

    A bookmark resolves to a page, not a position within it. Locates the section's
    own title (a literal, normalized text search -- not a semantic match) on the
    page and trims everything before it. Falls back to the untouched page if the
    title is too short/generic to trust or can't be found.
    """
    normalized_title = _normalize_punct(title)
    words = normalized_title.split()
    # single generic words ("Overview", "Summary") are too likely to recur in body
    # prose unrelated to the real heading; require >=2 words AND a length floor.
    if (
        len(words) < 2
        or len(normalized_title.replace(" ", "")) < MIN_HEADING_MATCH_CHARS
    ):
        return page_text
    match = _heading_pattern(normalized_title).search(_normalize_punct(page_text))
    return page_text[match.start() :] if match else page_text


def extract_section_text(
    reader: pypdf.PdfReader, page_start: int, page_end: int, title: str = ""
) -> str:
    page_end = min(page_end, len(reader.pages))
    texts = [reader.pages[i].extract_text() or "" for i in range(page_start, page_end)]
    if texts and title:
        texts[0] = _trim_to_heading(texts[0], title)
    return "\n".join(texts).strip()


def chunk_text(text: str, tokenizer, max_tokens: int, overlap: int) -> list[str]:
    """Split text into overlapping token windows, sliced from the ORIGINAL string.

    We window over the token stream but reconstruct each piece by slicing the
    original text at the tokens' character offsets, NOT by ``tokenizer.decode``.
    The embedding tokenizer is uncased, so decoding token ids would lowercase and
    re-space the text (e.g. "SDG&E" -> "sdg & e"), corrupting the stored content.
    Offset slicing keeps the original casing and punctuation while using the exact
    same window boundaries.
    """
    if not text:
        return []
    if not getattr(tokenizer, "is_fast", False):
        raise TypeError(
            "chunk_text requires a fast tokenizer (return_offsets_mapping support); "
            f"got {type(tokenizer).__name__}"
        )

    offsets = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)[
        "offset_mapping"
    ]
    if len(offsets) <= max_tokens:
        return [text]

    stride = max_tokens - overlap
    pieces: list[str] = []
    for start in range(0, len(offsets), stride):
        window = offsets[start : start + max_tokens]
        if not window:
            break
        char_start, char_end = window[0][0], window[-1][1]
        piece = text[char_start:char_end].strip()
        if piece:
            pieces.append(piece)
        if start + max_tokens >= len(offsets):
            break
    return pieces


def build_chunks(pdf_path: Path, reader: pypdf.PdfReader, tokenizer) -> list[Chunk]:
    chunks: list[Chunk] = []
    for section in build_leaf_sections(pdf_path, reader):
        text = extract_section_text(
            reader, section.page_start, section.page_end, section.title
        )
        if len(text) < MIN_CHUNK_CHARS:
            logger.debug(
                "Skipping near-empty section %r in %s",
                section.breadcrumb,
                pdf_path.name,
            )
            continue

        for index, piece in enumerate(
            chunk_text(text, tokenizer, MAX_TOKENS_PER_CHUNK, TOKEN_OVERLAP)
        ):
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


def ingest_signature() -> str:
    """Hash every setting that changes stored chunks or vectors."""
    from retrieval.ingest.pdf.structured_extraction import extractor_signature

    payload = {
        "recipe": "unified-ingest-v1",
        "embedding_provider": EMBEDDING_PROVIDER,
        "embedding_model": EMBEDDING_MODEL_NAME,
        "embedding_dimensions": _embedding_config["dimensions"],
        "max_tokens_per_chunk": MAX_TOKENS_PER_CHUNK,
        "token_overlap": TOKEN_OVERLAP,
        "minimum_chunk_chars": MIN_CHUNK_CHARS,
        "contextual_recipe": CONTEXTUAL_EMBEDDING_RECIPE,
        "hint_in_candidate_retrieval": HINT_IN_CANDIDATE_RETRIEVAL,
        "extraction": _config.get("extraction", {}),
        "structured_extractor": extractor_signature(),
        "object_storage": _config.get("object_storage", {}),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _existing_document(cur, filename: str) -> tuple[int, str, str] | None:
    cur.execute(
        "SELECT id, content_hash, ingest_signature "
        "FROM documents WHERE filename = %s",
        (filename,),
    )
    row = cur.fetchone()
    return (row[0], row[1], row[2]) if row else None


def upsert_document(
    cur,
    filename: str,
    page_count: int,
    content_hash: str,
    signature: str,
    chunk_counts: dict[str, int],
) -> int:
    cur.execute(
        """
        INSERT INTO documents (
            filename, page_count, content_hash, ingest_signature, chunk_counts
        )
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (filename)
        DO UPDATE SET page_count = EXCLUDED.page_count,
                      content_hash = EXCLUDED.content_hash,
                      ingest_signature = EXCLUDED.ingest_signature,
                      chunk_counts = EXCLUDED.chunk_counts,
                      ingested_at = now()
        RETURNING id
        """,
        (filename, page_count, content_hash, signature, Json(chunk_counts)),
    )
    return cur.fetchone()[0]


def replace_chunks(
    cur,
    document_id: int,
    chunks: Sequence[Chunk],
    raw_embeddings: Sequence[Sequence[float]],
    contextual_embeddings: Sequence[Sequence[float]],
) -> None:
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
            chunk.retrieval_hint,
            chunk.caption,
            Json(chunk.structured_data) if chunk.structured_data is not None else None,
            chunk.object_key,
            chunk.media_type,
            chunk.token_count,
            hashlib.sha256(chunk.content.encode()).hexdigest(),
            EMBEDDING_PROVIDER,
            EMBEDDING_MODEL_NAME,
            list(raw_embedding),
            EMBEDDING_MODEL_NAME,
            CONTEXTUAL_EMBEDDING_RECIPE,
            list(contextual_embedding),
            chunk.extractor,
        )
        for chunk, raw_embedding, contextual_embedding in zip(
            chunks, raw_embeddings, contextual_embeddings, strict=True
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
            contextual_embedding, extractor
        ) VALUES %s
        """,
        rows,
    )


def embed_chunks(
    model: SentenceTransformer, chunks: Sequence[Chunk]
) -> list[list[float]]:
    if not chunks:
        return []
    embeddings = model.encode(
        [chunk.content for chunk in chunks],
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    return [embedding.tolist() for embedding in embeddings]


def embed_contextual_chunks(
    model: SentenceTransformer, chunks: Sequence[Chunk]
) -> list[list[float]]:
    if not chunks:
        return []

    def contextual_input(chunk: Chunk) -> str:
        authoritative = chunk.content
        with_hint = authoritative + (
            f"\nRetrieval hint: {chunk.retrieval_hint}"
            if HINT_IN_CANDIDATE_RETRIEVAL and chunk.retrieval_hint
            else ""
        )
        try:
            return contextual_embedding_text_for_model(
                chunk.source_pdf,
                chunk.breadcrumb,
                with_hint,
                model.tokenizer,
                model.max_seq_length,
            )
        except ValueError:
            if with_hint == authoritative:
                raise
            logger.debug(
                "Dropping oversized retrieval hint from contextual embedding "
                "for %s page %d",
                chunk.source_pdf,
                chunk.page_start,
            )
            return contextual_embedding_text_for_model(
                chunk.source_pdf,
                chunk.breadcrumb,
                authoritative,
                model.tokenizer,
                model.max_seq_length,
            )

    embeddings = model.encode(
        [contextual_input(chunk) for chunk in chunks],
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    return [embedding.tolist() for embedding in embeddings]


def ingest_pdf(
    pdf_path: Path,
    conn,
    model: SentenceTransformer,
    *,
    storage=None,
) -> None:
    """Extract and replace all content types for one PDF atomically."""
    from retrieval.ingest.pdf.structured_extraction import (
        build_structured_chunks,
        extract_elements,
        safe_document_slug,
    )

    content_hash = hash_file(pdf_path)
    signature = ingest_signature()
    with conn.cursor() as cur:
        existing = _existing_document(cur, pdf_path.name)
    if existing and existing[1:] == (content_hash, signature):
        logger.info(
            "Skipping %s (content and ingest configuration unchanged)", pdf_path.name
        )
        return

    storage = storage or get_object_storage()
    old_object_prefixes: set[str] = set()
    if existing:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT object_key
                FROM chunks
                WHERE document_id = %s AND object_key IS NOT NULL
                """,
                (existing[0],),
            )
            old_object_prefixes = {
                str(PurePosixPath(row[0]).parent) for row in cur.fetchall()
            }
    reader = pypdf.PdfReader(pdf_path)
    narrative_chunks = build_chunks(pdf_path, reader, model.tokenizer)
    leaves = build_leaf_sections(pdf_path, reader)
    structured_enabled = bool(
        _config.get("extraction", {}).get("structured", {}).get("enabled", True)
    )
    structured_chunks: list[Chunk] = []
    object_prefix: str | None = None
    if structured_enabled:
        elements = extract_elements(pdf_path, len(reader.pages))
        object_prefix = (
            f"{safe_document_slug(pdf_path.name)}/"
            f"{content_hash[:12]}-{signature[:12]}"
        )

        def page_text(page_idx: int) -> str:
            try:
                return reader.pages[page_idx].extract_text() or ""
            except Exception:
                logger.debug(
                    "Could not extract page context from %s page %d",
                    pdf_path.name,
                    page_idx,
                    exc_info=True,
                )
                return ""

        structured_chunks = build_structured_chunks(
            pdf_path,
            elements,
            leaves,
            model.tokenizer,
            page_text_fn=page_text,
            storage=storage,
            object_prefix=object_prefix,
        )
    chunks = [*narrative_chunks, *structured_chunks]
    raw_embeddings = embed_chunks(model, chunks)
    contextual_embeddings = embed_contextual_chunks(model, chunks)
    chunk_counts = {
        content_type: sum(1 for chunk in chunks if chunk.content_type == content_type)
        for content_type in ("narrative", "table", "figure")
    }

    with conn.cursor() as cur:
        document_id = upsert_document(
            cur,
            pdf_path.name,
            len(reader.pages),
            content_hash,
            signature,
            chunk_counts,
        )
        replace_chunks(
            cur,
            document_id,
            chunks,
            raw_embeddings,
            contextual_embeddings,
        )
    conn.commit()
    for old_prefix in old_object_prefixes - (
        {object_prefix} if object_prefix else set()
    ):
        try:
            storage.clear_prefix(old_prefix)
        except Exception:
            logger.warning(
                "Ingest succeeded but stale figure prefix could not be cleared: %s",
                old_prefix,
                exc_info=True,
            )
    logger.info(
        "Ingested %s: %d narrative, %d table, %d figure chunks",
        pdf_path.name,
        chunk_counts["narrative"],
        chunk_counts["table"],
        chunk_counts["figure"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract, embed, and store all PDF content types."
    )
    parser.add_argument(
        "pdf_dir",
        type=Path,
        nargs="?",
        default=Path("resources/wmp/pdf"),
        help="Directory containing PDF files to ingest (default: resources/wmp/pdf).",
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="Only ingest PDFs whose filename contains this substring.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    setup_database()
    model = get_embedding_model()
    storage = get_object_storage()
    conn = connect_db()
    try:
        pdf_paths = sorted(args.pdf_dir.glob("*.pdf"))
        if args.only:
            pdf_paths = [
                path for path in pdf_paths if args.only.lower() in path.name.lower()
            ]
        if not pdf_paths:
            raise FileNotFoundError(
                f"No PDF files matched in {args.pdf_dir}"
                + (f" for --only {args.only!r}" if args.only else "")
            )
        failures: list[tuple[str, str]] = []
        for pdf_path in tqdm(pdf_paths, desc="Ingesting PDFs", unit="file"):
            try:
                ingest_pdf(pdf_path, conn, model, storage=storage)
            except Exception as exc:
                conn.rollback()
                log_failure("ingest_pdf", pdf_path.name, exc)
                logger.warning("Failed to ingest %s: %s", pdf_path.name, exc)
                failures.append((pdf_path.name, str(exc)))
        if failures:
            summary = "; ".join(f"{name}: {error}" for name, error in failures)
            raise RuntimeError(
                f"PDF ingestion failed for {len(failures)}/{len(pdf_paths)} file(s): "
                f"{summary}"
            )
    finally:
        conn.close()


if __name__ == "__main__":
    main()

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

3. Complete coverage: EVERY outline node becomes a section, not only the
   leaves, plus a synthetic span for the pages ahead of the first bookmark
   (cover, table of contents, executive summary). A parent heading's own prose
   -- the text between it and its first child -- is content, and leaf-only
   extraction never reached it. Parent-derived spans keep the owning node's
   breadcrumb so contextual embeddings stay correct, and are flagged
   `is_parent_span` so their contribution stays measurable and so they can be
   kept out of structured-element mapping.

4. Cut-point extraction: a bookmark resolves to a PAGE, not a position within
   it, so page boundaries alone cannot separate two sections that share a page.
   Each section instead runs from its own heading anchor to the NEXT section's
   heading anchor. A section therefore keeps the tail it owns at the top of its
   successor's first page. Trimming that tail without giving it to anyone was
   the single largest source of lost text in this corpus.

   Page RANGES (`page_start`/`page_end`) still describe only a section's own
   pages, so they remain a partition -- `map_page_to_leaf` and stored citations
   both depend on that.

5. Token-based sub-chunking: each section's text is split using the embedding
   model's own tokenizer into MAX_TOKENS_PER_CHUNK windows with TOKEN_OVERLAP
   overlap. Tables use their own, much larger window (see
   `structured_extraction`), because fragmenting a table destroys it while
   diluting a narrative chunk merely blurs it.

6. Edge-case guards found across this corpus:
   - Bookmarks with an unresolvable destination page number inherit the
     previous bookmark's page instead of crashing.
   - Zero-page sections (repeated merge-divider bookmarks, e.g. duplicate
     "Appendix D" entries between embedded sub-reports) are retained for
     cut-point purposes but contribute no chunks when their slice is empty.
   - Sections whose extracted text is near-empty (MIN_CHUNK_CHARS) are skipped
     rather than stored as noise (covers figure/map-only pages).
   - PDFs with no outline at all fall back to a single whole-document
     section (no breadcrumb) split purely by token count.
   - A heading anchor is only trusted when the title is >=2 words and
     >=MIN_HEADING_MATCH_CHARS long; otherwise the section is treated as
     starting at the top of its first page. Anchors are also forced
     non-decreasing among sections sharing a page, because a backwards cut
     would drop the slice between them instead of merely misattributing it.
   - `assert_extraction_coverage` fails the run if a document's chunked text
     falls below COVERAGE_MIN_RATIO of what pypdf can read from its pages.
     Both defects above were silent for the life of the project; this check,
     not the two patches, is what stops the class of bug.

7. Structured extraction: the same command invokes the local Docling extractor
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
from retrieval.source_manifest import title_for_filename
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
EMBEDDING_BATCH_SIZE = int(_embedding_config.get("embedding_batch_size", 32))
TABLE_EMBEDDING_BATCH_SIZE = int(
    _embedding_config.get("table_embedding_batch_size", EMBEDDING_BATCH_SIZE)
)
DEFAULT_CONTENT_TYPE = "narrative"

# A document must chunk at least this fraction of the text pypdf can read from
# its pages. Documents that are legitimately image-only have no extractable text
# to lose and are named here rather than lowering the threshold for everyone.
COVERAGE_MIN_RATIO = 0.95
COVERAGE_EXEMPT_FILENAMES: frozenset[str] = frozenset()
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
    title: str  # this section's own heading text (not the full breadcrumb path), used to anchor extraction
    breadcrumb: str
    sub_document: str | None
    section_number: str | None
    page_start: int
    page_end: int
    # Position in document order. Once parent spans exist, two sections can share
    # a page_start, so this -- not the page -- is a section's identity. The chunks
    # uniqueness key and parent reconstruction both rely on it.
    section_ordinal: int = 0
    # True for a span this module synthesises rather than reads off a leaf
    # bookmark: a parent heading's own prose, or the pages before the first
    # bookmark. Excluded from the list structured extraction maps elements onto.
    is_parent_span: bool = False


@dataclass(frozen=True)
class ExtractionCoverage:
    """How much of a document's readable text actually reached a chunk."""

    chunked_chars: int
    page_chars: int

    @property
    def ratio(self) -> float:
        return self.chunked_chars / self.page_chars if self.page_chars else 1.0


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
    section_ordinal: int = 0
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


def _document_sections(
    flat: list[SectionNode], doc_page_count: int, document_title: str
) -> tuple[LeafSection, ...]:
    """Every outline node as a section, in document order, plus front matter.

    A section's `page_end` is the next node's `page_start`, exclusive. For a leaf
    that is exactly what the old leaf-only walk produced (a leaf has no children,
    so the next node in pre-order is always at the same or a shallower depth); for
    a parent it is the first child's page, which is the span of the parent's own
    prose. Zero-page sections are kept rather than dropped: a heading that shares
    a page with the next one still owns the slice between the two headings.
    """
    sections: list[LeafSection] = []

    if flat and flat[0].page_start > 0:
        # A1: everything ahead of the first bookmark -- cover, table of contents,
        # executive summary. No outline node claims these pages, so leaf-only
        # extraction never read them at all.
        sections.append(
            LeafSection(
                title="",  # nothing to anchor on; this span starts at page 0, offset 0
                breadcrumb=document_title,
                sub_document=None,
                section_number=None,
                page_start=0,
                page_end=flat[0].page_start,
                is_parent_span=True,
            )
        )

    for index, node in enumerate(flat):
        following = flat[index + 1] if index + 1 < len(flat) else None
        page_end = following.page_start if following is not None else doc_page_count
        sections.append(
            LeafSection(
                title=node.title,
                breadcrumb=node.breadcrumb,
                sub_document=node.sub_document,
                section_number=node.section_number,
                page_start=node.page_start,
                # Merged filings and unresolvable destinations can order pages
                # backwards; a negative range would silently swallow content.
                page_end=max(page_end, node.page_start),
                is_parent_span=bool(node.children),
            )
        )

    return tuple(
        replace(section, section_ordinal=ordinal)
        for ordinal, section in enumerate(sections)
    )


def build_document_sections(
    pdf_path: Path, reader: pypdf.PdfReader
) -> tuple[LeafSection, ...]:
    """All sections in document order, for text extraction."""
    if not reader.outline:
        return (
            LeafSection(
                title="",  # no bookmark title to anchor on; extraction stays whole-page
                breadcrumb=pdf_path.name,
                sub_document=None,
                section_number=None,
                page_start=0,
                page_end=len(reader.pages),
            ),
        )

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
    return _document_sections(_flatten(tree), len(reader.pages), pdf_path.name)


def build_leaf_sections(pdf_path: Path, reader: pypdf.PdfReader) -> list[LeafSection]:
    """Only true leaf sections, with the page ranges structured extraction expects.

    `map_page_to_leaf` picks the containing leaf with the largest `page_start`, so
    handing it A1's parent spans would let a table on a child's page be attributed
    to the parent instead -- corrupting the breadcrumb on structured chunks and,
    through it, their contextual embeddings. Parent spans are for narrative text
    only; this list stays exactly what it was before A1.
    """
    return [
        section
        for section in build_document_sections(pdf_path, reader)
        if not section.is_parent_span and section.page_end > section.page_start
    ]


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


def _heading_offset(page_text: str, title: str) -> int:
    """Character offset on a page where this section's own heading begins.

    A bookmark resolves to a page, not a position within it, so the heading is
    located by a literal, whitespace/case/punctuation-tolerant text search -- not
    a semantic match. Returns 0, meaning "this section owns the page from the
    top", when the title is too short or generic to trust or cannot be found.
    Single generic words ("Overview", "Summary") recur in body prose unrelated to
    the real heading, hence the >=2-word and length floors.
    """
    if not title:
        return 0
    normalized_title = _normalize_punct(title)
    if (
        len(normalized_title.split()) < 2
        or len(normalized_title.replace(" ", "")) < MIN_HEADING_MATCH_CHARS
    ):
        return 0
    match = _heading_pattern(normalized_title).search(_normalize_punct(page_text))
    return match.start() if match else 0


def _visible_chars(text: str) -> int:
    """Non-whitespace character count.

    Coverage is measured on visible characters because section joins, strips, and
    heading cuts all legitimately change whitespace, and none of that is content.
    """
    return len("".join(text.split()))


class PageTextCache:
    """Reads each page's text at most once.

    `extract_text` is the expensive step of narrative ingest, and extraction, the
    coverage check, and figure page-context all need the same pages.
    """

    def __init__(self, reader: pypdf.PdfReader) -> None:
        self._reader = reader
        self._cache: dict[int, str] = {}

    @property
    def page_count(self) -> int:
        return len(self._reader.pages)

    def text(self, index: int) -> str:
        if not 0 <= index < self.page_count:
            return ""
        if index not in self._cache:
            try:
                self._cache[index] = self._reader.pages[index].extract_text() or ""
            except Exception:
                logger.warning(
                    "Could not extract text from page %d; treating it as empty",
                    index,
                    exc_info=True,
                )
                self._cache[index] = ""
        return self._cache[index]

    def total_visible_chars(self) -> int:
        return sum(_visible_chars(self.text(i)) for i in range(self.page_count))


def _section_cut_offsets(
    pages: PageTextCache, sections: Sequence[LeafSection]
) -> list[int]:
    """Heading offset for each section on its own first page.

    Offsets are forced non-decreasing among sections that start on the same page.
    A backwards cut would make the slice between two headings belong to neither
    section and disappear; clamping instead makes the later section swallow the
    earlier one's slice. Misattribution is recoverable, loss is not.
    """
    starts = [_clamp_page(section.page_start, pages.page_count) for section in sections]
    offsets = [
        _heading_offset(pages.text(start), section.title)
        for start, section in zip(starts, sections, strict=True)
    ]
    if offsets:
        # Every other section's page-top text is handed to its predecessor. The
        # first section has none, so it must own its page from character 0 --
        # otherwise a first bookmark that sits partway down page 0 silently drops
        # everything above it, which is the same defect A2 fixes, at the seam
        # where there is nobody to hand the text to.
        offsets[0] = 0
    for index in range(1, len(sections)):
        if starts[index] == starts[index - 1]:
            offsets[index] = max(offsets[index], offsets[index - 1])
    return offsets


def _clamp_page(page: int, page_count: int) -> int:
    return min(max(page, 0), max(page_count - 1, 0))


def extract_section_texts(
    pages: PageTextCache, sections: Sequence[LeafSection]
) -> list[str]:
    """Text for each section, cut at heading offsets instead of page boundaries.

    Each section runs from its own heading anchor to the next section's anchor, so
    it keeps the tail it owns at the top of its successor's first page. The old
    behaviour discarded that tail twice over -- the successor trimmed it away, and
    the predecessor's exclusive `page_end` meant it never read the page at all --
    which cost this corpus more text than the missing parent sections did.

    Every character of every page is claimed by exactly one section, which is what
    makes the coverage assertion meaningful.
    """
    if not sections:
        return []

    page_count = pages.page_count
    starts = [_clamp_page(section.page_start, page_count) for section in sections]
    offsets = _section_cut_offsets(pages, sections)

    texts: list[str] = []
    for index, start in enumerate(starts):
        if index + 1 < len(sections):
            stop_page, stop_offset = starts[index + 1], offsets[index + 1]
        else:
            stop_page, stop_offset = page_count, 0
        if stop_page < start:  # non-monotonic outline
            stop_page, stop_offset = start, offsets[index]

        if stop_page == start:
            texts.append(pages.text(start)[offsets[index] : stop_offset].strip())
            continue

        parts = [pages.text(start)[offsets[index] :]]
        parts.extend(pages.text(page) for page in range(start + 1, min(stop_page, page_count)))
        if stop_page < page_count and stop_offset > 0:
            parts.append(pages.text(stop_page)[:stop_offset])
        texts.append("\n".join(parts).strip())
    return texts


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


def build_chunks(
    pdf_path: Path,
    reader: pypdf.PdfReader,
    tokenizer,
    pages: PageTextCache | None = None,
) -> tuple[list[Chunk], ExtractionCoverage]:
    """Narrative chunks for one PDF, with the coverage its extraction achieved."""
    pages = pages if pages is not None else PageTextCache(reader)
    sections = build_document_sections(pdf_path, reader)
    texts = extract_section_texts(pages, sections)

    chunked_chars = 0
    chunks: list[Chunk] = []
    for section, text in zip(sections, texts, strict=True):
        if len(text) < MIN_CHUNK_CHARS:
            logger.debug(
                "Skipping near-empty section %r in %s",
                section.breadcrumb,
                pdf_path.name,
            )
            continue
        chunked_chars += _visible_chars(text)

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
                    section_ordinal=section.section_ordinal,
                )
            )
    return chunks, ExtractionCoverage(chunked_chars, pages.total_visible_chars())


def assert_extraction_coverage(filename: str, coverage: ExtractionCoverage) -> None:
    """Fail the run when a document's chunked text falls short of its pages.

    This is the fix. The leaf-only walk and the discarded heading trim each lost a
    large share of this corpus for the life of the project without producing a
    single error, and no amount of reranking can retrieve text that was never
    indexed. Any future change that silently stops reading part of a PDF stops the
    ingest instead.
    """
    if filename in COVERAGE_EXEMPT_FILENAMES:
        logger.info(
            "Coverage check skipped for %s (allowlisted as image-only)", filename
        )
        return
    if coverage.ratio >= COVERAGE_MIN_RATIO:
        logger.info(
            "Coverage %.1f%% for %s (%d of %d visible chars chunked)",
            coverage.ratio * 100,
            filename,
            coverage.chunked_chars,
            coverage.page_chars,
        )
        return
    raise RuntimeError(
        f"Extraction coverage for {filename} is {coverage.ratio:.1%} "
        f"({coverage.chunked_chars} of {coverage.page_chars} visible characters "
        f"chunked), below the {COVERAGE_MIN_RATIO:.0%} floor. "
        f"{coverage.page_chars - coverage.chunked_chars} characters would never "
        "reach the index. Add the file to COVERAGE_EXEMPT_FILENAMES only if it is "
        "genuinely image-only."
    )


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ingest_signature() -> str:
    """Hash every setting that changes stored chunks or vectors."""
    from retrieval.ingest.pdf.structured_extraction import extractor_signature

    from retrieval.ingest.pdf.structured_extraction import (
        TABLE_MAX_TOKENS_PER_CHUNK,
        TABLE_TOKEN_OVERLAP,
    )

    payload = {
        # v2: parent spans + cut-point extraction change what text every chunk
        # holds, so every previously ingested document must re-extract.
        "recipe": "unified-ingest-v2-full-coverage",
        "embedding_provider": EMBEDDING_PROVIDER,
        "embedding_model": EMBEDDING_MODEL_NAME,
        "embedding_dimensions": _embedding_config["dimensions"],
        "max_tokens_per_chunk": MAX_TOKENS_PER_CHUNK,
        "token_overlap": TOKEN_OVERLAP,
        "table_max_tokens_per_chunk": TABLE_MAX_TOKENS_PER_CHUNK,
        "table_token_overlap": TABLE_TOKEN_OVERLAP,
        "minimum_chunk_chars": MIN_CHUNK_CHARS,
        "coverage_min_ratio": COVERAGE_MIN_RATIO,
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
            chunk.section_ordinal,
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
            section_ordinal, page_start, page_end, chunk_index, content_type,
            content, retrieval_hint, caption, structured_data,
            object_key, media_type, token_count, content_hash,
            embedding_provider, embedding_model, embedding,
            contextual_embedding_model, contextual_embedding_recipe,
            contextual_embedding, extractor
        ) VALUES %s
        """,
        rows,
        # The Excel path already batches at 1000; two 1024-dim vectors per row is
        # ~30 KB of text, so 1000 rows is a few MB per statement.
        page_size=1000,
    )


def _batch_size_for(content_type: str) -> int:
    return TABLE_EMBEDDING_BATCH_SIZE if content_type == "table" else EMBEDDING_BATCH_SIZE


def _encode_by_content_type(
    model: SentenceTransformer, texts: Sequence[str], chunks: Sequence[Chunk]
) -> list[list[float]]:
    """Encode documents, grouping by content type so batch width stays sane.

    `encode` sorts by length internally to limit padding, but table chunks are an
    order of magnitude longer than narrative ones, so mixing them pads every
    narrative row out to a table's width. Separate calls with separate batch sizes
    keep peak unified memory bounded on Apple Silicon.

    No prompt is applied here. These are documents; only queries carry the
    asymmetric-model query prompt (see `retrieval.utils.encode_query`), and
    getting that backwards degrades retrieval silently.
    """
    vectors: list[list[float] | None] = [None] * len(texts)
    for content_type in dict.fromkeys(chunk.content_type for chunk in chunks):
        positions = [
            index
            for index, chunk in enumerate(chunks)
            if chunk.content_type == content_type
        ]
        encoded = model.encode(
            [texts[index] for index in positions],
            batch_size=_batch_size_for(content_type),
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        for index, vector in zip(positions, encoded, strict=True):
            vectors[index] = vector.tolist()
    return [vector for vector in vectors if vector is not None]


def embed_chunks(
    model: SentenceTransformer, chunks: Sequence[Chunk]
) -> list[list[float]]:
    if not chunks:
        return []
    return _encode_by_content_type(
        model, [chunk.content for chunk in chunks], chunks
    )


def embed_contextual_chunks(
    model: SentenceTransformer,
    chunks: Sequence[Chunk],
    document_title: str | None = None,
) -> list[list[float]]:
    """`document_title` is the manifest's readable title for this PDF.

    It becomes the first line of every contextual embedding. A filename is a poor
    thing to embed -- it tokenises badly and, on this corpus, once introduced the
    largest filing to the embedder under the wrong regulatory cycle. The readable
    title also distinguishes a filing from the guidelines it must comply with,
    which is a live confusion in the evaluation set.
    """
    if not chunks:
        return []

    def contextual_input(chunk: Chunk) -> str:
        title = document_title or chunk.source_pdf
        authoritative = chunk.content
        with_hint = authoritative + (
            f"\nRetrieval hint: {chunk.retrieval_hint}"
            if HINT_IN_CANDIDATE_RETRIEVAL and chunk.retrieval_hint
            else ""
        )
        try:
            return contextual_embedding_text_for_model(
                title,
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
                title,
                chunk.breadcrumb,
                authoritative,
                model.tokenizer,
                model.max_seq_length,
            )

    return _encode_by_content_type(
        model, [contextual_input(chunk) for chunk in chunks], chunks
    )


def ingest_pdf(
    pdf_path: Path,
    conn,
    model: SentenceTransformer,
    *,
    storage=None,
    structured_enabled: bool | None = None,
    skip_existing: bool = False,
) -> None:
    """Extract and replace all content types for one PDF atomically."""
    from retrieval.ingest.pdf.structured_extraction import (
        build_structured_chunks,
        extract_elements,
        safe_document_slug,
    )

    content_hash = hash_file(pdf_path)
    if structured_enabled is None:
        structured_enabled = bool(
            _config.get("extraction", {}).get("structured", {}).get("enabled", True)
        )
    signature = hashlib.sha256(
        f"{ingest_signature()}|structured={structured_enabled}".encode()
    ).hexdigest()
    with conn.cursor() as cur:
        existing = _existing_document(cur, pdf_path.name)
    if existing and skip_existing:
        logger.info("Skipping existing document %s", pdf_path.name)
        return
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
    pages = PageTextCache(reader)
    narrative_chunks, coverage = build_chunks(
        pdf_path, reader, model.tokenizer, pages=pages
    )
    assert_extraction_coverage(pdf_path.name, coverage)
    leaves = build_leaf_sections(pdf_path, reader)
    structured_chunks: list[Chunk] = []
    object_prefix: str | None = None
    if structured_enabled:
        elements = extract_elements(pdf_path, len(reader.pages))
        object_prefix = (
            f"{safe_document_slug(pdf_path.name)}/"
            f"{content_hash[:12]}-{signature[:12]}"
        )
        structured_chunks = build_structured_chunks(
            pdf_path,
            elements,
            leaves,
            model.tokenizer,
            page_text_fn=pages.text,
            storage=storage,
            object_prefix=object_prefix,
        )
    chunks = [*narrative_chunks, *structured_chunks]
    document_title = title_for_filename(pdf_path.name)
    raw_embeddings = embed_chunks(model, chunks)
    contextual_embeddings = embed_contextual_chunks(model, chunks, document_title)
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
        "Ingested %s: %d narrative, %d table, %d figure chunks (%.1f%% coverage)",
        pdf_path.name,
        chunk_counts["narrative"],
        chunk_counts["table"],
        chunk_counts["figure"],
        coverage.ratio * 100,
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
    parser.add_argument(
        "--narrative-only",
        action="store_true",
        help="Ingest PDF text without Docling table/figure extraction.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip any filename already present, regardless of ingest signature.",
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
                ingest_pdf(
                    pdf_path,
                    conn,
                    model,
                    storage=storage,
                    structured_enabled=False if args.narrative_only else None,
                    skip_existing=args.skip_existing,
                )
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

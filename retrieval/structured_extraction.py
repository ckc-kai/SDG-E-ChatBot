"""Specialized local table/figure extraction used by the unified ingest.

1. Extraction: each PDF runs once through docling's layout + TableFormer
   pipeline (local models, no cloud calls). Tables come back as cell grids;
   pictures come back with cropped images, captions, and -- when
   `structured.picture_description` is enabled -- a local SmolVLM-256M
   description.

2. Section mapping: every extracted element is mapped back into the SAME
   bookmark leaf-section tree the narrative ingest builds (reusing
   `retrieval.ingest.build_leaf_sections`), by page containment. Structured
   chunks therefore carry the same breadcrumb / sub_document / section_number
   metadata as their narrative siblings, and the contextual-embedding recipe
   applies unchanged.

3. Table output is dual: a Markdown flattening in ``content`` plus the exact cell grid in
   `structured_data` JSONB so downstream consumers can quote precise cells.
   Oversized tables are split row-aware -- every piece repeats the header row
   -- instead of blind token windows.

4. Figures keep authoritative caption/page context in ``content`` and return
   the local VLM description separately as ``retrieval_hint``. The unified
   ingest decides whether that hint participates in candidate generation; it
   is excluded from reranking and answer evidence by default.

Known limitations (accepted for this corpus, see module tests):
  - A table split across a `page_batch_size` boundary is extracted as two
    tables. Batching is off by default; the whole-document pass keeps
    docling's own multi-page table merging.
  - Table content also appears flattened inside narrative chunks. The mild
    duplication helps recall and avoids a risky "subtract table regions from
    page text" pass; the reranker sees both representations.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from retrieval.ingest import (
    Chunk,
    LeafSection,
    chunk_text,
)
from retrieval.object_storage import ObjectStorage
from retrieval.utils import embedding_config, load_config

logger = logging.getLogger(__name__)

_config = load_config()
_embedding_config = embedding_config()
_structured_config = _config.get("extraction", {}).get("structured", {})

MAX_TOKENS_PER_CHUNK = _embedding_config["max_tokens_per_chunk"]
MIN_CHUNK_CHARS = _embedding_config["minimum_chunk_chars"]

EXTRACTOR_VERSION = _structured_config.get("extractor_version", "docling-v1")
MIN_TABLE_ROWS = _structured_config.get("min_table_rows", 2)
MIN_TABLE_COLS = _structured_config.get("min_table_cols", 2)
MIN_FIGURE_AREA_PX = _structured_config.get("min_figure_area_px", 10000)
IMAGES_SCALE = float(_structured_config.get("images_scale", 2.0))
_description_config = _structured_config.get("figure_description", {})
PICTURE_DESCRIPTION = bool(_description_config.get("generate", True))
PAGE_BATCH_SIZE = int(_structured_config.get("page_batch_size", 0))

# Reserved headroom when packing table rows against MAX_TOKENS_PER_CHUNK, so
# joins/newlines never push a composed piece over the embedding window.
_TOKEN_PACK_SLACK = 8

# Bumped whenever the noise filters below change, so unchanged PDFs re-extract
# under the new rules (config's extractor_version covers deliberate re-runs).
_FILTERS_VERSION = 3

# Page text prepended to figure chunks ("Page context"). SmolVLM-256M
# descriptions are visually literal but semantically thin ("we can see a crane
# and a drone"); the figure's own page text (slide title, surrounding prose)
# carries the domain vocabulary retrieval actually matches on.
_FIGURE_PAGE_CONTEXT_CHARS = 300

# Dot-leader runs ("Section 2 ...... 14") appear in tables of contents that the
# layout model sometimes mislabels as data tables, and waste tokens inside
# legitimate cells.
_DOT_RUN_RE = re.compile(r"\.{4,}")
# Fraction of '.' characters above which a "table" is a table of contents.
_MAX_TABLE_DOT_RATIO = 0.25
# Captionless figures whose local-VLM description is about branding artwork.
_LOGO_DESCRIPTION_RE = re.compile(r"\b(logo|watermark|letterhead)\b", re.IGNORECASE)


@dataclass(frozen=True)
class StructuredElement:
    """One extracted table or figure, before chunking/embedding."""

    content_type: str  # 'table' | 'figure'
    page_start: int  # 0-based, matching the narrative pipeline
    page_end: int  # exclusive
    caption: str
    grid: tuple[tuple[str, ...], ...] | None = None  # tables only
    description: str = ""  # figures only
    image: object | None = None  # figures only (PIL.Image)


def extractor_signature() -> str:
    """Encodes every option that changes extraction output, so config changes
    invalidate the per-document skip cache."""
    try:
        from importlib.metadata import version

        docling_version = version("docling")
    except Exception:
        docling_version = "unknown"
    return (
        f"{EXTRACTOR_VERSION}|docling={docling_version}"
        f"|desc={PICTURE_DESCRIPTION}|scale={IMAGES_SCALE}"
        f"|min_table={MIN_TABLE_ROWS}x{MIN_TABLE_COLS}"
        f"|min_fig={MIN_FIGURE_AREA_PX}|page_batch={PAGE_BATCH_SIZE}"
        f"|filters={_FILTERS_VERSION}"
    )


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested without docling)
# ---------------------------------------------------------------------------


def _markdown_cell(text: str) -> str:
    cleaned = _DOT_RUN_RE.sub(" ", text or "")
    return " ".join(cleaned.split()).replace("|", "\\|")


def is_noise_table(grid: Sequence[Sequence[str]]) -> bool:
    """True for grids that are layout debris rather than data tables.

    The dominant failure observed on this corpus is a table-of-contents page
    mislabeled as a table: two columns of dot-leader rows. Those cells are
    mostly '.' characters, so a simple density check separates them from real
    data tables (which the corpus's QDR-style tables never approach).
    """
    joined = "".join(cell for row in grid for cell in row)
    stripped = "".join(joined.split())
    if not stripped:
        return True
    return stripped.count(".") / len(stripped) > _MAX_TABLE_DOT_RATIO


def is_noise_figure(caption: str, description: str) -> bool:
    """True for captionless branding artwork (logos on cover pages)."""
    return not caption and bool(_LOGO_DESCRIPTION_RE.search(description or ""))


def grid_to_markdown(grid: Sequence[Sequence[str]]) -> str:
    """Render a cell grid as a Markdown table; the first row is the header."""
    if not grid:
        return ""
    width = max(len(row) for row in grid)

    def line(row: Sequence[str]) -> str:
        cells = [_markdown_cell(cell) for cell in row] + [""] * (width - len(row))
        return "| " + " | ".join(cells) + " |"

    lines = [line(grid[0]), "| " + " | ".join(["---"] * width) + " |"]
    lines.extend(line(row) for row in grid[1:])
    return "\n".join(lines)


def chunk_table_grid(
    grid: Sequence[Sequence[str]],
    tokenizer,
    max_tokens: int,
    reserved_tokens: int = 0,
) -> list[tuple[int, int]]:
    """Split a table's BODY rows into [start, end) ranges that fit max_tokens.

    Every piece is rendered with the header row repeated, so budgeting reserves
    the header's (and separator's) tokens up front; `reserved_tokens` covers
    piece-level extras such as the caption line. A single body row larger than
    the remaining budget still gets its own piece (downstream guards re-split
    by tokens). Returns ranges over body-row indices (grid row 1..n-1).
    """
    body_count = max(len(grid) - 1, 0)
    if body_count == 0:
        return [(0, 0)]

    def token_len(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False))

    # grid_to_markdown of the header row alone renders header + separator.
    header_tokens = token_len(grid_to_markdown([grid[0]]))
    budget = max(max_tokens - header_tokens - reserved_tokens - _TOKEN_PACK_SLACK, 1)

    row_tokens = [
        token_len("| " + " | ".join(_markdown_cell(cell) for cell in row) + " |")
        for row in grid[1:]
    ]

    ranges: list[tuple[int, int]] = []
    start = 0
    used = 0
    for index, tokens in enumerate(row_tokens):
        if index > start and used + tokens > budget:
            ranges.append((start, index))
            start = index
            used = 0
        used += tokens
    ranges.append((start, body_count))
    return ranges


def piece_markdown(grid: Sequence[Sequence[str]], row_range: tuple[int, int]) -> str:
    """Markdown for one body-row range, header repeated."""
    start, end = row_range
    return grid_to_markdown([grid[0], *grid[1 + start : 1 + end]])


def compose_table_text(caption: str, markdown: str) -> str:
    parts = []
    if caption:
        parts.append(f"Table: {caption}")
    parts.append(markdown)
    return "\n".join(parts).strip()


def compose_figure_text(caption: str, page_context: str = "") -> str:
    parts = []
    if caption:
        parts.append(f"Figure: {caption}")
    if page_context:
        context = " ".join(page_context.split())[:_FIGURE_PAGE_CONTEXT_CHARS]
        if context:
            parts.append(f"Page context: {context}")
    return "\n".join(parts).strip()


def map_page_to_leaf(leaves: Sequence[LeafSection], page_idx: int) -> LeafSection | None:
    """Most specific leaf section containing a page.

    Boundary pages can belong to the tail of one section and the start of the
    next; the later-starting leaf wins because docling elements on a shared
    page overwhelmingly belong to the section that begins there.
    """
    best: LeafSection | None = None
    for leaf in leaves:
        if leaf.page_start <= page_idx < leaf.page_end:
            if best is None or leaf.page_start >= best.page_start:
                best = leaf
    return best


def safe_document_slug(filename: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", Path(filename).stem)


# ---------------------------------------------------------------------------
# Docling extraction (imports deferred: the module stays importable/testable
# without paying docling's multi-second import or model downloads)
# ---------------------------------------------------------------------------

_converter = None


def _get_converter():
    global _converter
    if _converter is not None:
        return _converter

    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import (
        PdfPipelineOptions,
        smolvlm_picture_description,
    )
    from docling.document_converter import DocumentConverter, PdfFormatOption

    options = PdfPipelineOptions()
    options.do_table_structure = True
    options.table_structure_options.do_cell_matching = True
    options.generate_picture_images = True
    options.images_scale = IMAGES_SCALE
    if PICTURE_DESCRIPTION:
        options.do_picture_description = True
        options.picture_description_options = smolvlm_picture_description

    _converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
    )
    return _converter


def _table_grid(table) -> tuple[tuple[str, ...], ...] | None:
    try:
        return tuple(
            tuple(getattr(cell, "text", "") or "" for cell in row)
            for row in table.data.grid
        )
    except Exception:
        logger.debug("Could not read table grid", exc_info=True)
        return None


def _prov_page_span(item) -> tuple[int, int] | None:
    """(page_start, page_end_exclusive), 0-based, from docling provenance."""
    pages = sorted({prov.page_no for prov in getattr(item, "prov", []) or []})
    if not pages:
        return None
    return pages[0] - 1, pages[-1]  # docling page_no is 1-based


def _picture_description(picture) -> str:
    for annotation in getattr(picture, "annotations", []) or []:
        if getattr(annotation, "kind", "") == "description":
            return (getattr(annotation, "text", "") or "").strip()
    return ""


def extract_elements(pdf_path: Path, page_count: int) -> list[StructuredElement]:
    converter = _get_converter()
    if PAGE_BATCH_SIZE > 0:
        page_ranges = [
            (start, min(start + PAGE_BATCH_SIZE - 1, page_count))
            for start in range(1, page_count + 1, PAGE_BATCH_SIZE)
        ]
    else:
        page_ranges = [(1, page_count)]

    elements: list[StructuredElement] = []
    for page_range in page_ranges:
        result = converter.convert(pdf_path, page_range=page_range, raises_on_error=True)
        document = result.document

        for table in document.tables:
            span = _prov_page_span(table)
            grid = _table_grid(table)
            if span is None or grid is None:
                continue
            if len(grid) < MIN_TABLE_ROWS or max(map(len, grid)) < MIN_TABLE_COLS:
                continue
            if is_noise_table(grid):
                logger.debug("Skipping noise table (TOC/dot-leaders) on page %d", span[0])
                continue
            elements.append(
                StructuredElement(
                    content_type="table",
                    page_start=span[0],
                    page_end=span[1],
                    caption=(table.caption_text(doc=document) or "").strip(),
                    grid=grid,
                )
            )

        for picture in document.pictures:
            span = _prov_page_span(picture)
            if span is None:
                continue
            image = None
            try:
                image = picture.get_image(document)
            except Exception:
                logger.debug("Could not render picture image", exc_info=True)
            if image is not None:
                # Normalize by render scale so the config threshold means
                # page-coordinate pixels regardless of images_scale.
                page_area = (image.width * image.height) / (IMAGES_SCALE**2)
                if page_area < MIN_FIGURE_AREA_PX:
                    continue  # decoration
            caption = (picture.caption_text(doc=document) or "").strip()
            description = _picture_description(picture)
            if is_noise_figure(caption, description):
                logger.debug("Skipping branding figure on page %d", span[0])
                continue
            elements.append(
                StructuredElement(
                    content_type="figure",
                    page_start=span[0],
                    page_end=span[1],
                    caption=caption,
                    description=description,
                    image=image,
                )
            )

    elements.sort(key=lambda el: (el.page_start, el.content_type))
    return elements


# ---------------------------------------------------------------------------
# Chunk building
# ---------------------------------------------------------------------------


def _save_figure_image(
    image,
    object_prefix: str,
    page_idx: int,
    ordinal: int,
    storage: ObjectStorage | None,
) -> str | None:
    if image is None:
        return None
    if storage is None:
        return None
    key = f"{object_prefix}/p{page_idx:04d}_f{ordinal}.png"
    return storage.put_png(key, image)


def build_structured_chunks(
    pdf_path: Path,
    elements: Sequence[StructuredElement],
    leaves: Sequence[LeafSection],
    tokenizer,
    page_text_fn=None,
    storage: ObjectStorage | None = None,
    object_prefix: str | None = None,
) -> list[Chunk]:
    """`page_text_fn(page_idx) -> str` supplies figure page context (optional)."""
    document_slug = safe_document_slug(pdf_path.name)
    object_prefix = object_prefix or document_slug
    counters: dict[tuple[str, int], int] = {}
    figure_ordinals: dict[int, int] = {}
    chunks: list[Chunk] = []

    def next_index(content_type: str, page_start: int) -> int:
        key = (content_type, page_start)
        counters[key] = counters.get(key, 0) + 1
        return counters[key] - 1

    def token_count(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False))

    def add_chunk(
        element: StructuredElement,
        content: str,
        retrieval_hint: str | None,
        structured_data: dict | None,
        object_key: str | None,
    ) -> None:
        leaf = map_page_to_leaf(leaves, element.page_start)
        chunks.append(
            Chunk(
                source_pdf=pdf_path.name,
                sub_document=leaf.sub_document if leaf else None,
                breadcrumb=leaf.breadcrumb if leaf else pdf_path.name,
                section_number=leaf.section_number if leaf else None,
                page_start=element.page_start,
                page_end=element.page_end,
                chunk_index=next_index(element.content_type, element.page_start),
                content_type=element.content_type,
                content=content,
                retrieval_hint=retrieval_hint,
                caption=element.caption or None,
                structured_data=structured_data,
                object_key=object_key,
                media_type="image/png" if object_key else None,
                token_count=token_count(content),
                extractor=extractor_signature(),
            )
        )

    for element in elements:
        if element.content_type == "table":
            grid = element.grid or ()
            full_markdown = grid_to_markdown(grid)
            base_data = {
                "caption": element.caption,
                "num_rows": len(grid),
                "num_cols": max(map(len, grid)) if grid else 0,
                "grid": [list(row) for row in grid],
            }
            caption_tokens = (
                token_count(f"Table: {element.caption}") if element.caption else 0
            )
            added_before = len(chunks)
            for row_range in chunk_table_grid(
                grid, tokenizer, MAX_TOKENS_PER_CHUNK, reserved_tokens=caption_tokens
            ):
                content = compose_table_text(element.caption, piece_markdown(grid, row_range))
                if len(content) < MIN_CHUNK_CHARS:
                    continue
                # A single monster row can exceed the window even alone; fall
                # back to plain token windows so the contextual-embedding
                # guard never rejects the piece downstream.
                if token_count(content) > MAX_TOKENS_PER_CHUNK:
                    pieces = chunk_text(content, tokenizer, MAX_TOKENS_PER_CHUNK, overlap=0)
                else:
                    pieces = [content]
                for piece in pieces:
                    add_chunk(
                        element,
                        piece,
                        None,
                        {**base_data, "piece_rows": list(row_range)},
                        None,
                    )
            # Guard against pathological tables where every piece was dropped:
            # fall back to one token-window pass over the whole flattening.
            if len(chunks) == added_before and len(full_markdown) >= MIN_CHUNK_CHARS:
                for piece in chunk_text(
                    compose_table_text(element.caption, full_markdown),
                    tokenizer,
                    MAX_TOKENS_PER_CHUNK,
                    overlap=0,
                ):
                    add_chunk(element, piece, None, base_data, None)
        else:
            page_context = page_text_fn(element.page_start) if page_text_fn else ""
            content = compose_figure_text(element.caption, page_context)
            if not content:
                logger.debug(
                    "Skipping figure without authoritative caption/page context "
                    "on page %d of %s",
                    element.page_start,
                    pdf_path.name,
                )
                continue
            ordinal = figure_ordinals.get(element.page_start, 0)
            figure_ordinals[element.page_start] = ordinal + 1
            object_key = _save_figure_image(
                element.image,
                object_prefix,
                element.page_start,
                ordinal,
                storage,
            )
            add_chunk(
                element,
                content,
                element.description or None,
                None,
                object_key,
            )

    return chunks

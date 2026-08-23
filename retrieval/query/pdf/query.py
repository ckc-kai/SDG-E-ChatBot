"""Retrieve PDF narrative, table, and figure chunks.

1. query_rewrite: rewrite complex/causal questions
2. search: embeds each (sub-)question with the same model used at ingestion
   time and does a pgvector cosine-distance search against the raw, contextual,
   or hybrid raw+contextual document embeddings in ``chunks``, plus a local
   PostgreSQL lexical lane over captions, breadcrumbs, and content
3. Candidate pooling: results from every sub-question are merged and deduped
   by chunk id.
4. When lanes are enabled, candidate generation and reranking happen
   independently for narrative, structured PDF, and Excel-card content.
5. Lane results are merged and exact-content deduplicated with no type quota.

Excel cards can participate in the candidate merge, but validated Excel answer
execution lives in ``retrieval.query.excel``.
"""

import argparse
import functools
import json
import logging
import math
import re
from dataclasses import dataclass, field, replace

from generation.features import feature_enabled
from retrieval.query.pdf.expansion import expand_to_parent_sections
from retrieval.query.calibration import Calibrator
from retrieval.contextual_embeddings import CONTEXTUAL_EMBEDDING_RECIPE
from retrieval.failure_log import get_failure_logger
from retrieval.query.lanes import (
    ALL_LANES,
    EXCEL,
    NARRATIVE,
    LaneOutcome,
    content_types_for,
    lane_confidence,
)
from retrieval.object_storage import get_object_storage
from retrieval.source_manifest import SourceManifest
from retrieval.utils import (
    connect_db,
    embedding_config,
    encode_query,
    get_anthropic_client,
    get_embedding_model,
    rerank_scores,
    load_config,
)

logger = logging.getLogger(__name__)
log_failure = get_failure_logger("query")

_config = load_config()
_rewrite_config = _config.get("query_rewrite", {})
_retrieval_config = _config.get("retrieval", {})
_embedding_config = embedding_config()
QUERY_REWRITE_MODEL = _rewrite_config.get("model", "claude-haiku-4-5-20251001")
QUERY_REWRITE_MODE = _rewrite_config.get("mode", "auto")
MAX_REWRITE_SUBQUESTIONS = _rewrite_config.get("max_subquestions", 2)
RETRIEVAL_TOP_K = _retrieval_config.get("retrieval_top_k", 30)
RERANK_TOP_K = _retrieval_config.get("rerank_top_k", 10)
RRF_K = _retrieval_config.get("rrf_k", 60)
DEFAULT_EMBEDDING_MODE = _retrieval_config.get("embedding_mode", "hybrid")
DEFAULT_HYBRID_POOL_MODE = _retrieval_config.get("hybrid_pool_mode", "union")
EMBEDDING_MODEL_NAME = _embedding_config["name"]
CAPTION_RERANK_WEIGHT = float(_retrieval_config.get("caption_rerank_weight", 0.25))
LEXICAL_CAPTION_WEIGHT = float(_retrieval_config.get("lexical_caption_weight", 4.0))
LEXICAL_HINT_WEIGHT = float(_retrieval_config.get("lexical_hint_weight", 0.25))
_lexical_query_mode = _retrieval_config.get("lexical_query_mode", "focused")
LEXICAL_QUERY_MODE = (
    "off" if _lexical_query_mode is False else str(_lexical_query_mode).lower()
)
RERANK_BATCH_SIZE = int(_retrieval_config.get("rerank_batch_size", 8))
DEDUPLICATE_EXACT_CONTENT = bool(
    _retrieval_config.get("deduplicate_exact_content", True)
)
_figure_description_config = (
    _config.get("extraction", {}).get("structured", {}).get("figure_description", {})
)
HINT_IN_LEXICAL_RETRIEVAL = bool(
    _figure_description_config.get("candidate_retrieval", True)
)
HINT_IN_RERANKING = bool(_figure_description_config.get("reranking", False))
LANE_MODE = _retrieval_config.get("lane_mode", "off")
CONFIDENCE_FLOOR = float(_retrieval_config.get("lane_confidence_floor", 0.0))
# Cross-lane score calibration is OFF by default and measured as harmful; see
# docs/retrieval_ranking_fix_plan.md. A per-content-type curve cannot be made
# comparable across lanes because the relevance base rate is a property of the
# question, not of the content type, and estimating it per type leaks the
# training suite's question mix into ranking. Enable only for diagnostics.
SCORE_CALIBRATION = str(_retrieval_config.get("score_calibration", "off")).lower()
_calibrator = (
    Calibrator.load()
    if SCORE_CALIBRATION not in {"off", "false", "0"}
    else Calibrator({})
)
EMBEDDING_MODES = ("raw", "contextual", "hybrid")
HYBRID_POOL_MODES = ("rrf", "union")
EVIDENCE_GROUP_CONTENT_TYPES: dict[str, tuple[str, ...]] = {
    NARRATIVE: ("narrative",),
    "table": ("table",),
    "figure": ("figure",),
    EXCEL: ("excel_card",),
}
EVIDENCE_GROUPS = tuple(EVIDENCE_GROUP_CONTENT_TYPES)
_EMBEDDING_COLUMNS = {
    "raw": "embedding",
    "contextual": "contextual_embedding",
}
_PARTIAL_INDEX_FILTERS: dict[tuple[str, ...], str] = {
    ("narrative",): "AND c.content_type = 'narrative'",
    ("table",): "AND c.content_type = 'table'",
    ("figure",): "AND c.content_type = 'figure'",
    ("excel_card",): "AND c.content_type = 'excel_card'",
}

_LEXICAL_STOPWORDS = {
    "about",
    "after",
    "also",
    "among",
    "and",
    "are",
    "before",
    "between",
    "did",
    "does",
    "for",
    "from",
    "have",
    "how",
    "into",
    "many",
    "much",
    "not",
    "over",
    "that",
    "the",
    "their",
    "them",
    "then",
    "these",
    "this",
    "through",
    "under",
    "what",
    "when",
    "where",
    "which",
    "who",
    "with",
    "would",
}
_LEXICAL_GENERIC_TERMS = {
    "compare",
    "explain",
    "mitigation",
    "plan",
    "provide",
    "report",
    "review",
    "risk",
    "sdge",
    "wildfire",
    "wmp",
}

_REWRITE_MODES = {"auto", "off", "always"}
_INTERROGATIVE = (
    r"(?:what|who|when|where|why|how|which|"
    r"is|are|was|were|do|does|did|has|have|can|could|should|would|will)"
)
_SECOND_INTERROGATIVE_CLAUSE_RE = re.compile(
    rf"(?:\b(?:and|or)\b|[.;?])\s*,?\s*{_INTERROGATIVE}\b",
    re.IGNORECASE,
)

_DECOMPOSE_SYSTEM_PROMPT = (
    "You split a compound question about SDG&E's Wildfire Mitigation Plan into "
    "focused, self-contained sub-questions that can each be answered independently. "
    f"Return no more than {MAX_REWRITE_SUBQUESTIONS} sub-questions. "
    "Respond with ONLY a JSON array of strings, no other text. "
    "If the question is already simple, return a single-element array containing it unchanged."
)

_JSON_FENCE_RE = re.compile(r"^```[a-zA-Z]*\s*\n?|\n?```\s*$")


def _strip_json_fence(text: str) -> str:
    """Strip a leading/trailing ```json ... ``` markdown fence, if present.

    Claude reliably follows "respond with ONLY a JSON array" for the array's
    CONTENT, but still sometimes wraps it in a markdown code fence. That leading
    backtick makes json.loads fail with "Expecting value: line 1 column 1", so we
    strip the fence (if any) before parsing. A no-op on already-bare JSON.
    """
    return _JSON_FENCE_RE.sub("", text.strip()).strip()


_rewrite_call_count = 0
_last_rewrite_trace: dict | None = None


def get_rewrite_call_count() -> int:
    """Total number of Claude API calls made by query_rewrite() since the last reset."""
    return _rewrite_call_count


def reset_rewrite_call_count() -> None:
    global _rewrite_call_count
    _rewrite_call_count = 0


def get_last_rewrite_trace() -> dict | None:
    """Full record of the most recent query_rewrite() call.

    {"question": str, "sub_questions": list[str], "source": str,
     "reason": str, "api_sub_questions": list[str], "error": str | None}
    `source` tells you what actually happened: "simple" = no API call needed,
    "api" = Claude successfully decomposed it, "fallback" = an API call was made
    but failed (see "error"), so the original question was used unchanged.
    """
    return _last_rewrite_trace


@dataclass(frozen=True)
class QueryObject:
    chunk_id: int
    source_pdf: str
    sub_document: str | None
    breadcrumb: str
    section_number: str | None
    page_start: int
    page_end: int
    chunk_index: int
    content_type: str
    content: str
    retrieval_hint: str | None
    caption: str | None
    structured_data: dict | None
    object_key: str | None
    media_type: str | None
    content_hash: str
    token_count: int
    distance: float
    retrieval_score: float | None = None


@dataclass(frozen=True)
class RankedResult:
    query_object: QueryObject
    rerank_score: float


@dataclass(frozen=True)
class RetrievalDiagnostics:
    search_queries: list[str]
    candidate_sets: list[list[QueryObject]]
    channel_candidate_sets: list[dict[str, list[QueryObject]]]
    pooled_candidates: list[QueryObject]
    reranked_candidates: list[RankedResult]
    lane_outcomes: list[LaneOutcome] = field(default_factory=list)


@dataclass(frozen=True)
class EvidenceGroup:
    """Independently ranked evidence for one authoritative content shape."""

    name: str
    content_types: tuple[str, ...]
    results: list[RankedResult]
    diagnostics: RetrievalDiagnostics


@dataclass(frozen=True)
class EvidenceRetrievalResult:
    """Retrieval output whose scores are comparable only inside each group."""

    question: str
    groups: dict[str, EvidenceGroup]


def _decomposition_reason(question: str) -> str | None:
    """Return a high-precision reason to decompose, independent of eval labels.

    Length, auxiliary verbs, identifiers containing periods, and noun lists are
    deliberately not treated as evidence of multiple retrieval intents. The API
    is reserved for a second explicit interrogative clause.
    """
    normalized = " ".join(question.strip().split())
    if normalized.count("?") > 1:
        return "multiple_questions"
    if _SECOND_INTERROGATIVE_CLAUSE_RE.search(normalized):
        return "independent_interrogative_clauses"
    return None


def _needs_decomposition(question: str) -> bool:
    return _decomposition_reason(question) is not None


@dataclass(frozen=True)
class SourceRole:
    """A document role explicitly required by the user's question."""

    name: str
    query: str
    filename_patterns: tuple[str, ...]


@functools.lru_cache(maxsize=1)
def _source_manifest() -> SourceManifest:
    return SourceManifest.load()


def _filenames_for_role(role: str, fallback: tuple[str, ...]) -> tuple[str, ...]:
    """Resolve stable role metadata to current corpus filenames.

    The fallback keeps an existing deployment operational if its checked-out
    manifest is unavailable during a rolling release.
    """
    if not feature_enabled("metadata_routing"):
        return fallback
    try:
        resolved = _source_manifest().filenames_for_role(role)
    except (OSError, ValueError, json.JSONDecodeError):
        logger.exception("Could not load source-role manifest; using compatibility path")
        return fallback
    return resolved or fallback


def required_source_roles(question: str) -> tuple[SourceRole, ...]:
    """Route explicit WMP/guideline comparisons without a model call."""
    normalized = " ".join(question.strip().split())
    lowered = normalized.casefold()
    if "wmp" not in lowered or "guideline" not in lowered:
        return ()
    if "oeis" in lowered:
        return (
            SourceRole(
                name="oeis_decision",
                query=(
                    "2023-2025 OEIS decision areas for continued improvement risk "
                    "methodology mitigation selection prioritization"
                ),
                filename_patterns=(
                    *_filenames_for_role(
                        "oeis_decision_2023",
                        ("FINAL_SDGE_20232025_WMP_Decision_and_Cover_Letter.pdf",),
                    ),
                ),
            ),
            SourceRole(
                name="wmp_guidelines",
                query=(
                    "2026-2028 WMP guidelines requirements risk methodology activity "
                    "selection prioritization scheduling risk reduction"
                ),
                filename_patterns=(
                    *_filenames_for_role(
                        "wmp_guidelines_2026_2028",
                        ("FINAL 2026-2028_Wildfire_Mitigation_Plan_Guidelines.pdf",),
                    ),
                ),
            ),
        )

    mentions_2023 = "2023" in lowered or "2023-2025" in lowered
    mentions_2026 = "2026" in lowered or "2026-2028" in lowered
    include_both = not mentions_2023 and not mentions_2026
    roles: list[SourceRole] = []
    if mentions_2023 or include_both:
        roles.extend(
            (
                SourceRole(
                    name="2023_wmp",
                    query=f"{normalized} Evidence from the 2023-2025 WMP.",
                    filename_patterns=(
                        *_filenames_for_role(
                            "wmp_2023_2025",
                            ("SDG&E_2023-2023_Base-WMP_R5-redacted.pdf",),
                        ),
                    ),
                ),
                SourceRole(
                    name="2023_guidelines",
                    query=f"{normalized} Requirements from the 2023-2025 guidelines.",
                    filename_patterns=(
                        *_filenames_for_role(
                            "wmp_guidelines_2023_2025",
                            ("2023-2025_WMP_TECHNICAL_GUIDELINES.pdf",),
                        ),
                    ),
                ),
            )
        )
    if mentions_2026 or include_both:
        roles.extend(
            (
                SourceRole(
                    name="2026_wmp",
                    query=f"{normalized} Evidence from the 2026-2028 WMP.",
                    filename_patterns=_filenames_for_role(
                        "wmp_2026_2028", ("SDG&E_2026-2028_Base-WMP_R2.pdf",)
                    ),
                ),
                SourceRole(
                    name="2026_guidelines",
                    query=f"{normalized} Requirements from the 2026-2028 guidelines.",
                    filename_patterns=(
                        *_filenames_for_role(
                            "wmp_guidelines_2026_2028",
                            ("FINAL 2026-2028_Wildfire_Mitigation_Plan_Guidelines.pdf",),
                        ),
                    ),
                ),
            )
        )
    return tuple(roles)


def _deduplicate_subquestions(question: str, sub_questions: list[str]) -> list[str]:
    seen = {question.casefold()}
    unique: list[str] = []
    for item in sub_questions:
        cleaned = " ".join(item.strip().split())
        signature = cleaned.casefold()
        if not cleaned or signature in seen:
            continue
        seen.add(signature)
        unique.append(cleaned)
        if len(unique) >= MAX_REWRITE_SUBQUESTIONS:
            break
    return unique


def query_rewrite(question: str, mode: str | None = None) -> list[str]:
    global _rewrite_call_count, _last_rewrite_trace
    question = question.strip()
    selected_mode = mode or QUERY_REWRITE_MODE
    if selected_mode not in _REWRITE_MODES:
        raise ValueError(
            f"Unknown rewrite mode {selected_mode!r}; expected one of {sorted(_REWRITE_MODES)}"
        )

    reason = _decomposition_reason(question)
    if selected_mode == "off":
        _last_rewrite_trace = {
            "question": question,
            "sub_questions": [question],
            "source": "disabled",
            "reason": "mode_off",
            "api_sub_questions": [],
            "error": None,
        }
        return [question]
    if selected_mode == "auto" and reason is None:
        _last_rewrite_trace = {
            "question": question,
            "sub_questions": [question],
            "source": "simple",
            "reason": "single_intent",
            "api_sub_questions": [],
            "error": None,
        }
        return [question]
    if selected_mode == "always":
        reason = "mode_always"

    try:
        _rewrite_call_count += 1
        response = get_anthropic_client().messages.create(
            model=QUERY_REWRITE_MODEL,
            max_tokens=512,
            system=_DECOMPOSE_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": question}],
        )
        sub_questions = json.loads(_strip_json_fence(response.content[0].text))
        if (
            not isinstance(sub_questions, list)
            or not sub_questions
            or not all(isinstance(item, str) for item in sub_questions)
        ):
            raise ValueError(f"Unexpected rewrite response shape: {sub_questions!r}")
        search_queries = [question, *_deduplicate_subquestions(question, sub_questions)]
        _last_rewrite_trace = {
            "question": question,
            "sub_questions": search_queries,
            "source": "api",
            "reason": reason,
            "api_sub_questions": sub_questions,
            "error": None,
        }
        return search_queries
    except Exception as exc:
        log_failure("query_rewrite", question, exc)
        logger.warning(
            "query_rewrite failed, falling back to original question: %s", exc
        )
        _last_rewrite_trace = {
            "question": question,
            "sub_questions": [question],
            "source": "fallback",
            "reason": reason,
            "api_sub_questions": [],
            "error": str(exc),
        }
        return [question]


def _validate_embedding_mode(conn, embedding_mode: str) -> str:
    if embedding_mode not in _EMBEDDING_COLUMNS:
        raise ValueError(
            f"Dense search requires one of {tuple(_EMBEDDING_COLUMNS)}; "
            f"got {embedding_mode!r}"
        )
    if embedding_mode == "contextual":
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    count(*) FILTER (
                        WHERE contextual_embedding IS NULL
                           OR contextual_embedding_model IS DISTINCT FROM %s
                    ) AS unusable,
                    array_agg(DISTINCT contextual_embedding_recipe) AS recipes
                FROM chunks
                """,
                (EMBEDDING_MODEL_NAME,),
            )
            unusable, recipes = cur.fetchone()
        if unusable:
            raise RuntimeError(
                f"{unusable} chunks have no contextual embedding or were "
                f"embedded with a model other than {EMBEDDING_MODEL_NAME!r}. "
                "Run `uv run python -m retrieval.ingest.pdf.ingest` to "
                "re-ingest them."
            )
        recipes = [recipe for recipe in (recipes or ()) if recipe]
        # A corpus embedded under a *mix* of recipes is half-migrated: its
        # vectors are not mutually comparable and ranking across them is
        # meaningless. A corpus embedded uniformly under an older recipe is
        # not: the recipe only describes how neighbouring context was
        # prepended before encoding, and the query vector is encoded plain
        # either way. Pinning this to the current recipe also rejected every
        # internally consistent older index, which is what the 768d bge
        # corpus on port 5433 is.
        if len(recipes) > 1:
            raise RuntimeError(
                "Contextual embeddings are half-migrated: the corpus mixes "
                f"recipes {sorted(recipes)}. Run `uv run python -m "
                "retrieval.ingest.pdf.ingest` to re-ingest them."
            )
        if recipes and recipes[0] != CONTEXTUAL_EMBEDDING_RECIPE:
            _warn_older_contextual_recipe(recipes[0])
    return _EMBEDDING_COLUMNS[embedding_mode]


_warned_recipes: set[str] = set()


def _warn_older_contextual_recipe(recipe: str) -> None:
    """Say once that the corpus predates the current contextualisation."""
    if recipe in _warned_recipes:
        return
    _warned_recipes.add(recipe)
    logger.warning(
        "Corpus contextual embeddings use recipe %r; the current recipe is "
        "%r. Retrieval is consistent but reflects the older contextualisation.",
        recipe,
        CONTEXTUAL_EMBEDDING_RECIPE,
    )


def _search_by_vector(
    query_vector,
    top_k: int,
    conn,
    *,
    embedding_mode: str,
    content_types: tuple[str, ...] | None = None,
    source_patterns: tuple[str, ...] | None = None,
    validate_embedding: bool = True,
) -> list[QueryObject]:
    embedding_column = (
        _validate_embedding_mode(conn, embedding_mode)
        if validate_embedding
        else _EMBEDDING_COLUMNS[embedding_mode]
    )
    # PostgreSQL cannot prove that ``content_type = ANY($1)`` implies a
    # partial-index predicate, even for a one-item runtime array. These four
    # clauses are fixed internal constants, not interpolated caller input, and
    # allow the independently ranked evidence groups to use their own HNSW
    # graphs. Multi-type compatibility lanes retain the parameterized filter.
    type_filter = _PARTIAL_INDEX_FILTERS.get(content_types) if content_types else None
    parameterized_type_filter = bool(content_types) and type_filter is None
    if parameterized_type_filter:
        type_filter = "AND c.content_type = ANY(%s)"
    type_filter = type_filter or ""
    source_filter = "AND d.filename ILIKE ANY(%s)" if source_patterns else ""
    params: list = [query_vector]
    if parameterized_type_filter:
        assert content_types is not None
        params.append(list(content_types))
    if source_patterns:
        params.append(list(source_patterns))
    params.append(top_k)
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT c.id, d.filename, c.sub_document, c.breadcrumb, c.section_number,
                   c.page_start, c.page_end, c.chunk_index, c.content_type,
                   c.content, c.retrieval_hint, c.caption, c.structured_data,
                   c.object_key, c.media_type, c.content_hash, c.token_count,
                   c.{embedding_column} <=> %s AS distance
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            WHERE true {type_filter} {source_filter}
            ORDER BY distance
            LIMIT %s
            """,
            params,
        )
        rows = cur.fetchall()

    return [
        QueryObject(
            chunk_id=row[0],
            source_pdf=row[1],
            sub_document=row[2],
            breadcrumb=row[3],
            section_number=row[4],
            page_start=row[5],
            page_end=row[6],
            chunk_index=row[7],
            content_type=row[8],
            content=row[9],
            retrieval_hint=row[10],
            caption=row[11],
            structured_data=row[12],
            object_key=row[13],
            media_type=row[14],
            content_hash=row[15],
            token_count=row[16],
            distance=row[17],
        )
        for row in rows
    ]


def search(
    question: str,
    top_k: int,
    conn,
    model,
    *,
    embedding_mode: str = DEFAULT_EMBEDDING_MODE,
) -> list[QueryObject]:
    if embedding_mode == "hybrid":
        raise ValueError("Use retrieve() for hybrid raw+contextual search.")
    query_vector = encode_query(model, question.strip())
    return _search_by_vector(
        query_vector,
        top_k,
        conn,
        embedding_mode=embedding_mode,
    )


def _broad_lexical_query(question: str) -> str:
    """Build an OR query from meaningful terms, including likely acronyms."""
    tokens: list[str] = []
    seen: set[str] = set()
    for token in re.findall(r"[A-Za-z0-9]+", question.lower()):
        if len(token) < 3 or token in _LEXICAL_STOPWORDS or token in seen:
            continue
        seen.add(token)
        tokens.append(token)
    for phrase in re.findall(r"\b[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){2,}\b", question):
        acronym = "".join(word[0] for word in phrase.split()).lower()
        if len(acronym) >= 3 and acronym not in seen:
            seen.add(acronym)
            tokens.append(acronym)
    return " OR ".join(tokens)


def _focused_lexical_query(question: str, max_terms: int = 4) -> str:
    """Build a selective AND query; dense channels preserve broad recall."""
    tokens = [
        token.casefold()
        for token in re.findall(r"[A-Za-z0-9]+", question)
        if len(token) >= 3
        and token.casefold() not in _LEXICAL_STOPWORDS
        and token.casefold() not in _LEXICAL_GENERIC_TERMS
    ]
    unique = list(dict.fromkeys(tokens))
    numeric = [token for token in unique if any(char.isdigit() for char in token)]
    distinctive = sorted(
        (token for token in unique if token not in numeric),
        key=lambda token: (-len(token), unique.index(token)),
    )
    selected = [*numeric, *distinctive][:max_terms]
    return " ".join(selected)


def _lexical_query(question: str) -> str:
    if LEXICAL_QUERY_MODE == "off":
        return ""
    if LEXICAL_QUERY_MODE == "focused":
        return _focused_lexical_query(question)
    return _broad_lexical_query(question)


def _search_lexical(
    question: str,
    top_k: int,
    conn,
    *,
    content_types: tuple[str, ...] | None = None,
    source_patterns: tuple[str, ...] | None = None,
) -> list[QueryObject]:
    expression = _lexical_query(question)
    if not expression:
        return []
    with conn.cursor() as cur:
        cur.execute(
            """
            WITH query AS (
                SELECT websearch_to_tsquery('english', %s) AS value
            ),
            corpus AS (
                SELECT c.*,
                       setweight(
                           to_tsvector(
                               'english', coalesce(c.caption, '')
                           ),
                           'A'
                       ) ||
                       setweight(
                           to_tsvector(
                               'english', coalesce(c.breadcrumb, '')
                           ),
                           'B'
                       ) ||
                       setweight(
                           to_tsvector('english', c.content),
                           'C'
                       ) AS authoritative_document,
                       to_tsvector(
                           'english', coalesce(c.retrieval_hint, '')
                       ) AS hint_document
                FROM chunks c
                WHERE (%s::text[] IS NULL
                   OR c.content_type = ANY(%s::text[]))
                  AND (%s::text[] IS NULL OR c.document_id IN (
                      SELECT id FROM documents WHERE filename ILIKE ANY(%s::text[])
                  ))
            )
            SELECT corpus.id, d.filename, corpus.sub_document,
                   corpus.breadcrumb, corpus.section_number,
                   corpus.page_start, corpus.page_end, corpus.chunk_index,
                   corpus.content_type, corpus.content, corpus.retrieval_hint,
                   corpus.caption, corpus.structured_data, corpus.object_key,
                   corpus.media_type, corpus.content_hash, corpus.token_count,
                   ts_rank_cd(corpus.authoritative_document, query.value)
                     + %s * ts_rank_cd(
                         to_tsvector(
                             'english', coalesce(corpus.caption, '')
                         ),
                         query.value
                     )
                     + CASE WHEN %s
                         THEN %s * ts_rank_cd(
                             corpus.hint_document, query.value
                         )
                         ELSE 0
                       END AS lexical_score
            FROM corpus
            JOIN documents d ON d.id = corpus.document_id
            CROSS JOIN query
            WHERE corpus.authoritative_document @@ query.value
               OR (%s AND corpus.hint_document @@ query.value)
            ORDER BY lexical_score DESC, corpus.id
            LIMIT %s
            """,
            (
                expression,
                list(content_types) if content_types else None,
                list(content_types) if content_types else None,
                list(source_patterns) if source_patterns else None,
                list(source_patterns) if source_patterns else None,
                LEXICAL_CAPTION_WEIGHT,
                HINT_IN_LEXICAL_RETRIEVAL,
                LEXICAL_HINT_WEIGHT,
                HINT_IN_LEXICAL_RETRIEVAL,
                top_k,
            ),
        )
        rows = cur.fetchall()
    return [
        QueryObject(
            chunk_id=row[0],
            source_pdf=row[1],
            sub_document=row[2],
            breadcrumb=row[3],
            section_number=row[4],
            page_start=row[5],
            page_end=row[6],
            chunk_index=row[7],
            content_type=row[8],
            content=row[9],
            retrieval_hint=row[10],
            caption=row[11],
            structured_data=row[12],
            object_key=row[13],
            media_type=row[14],
            content_hash=row[15],
            token_count=row[16],
            distance=-float(row[17]),
            retrieval_score=float(row[17]),
        )
        for row in rows
    ]


def reciprocal_rank_fusion(
    candidate_sets: list[list[QueryObject]],
    top_k: int,
    *,
    rrf_k: int = RRF_K,
) -> list[QueryObject]:
    """Fuse ranked channels by chunk id and retain a fixed candidate count."""
    if rrf_k < 1:
        raise ValueError("rrf_k must be positive")

    scores: dict[int, float] = {}
    best_rank: dict[int, int] = {}
    representative: dict[int, QueryObject] = {}
    for candidates in candidate_sets:
        for rank, candidate in enumerate(candidates, start=1):
            chunk_id = candidate.chunk_id
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (rrf_k + rank)
            best_rank[chunk_id] = min(best_rank.get(chunk_id, rank), rank)
            representative.setdefault(chunk_id, candidate)

    fused_ids = sorted(
        scores,
        key=lambda chunk_id: (-scores[chunk_id], best_rank[chunk_id], chunk_id),
    )[:top_k]
    return [
        replace(representative[chunk_id], retrieval_score=scores[chunk_id])
        for chunk_id in fused_ids
    ]


def raw_preserving_union(
    raw_candidates: list[QueryObject],
    contextual_candidates: list[QueryObject],
) -> list[QueryObject]:
    """Keep the complete raw ranking, then append contextual-only candidates.

    Raw is the current champion, so A3b must not allow the weaker contextual
    channel to evict a raw candidate. The existing reranker receives the full
    deduplicated union (up to twice the per-channel retrieval count).
    """
    combined = list(raw_candidates)
    seen_ids = {candidate.chunk_id for candidate in raw_candidates}
    for candidate in contextual_candidates:
        if candidate.chunk_id not in seen_ids:
            combined.append(candidate)
            seen_ids.add(candidate.chunk_id)
    return combined


def _merge_candidates(candidate_sets: list[list[QueryObject]]) -> list[QueryObject]:
    best_by_id: dict[int, QueryObject] = {}
    for candidates in candidate_sets:
        for candidate in candidates:
            existing = best_by_id.get(candidate.chunk_id)
            # RRF scores are comparable across hybrid sub-questions. Single-channel
            # candidates retain the existing cosine-distance comparison.
            candidate_quality = (
                candidate.retrieval_score
                if candidate.retrieval_score is not None
                else -candidate.distance
            )
            existing_quality = (
                existing.retrieval_score
                if existing is not None and existing.retrieval_score is not None
                else -existing.distance if existing is not None else float("-inf")
            )
            if existing is None or candidate_quality > existing_quality:
                best_by_id[candidate.chunk_id] = candidate
    return list(best_by_id.values())


def _candidate_text(candidate: QueryObject) -> str:
    context: list[str] = []
    if candidate.breadcrumb:
        context.append(f"Section: {candidate.breadcrumb}")
    if candidate.section_number:
        context.append(f"Section number: {candidate.section_number}")
    if candidate.content_type != "narrative":
        context.append(f"Element: {candidate.content_type}")
    if candidate.caption:
        context.append(f"Caption: {candidate.caption}")
    context.append(candidate.content)
    if HINT_IN_RERANKING and candidate.retrieval_hint:
        context.append(f"Unverified visual retrieval hint: {candidate.retrieval_hint}")
    return "\n".join(context)


def _lexical_terms(text: str) -> set[str]:
    terms = {
        token[:-1] if token.endswith("s") and len(token) > 4 else token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) >= 3 and token not in _LEXICAL_STOPWORDS
    }
    for phrase in re.findall(r"\b[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){2,}\b", text):
        terms.add("".join(word[0] for word in phrase.split()).lower())
    return terms


def _caption_relevance(question: str, candidate: QueryObject) -> float:
    """Normalized caption overlap; narrative chunks never receive this prior."""
    if candidate.content_type == "narrative" or not candidate.caption:
        return 0.0
    question_terms = _lexical_terms(question)
    caption_terms = _lexical_terms(candidate.caption)
    if not question_terms or not caption_terms:
        return 0.0
    overlap = len(question_terms & caption_terms)
    return overlap / (len(question_terms) * len(caption_terms)) ** 0.5


def deduplicate_ranked(
    ranked: list[RankedResult],
) -> list[RankedResult]:
    """Suppress byte-equivalent evidence without reserving content-type slots."""
    if not DEDUPLICATE_EXACT_CONTENT:
        return ranked
    deduplicated: list[RankedResult] = []
    seen_hashes: set[str] = set()
    for result in ranked:
        signature = result.query_object.content_hash
        if signature in seen_hashes:
            continue
        seen_hashes.add(signature)
        deduplicated.append(result)
    return deduplicated


def rerank(
    question: str, candidates: list[QueryObject], top_k: int | None
) -> list[RankedResult]:
    if not candidates:
        return []
    pairs = [(question, _candidate_text(candidate)) for candidate in candidates]
    scores = rerank_scores(pairs, batch_size=RERANK_BATCH_SIZE)
    ranked = sorted(
        (
            (
                candidate,
                float(score)
                + CAPTION_RERANK_WEIGHT * _caption_relevance(question, candidate),
            )
            for candidate, score in zip(candidates, scores, strict=True)
        ),
        key=lambda pair: pair[1],
        reverse=True,
    )
    results = [
        RankedResult(query_object=candidate, rerank_score=score)
        for candidate, score in ranked
    ]
    results = deduplicate_ranked(results)
    return results[:top_k] if top_k is not None else results


def _expand_parents(conn, results: list[RankedResult]) -> list[RankedResult]:
    """Widen returned narrative chunks to their surrounding section, if enabled.

    512-token children are a deliberate precision choice, and they are only safe
    because this restores the context a small chunk cannot carry. Turning the
    feature off without also raising the chunk size hands the model less context
    than either setting intends -- see the chunk-geometry note in config.yaml.
    """
    if not results or not feature_enabled("parent_child_expansion"):
        return results
    return expand_to_parent_sections(conn, results)


def retrieve_with_diagnostics(
    question: str,
    conn,
    *,
    rewrite_mode: str | None = None,
    retrieval_top_k: int = RETRIEVAL_TOP_K,
    rerank_top_k: int = RERANK_TOP_K,
    embedding_mode: str = DEFAULT_EMBEDDING_MODE,
    rrf_k: int = RRF_K,
    hybrid_pool_mode: str = DEFAULT_HYBRID_POOL_MODE,
    lanes: tuple[str, ...] | None = None,
    content_types: tuple[str, ...] | None = None,
    source_patterns: tuple[str, ...] | None = None,
    search_queries: list[str] | None = None,
    query_vectors: list | None = None,
    _skip_embedding_validation: bool = False,
) -> tuple[list[RankedResult], RetrievalDiagnostics]:
    """
    rewrite -> search -> merge -> rerank
    """
    if lanes and content_types:
        raise ValueError("lanes and content_types are mutually exclusive")
    if lanes is None and content_types is None and LANE_MODE not in {"off", "", None}:
        lanes = ALL_LANES
    if search_queries is None:
        search_queries = query_rewrite(question, mode=rewrite_mode)
    elif not search_queries:
        raise ValueError("search_queries must contain at least one query")
    if query_vectors is not None and len(query_vectors) != len(search_queries):
        raise ValueError("query_vectors must align one-to-one with search_queries")
    model = None if query_vectors is not None else get_embedding_model()
    if embedding_mode not in EMBEDDING_MODES:
        raise ValueError(
            f"Unknown embedding mode {embedding_mode!r}; expected one of {EMBEDDING_MODES}"
        )
    if hybrid_pool_mode not in HYBRID_POOL_MODES:
        raise ValueError(
            f"Unknown hybrid pool mode {hybrid_pool_mode!r}; "
            f"expected one of {HYBRID_POOL_MODES}"
        )
    if not _skip_embedding_validation:
        modes = (
            ("raw", "contextual") if embedding_mode == "hybrid" else (embedding_mode,)
        )
        for mode in modes:
            _validate_embedding_mode(conn, mode)

    # Each lane retrieves its OWN top-k. Filtering a single global top-k to a
    # lane would starve the smaller lanes at the candidate stage, which is a
    # recall bug rather than a ranking one.
    per_lane_sets: dict[str | None, list[list[QueryObject]]] = {}
    per_lane_channels: dict[str | None, list[dict[str, list[QueryObject]]]] = {}
    for lane in lanes or (None,):
        lane_sets, lane_channels = _generate_candidates(
            search_queries,
            conn,
            model,
            retrieval_top_k=retrieval_top_k,
            embedding_mode=embedding_mode,
            rrf_k=rrf_k,
            hybrid_pool_mode=hybrid_pool_mode,
            content_types=(content_types_for([lane]) if lane else content_types),
            source_patterns=source_patterns,
            query_vectors=query_vectors,
        )
        per_lane_sets[lane] = lane_sets
        per_lane_channels[lane] = lane_channels

    # Diagnostics index candidate_sets by search-query position, so the lanes
    # are folded back together per query rather than concatenated.
    candidate_sets = [
        _merge_candidates([per_lane_sets[lane][index] for lane in per_lane_sets])
        for index in range(len(search_queries))
    ]
    channel_candidate_sets = [
        {
            f"{lane}:{channel}" if lane else channel: candidates
            for lane in per_lane_channels
            for channel, candidates in per_lane_channels[lane][index].items()
        }
        for index in range(len(search_queries))
    ]
    lane_pools = {
        lane: _merge_candidates(sets)
        for lane, sets in per_lane_sets.items()
        if lane is not None
    }

    pooled = _merge_candidates(candidate_sets)
    if lanes:
        ranked, lane_outcomes = _rank_by_lane(question, lane_pools, lanes)
    else:
        ranked, lane_outcomes = rerank(question, pooled, top_k=None), []
    diagnostics = RetrievalDiagnostics(
        search_queries=search_queries,
        candidate_sets=candidate_sets,
        channel_candidate_sets=channel_candidate_sets,
        pooled_candidates=pooled,
        reranked_candidates=ranked,
        lane_outcomes=lane_outcomes,
    )
    # Expansion runs AFTER truncation to rerank_top_k, so it widens only what is
    # actually returned and never influences which chunks those are. Diagnostics
    # keep the un-expanded ranking, which is what ranking analysis wants.
    return _expand_parents(conn, ranked[:rerank_top_k]), diagnostics


def _rank_by_lane(
    question: str,
    lane_pools: dict[str, list[QueryObject]],
    lanes: tuple[str, ...],
) -> tuple[list[RankedResult], list[LaneOutcome]]:
    """Rerank inside each lane, then merge on calibrated scores.

    Cross-encoder logits are not comparable across content types whose text
    length differs by ~4x, so each lane is ranked on its own and only then
    mapped onto a shared scale. Without a fitted calibration the raw score is
    used, which keeps the lane split working before Phase 2 lands.
    """
    by_lane = {lane: lane_pools.get(lane, []) for lane in lanes}

    # Calibrate only when every content type in the merge has a curve; see
    # Calibrator.covers for why partial calibration is worse than none.
    present_types = {
        candidate.content_type
        for candidates in by_lane.values()
        for candidate in candidates
    }
    use_calibration = _calibrator.covers(present_types)
    if present_types and not use_calibration:
        logger.debug(
            "Falling back to raw scores; uncalibrated content types: %s",
            sorted(present_types - set(_calibrator.curves)),
        )

    outcomes: list[LaneOutcome] = []
    merged: list[tuple[float, RankedResult]] = []
    for lane, candidates in by_lane.items():
        if not candidates:
            outcomes.append(LaneOutcome(lane=lane, results=[], confidence=None))
            continue
        lane_ranked = rerank(question, candidates, top_k=None)
        lane_size = len(lane_ranked)
        calibrated = [
            (
                _calibrator.score(
                    result.query_object.content_type, result.rerank_score, lane_size
                )
                if use_calibration
                else result.rerank_score
            )
            for result in lane_ranked
        ]
        confidence = lane_confidence(lane, calibrated)
        outcomes.append(
            LaneOutcome(lane=lane, results=lane_ranked, confidence=confidence)
        )
        demoted = confidence.confidence < CONFIDENCE_FLOOR
        for rank, (result, score) in enumerate(zip(lane_ranked, calibrated)):
            # A demoted lane cannot occupy rank 1, but keeps every candidate.
            key = score - (1.0 if demoted and rank == 0 else 0.0)
            merged.append((key, result))

    merged.sort(key=lambda pair: pair[0], reverse=True)
    ranked = deduplicate_ranked([result for _, result in merged])
    return ranked, outcomes


def _generate_candidates(
    search_queries: list[str],
    conn,
    model,
    *,
    retrieval_top_k: int,
    embedding_mode: str,
    rrf_k: int,
    hybrid_pool_mode: str,
    content_types: tuple[str, ...] | None,
    source_patterns: tuple[str, ...] | None,
    query_vectors: list | None = None,
) -> tuple[list[list[QueryObject]], list[dict[str, list[QueryObject]]]]:
    candidate_sets: list[list[QueryObject]] = []
    channel_candidate_sets: list[dict[str, list[QueryObject]]] = []
    for query_index, search_query in enumerate(search_queries):
        query_vector = (
            query_vectors[query_index]
            if query_vectors is not None
            else encode_query(model, search_query.strip())
        )
        if embedding_mode == "hybrid":
            raw_candidates = _search_by_vector(
                query_vector,
                retrieval_top_k,
                conn,
                embedding_mode="raw",
                content_types=content_types,
                source_patterns=source_patterns,
                validate_embedding=False,
            )
            contextual_candidates = _search_by_vector(
                query_vector,
                retrieval_top_k,
                conn,
                embedding_mode="contextual",
                content_types=content_types,
                source_patterns=source_patterns,
                validate_embedding=False,
            )
            channels = {
                "raw": raw_candidates,
                "contextual": contextual_candidates,
            }
            if hybrid_pool_mode == "union":
                candidates = raw_preserving_union(
                    raw_candidates,
                    contextual_candidates,
                )
            else:
                candidates = reciprocal_rank_fusion(
                    [raw_candidates, contextual_candidates],
                    retrieval_top_k,
                    rrf_k=rrf_k,
                )
        else:
            candidates = _search_by_vector(
                query_vector,
                retrieval_top_k,
                conn,
                embedding_mode=embedding_mode,
                content_types=content_types,
                source_patterns=source_patterns,
                validate_embedding=False,
            )
            channels = {embedding_mode: candidates}
        lexical_candidates = _search_lexical(
            search_query,
            retrieval_top_k,
            conn,
            content_types=content_types,
            source_patterns=source_patterns,
        )
        channels["lexical"] = lexical_candidates
        seen_ids = {candidate.chunk_id for candidate in candidates}
        candidates = [
            *candidates,
            *(
                candidate
                for candidate in lexical_candidates
                if candidate.chunk_id not in seen_ids
            ),
        ]
        candidate_sets.append(candidates)
        channel_candidate_sets.append(channels)
    return candidate_sets, channel_candidate_sets


def retrieve(
    question: str,
    conn,
    *,
    rewrite_mode: str | None = None,
    retrieval_top_k: int = RETRIEVAL_TOP_K,
    rerank_top_k: int = RERANK_TOP_K,
    embedding_mode: str = DEFAULT_EMBEDDING_MODE,
    rrf_k: int = RRF_K,
    hybrid_pool_mode: str = DEFAULT_HYBRID_POOL_MODE,
    lanes: tuple[str, ...] | None = None,
    content_types: tuple[str, ...] | None = None,
    source_patterns: tuple[str, ...] | None = None,
) -> list[RankedResult]:
    ranked, _ = retrieve_with_diagnostics(
        question,
        conn,
        rewrite_mode=rewrite_mode,
        retrieval_top_k=retrieval_top_k,
        rerank_top_k=rerank_top_k,
        embedding_mode=embedding_mode,
        rrf_k=rrf_k,
        hybrid_pool_mode=hybrid_pool_mode,
        lanes=lanes,
        content_types=content_types,
        source_patterns=source_patterns,
    )
    return ranked


def retrieve_evidence(
    question: str,
    conn,
    *,
    groups: tuple[str, ...] = EVIDENCE_GROUPS,
    rewrite_mode: str | None = None,
    retrieval_top_k: int = RETRIEVAL_TOP_K,
    rerank_top_k: int = RERANK_TOP_K,
    embedding_mode: str = DEFAULT_EMBEDDING_MODE,
    rrf_k: int = RRF_K,
    hybrid_pool_mode: str = DEFAULT_HYBRID_POOL_MODE,
    source_patterns: tuple[str, ...] | None = None,
) -> EvidenceRetrievalResult:
    """Retrieve independently ranked evidence groups without score mixing.

    The same question is intentionally run against every requested group. A
    caller can use one or several groups without relying on a question-type
    classifier, and adding a new corpus cannot displace another group's
    results. Query rewriting is still controlled by ``rewrite_mode`` and is
    disabled by the recommended configuration.
    """
    unknown = sorted(set(groups) - set(EVIDENCE_GROUP_CONTENT_TYPES))
    if unknown:
        raise ValueError(
            f"Unknown evidence group(s) {unknown}; expected {list(EVIDENCE_GROUPS)}"
        )
    if not groups:
        return EvidenceRetrievalResult(question=question, groups={})

    retrieved: dict[str, EvidenceGroup] = {}
    source_roles = required_source_roles(question)
    if source_roles:
        modes = (
            ("raw", "contextual") if embedding_mode == "hybrid" else (embedding_mode,)
        )
        for mode in modes:
            _validate_embedding_mode(conn, mode)
        for group in groups:
            selected_types = EVIDENCE_GROUP_CONTENT_TYPES[group]
            if group in {EXCEL, "figure"}:
                results: list[RankedResult] = []
                diagnostics = RetrievalDiagnostics([], [], [], [], [])
            else:
                results, diagnostics = _retrieve_source_role_group(
                    source_roles,
                    conn,
                    content_types=selected_types,
                    retrieval_top_k=retrieval_top_k,
                    rerank_top_k=rerank_top_k,
                    embedding_mode=embedding_mode,
                    rrf_k=rrf_k,
                    hybrid_pool_mode=hybrid_pool_mode,
                )
            retrieved[group] = EvidenceGroup(
                name=group,
                content_types=selected_types,
                results=results,
                diagnostics=diagnostics,
            )
        return EvidenceRetrievalResult(question=question, groups=retrieved)

    # Rewriting is a question-level operation. Reuse it across evidence groups
    # so enabling the optional Claude path never multiplies API calls by the
    # number of independently ranked corpora.
    search_queries = query_rewrite(question, mode=rewrite_mode)
    model = get_embedding_model()
    query_vectors = [
        encode_query(model, search_query.strip())
        for search_query in search_queries
    ]
    modes = ("raw", "contextual") if embedding_mode == "hybrid" else (embedding_mode,)
    for mode in modes:
        _validate_embedding_mode(conn, mode)
    for group in groups:
        selected_types = EVIDENCE_GROUP_CONTENT_TYPES[group]
        results, diagnostics = retrieve_with_diagnostics(
            question,
            conn,
            rewrite_mode=rewrite_mode,
            retrieval_top_k=retrieval_top_k,
            rerank_top_k=rerank_top_k,
            embedding_mode=embedding_mode,
            rrf_k=rrf_k,
            hybrid_pool_mode=hybrid_pool_mode,
            content_types=selected_types,
            source_patterns=source_patterns,
            search_queries=search_queries,
            query_vectors=query_vectors,
            _skip_embedding_validation=True,
        )
        retrieved[group] = EvidenceGroup(
            name=group,
            content_types=selected_types,
            results=results,
            diagnostics=diagnostics,
        )
    return EvidenceRetrievalResult(question=question, groups=retrieved)


def _retrieve_source_role_group(
    source_roles: tuple[SourceRole, ...],
    conn,
    *,
    content_types: tuple[str, ...],
    retrieval_top_k: int,
    rerank_top_k: int,
    embedding_mode: str,
    rrf_k: int,
    hybrid_pool_mode: str,
) -> tuple[list[RankedResult], RetrievalDiagnostics]:
    """Retrieve each required document role independently and preserve coverage."""
    model = get_embedding_model()
    per_role_top_k = max(1, math.ceil(rerank_top_k / len(source_roles)))
    ranked_by_role: list[list[RankedResult]] = []
    candidate_sets: list[list[QueryObject]] = []
    channel_sets: list[dict[str, list[QueryObject]]] = []
    pooled: list[QueryObject] = []

    for role in source_roles:
        vector = encode_query(model, role.query)
        channels: dict[str, list[QueryObject]] = {}
        if embedding_mode == "hybrid":
            raw = _search_by_vector(
                vector,
                retrieval_top_k,
                conn,
                embedding_mode="raw",
                content_types=content_types,
                source_patterns=role.filename_patterns,
                validate_embedding=False,
            )
            contextual = _search_by_vector(
                vector,
                retrieval_top_k,
                conn,
                embedding_mode="contextual",
                content_types=content_types,
                source_patterns=role.filename_patterns,
                validate_embedding=False,
            )
            channels = {"raw": raw, "contextual": contextual}
            candidates = (
                raw_preserving_union(raw, contextual)
                if hybrid_pool_mode == "union"
                else reciprocal_rank_fusion(
                    [raw, contextual], retrieval_top_k, rrf_k=rrf_k
                )
            )
        else:
            candidates = _search_by_vector(
                vector,
                retrieval_top_k,
                conn,
                embedding_mode=embedding_mode,
                content_types=content_types,
                source_patterns=role.filename_patterns,
                validate_embedding=False,
            )
            channels = {embedding_mode: candidates}
        candidate_sets.append(candidates)
        channel_sets.append(channels)
        pooled.extend(candidates)
        ranked_by_role.append(rerank(role.query, candidates, top_k=per_role_top_k))

    results = _expand_parents(
        conn,
        deduplicate_ranked(_interleave_ranked_groups(ranked_by_role))[:rerank_top_k],
    )
    diagnostics = RetrievalDiagnostics(
        search_queries=[role.query for role in source_roles],
        candidate_sets=candidate_sets,
        channel_candidate_sets=channel_sets,
        pooled_candidates=_merge_candidates([pooled]),
        reranked_candidates=results,
    )
    return results, diagnostics


def _interleave_ranked_groups(
    groups: list[list[RankedResult]],
) -> list[RankedResult]:
    """Preserve required-role coverage under a small downstream token budget."""
    longest = max((len(group) for group in groups), default=0)
    return [
        group[rank] for rank in range(longest) for group in groups if rank < len(group)
    ]


def retrieve_configured(
    question: str,
    conn,
    *,
    output_mode: str | None = None,
    groups: tuple[str, ...] = EVIDENCE_GROUPS,
    **kwargs,
) -> EvidenceRetrievalResult | list[RankedResult]:
    """Public application dispatcher honoring ``retrieval.output_mode``.

    ``retrieve`` remains the stable low-level flat-list compatibility API.
    New application integrations should call this dispatcher or call
    ``retrieve_evidence`` explicitly when they require the grouped contract.
    """
    selected_mode = output_mode or str(
        _retrieval_config.get("output_mode", "legacy_flat")
    )
    if selected_mode == "grouped":
        return retrieve_evidence(question, conn, groups=groups, **kwargs)
    if selected_mode == "legacy_flat":
        return retrieve(question, conn, **kwargs)
    raise ValueError(
        f"Unknown output mode {selected_mode!r}; expected 'grouped' or 'legacy_flat'"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Query the SDG&E WMP knowledge base.")
    parser.add_argument("question", type=str, help="Question to ask.")
    parser.add_argument(
        "--embedding-mode",
        choices=EMBEDDING_MODES,
        default=DEFAULT_EMBEDDING_MODE,
        help="Dense retrieval strategy: raw, contextual, or hybrid.",
    )
    parser.add_argument(
        "--hybrid-pool-mode",
        choices=HYBRID_POOL_MODES,
        default=DEFAULT_HYBRID_POOL_MODE,
        help=(
            "How hybrid mode builds the reranker pool: fixed-size RRF or a "
            "raw-preserving deduplicated union "
            f"(default: {DEFAULT_HYBRID_POOL_MODE})."
        ),
    )
    parser.add_argument(
        "--rrf-k",
        type=int,
        default=RRF_K,
        help="Reciprocal-rank-fusion constant used by hybrid mode (default: 60).",
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        choices=EVIDENCE_GROUPS,
        default=list(EVIDENCE_GROUPS),
        help="Evidence groups to retrieve when grouped output is enabled.",
    )
    parser.add_argument(
        "--legacy-flat",
        action="store_true",
        help="Use the backward-compatible cross-content ranked list.",
    )
    args = parser.parse_args()
    if args.rrf_k < 1:
        parser.error("--rrf-k must be positive")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    conn = connect_db()
    try:
        output_mode = str(_retrieval_config.get("output_mode", "legacy_flat"))
        use_grouped = output_mode == "grouped" and not args.legacy_flat
        if use_grouped:
            evidence = retrieve_evidence(
                args.question,
                conn,
                groups=tuple(args.groups),
                embedding_mode=args.embedding_mode,
                rrf_k=args.rrf_k,
                hybrid_pool_mode=args.hybrid_pool_mode,
            )
            result_groups = {
                name: group.results for name, group in evidence.groups.items()
            }
        else:
            result_groups = {
                "legacy_flat": retrieve(
                    args.question,
                    conn,
                    embedding_mode=args.embedding_mode,
                    rrf_k=args.rrf_k,
                    hybrid_pool_mode=args.hybrid_pool_mode,
                )
            }
    finally:
        conn.close()

    if not any(result_groups.values()):
        print("No results found.")
        return

    storage = get_object_storage()
    for group_name, results in result_groups.items():
        print(f"\n=== {group_name.upper()} EVIDENCE ===")
        if not results:
            print("No results.")
            continue
        for rank, result in enumerate(results, start=1):
            qo = result.query_object
            retrieval_detail = (
                f"retrieval_score={qo.retrieval_score:.6f}"
                if qo.retrieval_score is not None
                else f"distance={qo.distance:.4f}"
            )
            print(
                f"\n[{rank}] [{qo.content_type}] "
                f"rerank_score={result.rerank_score:.4f} {retrieval_detail}"
            )
            print(f"    {qo.source_pdf} (p.{qo.page_start}-{qo.page_end})")
            print(f"    {qo.breadcrumb}")
            if qo.object_key:
                print(f"    object: {storage.uri(qo.object_key)}")
            print(f"    {qo.content[:300]}")


if __name__ == "__main__":
    main()

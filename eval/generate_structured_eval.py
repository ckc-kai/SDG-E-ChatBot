"""Generate a table/figure retrieval evaluation dataset from ingested chunks.

Samples table/figure chunks straight from the unified ``chunks`` table (so every gold
reference is guaranteed to resolve), then asks Claude Haiku -- the same model
and API key the query-rewrite path already uses -- to draft one natural
question per chunk that is answerable from that chunk alone. Evidence is
recorded by the stable key (source_pdf, content_type, page_start, chunk_index),
matching `eval.run_structured_eval`'s resolution rule.

Sampling filters favor informative elements:
  - tables: at least 3 rows and a minimum token count (skips layout debris)
  - figures: must have caption text (pure-description figures are usually
    generic chart summaries that make unanswerable questions)
Chunks are interleaved across documents so one PDF cannot dominate.

Usage:
    python -m eval.generate_structured_eval --tables 30 --figures 15 \
        --out eval/pdf/evaluation_structured.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
from pathlib import Path

from retrieval.failure_log import get_failure_logger
from retrieval.query.pdf.query import _strip_json_fence
from retrieval.utils import connect_db, get_anthropic_client, load_config

logger = logging.getLogger(__name__)
log_failure = get_failure_logger("generate_structured_eval")

GENERATION_MODEL = load_config().get("query_rewrite", {}).get(
    "model", "claude-haiku-4-5-20251001"
)

MIN_TABLE_ROWS_FOR_EVAL = 3
MIN_TABLE_TOKENS_FOR_EVAL = 40
MIN_FIGURE_CAPTION_CHARS = 15

_SYSTEM_PROMPT = (
    "You write evaluation questions for a retrieval system over SDG&E Wildfire "
    "Mitigation Plan regulatory filings. Given ONE table or figure from a filing, "
    "write ONE natural question that a regulator or analyst would ask, whose answer "
    "is contained in that table/figure content alone. Prefer questions about specific "
    "values, targets, comparisons, or trends actually present in the content. "
    "Do not mention 'the table', 'the figure', or 'the document' in the question; "
    "phrase it as a standalone information need. "
    'Respond with ONLY a JSON object: {"question": str, "expected_answer": str, '
    '"difficulty": "simple" | "medium" | "hard"}. '
    "expected_answer must quote the relevant values verbatim from the content. "
    'If the content is too uninformative for a fair question, respond {"skip": true}.'
)


def _fetch_candidates(conn, content_type: str) -> list[dict]:
    if content_type == "table":
        condition = (
            "c.content_type = 'table' "
            f"AND (c.structured_data->>'num_rows')::int >= {MIN_TABLE_ROWS_FOR_EVAL} "
            f"AND c.token_count >= {MIN_TABLE_TOKENS_FOR_EVAL}"
        )
    else:
        condition = (
            "c.content_type = 'figure' "
            f"AND length(coalesce(c.caption, '')) >= {MIN_FIGURE_CAPTION_CHARS}"
        )
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT c.id, d.filename, c.content_type, c.page_start, c.chunk_index,
                   c.breadcrumb, c.content, c.caption
            FROM chunks c JOIN documents d ON d.id = c.document_id
            WHERE {condition}
            ORDER BY c.id
            """
        )
        rows = cur.fetchall()
    return [
        {
            "chunk_id": row[0],
            "source_pdf": row[1],
            "content_type": row[2],
            "page_start": row[3],
            "chunk_index": row[4],
            "breadcrumb": row[5],
            "content": row[6],
            "caption": row[7],
        }
        for row in rows
    ]


def _interleave_by_document(candidates: list[dict], rng: random.Random) -> list[dict]:
    """Round-robin across documents so a single PDF cannot dominate the sample."""
    by_document: dict[str, list[dict]] = {}
    for candidate in candidates:
        by_document.setdefault(candidate["source_pdf"], []).append(candidate)
    for bucket in by_document.values():
        rng.shuffle(bucket)
    buckets = sorted(by_document.values(), key=len, reverse=True)
    interleaved: list[dict] = []
    index = 0
    while any(buckets):
        for bucket in buckets:
            if index < len(bucket):
                interleaved.append(bucket[index])
        index += 1
        if all(index >= len(bucket) for bucket in buckets):
            break
    return interleaved


def _draft_question(candidate: dict) -> dict | None:
    user_content = (
        f"Source filing: {candidate['source_pdf']}\n"
        f"Section: {candidate['breadcrumb']}\n"
        f"Element type: {candidate['content_type']}\n\n"
        f"Content:\n{candidate['content'][:6000]}"
    )
    response = get_anthropic_client().messages.create(
        model=GENERATION_MODEL,
        max_tokens=1024,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_content}],
    )
    payload = json.loads(_strip_json_fence(response.content[0].text))
    if not isinstance(payload, dict) or payload.get("skip"):
        return None
    question = " ".join(str(payload.get("question", "")).split())
    expected_answer = str(payload.get("expected_answer", "")).strip()
    difficulty = payload.get("difficulty", "medium")
    if not question or not expected_answer:
        return None
    if difficulty not in ("simple", "medium", "hard"):
        difficulty = "medium"
    return {
        "question": question,
        "expected_answer": expected_answer,
        "difficulty": difficulty,
    }


def _question_signature(question: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", "", question.lower()).strip()


def generate(
    conn, tables: int, figures: int, seed: int, out_path: Path
) -> tuple[int, int]:
    rng = random.Random(seed)
    rows: list[dict] = []
    seen_signatures: set[str] = set()
    counts = {"table": 0, "figure": 0}
    targets = {"table": tables, "figure": figures}

    for content_type in ("table", "figure"):
        candidates = _interleave_by_document(_fetch_candidates(conn, content_type), rng)
        logger.info("%d %s candidates available", len(candidates), content_type)
        for candidate in candidates:
            if counts[content_type] >= targets[content_type]:
                break
            try:
                draft = _draft_question(candidate)
            except Exception as exc:
                log_failure("draft_question", str(candidate["chunk_id"]), exc)
                logger.warning(
                    "Question generation failed for chunk %d: %s",
                    candidate["chunk_id"],
                    exc,
                )
                continue
            if draft is None:
                continue
            signature = _question_signature(draft["question"])
            if not signature or signature in seen_signatures:
                continue
            seen_signatures.add(signature)
            counts[content_type] += 1
            rows.append(
                {
                    "id": f"structured_eval_{len(rows) + 1:04d}",
                    "difficulty": draft["difficulty"],
                    "question_type": content_type,
                    "question": draft["question"],
                    "expected_answer": draft["expected_answer"],
                    "source_pdf": candidate["source_pdf"],
                    "evidence": [
                        {
                            "chunk_id": candidate["chunk_id"],
                            "source_pdf": candidate["source_pdf"],
                            "content_type": content_type,
                            "page_start_db": candidate["page_start"],
                            "chunk_index": candidate["chunk_index"],
                            "breadcrumb": candidate["breadcrumb"],
                            "content_excerpt": candidate["content"][:400],
                        }
                    ],
                    "generator": GENERATION_MODEL,
                }
            )
            logger.info(
                "[%s %d/%d] %s",
                content_type,
                counts[content_type],
                targets[content_type],
                draft["question"][:100],
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return counts["table"], counts["figure"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a structured (table/figure) retrieval eval dataset."
    )
    parser.add_argument("--tables", type=int, default=30, help="Table questions to generate.")
    parser.add_argument("--figures", type=int, default=15, help="Figure questions to generate.")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed.")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("eval/pdf/evaluation_structured.jsonl"),
        help="Output JSONL path.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    conn = connect_db()
    try:
        table_count, figure_count = generate(
            conn, args.tables, args.figures, args.seed, args.out
        )
    finally:
        conn.close()
    print(f"Wrote {table_count} table + {figure_count} figure questions to {args.out}")


if __name__ == "__main__":
    main()

"""Run one real Task 2 retrieval through the Task 3 mock pipeline.

Prerequisites:
  - config/config.yaml points to a populated PostgreSQL/pgvector database
  - the Change Order PDF has been ingested
  - embedding and reranker models are available locally or can be downloaded

This script does not call Anthropic, Bedrock, or any other answer-generation API.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from generation import (
    AnswerRequest,
    AnswerService,
    RecordingScriptedMockProvider,
    adapt_ranked_results,
)
from retrieval.query import retrieve
from retrieval.utils import connect_db


QUESTION = (
    "What is the target increase for SDG&E's Strategic Pole Replacement "
    "program in 2024, and why was it changed?"
)
EXPECTED_SOURCE = "2023-12-19_SDGE_2023_Change Order Report_R1.pdf"
EXPECTED_TEXT = "target increase from 200 to 267"


def main() -> None:
    conn = connect_db()
    try:
        ranked_results = retrieve(
            QUESTION,
            conn,
            rewrite_mode="off",
            retrieval_top_k=10,
            rerank_top_k=5,
            embedding_mode="raw",
        )
    finally:
        conn.close()

    chunks = adapt_ranked_results(ranked_results)
    if len(chunks) != 5:
        raise RuntimeError(f"Expected 5 reranked chunks, received {len(chunks)}")

    top = chunks[0]
    if top.metadata.source_file != EXPECTED_SOURCE:
        raise RuntimeError(f"Unexpected top source: {top.metadata.source_file}")
    if EXPECTED_TEXT not in top.content:
        raise RuntimeError("Top chunk does not contain the expected benchmark evidence")

    provider = RecordingScriptedMockProvider(
        {
            "answer": (
                "SDG&E increased the 2024 Strategic Pole Replacement target "
                "from 200 to 267 because it changed its approach and identified "
                "additional HFTD poles requiring remediation."
            ),
            "cited_chunk_ids": [top.chunk_id],
            "insufficient_context": False,
        }
    )
    request = AnswerRequest(
        request_id="smoke_task2_task3_001",
        question=QUESTION,
        chunks=chunks,
    )
    response = AnswerService(provider).answer(request)
    public_response = response.to_public_dict()

    if public_response.get("cited_chunk_ids") != [top.chunk_id]:
        raise RuntimeError("Task 3 did not preserve the selected top chunk ID")
    if public_response["citations"][0]["source_pdf"] != EXPECTED_SOURCE:
        raise RuntimeError("Task 3 citation hydration returned the wrong source")
    if provider.last_prompt is None or EXPECTED_TEXT not in provider.last_prompt:
        raise RuntimeError("Grounded prompt does not contain the expected evidence")

    print(
        json.dumps(
            {
                "retrieved_count": len(ranked_results),
                "top_chunk_id_current_db": top.chunk_id,
                "top_stable_key": {
                    "source_pdf": top.metadata.source_file,
                    "page_start": top.metadata.page_start,
                    "chunk_index": top.metadata.chunk_index,
                },
                "top_distance": top.metadata.distance,
                "top_rerank_score": top.metadata.rerank_score,
                "prompt_chars": len(provider.last_prompt),
                "public_response": public_response,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

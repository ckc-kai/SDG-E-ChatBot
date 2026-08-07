"""Run one real Task 2 retrieval through Task 3 with mock or local Ollama.

Prerequisites:
  - config/config.yaml points to a populated PostgreSQL/pgvector database
  - the Change Order PDF has been ingested
  - embedding and reranker models are available locally or can be downloaded

The default mock mode calls no model. Ollama mode calls only the configured
Ollama HTTP endpoint and never calls Anthropic, Bedrock, or AWS.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from generation import (
    AnswerRequest,
    AnswerService,
    ErrorResponse,
    OllamaProvider,
    RecordingScriptedMockProvider,
    adapt_ranked_results,
)
from generation.prompting import build_prompt
from retrieval.query import retrieve
from retrieval.utils import connect_db


QUESTION = (
    "What is the target increase for SDG&E's Strategic Pole Replacement "
    "program in 2024, and why was it changed?"
)
EXPECTED_SOURCE = "2023-12-19_SDGE_2023_Change Order Report_R1.pdf"
EXPECTED_TEXT = "target increase from 200 to 267"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test Task 2 -> Task 3.")
    parser.add_argument("--provider", choices=("mock", "ollama"), default="mock")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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

    if args.provider == "ollama":
        provider = OllamaProvider.from_env()
    else:
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
    prompt = build_prompt(request)
    response = AnswerService(provider).answer(request)
    if isinstance(response, ErrorResponse):
        raise RuntimeError(f"Task 3 generation failed with {args.provider}")
    public_response = response.to_public_dict()

    if args.provider == "mock" and public_response.get("cited_chunk_ids") != [
        top.chunk_id
    ]:
        raise RuntimeError("Task 3 did not preserve the selected top chunk ID")
    if not response.insufficient_context and not public_response["citations"]:
        raise RuntimeError("Answerable Task 3 response did not include a citation")
    if (
        args.provider == "mock"
        and public_response["citations"]
        and public_response["citations"][0]["source_pdf"] != EXPECTED_SOURCE
    ):
        raise RuntimeError("Task 3 citation hydration returned the wrong source")
    if args.provider == "mock" and (
        provider.last_prompt is None or EXPECTED_TEXT not in provider.last_prompt
    ):
        raise RuntimeError("Grounded prompt does not contain the expected evidence")

    report = {
        "request_id": request.request_id,
        "question": request.question,
        "retrieved_count": len(ranked_results),
        "retrieved_chunks": [asdict(chunk) for chunk in chunks],
        "top_chunk_id_current_db": top.chunk_id,
        "top_stable_key": {
            "source_pdf": top.metadata.source_file,
            "page_start": top.metadata.page_start,
            "chunk_index": top.metadata.chunk_index,
        },
        "top_distance": top.metadata.distance,
        "top_rerank_score": top.metadata.rerank_score,
        "provider": provider.model_id,
        "prompt_chars": len(prompt),
        "prompt": prompt,
        "raw_model_output": getattr(provider, "last_raw_text", None),
        "public_response": public_response,
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(f"Wrote {args.output}")
    else:
        print(rendered)


if __name__ == "__main__":
    main()

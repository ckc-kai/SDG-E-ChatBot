from __future__ import annotations

import unittest
import json
import tempfile
from pathlib import Path

from generation.providers.bedrock import BedrockProvider
from scripts.evaluate_task3_gold import (
    build_dry_run_report,
    load_suite,
    run_bedrock_report,
    select_rows,
)
from tests.test_bedrock_provider import FakeBedrockClient


def benchmark_row(row_id: str) -> dict:
    return {
        "id": row_id,
        "question": "What is the target?",
        "expected_answer": "100%",
        "evidence": [
            {
                "chunk_id": 7,
                "source_pdf": "WMP.pdf",
                "page_start_db": 4,
                "page_end_db_exclusive": 5,
                "breadcrumb": "Targets",
                "content_excerpt": "The target is 100%.",
            }
        ],
    }


class GoldRunnerTests(unittest.TestCase):
    def test_dry_run_builds_prompt_without_calling_a_model(self) -> None:
        report = build_dry_run_report([benchmark_row("eval_1")])
        self.assertFalse(report["model_called"])
        self.assertTrue(report["uses_gold_evidence"])
        self.assertEqual(report["records"][0]["chunk_ids"], ["7"])
        self.assertIn("The target is 100%.", report["records"][0]["prompt"])

    def test_select_rows_filters_ids_and_applies_limit(self) -> None:
        rows = [benchmark_row("a"), benchmark_row("b"), benchmark_row("c")]
        selected = select_rows(rows, request_ids={"b", "c"}, limit=1)
        self.assertEqual([row["id"] for row in selected], ["b"])

    def test_select_rows_rejects_unknown_id(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown evaluation IDs"):
            select_rows([benchmark_row("a")], request_ids={"missing"}, limit=5)

    def test_load_suite_returns_name_and_unique_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "suite.json"
            path.write_text(
                json.dumps(
                    {
                        "name": "smoke",
                        "cases": [{"id": "a"}, {"id": "b"}],
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(load_suite(path), ("smoke", {"a", "b"}))

    def test_fake_bedrock_response_runs_full_task3_gold_path(self) -> None:
        client = FakeBedrockClient(
            response={
                "output": {
                    "message": {
                        "content": [
                            {
                                "text": (
                                    '{"answer":"100%","cited_chunk_ids":["7"],'
                                    '"insufficient_context":false}'
                                )
                            }
                        ]
                    }
                },
                "usage": {"inputTokens": 90, "outputTokens": 15, "totalTokens": 105},
                "metrics": {"latencyMs": 200},
            }
        )
        report = run_bedrock_report(
            [benchmark_row("eval_1")],
            BedrockProvider(client, "amazon.nova-lite-v1:0"),
        )

        self.assertTrue(report["model_called"])
        self.assertEqual(report["errors"], 0)
        self.assertEqual(report["score_summary"]["citation_precision"], 1.0)
        response = report["records"][0]["response"]
        self.assertEqual(response["cited_chunk_ids"], ["7"])
        self.assertEqual(response["citations"][0]["source_pdf"], "WMP.pdf")
        self.assertEqual(report["records"][0]["usage"]["input_tokens"], 90)


if __name__ == "__main__":
    unittest.main()

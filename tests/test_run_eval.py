import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

from eval.run_eval import (
    GoldNotFoundError,
    QueryScore,
    _first_gold_rank_in_candidates,
    gold_signatures,
    load_eval,
    main,
    score_row,
)
from retrieval.query import QueryObject, RankedResult, RetrievalDiagnostics


def _candidate(chunk_id: int, content: str) -> QueryObject:
    return QueryObject(
        chunk_id=chunk_id,
        source_pdf="example.pdf",
        sub_document=None,
        breadcrumb="Example",
        section_number=None,
        page_start=1,
        page_end=2,
        chunk_index=0,
        content_type="text",
        content=content,
        token_count=3,
        distance=0.1,
    )


class EvaluationDiagnosticsTests(unittest.TestCase):
    def test_gold_rank_matches_id(self) -> None:
        candidates = [_candidate(10, "not gold"), _candidate(20, "gold")]

        rank = _first_gold_rank_in_candidates(candidates, {20}, {"gold"})

        self.assertEqual(rank, 2)

    def test_gold_rank_credits_identical_content_twin(self) -> None:
        ranked = [
            RankedResult(_candidate(10, "not gold"), 0.9),
            RankedResult(_candidate(99, "  GOLD\ncontent "), 0.8),
        ]

        rank = _first_gold_rank_in_candidates(
            ranked,
            gold_ids={20},
            gold_contents={"gold content"},
        )

        self.assertEqual(rank, 2)

    def test_gold_rank_is_none_when_candidate_pool_missed(self) -> None:
        candidates = [_candidate(10, "not gold")]

        rank = _first_gold_rank_in_candidates(candidates, {20}, {"gold"})

        self.assertIsNone(rank)

    @patch("eval.run_eval.get_last_rewrite_trace")
    @patch("eval.run_eval.get_rewrite_call_count", side_effect=[0, 1])
    @patch("eval.run_eval.retrieve_with_diagnostics")
    @patch("eval.run_eval.gold_signatures")
    def test_score_row_distinguishes_candidate_and_reranker_rank(
        self,
        gold_signatures: Mock,
        retrieve: Mock,
        _rewrite_count: Mock,
        rewrite_trace: Mock,
    ) -> None:
        non_gold = _candidate(10, "not gold")
        gold = _candidate(20, "gold")
        full_ranking = [RankedResult(non_gold, 0.9), RankedResult(gold, 0.8)]
        diagnostics = RetrievalDiagnostics(
            search_queries=["original", "focused"],
            candidate_sets=[[non_gold, gold], [gold]],
            pooled_candidates=[non_gold, gold],
            reranked_candidates=full_ranking,
        )
        gold_signatures.return_value = ({20}, {"gold"})
        retrieve.return_value = (full_ranking, diagnostics)
        rewrite_trace.return_value = {
            "source": "api",
            "reason": "independent_interrogative_clauses",
            "sub_questions": ["original", "focused"],
        }
        row = {
            "id": "row-1",
            "difficulty": "medium",
            "question_type": "risk_assessment",
            "question": "original",
            "evidence": [{}],
        }

        score = score_row(
            row,
            conn=object(),
            k=1,
            rewrite_mode="auto",
            retrieval_top_k=20,
            rerank_top_k=5,
        )

        self.assertEqual(score.gold_best_vector_rank, 1)
        self.assertEqual(score.gold_full_rerank_rank, 2)
        self.assertIsNone(score.first_gold_rank)
        self.assertEqual(score.api_rewrite_calls, 1)
        self.assertEqual(score.vector_candidate_ids, [[10, 20], [20]])
        retrieve.assert_called_once_with(
            "original",
            unittest.mock.ANY,
            rewrite_mode="auto",
            retrieval_top_k=20,
            rerank_top_k=5,
        )


class EvaluationHarnessTests(unittest.TestCase):
    def test_load_eval_reads_jsonl_and_rejects_empty_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            populated = Path(directory) / "rows.jsonl"
            populated.write_text('{"id": "one"}\n\n{"id": "two"}\n')
            empty = Path(directory) / "empty.jsonl"
            empty.write_text("")

            self.assertEqual([row["id"] for row in load_eval(str(populated))], ["one", "two"])
            with self.assertRaises(ValueError):
                load_eval(str(empty))

    def test_gold_signatures_resolve_content_and_report_stale_reference(self) -> None:
        conn = MagicMock()
        cursor = conn.cursor.return_value.__enter__.return_value
        evidence = [{"source_pdf": "doc.pdf", "page_start_db": 2, "chunk_index": 0}]
        cursor.fetchone.return_value = (20, " Gold\ncontent ")

        ids, contents = gold_signatures(conn, evidence)

        self.assertEqual(ids, {20})
        self.assertEqual(contents, {"gold content"})

        cursor.fetchone.return_value = None
        with self.assertRaises(GoldNotFoundError):
            gold_signatures(conn, evidence)

    @patch("builtins.print")
    @patch("eval.run_eval.score_row")
    @patch("eval.run_eval.connect_db")
    @patch("eval.run_eval.load_eval", return_value=[{"id": "row-1"}])
    def test_main_writes_diagnostic_result_payload(
        self,
        _load_rows: Mock,
        connect_db: Mock,
        score_row_mock: Mock,
        _print: Mock,
    ) -> None:
        score_row_mock.return_value = QueryScore(
            id="row-1",
            difficulty="simple",
            question_type="risk_assessment",
            num_gold=1,
            hit_at_1=0.0,
            recall_at_k=0.0,
            mrr=0.0,
            ndcg_at_k=0.0,
            first_gold_rank=None,
            retrieved_ids=[10],
            api_rewrite_calls=0,
            rewrite_source="disabled",
            rewrite_reason="mode_off",
            search_queries=["question"],
            vector_candidate_ids=[[10]],
            pooled_candidate_ids=[10],
            gold_best_vector_rank=2,
            gold_full_rerank_rank=6,
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "nested" / "result.json"
            argv = [
                "run_eval",
                "--eval",
                "unused.jsonl",
                "--out",
                str(output),
                "--rewrite-mode",
                "off",
                "--retrieval-top-k",
                "20",
                "--rerank-top-k",
                "5",
                "--misses",
            ]

            with patch.object(sys, "argv", argv):
                main()

            payload = json.loads(output.read_text())

        self.assertEqual(payload["retrieval_diagnostics"]["rewrite_mode"], "off")
        self.assertEqual(payload["retrieval_diagnostics"]["retrieval_top_k"], 20)
        self.assertEqual(payload["retrieval_diagnostics"]["reranker_top_k_misses"], 1)
        self.assertEqual(payload["per_query"][0]["gold_full_rerank_rank"], 6)
        connect_db.return_value.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()

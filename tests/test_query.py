import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from retrieval import query


def _candidate(
    chunk_id: int,
    *,
    breadcrumb: str = "2. Changes > Strategic Pole Replacement",
    section_number: str | None = "2.1",
    content: str = "The target increased from 200 to 267.",
    distance: float = 0.1,
) -> query.QueryObject:
    return query.QueryObject(
        chunk_id=chunk_id,
        source_pdf="example.pdf",
        sub_document=None,
        breadcrumb=breadcrumb,
        section_number=section_number,
        page_start=1,
        page_end=2,
        chunk_index=0,
        content_type="text",
        content=content,
        token_count=10,
        distance=distance,
    )


class RewriteRoutingTests(unittest.TestCase):
    def test_single_intent_questions_do_not_need_decomposition(self) -> None:
        questions = [
            "What does EVM stand for?",
            "What preparation exercises and training did SDG&E conduct to prepare for PSPS and support individuals with access and functional needs?",
            "What are the six components of SDG&E's risk analysis framework used to identify, evaluate, prioritize, and respond to risk events?",
            "On what date was the slip load test completed for sample 2.4.1?",
        ]

        for question_text in questions:
            with self.subTest(question=question_text):
                self.assertFalse(query._needs_decomposition(question_text))

    def test_independent_interrogative_clauses_need_decomposition(self) -> None:
        questions = [
            "What is the target increase, and what is the reason for the change?",
            "Who participates in the working group and what topics does it address?",
            "What changed; why did it change?",
            "What changed? Why did it change?",
            "Is the mitigation required, and does it apply in the HFTD?",
        ]

        for question_text in questions:
            with self.subTest(question=question_text):
                self.assertTrue(query._needs_decomposition(question_text))

    @patch("retrieval.query.get_anthropic_client")
    def test_auto_mode_bypasses_api_for_single_intent_question(self, client_factory: Mock) -> None:
        question_text = "What does EVM stand for?"

        rewritten = query.query_rewrite(question_text, mode="auto")

        self.assertEqual(rewritten, [question_text])
        client_factory.assert_not_called()
        self.assertEqual(query.get_last_rewrite_trace()["source"], "simple")

    @patch("retrieval.query.get_anthropic_client")
    def test_off_mode_is_an_internal_api_free_control(self, client_factory: Mock) -> None:
        question_text = "What changed, and why did it change?"

        rewritten = query.query_rewrite(question_text, mode="off")

        self.assertEqual(rewritten, [question_text])
        client_factory.assert_not_called()
        self.assertEqual(query.get_last_rewrite_trace()["source"], "disabled")

    @patch("retrieval.query.get_anthropic_client")
    def test_decomposition_keeps_original_and_caps_deduplicated_subquestions(
        self, client_factory: Mock
    ) -> None:
        question_text = "What changed, and what caused the change?"
        api_subquestions = [
            "What changed?",
            "What caused the change?",
            "What changed?",
            "What else happened?",
        ]
        response = SimpleNamespace(
            content=[SimpleNamespace(text=json.dumps(api_subquestions))]
        )
        client_factory.return_value.messages.create.return_value = response

        rewritten = query.query_rewrite(question_text, mode="auto")

        self.assertEqual(
            rewritten,
            [question_text, "What changed?", "What caused the change?"],
        )
        trace = query.get_last_rewrite_trace()
        self.assertEqual(trace["source"], "api")
        self.assertEqual(trace["reason"], "independent_interrogative_clauses")
        self.assertEqual(trace["api_sub_questions"], api_subquestions)

    def test_unknown_internal_mode_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            query.query_rewrite("What changed?", mode="sometimes")

    @patch("retrieval.query.log_failure")
    @patch("retrieval.query.get_anthropic_client")
    def test_api_failure_falls_back_to_original_question(
        self, client_factory: Mock, failure_log: Mock
    ) -> None:
        client_factory.return_value.messages.create.side_effect = RuntimeError("offline")
        question_text = "What changed, and why did it change?"

        rewritten = query.query_rewrite(question_text, mode="auto")

        self.assertEqual(rewritten, [question_text])
        self.assertEqual(query.get_last_rewrite_trace()["source"], "fallback")
        failure_log.assert_called_once()


class RetrievalTests(unittest.TestCase):
    def test_empty_candidate_list_does_not_load_reranker(self) -> None:
        with patch("retrieval.query.get_reranker_model") as model_factory:
            self.assertEqual(query.rerank("question", [], top_k=5), [])
            model_factory.assert_not_called()

    @patch("retrieval.query.get_reranker_model")
    def test_reranker_receives_section_context(self, model_factory: Mock) -> None:
        candidate = _candidate(1)
        model_factory.return_value.predict.return_value = [0.9]

        query.rerank("What changed?", [candidate], top_k=1)

        pairs = model_factory.return_value.predict.call_args.args[0]
        candidate_text = pairs[0][1]
        self.assertIn("Section: 2. Changes > Strategic Pole Replacement", candidate_text)
        self.assertIn("Section number: 2.1", candidate_text)
        self.assertIn(candidate.content, candidate_text)

    @patch("retrieval.query.rerank")
    @patch("retrieval.query.search")
    @patch("retrieval.query.get_embedding_model")
    @patch("retrieval.query.query_rewrite")
    def test_retrieve_with_diagnostics_preserves_full_candidate_flow(
        self,
        rewrite: Mock,
        embedding_model: Mock,
        search: Mock,
        rerank: Mock,
    ) -> None:
        first = _candidate(1, distance=0.1)
        second = _candidate(2, distance=0.2)
        rewrite.return_value = ["original question"]
        search.return_value = [first, second]
        rerank.return_value = [
            query.RankedResult(second, 0.9),
            query.RankedResult(first, 0.8),
        ]

        ranked, diagnostics = query.retrieve_with_diagnostics(
            "original question",
            conn=object(),
            rewrite_mode="auto",
            retrieval_top_k=20,
            rerank_top_k=1,
        )

        self.assertEqual([item.query_object.chunk_id for item in ranked], [2])
        self.assertEqual(diagnostics.search_queries, ["original question"])
        self.assertEqual(
            [[item.chunk_id for item in items] for items in diagnostics.candidate_sets],
            [[1, 2]],
        )
        self.assertEqual([item.chunk_id for item in diagnostics.pooled_candidates], [1, 2])
        self.assertEqual(
            [item.query_object.chunk_id for item in diagnostics.reranked_candidates],
            [2, 1],
        )
        self.assertEqual(search.call_args.args[1], 20)

    @patch("retrieval.query.retrieve_with_diagnostics")
    def test_retrieve_returns_ranked_results_without_diagnostics(self, retrieve_debug: Mock) -> None:
        expected = [query.RankedResult(_candidate(1), 0.8)]
        retrieve_debug.return_value = (
            expected,
            query.RetrievalDiagnostics([], [], [], []),
        )

        actual = query.retrieve("question", conn=object(), rewrite_mode="off")

        self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()

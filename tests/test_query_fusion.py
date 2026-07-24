import unittest
from unittest.mock import Mock, patch

from retrieval.query import (
    QueryObject,
    raw_preserving_union,
    retrieve_with_diagnostics,
)


def candidate(chunk_id: int, distance: float) -> QueryObject:
    return QueryObject(
        chunk_id=chunk_id,
        source_pdf="test.pdf",
        sub_document=None,
        breadcrumb="Test",
        section_number=None,
        page_start=1,
        page_end=2,
        chunk_index=0,
        content_type="text",
        content=f"chunk {chunk_id}",
        token_count=2,
        distance=distance,
    )


class RawPreservingUnionTests(unittest.TestCase):
    def test_keeps_all_raw_candidates_in_their_original_order(self) -> None:
        raw = [candidate(1, 0.1), candidate(2, 0.2), candidate(3, 0.3)]
        contextual = [candidate(4, 0.05), candidate(2, 0.06)]

        combined = raw_preserving_union(raw, contextual)

        self.assertEqual([item.chunk_id for item in combined], [1, 2, 3, 4])
        self.assertIs(combined[1], raw[1])

    def test_appends_contextual_only_candidates_in_contextual_rank_order(self) -> None:
        raw = [candidate(1, 0.1), candidate(2, 0.2)]
        contextual = [
            candidate(2, 0.01),
            candidate(5, 0.02),
            candidate(4, 0.03),
        ]

        combined = raw_preserving_union(raw, contextual)

        self.assertEqual([item.chunk_id for item in combined], [1, 2, 5, 4])

    @patch("retrieval.query.rerank")
    @patch("retrieval.query._search_by_vector")
    @patch("retrieval.query.get_embedding_model")
    def test_union_mode_sends_the_expanded_pool_to_the_reranker(
        self,
        get_embedding_model: Mock,
        search_by_vector: Mock,
        rerank: Mock,
    ) -> None:
        raw = [candidate(1, 0.1), candidate(2, 0.2)]
        contextual = [candidate(2, 0.01), candidate(3, 0.02)]
        get_embedding_model.return_value.encode.return_value = [0.0]
        search_by_vector.side_effect = [raw, contextual]
        rerank.return_value = []

        _, diagnostics = retrieve_with_diagnostics(
            "test question",
            conn=object(),
            rewrite_mode="off",
            retrieval_top_k=2,
            embedding_mode="hybrid",
            hybrid_pool_mode="union",
        )

        reranker_pool = rerank.call_args.args[1]
        self.assertEqual([item.chunk_id for item in reranker_pool], [1, 2, 3])
        self.assertEqual(
            [item.chunk_id for item in diagnostics.pooled_candidates],
            [1, 2, 3],
        )

    @patch("retrieval.query.rerank")
    @patch("retrieval.query._search_by_vector")
    @patch("retrieval.query.get_embedding_model")
    def test_rrf_mode_still_caps_the_pool_at_retrieval_top_k(
        self,
        get_embedding_model: Mock,
        search_by_vector: Mock,
        rerank: Mock,
    ) -> None:
        raw = [candidate(1, 0.1), candidate(2, 0.2)]
        contextual = [candidate(2, 0.01), candidate(3, 0.02)]
        get_embedding_model.return_value.encode.return_value = [0.0]
        search_by_vector.side_effect = [raw, contextual]
        rerank.return_value = []

        retrieve_with_diagnostics(
            "test question",
            conn=object(),
            rewrite_mode="off",
            retrieval_top_k=2,
            embedding_mode="hybrid",
            hybrid_pool_mode="rrf",
        )

        reranker_pool = rerank.call_args.args[1]
        self.assertEqual(len(reranker_pool), 2)


if __name__ == "__main__":
    unittest.main()

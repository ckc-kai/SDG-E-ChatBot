import unittest

from eval.run_eval import _first_gold_rank_in_candidates
from retrieval.query import QueryObject, RankedResult


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


if __name__ == "__main__":
    unittest.main()

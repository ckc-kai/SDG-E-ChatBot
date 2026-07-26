import unittest

from retrieval.structured_query import (
    RankedStructuredResult,
    StructuredQueryObject,
    _caption_relevance,
    _lexical_query,
    structured_lane_fusion,
)


def query_object(
    chunk_id: int,
    origin: str,
    caption: str | None = None,
) -> StructuredQueryObject:
    content_type = "table" if origin == "structured" else "narrative"
    return StructuredQueryObject(
        key=f"{origin}:{chunk_id}",
        chunk_id=chunk_id,
        origin=origin,
        source_pdf="test.pdf",
        sub_document=None,
        breadcrumb="Test",
        section_number=None,
        page_start=1,
        page_end=2,
        chunk_index=0,
        content_type=content_type,
        content=f"chunk {chunk_id}",
        caption=caption,
        structured_data=None,
        image_path=None,
        token_count=2,
        distance=0.1,
    )


def candidate(chunk_id: int, origin: str) -> RankedStructuredResult:
    return RankedStructuredResult(
        query_object=query_object(chunk_id, origin),
        rerank_score=1.0 - chunk_id / 100,
    )


class LexicalQueryTests(unittest.TestCase):
    def test_uses_or_semantics_and_drops_question_stopwords(self) -> None:
        expression = _lexical_query(
            "What does SDG&E project for covered conductor through 2031?"
        )

        self.assertNotIn("what", expression)
        self.assertNotIn("through", expression)
        self.assertIn("sdg OR project OR covered OR conductor OR 2031", expression)

    def test_adds_acronym_for_title_cased_phrase(self) -> None:
        expression = _lexical_query(
            "Which event peaked on the Santa Ana Wildfire Threat Index?"
        )

        self.assertIn("sawti", expression.split(" OR "))


class CaptionRelevanceTests(unittest.TestCase):
    def test_prefers_caption_matching_minimum_and_topic(self) -> None:
        question = (
            "How does SDG&E's minimum vegetation-management maturity change?"
        )
        minimum = query_object(
            1,
            "structured",
            "Cross-Utility Maturity for Vegetation Management (Minimum Values)",
        )
        average = query_object(
            2,
            "structured",
            "Cross-Utility Maturity for Vegetation Management (Average Values)",
        )

        self.assertGreater(
            _caption_relevance(question, minimum),
            _caption_relevance(question, average),
        )

    def test_does_not_apply_caption_prior_to_narrative(self) -> None:
        narrative = query_object(1, "narrative", "Perfectly matching caption")

        self.assertEqual(
            _caption_relevance("Perfectly matching caption", narrative),
            0.0,
        )


class StructuredLaneFusionTests(unittest.TestCase):
    def test_reserves_four_of_first_five_slots_for_structured_results(self) -> None:
        ranked = [
            candidate(1, "narrative"),
            candidate(2, "narrative"),
            candidate(3, "structured"),
            candidate(4, "narrative"),
            candidate(5, "structured"),
            candidate(6, "structured"),
            candidate(7, "structured"),
            candidate(8, "structured"),
        ]

        fused = structured_lane_fusion(ranked)

        self.assertEqual(
            [result.query_object.origin for result in fused[:5]],
            ["structured", "narrative", "structured", "structured", "structured"],
        )
        self.assertEqual(
            [result.query_object.chunk_id for result in fused if result.query_object.origin == "structured"],
            [3, 5, 6, 7, 8],
        )

    def test_returns_original_order_when_only_one_lane_is_present(self) -> None:
        ranked = [candidate(1, "structured"), candidate(2, "structured")]

        self.assertEqual(structured_lane_fusion(ranked), ranked)


if __name__ == "__main__":
    unittest.main()

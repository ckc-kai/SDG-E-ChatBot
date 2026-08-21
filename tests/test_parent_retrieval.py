"""Parent-child retrieval and asymmetric query encoding.

Both are settings that fail silently rather than loudly. A parent expansion that
duplicates seam text still returns an answer, and a query encoded without its
prompt still returns results -- just worse ones. These tests are the only place
either becomes visible.
"""

from __future__ import annotations

import unittest
from dataclasses import dataclass
from unittest.mock import Mock

from retrieval.query.pdf.expansion import (
    PARENT_TOKEN_BUDGET,
    SectionSibling,
    join_without_overlap,
    select_parent_window,
)
from retrieval.utils import encode_query, model_encodes_asymmetrically


def sibling(index: int, content: str, tokens: int = 100) -> SectionSibling:
    return SectionSibling(
        chunk_id=1000 + index,
        chunk_index=index,
        content=content,
        token_count=tokens,
    )


class SeamJoinTests(unittest.TestCase):
    def test_repeated_overlap_is_removed_once(self):
        first = "The utility inspected 24,643 circuit miles during"
        second = "circuit miles during the reporting quarter."
        joined = join_without_overlap(first, second)
        self.assertEqual(
            joined,
            "The utility inspected 24,643 circuit miles during the reporting quarter.",
        )
        self.assertEqual(joined.count("circuit miles during"), 1)

    def test_disjoint_pieces_are_separated_not_glued(self):
        joined = join_without_overlap("first piece", "second piece")
        self.assertEqual(joined, "first piece\nsecond piece")

    def test_empty_operands_are_identity(self):
        self.assertEqual(join_without_overlap("", "text"), "text")
        self.assertEqual(join_without_overlap("text", ""), "text")

    def test_a_value_split_across_a_seam_survives_reconstruction(self):
        # The concrete pdf_003 failure: a figure that no single chunk contained.
        pieces = ["risk spend totalled 24,", "24,643 across the", "across the portfolio"]
        content = ""
        for piece in pieces:
            content = join_without_overlap(content, piece)
        self.assertIn("24,643", content)


class ParentWindowTests(unittest.TestCase):
    def test_a_section_inside_the_budget_is_returned_whole(self):
        siblings = [sibling(index, f"chunk {index}", 100) for index in range(5)]
        window = select_parent_window(siblings, anchor_index=2)
        self.assertEqual(len(window), 5)

    def test_an_oversized_section_is_capped(self):
        siblings = [sibling(index, f"chunk {index}", 600) for index in range(20)]
        window = select_parent_window(siblings, anchor_index=10)
        self.assertLessEqual(
            sum(item.token_count for item in window), PARENT_TOKEN_BUDGET
        )
        self.assertLess(len(window), 20)

    def test_the_anchor_is_always_inside_the_window(self):
        siblings = [sibling(index, f"chunk {index}", 900) for index in range(20)]
        for anchor in (0, 7, 19):
            window = select_parent_window(siblings, anchor_index=anchor)
            self.assertIn(anchor, [item.chunk_index for item in window])

    def test_the_window_is_contiguous(self):
        siblings = [sibling(index, f"chunk {index}", 700) for index in range(12)]
        indexes = [item.chunk_index for item in select_parent_window(siblings, 5)]
        self.assertEqual(indexes, list(range(indexes[0], indexes[-1] + 1)))

    def test_a_missing_anchor_expands_nothing(self):
        siblings = [sibling(index, f"chunk {index}") for index in range(3)]
        self.assertEqual(select_parent_window(siblings, anchor_index=99), [])

    def test_a_single_oversized_child_is_still_returned(self):
        siblings = [sibling(0, "huge", PARENT_TOKEN_BUDGET * 2)]
        self.assertEqual(len(select_parent_window(siblings, anchor_index=0)), 1)


@dataclass(frozen=True)
class FakeQueryObject:
    chunk_id: int
    chunk_index: int
    content: str
    token_count: int
    content_type: str = "narrative"


@dataclass(frozen=True)
class FakeRanked:
    query_object: FakeQueryObject
    rerank_score: float = 1.0


class SectionDeduplicationTests(unittest.TestCase):
    """Two hits from one section must not each pay the full parent budget."""

    def setUp(self) -> None:
        from retrieval.query.pdf import expansion

        self.expansion = expansion
        self.section = ((1, 7), [sibling(0, "alpha text"), sibling(1, "beta text")])
        self.original = expansion._fetch_section_siblings
        expansion._fetch_section_siblings = lambda conn, ids: {
            anchor: self.section for anchor in ids
        }
        self.addCleanup(
            setattr, expansion, "_fetch_section_siblings", self.original
        )

    def test_only_the_first_hit_in_a_section_is_expanded(self):
        results = [
            FakeRanked(FakeQueryObject(chunk_id=1000, chunk_index=0, content="alpha text", token_count=100)),
            FakeRanked(FakeQueryObject(chunk_id=1001, chunk_index=1, content="beta text", token_count=100)),
        ]
        expanded = self.expansion.expand_to_parent_sections(None, results)
        self.assertIn("beta text", expanded[0].query_object.content)
        self.assertEqual(expanded[1].query_object.content, "beta text")

    def test_ranking_order_and_scores_are_preserved(self):
        results = [
            FakeRanked(FakeQueryObject(chunk_id=1000, chunk_index=0, content="alpha text", token_count=100), 0.9),
            FakeRanked(FakeQueryObject(chunk_id=1001, chunk_index=1, content="beta text", token_count=100), 0.4),
        ]
        expanded = self.expansion.expand_to_parent_sections(None, results)
        self.assertEqual([item.rerank_score for item in expanded], [0.9, 0.4])
        self.assertEqual(
            [item.query_object.chunk_id for item in expanded], [1000, 1001]
        )

    def test_a_query_failure_degrades_to_unexpanded_results(self):
        def boom(conn, ids):
            raise RuntimeError("database is gone")

        self.expansion._fetch_section_siblings = boom
        connection = Mock()
        results = [
            FakeRanked(FakeQueryObject(chunk_id=1000, chunk_index=0, content="alpha", token_count=10))
        ]
        self.assertEqual(
            self.expansion.expand_to_parent_sections(connection, results), results
        )
        connection.rollback.assert_called_once_with()


class AsymmetricEncodingTests(unittest.TestCase):
    """The configured model must encode a query differently from a document.

    Applying the prompt to documents, or omitting it on queries, degrades
    retrieval with no error -- so it is asserted rather than assumed.
    """

    @classmethod
    def setUpClass(cls) -> None:
        from retrieval.utils import get_embedding_model

        cls.model = get_embedding_model()

    def test_query_and_document_encodings_of_the_same_string_differ(self):
        if not model_encodes_asymmetrically(self.model):
            self.skipTest("configured embedding model has no query prompt")
        text = "how many circuit miles were inspected"
        as_query = encode_query(self.model, text)
        as_document = self.model.encode(text, normalize_embeddings=True)
        self.assertFalse(
            (as_query == as_document).all(),
            "query and document encodings are identical: the query prompt is "
            "not being applied",
        )

    def test_encode_query_still_works_without_a_query_prompt(self):
        @dataclass
        class PromptlessModel:
            prompts: dict

            def encode(self, text, **kwargs):
                assert "prompt_name" not in kwargs
                return [0.0, 1.0]

        self.assertEqual(encode_query(PromptlessModel(prompts={}), "q"), [0.0, 1.0])


if __name__ == "__main__":
    unittest.main()

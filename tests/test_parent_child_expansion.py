import unittest
from types import SimpleNamespace

from retrieval.query.pdf.expansion import select_sibling_context


def chunk(identifier, index, tokens, breadcrumb="Section A"):
    return SimpleNamespace(
        chunk_id=identifier,
        source_pdf="wmp.pdf",
        sub_document=None,
        breadcrumb=breadcrumb,
        chunk_index=index,
        token_count=tokens,
        content_type="narrative",
    )


class ParentChildExpansionTests(unittest.TestCase):
    def test_selects_nearest_same_section_siblings_within_token_cap(self):
        anchor = chunk(10, 5, 300)
        selected = select_sibling_context(
            (anchor,),
            (
                chunk(9, 4, 700),
                chunk(11, 6, 700),
                chunk(12, 7, 700),
                chunk(20, 4, 100, breadcrumb="Other"),
            ),
            token_budget=1400,
        )
        self.assertEqual([item.chunk_id for item in selected], [9, 11])
        self.assertLessEqual(sum(item.token_count for item in selected), 1400)


if __name__ == "__main__":
    unittest.main()

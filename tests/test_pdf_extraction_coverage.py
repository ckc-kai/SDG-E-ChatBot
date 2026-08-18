"""Extraction-coverage tests for the PDF ingest.

Both defects these cover -- leaf-only extraction (A1) and the discarded heading
trim (A2) -- were silent for the life of the project, costing 31% of the corpus
without one error or log line. The fixtures are synthetic outlines rather than
real PDFs so the whole class of bug is testable in milliseconds, with no
dependency on which files happen to be on disk.
"""

from __future__ import annotations

import unittest
from pathlib import Path

from retrieval.ingest.pdf.ingest import (
    COVERAGE_MIN_RATIO,
    ExtractionCoverage,
    LeafSection,
    SectionNode,
    _document_sections,
    _visible_chars,
    assert_extraction_coverage,
    extract_section_texts,
)
from retrieval.ingest.pdf.structured_extraction import map_page_to_leaf


class FakePages:
    """Stands in for PageTextCache over a fixed list of page texts."""

    def __init__(self, pages: list[str]) -> None:
        self._pages = pages

    @property
    def page_count(self) -> int:
        return len(self._pages)

    def text(self, index: int) -> str:
        return self._pages[index] if 0 <= index < len(self._pages) else ""

    def total_visible_chars(self) -> int:
        return sum(_visible_chars(page) for page in self._pages)


def node(
    title: str,
    page_start: int,
    depth: int = 0,
    children: tuple[SectionNode, ...] = (),
) -> SectionNode:
    return SectionNode(
        title=title,
        depth=depth,
        page_start=page_start,
        breadcrumb=title,
        sub_document=None,
        section_number=None,
        children=children,
    )


def flatten(nodes: tuple[SectionNode, ...]) -> list[SectionNode]:
    flat: list[SectionNode] = []
    for item in nodes:
        flat.append(item)
        flat.extend(flatten(item.children))
    return flat


def lossless(pages: FakePages, texts: list[str]) -> bool:
    """Every visible character on every page reached exactly one section."""
    return sum(_visible_chars(text) for text in texts) == pages.total_visible_chars()


class SharedBoundaryPageTests(unittest.TestCase):
    """A2: the tail above a heading belongs to the previous section."""

    def setUp(self) -> None:
        self.pages = FakePages(
            [
                "1.1 First Section heading\nopening prose",
                "tail of the first section\n1.2 Second Section heading\nsecond prose",
                "more of the second section",
            ]
        )
        tree = (node("1.1 First Section", 0), node("1.2 Second Section", 1))
        self.sections = _document_sections(flatten(tree), 3, "doc.pdf")
        self.texts = extract_section_texts(self.pages, self.sections)

    def test_previous_section_keeps_its_tail_on_the_shared_page(self):
        self.assertIn("tail of the first section", self.texts[0])

    def test_next_section_still_starts_at_its_own_heading(self):
        self.assertTrue(self.texts[1].startswith("1.2 Second Section heading"))
        self.assertNotIn("tail of the first section", self.texts[1])

    def test_no_text_is_lost_at_the_boundary(self):
        self.assertTrue(lossless(self.pages, self.texts))


class ParentSectionTests(unittest.TestCase):
    """A1: a parent heading's own prose is content, not just a breadcrumb."""

    def setUp(self) -> None:
        child = node("2.1 Child Section", 2, depth=1)
        parent = node("2 Parent Section", 0, children=(child,))
        self.pages = FakePages(
            [
                "2 Parent Section heading\nparent prose one",
                "parent prose two",
                "2.1 Child Section heading\nchild prose",
            ]
        )
        self.sections = _document_sections(flatten((parent,)), 3, "doc.pdf")
        self.texts = extract_section_texts(self.pages, self.sections)

    def test_parent_prose_is_extracted(self):
        self.assertIn("parent prose one", self.texts[0])
        self.assertIn("parent prose two", self.texts[0])

    def test_parent_span_is_flagged_and_child_is_not(self):
        self.assertTrue(self.sections[0].is_parent_span)
        self.assertFalse(self.sections[1].is_parent_span)

    def test_parent_span_does_not_overlap_its_child(self):
        parent, child = self.sections
        self.assertEqual(parent.page_end, child.page_start)
        self.assertNotIn("child prose", self.texts[0])

    def test_no_text_is_lost(self):
        self.assertTrue(lossless(self.pages, self.texts))


class FrontMatterTests(unittest.TestCase):
    """A1: pages before the first bookmark are claimed by no outline node."""

    def setUp(self) -> None:
        self.pages = FakePages(
            [
                "cover page",
                "table of contents",
                "executive summary body",
                "1 Introduction heading\nintro prose",
            ]
        )
        tree = (node("1 Introduction", 3),)
        self.sections = _document_sections(flatten(tree), 4, "doc.pdf")
        self.texts = extract_section_texts(self.pages, self.sections)

    def test_pre_bookmark_pages_become_a_section(self):
        self.assertEqual(self.sections[0].page_start, 0)
        self.assertEqual(self.sections[0].page_end, 3)
        self.assertTrue(self.sections[0].is_parent_span)

    def test_executive_summary_is_extracted(self):
        self.assertIn("executive summary body", self.texts[0])

    def test_front_matter_breadcrumb_falls_back_to_the_document(self):
        self.assertEqual(self.sections[0].breadcrumb, "doc.pdf")

    def test_no_text_is_lost(self):
        self.assertTrue(lossless(self.pages, self.texts))


class HeadingAnchorGuardTests(unittest.TestCase):
    def test_generic_one_word_title_is_not_anchored(self):
        pages = FakePages(["prose mentioning Overview in passing\nOverview heading"])
        tree = (node("Overview", 0),)
        texts = extract_section_texts(pages, _document_sections(flatten(tree), 1, "d.pdf"))
        self.assertTrue(texts[0].startswith("prose mentioning"))

    def test_title_below_the_character_floor_is_not_anchored(self):
        pages = FakePages(["leading prose\nA B heading"])
        tree = (node("A B", 0),)
        texts = extract_section_texts(pages, _document_sections(flatten(tree), 1, "d.pdf"))
        self.assertTrue(texts[0].startswith("leading prose"))

    def test_unfindable_title_leaves_the_page_untouched(self):
        pages = FakePages(["page content that never names the section"])
        tree = (node("Nowhere To Be Found", 0),)
        texts = extract_section_texts(pages, _document_sections(flatten(tree), 1, "d.pdf"))
        self.assertEqual(texts[0], "page content that never names the section")

    def test_backwards_anchors_on_a_shared_page_lose_nothing(self):
        # The second heading resolves EARLIER on the page than the first. Clamping
        # makes the later section swallow the slice rather than dropping it.
        pages = FakePages(["Second Heading Here first\nFirst Heading Here second"])
        tree = (node("First Heading Here", 0), node("Second Heading Here", 0))
        sections = _document_sections(flatten(tree), 1, "d.pdf")
        texts = extract_section_texts(pages, sections)
        self.assertTrue(lossless(pages, texts))

    def test_text_above_a_first_bookmark_on_page_zero_is_kept(self):
        # Nothing precedes the first section, so it must own its page from the
        # top or the cover text above its heading has no owner at all.
        pages = FakePages(["SDG&E cover title block\n1 Introduction heading\nprose"])
        tree = (node("1 Introduction", 0),)
        sections = _document_sections(flatten(tree), 1, "d.pdf")
        texts = extract_section_texts(pages, sections)
        self.assertIn("SDG&E cover title block", texts[0])
        self.assertTrue(lossless(pages, texts))


class UnresolvableDestinationTests(unittest.TestCase):
    def test_page_beyond_the_document_is_clamped_not_crashed(self):
        pages = FakePages(["only page"])
        tree = (node("Real Section Heading", 0), node("Broken Section Heading", 9))
        sections = _document_sections(flatten(tree), 1, "d.pdf")
        texts = extract_section_texts(pages, sections)
        self.assertEqual(len(texts), 2)
        # Which of the two collapsed sections ends up holding the page is not
        # something the caller should depend on; that the page survives is.
        self.assertIn("only page", "".join(texts))
        self.assertTrue(lossless(pages, texts))

    def test_backwards_page_order_does_not_produce_a_negative_span(self):
        tree = (node("Later Section Heading", 5), node("Earlier Section Heading", 2))
        sections = _document_sections(flatten(tree), 10, "d.pdf")
        for section in sections:
            self.assertGreaterEqual(section.page_end, section.page_start)


class SectionOrdinalTests(unittest.TestCase):
    """Trap 1: parent spans let two sections share a page_start."""

    def test_sections_sharing_a_page_start_get_distinct_ordinals(self):
        child = node("3.1 Child Heading", 0, depth=1)
        parent = node("3 Parent Heading", 0, children=(child,))
        sections = _document_sections(flatten((parent,)), 2, "d.pdf")
        self.assertEqual(sections[0].page_start, sections[1].page_start)
        self.assertNotEqual(sections[0].section_ordinal, sections[1].section_ordinal)

    def test_ordinals_are_document_order_and_dense(self):
        tree = (node("One Heading Here", 0), node("Two Heading Here", 1))
        sections = _document_sections(flatten(tree), 2, "d.pdf")
        self.assertEqual([s.section_ordinal for s in sections], [0, 1])


class StructuredMappingTests(unittest.TestCase):
    """Trap 2: parent spans must not capture a child's tables and figures."""

    def test_a_table_on_a_child_page_maps_to_the_child_not_the_parent(self):
        parent = LeafSection(
            title="4 Parent",
            breadcrumb="4 Parent",
            sub_document=None,
            section_number="4",
            page_start=10,
            page_end=20,  # deliberately wide, as a mis-built parent span would be
            section_ordinal=0,
            is_parent_span=True,
        )
        child = LeafSection(
            title="4.1 Child",
            breadcrumb="4 Parent > 4.1 Child",
            sub_document=None,
            section_number="4.1",
            page_start=15,
            page_end=20,
            section_ordinal=1,
        )
        leaves = [section for section in (parent, child) if not section.is_parent_span]
        mapped = map_page_to_leaf(leaves, 16)
        self.assertIsNotNone(mapped)
        self.assertEqual(mapped.breadcrumb, "4 Parent > 4.1 Child")

    def test_build_leaf_sections_excludes_parent_spans(self):
        child = node("5.1 Child Heading", 3, depth=1)
        parent = node("5 Parent Heading", 0, children=(child,))
        sections = _document_sections(flatten((parent,)), 6, "d.pdf")
        leaves = [
            section
            for section in sections
            if not section.is_parent_span and section.page_end > section.page_start
        ]
        self.assertEqual([section.title for section in leaves], ["5.1 Child Heading"])


class CoverageAssertionTests(unittest.TestCase):
    def test_shortfall_fails_the_run(self):
        with self.assertRaises(RuntimeError) as caught:
            assert_extraction_coverage("x.pdf", ExtractionCoverage(500, 1000))
        self.assertIn("50.0%", str(caught.exception))
        self.assertIn("500", str(caught.exception))

    def test_coverage_at_the_floor_passes(self):
        chunked = int(1000 * COVERAGE_MIN_RATIO)
        assert_extraction_coverage("x.pdf", ExtractionCoverage(chunked, 1000))

    def test_a_document_with_no_extractable_text_does_not_divide_by_zero(self):
        assert_extraction_coverage("scan.pdf", ExtractionCoverage(0, 0))


class NoOutlineFallbackTests(unittest.TestCase):
    def test_whole_document_becomes_one_untitled_section(self):
        from retrieval.ingest.pdf.ingest import build_document_sections

        class FakeReader:
            outline = None
            pages = [object(), object(), object()]

        sections = build_document_sections(Path("no_outline.pdf"), FakeReader())
        self.assertEqual(len(sections), 1)
        self.assertEqual(sections[0].title, "")
        self.assertEqual(sections[0].breadcrumb, "no_outline.pdf")
        self.assertEqual((sections[0].page_start, sections[0].page_end), (0, 3))


if __name__ == "__main__":
    unittest.main()

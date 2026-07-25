import unittest
from pathlib import Path

from retrieval.ingest import LeafSection
from retrieval.structured_ingest import (
    StructuredElement,
    build_structured_chunks,
    chunk_table_grid,
    compose_figure_text,
    compose_table_text,
    grid_to_markdown,
    is_noise_figure,
    is_noise_table,
    map_page_to_leaf,
    piece_markdown,
    safe_document_slug,
)


class FakeTokenizer:
    """Whitespace tokenizer implementing the two interfaces the ingest uses:
    `encode` for token counting and offset-mapping calls for chunk_text."""

    is_fast = True

    def encode(self, text, add_special_tokens=False, **kwargs):
        return text.split()

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=True):
        offsets = []
        position = 0
        for token in text.split():
            start = text.index(token, position)
            offsets.append((start, start + len(token)))
            position = start + len(token)
        return {"offset_mapping": offsets}


def leaf(breadcrumb: str, page_start: int, page_end: int, **overrides) -> LeafSection:
    defaults = dict(
        title=breadcrumb,
        breadcrumb=breadcrumb,
        sub_document=None,
        section_number=None,
        page_start=page_start,
        page_end=page_end,
    )
    defaults.update(overrides)
    return LeafSection(**defaults)


class GridToMarkdownTests(unittest.TestCase):
    def test_renders_header_separator_and_body_rows(self) -> None:
        markdown = grid_to_markdown(
            [["Initiative", "Target"], ["Pole Replacement", "267"]]
        )

        lines = markdown.splitlines()
        self.assertEqual(lines[0], "| Initiative | Target |")
        self.assertEqual(lines[1], "| --- | --- |")
        self.assertEqual(lines[2], "| Pole Replacement | 267 |")

    def test_escapes_pipes_and_collapses_internal_whitespace(self) -> None:
        markdown = grid_to_markdown([["a|b", "x\n y"], ["1", "2"]])

        self.assertIn("a\\|b", markdown)
        self.assertIn("x y", markdown)

    def test_pads_ragged_rows_to_uniform_width(self) -> None:
        markdown = grid_to_markdown([["h1", "h2", "h3"], ["only-one"]])

        body = markdown.splitlines()[2]
        self.assertEqual(body.count("|"), 4)

    def test_empty_grid_returns_empty_string(self) -> None:
        self.assertEqual(grid_to_markdown([]), "")


class ChunkTableGridTests(unittest.TestCase):
    def test_small_table_fits_in_one_range(self) -> None:
        grid = [["h1", "h2"], ["a", "b"], ["c", "d"]]

        ranges = chunk_table_grid(grid, FakeTokenizer(), max_tokens=400)

        self.assertEqual(ranges, [(0, 2)])

    def test_ranges_cover_all_body_rows_without_overlap(self) -> None:
        grid = [["header", "row"]] + [[f"cell{i}", f"value{i}"] for i in range(40)]

        ranges = chunk_table_grid(grid, FakeTokenizer(), max_tokens=30)

        self.assertGreater(len(ranges), 1)
        self.assertEqual(ranges[0][0], 0)
        self.assertEqual(ranges[-1][1], 40)
        for (_, previous_end), (next_start, _) in zip(ranges, ranges[1:]):
            self.assertEqual(previous_end, next_start)

    def test_single_oversized_row_still_gets_its_own_range(self) -> None:
        wide_row = [f"word{i}" for i in range(100)]
        grid = [["h"] * 100, wide_row, ["tiny"]]

        ranges = chunk_table_grid(grid, FakeTokenizer(), max_tokens=20)

        self.assertEqual(ranges[0], (0, 1))
        self.assertEqual(ranges[-1][1], 2)

    def test_header_only_table_returns_empty_body_range(self) -> None:
        ranges = chunk_table_grid([["h1", "h2"]], FakeTokenizer(), max_tokens=400)

        self.assertEqual(ranges, [(0, 0)])


class PieceMarkdownTests(unittest.TestCase):
    def test_every_piece_repeats_the_header(self) -> None:
        grid = [["name", "value"], ["a", "1"], ["b", "2"], ["c", "3"]]

        piece = piece_markdown(grid, (1, 3))

        lines = piece.splitlines()
        self.assertEqual(lines[0], "| name | value |")
        self.assertEqual(lines[2], "| b | 2 |")
        self.assertEqual(lines[3], "| c | 3 |")
        self.assertEqual(len(lines), 4)


class ComposeTextTests(unittest.TestCase):
    def test_table_text_includes_caption_when_present(self) -> None:
        text = compose_table_text("Table 12: Targets", "| a |")

        self.assertTrue(text.startswith("Table: Table 12: Targets"))
        self.assertIn("| a |", text)

    def test_table_text_without_caption_is_just_markdown(self) -> None:
        self.assertEqual(compose_table_text("", "| a |"), "| a |")

    def test_figure_text_combines_caption_and_description(self) -> None:
        text = compose_figure_text("Figure 3: Risk curve", "A line chart trending up.")

        self.assertIn("Figure: Figure 3: Risk curve", text)
        self.assertIn("Description: A line chart trending up.", text)

    def test_figure_text_empty_when_no_caption_or_description(self) -> None:
        self.assertEqual(compose_figure_text("", ""), "")

    def test_figure_text_appends_truncated_page_context(self) -> None:
        text = compose_figure_text(
            "", "A chart.", page_context="Drone   Assessments\nof transmission assets " + "x" * 500
        )

        self.assertIn("Page context: Drone Assessments of transmission assets", text)
        self.assertLess(len(text.splitlines()[-1]), 320)


class NoiseFilterTests(unittest.TestCase):
    def test_dot_leader_toc_grid_is_noise(self) -> None:
        grid = [
            ["Change Order " + "." * 90 + " 4", "4"],
            ["Covered Conductor " + "." * 80 + " 5", "5"],
        ]

        self.assertTrue(is_noise_table(grid))

    def test_data_table_is_not_noise(self) -> None:
        grid = [
            ["Initiative", "2023", "2024"],
            ["Strategic Pole Replacement", "200", "267"],
        ]

        self.assertFalse(is_noise_table(grid))

    def test_empty_grid_is_noise(self) -> None:
        self.assertTrue(is_noise_table([["", ""], ["", ""]]))

    def test_captionless_logo_description_is_noise_figure(self) -> None:
        self.assertTrue(
            is_noise_figure("", "The image features a logo consisting of two circles.")
        )

    def test_captioned_figure_is_kept_even_if_description_mentions_logo(self) -> None:
        self.assertFalse(
            is_noise_figure("Figure 1: Corporate structure", "Chart with a logo inset.")
        )

    def test_dot_runs_inside_cells_are_collapsed_in_markdown(self) -> None:
        markdown = grid_to_markdown(
            [["Section", "Page"], ["Change Order " + "." * 60, "4"]]
        )

        self.assertNotIn("....", markdown)
        self.assertIn("Change Order", markdown)


class MapPageToLeafTests(unittest.TestCase):
    def test_returns_containing_leaf(self) -> None:
        leaves = [leaf("A", 0, 5), leaf("B", 5, 10)]

        self.assertEqual(map_page_to_leaf(leaves, 7).breadcrumb, "B")

    def test_boundary_page_prefers_later_starting_leaf(self) -> None:
        # Page 5 holds the tail of A (0-6) and the start of B (5-10).
        leaves = [leaf("A", 0, 6), leaf("B", 5, 10)]

        self.assertEqual(map_page_to_leaf(leaves, 5).breadcrumb, "B")

    def test_returns_none_when_no_leaf_contains_page(self) -> None:
        self.assertIsNone(map_page_to_leaf([leaf("A", 0, 5)], 12))


class SafeDocumentSlugTests(unittest.TestCase):
    def test_replaces_shell_hostile_characters(self) -> None:
        slug = safe_document_slug("SDG&E_2023-2023_Base-WMP_R5-redacted.pdf")

        self.assertNotIn("&", slug)
        self.assertNotIn(" ", slug)
        self.assertTrue(slug.startswith("SDG_E_2023-2023"))


class BuildStructuredChunksTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tokenizer = FakeTokenizer()
        self.pdf_path = Path("resources/wmp/pdf/sample.pdf")
        self.leaves = [
            leaf("8 Grid Design > 8.1 Overview", 0, 3),
            leaf("8 Grid Design > 8.2 Targets", 3, 9),
        ]

    def table_element(self, page: int, caption: str = "Table 1") -> StructuredElement:
        return StructuredElement(
            content_type="table",
            page_start=page,
            page_end=page + 1,
            caption=caption,
            grid=(
                ("Initiative", "2023", "2024"),
                ("Strategic Pole Replacement", "200", "267"),
                ("Weather Stations", "219", "222"),
            ),
        )

    def figure_element(self, page: int, caption: str, description: str) -> StructuredElement:
        return StructuredElement(
            content_type="figure",
            page_start=page,
            page_end=page + 1,
            caption=caption,
            description=description,
            image=None,
        )

    def test_table_chunk_carries_section_metadata_and_grid(self) -> None:
        chunks = build_structured_chunks(
            self.pdf_path, [self.table_element(page=4)], self.leaves, self.tokenizer
        )

        self.assertEqual(len(chunks), 1)
        chunk = chunks[0]
        self.assertEqual(chunk.content_type, "table")
        self.assertEqual(chunk.breadcrumb, "8 Grid Design > 8.2 Targets")
        self.assertEqual(chunk.structured_data["num_rows"], 3)
        self.assertEqual(
            chunk.structured_data["grid"][1],
            ["Strategic Pole Replacement", "200", "267"],
        )
        self.assertIn("| Strategic Pole Replacement | 200 | 267 |", chunk.content)
        self.assertTrue(chunk.content.startswith("Table: Table 1"))

    def test_unmapped_page_falls_back_to_document_breadcrumb(self) -> None:
        chunks = build_structured_chunks(
            self.pdf_path, [self.table_element(page=50)], self.leaves, self.tokenizer
        )

        self.assertEqual(chunks[0].breadcrumb, "sample.pdf")
        self.assertIsNone(chunks[0].sub_document)

    def test_chunk_indices_are_unique_per_content_type_and_page(self) -> None:
        elements = [
            self.table_element(page=4, caption="Table A"),
            self.table_element(page=4, caption="Table B"),
            self.figure_element(page=4, caption="Figure 1", description="A map."),
        ]

        chunks = build_structured_chunks(self.pdf_path, elements, self.leaves, self.tokenizer)

        keys = [(c.content_type, c.page_start, c.chunk_index) for c in chunks]
        self.assertEqual(len(keys), len(set(keys)))
        table_indices = sorted(c.chunk_index for c in chunks if c.content_type == "table")
        self.assertEqual(table_indices, [0, 1])
        figure_indices = [c.chunk_index for c in chunks if c.content_type == "figure"]
        self.assertEqual(figure_indices, [0])

    def test_text_empty_figure_is_skipped(self) -> None:
        chunks = build_structured_chunks(
            self.pdf_path,
            [self.figure_element(page=1, caption="", description="")],
            self.leaves,
            self.tokenizer,
        )

        self.assertEqual(chunks, [])

    def test_figure_without_image_has_no_image_path(self) -> None:
        chunks = build_structured_chunks(
            self.pdf_path,
            [self.figure_element(page=1, caption="Figure 2: PSPS map", description="Shaded regions.")],
            self.leaves,
            self.tokenizer,
        )

        self.assertEqual(len(chunks), 1)
        self.assertIsNone(chunks[0].image_path)
        self.assertIn("Figure: Figure 2: PSPS map", chunks[0].content)

    def test_figure_chunk_includes_page_context_when_provided(self) -> None:
        chunks = build_structured_chunks(
            self.pdf_path,
            [self.figure_element(page=1, caption="", description="A bar chart.")],
            self.leaves,
            self.tokenizer,
            page_text_fn=lambda page_idx: f"Slide title for page {page_idx}",
        )

        self.assertEqual(len(chunks), 1)
        self.assertIn("Page context: Slide title for page 1", chunks[0].content)

    def test_oversized_table_is_split_with_header_repeated_in_each_piece(self) -> None:
        big_grid = tuple(
            [("Initiative", "Detail")]
            + [(f"row-{i}", " ".join(f"w{j}" for j in range(20))) for i in range(60)]
        )
        element = StructuredElement(
            content_type="table",
            page_start=4,
            page_end=5,
            caption="Big table",
            grid=big_grid,
        )

        chunks = build_structured_chunks(self.pdf_path, [element], self.leaves, self.tokenizer)

        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertIn("| Initiative | Detail |", chunk.content)
        covered = []
        for chunk in chunks:
            covered.extend(
                range(chunk.structured_data["piece_rows"][0], chunk.structured_data["piece_rows"][1])
            )
        self.assertEqual(sorted(set(covered)), list(range(60)))


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from openpyxl import Workbook

from retrieval.ingest.excel.workbook import build_workbook_chunks


class FakeTokenizer:
    def encode(self, text, **_kwargs):
        return text.split()

    def decode(self, tokens, **_kwargs):
        return " ".join(tokens)


class WorkbookIngestTests(unittest.TestCase):
    def test_preserves_sheet_rows_and_skips_empty_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.xlsx"
            workbook = Workbook()
            sheet = workbook.active
            sheet.title = "Targets"
            sheet.append(["Metric", "Year", "Value"])
            sheet.append(["Pole replacement", 2025, 100])
            sheet.append([None, None, None])
            workbook.save(path)
            workbook.close()

            chunks = build_workbook_chunks(path, FakeTokenizer(), max_tokens=100)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].sheet, "Targets")
        self.assertEqual((chunks[0].row_start, chunks[0].row_end), (1, 2))
        self.assertIn("Pole replacement", chunks[0].content)

    def test_ignores_formatting_only_cells_at_excel_limits(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "formatted-range.xlsx"
            workbook = Workbook()
            sheet = workbook.active
            sheet.title = "Targets"
            sheet.append(["Metric", "Value"])
            sheet.append(["Detailed inspection", 100])
            sheet["XFD1048576"].number_format = "0.00"
            workbook.save(path)
            workbook.close()

            chunks = build_workbook_chunks(path, FakeTokenizer(), max_tokens=100)

        self.assertEqual(len(chunks), 1)
        self.assertEqual((chunks[0].row_start, chunks[0].row_end), (1, 2))
        self.assertNotIn("1048576", chunks[0].content)


if __name__ == "__main__":
    unittest.main()

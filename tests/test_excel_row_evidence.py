"""The workbook rows behind a retrieved Excel card must reach the prompt."""

from __future__ import annotations

import unittest
from unittest.mock import Mock

from retrieval.query.excel import rows as excel_rows
from retrieval.query.excel.rows import (
    MAX_CARDS,
    MAX_ROWS_PER_CARD,
    ExcelRowSlice,
    fetch_card_rows,
)


def _card(chunk_id: str, structured: dict, caption: str = "a card"):
    return Mock(
        query_object=Mock(
            chunk_id=chunk_id, structured_data=structured, caption=caption
        )
    )


class _FakeCursor:
    def __init__(self, rows, captured):
        self._rows = rows
        self._captured = captured

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params):
        self._captured.append((sql, params))

    def fetchall(self):
        return self._rows


class _FakeConnection:
    def __init__(self, rows):
        self.rows = rows
        self.captured: list = []

    def cursor(self):
        return _FakeCursor(self.rows, self.captured)


class RenderTests(unittest.TestCase):
    def test_jsonb_values_are_flattened_rather_than_dropped(self):
        """A record's numbers live in ``attributes``; rendering must show them."""
        row_slice = ExcelRowSlice(
            table_number=1,
            card_chunk_id="7",
            caption="Undergrounding",
            columns=("entity_key", "attributes"),
            rows=(
                (
                    "WMP.455",
                    {
                        "annual_quant_target": 60.0,
                        "quant_actual_progress_q1_4": 35.91,
                        "internal_note": "ignored",
                        "percent_complete": None,
                    },
                ),
            ),
            truncated=False,
        )

        rendered = row_slice.render()

        self.assertIn("annual_quant_target=60.0", rendered)
        self.assertIn("quant_actual_progress_q1_4=35.91", rendered)
        self.assertNotIn("internal_note", rendered)
        # A null attribute is noise, not evidence.
        self.assertNotIn("percent_complete", rendered)

    def test_truncation_is_disclosed(self):
        row_slice = ExcelRowSlice(
            table_number=15,
            card_chunk_id="9",
            caption="Risk",
            columns=("metric_name",),
            rows=(("Overall utility risk",),),
            truncated=True,
        )

        self.assertIn("more exist", row_slice.render())

    def test_source_file_matches_the_manifest_name(self):
        row_slice = ExcelRowSlice(1, "7", "c", ("a",), (("b",),), False)

        self.assertEqual(row_slice.source_file, "sdge_table01_rag_ready.csv")


class FetchTests(unittest.TestCase):
    def test_entity_card_narrows_to_its_own_activity(self):
        connection = _FakeConnection([("WMP.455", "t", 2024, 4, "Delayed", {})])

        fetch_card_rows(
            "What happened in 2024?",
            [_card("7", {"table_number": 1, "entity_key": "WMP.455"})],
            connection,
        )

        sql, params = connection.captured[0]
        self.assertIn("excel_records", sql)
        self.assertIn("t.entity_key = %s", sql)
        self.assertIn("WMP.455", params)
        # A year named in the question must narrow the window.
        self.assertIn([2024], params)

    def test_fact_card_without_a_metric_key_does_not_filter_by_metric(self):
        connection = _FakeConnection([("Risk", 2023, None, "T3", "D", "1", "", {})])

        fetch_card_rows(
            "Which segments are riskiest?",
            [_card("9", {"table_number": 15})],
            connection,
        )

        sql, _ = connection.captured[0]
        self.assertIn("excel_facts", sql)
        self.assertNotIn("semantic_metric_key", sql)

    def test_truncation_is_detected_without_returning_the_extra_row(self):
        connection = _FakeConnection(
            [("Risk", 2023, None, "T3", "D", str(index), "", {})
             for index in range(MAX_ROWS_PER_CARD + 1)]
        )

        (row_slice,) = fetch_card_rows(
            "q", [_card("9", {"table_number": 15})], connection
        )

        self.assertEqual(len(row_slice.rows), MAX_ROWS_PER_CARD)
        self.assertTrue(row_slice.truncated)

    def test_cards_describing_the_same_scope_are_read_once(self):
        connection = _FakeConnection([("Risk", 2023, None, "T3", "D", "1", "", {})])
        same = {"table_number": 15, "semantic_metric_key": "overall_risk"}

        slices = fetch_card_rows(
            "q", [_card("9", same), _card("10", same)], connection
        )

        self.assertEqual(len(slices), 1)
        window_queries = [
            sql for sql, _ in connection.captured if "DISTINCT" not in sql
        ]
        self.assertEqual(len(window_queries), 1)

    def test_no_more_than_the_card_cap_is_read(self):
        connection = _FakeConnection([("Risk", 2023, None, "T3", "D", "1", "", {})])
        cards = [
            _card(str(index), {"table_number": 15, "semantic_metric_key": str(index)})
            for index in range(MAX_CARDS + 3)
        ]

        self.assertEqual(len(fetch_card_rows("q", cards, connection)), MAX_CARDS)

    def test_a_database_failure_never_breaks_retrieval(self):
        class Exploding:
            def cursor(self):
                raise RuntimeError("connection lost")

        self.assertEqual(
            fetch_card_rows("q", [_card("9", {"table_number": 15})], Exploding()),
            (),
        )

    def test_a_card_without_a_table_number_is_skipped(self):
        connection = _FakeConnection([("x",)])

        self.assertEqual(fetch_card_rows("q", [_card("9", {})], connection), ())
        self.assertEqual(connection.captured, [])


if __name__ == "__main__":
    unittest.main()

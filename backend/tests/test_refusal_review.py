"""A refusal is checked against the record before it is accepted.

The answer model judges evidence sufficiency itself and is measurably wrong
about it: the 2026-08-23 beta run refused 10 of 27 answerable questions while
holding the data. These assert the three properties that make the review safe
-- it fires only when the record contradicts the refusal, it never fires on an
empty record, and it never loops.
"""

from __future__ import annotations

import sys
import types
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from generation.schemas import AnswerRequest, Chunk, ChunkMetadata  # noqa: E402
from services.generation_service import GenerationService  # noqa: E402


def _bundle(*, verified_rows: int, ranked: int):
    """A bundle carrying only what the snapshot reads."""
    answers = ()
    if verified_rows:
        result = types.SimpleNamespace(rows=tuple(range(verified_rows)))
        answers = (types.SimpleNamespace(result=result),)
    group = types.SimpleNamespace(results=tuple(range(ranked)))
    groups = {"narrative": group} if ranked else {}
    return types.SimpleNamespace(
        evidence=types.SimpleNamespace(groups=groups),
        excel_rows=(),
        verified_excels=answers,
        verified_excel=answers[0] if answers else None,
    )


def _request():
    chunk = Chunk(
        source_id="s",
        chunk_id="c1",
        content="text",
        metadata=ChunkMetadata(source_file="f.pdf"),
    )
    return AnswerRequest(request_id="r", question="q?", chunks=(chunk,))


def _response(*, insufficient: bool):
    return types.SimpleNamespace(
        insufficient_context=insufficient, timings=None, answer="a"
    )


class RefusalReviewTests(unittest.TestCase):
    def _service(self, responses):
        answer_service = MagicMock()
        answer_service.answer.side_effect = responses
        return GenerationService(answer_service=answer_service), answer_service

    def test_verified_rows_contradict_a_refusal_and_trigger_one_review(self):
        # ``_review_refusal`` receives the first answer and makes exactly one
        # call of its own, so the mock supplies only the review's response.
        service, answer_service = self._service([_response(insufficient=False)])
        result = service._review_refusal(
            _request(), _response(insufficient=True), _bundle(verified_rows=9, ranked=0)
        )
        self.assertFalse(result.insufficient_context)
        self.assertEqual(answer_service.answer.call_count, 1)
        notice = answer_service.answer.call_args[0][0].evidence_notice
        self.assertIn("9 execution-verified workbook rows", notice)

    def test_an_empty_record_leaves_the_refusal_alone(self):
        service, answer_service = self._service([_response(insufficient=False)])
        original = _response(insufficient=True)
        result = service._review_refusal(
            _request(), original, _bundle(verified_rows=0, ranked=0)
        )
        self.assertIs(result, original)
        answer_service.answer.assert_not_called()

    def test_thin_text_evidence_alone_leaves_the_refusal_alone(self):
        service, answer_service = self._service([_response(insufficient=False)])
        original = _response(insufficient=True)
        result = service._review_refusal(
            _request(), original, _bundle(verified_rows=0, ranked=2)
        )
        self.assertIs(result, original)
        answer_service.answer.assert_not_called()

    def test_a_second_refusal_is_accepted_rather_than_re_asked(self):
        service, answer_service = self._service([_response(insufficient=True)])
        original = _response(insufficient=True)
        result = service._review_refusal(
            _request(), original, _bundle(verified_rows=9, ranked=0)
        )
        self.assertIs(result, original)
        self.assertEqual(answer_service.answer.call_count, 1)

    def test_the_review_never_reviews_its_own_review(self):
        service, answer_service = self._service([])
        request = replace(_request(), evidence_notice="already reviewed")
        original = _response(insufficient=True)
        result = service._review_refusal(
            request, original, _bundle(verified_rows=9, ranked=0)
        )
        self.assertIs(result, original)
        answer_service.answer.assert_not_called()


if __name__ == "__main__":
    unittest.main()


class LanePromptBudgetTests(unittest.TestCase):
    """Rows earn the wide budget; a descriptive card does not.

    Keying on the presence of an ``excel_card`` sent narrative questions down
    the workbook path because retrieval had surfaced one card among twenty
    passages, and they were answered at the wide budget they do not want.
    """

    def _budget(self, *, verified_rows=0, row_slices=0, excel_group=0):
        from services.generation_service import (
            _PDF_PROMPT_TOKEN_BUDGET,
            _lane_prompt_budget,
        )

        bundle = _bundle(verified_rows=verified_rows, ranked=4)
        bundle.excel_rows = tuple(range(row_slices))
        if excel_group:
            bundle.evidence.groups["excel"] = types.SimpleNamespace(
                results=tuple(range(excel_group))
            )
        return _lane_prompt_budget(bundle), _PDF_PROMPT_TOKEN_BUDGET

    def test_executed_rows_take_the_provider_ceiling(self):
        budget, _ = self._budget(verified_rows=128)
        self.assertIsNone(budget)

    def test_a_row_window_also_takes_the_ceiling(self):
        budget, _ = self._budget(row_slices=3)
        self.assertIsNone(budget)

    def test_a_card_without_rows_stays_on_the_narrative_budget(self):
        budget, pdf_budget = self._budget(excel_group=5)
        self.assertEqual(budget, pdf_budget)

    def test_narrative_only_stays_on_the_narrative_budget(self):
        budget, pdf_budget = self._budget()
        self.assertEqual(budget, pdf_budget)

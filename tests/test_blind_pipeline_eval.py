from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from eval.blind_pipeline_eval import (
    build_blind_cases,
    pending_cases,
    select_cases,
    stratified_sample_indices,
)


class StratifiedSamplingTests(unittest.TestCase):
    def test_sample_is_reproducible_and_spans_every_bin(self) -> None:
        indices = stratified_sample_indices(14, 10, seed=20260817)

        self.assertEqual(indices, [0, 1, 2, 4, 5, 7, 8, 9, 11, 12])
        for bin_index, selected in enumerate(indices):
            self.assertGreaterEqual(selected, bin_index * 14 // 10)
            self.assertLess(selected, (bin_index + 1) * 14 // 10)

    def test_sample_rejects_more_bins_than_questions(self) -> None:
        with self.assertRaisesRegex(ValueError, "sample_size"):
            stratified_sample_indices(3, 4, seed=1)


class BlindCaseTests(unittest.TestCase):
    def test_manifest_never_copies_reference_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            user_path = root / "users.json"
            beta_path = root / "beta.json"
            user_path.write_text(
                json.dumps(
                    [
                        {
                            "id": f"user_{index}",
                            "original_number": str(index),
                            "category": "pdf",
                            "question": f"user question {index}",
                        }
                        for index in range(1, 18)
                    ]
                ),
                encoding="utf-8",
            )
            beta_path.write_text(
                json.dumps(
                    {
                        "questions": [
                            {
                                "id": f"beta_{index}",
                                "category": "pdf",
                                "question": f"beta question {index}",
                                "golden_answer": f"SECRET_ANSWER_{index}",
                                "expected_response_behavior": "SECRET_BEHAVIOR",
                                "verification": {"secret": True},
                            }
                            for index in range(14)
                        ]
                    }
                ),
                encoding="utf-8",
            )

            cases = build_blind_cases(
                user_path,
                beta_path,
                user_limit=16,
                beta_sample_size=10,
                seed=20260817,
            )

            serialized = json.dumps(cases)
            self.assertEqual(len(cases), 26)
            self.assertNotIn("SECRET", serialized)
            self.assertEqual(
                set(cases[0]),
                {"case_id", "source", "source_index", "category", "question"},
            )

    def test_pending_cases_skips_only_successful_case_ids(self) -> None:
        cases = [
            {"case_id": "one"},
            {"case_id": "two"},
            {"case_id": "three"},
        ]
        results = [
            {"case_id": "one", "status": "success"},
            {"case_id": "two", "status": "error"},
        ]

        self.assertEqual(
            [case["case_id"] for case in pending_cases(cases, results)],
            ["two", "three"],
        )

    def test_select_cases_can_limit_an_integrated_rerun_to_beta(self) -> None:
        cases = [
            {"case_id": "user", "source": "user_questions"},
            {"case_id": "beta", "source": "beta_golden_questions"},
        ]

        self.assertEqual(
            select_cases(cases, "beta_golden_questions", None),
            [cases[1]],
        )

    def test_select_cases_can_target_specific_case_ids(self) -> None:
        cases = [
            {"case_id": "one", "source": "beta_golden_questions"},
            {"case_id": "two", "source": "beta_golden_questions"},
            {"case_id": "three", "source": "user_questions"},
        ]

        self.assertEqual(
            select_cases(cases, None, ("two", "three")),
            [cases[1], cases[2]],
        )


if __name__ == "__main__":
    unittest.main()

import unittest

from generation.multistep import PlannedSubanswer, protect_synthesis_coverage
from generation.planning import Requirement
from generation.schemas import ModelAnswer


class MultistepSafetyTests(unittest.TestCase):
    def test_synthesis_cannot_clear_subanswer_insufficient_context(self):
        subanswer = ModelAnswer(
            answer="The retrieved evidence covers only 2024.",
            cited_chunk_ids=("660",),
            insufficient_context=True,
            answered_requirements=("2024 inspection count",),
            missing_requirements=("2022-2023 and 2025 inspection counts",),
        )
        synthesis = ModelAnswer(
            answer="The inspection count was reported.",
            cited_chunk_ids=("660",),
            insufficient_context=False,
            answered_requirements=("inspection count",),
            missing_requirements=(),
        )

        protected = protect_synthesis_coverage(
            synthesis,
            (Requirement("R1", "inspection counts"),),
            (PlannedSubanswer(("R1",), subanswer),),
        )

        self.assertTrue(protected.insufficient_context)
        self.assertEqual(
            protected.missing_requirements,
            ("R1",),
        )

    def test_synthesis_can_remain_sufficient_when_all_steps_are_sufficient(self):
        subanswers = (
            ModelAnswer("A", ("1",), False),
            ModelAnswer("B", ("2",), False),
        )
        synthesis = ModelAnswer(
            "A and B", ("1", "2"), False,
            answered_requirements=("R1", "R2"),
        )

        protected = protect_synthesis_coverage(
            synthesis,
            (Requirement("R1", "A"), Requirement("R2", "B")),
            (
                PlannedSubanswer(("R1",), subanswers[0]),
                PlannedSubanswer(("R2",), subanswers[1]),
            ),
        )

        self.assertFalse(protected.insufficient_context)
        self.assertEqual(protected.missing_requirements, ())

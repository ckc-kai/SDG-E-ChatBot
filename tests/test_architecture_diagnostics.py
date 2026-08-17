import json
import tempfile
import unittest
from pathlib import Path

from eval.build_architecture_diagnostics import build_diagnostics


class ArchitectureDiagnosticBuilderTests(unittest.TestCase):
    def test_builds_three_frozen_24_case_suites_and_blind_projection(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            manifest = build_diagnostics(output)
            self.assertEqual(
                manifest["counts"],
                {
                    "excel_execution": 24,
                    "cross_resource_computation": 24,
                    "modality_gating": 24,
                },
            )
            blind = [
                json.loads(line)
                for line in (output / "blind/cross_resource_computation.jsonl")
                .read_text()
                .splitlines()
            ]
        self.assertEqual(set(blind[0]), {"id", "question"})


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from eval.run_eval import load_eval


class EvaluationLoadingTests(unittest.TestCase):
    def test_load_eval_reads_utf8_jsonl_on_windows(self) -> None:
        row = {
            "id": "utf8",
            "question": "What does “projected” mean?",
            "evidence": [],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evaluation.jsonl"
            path.write_text(
                json.dumps(row, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            self.assertEqual(load_eval(path)[0], row)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from generation.providers.dsh import DshPlannerProvider


class DshPlannerProviderTests(unittest.TestCase):
    def test_structured_call_uses_isolated_runner_and_returns_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "scripts").mkdir()
            (root / "scripts" / "run_dsh_planner.py").touch()
            sdk = root / "sdk"
            sdk.mkdir()
            provider = DshPlannerProvider(
                model="qwen3:14b",
                max_tokens=1400,
                project_root=root,
                environ={"DSH_SDK_PATH": str(sdk), "PATH": "/usr/bin"},
            )
            envelope = json.dumps(
                {
                    "output": '{"requirements":[],"tasks":[]}',
                    "finish_reason": "completed",
                    "latency_ms": 123,
                }
            )
            completed = subprocess.CompletedProcess(
                args=[], returncode=0, stdout=envelope, stderr=""
            )

            with patch(
                "generation.providers.dsh.subprocess.run", return_value=completed
            ) as run:
                output = provider.generate_structured(
                    "plan this", {"type": "object"}
                )

            self.assertEqual(output, '{"requirements":[],"tasks":[]}')
            self.assertEqual(provider.model_id, "dsh/qwen3:14b")
            self.assertEqual(provider.last_finish_reason, "completed")
            payload = json.loads(run.call_args.kwargs["input"])
            self.assertEqual(payload["model"], "qwen3:14b")
            self.assertEqual(payload["max_tokens"], 1400)
            self.assertIn(str(sdk), run.call_args.kwargs["env"]["PYTHONPATH"])

    def test_dsh_model_has_a_separate_environment_setting(self):
        provider = DshPlannerProvider.from_env(
            environ={
                "TASK3_PLANNER_MODEL": "qwen3:4b",
                "DSH_PLANNER_MODEL": "qwen3:14b",
            }
        )

        self.assertEqual(provider.model, "qwen3:14b")

    def test_dsh_endpoint_is_independent_from_final_answer_endpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "scripts").mkdir()
            (root / "scripts" / "run_dsh_planner.py").touch()
            (root / ".dsh-python").mkdir()
            provider = DshPlannerProvider(
                project_root=root,
                environ={
                    "DEEPSEEK_BASE_URL": "https://api.deepseek.com",
                    "PATH": "/usr/bin",
                },
            )
            completed = subprocess.CompletedProcess(
                args=[], returncode=0, stdout='{"output":"{}"}', stderr=""
            )

            with patch(
                "generation.providers.dsh.subprocess.run",
                return_value=completed,
            ) as run:
                provider.generate_structured("plan this", {"type": "object"})

            environment = run.call_args.kwargs["env"]
            self.assertEqual(
                environment["DSH_BASE_URL"],
                "http://127.0.0.1:11434/v1",
            )
            self.assertEqual(
                environment["DEEPSEEK_BASE_URL"],
                "https://api.deepseek.com",
            )


if __name__ == "__main__":
    unittest.main()

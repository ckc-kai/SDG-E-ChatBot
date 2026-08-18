import unittest

from eval.environment import evaluation_environment


class EvaluationEnvironmentTests(unittest.TestCase):
    def test_required_performance_fields_are_always_present(self):
        metadata = evaluation_environment(
            model_name="qwen3.5:9b", backend="metal_ollama", context_limit=4096
        )
        self.assertEqual(metadata["environment_id"], "local_mps")
        self.assertEqual(metadata["model_name"], "qwen3.5:9b")
        self.assertEqual(metadata["backend"], "metal_ollama")
        for key in (
            "machine_model",
            "chip_or_gpu",
            "total_memory",
            "quantization",
            "context_limit",
            "batch_or_concurrency",
            "software_versions",
        ):
            self.assertIn(key, metadata)


if __name__ == "__main__":
    unittest.main()

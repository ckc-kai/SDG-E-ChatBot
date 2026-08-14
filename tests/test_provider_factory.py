from __future__ import annotations

import unittest

from generation.providers import GroqProvider, OllamaProvider, ProviderError, create_provider_from_env


class ProviderFactoryTests(unittest.TestCase):
    def test_selects_ollama_from_environment(self) -> None:
        provider = create_provider_from_env(
            environ={"TASK3_PROVIDER": "ollama", "OLLAMA_MODEL": "qwen3:4b"}
        )
        self.assertIsInstance(provider, OllamaProvider)
        self.assertEqual(provider.model_id, "ollama/qwen3:4b")

    def test_explicit_name_overrides_environment_selection(self) -> None:
        provider = create_provider_from_env(
            "ollama",
            environ={"TASK3_PROVIDER": "unknown", "OLLAMA_MODEL": "qwen3:4b"},
        )
        self.assertIsInstance(provider, OllamaProvider)

    def test_selects_groq_from_environment_without_calling_api(self) -> None:
        provider = create_provider_from_env(
            environ={"TASK3_PROVIDER": "groq", "GROQ_API_KEY": "test-key"}
        )
        self.assertIsInstance(provider, GroqProvider)
        self.assertEqual(provider.model_id, "groq/openai/gpt-oss-120b")

    def test_missing_selection_is_rejected(self) -> None:
        with self.assertRaisesRegex(ProviderError, "TASK3_PROVIDER is required"):
            create_provider_from_env(environ={})

    def test_unknown_selection_is_rejected(self) -> None:
        with self.assertRaisesRegex(ProviderError, "Unsupported TASK3_PROVIDER"):
            create_provider_from_env(environ={"TASK3_PROVIDER": "other"})


if __name__ == "__main__":
    unittest.main()

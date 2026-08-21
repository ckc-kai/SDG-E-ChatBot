from __future__ import annotations

import unittest
from typing import Any

from generation.providers.base import ProviderError
from generation.providers.bedrock import BedrockProvider, BedrockUsage


class FakeBedrockClient:
    def __init__(
        self,
        response: dict[str, Any] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.response = response or {
            "output": {
                "message": {
                    "content": [
                        {
                            "text": (
                                '{"answer":"Supported","cited_chunk_ids":["1"],'
                                '"insufficient_context":false}'
                            )
                        }
                    ]
                }
            },
            "usage": {"inputTokens": 100, "outputTokens": 20, "totalTokens": 120},
            "metrics": {"latencyMs": 250},
        }
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def converse(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.response


class BedrockProviderTests(unittest.TestCase):
    def test_generate_uses_converse_and_records_usage(self) -> None:
        client = FakeBedrockClient()
        provider = BedrockProvider(
            client,
            "amazon.nova-lite-v1:0",
            max_tokens=300,
            temperature=0,
        )

        raw = provider.generate("grounded prompt")

        self.assertIn('"answer":"Supported"', raw)
        self.assertEqual(
            client.calls,
            [
                {
                    "modelId": "amazon.nova-lite-v1:0",
                    "messages": [
                        {
                            "role": "user",
                            "content": [{"text": "grounded prompt"}],
                        }
                    ],
                    "inferenceConfig": {"maxTokens": 300, "temperature": 0},
                }
            ],
        )
        self.assertEqual(provider.last_usage, BedrockUsage(100, 20, 120, 250))
        self.assertEqual(provider.capabilities.context_window, 4096)
        self.assertEqual(provider.capabilities.prompt_token_budget, 3796)

    def test_multiple_text_blocks_are_joined(self) -> None:
        client = FakeBedrockClient(
            response={
                "output": {
                    "message": {
                        "content": [{"text": '{"answer":'}, {"text": '"ok"}'}]
                    }
                }
            }
        )
        provider = BedrockProvider(client, "model")
        self.assertEqual(provider.generate("prompt"), '{"answer":"ok"}')

    def test_client_failure_becomes_provider_error_without_exposing_detail(self) -> None:
        client = FakeBedrockClient(error=TimeoutError("private AWS detail"))
        provider = BedrockProvider(client, "model")
        with self.assertRaisesRegex(ProviderError, "Converse request failed") as captured:
            provider.generate("prompt")
        self.assertNotIn("private AWS detail", str(captured.exception))

    def test_missing_response_text_is_rejected(self) -> None:
        provider = BedrockProvider(
            FakeBedrockClient(response={"output": {"message": {"content": []}}}),
            "model",
        )
        with self.assertRaisesRegex(ProviderError, "contains no text"):
            provider.generate("prompt")

    def test_from_env_uses_injected_client_without_boto3(self) -> None:
        client = FakeBedrockClient()
        provider = BedrockProvider.from_env(
            environ={
                "AWS_REGION": "us-east-1",
                "BEDROCK_MODEL_ID": "amazon.nova-lite-v1:0",
                "BEDROCK_MAX_TOKENS": "250",
                "BEDROCK_TEMPERATURE": "0",
            },
            client=client,
        )
        self.assertEqual(provider.model_id, "amazon.nova-lite-v1:0")
        self.assertEqual(provider.max_tokens, 250)
        self.assertEqual(provider.temperature, 0)

    def test_from_env_requires_model_id(self) -> None:
        with self.assertRaisesRegex(ProviderError, "BEDROCK_MODEL_ID is required"):
            BedrockProvider.from_env(environ={}, client=FakeBedrockClient())

    def test_invalid_configuration_is_rejected(self) -> None:
        with self.assertRaisesRegex(ProviderError, "Invalid Bedrock"):
            BedrockProvider.from_env(
                environ={"BEDROCK_MODEL_ID": "model", "BEDROCK_MAX_TOKENS": "many"},
                client=FakeBedrockClient(),
            )


if __name__ == "__main__":
    unittest.main()

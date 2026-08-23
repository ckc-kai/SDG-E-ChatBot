"""Shared model-capability and provider-quota metadata."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
import re
import time


@dataclass(frozen=True)
class ModelCapabilities:
    """Model hard limits plus the application's operational input budget."""

    context_window: int
    max_output_tokens: int
    requested_output_tokens: int
    prompt_token_budget: int

    def __post_init__(self) -> None:
        if min(
            self.context_window,
            self.max_output_tokens,
            self.requested_output_tokens,
            self.prompt_token_budget,
        ) <= 0:
            raise ValueError("model capability limits must be positive")
        if self.requested_output_tokens > self.max_output_tokens:
            raise ValueError("requested output exceeds the model output limit")
        if self.prompt_token_budget + self.requested_output_tokens > self.context_window:
            raise ValueError("prompt and output budgets exceed the model context window")


@dataclass(frozen=True)
class RateLimitState:
    """Latest provider quota snapshot; missing headers remain unknown."""

    request_limit: int | None = None
    remaining_requests: int | None = None
    token_limit: int | None = None
    remaining_tokens: int | None = None
    request_reset: str | None = None
    token_reset: str | None = None
    retry_after_seconds: float | None = None
    captured_at_monotonic: float | None = None

    def available_tokens(self) -> int | None:
        if self._window_has_reset(self.token_reset):
            return None
        return self.remaining_tokens

    def available_requests(self) -> int | None:
        if self._window_has_reset(self.request_reset):
            return None
        return self.remaining_requests

    def seconds_until_token_reset(self) -> float | None:
        """Return the remaining token-window wait, or zero after it resets."""
        reset_seconds = _duration_seconds(self.token_reset)
        if reset_seconds is None or self.captured_at_monotonic is None:
            return None
        elapsed = time.monotonic() - self.captured_at_monotonic
        return max(0.0, reset_seconds - elapsed)

    def _window_has_reset(self, duration: str | None) -> bool:
        reset_seconds = _duration_seconds(duration)
        if reset_seconds is None or self.captured_at_monotonic is None:
            return False
        return time.monotonic() - self.captured_at_monotonic >= reset_seconds


def rate_limit_state_from_headers(headers: Mapping[str, str]) -> RateLimitState:
    normalized = {str(key).casefold(): str(value) for key, value in headers.items()}
    return RateLimitState(
        request_limit=_optional_int(normalized.get("x-ratelimit-limit-requests")),
        remaining_requests=_optional_int(
            normalized.get("x-ratelimit-remaining-requests")
        ),
        token_limit=_optional_int(normalized.get("x-ratelimit-limit-tokens")),
        remaining_tokens=_optional_int(
            normalized.get("x-ratelimit-remaining-tokens")
        ),
        request_reset=normalized.get("x-ratelimit-reset-requests"),
        token_reset=normalized.get("x-ratelimit-reset-tokens"),
        retry_after_seconds=_optional_float(normalized.get("retry-after")),
        captured_at_monotonic=time.monotonic(),
    )


def _optional_int(value: str | None) -> int | None:
    try:
        return int(value) if value is not None else None
    except ValueError:
        return None


def _optional_float(value: str | None) -> float | None:
    try:
        return float(value) if value is not None else None
    except ValueError:
        return None


_DURATION_PART_RE = re.compile(r"(?P<value>\d+(?:\.\d+)?)(?P<unit>[hms])")


def _duration_seconds(value: str | None) -> float | None:
    if not value:
        return None
    multipliers = {"h": 3600.0, "m": 60.0, "s": 1.0}
    parts = list(_DURATION_PART_RE.finditer(value.strip().casefold()))
    if not parts or "".join(match.group(0) for match in parts) != value.strip().casefold():
        return None
    return sum(
        float(match.group("value")) * multipliers[match.group("unit")]
        for match in parts
    )

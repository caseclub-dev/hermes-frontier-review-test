"""Transport layer — ABC, MockAnthropicTransport, and RealAnthropicTransport stub.

Design: every transport implements the same interface so AnthropicClient
can swap transports without knowing which one it uses.

Mock mode never falls through to the real API.
"""

from __future__ import annotations

import json
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional


# ── Exception hierarchy ──────────────────────────────────────────────

class TransportError(Exception):
    """Base exception for transport errors."""


class MockServerError(TransportError):
    """Raised when the mock server encounters an internal error."""


class MockRateLimitError(TransportError):
    """Raised when the API rate limit is exceeded (mock)."""


class MockTimeoutError(TransportError):
    """Raised when a request times out (mock or real)."""


class RealApiDisabledError(TransportError):
    """Raised when code tries to call the real API in offline mode."""


# ── Transport ABC ────────────────────────────────────────────────────

class BaseTransport(ABC):
    """Abstract base class for all transports.

    Both MockAnthropicTransport and RealAnthropicTransport must implement
    these three methods with identical signatures.
    """

    @abstractmethod
    def send_request(
        self,
        url: str,
        method: str,
        headers: dict[str, str],
        body: bytes | str,
        timeout: int = 300,
    ) -> bytes:
        """Send an HTTP request and return the raw response body as bytes."""

    @abstractmethod
    def receive_response(self, raw_bytes: bytes) -> dict[str, Any]:
        """Parse the raw response bytes into a structured dict."""


# ── MockAnthropicTransport ───────────────────────────────────────────

@dataclass
class ScenarioConfig:
    """Configuration for a single mock HTTP response scenario.

    Attributes:
        status_code: HTTP status code to return.
        headers: Response headers dict.
        _raw_body: Pre-formed JSON body string (bypasses re-encoding).
        input_tokens: Simulated input token count.
        output_tokens: Simulated output token count.
        thinking_tokens: Simulated thinking token count.
        cache_creation_tokens: Simulated cache creation tokens.
        cache_read_tokens: Simulated cache read tokens.
    """

    status_code: int = 200
    headers: dict[str, str] = field(default_factory=dict)
    _raw_body: Optional[str] = None
    input_tokens: int = 15000
    output_tokens: int = 2000
    thinking_tokens: int = 0
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0


class MockAnthropicTransport(BaseTransport):
    """Simulated Anthropic API transport for offline testing.

    Supports single-scenario configuration and sequential scenario queues.
    All method signatures match the real transport interface.

    Mock mode NEVER falls through to a real API call — send_request
    only returns pre-configured responses or raises mock errors.
    """

    def __init__(self) -> None:
        self._configured_scenario: Optional[ScenarioConfig] = None
        self._sequence: list[ScenarioConfig] = []
        self._sequence_index: int = 0

    # ── Public configuration API ────────────────────────────────────

    def configure(self, scenario: ScenarioConfig) -> None:
        """Configure a single-scenario mock response."""
        self._configured_scenario = scenario
        self._sequence = [scenario]
        self._sequence_index = 0

    def configure_sequence(self, scenarios: list[ScenarioConfig]) -> None:
        """Configure a sequence of responses (consumed in order)."""
        self._configured_scenario = None
        self._sequence = list(scenarios)
        self._sequence_index = 0

    # ── Transport interface ─────────────────────────────────────────

    def send_request(
        self,
        url: str,
        method: str,
        headers: dict[str, str],
        body: bytes | str,
        timeout: int = 300,
    ) -> bytes:
        """Send a request and return the pre-configured response body as bytes.

        In single-scenario mode, the scenario is consumed on first call and
        subsequent calls raise MockServerError (sequence exhausted).
        No network access occurs.
        """
        if self._configured_scenario is not None:
            # Single-scenario: consume it and fall through to sequence check
            raw = self._configured_scenario._raw_body
            consumed_raw = raw.encode("utf-8") if raw else b""
            self._configured_scenario = None
            self._sequence_index = 1  # mark the one scenario as consumed in sequence too
            return consumed_raw

        # Sequential mode — consume next scenario in queue
        if self._sequence_index >= len(self._sequence):
            raise MockServerError(
                "HTTP 500: Server Error — mock sequence exhausted"
            )

        scenario = self._sequence[self._sequence_index]
        self._sequence_index += 1

        raw = scenario._raw_body
        return raw.encode("utf-8") if raw else b""

    def receive_response(self, raw_bytes: bytes) -> dict[str, Any]:
        """Parse the response JSON — exercises the same parser as real client."""
        try:
            resp = json.loads(raw_bytes.decode("utf-8"))
            return resp
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise TransportError(f"Failed to parse response: {exc}") from exc

    # ── Scenario builders ───────────────────────────────────────────

    @staticmethod
    def ok_response(
        model: str = "claude-opus-5-5",
        status: str = "pass",
        summary: str = "Review completed.",
        findings: list[dict[str, Any]] | None = None,
        input_tokens: int = 15000,
        output_tokens: int = 2000,
    ) -> ScenarioConfig:
        """Create a successful API response scenario.

        The response body mimics the structure Anthropic's Messages API
        would return for a structured-review prompt.
        """
        if findings is None:
            findings = []

        # Build the inner review payload that matches _parse_json_response expectations
        review_payload = {
            "status": status,
            "summary": summary,
            "findings": findings,
            "requirements_met": [],
            "blockers": [] if status != "blocked" else ["reason this cannot proceed"],
            "suggestions": ["improvement recommendations"],
        }

        body = {
            "id": f"msg-{uuid.uuid4().hex[:8]}",
            "model": model,
            "content": [
                {"type": "text", "text": json.dumps(review_payload)}
            ],
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
                "thinking_tokens": 0,
                "cache_creation_tokens": 0,
                "cache_read_tokens": 0,
            },
        }

        return ScenarioConfig(
            status_code=200,
            headers={"content-type": "application/json"},
            _raw_body=json.dumps(body),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    @staticmethod
    def server_error_response(message: str = "Internal server error") -> ScenarioConfig:
        """Return a config for a 500 error scenario."""
        return ScenarioConfig(
            status_code=500,
            headers={"content-type": "application/json"},
            _raw_body=json.dumps({
                "error": {"type": "api_error", "message": message},
            }),
        )


# ── RealAnthropicTransport (stub) ───────────────────────────────────

class RealAnthropicTransport(BaseTransport):
    """Real Anthropic API transport — stub for future implementation.

    In M1–M5 this transport is DISABLED. Instantiating it without an
    explicit enable flag will raise RealApiDisabledError on any call.

    When implemented in a later phase, it will:
    - Read ANTHROPIC_API_KEY from environment or vault
    - Make actual POST requests to https://api.anthropic.com/v1/messages
    - Return raw response bytes via send_request
    - Parse JSON responses via receive_response
    """

    def __init__(self, api_key: Optional[str] = None) -> None:
        self._api_key = api_key
        # Fail-closed: real transport is not usable without explicit enable
        raise RealApiDisabledError(
            "Real API transport is disabled in M1. Use MockAnthropicTransport."
        )

    def send_request(
        self,
        url: str,
        method: str,
        headers: dict[str, str],
        body: bytes | str,
        timeout: int = 300,
    ) -> bytes:
        raise RealApiDisabledError("Real API transport is disabled in offline mode.")

    def receive_response(self, raw_bytes: bytes) -> dict[str, Any]:
        try:
            return json.loads(raw_bytes.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise TransportError(f"Failed to parse response: {exc}") from exc

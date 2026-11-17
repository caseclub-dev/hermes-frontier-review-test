"""AnthropicClient — unified interface for review requests (real + mock).

Transport resolution order:
1. If a MockAnthropicTransport is injected → use it (regardless of offline mode)
2. If RealAnthropicTransport is injected → use it (explicit opt-in only)
3. If neither and real API is not enabled → raise an error (fail-closed)

The client exercises the full request-building path even for mock transport,
so response parsing and post-processing are tested against production code.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from .transport import (
    BaseTransport,
    MockAnthropicTransport,
    RealApiDisabledError,
    TransportError,
)

logger = logging.getLogger(__name__)


class AnthropicClient:
    """Client for the Anthropic Messages API.

    Supports transport injection for offline testing without network access.
    When used without an injected transport and real API is disabled,
    all calls fail fast with RealApiDisabledError (fail-closed).
    """

    def __init__(
        self,
        model: str = "claude-opus-5-5",
        timeout_s: int = 300,
    ) -> None:
        from .config import ALLOWED_MODEL, validate_model

        if not validate_model(model):
            raise ValueError(
                f"Model '{model}' is not in the allowed model list "
                f"(only '{ALLOWED_MODEL}' is permitted)."
            )

        self.model = model
        self.timeout_s = timeout_s
        self._transport: Optional[BaseTransport] = None

    @property
    def transport(self) -> Optional[BaseTransport]:
        """The currently injected transport, or None."""
        return self._transport

    def set_transport(self, transport: BaseTransport) -> None:
        """Inject a transport (mock or real)."""
        if not isinstance(transport, BaseTransport):
            raise TypeError(f"Expected BaseTransport subclass, got {type(transport)}")
        self._transport = transport

    # ── Core chat method ──────────────────────────────────────────────

    def chat(
        self,
        system_prompt: str,
        user_message: str,
        stop_sequences: Optional[list[str]] = None,
    ) -> tuple[str, dict[str, int]]:
        """Send a review request and return (response_text, usage_dict).

        Transport resolution:
        1. Injected mock transport → use it (no network)
        2. No transport + real API disabled → error (fail-closed)
        3. Real transport injected → make HTTP call
        """
        if self._transport is not None:
            return self._chat_via_transport(system_prompt, user_message, stop_sequences)

        raise RealApiDisabledError(
            "No transport injected and real API is disabled in offline mode. "
            "Inject a MockAnthropicTransport for testing."
        )

    def _chat_via_transport(
        self,
        system_prompt: str,
        user_message: str,
        stop_sequences: Optional[list[str]] = None,
    ) -> tuple[str, dict[str, int]]:
        """Route through the injected transport — exercises full code path."""
        body = self._build_request_body(system_prompt, user_message, stop_sequences)

        try:
            raw_bytes = self._transport.send_request(
                url="https://api.anthropic.com/v1/messages",
                method="POST",
                headers={"content-type": "application/json"},
                body=body,
                timeout=self.timeout_s,
            )

            resp = self._transport.receive_response(raw_bytes)

            # Extract text content from response
            text = self._extract_text(resp)

            # Extract usage from mock response
            usage_dict = self._extract_usage(resp)

            # ── Response model drift check ────────────────────────
            self._check_model_drift(resp.get("model", ""))

            return text, usage_dict

        except TransportError:
            raise
        except Exception as exc:
            logger.error(f"Unexpected transport error: {exc}")
            raise

    # ── Request building ──────────────────────────────────────────────

    def _build_request_body(
        self,
        system_prompt: str,
        user_message: str,
        stop_sequences: Optional[list[str]] = None,
    ) -> str:
        """Build the JSON request body for Claude Messages API.

        This matches Anthropic's actual API structure — both mock and real
        transports exercise this method, so any bugs affect production code.
        """
        messages = [{"role": "user", "content": user_message}]

        body: dict[str, Any] = {
            "model": self.model,
            "max_tokens": 32768,
            "system": system_prompt,
            "messages": messages,
        }

        if stop_sequences:
            body["stop_sequences"] = stop_sequences

        # Add thinking params for opus models
        thinking_param = self._build_thinking_param()
        if thinking_param is not None:
            body["thinking"] = thinking_param

        return json.dumps(body)

    def _build_thinking_param(self) -> dict[str, str] | None:
        """Return thinking block params when using an opus model."""
        if "opus" in self.model.lower():
            return {"type": "adaptive"}
        return None

    # ── Response helpers ──────────────────────────────────────────────

    def _extract_text(self, response: dict[str, Any]) -> str:
        """Extract text content from an API response."""
        content_blocks = response.get("content", [])
        text_parts: list[str] = []
        for block in content_blocks:
            if isinstance(block, dict):
                text_parts.append(block.get("text", ""))
            elif isinstance(block, str):
                text_parts.append(block)
        return "".join(text_parts)

    def _extract_usage(self, response: dict[str, Any]) -> dict[str, int]:
        """Extract token usage from an API response."""
        usage_info = response.get("usage", {})
        return {
            "input_tokens": usage_info.get("input_tokens", 0),
            "output_tokens": usage_info.get("output_tokens", 0),
            "total_tokens": usage_info.get("total_tokens", 0),
            "thinking_tokens": usage_info.get("thinking_tokens", 0),
            "cache_creation_tokens": usage_info.get("cache_creation_tokens", 0),
            "cache_read_tokens": usage_info.get("cache_read_tokens", 0),
        }

    def _check_model_drift(self, response_model: str) -> None:
        """Raise RuntimeError if the API returned a different model than requested."""
        from .config import ALLOWED_MODEL

        if response_model != ALLOWED_MODEL:
            raise RuntimeError(
                f"Model drift detected: expected '{ALLOWED_MODEL}' "
                f"but got '{response_model}'. This may indicate unsafe behavior."
            )

    def health_check(self) -> dict[str, Any]:
        """Lightweight health check."""
        if self._transport is not None:
            return {"status": "ok", "model": self.model, "transport": "mock"}
        return {
            "status": "offline",
            "model": self.model,
            "transport": "none",
            "note": "No transport injected; real API disabled.",
        }

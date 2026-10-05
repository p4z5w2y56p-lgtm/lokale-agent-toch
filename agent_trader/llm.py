"""Provider adapters. Each is a callable (system, user) -> text, so the agent does not
care whether the brain is Claude or Gemini. Keys are passed in by the caller (read from
the environment), never hard-coded, and never put in a URL."""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable


class LLMError(Exception):
    pass


@dataclass
class AnthropicLLM:
    client: Any                      # anthropic.Anthropic() or a fake
    model: str = "claude-haiku-4-5-20251001"
    max_tokens: int = 600
    usage: Any = None                # optional UsageTracker: logs cost, hard-stops at the cap
    tag: str = ""

    def __call__(self, system: str, user: str) -> str:
        if self.usage is not None:
            self.usage.check()
        msg = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        if self.usage is not None and getattr(msg, "usage", None) is not None:
            self.usage.record(self.model, msg.usage.input_tokens, msg.usage.output_tokens, self.tag)
        return "".join(getattr(b, "text", "") for b in msg.content)


GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


def _urlopen_json(req: urllib.request.Request) -> dict:
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.load(resp)
    except urllib.error.HTTPError as exc:
        # Do not echo the body: it can contain request details. Status is enough.
        raise LLMError(f"Gemini HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise LLMError(f"Gemini network error: {exc.reason}") from exc


@dataclass
class GeminiLLM:
    api_key: str = ""
    model: str = "gemini-2.5-flash"   # override with GEMINI_MODEL if Google renames it
    max_tokens: int = 600
    post: Callable[[urllib.request.Request], dict] = _urlopen_json

    def __post_init__(self) -> None:
        if not self.api_key:
            raise LLMError("GEMINI_API_KEY is empty")

    def __call__(self, system: str, user: str) -> str:
        body = {
            "system_instruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {
                "maxOutputTokens": self.max_tokens,
                "temperature": 0.2,
                "responseMimeType": "application/json",
            },
        }
        req = urllib.request.Request(
            GEMINI_URL.format(model=self.model),
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json", "x-goog-api-key": self.api_key},
            method="POST",
        )
        data = self.post(req)
        try:
            parts = data["candidates"][0]["content"]["parts"]
            return "".join(p.get("text", "") for p in parts)
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError("Gemini returned no usable answer") from exc

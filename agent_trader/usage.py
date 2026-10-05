"""Usage tracker: every model call logs its tokens and estimated euro cost to a local
jsonl (no key, no prompt text), and refuses new calls once the budget cap is hit.

This is an estimate from the token counts the API returns. The real bill lives at
console.anthropic.com -> Settings -> Billing / Usage."""
from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

# USD per million tokens (input, output). Haiku 4.5 list price.
PRICES_USD = {
    "claude-haiku-4-5-20251001": (1.0, 5.0),
    "claude-sonnet-5-5": (3.0, 15.0),
    "claude-opus-5-5": (5.0, 25.0),
}
DEFAULT_PRICE_USD = (3.0, 15.0)   # unknown model: assume the pricier tier, never under-count


class BudgetExceeded(Exception):
    pass


def cost_eur(model: str, input_tokens: int, output_tokens: int, eur_per_usd: float) -> float:
    pin, pout = PRICES_USD.get(model, DEFAULT_PRICE_USD)
    return (input_tokens * pin + output_tokens * pout) / 1_000_000 * eur_per_usd


@dataclass
class UsageTracker:
    path: Path = Path("runs/usage.jsonl")
    budget_eur: float = 5.0
    eur_per_usd: float = 0.86
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _spent: float | None = field(default=None, init=False)

    @classmethod
    def from_env(cls) -> "UsageTracker":
        return cls(
            Path(os.environ.get("USAGE_LOG", "runs/usage.jsonl")),
            float(os.environ.get("BUDGET_EUR", "5")),
            float(os.environ.get("EUR_PER_USD", "0.86")),
        )

    def spent(self) -> float:
        with self._lock:
            return self._load()

    def _load(self) -> float:
        if self._spent is None:
            total = 0.0
            if self.path.exists():
                for line in self.path.read_text().splitlines():
                    try:
                        total += float(json.loads(line).get("eur", 0.0))
                    except (ValueError, AttributeError):
                        continue
            self._spent = total
        return self._spent

    def check(self) -> None:
        """Call before every request. Raises once the cap is reached."""
        spent = self.spent()
        if spent >= self.budget_eur:
            raise BudgetExceeded(f"budget cap reached: {spent:.3f} of {self.budget_eur:.2f} EUR")

    def record(self, model: str, input_tokens: int, output_tokens: int, tag: str = "") -> float:
        eur = cost_eur(model, input_tokens, output_tokens, self.eur_per_usd)
        row = {"ts": round(time.time(), 1), "model": model, "in": input_tokens,
               "out": output_tokens, "eur": round(eur, 6), "tag": tag}
        with self._lock:
            self._load()
            self._spent += eur
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a") as fh:
                fh.write(json.dumps(row) + "\n")
        return eur

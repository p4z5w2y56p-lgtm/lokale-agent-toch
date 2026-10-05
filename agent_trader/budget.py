"""Euro budget for LLM calls: records cost per call, warns at 50%/80%, stops at 100%."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Claude Haiku 4.5 list prices, USD per million tokens.
INPUT_USD_PER_M = 1.0
OUTPUT_USD_PER_M = 5.0
USD_EUR = 0.92
WARN_LEVELS = (0.5, 0.8)


class BudgetExceeded(Exception):
    pass


def cost_eur(input_tokens: int, output_tokens: int) -> float:
    usd = input_tokens / 1e6 * INPUT_USD_PER_M + output_tokens / 1e6 * OUTPUT_USD_PER_M
    return usd * USD_EUR


class BudgetMeter:
    def __init__(self, path: str | Path | None = None, cap_eur: float | None = None, out=None) -> None:
        self.path = Path(path or os.environ.get("AGENT_BUDGET_FILE", "runs/budget.json"))
        self.cap_eur = float(cap_eur if cap_eur is not None else os.environ.get("AGENT_BUDGET_EUR", "5.0"))
        self.out = out
        self.state = {"spent_eur": 0.0, "calls": 0, "input_tokens": 0, "output_tokens": 0, "warned": []}
        if self.path.exists():
            try:
                self.state.update(json.loads(self.path.read_text()))
            except (OSError, ValueError):
                pass

    @property
    def spent_eur(self) -> float:
        return float(self.state["spent_eur"])

    def check(self) -> None:
        """Raise before a call if the cap is already used up."""
        if self.cap_eur > 0 and self.spent_eur >= self.cap_eur:
            raise BudgetExceeded(f"LLM budget used up: EUR {self.spent_eur:.4f} of {self.cap_eur:.2f}")

    def record(self, input_tokens: int, output_tokens: int) -> float:
        cost = cost_eur(input_tokens, output_tokens)
        s = self.state
        s["spent_eur"] = self.spent_eur + cost
        s["calls"] += 1
        s["input_tokens"] += int(input_tokens)
        s["output_tokens"] += int(output_tokens)
        frac = s["spent_eur"] / self.cap_eur if self.cap_eur > 0 else 0.0
        for level in WARN_LEVELS:
            if frac >= level and level not in s["warned"]:
                s["warned"].append(level)
                print(f"WARNING: LLM budget {level:.0%} reached: EUR {s['spent_eur']:.4f} of "
                      f"{self.cap_eur:.2f}", file=self.out or sys.stderr)
        self._save()
        if self.cap_eur > 0 and frac >= 1.0:
            raise BudgetExceeded(f"LLM budget exceeded: EUR {s['spent_eur']:.4f} of {self.cap_eur:.2f}")
        return cost

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.state, indent=2))

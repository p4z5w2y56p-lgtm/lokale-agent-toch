"""Budget alerts: warn at 50% and 80% of the budget, hard-stop at 100%. Each fires once.

Interface to the usage meter (plug-in contract)
-----------------------------------------------
The usage meter writes a JSONL file, one JSON object per line, one line per model call.
This module only needs ONE field from each record:

    {"cost_eur": 0.0042, ...anything else (tokens, model, ts)...}

`cost_eur` is a non-negative number in euro. Blank lines, malformed lines and records
without a usable `cost_eur` are skipped and counted (see `UsageTotal.skipped`), never
fatal: a half-written last line must not crash the agent. A missing file means 0 spent.
If the meter lands with a different file name or field name, change `read_usage_total`'s
arguments (`cost_field=`), nothing else here needs to change. Alternatively skip the file
and hand `BudgetAlerts.check()` a total directly.

Alerts go to stdout and to an append-only JSONL alerts file (default runs/budget_alerts.jsonl).
The alerts file doubles as the memory of what already fired, so a restarted process does
not repeat itself. If the budget amount changes, alerts start over for the new amount.

The 100% level is a hard stop: `BudgetAlerts.check()` returns the alert and
`BudgetAlerts.exhausted(total)` is True from then on. Callers (paper trading, replay with an
LLM agent) must stop making paid calls when it is True. This is advice enforced by the
caller, not a kill switch; it cannot cancel a call already in flight.
"""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Mapping

DEFAULT_BUDGET_EUR = 5.0
DEFAULT_THRESHOLDS = (0.5, 0.8, 1.0)
DEFAULT_ALERTS_PATH = "runs/budget_alerts.jsonl"
DEFAULT_USAGE_PATH = "runs/usage.jsonl"   # assumption: where the usage meter writes


@dataclass(frozen=True)
class UsageTotal:
    cost_eur: float
    records: int
    skipped: int


def read_usage_total(path: str | Path, cost_field: str = "cost_eur") -> UsageTotal:
    """Sum `cost_field` over a JSONL usage log. Tolerant of junk lines; missing file = 0."""
    total, records, skipped = 0.0, 0, 0
    try:
        fh = open(path, encoding="utf-8")
    except FileNotFoundError:
        return UsageTotal(0.0, 0, 0)
    with fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)[cost_field]
                cost = float(value)
                if isinstance(value, bool) or not math.isfinite(cost) or cost < 0:
                    raise ValueError(value)
            except (ValueError, KeyError, TypeError):
                skipped += 1
                continue
            total += cost
            records += 1
    return UsageTotal(total, records, skipped)


@dataclass(frozen=True)
class Alert:
    ts: float
    level: str            # "info" (50%), "warning" (80%), "hard_stop" (100%)
    threshold: float      # 0.5 / 0.8 / 1.0
    budget_eur: float
    spent_eur: float
    message: str


def _level(threshold: float) -> str:
    if threshold >= 1.0:
        return "hard_stop"
    return "warning" if threshold >= 0.8 else "info"


class BudgetAlerts:
    def __init__(
        self,
        budget_eur: float = DEFAULT_BUDGET_EUR,
        alerts_path: str | Path | None = DEFAULT_ALERTS_PATH,
        thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS,
        emit: Callable[[str], None] = print,
    ) -> None:
        if not math.isfinite(budget_eur) or budget_eur <= 0:
            raise ValueError(f"budget_eur must be a positive number, got {budget_eur!r}")
        if not thresholds or any(not 0 < t <= 1.0 for t in thresholds):
            raise ValueError("thresholds must be fractions in (0, 1]")
        self.budget_eur = budget_eur
        self.thresholds = tuple(sorted(set(thresholds)))
        self.alerts_path = Path(alerts_path) if alerts_path else None
        self.emit = emit
        self.fired: set[float] = set()
        self._load_fired()

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None, **kw) -> "BudgetAlerts":
        """BUDGET_EUR overrides the 5 euro default."""
        env = os.environ if env is None else env
        return cls(budget_eur=float(env.get("BUDGET_EUR", DEFAULT_BUDGET_EUR)), **kw)

    def _load_fired(self) -> None:
        if not self.alerts_path or not self.alerts_path.exists():
            return
        for line in self.alerts_path.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
                if float(rec["budget_eur"]) == self.budget_eur:
                    self.fired.add(float(rec["threshold"]))
            except (ValueError, KeyError, TypeError):
                continue

    def exhausted(self, spent_eur: float) -> bool:
        """True once the budget is used up. Independent of what was already announced."""
        return spent_eur >= self.budget_eur

    def check(self, spent_eur: float) -> list[Alert]:
        """Emit (stdout + file) an alert for every threshold crossed that has not fired yet."""
        new: list[Alert] = []
        for t in self.thresholds:
            if t in self.fired or spent_eur < t * self.budget_eur:
                continue
            self.fired.add(t)
            level = _level(t)
            if level == "hard_stop":
                msg = (f"BUDGET HARD STOP: EUR {spent_eur:.2f} spent of EUR {self.budget_eur:.2f} "
                       f"({spent_eur / self.budget_eur:.0%}). Stop making paid model calls.")
            else:
                msg = (f"Budget alert: {t:.0%} of EUR {self.budget_eur:.2f} reached "
                       f"(EUR {spent_eur:.2f} spent).")
            alert = Alert(time.time(), level, t, self.budget_eur, spent_eur, msg)
            new.append(alert)
            self.emit(msg)
            self._persist(alert)
        return new

    def check_file(self, usage_path: str | Path = DEFAULT_USAGE_PATH, cost_field: str = "cost_eur") -> UsageTotal:
        """Read the usage log, fire any due alerts, return the total."""
        total = read_usage_total(usage_path, cost_field)
        self.check(total.cost_eur)
        return total

    def _persist(self, alert: Alert) -> None:
        if not self.alerts_path:
            return
        self.alerts_path.parent.mkdir(parents=True, exist_ok=True)
        with self.alerts_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(alert)) + "\n")


def register(sub) -> None:
    """Adds `budget` to the CLI: python -m agent_trader budget [--usage F] [--budget EUR]."""

    def run(args) -> int:
        try:
            alerts = BudgetAlerts(args.budget, args.alerts)
        except ValueError as exc:
            print(f"Budget refused: {exc}")
            return 2
        total = alerts.check_file(args.usage)
        print(f"Spent EUR {total.cost_eur:.4f} of EUR {args.budget:.2f} "
              f"({total.cost_eur / args.budget:.0%}), {total.records} calls"
              + (f", {total.skipped} unreadable lines skipped" if total.skipped else ""))
        return 3 if alerts.exhausted(total.cost_eur) else 0

    p = sub.add_parser("budget", help="check the usage log against the budget and fire alerts")
    p.add_argument("--usage", default=DEFAULT_USAGE_PATH, help="usage JSONL with a cost_eur field per call")
    p.add_argument("--budget", type=float, default=float(os.environ.get("BUDGET_EUR", DEFAULT_BUDGET_EUR)),
                   help="budget in euro (default 5, or env BUDGET_EUR)")
    p.add_argument("--alerts", default=DEFAULT_ALERTS_PATH)
    p.set_defaults(fn=run)

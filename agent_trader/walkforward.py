"""Walk-forward evaluation: many rolling train/test windows over several data series.

One replay on one seed says almost nothing. Here every series (synthetic seed or real
CSV) is cut into consecutive folds of `train_bars` followed by `test_bars`; each agent
is run fresh on each part, and only the test parts count. Nothing is tuned on train:
it is reported so overfitting would show up as a big train/test gap.

LLM folds cost money, so the LLM agent only runs on the first `llm_folds` folds, and an
estimated euro cap stops further calls.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Callable

from .config import Config
from .journal import Journal
from .killswitch import KillSwitch
from .models import Candle
from .replay import run_replay

# Haiku 4.5 list price in USD per million tokens, and a rough USD->EUR rate.
PRICE_IN, PRICE_OUT, USD_EUR = 1.0, 5.0, 0.92


class BudgetStop(Exception):
    """Raised by MeteredLLM over the cap; LLMAgent fails closed, and the fold is dropped."""


@dataclass
class MeteredLLM:
    """Wraps an LLM callable; estimates tokens (~4 chars each) and refuses to go over a euro cap."""

    llm: Callable[[str, str], str]
    max_eur: float
    calls: int = 0
    chars_in: int = 0
    chars_out: int = 0
    blocked: int = 0       # calls refused because the cap was hit

    @property
    def eur(self) -> float:
        usd = self.chars_in / 4 / 1e6 * PRICE_IN + self.chars_out / 4 / 1e6 * PRICE_OUT
        return usd * USD_EUR

    def __call__(self, system: str, user: str) -> str:
        if self.eur >= self.max_eur:
            self.blocked += 1
            raise BudgetStop(f"estimated spend {self.eur:.2f} EUR reached cap {self.max_eur:.2f} EUR")
        out = self.llm(system, user)
        self.calls += 1
        self.chars_in += len(system) + len(user)
        self.chars_out += len(out)
        return out


@dataclass
class FoldResult:
    series: str
    fold: int
    part: str              # "train" or "test"
    agent: str
    ret: float
    buy_hold: float
    max_drawdown: float
    trades: int
    halted: str | None


@dataclass
class Summary:
    agent: str
    part: str
    n: int
    mean: float
    median: float
    worst: float
    best: float
    mean_buy_hold: float
    beat_buy_hold: float   # share of folds with return > buy-and-hold
    beat_cash: float       # share of folds with return > 0
    halts: int


@dataclass
class WalkForwardReport:
    folds: list[FoldResult] = field(default_factory=list)
    llm_note: str = ""

    def summaries(self) -> list[Summary]:
        keys = sorted({(f.agent, f.part) for f in self.folds}, key=lambda k: (k[1] != "test", k[0]))
        out = []
        for agent, part in keys:
            rows = [f for f in self.folds if f.agent == agent and f.part == part]
            rets = [f.ret for f in rows]
            out.append(Summary(
                agent, part, len(rows), statistics.fmean(rets), statistics.median(rets),
                min(rets), max(rets), statistics.fmean(f.buy_hold for f in rows),
                sum(f.ret > f.buy_hold for f in rows) / len(rows),
                sum(f.ret > 0 for f in rows) / len(rows),
                sum(f.halted is not None for f in rows),
            ))
        return out


def make_folds(n: int, train_bars: int, test_bars: int) -> list[tuple[slice, slice]]:
    """Consecutive, non-overlapping test windows, each preceded by its own train window."""
    folds, start = [], 0
    while start + train_bars + test_bars <= n:
        mid = start + train_bars
        folds.append((slice(start, mid), slice(mid, mid + test_bars)))
        start += test_bars
    return folds


def run_walkforward(
    series: dict[str, dict[str, list[Candle]]],
    agents: dict[str, Callable[[], object]],
    config: Config,
    train_bars: int,
    test_bars: int,
    llm_agents: frozenset[str] = frozenset(),
    llm_folds: int = 3,
    meter: MeteredLLM | None = None,
    journal: Journal | None = None,
) -> WalkForwardReport:
    """`series` maps a label (e.g. "synthetic seed 3") to aligned candles. `agents` maps a
    name to a factory so every run starts with a fresh agent. Agents named in
    `llm_agents` run only on test windows of the first `llm_folds` folds overall."""
    journal = journal or Journal(None)
    report = WalkForwardReport()
    llm_used = 0
    for label, candles in series.items():
        n = len(next(iter(candles.values())))
        for k, (tr, te) in enumerate(make_folds(n, train_bars, test_bars)):
            is_llm_fold = llm_used < llm_folds
            for name, factory in agents.items():
                is_llm = name in llm_agents
                if is_llm and not is_llm_fold:
                    continue
                parts = [("test", te)] if is_llm else [("train", tr), ("test", te)]
                for part, sl in parts:
                    window = {s: c[sl] for s, c in candles.items()}
                    if is_llm and meter is not None and meter.blocked:
                        continue   # budget already hit: skip rather than report a fake "no trades" fold
                    ks = KillSwitch()
                    ks.path = _NoFile()
                    r = run_replay(window, factory(), config, ks, journal)
                    if is_llm and meter is not None and meter.blocked:
                        report.llm_note = (f"estimated cap {meter.max_eur:.2f} EUR reached; "
                                           f"the fold it hit was dropped")
                        continue
                    report.folds.append(FoldResult(
                        label, k, part, name, r.scorecard.total_return, r.scorecard.buy_hold_return,
                        r.scorecard.max_drawdown, r.scorecard.trades, r.halted,
                    ))
            if is_llm_fold and llm_agents:
                llm_used += 1
    return report


class _NoFile:
    """Path stand-in so walk-forward never reads or writes the human KILL file."""

    def exists(self) -> bool:
        return False

    def write_text(self, _: str) -> None:
        pass

    def unlink(self, missing_ok: bool = False) -> None:
        pass


def format_report(report: WalkForwardReport, data_label: str, meter: MeteredLLM | None = None) -> str:
    lines = [f"Data: {data_label}.  Only TEST rows count; nothing is tuned on TRAIN.", ""]
    lines.append(f"{'part':5} {'agent':9} {'folds':>5} {'mean':>8} {'median':>8} {'worst':>8} "
                 f"{'best':>8} {'B&H mean':>9} {'beat B&H':>9} {'>0':>5} {'halts':>5}")
    for s in report.summaries():
        lines.append(
            f"{s.part:5} {s.agent:9} {s.n:5d} {s.mean:+8.2%} {s.median:+8.2%} {s.worst:+8.2%} "
            f"{s.best:+8.2%} {s.mean_buy_hold:+9.2%} {s.beat_buy_hold:9.0%} {s.beat_cash:5.0%} {s.halts:5d}"
        )
    if meter is not None and meter.calls:
        lines.append(f"\nLLM calls: {meter.calls}, estimated cost {meter.eur:.3f} EUR (chars/4 token estimate)")
    if report.llm_note:
        lines.append(f"\nLLM stopped: {report.llm_note}")
    return "\n".join(lines)

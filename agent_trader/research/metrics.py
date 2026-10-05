"""Period metrics on the continuous equity curve, the dev gate, the holdout success bar and the
dev tuning rule."""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass

from .backtest import DAY, Market, Run

N_BARS = 35_999
WARMUP_END = 3598                 # bars 0..3598: warm-up only, never scored
DEV = (3599, 25198)               # 30 months of 720 bars
HOLDOUT = (25199, 35998)          # 15 months of 720 bars (= data.split(candles, 0.3))
MONTH_BARS = 720
FLAT = 0.0025                     # |monthly return| below this counts as a flat month


@dataclass(frozen=True)
class Metrics:
    compounded: float
    cagr: float
    mean_month: float             # arithmetic mean of the 720-bar month returns
    months: tuple[float, ...]
    positive: int
    flat: int
    negative: int
    worst_month: float
    max_dd: float                 # on hourly marks, peak reset at the period start
    underwater_days: float        # longest stretch below the running peak
    sharpe: float                 # daily returns from 23:00 marks, x sqrt(365), rf 0
    calmar: float                 # CAGR / max DD
    fills: int
    cost_drag: float              # fees + slippage per year / average equity
    exposure: float               # mean invested fraction at hourly closes


def drawdown(equity: list[float], ts: list[int]) -> tuple[float, float]:
    """Max drawdown and the longest time under water in days (an unrecovered one counts to the end)."""
    peak, peak_ts = equity[0], ts[0]
    worst = longest = 0.0
    under = False
    for x, t in zip(equity, ts):
        if x >= peak:
            if under:
                longest = max(longest, t - peak_ts)
                under = False
            peak, peak_ts = x, t
        else:
            under = True
            worst = max(worst, 1 - x / peak)
    if under:
        longest = max(longest, ts[-1] - peak_ts)
    return worst, longest / DAY


def period_metrics(m: Market, run: Run, start: int, end: int) -> Metrics:
    """Metrics for bars start..end of one continuous run; the base is the close of bar start-1."""
    if start < 1 or end >= len(run.equity) or (end - start + 1) % MONTH_BARS:
        raise ValueError("a period must be whole 720-bar months after bar 0")
    eq = run.equity
    base = start - 1
    months = tuple(eq[i + MONTH_BARS - 1] / eq[i - 1] - 1 for i in range(start, end + 1, MONTH_BARS))
    max_dd, underwater = drawdown(eq[base:end + 1], m.ts[base:end + 1])
    marks = [eq[t] for t in m.day_bars if base <= t <= end]
    daily = [b / a - 1 for a, b in zip(marks, marks[1:])]
    sd = statistics.stdev(daily)
    years = (m.ts[end] - m.ts[base]) / (365 * DAY)
    growth = eq[end] / eq[base]
    cagr = growth ** (1 / years) - 1
    fills = [f for f in run.fills if start <= f.bar <= end]
    return Metrics(
        compounded=growth - 1,
        cagr=cagr,
        mean_month=statistics.fmean(months),
        months=months,
        positive=sum(r >= FLAT for r in months),
        flat=sum(abs(r) < FLAT for r in months),
        negative=sum(r <= -FLAT for r in months),
        worst_month=min(months),
        max_dd=max_dd,
        underwater_days=underwater,
        sharpe=statistics.fmean(daily) / sd * math.sqrt(365) if sd > 0 else 0.0,
        calmar=cagr / max_dd if max_dd > 0 else math.inf,
        fills=len(fills),
        cost_drag=sum(f.fee + f.slippage for f in fills) / statistics.fmean(eq[start:end + 1]) / years,
        exposure=statistics.fmean(run.exposure[start:end + 1]),
    )


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str


def _pct(x: float) -> str:
    return f"{x * 100:+.2f}%"


def dev_gate(s: Metrics, bh: Metrics) -> list[Check]:
    """All must hold, else the strategy is FAILED-DEV and never runs on holdout."""
    return [
        Check("Sharpe >= 0.9 x B&H", s.sharpe >= 0.9 * bh.sharpe,
              f"{s.sharpe:.2f} vs needed {0.9 * bh.sharpe:.2f}"),
        Check("max DD <= 0.6 x B&H", s.max_dd <= 0.6 * bh.max_dd,
              f"{s.max_dd * 100:.1f}% vs allowed {0.6 * bh.max_dd * 100:.1f}%"),
        Check("compounded return > 0", s.compounded > 0, _pct(s.compounded)),
        Check("cost drag <= 2%/yr", s.cost_drag <= 0.02, f"{s.cost_drag * 100:.2f}%/yr"),
    ]


def success_bar(s: Metrics, bh: Metrics) -> list[Check]:
    """The five holdout criteria (Strategist plan, section 4)."""
    return [
        Check("1 makes money: return > 0 and mean month >= B&H - 1pp",
              s.compounded > 0 and s.mean_month >= bh.mean_month - 0.01,
              f"{_pct(s.compounded)}; mean month {_pct(s.mean_month)} vs needed {_pct(bh.mean_month - 0.01)}"),
        Check("2 hit rate: positive months >= B&H", s.positive >= bh.positive,
              f"{s.positive} vs {bh.positive}"),
        Check("3 worst month >= half of B&H's", s.worst_month >= 0.5 * bh.worst_month,
              f"{_pct(s.worst_month)} vs needed {_pct(0.5 * bh.worst_month)}"),
        Check("4 max DD <= half of B&H and <= 15%, under water <= B&H",
              s.max_dd <= 0.5 * bh.max_dd and s.max_dd <= 0.15 and s.underwater_days <= bh.underwater_days,
              f"DD {s.max_dd * 100:.1f}% vs allowed {min(0.5 * bh.max_dd, 0.15) * 100:.1f}%; "
              f"under water {s.underwater_days:.0f} vs {bh.underwater_days:.0f} days"),
        Check("5 Sharpe and Calmar > B&H", s.sharpe > bh.sharpe and s.calmar > bh.calmar,
              f"Sharpe {s.sharpe:.2f} vs {bh.sharpe:.2f}; Calmar {s.calmar:.2f} vs {bh.calmar:.2f}"),
    ]


def verdict(checks: list[Check]) -> str:
    passed = sum(c.ok for c in checks)
    return "PASS" if passed == len(checks) else "NEAR MISS" if passed == len(checks) - 1 else "FAIL"


def select(results: dict[tuple, Metrics], axes: list[tuple]) -> tuple[tuple, dict[tuple, float]]:
    """Dev tuning rule: highest median dev Sharpe over a cell and its direct neighbours (one step
    in one parameter); ties go to lower dev max DD, then fewer fills. Returns (pick, scores)."""
    def neighbours(p: tuple):
        for i, axis in enumerate(axes):
            j = axis.index(p[i])
            for k in (j - 1, j + 1):
                if 0 <= k < len(axis):
                    yield p[:i] + (axis[k],) + p[i + 1:]

    scores = {p: statistics.median([results[p].sharpe] + [results[q].sharpe for q in neighbours(p)])
              for p in results}
    pick = min(results, key=lambda p: (-scores[p], results[p].max_dd, results[p].fills))
    return pick, scores

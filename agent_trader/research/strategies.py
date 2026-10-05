"""Rule-based long-or-cash strategies for ETH and BTC (Strategist plan, 2026-10-05).

Each builder reads only prices up to each decision bar and returns {decision bar: target weights}.
Daily rules decide at the 23:00 close of each complete UTC day; weekly rules only at Sunday's.
No target is issued until every indicator is defined, so a run sits in cash until then.
"""
from __future__ import annotations

import math
import statistics

from .backtest import Decisions, Market

CADENCES = ("daily", "weekly")

# Dev grids (axis order matters for the neighbour rule in metrics.select) and the plan's defaults.
S1_GRID = {"b": (10, 20, 30), "cadence": CADENCES}
S2_GRID = {"vol_target": (0.40, 0.60, 0.80), "n": (20, 60)}
S3_GRID = {"lookback": (30, 60, 90), "tilt": (0.5, 0.75, 1.0)}
S4_GRID = {"max_dd": (0.10, 0.15, 0.20), "window": (30, 60, 90)}
DEFAULTS = {
    "S1": {"b": 20, "cadence": "weekly"},
    "S2": {"vol_target": 0.60, "n": 20},
    "S3": {"lookback": 60, "tilt": 0.75},
    "S4": {"max_dd": 0.15, "window": 60},
}


def _decision_days(m: Market, cadence: str, first: int) -> list[int]:
    if cadence not in CADENCES:
        raise ValueError(f"cadence must be one of {CADENCES}")
    return [d for d in range(first, len(m.day_bars)) if cadence == "daily" or m.is_sunday(d)]


def _ladder(closes: list[float], d: int, b: int) -> float:
    """0.5 * (number of lookbacks L in {b, 2b, 4b} days with close_d > close_{d-L}) / 3."""
    return 0.5 * sum(closes[d] > closes[d - lb] for lb in (b, 2 * b, 4 * b)) / 3


def trend_ladder(m: Market, b: int, cadence: str) -> Decisions:
    """S1: multi-horizon time-series momentum with graded exposure (0, 1/6, 1/3 or 1/2 per asset)."""
    return {m.day_bars[d]: {s: _ladder(m.day_close[s], d, b) for s in m.symbols}
            for d in _decision_days(m, cadence, first=4 * b)}


def vol_target_ladder(m: Market, b: int, cadence: str, vol_target: float, n: int) -> Decisions:
    """S2: S1 times min(1, vol_target / vol_hat); vol_hat = sqrt(365) * stdev (ddof 1) of the
    last n daily log returns up to and including day d."""
    out: Decisions = {}
    for d in _decision_days(m, cadence, first=max(4 * b, n)):
        w = {}
        for s in m.symbols:
            c = m.day_close[s]
            vol = math.sqrt(365) * statistics.stdev(
                [math.log(c[i] / c[i - 1]) for i in range(d - n + 1, d + 1)])
            w[s] = _ladder(c, d, b) * (min(1.0, vol_target / vol) if vol > 0 else 1.0)
        out[m.day_bars[d]] = w
    return out


def dual_momentum(m: Market, lookback: int, tilt: float) -> Decisions:
    """S3 (weekly): leader by lookback return gets tilt, laggard 1 - tilt (a tie goes to BTC);
    any asset whose own lookback return is <= 0 gets 0 and its share stays in cash."""
    out: Decisions = {}
    for d in _decision_days(m, "weekly", first=lookback):
        r = {s: m.day_close[s][d] / m.day_close[s][d - lookback] - 1 for s in m.symbols}
        leader = "ETH" if r["ETH"] > r["BTC"] else "BTC"
        out[m.day_bars[d]] = {s: ((tilt if s == leader else 1 - tilt) if r[s] > 0 else 0.0)
                              for s in m.symbols}
    return out


def drawdown_brake(m: Market, max_dd: float, window: int) -> Decisions:
    """S4 (daily, sleeve 0.5 per asset): M = highest daily close of the last `window` days incl.
    day d, DD = 1 - close_d / M. ON on the first day M is defined; ON -> OFF when DD >= max_dd;
    OFF -> ON when DD < max_dd / 2."""
    out: Decisions = {}
    on: dict[str, bool | None] = {s: None for s in m.symbols}
    for d in _decision_days(m, "daily", first=window - 1):
        w = {}
        for s in m.symbols:
            c = m.day_close[s]
            dd = 1 - c[d] / max(c[d - window + 1: d + 1])
            if on[s] is None:
                on[s] = True
            elif on[s] and dd >= max_dd:
                on[s] = False
            elif not on[s] and dd < max_dd / 2:
                on[s] = True
            w[s] = 0.5 if on[s] else 0.0
        out[m.day_bars[d]] = w
    return out


def ref_sma(m: Market, n: int = 336, band: float = 0.01) -> Decisions:
    """REF-SMA336 (reference, not a candidate): per asset sleeve 0.5, hourly decisions; in when
    close > SMA(n)*(1+band), out when close < SMA(n)*(1-band), else keep."""
    out: Decisions = {}
    total = {s: 0.0 for s in m.symbols}
    inside = {s: False for s in m.symbols}
    for t in range(m.n):
        for s in m.symbols:
            total[s] += m.close[s][t] - (m.close[s][t - n] if t >= n else 0.0)
        if t < n - 1:
            continue
        for s in m.symbols:
            sma = total[s] / n
            if m.close[s][t] > sma * (1 + band):
                inside[s] = True
            elif m.close[s][t] < sma * (1 - band):
                inside[s] = False
        out[t] = {s: 0.5 if inside[s] else 0.0 for s in m.symbols}
    return out

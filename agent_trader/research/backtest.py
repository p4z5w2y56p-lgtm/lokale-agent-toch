"""Continuous long-or-cash backtest with next-bar fills.

A strategy hands in target weights per decision bar (weights >= 0, sum <= 1, rest is cash).
A decision at bar t may use data up to close[t] only and fills at open[t+1]: buys at
open*(1+5 bps), sells at open*(1-5 bps), plus a 10 bps fee on the fill notional. A decision on
the last bar never fills. An asset trades only if its weight must move by >= 0.05, or its target
is 0 while it is still held; it then trades exactly to target, sells before buys, and buys are
scaled down if cash is short. Equity is marked at every hourly close.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from ..data import align, load_csv

FEE = 0.001          # 10 bps of fill notional
SLIPPAGE = 0.0005    # 5 bps against us on every fill
MIN_TRADE = 0.05     # minimum weight change worth trading
START_CASH = 10_000.0
SYMBOLS = ("ETH", "BTC")
HOUR = 3600
DAY = 86400

Decisions = dict[int, dict[str, float]]  # decision bar index -> target weight per symbol


class Market:
    """Aligned hourly bars plus a daily view: close_d = close of each complete UTC day's 23:00 bar."""

    def __init__(self, ts, opens, closes, symbols=SYMBOLS):
        self.symbols = tuple(symbols)
        self.ts = list(ts)
        self.open = {s: list(opens[s]) for s in self.symbols}
        self.close = {s: list(closes[s]) for s in self.symbols}
        self.n = len(self.ts)
        if any(len(v) != self.n for v in (*self.open.values(), *self.close.values())):
            raise ValueError("every symbol needs one open and one close per bar")
        # A complete UTC day has both its 00:00 and its 23:00 bar (one missing hour inside is fine).
        started: set[int] = set()
        self.day_bars: list[int] = []
        for i, t in enumerate(self.ts):
            day, sec = divmod(t, DAY)
            if sec == 0:
                started.add(day)
            elif sec == 23 * HOUR and day in started:
                self.day_bars.append(i)
        self.day_close = {s: [self.close[s][i] for i in self.day_bars] for s in self.symbols}

    def is_sunday(self, d: int) -> bool:
        """Is daily bar d a Sunday? (1970-01-01, day 0, was a Thursday; Monday = 0.)"""
        return (self.ts[self.day_bars[d]] // DAY + 3) % 7 == 6

    def head(self, n: int) -> "Market":
        """The first n bars only: lets tuning run without ever touching later (holdout) bars."""
        return Market(self.ts[:n], {s: v[:n] for s, v in self.open.items()},
                      {s: v[:n] for s, v in self.close.items()}, self.symbols)

    def scaled_after(self, t: int, factor: float) -> "Market":
        """Copy with every price after bar t multiplied by factor (for the no-lookahead test)."""
        def scale(v):
            return v[: t + 1] + [x * factor for x in v[t + 1:]]
        return Market(self.ts, {s: scale(v) for s, v in self.open.items()},
                      {s: scale(v) for s, v in self.close.items()}, self.symbols)


def load_market(data_dir: str | Path) -> Market:
    """Load eth_1h.csv and btc_1h.csv (ts,open,high,low,close,volume) from data_dir."""
    raw = align({s: load_csv(Path(data_dir) / f"{s.lower()}_1h.csv", s) for s in SYMBOLS})
    ts = [c.ts for c in raw[SYMBOLS[0]]]
    if any(t % HOUR for t in ts):
        raise ValueError("timestamps must be whole UTC hours")
    return Market(ts, {s: [c.open for c in raw[s]] for s in SYMBOLS},
                  {s: [c.close for c in raw[s]] for s in SYMBOLS})


@dataclass(frozen=True)
class Fill:
    bar: int        # bar whose open it filled at
    symbol: str
    side: str       # "buy" or "sell"
    qty: float
    price: float    # open * (1 +/- slippage)
    fee: float      # FEE * qty * price
    slippage: float  # qty * open * SLIPPAGE: cost versus the open


@dataclass
class Run:
    equity: list[float]    # mark at every hourly close
    exposure: list[float]  # invested fraction of equity at every hourly close
    fills: list[Fill]


def simulate(m: Market, decisions: Decisions, start_cash: float = START_CASH) -> Run:
    """One continuous run over every bar of m, starting in cash at bar 0."""
    cash = start_cash
    qty = {s: 0.0 for s in m.symbols}
    equity: list[float] = []
    exposure: list[float] = []
    fills: list[Fill] = []
    pending: dict[str, float] | None = None
    for t in range(m.n):
        if pending is not None:
            cash = _trade_to_target(m, t, pending, qty, cash, fills)
            pending = None
        held = sum(qty[s] * m.close[s][t] for s in m.symbols)
        equity.append(cash + held)
        exposure.append(held / (cash + held))
        if t in decisions and t + 1 < m.n:   # a decision on the last bar never fills
            target = decisions[t]
            if any(w < 0 for w in target.values()) or sum(target.values()) > 1 + 1e-9:
                raise ValueError(f"bad target weights at bar {t}: {target}")
            pending = target
    return Run(equity, exposure, fills)


def _trade_to_target(m: Market, t: int, target: dict[str, float], qty: dict[str, float],
                     cash: float, fills: list[Fill]) -> float:
    """Fill at open[t]: sells first, then buys (scaled down if cash is short). Returns new cash."""
    px = {s: m.open[s][t] for s in m.symbols}
    eq = cash + sum(qty[s] * px[s] for s in m.symbols)
    buys: dict[str, float] = {}
    for s in m.symbols:
        now = qty[s] * px[s] / eq
        want = target.get(s, 0.0)
        exit_all = want == 0.0 and qty[s] > 0
        if abs(want - now) < MIN_TRADE and not exit_all:
            continue
        if want < now:
            q = qty[s] if want == 0.0 else (now - want) * eq / px[s]
            price = px[s] * (1 - SLIPPAGE)
            fee = q * price * FEE
            cash += q * price - fee
            qty[s] = 0.0 if want == 0.0 else qty[s] - q
            fills.append(Fill(t, s, "sell", q, price, fee, q * px[s] * SLIPPAGE))
        elif want > now:
            buys[s] = (want - now) * eq / px[s]
    if buys:
        cost = sum(q * px[s] * (1 + SLIPPAGE) * (1 + FEE) for s, q in buys.items())
        scale = min(1.0, cash / cost)
        for s, q in buys.items():
            q *= scale
            price = px[s] * (1 + SLIPPAGE)
            fee = q * price * FEE
            cash -= q * price + fee
            qty[s] += q
            fills.append(Fill(t, s, "buy", q, price, fee, q * px[s] * SLIPPAGE))
        cash = max(cash, 0.0)  # float dust after a scaled-down buy
    return cash


def buy_and_hold(m: Market, base: int, start_cash: float = START_CASH) -> Run:
    """50/50 ETH/BTC bought at close[base], never rebalanced, no fees. Cash before base."""
    qty = {s: start_cash / len(m.symbols) / m.close[s][base] for s in m.symbols}
    equity = [start_cash if t < base else sum(qty[s] * m.close[s][t] for s in m.symbols)
              for t in range(m.n)]
    exposure = [0.0 if t < base else 1.0 for t in range(m.n)]
    return Run(equity, exposure, [])


def fills_hash(fills: list[Fill]) -> str:
    """Determinism fingerprint of a fill list (exact float reprs)."""
    h = hashlib.sha256()
    for f in fills:
        h.update(repr((f.bar, f.symbol, f.side, f.qty, f.price, f.fee)).encode())
    return h.hexdigest()[:16]

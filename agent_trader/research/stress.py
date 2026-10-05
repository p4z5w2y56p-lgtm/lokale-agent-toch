"""Round 2: stress-test the round-1 conclusion ("the rules cut drawdowns through lower exposure, not
timing") on real data that round 1 never scored, without touching the holdout.

A) Older-data one-shot: frozen round-1 picks, references and static allocations on 2017-2022 data.
B) Walk-forward: on joined pre-holdout data (older + dev era), retune each strategy on 24 train
   months with the round-1 grid and selection rule, apply the pick to the next 6 months, and chain
   the test windows into one continuous out-of-sample curve.
C) Exposure study: static allocations 0..100% on the older data and on dev.
Every rule is compared with a static allocation at the same average exposure.

Holdout safety: CSVs are read only up to the last bar before HOLDOUT_START_TS (reading stops at the
first row at or after it), and every entry point re-checks that no bar at or after it is present.

Run:  python -m agent_trader.research.stress --old data/binance_early --dev data/binance --out FILE
"""
from __future__ import annotations

import argparse
import csv
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Callable

from .backtest import HOUR, SYMBOLS, Decisions, Market, Run, buy_and_hold, simulate
from .metrics import DEV, MONTH_BARS, Metrics, drawdown, period_metrics, select
from .run import _git_rev
from .strategies import (S1_GRID, S2_GRID, S3_GRID, S4_GRID, drawdown_brake, dual_momentum, ref_sma,
                         trend_ladder, vol_target_ladder)

HOLDOUT_START_TS = int(datetime(2025, 7, 12, 19, tzinfo=timezone.utc).timestamp())  # bar 25199 of data/binance
WARMUP = DEV[0]                  # 3599 unscored bars before every scored window, as in round 1
TRAIN_MONTHS, TEST_MONTHS = 24, 6
STATIC_LEVELS = (0.2, 0.4, 0.6, 0.8)
EXPOSURE_GRID = tuple(i / 10 for i in range(11))
MAX_COST_DRAG = 0.02
JUNCTION_JUMP = 0.02             # older close vs next dev-era open: a bigger jump means misaligned files

# Frozen round-1 picks (results.md, code 482d9d1) and the round-1 grids, keyed like round 1.
FROZEN = {
    "S1": {"b": 30, "cadence": "daily"},
    "S2": {"b": 30, "cadence": "daily", "vol_target": 0.40, "n": 60},
    "S3": {"lookback": 30, "tilt": 0.5},
    "S4": {"max_dd": 0.10, "window": 30},
}
BUILD = {"S1": trend_ladder, "S2": vol_target_ladder, "S3": dual_momentum, "S4": drawdown_brake}
GRIDS = {"S1": S1_GRID, "S2": S2_GRID, "S3": S3_GRID, "S4": S4_GRID}
TITLES = {"S1": "Trend ladder", "S2": "Vol-targeted ladder", "S3": "ETH/BTC dual momentum",
          "S4": "Drawdown brake", "REF": "REF-SMA336"}
RULES = ("S1", "S2", "S3", "S4")


# ----------------------------------------------------------------------------- data


def _read_pre_holdout(path: Path, cutoff: int) -> tuple[list[int], list[float], list[float]]:
    """ts, opens, closes of every row before cutoff; stops reading at the first row at or after it."""
    ts: list[int] = []
    opens: list[float] = []
    closes: list[float] = []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            t = int(row["ts"])
            if t >= cutoff:
                break
            if ts and t <= ts[-1]:
                raise ValueError(f"{path}: timestamps must be strictly increasing (ts={t})")
            o, c = float(row["open"]), float(row["close"])
            if not (o > 0 and c > 0):
                raise ValueError(f"{path}: bad price at ts={t}")
            ts.append(t)
            opens.append(o)
            closes.append(c)
    return ts, opens, closes


def load_pre_holdout(data_dir: str | Path, cutoff: int = HOLDOUT_START_TS) -> Market:
    """eth_1h.csv and btc_1h.csv from data_dir, bars before cutoff only. Both symbols must carry the
    exact same timestamps (no silent intersection) on whole UTC hours."""
    cols = {s: _read_pre_holdout(Path(data_dir) / f"{s.lower()}_1h.csv", cutoff) for s in SYMBOLS}
    ts = cols[SYMBOLS[0]][0]
    for s in SYMBOLS[1:]:
        if cols[s][0] != ts:
            raise ValueError(f"{data_dir}: {s} timestamps differ from {SYMBOLS[0]}'s")
    if not ts or any(t % HOUR for t in ts):
        raise ValueError(f"{data_dir}: need bars on whole UTC hours")
    m = Market(ts, {s: cols[s][1] for s in SYMBOLS}, {s: cols[s][2] for s in SYMBOLS})
    assert_pre_holdout(m, cutoff)
    return m


def join_markets(older: Market, newer: Market) -> Market:
    """older followed by newer. Refuses duplicates, overlaps, unordered or off-hour timestamps,
    different symbols, and a price jump at the seam (a sign of swapped or misaligned files)."""
    if older.symbols != newer.symbols:
        raise ValueError("both markets need the same symbols")
    ts = older.ts + newer.ts
    if any(b <= a for a, b in zip(ts, ts[1:])):
        raise ValueError("joined timestamps must be strictly increasing (duplicate or overlapping bars)")
    if any(t % HOUR for t in ts):
        raise ValueError("joined timestamps must be whole UTC hours")
    for s in older.symbols:
        jump = newer.open[s][0] / older.close[s][-1] - 1
        if abs(jump) > JUNCTION_JUMP:
            raise ValueError(f"{s}: {jump:+.1%} jump at the seam; the files do not line up")
    return Market(ts, {s: older.open[s] + newer.open[s] for s in older.symbols},
                  {s: older.close[s] + newer.close[s] for s in older.symbols}, older.symbols)


def assert_pre_holdout(m: Market, cutoff: int = HOLDOUT_START_TS) -> None:
    if m.ts and max(m.ts) >= cutoff:
        raise ValueError("holdout bars present: round 2 must never see bars at or after the holdout start")


def window(m: Market, a: int, b: int) -> Market:
    """Bars a..b (inclusive) as their own market."""
    return Market(m.ts[a:b + 1], {s: v[a:b + 1] for s, v in m.open.items()},
                  {s: v[a:b + 1] for s, v in m.close.items()}, m.symbols)


# ----------------------------------------------------------------------------- building blocks


def static_decisions(m: Market, level: float) -> Decisions:
    """level of equity split 50/50 ETH/BTC, rebalanced at every Sunday close (same engine and costs)."""
    w = {s: level / len(m.symbols) for s in m.symbols}
    return {m.day_bars[d]: dict(w) for d in range(len(m.day_bars)) if m.is_sunday(d)}


def static_run(m: Market, level: float) -> Run:
    return simulate(m, static_decisions(m, level))


def matched_static(m: Market, target: float, start: int, end: int, tol: float = 0.002,
                   tries: int = 6) -> tuple[float, Run]:
    """Static level whose measured average exposure on start..end matches target within tol."""
    level = min(1.0, max(0.0, target))
    run = static_run(m, level)
    for _ in range(tries):
        got = statistics.fmean(run.exposure[start:end + 1])
        if abs(got - target) <= tol:
            break
        level = min(1.0, max(0.0, level + target - got))
        run = static_run(m, level)
    return level, run


def rule_decisions(m: Market, key: str, params: dict) -> Decisions:
    return ref_sma(m) if key == "REF" else BUILD[key](m, **params)


def year_split(m: Market, run: Run, start: int, end: int) -> list[tuple[int, float, float]]:
    """(year, return, max DD) per calendar year inside start..end; each from the previous year's last close."""
    years: dict[int, list[int]] = {}
    for i in range(start, end + 1):
        years.setdefault(datetime.fromtimestamp(m.ts[i], tz=timezone.utc).year, []).append(i)
    out = []
    for y, bars in years.items():
        base, last = bars[0] - 1, bars[-1]
        dd, _ = drawdown(run.equity[base:last + 1], m.ts[base:last + 1])
        out.append((y, run.equity[last] / run.equity[base] - 1, dd))
    return out


@dataclass
class Row:
    name: str
    metrics: Metrics
    run: Run
    note: str = ""


# ----------------------------------------------------------------------------- A) older one-shot


@dataclass
class OneShot:
    m: Market
    start: int
    end: int
    rows: dict[str, Row]          # S1..S4, REF, B&H, static levels
    matched: dict[str, Row]       # rule key -> its matched static


def one_shot(m: Market) -> OneShot:
    """Frozen picks, references and statics on m, scored on whole 720-bar months ending at its last
    bar after at least WARMUP warm-up bars. No retuning."""
    assert_pre_holdout(m)
    months = (m.n - WARMUP) // MONTH_BARS
    end = m.n - 1
    start = end - months * MONTH_BARS + 1
    rows: dict[str, Row] = {}
    for key in (*RULES, "REF"):
        run = simulate(m, rule_decisions(m, key, FROZEN.get(key, {})))
        rows[key] = Row(key, period_metrics(m, run, start, end), run)
    bh = buy_and_hold(m, start - 1)
    rows["B&H"] = Row("B&H", period_metrics(m, bh, start, end), bh)
    for lv in STATIC_LEVELS:
        run = static_run(m, lv)
        rows[f"ST{lv:.0%}"] = Row(f"Static {lv:.0%}", period_metrics(m, run, start, end), run)
    matched = {}
    for key in (*RULES, "REF"):
        lv, run = matched_static(m, rows[key].metrics.exposure, start, end)
        matched[key] = Row(f"Static {lv:.1%}", period_metrics(m, run, start, end), run, note=f"{lv:.3f}")
    return OneShot(m, start, end, rows, matched)


# ----------------------------------------------------------------------------- B) walk-forward


@dataclass(frozen=True)
class Fold:
    train: tuple[int, int]        # inclusive bar range scored while tuning
    test: tuple[int, int]         # inclusive bar range of the out-of-sample window
    warm: int                     # first bar of the tuning slice (warm-up before train)


def folds(n: int, train_bars: int, test_bars: int, warmup: int) -> list[Fold]:
    """Back-to-back test windows ending at bar n-1; each train window is the train_bars right
    before its test window, with warmup bars of earlier history in front."""
    k = (n - warmup - train_bars) // test_bars
    if k < 1:
        raise ValueError("not enough bars for one fold")
    out = []
    for i in range(k):
        te_s = n - (k - i) * test_bars
        tr_s = te_s - train_bars
        out.append(Fold((tr_s, te_s - 1), (te_s, te_s + test_bars - 1), tr_s - warmup))
    return out


def tune_on(m: Market, build: Callable, grid: dict, start: int, end: int,
            fixed: dict | None = None) -> tuple[dict, dict]:
    """Round-1 selection rule on bars start..end of m. Returns (picked params, picked Metrics)."""
    fixed = fixed or {}
    results = {}
    for point in product(*grid.values()):
        run = simulate(m, build(m, **fixed, **dict(zip(grid, point))))
        results[point] = period_metrics(m, run, start, end)
    pick, _ = select(results, list(grid.values()))
    return {**fixed, **dict(zip(grid, pick))}, results[pick]


def tune_fold(m: Market, f: Fold, grids: dict = GRIDS, build: dict = BUILD) -> dict[str, dict]:
    """Picks for one fold. The tuning market is the slice f.warm..f.train[1]: no test or later bar."""
    tm = window(m, f.warm, f.train[1])
    a, b = f.train[0] - f.warm, f.train[1] - f.warm
    picks: dict[str, dict] = {}
    picks["S1"], _ = tune_on(tm, build["S1"], grids["S1"], a, b)
    picks["S2"], _ = tune_on(tm, build["S2"], grids["S2"], a, b,
                             fixed={k: picks["S1"][k] for k in ("b", "cadence")})
    picks["S3"], _ = tune_on(tm, build["S3"], grids["S3"], a, b)
    picks["S4"], _ = tune_on(tm, build["S4"], grids["S4"], a, b)
    return picks


def chain_decisions(m: Market, fs: list[Fold], key: str, picks: list[dict],
                    build: dict = BUILD) -> Decisions:
    """Decisions of fold i's pick for bars inside test window i, built from data up to the end of
    that window only. At the last train bar the strategy switches to the new pick's latest target."""
    out: Decisions = {}
    for f, p in zip(fs, picks):
        sub = window(m, 0, f.test[1])
        dec = ref_sma(sub) if key == "REF" else build[key](sub, **p)
        switch = f.test[0] - 1
        before = [t for t in dec if t <= switch]
        if before:
            out[switch] = dict(dec[max(before)])
        out.update({t: w for t, w in dec.items() if f.test[0] <= t <= f.test[1]})
    return out


@dataclass
class WalkForward:
    m: Market
    folds: list[Fold]
    picks: list[dict[str, dict]]
    start: int
    end: int
    rows: dict[str, Row]
    matched: dict[str, Row]


def walk_forward(m: Market, train_months: int = TRAIN_MONTHS, test_months: int = TEST_MONTHS,
                 warmup: int = WARMUP, grids: dict = GRIDS, build: dict = BUILD,
                 levels: tuple = STATIC_LEVELS) -> WalkForward:
    assert_pre_holdout(m)
    fs = folds(m.n, train_months * MONTH_BARS, test_months * MONTH_BARS, warmup)
    picks = [tune_fold(m, f, grids, build) for f in fs]
    start, end = fs[0].test[0], fs[-1].test[1]
    rows: dict[str, Row] = {}
    for key in (*RULES, "REF"):
        per_fold = [p.get(key, {}) for p in picks]
        run = simulate(m, chain_decisions(m, fs, key, per_fold, build))
        rows[key] = Row(key, period_metrics(m, run, start, end), run)
    bh = buy_and_hold(m, start - 1)
    rows["B&H"] = Row("B&H", period_metrics(m, bh, start, end), bh)
    for lv in levels:
        run = static_run(m, lv)
        rows[f"ST{lv:.0%}"] = Row(f"Static {lv:.0%}", period_metrics(m, run, start, end), run)
    matched = {}
    for key in (*RULES, "REF"):
        lv, run = matched_static(m, rows[key].metrics.exposure, start, end)
        matched[key] = Row(f"Static {lv:.1%}", period_metrics(m, run, start, end), run, note=f"{lv:.3f}")
    return WalkForward(m, fs, picks, start, end, rows, matched)


# ----------------------------------------------------------------------------- C) exposure study


def exposure_study(m: Market, start: int, end: int) -> list[tuple[float, Metrics]]:
    assert_pre_holdout(m)
    return [(lv, period_metrics(m, static_run(m, lv), start, end)) for lv in EXPOSURE_GRID]


# ----------------------------------------------------------------------------- judgment


@dataclass(frozen=True)
class Verdict:
    key: str
    checks: dict[str, bool]       # "A Sharpe", "A max DD", ... "B cost drag"

    @property
    def timing_skill(self) -> bool:
        return all(self.checks.values())


def beats(rule: Metrics, static: Metrics) -> dict[str, bool]:
    share = lambda x: x.positive / len(x.months)  # noqa: E731
    return {"Sharpe": rule.sharpe > static.sharpe, "max DD": rule.max_dd < static.max_dd,
            "positive-month share": share(rule) > share(static), "cost drag <= 2%/yr": rule.cost_drag <= MAX_COST_DRAG}


def judge(key: str, a: OneShot, b: WalkForward) -> Verdict:
    checks = {f"A {k}": v for k, v in beats(a.rows[key].metrics, a.matched[key].metrics).items()}
    checks.update({f"B {k}": v for k, v in beats(b.rows[key].metrics, b.matched[key].metrics).items()})
    return Verdict(key, checks)


# ----------------------------------------------------------------------------- report


def _date(m: Market, bar: int) -> str:
    return datetime.fromtimestamp(m.ts[bar], tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


def _p(x: float, d: int = 1) -> str:
    return f"{x * 100:+.{d}f}%"


def _params(p: dict) -> str:
    return ", ".join(f"{k}={v}" for k, v in p.items()) or "-"


HEAD = ("| Strategy | Return | Sharpe | Calmar | Max DD | Positive months | Worst month | Cost drag | "
        "Avg exposure |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|")


def _line(name: str, x: Metrics) -> str:
    return (f"| {name} | {_p(x.compounded)} | {x.sharpe:.2f} | {x.calmar:.2f} | {x.max_dd * 100:.1f}% | "
            f"{x.positive}/{len(x.months)} | {_p(x.worst_month)} | {x.cost_drag * 100:.2f}%/yr | "
            f"{x.exposure * 100:.1f}% |")


def _label(key: str, params: dict | None = None) -> str:
    base = f"{key} {TITLES[key]}" if key in TITLES else key
    return f"{base} ({_params(params)})" if params else base


def _main_table(rows: dict[str, Row], matched: dict[str, Row], params: dict[str, dict]) -> list[str]:
    out = [HEAD]
    for key in (*RULES, "REF"):
        out.append(_line(_label(key, params.get(key)), rows[key].metrics))
        out.append(_line(f"  matched {matched[key].name}", matched[key].metrics))
    out.append(_line("B&H 50/50 (no fees, never rebalanced)", rows["B&H"].metrics))
    for lv in STATIC_LEVELS:
        k = f"ST{lv:.0%}"
        if k in rows:
            out.append(_line(rows[k].name + " (weekly rebalance)", rows[k].metrics))
    return out


def _vs_table(rows: dict[str, Row], matched: dict[str, Row]) -> list[str]:
    out = ["| Rule | Sharpe rule / static | Max DD rule / static | Positive months rule / static | "
           "Months +/=/- rule; static | Return rule / static | Cost drag rule | Beats on all three |",
           "|---|---:|---:|---:|---:|---:|---:|---|"]
    for key in (*RULES, "REF"):
        r, s = rows[key].metrics, matched[key].metrics
        b = beats(r, s)
        allthree = b["Sharpe"] and b["max DD"] and b["positive-month share"]
        out.append(f"| {key} | {r.sharpe:.2f} / {s.sharpe:.2f} | {r.max_dd * 100:.1f}% / {s.max_dd * 100:.1f}% | "
                   f"{r.positive} / {s.positive} of {len(r.months)} | {r.positive}/{r.flat}/{r.negative}; "
                   f"{s.positive}/{s.flat}/{s.negative} | {_p(r.compounded, 0)} / {_p(s.compounded, 0)} | "
                   f"{r.cost_drag * 100:.2f}%/yr | "
                   f"{'yes' if allthree else 'no'} |")
    return out


def report(a: OneShot | None, b: WalkForward, c_old: list | None, c_dev: list,
           dev_ctx: OneShot, old_src: str, dev_src: str) -> str:
    L: list[str] = []
    rev = _git_rev()
    L += [f"All numbers below are **real data** (Binance hourly ETHUSDT and BTCUSDT; older data from `{old_src}`, "
          f"dev-era data from `{dev_src}` read only up to 2025-07-12 18:00 UTC). Written by "
          f"`python -m agent_trader.research.stress` at code `{rev}`. No holdout bar was loaded; no LLM or API calls.", ""]

    # A
    if a is None:
        L += ["### A) Older-data one-shot: SKIPPED (no older data)", ""]
    else:
        m = a.m
        L += [f"### A) Older-data one-shot (real data, older-data window {_date(m, a.start)} to {_date(m, a.end)} "
              f"UTC, {len(a.rows['B&H'].metrics.months)} months; warm-up bars before it unscored; no retuning)", ""]
        L += _main_table(a.rows, a.matched, FROZEN)
        L += ["", "Rule vs its matched static (same average exposure), older data:", ""]
        L += _vs_table(a.rows, a.matched)
        L += ["", "Calendar-year split, older data (return / max DD within the year; 2018 starts 2018-01-15 19:00, "
              "2022 ends 2022-08-27 18:00):", ""]
        keys = [*RULES, "REF", "B&H", *(f"ST{lv:.0%}" for lv in STATIC_LEVELS)]
        splits = {k: year_split(m, a.rows[k].run, a.start, a.end) for k in keys}
        years = [y for y, _, _ in splits["B&H"]]
        L.append("| Year | " + " | ".join(k for k in keys) + " |")
        L.append("|---|" + "---:|" * len(keys))
        for i, y in enumerate(years):
            L.append(f"| {y} | " + " | ".join(f"{_p(splits[k][i][1], 0)} / {splits[k][i][2] * 100:.0f}%"
                                               for k in keys) + " |")
        L.append("")

    # B
    m = b.m
    L += [f"### B) Walk-forward retuning (real data, walk-forward test windows chained {_date(m, b.start)} to "
          f"{_date(m, b.end)} UTC, {len(b.folds)} folds x {TEST_MONTHS} months; train {TRAIN_MONTHS} months each)", ""]
    L += _main_table(b.rows, b.matched, {})
    L += ["", "Rule vs its matched static (same average exposure), walk-forward test windows:", ""]
    L += _vs_table(b.rows, b.matched)
    L += ["", "Picks per fold (train window -> test window; round-1 grid and selection rule on the train window "
          "only):", "",
          "| Fold | Train (UTC) | Test (UTC) | S1 | S2 vol_target, n | S3 | S4 |", "|---:|---|---|---|---|---|---|"]
    for i, (f, p) in enumerate(zip(b.folds, b.picks)):
        L.append(f"| {i} | {_date(m, f.train[0])} -> {_date(m, f.train[1])} | {_date(m, f.test[0])} -> "
                 f"{_date(m, f.test[1])} | b={p['S1']['b']} {p['S1']['cadence']} | "
                 f"{p['S2']['vol_target']}, {p['S2']['n']} | L={p['S3']['lookback']} T={p['S3']['tilt']} | "
                 f"D={p['S4']['max_dd']} H={p['S4']['window']} |")
    L += ["", "Return per test window (walk-forward test, chained curve):", "",
          "| Fold | Test starts | " + " | ".join([*RULES, "REF", "B&H"]) + " |",
          "|---:|---|" + "---:|" * 6]
    for i, f in enumerate(b.folds):
        cells = []
        for k in (*RULES, "REF", "B&H"):
            eq = b.rows[k].run.equity
            cells.append(_p(eq[f.test[1]] / eq[f.test[0] - 1] - 1))
        L.append(f"| {i} | {_date(m, f.test[0])} | " + " | ".join(cells) + " |")
    L.append("")

    # C
    L += ["### C) Exposure study: static 50/50 ETH/BTC, weekly rebalance (real data)", ""]
    for title, rows in (("older-data window " + (f"{_date(a.m, a.start)} to {_date(a.m, a.end)}" if a else ""),
                         c_old),
                        (f"dev window {_date(dev_ctx.m, dev_ctx.start)} to {_date(dev_ctx.m, dev_ctx.end)}", c_dev)):
        if rows is None:
            L += [f"{title}: skipped (no older data)", ""]
            continue
        L += [f"{title} UTC:", "",
              "| Target | Avg exposure | Return | Sharpe | Max DD | Positive months | Worst month | Cost drag |",
              "|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for lv, x in rows:
            L.append(f"| {lv:.0%} | {x.exposure * 100:.1f}% | {_p(x.compounded)} | {x.sharpe:.2f} | "
                     f"{x.max_dd * 100:.1f}% | {x.positive}/{len(x.months)} | {_p(x.worst_month)} | "
                     f"{x.cost_drag * 100:.2f}%/yr |")
        L.append("")
    L += ["Dev context, not part of the judgment (real data, dev window; frozen round-1 picks vs matched static; "
          "rule rows can be checked against results.md):", ""]
    L += _vs_table(dev_ctx.rows, dev_ctx.matched)
    L.append("")

    # Judgment
    verdicts = [judge(k, a, b) for k in (*RULES, "REF")] if a is not None else []
    L += ["### Judgment (rule fixed before results)", ""]
    if not verdicts:
        L += ["A was skipped, so no rule can meet the rule (it needs A and B). The round-1 conclusion stands.", ""]
    else:
        names = list(verdicts[0].checks)
        L += ["| Rule | " + " | ".join(names) + " | Timing skill |", "|---|" + "---|" * (len(names) + 1)]
        for v in verdicts:
            tag = " (reference, information only)" if v.key == "REF" else ""
            L.append(f"| {v.key}{tag} | " + " | ".join("yes" if v.checks[n] else "no" for n in names)
                     + f" | **{'YES' if v.timing_skill else 'no'}** |")
        skilled = [v.key for v in verdicts if v.timing_skill and v.key != "REF"]
        L += ["", ("**Result: " + ", ".join(skilled) + " met the timing-skill rule.**") if skilled else
              "**Result: no rule meets the timing-skill rule. The round-1 conclusion stands: the rules cut "
              "drawdowns mainly through lower exposure, not timing.**", ""]
    return "\n".join(L)


def stress(old_dir: str | None, dev_dir: str) -> str:
    dev = load_pre_holdout(dev_dir)
    old = load_pre_holdout(old_dir) if old_dir and Path(old_dir, "eth_1h.csv").exists() else None
    dev_ctx = one_shot(dev)
    if dev_ctx.start != DEV[0] or dev_ctx.end != DEV[1]:
        raise RuntimeError("dev window does not match round 1")
    c_dev = exposure_study(dev, DEV[0], DEV[1])
    if old is not None:
        a = one_shot(old)
        c_old = exposure_study(old, a.start, a.end)
        b = walk_forward(join_markets(old, dev))
    else:
        a = c_old = None
        b = walk_forward(dev, train_months=12, test_months=3)
    return report(a, b, c_old, c_dev, dev_ctx, old_dir or "-", dev_dir)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m agent_trader.research.stress", description=__doc__)
    ap.add_argument("--old", default="data/binance_early", help="older eth_1h.csv/btc_1h.csv (optional)")
    ap.add_argument("--dev", default="data/binance", help="dev-era eth_1h.csv/btc_1h.csv (read up to the holdout)")
    ap.add_argument("--out", default="runs/stress_results.md")
    args = ap.parse_args(argv)
    text = stress(args.old, args.dev)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(text)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

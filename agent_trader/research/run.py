"""Run the Strategist plan end to end and write a markdown report.

Order: tune every strategy on its dev grid using a market cut at the last dev bar (holdout bars
are never passed to tuning) -> freeze -> dev gate -> one holdout run, only for frozen configs
that pass the gate. Every frozen config is run twice and its fill hashes must match.
"""
from __future__ import annotations

import argparse
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Callable

from .backtest import FEE, MIN_TRADE, SLIPPAGE, Market, Run, buy_and_hold, fills_hash, load_market, simulate
from .metrics import (DEV, HOLDOUT, MONTH_BARS, N_BARS, WARMUP_END, Check, Metrics, dev_gate,
                      period_metrics, select, success_bar, verdict)
from .strategies import (DEFAULTS, S1_GRID, S2_GRID, S3_GRID, S4_GRID, drawdown_brake, dual_momentum,
                         ref_sma, trend_ladder, vol_target_ladder)


@dataclass
class Tuned:
    key: str
    title: str
    build: Callable
    fixed: dict
    grid: dict
    results: dict          # grid point -> dev Metrics
    scores: dict           # grid point -> median dev Sharpe over the point and its neighbours
    pick: tuple
    dev_run: Run
    gate: list[Check]
    hashes: tuple[str, str]
    holdout: Metrics | None = None
    bar: list[Check] | None = None
    full_hashes: tuple[str, str] | None = None

    @property
    def params(self) -> dict:
        return {**self.fixed, **dict(zip(self.grid, self.pick))}

    @property
    def dev(self) -> Metrics:
        return self.results[self.pick]

    @property
    def passed_gate(self) -> bool:
        return all(c.ok for c in self.gate)


def _twice(m: Market, build: Callable, params: dict) -> tuple[Run, tuple[str, str]]:
    """Run a frozen config twice from scratch; the fill lists must hash the same."""
    first = simulate(m, build(m, **params))
    second = simulate(m, build(m, **params))
    hashes = (fills_hash(first.fills), fills_hash(second.fills))
    if hashes[0] != hashes[1]:
        raise RuntimeError(f"not deterministic: {build.__name__} {params}")
    return first, hashes


def tune(key: str, title: str, build: Callable, grid: dict, dev_m: Market, bh_dev: Metrics,
         fixed: dict | None = None) -> Tuned:
    fixed = fixed or {}
    results = {}
    for point in product(*grid.values()):
        run = simulate(dev_m, build(dev_m, **fixed, **dict(zip(grid, point))))
        results[point] = period_metrics(dev_m, run, *DEV)
    pick, scores = select(results, list(grid.values()))
    params = {**fixed, **dict(zip(grid, pick))}
    run, hashes = _twice(dev_m, build, params)
    if period_metrics(dev_m, run, *DEV) != results[pick]:
        raise RuntimeError(f"{key}: rerun of the frozen config differs from its grid run")
    return Tuned(key, title, build, fixed, grid, results, scores, pick, run,
                 dev_gate(results[pick], bh_dev), hashes)


def _full_run(m: Market, build: Callable, params: dict, dev_run: Run) -> tuple[Run, tuple[str, str]]:
    """The one continuous run over all bars. Up to the last dev bar it must equal the dev-only run
    (a no-lookahead check on the real data: adding later bars changed nothing before them)."""
    run, hashes = _twice(m, build, params)
    if run.equity[:DEV[1] + 1] != dev_run.equity or [f for f in run.fills if f.bar <= DEV[1]] != dev_run.fills:
        raise RuntimeError(f"{build.__name__} {params}: full run differs from the dev-only run on dev bars")
    return run, hashes


def research(m: Market, source: str = "") -> str:
    if m.n != N_BARS:
        raise ValueError(f"the dev/holdout split is defined for {N_BARS} bars, got {m.n}")
    dev_m = m.head(DEV[1] + 1)
    bh_dev = period_metrics(dev_m, buy_and_hold(dev_m, DEV[0] - 1), *DEV)
    ref_dev_run, ref_dev_hashes = _twice(dev_m, ref_sma, {})
    ref_dev = period_metrics(dev_m, ref_dev_run, *DEV)

    # Build order S1, S2, S4, S3. S2 reuses S1's frozen (b, cadence), even if S1 fails its gate.
    s1 = tune("S1", "Trend ladder", trend_ladder, S1_GRID, dev_m, bh_dev)
    s2 = tune("S2", "Volatility-targeted trend ladder", vol_target_ladder, S2_GRID, dev_m, bh_dev,
              fixed=s1.params)
    s4 = tune("S4", "Drawdown brake", drawdown_brake, S4_GRID, dev_m, bh_dev)
    s3 = tune("S3", "ETH/BTC dual-momentum tilt", dual_momentum, S3_GRID, dev_m, bh_dev)
    tuned = [s1, s2, s3, s4]

    # Holdout: frozen configs only, each once (plus an identical rerun for the determinism hash).
    bh_hold = period_metrics(m, buy_and_hold(m, HOLDOUT[0] - 1), *HOLDOUT)
    ref_full, ref_full_hashes = _full_run(m, ref_sma, {}, ref_dev_run)
    ref_hold = period_metrics(m, ref_full, *HOLDOUT)
    for t in tuned:
        if t.passed_gate:
            full, t.full_hashes = _full_run(m, t.build, t.params, t.dev_run)
            t.holdout = period_metrics(m, full, *HOLDOUT)
            t.bar = success_bar(t.holdout, bh_hold)

    return _report(m, source, tuned, bh_dev, ref_dev, bh_hold, ref_hold, (ref_dev_hashes, ref_full_hashes))


# ----------------------------------------------------------------------------- report


def _date(m: Market, bar: int) -> str:
    return datetime.fromtimestamp(m.ts[bar], tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


def _p(x: float, digits: int = 1) -> str:
    return f"{x * 100:+.{digits}f}%"


ROWS: list[tuple[str, Callable[[Metrics], str]]] = [
    ("Compounded return", lambda x: _p(x.compounded)),
    ("CAGR", lambda x: _p(x.cagr)),
    ("Mean monthly return (arithmetic)", lambda x: _p(x.mean_month, 2)),
    ("Months positive / flat / negative", lambda x: f"{x.positive} / {x.flat} / {x.negative}"),
    ("Worst month", lambda x: _p(x.worst_month)),
    ("Max drawdown (hourly marks)", lambda x: f"{x.max_dd * 100:.1f}%"),
    ("Longest time under water", lambda x: f"{x.underwater_days:.0f} days"),
    ("Sharpe (daily, x sqrt(365))", lambda x: f"{x.sharpe:.2f}"),
    ("Calmar (CAGR / max DD)", lambda x: f"{x.calmar:.2f}"),
    ("Fills", lambda x: str(x.fills)),
    ("Cost drag (fees + slippage)", lambda x: f"{x.cost_drag * 100:.2f}%/yr"),
    ("Average exposure", lambda x: f"{x.exposure * 100:.0f}%"),
]


def _metric_table(columns: list[tuple[str, Metrics]]) -> list[str]:
    out = ["| Metric | " + " | ".join(name for name, _ in columns) + " |",
           "|---|" + "---:|" * len(columns)]
    for label, fmt in ROWS:
        out.append(f"| {label} | " + " | ".join(fmt(x) for _, x in columns) + " |")
    return out


def _checks(checks: list[Check]) -> list[str]:
    return [f"- {'PASS' if c.ok else 'FAIL'}: {c.name} ({c.detail})" for c in checks]


def _params(p: dict) -> str:
    return ", ".join(f"{k}={v}" for k, v in p.items())


def _months_table(m: Market, start: int, columns: list[tuple[str, Metrics]]) -> list[str]:
    out = ["| # | Month starts (UTC) | " + " | ".join(n for n, _ in columns) + " |",
           "|---:|---|" + "---:|" * len(columns)]
    for i in range(len(columns[0][1].months)):
        out.append(f"| {i + 1} | {_date(m, start + i * MONTH_BARS)} | "
                   + " | ".join(_p(x.months[i]) for _, x in columns) + " |")
    return out


def _hypothesis(t: Tuned, tuned: dict[str, Tuned], bh_dev: Metrics) -> str:
    if t.key == "S1":
        return "S1 is falsified if it fails the dev gate or the holdout bar (see the verdicts)."
    if t.key == "S2":
        s1 = tuned["S1"].dev
        better = [p for p, r in t.results.items() if r.sharpe > s1.sharpe and r.worst_month > s1.worst_month]
        found = ", ".join(_params(dict(zip(t.grid, p))) for p in better) or "none"
        return (f"Grid points that beat frozen S1 on both dev Sharpe ({s1.sharpe:.2f}) and dev worst month "
                f"({_p(s1.worst_month)}): {found}. "
                + ("Vol scaling adds value on dev." if better else "FALSIFIED: vol scaling adds nothing on dev."))
    if t.key == "S3":
        lb = t.pick[0]
        base = t.results[(lb, 0.5)].sharpe
        tilt = {tt: t.results[(lb, tt)].sharpe for tt in (0.75, 1.0)}
        beats = [tt for tt, v in tilt.items() if v > base]
        return (f"At the selected lookback {lb}: dev Sharpe T=0.5 {base:.2f}, T=0.75 {tilt[0.75]:.2f}, "
                f"T=1.0 {tilt[1.0]:.2f}. "
                + (f"Rotation beats plain absolute momentum at T={', '.join(map(str, beats))}."
                   if beats else "FALSIFIED: rotation adds nothing; this is plain absolute momentum."))
    d = t.dev
    ok = d.worst_month >= 0.6 * bh_dev.worst_month and d.max_dd <= 0.6 * bh_dev.max_dd
    return (f"Needs dev worst month and max DD both at least 40% smaller than B&H: worst month "
            f"{_p(d.worst_month)} vs B&H {_p(bh_dev.worst_month)}, max DD {d.max_dd * 100:.1f}% vs "
            f"B&H {bh_dev.max_dd * 100:.1f}%. " + ("Holds." if ok else "FALSIFIED."))


def _git_rev() -> str:
    try:
        out = subprocess.run(["git", "describe", "--always", "--dirty"], capture_output=True, text=True,
                             cwd=Path(__file__).parent, timeout=10, check=True)
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _report(m: Market, source: str, tuned: list[Tuned], bh_dev: Metrics, ref_dev: Metrics,
            bh_hold: Metrics, ref_hold: Metrics, ref_hashes) -> str:
    by_key = {t.key: t for t in tuned}
    passed = [t for t in tuned if t.bar is not None and verdict(t.bar) == "PASS"]
    near = [t for t in tuned if t.bar is not None and verdict(t.bar) == "NEAR MISS"]
    L: list[str] = [
        "# OMIN consistency strategies: results (Builder, generated)",
        "",
        f"**All numbers are real data:** Binance hourly candles for ETHUSDT and BTCUSDT ({source or 'see --data'}), "
        f"{m.n:,} aligned bars from {_date(m, 0)} to {_date(m, m.n - 1)} UTC. Nothing here is synthetic or "
        f"estimated by hand; the whole file is written by `python -m agent_trader.research` (code at "
        f"`{_git_rev()}`, branch claude/workflow-consistent-strategies). No LLM or API calls.",
        "",
        "## Verdict",
        "",
        "| Strategy | Frozen settings (picked on dev) | Dev gate | Holdout success bar |",
        "|---|---|---|---|",
    ]
    for t in tuned:
        if t.bar is None:
            hold = "not run (failed the dev gate)"
        else:
            hold = f"{verdict(t.bar)} ({sum(c.ok for c in t.bar)}/5)"
        L.append(f"| {t.key} {t.title} | {_params(t.params)} | {'PASS' if t.passed_gate else 'FAILED-DEV'} | {hold} |")
    L += ["", ("**Workflow verdict: SUCCESS.** " + ", ".join(t.key for t in passed)
               + " passed the holdout success bar; next step is 3+ months of unchanged live paper trading.")
          if passed else
          ("**Workflow verdict: nothing passes.** "
           + (f"Near miss (4 of 5, report only, do not promote): {', '.join(t.key for t in near)}. " if near else "")
           + "The standing verdict holds: hold ETH+BTC and use the best rule at most as a crash brake."), ""]

    L += [
        "## Harness (fixed before any test)",
        "",
        f"- Decision at bar t uses data up to close[t] only; it fills at open[t+1]: buys at open x "
        f"{1 + SLIPPAGE}, sells at open x {1 - SLIPPAGE} ({SLIPPAGE * 1e4:.0f} bps slippage), plus a "
        f"{FEE * 1e4:.0f} bps fee on the fill notional. A decision on the last bar never fills. "
        "No PolicyEngine (its caps would distort the rules).",
        f"- One continuous compounded run per strategy from bar 0, starting in cash; no monthly restarts. "
        f"An asset trades only if its weight must move by >= {MIN_TRADE}, or its target is 0 while held; "
        "then exactly to target, sells before buys, buys scaled down if cash is short. Marks at every hourly close.",
        "- Daily rules decide at the 23:00 UTC close of each complete day and fill at 00:00; weekly rules decide "
        "only at Sunday's close and fill Monday 00:00. REF-SMA336 decides hourly.",
        f"- Warm-up bars 0..{WARMUP_END} (never scored). Dev bars {DEV[0]}..{DEV[1]} ({_date(m, DEV[0])} to "
        f"{_date(m, DEV[1])} UTC, 30 months). Holdout bars {HOLDOUT[0]}..{HOLDOUT[1]} ({_date(m, HOLDOUT[0])} to "
        f"{_date(m, HOLDOUT[1])} UTC, 15 months; the same split as `data.split(candles, 0.3)`). A month is a "
        "720-bar window; its return runs from the previous window's last close.",
        "- Tuning used a market cut at the last dev bar, so holdout bars were never seen while tuning. Pick: highest "
        "median dev Sharpe over the cell and its direct neighbours; ties to lower dev max DD, then fewer fills.",
        "- B&H: 50/50 ETH/BTC bought at the close of the bar before each period, never rebalanced, no fees "
        "(so it shows 0 fills and 0 cost). Max DD and time under water start fresh at each period's start. "
        "Sharpe uses daily simple returns between 23:00 marks (stdev ddof 1). A flat month is |r| < 0.25%.",
        "",
        f"## Dev ({_date(m, DEV[0])} to {_date(m, DEV[1])} UTC, 30 months)",
        "",
        "Reference rows on dev: B&H and REF-SMA336 (old best rule rerun in the fixed harness).",
        "",
    ]
    L += _metric_table([("B&H 50/50", bh_dev), ("REF-SMA336", ref_dev)])
    for t in tuned:
        names = list(t.grid)
        L += ["", f"### {t.key} {t.title}", ""]
        if t.fixed:
            L.append(f"Fixed from frozen S1: {_params(t.fixed)}.")
            L.append("")
        L.append(f"Dev grid ({len(t.results)} runs; plan default {_params(DEFAULTS[t.key])}; ** = picked):")
        L.append("")
        L.append("| " + " | ".join(names) + " | Sharpe | Nbhd median Sharpe | Compounded | Max DD | Worst month "
                 "| Months +/=/- | Fills | Cost drag | Exposure |")
        L.append("|" + "---|" * len(names) + "---:|" * 9)
        for p, r in t.results.items():
            mark = "**" if p == t.pick else ""
            L.append("| " + " | ".join(f"{mark}{v}{mark}" for v in p)
                     + f" | {r.sharpe:.2f} | {t.scores[p]:.2f} | {_p(r.compounded)} | {r.max_dd * 100:.1f}% "
                     f"| {_p(r.worst_month)} | {r.positive}/{r.flat}/{r.negative} | {r.fills} "
                     f"| {r.cost_drag * 100:.2f}% | {r.exposure * 100:.0f}% |")
        L += ["", f"Frozen: {_params(t.params)}.", ""]
        L += _metric_table([(f"{t.key} (dev)", t.dev), ("B&H 50/50", bh_dev), ("REF-SMA336", ref_dev)])
        L += ["", f"Dev gate: **{'PASS' if t.passed_gate else 'FAILED-DEV (never runs on holdout)'}**", ""]
        L += _checks(t.gate)
        L += ["", f"Hypothesis check: {_hypothesis(t, by_key, bh_dev)}", "",
              "Success-bar criteria on dev, for information only (the bar is judged on holdout): "
              + "; ".join(f"{c.name.split(':')[0]}: {'yes' if c.ok else 'no'}"
                          for c in success_bar(t.dev, bh_dev)) + "."]

    L += ["", "### Monthly returns on dev (frozen configs)", ""]
    L += _months_table(m, DEV[0], [("B&H", bh_dev), ("REF", ref_dev)] + [(t.key, t.dev) for t in tuned])

    L += ["", f"## Holdout ({_date(m, HOLDOUT[0])} to {_date(m, HOLDOUT[1])} UTC, 15 months)", "",
          "Each frozen config that passed the dev gate ran once over all bars; nothing was changed after. "
          "B&H and REF-SMA336 (no tunable settings) are the references.", ""]
    ran = [t for t in tuned if t.holdout is not None]
    if not ran:
        L += ["No strategy passed the dev gate, so no strategy ran on holdout. Reference rows only:", ""]
        L += _metric_table([("B&H 50/50", bh_hold), ("REF-SMA336", ref_hold)])
    for t in ran:
        L += [f"### {t.key} {t.title} ({_params(t.params)}): **{verdict(t.bar)}** "
              f"({sum(c.ok for c in t.bar)}/5)", ""]
        L += _metric_table([(f"{t.key} (holdout)", t.holdout), ("B&H 50/50", bh_hold), ("REF-SMA336", ref_hold)])
        L += [""] + _checks(t.bar) + [""]
    if ran:
        L += ["### Monthly returns on holdout", ""]
        L += _months_table(m, HOLDOUT[0], [("B&H", bh_hold), ("REF", ref_hold)] + [(t.key, t.holdout) for t in ran])
        L.append("")

    L += ["## Determinism (each frozen config run twice from scratch; sha256 of the fill list, first 16 hex)", "",
          "| Run | Hash, run 1 | Hash, run 2 | Same |", "|---|---|---|---|"]
    rows = [("REF-SMA336 dev-only", ref_hashes[0]), ("REF-SMA336 all bars", ref_hashes[1])]
    for t in tuned:
        rows.append((f"{t.key} {_params(t.params)} dev-only", t.hashes))
        if t.full_hashes:
            rows.append((f"{t.key} {_params(t.params)} all bars", t.full_hashes))
    L += [f"| {name} | `{a}` | `{b}` | {'yes' if a == b else 'NO'} |" for name, (a, b) in rows]
    L += ["", "Every all-bars run was also checked to equal its dev-only run on every dev bar (equity marks and "
          "fills), a no-lookahead check on the real data.", ""]
    return "\n".join(L)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m agent_trader.research", description=__doc__)
    ap.add_argument("--data", default="data", help="folder with eth_1h.csv and btc_1h.csv")
    ap.add_argument("--out", default="runs/research_results.md")
    args = ap.parse_args(argv)
    text = research(load_market(args.data), source=args.data)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(text)
    print(f"wrote {args.out}")

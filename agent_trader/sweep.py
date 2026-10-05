"""Capability sweep: where does the model stop beating the dumb baseline?

Runs the LLM agent, the momentum baseline and buy-and-hold over several market
regimes (picked from real candles), decision frequencies and risk settings, and
writes every scorecard to one JSON file the control panel reads."""
from __future__ import annotations

import json
import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path

from .agent import LLMAgent, MomentumBaseline
from .config import Config, Limits
from .killswitch import KillSwitch
from .models import Candle
from .replay import run_replay

RISK = {
    "normal": Limits(),
    "strict": Limits(max_trade_pct=0.05, max_position_pct=0.15, max_trades_per_day=4, max_daily_loss_pct=0.03),
}


def pick_regimes(candles: dict[str, list[Candle]], window: int) -> dict[str, tuple[int, int]]:
    """Non-overlapping windows on the first symbol: biggest rally, biggest drop,
    flattest, and most volatile."""
    closes = [c.close for c in next(iter(candles.values()))]
    stats = []
    for start in range(0, len(closes) - window, window // 2):
        seg = closes[start:start + window]
        ret = seg[-1] / seg[0] - 1
        rets = [math.log(b / a) for a, b in zip(seg, seg[1:])]
        mean = sum(rets) / len(rets)
        vol = math.sqrt(sum((r - mean) ** 2 for r in rets) / len(rets))
        stats.append((start, ret, vol))
    chosen: dict[str, tuple[int, int]] = {}

    def take(name, key):
        for start, *_ in sorted(stats, key=key):
            if all(abs(start - s) >= window for s, _ in chosen.values()):
                chosen[name] = (start, start + window)
                return

    take("bull", lambda s: -s[1])
    take("bear", lambda s: s[1])
    take("sideways", lambda s: abs(s[1]))
    take("high_vol", lambda s: -s[2])
    return chosen


def run_cell(candles, agent, limits: Limits) -> dict:
    config = replace(Config(), limits=limits)
    card = run_replay(candles, agent, config, KillSwitch()).scorecard
    return asdict(card)


def sweep(candles, llm_factory, frequencies=(3, 6, 12), window=240, out="runs/sweep.json", workers=8) -> dict:
    regimes = pick_regimes(candles, window)
    jobs = []
    for name, (a, b) in regimes.items():
        seg = {s: v[a:b] for s, v in candles.items()}
        for risk in RISK:
            jobs.append((name, risk, "baseline", None, seg))
            freqs = frequencies if risk == "normal" else (6,)
            for f in freqs:
                jobs.append((name, risk, "haiku", f, seg))

    def work(job):
        name, risk, kind, f, seg = job
        agent = MomentumBaseline() if kind == "baseline" else LLMAgent(llm=llm_factory(f"{name}/{risk}/{f}"), decide_every=f)
        row = {"regime": name, "risk": risk, "agent": kind, "decide_every": f, **run_cell(seg, agent, RISK[risk])}
        if kind != "baseline":
            row["error"] = agent.last_error
        return row

    with ThreadPoolExecutor(workers) as pool:
        rows = list(pool.map(work, jobs))
    first = next(iter(candles.values()))
    result = {
        "regimes": {n: {"start_ts": first[a].ts, "end_ts": first[b - 1].ts,
                        "move": first[b - 1].close / first[a].close - 1} for n, (a, b) in regimes.items()},
        "rows": rows,
    }
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(result, indent=1))
    return result

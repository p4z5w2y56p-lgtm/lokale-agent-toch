"""Free (no LLM) parameter sweep of the momentum rule on REAL walk-forward windows.
Pick the best setting on dev windows only, then report it on holdout windows."""
from __future__ import annotations

import itertools
import os
import statistics
import sys
from dataclasses import replace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from agent_trader.agent import MomentumBaseline  # noqa: E402
from agent_trader.config import load_config  # noqa: E402
from agent_trader.data import align, load_csv  # noqa: E402
from agent_trader.journal import Journal  # noqa: E402
from agent_trader.killswitch import KillSwitch  # noqa: E402
from agent_trader.replay import run_replay  # noqa: E402

from walkforward import windows  # noqa: E402

candles = align({"ETH": load_csv(os.path.join(ROOT, "data/eth_1h.csv"), "ETH"), "BTC": load_csv(os.path.join(ROOT, "data/btc_1h.csv"), "BTC")})
config = replace(load_config(), fee_bps=10.0, slippage_bps=5.0)
ks = KillSwitch(".sweep_kill")
wins = windows(candles, 30 * 24, 0)
split_at = len(wins) - round(len(wins) * 0.3)


def score(agent_kw, part):
    idx = range(split_at) if part == "dev" else range(split_at, len(wins))
    ex, beat, ret = [], 0, []
    for i in idx:
        c = run_replay(wins[i], MomentumBaseline(**agent_kw), config, ks, Journal(None)).scorecard
        ex.append(c.total_return - c.buy_hold_return)
        ret.append(c.total_return)
        beat += c.excess_return > 0
    return statistics.mean(ret), statistics.mean(ex), beat, len(ex)


grid = list(itertools.product([24, 72, 168], [0.01, 0.03, 0.06], [0.05, 0.25, 0.45]))
res = []
for lb, th, fr in grid:
    kw = dict(lookback=lb, threshold=th, trade_fraction=fr)
    r, e, b, n = score(kw, "dev")
    res.append((e, kw, r, b, n))
    print(f"dev {kw} ret {r:+.4f} excess {e:+.4f} beats {b}/{n}", flush=True)
res.sort(key=lambda x: -x[0])
for e, kw, r, b, n in res[:3]:
    hr, he, hb, hn = score(kw, "holdout")
    print(f"TOP {kw}: dev ret {r:+.4f} excess {e:+.4f} beats {b}/{n} | HOLDOUT ret {hr:+.4f} excess {he:+.4f} beats {hb}/{hn}")

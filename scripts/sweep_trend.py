"""Free sweep of a mostly-invested trend rule on REAL walk-forward windows (paper money).
Hold each coin (up to its position cap) while price > SMA(n); go to cash when below.
Sim-only limits are loosened so the bot can actually be invested. Pick on dev, report holdout."""
from __future__ import annotations

import os
import statistics
import sys
from dataclasses import dataclass, replace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from agent_trader.agent import Observation  # noqa: E402
from agent_trader.config import load_config  # noqa: E402
from agent_trader.data import align, load_csv  # noqa: E402
from agent_trader.journal import Journal  # noqa: E402
from agent_trader.killswitch import KillSwitch  # noqa: E402
from agent_trader.models import Side, TradeProposal  # noqa: E402
from agent_trader.replay import run_replay  # noqa: E402
from walkforward import windows  # noqa: E402


@dataclass
class TrendHold:
    sma: int = 168
    band: float = 0.0
    target: float = 0.49

    def decide(self, obs: Observation):
        out = []
        for s, h in obs.candles.items():
            if len(h) <= self.sma:
                continue
            avg = sum(c.close for c in h[-self.sma:]) / self.sma
            p = obs.prices[s]
            held = obs.portfolio.holdings.get(s, 0.0)
            if p > avg * (1 + self.band) and held == 0:
                out.append(TradeProposal(s, Side.BUY, self.target * obs.portfolio.equity / p, p, "above sma"))
            elif p < avg * (1 - self.band) and held > 0:
                out.append(TradeProposal(s, Side.SELL, held, p, "below sma"))
        return out


candles = align({s: load_csv(os.path.join(ROOT, f"data/{s.lower()}_1h.csv"), s) for s in ("ETH", "BTC")})
base = load_config()
config = replace(base, fee_bps=10.0, slippage_bps=5.0,
                 limits=replace(base.limits, max_trade_pct=0.5, max_position_pct=0.5))
ks = KillSwitch(os.path.join(ROOT, ".sweep_kill"))
wins = windows(candles, 30 * 24, 0)
split_at = len(wins) - round(len(wins) * 0.3)


def score(kw, idx):
    ret, ex, dd = [], [], []
    for i in idx:
        c = run_replay(wins[i], TrendHold(**kw), config, ks, Journal(None)).scorecard
        ret.append(c.total_return)
        ex.append(c.total_return - c.buy_hold_return)
        dd.append(c.max_drawdown)
    return statistics.mean(ret), statistics.mean(ex), sum(e > 0 for e in ex), len(ex), max(dd)


res = []
for sma in (24, 72, 168, 336):
    for band in (0.0, 0.01, 0.03):
        kw = dict(sma=sma, band=band)
        r = score(kw, range(split_at))
        res.append((r[1], kw, r))
        print(f"dev {kw} ret {r[0]:+.4f} excess {r[1]:+.4f} beats {r[2]}/{r[3]} worstDD {r[4]:.3f}", flush=True)
res.sort(key=lambda x: -x[0])
for e, kw, r in res[:3]:
    h = score(kw, range(split_at, len(wins)))
    print(f"TOP {kw}: dev ret {r[0]:+.4f} excess {r[1]:+.4f} beats {r[2]}/{r[3]} | HOLDOUT ret {h[0]:+.4f} "
          f"excess {h[1]:+.4f} beats {h[2]}/{h[3]} worstDD {h[4]:.3f}", flush=True)

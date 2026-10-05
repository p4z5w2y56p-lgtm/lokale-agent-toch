"""Walk-forward on REAL candles: cut history into back-to-back windows, run each agent
fresh in every window, compare with buy-and-hold. Older windows = "dev", newest = "holdout".
Haiku is metered and hard-stopped at --budget-eur.

  ANTHROPIC_API_KEY=... python scripts/walkforward.py --csv data/eth_1h.csv,data/btc_1h.csv --symbols ETH,BTC
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from dataclasses import replace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent_trader.agent import LLMAgent, MomentumBaseline  # noqa: E402
from agent_trader.config import load_config  # noqa: E402
from agent_trader.data import align, load_csv  # noqa: E402
from agent_trader.journal import Journal  # noqa: E402
from agent_trader.killswitch import KillSwitch  # noqa: E402
from agent_trader.replay import run_replay  # noqa: E402

# Haiku 4.5 list price, USD per million tokens; EUR at ~0.92.
PRICE_IN, PRICE_OUT, USD_EUR = 1.0, 5.0, 0.92
# Shared meter across threads; hard stop when the project total reaches BUDGET_EUR.
USAGE_LOG = os.environ.get("USAGE_LOG")
SHARED_CAP = float(os.environ.get("BUDGET_EUR", "5"))


class BudgetExceeded(Exception):
    pass


class MeteredHaiku:
    def __init__(self, client, budget_eur, model="claude-haiku-4-5-20251001"):
        self.client, self.budget, self.model = client, budget_eur, model
        self.calls = self.tin = self.tout = 0
        self.stopped = False

    @property
    def eur(self):
        return (self.tin * PRICE_IN + self.tout * PRICE_OUT) / 1e6 * USD_EUR

    def __call__(self, system, user):
        if self.eur >= self.budget or shared_total() >= SHARED_CAP:
            self.stopped = True
            raise BudgetExceeded(f"budget reached (mine {self.eur:.2f}, shared {shared_total():.2f} EUR)")
        msg = self.client.messages.create(model=self.model, max_tokens=600, system=system,
                                          messages=[{"role": "user", "content": user}])
        self.calls += 1
        self.tin += msg.usage.input_tokens
        self.tout += msg.usage.output_tokens
        u = msg.usage
        cost = (u.input_tokens * PRICE_IN + u.output_tokens * PRICE_OUT) / 1e6 * USD_EUR
        if USAGE_LOG:
            with open(USAGE_LOG, "a") as fh:
                fh.write(json.dumps({"ts": time.time(), "model": self.model, "in": u.input_tokens,
                                     "out": u.output_tokens, "eur": round(cost, 6), "tag": "walkforward-real"}) + "\n")
        return "".join(getattr(b, "text", "") for b in msg.content)


def shared_total():
    if not USAGE_LOG or not os.path.exists(USAGE_LOG):
        return 0.0
    with open(USAGE_LOG) as fh:
        return sum(json.loads(line).get("eur", 0) for line in fh if line.strip())


def windows(candles, size, warmup):
    n = len(next(iter(candles.values())))
    out, start = [], 0
    while start + warmup + size <= n:
        out.append({s: c[start:start + warmup + size] for s, c in candles.items()})
        start += size
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--symbols", default="ETH,BTC")
    ap.add_argument("--window-days", type=int, default=30)
    ap.add_argument("--haiku-windows", type=int, default=8, help="spread evenly, newest always included")
    ap.add_argument("--decide-every", type=int, default=12)
    ap.add_argument("--holdout", type=float, default=0.3)
    ap.add_argument("--budget-eur", type=float, default=1.0)
    ap.add_argument("--fee-bps", type=float, default=10.0)
    ap.add_argument("--slippage-bps", type=float, default=5.0)
    ap.add_argument("--out", default="walkforward_results.json")
    a = ap.parse_args()

    syms = [s.strip().upper() for s in a.symbols.split(",")]
    candles = align({s: load_csv(p.strip(), s) for s, p in zip(syms, a.csv.split(","))})
    config = replace(load_config(), fee_bps=a.fee_bps, slippage_bps=a.slippage_bps)
    ks = KillSwitch(os.path.join(os.path.dirname(a.out) or ".", ".wf_kill"))
    wins = windows(candles, a.window_days * 24, 0)
    n_hold = max(1, round(len(wins) * a.holdout))
    split_at = len(wins) - n_hold

    k = min(a.haiku_windows, len(wins))
    haiku_idx = sorted({round(i * (len(wins) - 1) / max(1, k - 1)) for i in range(k)})
    llm = None
    if os.environ.get("ANTHROPIC_API_KEY") and k:
        import anthropic
        llm = MeteredHaiku(anthropic.Anthropic(), a.budget_eur)

    rows = []
    for i, w in enumerate(wins):
        first = next(iter(w.values()))[0].ts
        row = {"window": i, "start_ts": first, "set": "holdout" if i >= split_at else "dev"}
        base = run_replay(w, MomentumBaseline(), config, ks, Journal(None)).scorecard
        row.update(bh=base.buy_hold_return, base=base.total_return, base_trades=base.trades)
        if llm and i in haiku_idx:
            if llm.stopped:
                print("STOP: budget reached, skipping Haiku window", i)
            else:
                ag = LLMAgent(llm=llm, decide_every=a.decide_every)
                h = run_replay(w, ag, config, ks, Journal(None)).scorecard
                if ag.last_error:
                    row["haiku_error"] = ag.last_error[:120]
                row.update(haiku=h.total_return, haiku_trades=h.trades, haiku_dd=h.max_drawdown,
                           haiku_rejected=h.proposals_rejected, haiku_calls_total=llm.calls)
        rows.append(row)
        print(json.dumps({k2: (round(v, 4) if isinstance(v, float) else v) for k2, v in row.items()}), flush=True)

    def summary(name, key):
        out = {}
        for part in ("dev", "holdout"):
            r = [x for x in rows if x["set"] == part and key in x]
            if not r:
                continue
            ex = [x[key] - x["bh"] for x in r]
            out[part] = {"windows": len(r), "mean_return": statistics.mean(x[key] for x in r),
                         "mean_bh": statistics.mean(x["bh"] for x in r),
                         "beats_bh": sum(e > 0 for e in ex), "mean_excess": statistics.mean(ex)}
            if key == "haiku":
                out[part]["beats_baseline"] = sum(x["haiku"] > x["base"] for x in r)
        return out

    result = {"data": "REAL Binance hourly", "fee_bps": a.fee_bps, "slippage_bps": a.slippage_bps,
              "window_days": a.window_days, "baseline": summary("baseline", "base"),
              "haiku": summary("haiku", "haiku"), "rows": rows}
    if llm:
        result["haiku_usage"] = {"calls": llm.calls, "in_tokens": llm.tin, "out_tokens": llm.tout,
                                 "eur": round(llm.eur, 3)}
    with open(a.out, "w") as fh:
        json.dump(result, fh, indent=1)
    print(json.dumps({k2: v for k2, v in result.items() if k2 != "rows"}, indent=1))


if __name__ == "__main__":
    main()

"""Trading costs for the simulation: taker fee + slippage, and a gross-vs-net report.

The PaperBroker already charges `config.fee_bps` and `config.slippage_bps` on every fill
(buys fill above the price, sells below it, and the fee is taken on the notional). This
module adds the missing pieces:

* `CostModel`: the two knobs with realistic defaults (taker fee 0.1% per side = 10 bps,
  slippage 5 bps per fill), validated, settable from env vars or CLI flags.
* `summarize_costs`: how much the run paid in fees and slippage, and what the return
  would have been *gross* (same fills, no costs) versus *net* (what the account got).
* `compare_costs`: the same agent on the same data with zero costs and with costs.

Gross = net final equity + fees paid + slippage paid. It is exact for "the same fills with
zero costs" and ignores second-order effects (with costs the account is a bit smaller, so
equity-based sizing differs by a hair). `compare_costs` runs the real zero-cost replay if
you want the exact number.

Run `python -m agent_trader.costs` for a before/after table on synthetic data.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, replace
from typing import Callable, Mapping

from .agent import Agent
from .config import Config
from .killswitch import KillSwitch
from .models import Candle, Fill, Side

DEFAULT_TAKER_FEE_BPS = 10.0   # 0.1% per side (Binance spot taker, before any BNB discount)
DEFAULT_SLIPPAGE_BPS = 5.0


@dataclass(frozen=True)
class CostModel:
    taker_fee_bps: float = DEFAULT_TAKER_FEE_BPS
    slippage_bps: float = DEFAULT_SLIPPAGE_BPS

    def __post_init__(self) -> None:
        for name in ("taker_fee_bps", "slippage_bps"):
            v = getattr(self, name)
            if not math.isfinite(v) or v < 0:
                raise ValueError(f"{name} must be a finite number >= 0, got {v!r}")
        if self.slippage_bps >= 10_000:
            raise ValueError("slippage_bps must be below 10000 (100%)")

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "CostModel":
        """TAKER_FEE_BPS and SLIPPAGE_BPS override the defaults."""
        env = os.environ if env is None else env
        return cls(
            taker_fee_bps=float(env.get("TAKER_FEE_BPS", DEFAULT_TAKER_FEE_BPS)),
            slippage_bps=float(env.get("SLIPPAGE_BPS", DEFAULT_SLIPPAGE_BPS)),
        )

    def apply(self, config: Config) -> Config:
        """A copy of `config` whose fee and slippage are this model's."""
        return replace(config, fee_bps=self.taker_fee_bps, slippage_bps=self.slippage_bps)


NO_COSTS = CostModel(0.0, 0.0)


@dataclass(frozen=True)
class CostSummary:
    fills: int
    turnover: float          # total notional traded (quote currency)
    fees: float
    slippage: float
    starting_cash: float
    final_equity: float

    @property
    def total_costs(self) -> float:
        return self.fees + self.slippage

    @property
    def net_return(self) -> float:
        return self.final_equity / self.starting_cash - 1

    @property
    def gross_return(self) -> float:
        return (self.final_equity + self.total_costs) / self.starting_cash - 1

    @property
    def cost_drag(self) -> float:
        """Return points lost to costs (gross minus net)."""
        return self.gross_return - self.net_return


def slippage_paid(fill: Fill, slippage_bps: float) -> float:
    """Quote-currency cost of the slippage on one fill, versus the un-slipped price."""
    slip = slippage_bps / 10_000
    if fill.side is Side.BUY:
        reference = fill.price / (1 + slip)
        return fill.quantity * (fill.price - reference)
    reference = fill.price / (1 - slip)
    return fill.quantity * (reference - fill.price)


def summarize_costs(fills: list[Fill], equity_curve: list[float], config: Config) -> CostSummary:
    final = equity_curve[-1] if equity_curve else config.starting_cash
    return CostSummary(
        fills=len(fills),
        turnover=sum(f.quantity * f.price for f in fills),
        fees=sum(f.fee for f in fills),
        slippage=sum(slippage_paid(f, config.slippage_bps) for f in fills),
        starting_cash=config.starting_cash,
        final_equity=final,
    )


def format_cost_report(s: CostSummary, config: Config, indent: str = "") -> str:
    lines = [
        f"{indent}Costs: fee {config.fee_bps:g} bps + slippage {config.slippage_bps:g} bps per fill, {s.fills} fills",
        f"{indent}Gross return:  {s.gross_return:+.2%}   (before costs)",
        f"{indent}Net return:    {s.net_return:+.2%}   (after costs; this is the real number)",
        f"{indent}Cost drag:     {s.cost_drag:.2%}   fees {s.fees:.2f} + slippage {s.slippage:.2f} on {s.turnover:.0f} traded",
    ]
    return "\n".join(lines)


def compare_costs(
    candles: dict[str, list[Candle]],
    make_agent: Callable[[], Agent],
    config: Config,
    killswitch: KillSwitch,
    model: CostModel | None = None,
    steps_per_day: int = 24,
):
    """Run the same agent on the same data twice: zero costs, then `model` costs.
    Returns (gross_summary, gross_config, net_summary, net_config). Only for cheap agents
    (the rule-based baseline); an LLM agent would pay for two runs."""
    from .journal import Journal
    from .replay import run_replay

    model = model or CostModel()
    out = []
    for m in (NO_COSTS, model):
        cfg = m.apply(config)
        res = run_replay(candles, make_agent(), cfg, killswitch, Journal(), steps_per_day)
        out.append((summarize_costs(res.fills, res.equity_curve, cfg), cfg, res))
    (g, gc, gres), (n, nc, nres) = out
    return g, gc, gres, n, nc, nres


def main(argv: list[str] | None = None) -> int:
    """Before/after table: rule-based baseline on synthetic data, no costs vs costs."""
    import argparse
    import tempfile
    from pathlib import Path

    from .agent import MomentumBaseline
    from .data import synthetic

    p = argparse.ArgumentParser(prog="python -m agent_trader.costs")
    p.add_argument("--seeds", default="1,2,3,4,5")
    p.add_argument("--steps", type=int, default=720)
    p.add_argument("--fee-bps", type=float, default=DEFAULT_TAKER_FEE_BPS)
    p.add_argument("--slippage-bps", type=float, default=DEFAULT_SLIPPAGE_BPS)
    args = p.parse_args(argv)
    model = CostModel(args.fee_bps, args.slippage_bps)
    starts = {"ETH": 2000.0, "WBTC": 60000.0}

    print(f"SYNTHETIC DATA, rule-based baseline, {args.steps} hourly bars, "
          f"fee {model.taker_fee_bps:g} bps + slippage {model.slippage_bps:g} bps")
    print(f"{'seed':>4} {'fills':>5} {'no-cost':>9} {'with-cost':>10} {'gross*':>8} {'fees':>7} {'slip':>7} {'drag':>7}")
    tot_g = tot_n = 0.0
    seeds = [int(s) for s in args.seeds.split(",")]
    with tempfile.TemporaryDirectory() as tmp:
        ks = KillSwitch(Path(tmp) / "KILL")
        for seed in seeds:
            candles = {s: synthetic(s, args.steps, seed=seed + i, start_price=starts[s])
                       for i, s in enumerate(("ETH", "WBTC"))}
            g, _, _, n, _, _ = compare_costs(candles, MomentumBaseline, Config(), ks, model)
            tot_g += g.net_return
            tot_n += n.net_return
            print(f"{seed:>4} {n.fills:>5} {g.net_return:>+9.2%} {n.net_return:>+10.2%} "
                  f"{n.gross_return:>+8.2%} {n.fees:>7.2f} {n.slippage:>7.2f} {n.cost_drag:>7.2%}")
    k = len(seeds)
    print(f"{'mean':>4} {'':>5} {tot_g / k:>+9.2%} {tot_n / k:>+10.2%}")
    print("no-cost = replay with fee 0 and slippage 0; with-cost = same agent and data with costs;")
    print("gross* = with-cost run's net equity plus the fees and slippage it paid (same fills, no costs).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

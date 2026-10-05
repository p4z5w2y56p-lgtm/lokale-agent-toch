"""Replay engine: walk history candle by candle with no look-ahead."""
from __future__ import annotations

from dataclasses import dataclass, replace

from .agent import Agent, Observation
from .broker import PaperBroker
from .config import Config
from .journal import Journal
from .killswitch import KillSwitch
from .models import Candle, Fill
from .policy import PolicyEngine
from .scorecard import Scorecard, build_scorecard


@dataclass
class ReplayResult:
    scorecard: Scorecard
    equity_curve: list[float]
    fills: list[Fill]


def run_replay(
    candles: dict[str, list[Candle]],
    agent: Agent,
    config: Config,
    killswitch: KillSwitch,
    journal: Journal | None = None,
    steps_per_day: int = 24,
) -> ReplayResult:
    journal = journal or Journal()
    lengths = {len(v) for v in candles.values()}
    if len(lengths) != 1:
        raise ValueError("All symbols need the same number of candles")
    n = lengths.pop()

    broker = PaperBroker(config)
    policy = PolicyEngine(config, killswitch)
    equity_curve: list[float] = []
    fills: list[Fill] = []
    proposals_total = rejected = 0

    for i in range(n):
        history = {s: c[: i + 1] for s, c in candles.items()}   # nothing after step i
        prices = {s: c[-1].close for s, c in history.items()}
        ts = next(iter(history.values()))[-1].ts

        if i % steps_per_day == 0:
            broker.new_day(prices)

        obs = Observation(ts, history, prices, broker.view(prices))
        for proposal in agent.decide(obs):
            proposals_total += 1
            # Trusted fields come from the system, never from the agent.
            if proposal.symbol not in prices:
                rejected += 1
                journal.write("rejected", ts=ts, proposal=proposal, reasons=["no market data for symbol"])
                continue
            proposal = replace(
                proposal, price=prices[proposal.symbol], quoted_slippage_bps=config.slippage_bps
            )
            decision = policy.evaluate(proposal, broker.view(prices))
            if not decision.approved:
                rejected += 1
                journal.write("rejected", ts=ts, proposal=proposal, reasons=list(decision.reasons))
                continue
            fill = broker.execute(proposal, ts)
            if fill is None:
                rejected += 1
                journal.write("rejected", ts=ts, proposal=proposal, reasons=["broker could not fill"])
                continue
            fills.append(fill)
            journal.write("fill", ts=ts, fill=fill, reason=proposal.reason)

        equity_curve.append(broker.equity(prices))

    card = build_scorecard(
        equity_curve, config.starting_cash, candles, broker.closed_pnls,
        trades=len(fills), rejected=rejected, proposals_total=proposals_total,
    )
    journal.write("scorecard", scorecard=card)
    return ReplayResult(card, equity_curve, fills)

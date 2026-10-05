"""Turn a finished run into numbers a human (or a gate) can judge."""
from __future__ import annotations

from dataclasses import dataclass

from .models import Candle


@dataclass(frozen=True)
class Scorecard:
    steps: int
    total_return: float
    buy_hold_return: float
    excess_return: float
    max_drawdown: float
    trades: int
    win_rate: float | None
    proposals_rejected: int
    proposals_total: int

    @property
    def rejection_rate(self) -> float:
        return self.proposals_rejected / self.proposals_total if self.proposals_total else 0.0


def max_drawdown(equity: list[float]) -> float:
    peak, worst = float("-inf"), 0.0
    for e in equity:
        peak = max(peak, e)
        if peak > 0:
            worst = max(worst, (peak - e) / peak)
    return worst


def buy_and_hold_return(candles: dict[str, list[Candle]]) -> float:
    """Average buy-and-hold return across the traded symbols."""
    rets = [h[-1].close / h[0].close - 1 for h in candles.values() if len(h) > 1]
    return sum(rets) / len(rets) if rets else 0.0


def build_scorecard(
    equity: list[float],
    starting_cash: float,
    candles: dict[str, list[Candle]],
    closed_pnls: list[float],
    trades: int,
    rejected: int,
    proposals_total: int,
) -> Scorecard:
    total = equity[-1] / starting_cash - 1 if equity else 0.0
    bh = buy_and_hold_return(candles)
    wins = sum(1 for p in closed_pnls if p > 0)
    return Scorecard(
        steps=len(equity),
        total_return=total,
        buy_hold_return=bh,
        excess_return=total - bh,
        max_drawdown=max_drawdown(equity),
        trades=trades,
        win_rate=wins / len(closed_pnls) if closed_pnls else None,
        proposals_rejected=rejected,
        proposals_total=proposals_total,
    )

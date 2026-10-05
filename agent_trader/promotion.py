"""Stage gates. ADVISORY ONLY: this reports whether the agent has earned a promotion;
a human moves the stage. Nothing here can change the stage or unlock real money."""
from __future__ import annotations

from dataclasses import dataclass

from .config import Stage
from .scorecard import Scorecard


@dataclass(frozen=True)
class Gate:
    min_steps: int
    min_trades: int
    min_excess_return: float
    max_drawdown: float
    max_rejection_rate: float


GATES: dict[Stage, Gate] = {
    # Replay -> live paper: beat plain buy-and-hold over a decent history, stay in bounds.
    Stage.REPLAY: Gate(min_steps=500, min_trades=20, min_excess_return=0.0,
                       max_drawdown=0.15, max_rejection_rate=0.10),
    # Live paper -> testnet: a longer, stricter run (about two weeks of hourly bars).
    Stage.LIVE_PAPER: Gate(min_steps=24 * 14, min_trades=30, min_excess_return=0.0,
                           max_drawdown=0.10, max_rejection_rate=0.05),
    # Testnet -> mainnet: needs flawless execution; judged by a human, not by this table.
}


@dataclass(frozen=True)
class PromotionResult:
    eligible: bool
    failures: tuple[str, ...]
    note: str


def check_promotion(stage: Stage, card: Scorecard) -> PromotionResult:
    gate = GATES.get(stage)
    if gate is None:
        return PromotionResult(False, (), "No automatic gate for this stage; a human decides.")

    failures: list[str] = []
    if card.steps < gate.min_steps:
        failures.append(f"only {card.steps} steps, need {gate.min_steps}")
    if card.trades < gate.min_trades:
        failures.append(f"only {card.trades} trades, need {gate.min_trades}")
    if card.excess_return < gate.min_excess_return:
        failures.append(f"excess return {card.excess_return:+.1%} vs buy-and-hold, need >= {gate.min_excess_return:+.1%}")
    if card.max_drawdown > gate.max_drawdown:
        failures.append(f"max drawdown {card.max_drawdown:.1%}, limit {gate.max_drawdown:.1%}")
    if card.rejection_rate > gate.max_rejection_rate:
        failures.append(f"{card.rejection_rate:.1%} of proposals broke the rules, limit {gate.max_rejection_rate:.1%}")

    next_stage = Stage(stage + 1).name
    note = f"Eligible to consider {next_stage}. A human makes the call." if not failures else "Not ready."
    return PromotionResult(not failures, tuple(failures), note)

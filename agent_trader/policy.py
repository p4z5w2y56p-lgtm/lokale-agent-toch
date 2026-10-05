"""The policy engine: the part of the system the agent cannot talk its way past.

It is plain code with no LLM in it. Every rule that fails adds a reason, so the
journal shows exactly why a trade was refused.
"""
from __future__ import annotations

import math

from .config import Config, Stage
from .killswitch import KillSwitch
from .models import Decision, PortfolioView, Side, TradeProposal


class PolicyEngine:
    def __init__(self, config: Config, killswitch: KillSwitch) -> None:
        self.config = config
        self.killswitch = killswitch

    def evaluate(self, p: TradeProposal, pv: PortfolioView) -> Decision:
        limits = self.config.limits
        reasons: list[str] = []

        if self.killswitch.engaged():
            reasons.append("kill switch is engaged")

        if self.config.stage >= Stage.MAINNET_CAPPED and not self.config.mainnet_unlocked:
            reasons.append("stage is locked (real-money stages need an explicit human unlock)")

        if p.symbol not in limits.allowed_symbols:
            reasons.append(f"{p.symbol!r} is not on the allowlist")

        if not _positive_finite(p.quantity) or not _positive_finite(p.price):
            reasons.append("quantity and price must be finite positive numbers")
            return Decision(False, tuple(reasons))  # the maths below is meaningless

        if not _positive_finite(pv.equity):
            reasons.append("equity is not positive")
            return Decision(False, tuple(reasons))

        notional = p.quantity * p.price

        # sells only reduce exposure, so a stop-loss must never be blocked by size
        if p.side is Side.BUY and notional > limits.max_trade_pct * pv.equity:
            reasons.append(
                f"trade is {notional / pv.equity:.1%} of equity, over max_trade_pct "
                f"{limits.max_trade_pct:.1%}"
            )

        if pv.trades_today >= limits.max_trades_per_day:
            reasons.append(f"already {pv.trades_today} trades today, over max_trades_per_day")

        if pv.day_start_equity > 0:
            loss = (pv.day_start_equity - pv.equity) / pv.day_start_equity
            if loss > limits.max_daily_loss_pct:
                reasons.append(
                    f"down {loss:.1%} today, over max_daily_loss_pct "
                    f"{limits.max_daily_loss_pct:.1%}: trading halted for the day"
                )

        if p.quoted_slippage_bps > limits.max_slippage_bps:
            reasons.append(
                f"quoted slippage {p.quoted_slippage_bps:.0f} bps is over max_slippage_bps "
                f"{limits.max_slippage_bps:.0f}"
            )

        held = pv.holdings.get(p.symbol, 0.0)
        if p.side is Side.BUY:
            # budget for fee AND slippage so an approved buy can never overdraw cash
            cost = notional * (1 + (self.config.fee_bps + self.config.slippage_bps) / 10_000)
            if cost > pv.cash:
                reasons.append(f"needs {cost:.2f} cash but only {pv.cash:.2f} available")
            position_after = (held * p.price + notional) / pv.equity
            if position_after > limits.max_position_pct:
                reasons.append(
                    f"position would be {position_after:.1%} of equity, over "
                    f"max_position_pct {limits.max_position_pct:.1%}"
                )
        else:
            if p.quantity > held:
                reasons.append(f"sells {p.quantity} {p.symbol} but holds only {held}")

        return Decision(approved=not reasons, reasons=tuple(reasons))


def _positive_finite(x: float) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x) and x > 0

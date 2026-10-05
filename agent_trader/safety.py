"""Hard safety rules: max loss per day, max drawdown from peak, kill switch.

Plain code, no LLM. They only ever block buys: once a limit is hit the bot may still
sell to reduce risk, but it cannot add new exposure. A drawdown breach trips the
kill switch, which then stays engaged for the rest of the run.
"""
from __future__ import annotations

from .config import Limits
from .killswitch import KillSwitch
from .models import PortfolioView


def drawdown(pv: PortfolioView) -> float:
    return (pv.peak_equity - pv.equity) / pv.peak_equity if pv.peak_equity > 0 else 0.0


def daily_loss(pv: PortfolioView) -> float:
    return (pv.day_start_equity - pv.equity) / pv.day_start_equity if pv.day_start_equity > 0 else 0.0


def check(limits: Limits, pv: PortfolioView, killswitch: KillSwitch) -> None:
    """Trip the kill switch when drawdown from peak breaches the limit."""
    dd = drawdown(pv)
    if dd > limits.max_drawdown_pct:
        killswitch.trip(
            f"drawdown {dd:.1%} from peak {pv.peak_equity:.2f} is over max_drawdown_pct "
            f"{limits.max_drawdown_pct:.1%}"
        )


def safety_reasons(limits: Limits, pv: PortfolioView, killswitch: KillSwitch) -> list[str]:
    """Reasons a new buy must be refused. Empty list means the safety rules allow it."""
    check(limits, pv, killswitch)
    reasons: list[str] = []
    if killswitch.engaged():
        reasons.append(f"kill switch is engaged ({killswitch.reason or 'manual'}): no new buys")
    loss = daily_loss(pv)
    if loss > limits.max_daily_loss_pct:
        reasons.append(
            f"down {loss:.1%} today, over max_daily_loss_pct "
            f"{limits.max_daily_loss_pct:.1%}: no new buys for the rest of the day"
        )
    return reasons

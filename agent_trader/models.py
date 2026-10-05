"""Plain data types shared by every other module."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"


@dataclass(frozen=True)
class Candle:
    ts: int
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0


@dataclass(frozen=True)
class TradeProposal:
    symbol: str
    side: Side
    quantity: float
    price: float
    reason: str = ""
    quoted_slippage_bps: float = 0.0


@dataclass(frozen=True)
class Decision:
    approved: bool
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class Fill:
    ts: int
    symbol: str
    side: Side
    quantity: float
    price: float
    fee: float


@dataclass(frozen=True)
class PortfolioView:
    """Read-only snapshot the policy engine judges a proposal against."""

    cash: float
    holdings: dict[str, float] = field(default_factory=dict)
    equity: float = 0.0
    day_start_equity: float = 0.0
    trades_today: int = 0

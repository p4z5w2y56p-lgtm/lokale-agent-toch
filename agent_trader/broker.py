"""Paper broker: fake money, realistic frictions (fees + slippage)."""
from __future__ import annotations

from dataclasses import dataclass

from .config import Config
from .models import Fill, PortfolioView, Side, TradeProposal


@dataclass
class _Position:
    qty: float = 0.0
    avg_cost: float = 0.0  # per unit, including buy fees


class PaperBroker:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.cash = config.starting_cash
        self.positions: dict[str, _Position] = {}
        self.closed_pnls: list[float] = []
        self.trades_today = 0
        self.day_start_equity = config.starting_cash

    # --- valuation -----------------------------------------------------
    def holdings(self) -> dict[str, float]:
        return {s: p.qty for s, p in self.positions.items() if p.qty > 0}

    def equity(self, prices: dict[str, float]) -> float:
        return self.cash + sum(p.qty * prices.get(s, 0.0) for s, p in self.positions.items())

    def view(self, prices: dict[str, float]) -> PortfolioView:
        return PortfolioView(
            cash=self.cash,
            holdings=self.holdings(),
            equity=self.equity(prices),
            day_start_equity=self.day_start_equity,
            trades_today=self.trades_today,
        )

    def new_day(self, prices: dict[str, float]) -> None:
        self.trades_today = 0
        self.day_start_equity = self.equity(prices)

    # --- execution -----------------------------------------------------
    def execute(self, p: TradeProposal, ts: int) -> Fill | None:
        """Fill a proposal at its price +/- slippage. Returns None if it cannot be
        filled (never lets cash or holdings go negative)."""
        slip = self.config.slippage_bps / 10_000
        fee_rate = self.config.fee_bps / 10_000
        pos = self.positions.setdefault(p.symbol, _Position())

        if p.side is Side.BUY:
            price = p.price * (1 + slip)
            notional = p.quantity * price
            fee = notional * fee_rate
            if notional + fee > self.cash:
                return None
            total_cost = pos.qty * pos.avg_cost + notional + fee
            pos.qty += p.quantity
            pos.avg_cost = total_cost / pos.qty
            self.cash -= notional + fee
        else:
            if p.quantity > pos.qty:
                return None
            price = p.price * (1 - slip)
            notional = p.quantity * price
            fee = notional * fee_rate
            self.closed_pnls.append((price - pos.avg_cost) * p.quantity - fee)
            pos.qty -= p.quantity
            if pos.qty <= 1e-12:
                pos.qty, pos.avg_cost = 0.0, 0.0
            self.cash += notional - fee

        self.trades_today += 1
        return Fill(ts, p.symbol, p.side, p.quantity, price, fee)

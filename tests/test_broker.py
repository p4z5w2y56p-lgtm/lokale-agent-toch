import pytest

from agent_trader.broker import PaperBroker
from agent_trader.config import Config
from agent_trader.models import Side, TradeProposal


def make(**kw):
    return PaperBroker(Config(fee_bps=30, slippage_bps=10, starting_cash=1000.0, **kw))


def test_buy_applies_slippage_and_fee():
    b = make()
    fill = b.execute(TradeProposal("ETH", Side.BUY, 0.1, 2000.0), ts=0)
    assert fill.price == pytest.approx(2000 * 1.001)
    assert fill.fee == pytest.approx(0.1 * 2000 * 1.001 * 0.003)
    assert b.cash == pytest.approx(1000 - 0.1 * fill.price - fill.fee)
    assert b.holdings() == {"ETH": 0.1}


def test_cannot_overdraw_cash():
    b = make()
    assert b.execute(TradeProposal("ETH", Side.BUY, 10.0, 2000.0), ts=0) is None
    assert b.cash == 1000.0 and b.holdings() == {}


def test_cannot_sell_what_you_do_not_have():
    b = make()
    assert b.execute(TradeProposal("ETH", Side.SELL, 1.0, 2000.0), ts=0) is None


def test_round_trip_loses_to_fees_when_price_is_flat():
    b = make()
    b.execute(TradeProposal("ETH", Side.BUY, 0.1, 2000.0), ts=0)
    b.execute(TradeProposal("ETH", Side.SELL, 0.1, 2000.0), ts=1)
    assert b.cash < 1000.0
    assert b.closed_pnls[0] < 0
    assert b.holdings() == {}


def test_profitable_round_trip_counts_as_win():
    b = make()
    b.execute(TradeProposal("ETH", Side.BUY, 0.1, 2000.0), ts=0)
    b.execute(TradeProposal("ETH", Side.SELL, 0.1, 2200.0), ts=1)
    assert b.closed_pnls[0] > 0
    assert b.cash > 1000.0


def test_new_day_resets_counters():
    b = make()
    b.execute(TradeProposal("ETH", Side.BUY, 0.01, 2000.0), ts=0)
    assert b.trades_today == 1
    b.new_day({"ETH": 2000.0})
    assert b.trades_today == 0

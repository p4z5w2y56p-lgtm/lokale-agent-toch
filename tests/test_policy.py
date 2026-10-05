import math

import pytest

from agent_trader.config import Config, Limits, Stage
from agent_trader.killswitch import KillSwitch
from agent_trader.models import PortfolioView, Side, TradeProposal
from agent_trader.policy import PolicyEngine


@pytest.fixture
def ks(tmp_path):
    return KillSwitch(tmp_path / "KILL")


@pytest.fixture
def engine(ks):
    return PolicyEngine(Config(), ks)


def view(**kw):
    base = dict(cash=1000.0, holdings={}, equity=1000.0, day_start_equity=1000.0, trades_today=0)
    base.update(kw)
    return PortfolioView(**base)


def buy(qty=0.04, price=2000.0, symbol="ETH", **kw):
    return TradeProposal(symbol=symbol, side=Side.BUY, quantity=qty, price=price, **kw)


def sell(qty=0.01, price=2000.0, symbol="ETH", **kw):
    return TradeProposal(symbol=symbol, side=Side.SELL, quantity=qty, price=price, **kw)


def test_reasonable_buy_is_approved(engine):
    d = engine.evaluate(buy(qty=0.04), view())  # $80 = 8% of equity
    assert d.approved, d.reasons


def test_kill_switch_blocks_everything(engine, ks):
    ks.engage("test")
    d = engine.evaluate(buy(), view())
    assert not d.approved
    assert any("kill switch" in r for r in d.reasons)


def test_locked_stage_is_rejected(ks):
    engine = PolicyEngine(Config(stage=Stage.MAINNET_CAPPED), ks)
    d = engine.evaluate(buy(), view())
    assert not d.approved
    assert any("locked" in r for r in d.reasons)


def test_unknown_token_is_rejected(engine):
    d = engine.evaluate(buy(symbol="SCAMCOIN"), view())
    assert not d.approved
    assert any("allowlist" in r for r in d.reasons)


def test_trade_too_big_is_rejected(engine):
    d = engine.evaluate(buy(qty=0.1), view())  # $200 = 20% > 10%
    assert not d.approved
    assert any("max_trade_pct" in r for r in d.reasons)


def test_position_cap_is_rejected(engine):
    held = {"ETH": 0.14}  # $280 = 28% of equity already
    d = engine.evaluate(buy(qty=0.04), view(holdings=held))  # +$80 -> 36% > 30%
    assert not d.approved
    assert any("max_position_pct" in r for r in d.reasons)


def test_daily_trade_count_is_rejected(engine):
    d = engine.evaluate(buy(), view(trades_today=10))
    assert not d.approved
    assert any("max_trades_per_day" in r for r in d.reasons)


def test_daily_loss_halts_trading(engine):
    d = engine.evaluate(buy(), view(equity=940.0, day_start_equity=1000.0))  # -6%
    assert not d.approved
    assert any("max_daily_loss_pct" in r for r in d.reasons)


def test_slippage_cap_is_rejected(engine):
    d = engine.evaluate(buy(quoted_slippage_bps=200.0), view())
    assert not d.approved
    assert any("max_slippage_bps" in r for r in d.reasons)


def test_cannot_buy_with_cash_you_do_not_have(engine):
    d = engine.evaluate(buy(qty=0.04), view(cash=10.0))
    assert not d.approved
    assert any("cash" in r for r in d.reasons)


def test_cannot_sell_more_than_held(engine):
    d = engine.evaluate(sell(qty=0.05), view(holdings={"ETH": 0.01}))
    assert not d.approved
    assert any("holds" in r for r in d.reasons)


def test_sell_of_held_amount_is_approved(engine):
    d = engine.evaluate(sell(qty=0.01), view(holdings={"ETH": 0.02}))
    assert d.approved, d.reasons


def test_multiple_reasons_are_all_reported(engine):
    d = engine.evaluate(buy(symbol="SCAMCOIN", qty=1.0), view(trades_today=99))
    assert not d.approved
    assert len(d.reasons) >= 3


@pytest.mark.parametrize("qty", [0.0, -1.0, math.nan, math.inf])
def test_hostile_quantity_is_rejected(engine, qty):
    d = engine.evaluate(buy(qty=qty), view())
    assert not d.approved


@pytest.mark.parametrize("price", [0.0, -5.0, math.nan, math.inf])
def test_hostile_price_is_rejected(engine, price):
    d = engine.evaluate(buy(price=price), view())
    assert not d.approved


def test_zero_equity_is_rejected(engine):
    d = engine.evaluate(buy(), view(equity=0.0, cash=0.0))
    assert not d.approved

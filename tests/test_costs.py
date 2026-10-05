import pytest

from agent_trader.agent import MomentumBaseline
from agent_trader.broker import PaperBroker
from agent_trader.config import Config
from agent_trader.costs import (
    NO_COSTS,
    CostModel,
    compare_costs,
    format_cost_report,
    slippage_paid,
    summarize_costs,
)
from agent_trader.data import synthetic
from agent_trader.killswitch import KillSwitch
from agent_trader.models import Fill, Side, TradeProposal


def data(n=400, seed=1):
    return {
        "ETH": synthetic("ETH", n, seed=seed),
        "WBTC": synthetic("WBTC", n, seed=seed + 1, start_price=60000.0),
    }


def test_defaults_are_ten_bps_fee_and_five_bps_slippage():
    m = CostModel()
    assert (m.taker_fee_bps, m.slippage_bps) == (10.0, 5.0)
    cfg = Config()
    assert (cfg.fee_bps, cfg.slippage_bps) == (10.0, 5.0)


def test_apply_overrides_config_only_for_costs():
    cfg = CostModel(25, 12).apply(Config(starting_cash=500.0))
    assert (cfg.fee_bps, cfg.slippage_bps, cfg.starting_cash) == (25, 12, 500.0)


def test_from_env_and_validation():
    assert CostModel.from_env({}) == CostModel()
    assert CostModel.from_env({"TAKER_FEE_BPS": "7", "SLIPPAGE_BPS": "2"}) == CostModel(7, 2)
    with pytest.raises(ValueError):
        CostModel(-1, 5)
    with pytest.raises(ValueError):
        CostModel(10, float("nan"))
    with pytest.raises(ValueError):
        CostModel(10, 10_000)
    with pytest.raises(ValueError):
        CostModel.from_env({"TAKER_FEE_BPS": "abc"})


def test_every_simulated_trade_pays_fee_and_slippage():
    b = PaperBroker(CostModel(10, 5).apply(Config(starting_cash=1000.0)))
    buy = b.execute(TradeProposal("ETH", Side.BUY, 0.1, 1000.0), 0)
    assert buy.price == pytest.approx(1000.0 * 1.0005)
    assert buy.fee == pytest.approx(0.1 * 1000.0 * 1.0005 * 0.001)
    sell = b.execute(TradeProposal("ETH", Side.SELL, 0.1, 1000.0), 1)
    assert sell.price == pytest.approx(1000.0 * 0.9995)
    assert sell.fee > 0
    # round trip at an unchanged price loses roughly 2 * (10 + 5) bps of notional
    loss = 1000.0 - b.cash
    assert loss == pytest.approx(100.0 * 2 * 0.0015, rel=0.02)


def test_slippage_paid_matches_fill_prices():
    buy = Fill(0, "ETH", Side.BUY, 2.0, 1000.5, 1.0)       # 5 bps above 1000
    sell = Fill(0, "ETH", Side.SELL, 2.0, 999.5, 1.0)      # 5 bps below 1000
    assert slippage_paid(buy, 5.0) == pytest.approx(2.0 * 0.5, rel=1e-3)
    assert slippage_paid(sell, 5.0) == pytest.approx(2.0 * 0.5, rel=1e-3)
    assert slippage_paid(buy, 0.0) == pytest.approx(0.0)


def test_summary_gross_is_net_plus_costs():
    fills = [Fill(0, "ETH", Side.BUY, 1.0, 100.05, 0.10), Fill(1, "ETH", Side.SELL, 1.0, 99.95, 0.10)]
    s = summarize_costs(fills, [1000.0, 998.0], Config(starting_cash=1000.0, fee_bps=10, slippage_bps=5))
    assert s.fees == pytest.approx(0.20)
    assert s.total_costs == pytest.approx(s.fees + s.slippage)
    assert s.net_return == pytest.approx(-0.002)
    assert s.gross_return == pytest.approx(-0.002 + s.total_costs / 1000.0)
    assert s.cost_drag == pytest.approx(s.total_costs / 1000.0)
    assert "Gross return" in format_cost_report(s, Config())
    assert "Net return" in format_cost_report(s, Config())


def test_no_fills_means_no_drag():
    s = summarize_costs([], [1000.0, 1000.0], Config(starting_cash=1000.0))
    assert s.total_costs == 0 and s.gross_return == s.net_return == 0.0


def test_costs_reduce_net_return_and_gross_matches_zero_cost_run(tmp_path):
    ks = KillSwitch(tmp_path / "KILL")
    g, _, _, n, _, _ = compare_costs(data(), MomentumBaseline, Config(), ks, CostModel())
    assert n.fills > 0
    assert n.net_return < g.net_return                       # costs hurt
    assert n.fees > 0 and n.slippage > 0
    assert n.gross_return == pytest.approx(g.net_return, abs=5e-4)   # gross* ~ true zero-cost run
    assert g.total_costs == 0


def test_zero_cost_model_charges_nothing(tmp_path):
    ks = KillSwitch(tmp_path / "KILL")
    _, _, _, n, _, _ = compare_costs(data(), MomentumBaseline, Config(), ks, NO_COSTS)
    assert n.total_costs == 0 and n.cost_drag == 0

from dataclasses import replace

import pytest

from agent_trader.data import synthetic
from agent_trader.research.backtest import START_CASH, Market, fills_hash, simulate
from agent_trader.research.metrics import Metrics, drawdown, select
from agent_trader.research.strategies import (drawdown_brake, dual_momentum, ref_sma, trend_ladder,
                                              vol_target_ladder)

BUILDERS = {
    "S1 daily": lambda m: trend_ladder(m, 10, "daily"),
    "S1 weekly": lambda m: trend_ladder(m, 20, "weekly"),
    "S2": lambda m: vol_target_ladder(m, 10, "daily", 0.6, 20),
    "S3": lambda m: dual_momentum(m, 30, 0.75),
    "S4": lambda m: drawdown_brake(m, 0.10, 30),
    "REF-SMA336": lambda m: ref_sma(m),
}


def _market(days: int = 300) -> Market:
    """Two seeded random walks starting 1970-01-01 00:00 UTC (whole days, so every day is complete)."""
    eth = synthetic("ETH", days * 24, seed=1, start_price=1500, vol=0.012)
    btc = synthetic("BTC", days * 24, seed=2, start_price=20000, vol=0.008)
    return Market([c.ts for c in eth], {"ETH": [c.open for c in eth], "BTC": [c.open for c in btc]},
                  {"ETH": [c.close for c in eth], "BTC": [c.close for c in btc]})


def _unchanged_up_to(build, m: Market, t: int) -> bool:
    """Multiply every bar after t by 10: all target weights decided at or before t must stay."""
    base = {k: v for k, v in build(m).items() if k <= t}
    shocked = {k: v for k, v in build(m.scaled_after(t, 10.0)).items() if k <= t}
    assert base, "the check needs decisions at or before t"
    return shocked == base


@pytest.mark.parametrize("name", BUILDERS)
def test_no_lookahead(name):
    m = _market()
    for t in (3000, 4007, 5555):  # 4007 is a 23:00 bar, i.e. a daily decision bar
        assert _unchanged_up_to(BUILDERS[name], m, t), f"{name} looked ahead at t={t}"


def test_no_lookahead_check_catches_a_leak():
    def leaky(m):  # peeks at the next close (an hourly doubling never happens unshocked)
        return {t: {"ETH": 0.5 if m.close["ETH"][t + 1] > 2 * m.close["ETH"][t] else 0.0, "BTC": 0.0}
                for t in range(m.n - 1)}
    assert not _unchanged_up_to(leaky, _market(), 3000)


def _tiny(n: int = 6, gap: float = 0.5) -> Market:
    """Opens differ from the previous close, so a fill at the close would show."""
    return Market([i * 3600 for i in range(n)],
                  {"ETH": [100.0 + i for i in range(n)], "BTC": [200.0] * n},
                  {"ETH": [100.0 + i + gap for i in range(n)], "BTC": [200.0] * n})


def test_decision_at_t_fills_at_next_open_and_pays_15_bps():
    m = _tiny()
    run = simulate(m, {2: {"ETH": 0.5, "BTC": 0.0}})
    assert [(f.bar, f.symbol, f.side) for f in run.fills] == [(3, "ETH", "buy")]
    f = run.fills[0]
    open3 = m.open["ETH"][3]
    assert f.price == pytest.approx(open3 * 1.0005)
    assert f.qty * open3 == pytest.approx(0.5 * START_CASH)
    paid_bps = (f.qty * (f.price - open3) + f.fee) / (f.qty * open3) * 1e4
    assert paid_bps == pytest.approx(15.0, abs=0.01)
    assert run.equity[:3] == [START_CASH] * 3  # nothing happens before the fill bar
    expected = START_CASH - f.qty * f.price - f.fee + f.qty * m.close["ETH"][3]
    assert run.equity[3] == pytest.approx(expected)


def test_round_trip_costs_about_30_bps_and_last_bar_never_fills():
    m = Market([i * 3600 for i in range(8)], {"ETH": [100.0] * 8, "BTC": [50.0] * 8},
               {"ETH": [100.0] * 8, "BTC": [50.0] * 8})
    run = simulate(m, {1: {"ETH": 0.5, "BTC": 0.0}, 3: {"ETH": 0.0, "BTC": 0.0}, 7: {"ETH": 0.5, "BTC": 0.0}})
    assert [(f.bar, f.side) for f in run.fills] == [(2, "buy"), (4, "sell")]
    assert run.equity[-1] / START_CASH - 1 == pytest.approx(-0.5 * 0.0030, abs=1e-6)


def test_trade_threshold_full_exit_and_cash_scaling():
    m = Market([i * 3600 for i in range(8)], {"ETH": [100.0] * 8, "BTC": [50.0] * 8},
               {"ETH": [100.0] * 8, "BTC": [50.0] * 8})
    run = simulate(m, {0: {"ETH": 0.5, "BTC": 0.5},     # all-in: buys scaled to the cash
                       2: {"ETH": 0.46, "BTC": 0.5},    # move < 0.05: no trade
                       4: {"ETH": 0.0, "BTC": 0.47}})   # exit ETH fully, BTC move < 0.05: keep
    assert [(f.bar, f.symbol, f.side) for f in run.fills] == [(1, "ETH", "buy"), (1, "BTC", "buy"), (5, "ETH", "sell")]
    assert run.exposure[1] == pytest.approx(1.0)        # no cash left, and none overdrawn
    assert run.exposure[5] == pytest.approx(0.5, abs=0.002)


@pytest.mark.parametrize("name", BUILDERS)
def test_determinism_hash(name):
    m = _market()
    first = simulate(m, BUILDERS[name](m)).fills
    second = simulate(_market(), BUILDERS[name](_market())).fills
    assert first, "a determinism check needs fills"
    assert fills_hash(first) == fills_hash(second)
    assert fills_hash(first) != fills_hash(first[:-1])


def test_drawdown_and_time_under_water():
    dd, days = drawdown([100, 110, 99, 105, 111, 90], [i * 3600 for i in range(6)])
    assert dd == pytest.approx(1 - 90 / 111)
    assert days == pytest.approx(3 / 24)  # 110 at hour 1, first back above at hour 4


def test_select_uses_neighbourhood_median_not_the_single_best_cell():
    base = Metrics(compounded=0, cagr=0, mean_month=0, months=(), positive=0, flat=0, negative=0,
                   worst_month=0, max_dd=0.1, underwater_days=0, sharpe=0, calmar=0, fills=1,
                   cost_drag=0, exposure=0)
    sharpe = {(1, "a"): 0.1, (2, "a"): 1.0, (3, "a"): 0.2, (1, "b"): 0.3, (2, "b"): 0.4, (3, "b"): 0.5}
    pick, scores = select({p: replace(base, sharpe=s) for p, s in sharpe.items()}, [(1, 2, 3), ("a", "b")])
    assert scores[(2, "a")] == pytest.approx(0.3)  # lone spike among weak neighbours
    assert pick == (3, "a") and scores[pick] == pytest.approx(0.5)

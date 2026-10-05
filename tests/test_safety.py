from agent_trader.agent import MomentumBaseline
from agent_trader.config import Config, Limits
from agent_trader.killswitch import KillSwitch
from agent_trader.models import Candle, PortfolioView, Side, TradeProposal
from agent_trader.policy import PolicyEngine
from agent_trader.replay import run_replay


def view(**kw):
    base = dict(cash=500.0, holdings={"ETH": 0.05}, equity=1000.0,
                day_start_equity=1000.0, trades_today=0, peak_equity=1000.0)
    base.update(kw)
    return PortfolioView(**base)


def buy():
    return TradeProposal("ETH", Side.BUY, 0.02, 2000.0)


def sell():
    return TradeProposal("ETH", Side.SELL, 0.04, 2000.0)


def engine(tmp_path, **limits):
    ks = KillSwitch(tmp_path / "KILL")
    return PolicyEngine(Config(limits=Limits(**limits)), ks), ks


def test_kill_switch_blocks_buys_but_allows_sells(tmp_path):
    e, ks = engine(tmp_path)
    ks.engage("test")
    assert not e.evaluate(buy(), view()).approved
    assert e.evaluate(sell(), view()).approved


def test_daily_loss_blocks_buys_not_sells(tmp_path):
    e, _ = engine(tmp_path)
    pv = view(equity=940.0, peak_equity=1000.0)  # -6% today, -6% drawdown
    d = e.evaluate(buy(), pv)
    assert not d.approved and any("max_daily_loss_pct" in r for r in d.reasons)
    assert e.evaluate(sell(), pv).approved


def test_drawdown_trips_kill_switch_and_it_stays_tripped(tmp_path):
    e, ks = engine(tmp_path, max_drawdown_pct=0.10)
    pv = view(equity=880.0, day_start_equity=880.0, peak_equity=1000.0)  # -12% from peak
    d = e.evaluate(buy(), pv)
    assert not d.approved
    assert ks.engaged() and "max_drawdown_pct" in ks.reason
    # recovered equity does not release it: only a human can
    assert not e.evaluate(buy(), view(equity=1000.0, day_start_equity=1000.0)).approved
    assert not (tmp_path / "KILL").exists()  # automatic trip never writes the human switch file


def test_small_drawdown_is_fine(tmp_path):
    e, ks = engine(tmp_path, max_drawdown_pct=0.10)
    assert e.evaluate(buy(), view(equity=950.0, day_start_equity=950.0)).approved
    assert not ks.engaged()


def crash(n=300):
    """Rises 30% then falls 60%: any long position draws down hard."""
    out, price = [], 2000.0
    for i in range(n):
        price *= 1.004 if i < n // 3 else 0.995
        out.append(Candle(i * 3600, "ETH", price, price, price, price))
    return {"ETH": out}


def test_replay_reports_drawdown_trip(tmp_path):
    ks = KillSwitch(tmp_path / "KILL")
    config = Config(limits=Limits(max_drawdown_pct=0.01, max_position_pct=0.9, max_trade_pct=0.9),
                    fee_bps=0, slippage_bps=0)
    agent = MomentumBaseline(trade_fraction=0.8, threshold=0.01)
    r = run_replay(crash(), agent, config, ks)
    assert r.halted and "max_drawdown_pct" in r.halted
    buys_after = [f for f in r.fills if f.side is Side.BUY and f.ts > int(r.halted.split("ts=")[1].rstrip(")"))]
    assert buys_after == []

from types import SimpleNamespace

import pytest

from agent_trader.agent import MomentumBaseline
from agent_trader.broker import PaperBroker
from agent_trader.budget import BudgetExceeded, BudgetMeter, cost_eur
from agent_trader.cli import main
from agent_trader.config import Config
from agent_trader.killswitch import KillSwitch
from agent_trader.llm import AnthropicLLM
from agent_trader.models import Candle, Side, TradeProposal
from agent_trader.paper_live import PaperLive, PriceSourceError


# --- fees and slippage ---------------------------------------------------
def test_defaults_are_realistic():
    c = Config()
    assert 5 <= c.fee_bps <= 30 and 1 <= c.slippage_bps <= 20


def test_round_trip_at_flat_price_loses_money():
    b = PaperBroker(Config())
    buy = b.execute(TradeProposal("ETH", Side.BUY, 0.1, 2000.0), ts=0)
    sell = b.execute(TradeProposal("ETH", Side.SELL, 0.1, 2000.0), ts=1)
    assert buy and sell
    assert b.cash < 1000.0
    expected = 0.1 * 2000 * (2 * Config().slippage_bps + 2 * Config().fee_bps) / 10_000
    assert 1000.0 - b.cash == pytest.approx(expected, rel=0.01)
    assert b.closed_pnls[0] < 0


def test_replay_cli_prints_fees_and_accepts_overrides(capsys, tmp_path):
    rc = main(["replay", "--steps", "200", "--fee-bps", "25", "--slippage-bps", "3",
               "--journal", str(tmp_path / "j.jsonl")])
    out = capsys.readouterr().out
    assert rc == 0 and "Fees paid:" in out and "fee 25 bps" in out and "slippage 3 bps" in out


# --- paper trading on live prices ----------------------------------------
class FakeSource:
    def __init__(self):
        self.n = 30

    def __call__(self, symbol, hours):
        base = 2000.0 if symbol == "ETH" else 60000.0
        # strong uptrend so the baseline buys
        return [Candle(i * 3600, symbol, base * 1.01 ** i, base * 1.01 ** i, base * 1.01 ** i, base * 1.01 ** i)
                for i in range(self.n)]



def test_paper_live_tick_trades_and_skips_stale_bar(tmp_path):
    src = FakeSource()
    from agent_trader.journal import Journal
    j = Journal()
    live = PaperLive(MomentumBaseline(), Config(), ["ETH", "WBTC"], src, journal=j)
    s1 = live.tick()
    assert s1["new_bar"] and s1["fills"] >= 1 and live.fees_paid > 0
    s2 = live.tick()
    assert not s2["new_bar"] and s2["fills"] == 0
    src.n += 1
    assert live.tick()["new_bar"]
    assert any(e["event"] == "paper_tick" for e in j.events)


def test_paper_live_run_max_ticks_sleeps_between():
    sleeps = []
    live = PaperLive(MomentumBaseline(), Config(), ["ETH"], FakeSource())
    assert live.run(3, interval_s=7, sleep=sleeps.append) == 3
    assert sleeps == [7, 7]


def test_paper_live_empty_source_fails_clearly():
    live = PaperLive(MomentumBaseline(), Config(), ["ETH"], lambda s, h: [])
    with pytest.raises(PriceSourceError):
        live.tick()


def test_paper_cli_network_blocked(monkeypatch, capsys, tmp_path):
    import agent_trader.paper_live as pl

    def blocked(symbol, hours):
        raise PriceSourceError("Binance answered HTTP 403")
    monkeypatch.setattr(pl, "binance_source", blocked)
    rc = main(["paper", "--once", "--journal", str(tmp_path / "p.jsonl")])
    assert rc == 1 and "403" in capsys.readouterr().err


# --- budget ----------------------------------------------------------------
def test_cost_eur_haiku_prices():
    assert cost_eur(1_000_000, 0) == pytest.approx(0.92)
    assert cost_eur(0, 1_000_000) == pytest.approx(4.60)


def test_budget_warns_once_and_raises(tmp_path, capsys):
    path = tmp_path / "b.json"
    m = BudgetMeter(path, cap_eur=0.92)          # cap = 1M input tokens
    m.record(550_000, 0)
    m.record(10_000, 0)
    err = capsys.readouterr().err
    assert err.count("50%") == 1 and "80%" not in err
    m.record(260_000, 0)
    assert "80%" in capsys.readouterr().err
    with pytest.raises(BudgetExceeded):
        m.record(200_000, 0)
    again = BudgetMeter(path, cap_eur=0.92)      # persisted
    assert again.spent_eur == pytest.approx(m.spent_eur)
    with pytest.raises(BudgetExceeded):
        again.check()


def test_budget_env_defaults(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_BUDGET_FILE", str(tmp_path / "x.json"))
    monkeypatch.setenv("AGENT_BUDGET_EUR", "2.5")
    m = BudgetMeter()
    assert m.cap_eur == 2.5 and m.path == tmp_path / "x.json"


def test_anthropic_llm_records_usage(tmp_path):
    msg = SimpleNamespace(content=[SimpleNamespace(text="{}")],
                          usage=SimpleNamespace(input_tokens=1000, output_tokens=200))
    client = SimpleNamespace(messages=SimpleNamespace(create=lambda **kw: msg))
    m = BudgetMeter(tmp_path / "b.json", cap_eur=5)
    assert AnthropicLLM(client, budget=m)("s", "u") == "{}"
    assert m.state["calls"] == 1 and m.spent_eur == pytest.approx(cost_eur(1000, 200))


def test_budget_cli(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("AGENT_BUDGET_FILE", str(tmp_path / "b.json"))
    assert main(["budget"]) == 0
    assert "EUR 0.0000 of 5.00" in capsys.readouterr().out

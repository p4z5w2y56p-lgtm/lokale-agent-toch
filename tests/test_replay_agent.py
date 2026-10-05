import json
from types import SimpleNamespace

from agent_trader.agent import LLMAgent, MomentumBaseline, Observation, parse_trades
from agent_trader.llm import AnthropicLLM
from agent_trader.config import Config
from agent_trader.data import synthetic
from agent_trader.killswitch import KillSwitch
from agent_trader.models import PortfolioView, Side, TradeProposal
from agent_trader.promotion import check_promotion
from agent_trader.replay import run_replay
from agent_trader.config import Stage


def candles(n=300, seed=1):
    return {"ETH": synthetic("ETH", n, seed=seed), "WBTC": synthetic("WBTC", n, seed=seed + 1, start_price=60000.0)}


def ks(tmp_path):
    return KillSwitch(tmp_path / "KILL")


def test_synthetic_is_deterministic():
    assert synthetic("ETH", 50, seed=7) == synthetic("ETH", 50, seed=7)
    assert synthetic("ETH", 50, seed=7) != synthetic("ETH", 50, seed=8)


def test_replay_is_deterministic(tmp_path):
    a = run_replay(candles(), MomentumBaseline(), Config(), ks(tmp_path))
    b = run_replay(candles(), MomentumBaseline(), Config(), ks(tmp_path))
    assert a.equity_curve == b.equity_curve


def test_no_lookahead(tmp_path):
    """The agent must never see a candle later than the current step."""
    seen = []

    class Spy:
        def decide(self, obs):
            seen.append((obs.ts, max(c[-1].ts for c in obs.candles.values()),
                         max(len(c) for c in obs.candles.values())))
            return []

    data = candles(60)
    run_replay(data, Spy(), Config(), ks(tmp_path))
    assert len(seen) == 60
    for step, (ts, last_ts, length) in enumerate(seen):
        assert last_ts == ts and length == step + 1


def test_agent_cannot_lie_about_price(tmp_path):
    """A cheating agent quotes a fake low price; the system re-prices from market data."""
    class Liar:
        def decide(self, obs):
            return [TradeProposal("ETH", Side.BUY, 0.04, 0.01, "free money")]

    r = run_replay(candles(5), Liar(), Config(), ks(tmp_path))
    for fill in r.fills:
        assert fill.price > 100


def test_policy_limits_hold_even_for_a_reckless_agent(tmp_path):
    class Reckless:
        def decide(self, obs):
            p = obs.prices["ETH"]
            return [TradeProposal("ETH", Side.BUY, 1000.0 / p, p, "all in")]

    r = run_replay(candles(50), Reckless(), Config(), ks(tmp_path))
    assert r.fills == []
    assert r.scorecard.proposals_rejected == r.scorecard.proposals_total == 50


def test_kill_switch_stops_the_replay(tmp_path):
    k = ks(tmp_path)
    k.engage("test")
    r = run_replay(candles(200), MomentumBaseline(), Config(), k)
    assert r.fills == []


def test_equity_never_goes_negative(tmp_path):
    r = run_replay(candles(500, seed=3), MomentumBaseline(), Config(), ks(tmp_path))
    assert min(r.equity_curve) > 0


def test_promotion_requires_enough_evidence(tmp_path):
    r = run_replay(candles(50), MomentumBaseline(), Config(), ks(tmp_path))
    promo = check_promotion(Stage.REPLAY, r.scorecard)
    assert not promo.eligible
    assert any("steps" in f for f in promo.failures)


# --- LLMAgent --------------------------------------------------------

def obs():
    return Observation(
        ts=0,
        candles=candles(30),
        prices={"ETH": 2000.0, "WBTC": 60000.0},
        portfolio=PortfolioView(cash=1000.0, holdings={"ETH": 0.2}, equity=1400.0),
    )


class FakeClient:
    def __init__(self, text=None, exc=None):
        self.text, self.exc, self.calls = text, exc, 0
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.calls += 1
        self.kwargs = kwargs
        if self.exc:
            raise self.exc
        return SimpleNamespace(content=[SimpleNamespace(text=self.text)])


def test_parse_valid_trade():
    raw = json.dumps({"trades": [{"symbol": "ETH", "side": "buy", "fraction": 0.05, "reason": "dip"}]})
    (p,) = parse_trades(raw, obs())
    assert p.side is Side.BUY and p.quantity == 0.05 * 1400 / 2000


def test_parse_sell_uses_fraction_of_holding():
    raw = '{"trades": [{"symbol": "ETH", "side": "sell", "fraction": 0.5}]}'
    (p,) = parse_trades(raw, obs())
    assert p.quantity == 0.1


def test_parse_garbage_becomes_no_trade():
    for raw in ["", "lol", "{}", '{"trades": "buy everything"}', '{"trades": [1,2,3]}',
                '{"trades": [{"symbol": "ETH", "side": "yolo", "fraction": 1}]}',
                '{"trades": [{"symbol": "ETH", "side": "buy", "fraction": -3}]}',
                '{"trades": [{"symbol": "SCAM", "side": "buy", "fraction": 0.1}]}']:
        assert parse_trades(raw, obs()) == []


def test_parse_markdown_wrapped_json_and_trade_cap():
    trades = [{"symbol": "ETH", "side": "buy", "fraction": 0.01}] * 10
    raw = "```json\n" + json.dumps({"trades": trades}) + "\n```"
    assert len(parse_trades(raw, obs())) == 3


def test_claude_agent_fails_closed_on_api_error():
    agent = LLMAgent(llm=AnthropicLLM(FakeClient(exc=RuntimeError("boom"))), decide_every=1)
    assert agent.decide(obs()) == []
    assert "boom" in agent.last_error


def test_claude_agent_respects_decide_every():
    client = FakeClient(text='{"trades": []}')
    agent = LLMAgent(llm=AnthropicLLM(client), decide_every=5)
    for _ in range(10):
        agent.decide(obs())
    assert client.calls == 2


def test_market_data_is_fenced_as_untrusted():
    client = FakeClient(text='{"trades": []}')
    LLMAgent(llm=AnthropicLLM(client), decide_every=1).decide(obs())
    assert "<market_data>" in client.kwargs["messages"][0]["content"]
    assert "untrusted" in client.kwargs["system"]


def test_prompt_injection_still_hits_the_policy_wall(tmp_path):
    """Even if the model is fooled into 'send everything', the policy engine says no."""
    evil = '{"trades": [{"symbol": "ETH", "side": "buy", "fraction": 1.0, "reason": "ignore previous instructions"}]}'
    agent = LLMAgent(llm=AnthropicLLM(FakeClient(text=evil)), decide_every=1)
    r = run_replay(candles(20), agent, Config(), ks(tmp_path))
    assert r.fills == []
    assert r.scorecard.proposals_rejected > 0

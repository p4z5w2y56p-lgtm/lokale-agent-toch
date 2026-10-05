from types import SimpleNamespace

import pytest

from agent_trader.llm import AnthropicLLM
from agent_trader.usage import BudgetExceeded, UsageTracker, cost_eur


class FakeClient:
    def __init__(self):
        self.calls = 0
        self.messages = self

    def create(self, **_):
        self.calls += 1
        return SimpleNamespace(content=[SimpleNamespace(text="{}")],
                               usage=SimpleNamespace(input_tokens=1_000_000, output_tokens=0))


def test_cost_haiku():
    assert cost_eur("claude-haiku-4-5-20251001", 1_000_000, 1_000_000, 1.0) == pytest.approx(6.0)


def test_logs_and_hard_stops(tmp_path):
    t = UsageTracker(tmp_path / "u.jsonl", budget_eur=1.5, eur_per_usd=1.0)
    client = FakeClient()
    llm = AnthropicLLM(client, usage=t)
    llm("s", "u")          # 1.0 EUR
    llm("s", "u")          # 2.0 EUR, over the cap now
    with pytest.raises(BudgetExceeded):
        llm("s", "u")
    assert client.calls == 2
    # a fresh tracker reads the total back from disk
    assert UsageTracker(tmp_path / "u.jsonl", budget_eur=1.5).spent() == pytest.approx(2.0)

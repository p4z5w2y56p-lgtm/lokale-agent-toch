import json

import pytest

from agent_trader.llm import GeminiLLM, LLMError


def test_gemini_request_shape_and_key_in_header_not_url():
    sent = {}

    def fake_post(req):
        sent["url"], sent["headers"], sent["body"] = req.full_url, dict(req.header_items()), json.loads(req.data)
        return {"candidates": [{"content": {"parts": [{"text": '{"trades": '}, {"text": "[]}"}]}}]}

    llm = GeminiLLM(api_key="SECRET123", model="gemini-test", post=fake_post)
    assert llm("sys prompt", "user msg") == '{"trades": []}'
    assert "SECRET123" not in sent["url"]
    assert sent["headers"]["X-goog-api-key"] == "SECRET123"
    assert "gemini-test:generateContent" in sent["url"]
    assert sent["body"]["system_instruction"]["parts"][0]["text"] == "sys prompt"
    assert sent["body"]["contents"][0]["parts"][0]["text"] == "user msg"


def test_gemini_requires_a_key():
    with pytest.raises(LLMError):
        GeminiLLM(api_key="")


@pytest.mark.parametrize("response", [{}, {"candidates": []}, {"candidates": [{"content": {}}]}])
def test_gemini_bad_response_is_an_error(response):
    llm = GeminiLLM(api_key="k", post=lambda req: response)
    with pytest.raises(LLMError):
        llm("s", "u")


def test_gemini_failure_makes_agent_do_nothing():
    from tests.test_replay_agent import obs
    from agent_trader.agent import LLMAgent

    def boom(req):
        raise LLMError("Gemini HTTP 429")

    agent = LLMAgent(llm=GeminiLLM(api_key="k", post=boom), decide_every=1)
    assert agent.decide(obs()) == []
    assert "429" in agent.last_error

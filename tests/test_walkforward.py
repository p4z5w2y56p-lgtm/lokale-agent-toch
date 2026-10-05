from agent_trader.agent import LLMAgent, MomentumBaseline
from agent_trader.config import Config
from agent_trader.data import synthetic
from agent_trader.walkforward import MeteredLLM, format_report, make_folds, run_walkforward


def series(seeds=2, n=600):
    return {f"s{k}": {"ETH": synthetic("ETH", n, seed=k)} for k in range(seeds)}


def test_folds_do_not_overlap_and_test_follows_train():
    folds = make_folds(1000, 400, 200)
    assert len(folds) == 3
    for tr, te in folds:
        assert tr.stop == te.start and te.stop - te.start == 200
    assert [te.start for _, te in folds] == [400, 600, 800]


def test_baseline_runs_on_every_fold_train_and_test():
    r = run_walkforward(series(), {"baseline": MomentumBaseline}, Config(), 200, 100)
    assert len(r.folds) == 2 * 4 * 2
    parts = {(s.agent, s.part): s for s in r.summaries()}
    assert parts[("baseline", "test")].n == 8
    assert "SYNTHETIC" in format_report(r, "SYNTHETIC test")


def test_llm_limited_to_few_test_folds_and_budget_cap_drops_fold():
    calls = []

    def fake(system, user):
        calls.append(1)
        return '{"trades": []}'

    meter = MeteredLLM(fake, max_eur=1.0)
    agents = {"baseline": MomentumBaseline, "claude": lambda: LLMAgent(llm=meter, decide_every=50)}
    r = run_walkforward(series(), agents, Config(), 200, 100, llm_agents=frozenset({"claude"}),
                        llm_folds=2, meter=meter)
    llm_rows = [f for f in r.folds if f.agent == "claude"]
    assert len(llm_rows) == 2 and all(f.part == "test" for f in llm_rows)

    broke = MeteredLLM(fake, max_eur=0.0)
    agents["claude"] = lambda: LLMAgent(llm=broke, decide_every=50)
    r = run_walkforward(series(), agents, Config(), 200, 100, llm_agents=frozenset({"claude"}),
                        llm_folds=2, meter=broke)
    assert not [f for f in r.folds if f.agent == "claude"]
    assert "cap" in r.llm_note

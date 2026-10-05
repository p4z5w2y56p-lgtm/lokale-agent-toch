import json

import pytest

from agent_trader.budget import BudgetAlerts, read_usage_total


def write_usage(path, costs, extra=""):
    lines = [json.dumps({"model": "haiku", "tokens": 10, "cost_eur": c}) for c in costs]
    path.write_text("\n".join(lines) + "\n" + extra)


def make(tmp_path, budget=5.0, out=None):
    out = [] if out is None else out
    return BudgetAlerts(budget, tmp_path / "alerts.jsonl", emit=out.append), out


def test_read_usage_total_sums_cost_eur(tmp_path):
    f = tmp_path / "usage.jsonl"
    write_usage(f, [0.5, 0.25, 1.0])
    t = read_usage_total(f)
    assert t.cost_eur == pytest.approx(1.75) and t.records == 3 and t.skipped == 0


def test_read_usage_total_skips_junk_and_missing_file(tmp_path):
    f = tmp_path / "usage.jsonl"
    write_usage(f, [1.0], extra='\nnot json\n{"tokens": 5}\n{"cost_eur": -3}\n{"cost_eur": "x"}\n{"cost_eur": true}\n{"cost_eur": 2')
    t = read_usage_total(f)
    assert t.cost_eur == 1.0 and t.records == 1 and t.skipped == 6
    assert read_usage_total(tmp_path / "nope.jsonl").cost_eur == 0.0


def test_nothing_fires_below_half(tmp_path):
    a, out = make(tmp_path)
    assert a.check(2.49) == [] and out == []
    assert not (tmp_path / "alerts.jsonl").exists()


def test_fires_at_50_80_and_100_once_each(tmp_path):
    a, out = make(tmp_path)
    assert [x.threshold for x in a.check(2.5)] == [0.5]
    assert a.check(2.6) == []                       # not again
    assert [x.threshold for x in a.check(4.0)] == [0.8]
    assert a.check(4.5) == []
    hard = a.check(5.0)
    assert [x.level for x in hard] == ["hard_stop"]
    assert a.check(9.0) == []
    assert len(out) == 3 and "HARD STOP" in out[2]


def test_jump_past_several_thresholds_fires_all_in_order(tmp_path):
    a, out = make(tmp_path)
    assert [x.threshold for x in a.check(6.0)] == [0.5, 0.8, 1.0]
    assert len(out) == 3


def test_exhausted_is_true_at_and_over_budget(tmp_path):
    a, _ = make(tmp_path)
    assert not a.exhausted(4.99) and a.exhausted(5.0) and a.exhausted(7.0)


def test_alerts_persist_and_survive_restart(tmp_path):
    a, _ = make(tmp_path)
    a.check(4.2)
    lines = (tmp_path / "alerts.jsonl").read_text().splitlines()
    assert [json.loads(x)["threshold"] for x in lines] == [0.5, 0.8]
    assert json.loads(lines[0])["budget_eur"] == 5.0
    b, out = make(tmp_path)                         # new process, same file
    assert b.check(4.3) == [] and out == []
    assert [x.threshold for x in b.check(5.5)] == [1.0]


def test_changed_budget_starts_alerts_over(tmp_path):
    a, _ = make(tmp_path, budget=5.0)
    a.check(5.0)
    b, out = make(tmp_path, budget=10.0)
    assert [x.threshold for x in b.check(5.0)] == [0.5]


def test_check_file_reads_the_usage_log(tmp_path):
    f = tmp_path / "usage.jsonl"
    write_usage(f, [1.0, 1.6])
    a, out = make(tmp_path)
    total = a.check_file(f)
    assert total.cost_eur == pytest.approx(2.6) and len(out) == 1 and "50%" in out[0]
    write_usage(f, [1.0, 1.6, 1.5])
    a.check_file(f)
    assert len(out) == 2 and "80%" in out[1]


def test_default_budget_is_five_euro_and_env_override(tmp_path):
    assert BudgetAlerts(alerts_path=None).budget_eur == 5.0
    assert BudgetAlerts.from_env({"BUDGET_EUR": "12"}, alerts_path=None).budget_eur == 12.0


def test_invalid_budget_rejected():
    for bad in (0, -1, float("nan")):
        with pytest.raises(ValueError):
            BudgetAlerts(bad, None)
    with pytest.raises(ValueError):
        BudgetAlerts(5.0, None, thresholds=(1.5,))

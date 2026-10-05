"""Fast training loop: replay many windows in parallel, let a coach rewrite the
playbook from the results, and repeat until the agent makes the profit target on
held-out test windows it never trained on, or the budget cap stops it.

The budget cap always wins over the target: a bot that never reaches the target
cannot spend past the cap."""
from __future__ import annotations

import json
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .agent import LLMAgent
from .config import Config
from .killswitch import KillSwitch
from .replay import run_replay
from .usage import BudgetExceeded

COACH_PROMPT = """You coach a crypto trading agent. Below is its current playbook and how it
did on several training windows (return vs buy-and-hold, trades, rejected proposals).
Rewrite the playbook so it makes more money on unseen data. Keep it short (under 40
lines), concrete, with numeric rules the agent can apply to recent closes. The safety
limits are fixed: max 10% of equity per trade, max 30% per token. Reply with ONLY the
new playbook in markdown."""


def _windows(n_total: int, length: int, count: int, lo: int, hi: int, rng: random.Random) -> list[tuple[int, int]]:
    return [(s, s + length) for s in (rng.randrange(lo, hi - length) for _ in range(count))]


def train(candles, make_llm, usage, *, target=0.10, rounds=20, decide_every=12, train_len=240,
          test_len=720, n_train=4, n_test=3, seed=7, out_dir="runs/train", workers=8,
          start_playbook: Path = Path(__file__).with_name("playbook.md"), log=print) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    n = len(next(iter(candles.values())))
    cut = int(n * 0.7)                              # train on the past, test on the newest 30%
    rng = random.Random(seed)
    tests = [(cut + i * (n - cut - test_len) // max(n_test - 1, 1),) for i in range(n_test)]
    tests = [(a[0], a[0] + test_len) for a in tests]
    playbook = start_playbook.read_text()
    history: list[dict] = []
    stop = "max rounds"

    def run(win, pb_path, tag):
        a, b = win
        seg = {s: v[a:b] for s, v in candles.items()}
        agent = LLMAgent(llm=make_llm(tag), playbook_path=pb_path, decide_every=decide_every)
        card = run_replay(seg, agent, Config(), KillSwitch()).scorecard
        if agent.last_error and "budget" in agent.last_error:
            raise BudgetExceeded(agent.last_error)
        return {"ret": card.total_return, "bh": card.buy_hold_return, "trades": card.trades,
                "rejected": card.proposals_rejected, "proposals": card.proposals_total}

    for r in range(1, rounds + 1):
        pb_path = out / f"playbook_r{r}.md"
        pb_path.write_text(playbook)
        trains = _windows(n, train_len, n_train, 0, cut, rng)
        try:
            with ThreadPoolExecutor(workers) as pool:
                tr = list(pool.map(lambda w: run(w, pb_path, f"train r{r}"), trains))
                te = list(pool.map(lambda w: run(w, pb_path, f"test r{r}"), tests))
        except BudgetExceeded:
            stop = "budget cap"
            break
        mean_test = sum(x["ret"] for x in te) / len(te)
        row = {"round": r, "train": tr, "test": te, "mean_test": mean_test,
               "min_test": min(x["ret"] for x in te), "spent_eur": usage.spent()}
        history.append(row)
        log(f"round {r}: test mean {mean_test:+.2%} (min {row['min_test']:+.2%}), "
            f"train mean {sum(x['ret'] for x in tr) / len(tr):+.2%}, spent {usage.spent():.2f} EUR")
        if mean_test >= target:
            stop = "target reached"
            break
        report = json.dumps({"train_windows": tr}, indent=0)
        try:
            playbook = make_llm(f"coach r{r}", max_tokens=1200)(COACH_PROMPT, f"<playbook>{playbook}</playbook>\n<results>{report}</results>")
        except BudgetExceeded:
            stop = "budget cap"
            break
    result = {"stop": stop, "target": target, "rounds": history, "spent_eur": usage.spent(),
              "best_round": max(history, key=lambda h: h["mean_test"])["round"] if history else None}
    (out / "result.json").write_text(json.dumps(result, indent=1))
    return result

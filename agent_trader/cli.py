"""Command line: replay, status, kill, unkill."""
from __future__ import annotations

import argparse
import os
import sys

from .agent import LLMAgent, MomentumBaseline
from .config import ConfigError, load_config
from .data import load_csv, synthetic
from .journal import Journal
from .killswitch import KillSwitch
from .llm import AnthropicLLM, GeminiLLM
from .promotion import check_promotion
from .replay import run_replay


def _build_candles(args: argparse.Namespace) -> dict:
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    if args.csv:
        paths = [p.strip() for p in args.csv.split(",")]
        if len(paths) != len(symbols):
            sys.exit("--csv needs one file per symbol, comma separated, same order as --symbols")
        return {s: load_csv(p, s) for s, p in zip(symbols, paths)}
    starts = {"ETH": 2000.0, "WBTC": 60000.0}
    return {
        s: synthetic(s, args.steps, seed=args.seed + i, start_price=starts.get(s, 100.0))
        for i, s in enumerate(symbols)
    }


def _cmd_replay(args: argparse.Namespace) -> int:
    try:
        config = load_config()
    except ConfigError as exc:
        print(f"Config refused: {exc}", file=sys.stderr)
        return 2
    candles = _build_candles(args)
    if args.agent == "claude":
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print("ANTHROPIC_API_KEY is not set. Add it to the environment secrets first.", file=sys.stderr)
            return 2
        import anthropic

        llm = AnthropicLLM(anthropic.Anthropic(), os.environ.get("AGENT_MODEL", "claude-sonnet-5-5"))
        agent = LLMAgent(llm=llm, decide_every=args.decide_every)
    elif args.agent == "gemini":
        if not os.environ.get("GEMINI_API_KEY"):
            print("GEMINI_API_KEY is not set. Add it to the environment secrets first.", file=sys.stderr)
            return 2
        llm = GeminiLLM(os.environ["GEMINI_API_KEY"], os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"))
        agent = LLMAgent(llm=llm, decide_every=args.decide_every)
    else:
        agent = MomentumBaseline()

    killswitch = KillSwitch()
    result = run_replay(candles, agent, config, killswitch, Journal(args.journal))
    card = result.scorecard
    print(f"Agent: {args.agent}   Stage: {config.stage.name}   Steps: {card.steps}")
    print(f"Return:        {card.total_return:+.2%}")
    print(f"Buy-and-hold:  {card.buy_hold_return:+.2%}")
    print(f"Excess:        {card.excess_return:+.2%}")
    print(f"Max drawdown:  {card.max_drawdown:.2%}")
    print(f"Trades:        {card.trades}   Win rate: "
          f"{'n/a' if card.win_rate is None else f'{card.win_rate:.0%}'}")
    print(f"Rejected:      {card.proposals_rejected}/{card.proposals_total} proposals")
    if getattr(agent, "last_error", None):
        print(f"Agent error:   {agent.last_error}")
    promo = check_promotion(config.stage, card)
    print(f"\nPromotion: {'ELIGIBLE' if promo.eligible else 'NOT READY'} - {promo.note}")
    for failure in promo.failures:
        print(f"  - {failure}")
    return 0


def _cmd_status(_: argparse.Namespace) -> int:
    ks = KillSwitch()
    print("KILL SWITCH: " + ("ENGAGED (nothing can trade)" if ks.engaged() else "off"))
    try:
        print(f"Stage: {load_config().stage.name}")
    except ConfigError as exc:
        print(f"Config refused: {exc}")
    return 0


def _cmd_kill(_: argparse.Namespace) -> int:
    KillSwitch().engage("manual via cli")
    print("Kill switch ENGAGED. Run `agent-trader unkill` to release it.")
    return 0


def _cmd_unkill(_: argparse.Namespace) -> int:
    KillSwitch().release()
    print("Kill switch released.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="agent-trader")
    sub = parser.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("replay", help="run the agent over historical or synthetic data")
    r.add_argument("--agent", choices=["baseline", "claude", "gemini"], default="baseline")
    r.add_argument("--symbols", default="ETH,WBTC")
    r.add_argument("--csv", help="comma separated CSV paths, one per symbol")
    r.add_argument("--steps", type=int, default=720, help="synthetic candles per symbol")
    r.add_argument("--seed", type=int, default=1)
    r.add_argument("--decide-every", type=int, default=6, help="claude/gemini: call the model every N bars")
    r.add_argument("--journal", default="runs/journal.jsonl")
    r.set_defaults(fn=_cmd_replay)

    for name, fn, help_ in [("status", _cmd_status, "show kill switch and stage"),
                            ("kill", _cmd_kill, "engage the kill switch"),
                            ("unkill", _cmd_unkill, "release the kill switch")]:
        sub.add_parser(name, help=help_).set_defaults(fn=fn)

    args = parser.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())

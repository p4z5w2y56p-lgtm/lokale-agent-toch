"""Command line: replay, paper, fetch, budget, wallet, status, kill, unkill."""
from __future__ import annotations

import argparse
import os
import sys

from .agent import LLMAgent, MomentumBaseline
from .config import ConfigError, load_config
from .data import align, fetch_binance, load_csv, save_csv, split, synthetic
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
        return align({s: load_csv(p, s) for s, p in zip(symbols, paths)})
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
    config = _with_costs(config, args)
    candles = _build_candles(args)
    if args.agent == "claude":
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print("ANTHROPIC_API_KEY is not set. Add it to the environment secrets first.", file=sys.stderr)
            return 2
        import anthropic

        from .budget import BudgetMeter

        llm = AnthropicLLM(anthropic.Anthropic(), os.environ.get("AGENT_MODEL", "claude-haiku-4-5-20251001"),
                           budget=BudgetMeter())
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
    if args.holdout:
        train, test = split(candles, args.holdout)
        print("=== TRAIN (tune the playbook on this) ===")
        _report(args, config, agent, run_replay(train, agent, config, killswitch, Journal(args.journal)))
        print("\n=== TEST (never tuned on: this is the number that counts) ===")
        agent = _fresh(agent)
        return _report(args, config, agent, run_replay(test, agent, config, killswitch, Journal(args.journal)))
    return _report(args, config, agent, run_replay(candles, agent, config, killswitch, Journal(args.journal)))


def _with_costs(config, args):
    from dataclasses import replace

    changes = {k: v for k, v in (("fee_bps", getattr(args, "fee_bps", None)),
                                 ("slippage_bps", getattr(args, "slippage_bps", None))) if v is not None}
    return replace(config, **changes) if changes else config


def _fresh(agent):
    """New agent with the same settings, so the test run carries no state from training."""
    if isinstance(agent, LLMAgent):
        return LLMAgent(llm=agent.llm, decide_every=agent.decide_every)
    return type(agent)()


def _report(args, config, agent, result) -> int:
    card = result.scorecard
    print(f"Agent: {args.agent}   Stage: {config.stage.name}   Steps: {card.steps}")
    print(f"Return:        {card.total_return:+.2%}")
    print(f"Buy-and-hold:  {card.buy_hold_return:+.2%}")
    print(f"Excess:        {card.excess_return:+.2%}")
    print(f"Max drawdown:  {card.max_drawdown:.2%}")
    print(f"Trades:        {card.trades}   Win rate: "
          f"{'n/a' if card.win_rate is None else f'{card.win_rate:.0%}'}")
    print(f"Rejected:      {card.proposals_rejected}/{card.proposals_total} proposals")
    print(f"Fees paid:     {sum(f.fee for f in result.fills):.2f}   "
          f"(fee {config.fee_bps:g} bps, slippage {config.slippage_bps:g} bps)")
    if getattr(agent, "last_error", None):
        print(f"Agent error:   {agent.last_error}")
    promo = check_promotion(config.stage, card)
    print(f"\nPromotion: {'ELIGIBLE' if promo.eligible else 'NOT READY'} - {promo.note}")
    for failure in promo.failures:
        print(f"  - {failure}")
    return 0


def _cmd_fetch(args: argparse.Namespace) -> int:
    from datetime import datetime, timezone

    end = datetime.now(timezone.utc)
    start_ms = int(end.timestamp() * 1000) - args.days * 86_400_000
    for sym in [s.strip().upper() for s in args.symbols.split(",") if s.strip()]:
        try:
            candles = fetch_binance(sym, start_ms, int(end.timestamp() * 1000), args.interval)
        except OSError as exc:
            print(f"{sym}: download failed ({exc})", file=sys.stderr)
            return 1
        path = f"{args.out}/{sym.lower()}_{args.interval}.csv"
        save_csv(candles, path)
        print(f"{sym}: {len(candles)} candles -> {path}")
    return 0


def _cmd_paper(args: argparse.Namespace) -> int:
    from .paper_live import PaperLive, PriceSourceError, binance_source

    try:
        config = load_config()
    except ConfigError as exc:
        print(f"Config refused: {exc}", file=sys.stderr)
        return 2
    config = _with_costs(config, args)
    if args.agent != "baseline":
        print("paper currently runs the baseline agent only (no paid LLM calls).", file=sys.stderr)
        return 2
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    live = PaperLive(MomentumBaseline(), config, symbols, binance_source, journal=Journal(args.journal))

    def show(s: dict) -> None:
        px = "  ".join(f"{k} {v:,.2f}" for k, v in s["prices"].items())
        print(f"bar {s['ts']}  {px}  equity {s['equity']:.2f}  fills {s['fills']}  "
              f"rejected {s['rejected']}{'' if s['new_bar'] else '  (no new bar)'}")

    try:
        live.run(1 if args.once else args.max_ticks, interval_s=args.interval, on_tick=show)
    except PriceSourceError as exc:
        print(f"Live prices unavailable: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Stopped.")
    print(f"Fees paid: {live.fees_paid:.2f}   Journal: {args.journal}")
    return 0


def _cmd_budget(_: argparse.Namespace) -> int:
    from .budget import BudgetMeter

    m = BudgetMeter()
    pct = m.spent_eur / m.cap_eur if m.cap_eur > 0 else 0.0
    print(f"LLM budget: EUR {m.spent_eur:.4f} of {m.cap_eur:.2f} ({pct:.0%})   "
          f"calls {m.state['calls']}   file {m.path}")
    return 0


def _cmd_wallet(args: argparse.Namespace) -> int:
    from .signer import SignerError, TestnetWallet

    if KillSwitch().engaged() and args.action != "status":
        print("Kill switch is ENGAGED. Refusing.", file=sys.stderr)
        return 2
    try:
        config = load_config()
        wallet = TestnetWallet.connect(os.environ, config.stage)
        if args.action == "status":
            print(f"Address: {wallet.address}   Chain: {wallet.settings.chain_id}")
            for sym, bal in wallet.balances().items():
                print(f"{sym}: {bal:.6f}")
            return 0
        tx = wallet.build_wrap(args.amount) if args.action == "wrap" else wallet.build_unwrap(args.amount)
        print(f"{args.action} {args.amount} on chain {tx['chainId']} from {wallet.address}")
        if not args.send:
            print("Dry run. Add --send to sign and broadcast.")
            return 0
        tx_hash = wallet.send(tx)
        print(f"Sent: https://sepolia.basescan.org/tx/0x{tx_hash.removeprefix('0x')}")
        return 0
    except (ConfigError, SignerError) as exc:
        print(f"Wallet refused: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"Could not reach the RPC node: {exc}", file=sys.stderr)
        return 1


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


def load_dotenv(path: str = ".env") -> None:
    """Read KEY=value lines from a local .env into the environment. Real env vars win."""
    try:
        lines = open(path).read().splitlines()
    except OSError:
        return
    for line in lines:
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            if value.strip():
                os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
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
    r.add_argument("--holdout", type=float, default=0.0,
                   help="fraction of the newest data kept for an out-of-sample test, e.g. 0.3")
    r.add_argument("--fee-bps", type=float, help="override the fee per trade, in basis points")
    r.add_argument("--slippage-bps", type=float, help="override simulated slippage, in basis points")
    r.set_defaults(fn=_cmd_replay)

    pp = sub.add_parser("paper", help="paper trade on live public hourly prices (fake money)")
    pp.add_argument("--agent", choices=["baseline"], default="baseline")
    pp.add_argument("--symbols", default="ETH,WBTC")
    pp.add_argument("--once", action="store_true", help="one tick, then exit")
    pp.add_argument("--max-ticks", type=int, default=24)
    pp.add_argument("--interval", type=float, default=300.0, help="seconds between polls")
    pp.add_argument("--journal", default="runs/paper_journal.jsonl")
    pp.add_argument("--fee-bps", type=float)
    pp.add_argument("--slippage-bps", type=float)
    pp.set_defaults(fn=_cmd_paper)

    sub.add_parser("budget", help="show LLM spend vs cap").set_defaults(fn=_cmd_budget)

    f = sub.add_parser("fetch", help="download real hourly candles (public Binance data, no key)")
    f.add_argument("--symbols", default="ETH,WBTC")
    f.add_argument("--days", type=int, default=365)
    f.add_argument("--interval", default="1h")
    f.add_argument("--out", default="data")
    f.set_defaults(fn=_cmd_fetch)

    w = sub.add_parser("wallet", help="testnet wallet: status, wrap/unwrap ETH (AGENT_STAGE=2 only)")
    w.add_argument("action", choices=["status", "wrap", "unwrap"])
    w.add_argument("--amount", type=float, default=0.001)
    w.add_argument("--send", action="store_true", help="actually sign and broadcast (default: dry run)")
    w.set_defaults(fn=_cmd_wallet)

    for name, fn, help_ in [("status", _cmd_status, "show kill switch and stage"),
                            ("kill", _cmd_kill, "engage the kill switch"),
                            ("unkill", _cmd_unkill, "release the kill switch")]:
        sub.add_parser(name, help=help_).set_defaults(fn=fn)

    args = parser.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())

"""Paper trading on live prices: poll public hourly closes, run the agent through the
policy engine and the paper broker. Fake money only: no keys, nothing on-chain."""
from __future__ import annotations

import time
import urllib.error
from dataclasses import replace
from typing import Callable

from .agent import Agent, Observation
from .broker import PaperBroker
from .config import Config
from .data import fetch_binance
from .journal import Journal
from .killswitch import KillSwitch
from .models import Candle
from .policy import PolicyEngine

# (symbol, hours of history) -> closed candles, oldest first
PriceSource = Callable[[str, int], list[Candle]]


class PriceSourceError(Exception):
    pass


def binance_source(symbol: str, hours: int) -> list[Candle]:
    """Latest closed 1h candles from data-api.binance.vision (public, no key)."""
    now_ms = int(time.time() * 1000)
    start = now_ms - (hours + 1) * 3_600_000
    try:
        candles = fetch_binance(symbol, start, now_ms, "1h")
    except urllib.error.HTTPError as exc:
        raise PriceSourceError(
            f"Binance answered HTTP {exc.code} for {symbol}. If this is 403, the network "
            "(e.g. a cloud proxy) blocks data-api.binance.vision; run `paper` from a machine "
            "with open internet access."
        ) from exc
    except OSError as exc:
        raise PriceSourceError(f"Could not reach data-api.binance.vision for {symbol}: {exc}") from exc
    # The newest kline is still forming; only act on closed bars.
    cutoff = now_ms // 1000 - 3600
    return [c for c in candles if c.ts <= cutoff]


class PaperLive:
    def __init__(self, agent: Agent, config: Config, symbols: list[str], source: PriceSource,
                 killswitch: KillSwitch | None = None, journal: Journal | None = None,
                 history_hours: int = 72) -> None:
        self.agent, self.config, self.symbols, self.source = agent, config, symbols, source
        self.killswitch = killswitch or KillSwitch()
        self.journal = journal or Journal()
        self.history_hours = history_hours
        self.broker = PaperBroker(config)
        self.policy = PolicyEngine(config, self.killswitch)
        self.last_ts: int | None = None
        self.day: int | None = None
        self.fees_paid = 0.0

    def tick(self) -> dict:
        """One poll. Acts only when a new closed bar exists. Returns a small summary."""
        hist = {s: self.source(s, self.history_hours) for s in self.symbols}
        if any(not v for v in hist.values()):
            raise PriceSourceError("price source returned no candles")
        common = set.intersection(*({c.ts for c in v} for v in hist.values()))
        if not common:
            raise PriceSourceError("symbols share no common timestamp")
        ts = max(common)
        hist = {s: [c for c in v if c.ts <= ts and c.ts in common] for s, v in hist.items()}
        prices = {s: v[-1].close for s, v in hist.items()}
        summary = {"ts": ts, "prices": prices, "new_bar": False, "fills": 0, "rejected": 0}
        if self.last_ts is not None and ts <= self.last_ts:
            summary["equity"] = self.broker.equity(prices)
            return summary
        self.last_ts = ts
        summary["new_bar"] = True
        if self.day != ts // 86_400:
            self.day = ts // 86_400
            self.broker.new_day(prices)

        for proposal in self.agent.decide(Observation(ts, hist, prices, self.broker.view(prices))):
            if proposal.symbol not in prices:
                summary["rejected"] += 1
                self.journal.write("rejected", ts=ts, proposal=proposal, reasons=["no market data for symbol"])
                continue
            proposal = replace(proposal, price=prices[proposal.symbol],
                               quoted_slippage_bps=self.config.slippage_bps)
            decision = self.policy.evaluate(proposal, self.broker.view(prices))
            if not decision.approved:
                summary["rejected"] += 1
                self.journal.write("rejected", ts=ts, proposal=proposal, reasons=list(decision.reasons))
                continue
            fill = self.broker.execute(proposal, ts)
            if fill is None:
                summary["rejected"] += 1
                self.journal.write("rejected", ts=ts, proposal=proposal, reasons=["broker could not fill"])
                continue
            self.fees_paid += fill.fee
            summary["fills"] += 1
            self.journal.write("fill", ts=ts, fill=fill, reason=proposal.reason, mode="paper_live")
        summary["equity"] = self.broker.equity(prices)
        self.journal.write("paper_tick", ts=ts, prices=prices, equity=summary["equity"],
                           cash=self.broker.cash, holdings=self.broker.holdings(), fees_paid=self.fees_paid)
        return summary

    def run(self, max_ticks: int, interval_s: float = 300.0, sleep=time.sleep, on_tick=None) -> int:
        for i in range(max_ticks):
            s = self.tick()
            if on_tick:
                on_tick(s)
            if i < max_ticks - 1:
                sleep(interval_s)
        return max_ticks

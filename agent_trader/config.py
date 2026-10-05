"""Stages, limits and the mainnet lock."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Mapping


class ConfigError(Exception):
    pass


class Stage(IntEnum):
    REPLAY = 0
    LIVE_PAPER = 1
    TESTNET = 2
    MAINNET_CAPPED = 3
    MAINNET_AUTONOMOUS = 4


# Base Sepolia, Arbitrum Sepolia, Ethereum Sepolia
TESTNET_CHAIN_IDS = frozenset({84532, 421614, 11155111})
# Ethereum, Optimism, Base, Arbitrum One
MAINNET_CHAIN_IDS = frozenset({1, 10, 8453, 42161})

UNLOCK_PHRASE = "I_UNDERSTAND_THIS_USES_REAL_MONEY"


@dataclass(frozen=True)
class Limits:
    max_trade_pct: float = 0.10        # of equity, per trade
    max_position_pct: float = 0.30     # of equity, per token
    max_trades_per_day: int = 10
    max_daily_loss_pct: float = 0.05   # blocks new buys for the rest of the day
    max_drawdown_pct: float = 0.15     # from peak equity: trips the kill switch for the run
    max_slippage_bps: float = 50.0
    allowed_symbols: frozenset[str] = field(
        default_factory=lambda: frozenset({"ETH", "WBTC"})
    )


@dataclass(frozen=True)
class Config:
    stage: Stage = Stage.REPLAY
    limits: Limits = field(default_factory=Limits)
    starting_cash: float = 1000.0
    fee_bps: float = 10.0              # ~0.10%: CEX taker fee, or a low-fee DEX pool (Uniswap v3 0.05-0.30%)
    slippage_bps: float = 5.0          # simulated slippage on liquid pairs (ETH, BTC) at small size
    mainnet_unlocked: bool = False


def load_config(env: Mapping[str, str] | None = None) -> Config:
    """Build a Config from environment variables. Refuses locked stages."""
    env = os.environ if env is None else env
    try:
        stage = Stage(int(env.get("AGENT_STAGE", "0")))
    except ValueError as exc:
        raise ConfigError(f"Invalid AGENT_STAGE: {env.get('AGENT_STAGE')!r}") from exc

    unlocked = env.get("ALLOW_MAINNET", "") == UNLOCK_PHRASE
    if stage >= Stage.MAINNET_CAPPED and not unlocked:
        raise ConfigError(
            "Stages 3-4 use real money and are locked. Set ALLOW_MAINNET to the "
            "unlock phrase in the environment to enable them (a human decision)."
        )
    return Config(stage=stage, mainnet_unlocked=unlocked)

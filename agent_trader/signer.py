"""Testnet-only wallet guard.

This build does NOT sign or send anything. It defines the one rule that matters before
any signer exists: the chain must be an allowlisted TESTNET, and the stage must allow it.
The private key is never read here; when a real signer is added (milestone 2) it will be
the only module allowed to read SIGNER_PRIVATE_KEY.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .config import MAINNET_CHAIN_IDS, TESTNET_CHAIN_IDS, Stage


class SignerError(Exception):
    pass


def require_testnet(chain_id: int, stage: Stage) -> None:
    if chain_id in MAINNET_CHAIN_IDS:
        raise SignerError(f"Chain {chain_id} is a MAINNET. Refusing: this build is testnet-only.")
    if chain_id not in TESTNET_CHAIN_IDS:
        raise SignerError(f"Chain {chain_id} is not an allowlisted testnet.")
    if stage != Stage.TESTNET:
        raise SignerError(f"Stage is {stage.name}; wallet access needs stage TESTNET.")


@dataclass(frozen=True)
class TestnetSettings:
    rpc_url: str
    chain_id: int

    __test__ = False  # not a pytest class

    @classmethod
    def from_env(cls, env: Mapping[str, str], stage: Stage) -> "TestnetSettings":
        try:
            chain_id = int(env.get("CHAIN_ID", ""))
        except ValueError as exc:
            raise SignerError("CHAIN_ID must be a number") from exc
        require_testnet(chain_id, stage)
        rpc_url = env.get("RPC_URL", "")
        if not rpc_url.startswith("https://"):
            raise SignerError("RPC_URL must be set to an https:// endpoint")
        return cls(rpc_url=rpc_url, chain_id=chain_id)

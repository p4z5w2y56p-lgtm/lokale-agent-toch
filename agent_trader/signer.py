"""Testnet-only wallet.

The one rule: the chain must be an allowlisted TESTNET, and the stage must allow it. That is
checked twice: against CHAIN_ID from the environment, and against what the RPC node itself
reports, so a mainnet RPC URL can never slip through with a testnet CHAIN_ID.

This is the only module that reads SIGNER_PRIVATE_KEY. The key is never printed or logged.
For now the wallet can only wrap/unwrap ETH <-> WETH: a real signed transaction with no
liquidity or price risk, to prove the pipeline before any swap code exists.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

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


# WETH predeploy on OP-stack chains (same address on Base Sepolia).
WETH_ADDRESSES = {84532: "0x4200000000000000000000000000000000000006"}
WETH_ABI = [
    {"name": "deposit", "type": "function", "stateMutability": "payable", "inputs": [], "outputs": []},
    {"name": "withdraw", "type": "function", "stateMutability": "nonpayable",
     "inputs": [{"name": "wad", "type": "uint256"}], "outputs": []},
    {"name": "balanceOf", "type": "function", "stateMutability": "view",
     "inputs": [{"name": "owner", "type": "address"}], "outputs": [{"name": "", "type": "uint256"}]},
]
MAX_TX_ETH = 0.01  # hard cap per transaction, even on testnet


class TestnetWallet:
    """Signs and sends testnet transactions. Build with `connect`, or pass a web3-like
    object and account for tests."""

    __test__ = False

    def __init__(self, w3: Any, account: Any, settings: TestnetSettings):
        actual = int(w3.eth.chain_id)
        if actual != settings.chain_id:
            raise SignerError(f"RPC reports chain {actual}, but CHAIN_ID is {settings.chain_id}. Refusing.")
        require_testnet(actual, Stage.TESTNET)
        if actual not in WETH_ADDRESSES:
            raise SignerError(f"No WETH address known for chain {actual}")
        self.w3, self.account, self.settings = w3, account, settings
        self.weth = w3.eth.contract(address=WETH_ADDRESSES[actual], abi=WETH_ABI)

    @classmethod
    def connect(cls, env: Mapping[str, str], stage: Stage) -> "TestnetWallet":
        settings = TestnetSettings.from_env(env, stage)
        key = env.get("SIGNER_PRIVATE_KEY", "").strip()
        if not key:
            raise SignerError("SIGNER_PRIVATE_KEY is not set (use a throwaway testnet wallet)")
        try:
            from eth_account import Account
            from web3 import Web3
        except ImportError as exc:
            raise SignerError('web3 is not installed: pip install -e ".[testnet]"') from exc
        try:
            account = Account.from_key(key)
        except Exception:
            raise SignerError("SIGNER_PRIVATE_KEY is not a valid private key") from None
        return cls(Web3(Web3.HTTPProvider(settings.rpc_url)), account, settings)

    @property
    def address(self) -> str:
        return self.account.address

    def balances(self) -> dict[str, float]:
        eth = self.w3.eth.get_balance(self.address) / 1e18
        weth = self.weth.functions.balanceOf(self.address).call() / 1e18
        return {"ETH": eth, "WETH": weth}

    def build_wrap(self, amount_eth: float) -> dict:
        wei = _to_wei(amount_eth)
        return self._finish(self.weth.functions.deposit().build_transaction(self._base(value=wei)))

    def build_unwrap(self, amount_eth: float) -> dict:
        wei = _to_wei(amount_eth)
        return self._finish(self.weth.functions.withdraw(wei).build_transaction(self._base()))

    def send(self, tx: dict) -> str:
        if int(tx.get("chainId", -1)) != self.settings.chain_id:
            raise SignerError("Transaction chainId does not match the testnet. Refusing to sign.")
        if int(tx.get("value", 0)) > _to_wei(MAX_TX_ETH):
            raise SignerError(f"Transaction value is above the {MAX_TX_ETH} ETH per-transaction cap. Refusing to sign.")
        if str(tx.get("to", "")).lower() != self.weth.address.lower():
            raise SignerError("Transaction is not to the WETH contract. Refusing to sign.")
        signed = self.account.sign_transaction(tx)
        tx_hash = self.w3.eth.send_raw_transaction(signed.raw_transaction)
        return tx_hash.hex() if hasattr(tx_hash, "hex") else str(tx_hash)

    def _base(self, value: int = 0) -> dict:
        return {
            "from": self.address,
            "value": value,
            "chainId": self.settings.chain_id,
            "nonce": self.w3.eth.get_transaction_count(self.address),
        }

    def _finish(self, tx: dict) -> dict:
        tx.setdefault("gas", 100_000)
        return tx


def _to_wei(amount_eth: float) -> int:
    if not amount_eth > 0:
        raise SignerError("Amount must be positive")
    if amount_eth > MAX_TX_ETH:
        raise SignerError(f"Amount {amount_eth} ETH is above the {MAX_TX_ETH} ETH per-transaction cap")
    return int(round(amount_eth * 1e18))

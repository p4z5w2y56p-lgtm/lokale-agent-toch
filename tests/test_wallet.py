from types import SimpleNamespace

import pytest

from agent_trader.config import Stage
from agent_trader.signer import MAX_TX_ETH, SignerError, TestnetSettings, TestnetWallet


class FakeFn:
    def __init__(self, name, to=None):
        self.name, self.to = name, to

    def build_transaction(self, base):
        return {**base, "to": self.to, "data": self.name}

    def call(self):
        return 5 * 10**17


class FakeEth:
    def __init__(self, chain_id):
        self.chain_id = chain_id
        self.sent = []

    def contract(self, address, abi):
        fns = SimpleNamespace(deposit=lambda: FakeFn("deposit", address),
                              withdraw=lambda wad: FakeFn(f"withdraw:{wad}", address),
                              balanceOf=lambda a: FakeFn("bal"))
        return SimpleNamespace(address=address, functions=fns)

    def get_balance(self, a):
        return 10**18

    def get_transaction_count(self, a):
        return 7

    def send_raw_transaction(self, raw):
        self.sent.append(raw)
        return b"\xab"


class FakeAccount:
    address = "0xabc"

    def sign_transaction(self, tx):
        return SimpleNamespace(raw_transaction=b"signed")


SETTINGS = TestnetSettings(rpc_url="https://x", chain_id=84532)


def wallet(rpc_chain=84532):
    return TestnetWallet(SimpleNamespace(eth=FakeEth(rpc_chain)), FakeAccount(), SETTINGS)


def test_refuses_when_rpc_is_actually_mainnet():
    with pytest.raises(SignerError, match="RPC reports chain 8453"):
        wallet(rpc_chain=8453)


def test_wrap_builds_testnet_tx_with_value():
    tx = wallet().build_wrap(0.001)
    assert tx["chainId"] == 84532 and tx["value"] == 10**15 and tx["data"] == "deposit"
    assert tx["nonce"] == 7


def test_unwrap_has_no_value():
    tx = wallet().build_unwrap(0.002)
    assert tx["value"] == 0 and tx["data"] == f"withdraw:{2 * 10**15}"


@pytest.mark.parametrize("amount", [0, -1, MAX_TX_ETH * 2])
def test_amount_cap_and_sign(amount):
    with pytest.raises(SignerError):
        wallet().build_wrap(amount)


def test_send_refuses_foreign_chain_id():
    w = wallet()
    tx = w.build_wrap(0.001) | {"chainId": 8453}
    with pytest.raises(SignerError):
        w.send(tx)
    assert w.w3.eth.sent == []


def test_send_signs_and_broadcasts():
    w = wallet()
    assert w.send(w.build_wrap(0.001)) == "ab"
    assert w.w3.eth.sent == [b"signed"]


def test_balances():
    assert wallet().balances() == {"ETH": 1.0, "WETH": 0.5}


def test_connect_needs_key_and_testnet_stage():
    env = {"CHAIN_ID": "84532", "RPC_URL": "https://x"}
    with pytest.raises(SignerError, match="SIGNER_PRIVATE_KEY"):
        TestnetWallet.connect(env, Stage.TESTNET)
    with pytest.raises(SignerError, match="TESTNET"):
        TestnetWallet.connect(env | {"SIGNER_PRIVATE_KEY": "0x" + "1" * 64}, Stage.REPLAY)


def test_connect_rejects_bad_key_without_echoing_it():
    env = {"CHAIN_ID": "84532", "RPC_URL": "https://x", "SIGNER_PRIVATE_KEY": "secretgarbage"}
    with pytest.raises(SignerError) as exc:
        TestnetWallet.connect(env, Stage.TESTNET)
    assert "secretgarbage" not in str(exc.value)


def test_load_dotenv(tmp_path, monkeypatch):
    from agent_trader.cli import load_dotenv

    (tmp_path / ".env").write_text('# c\nFOO_T=1\nBAR_T="two"\nEMPTY_T=\nKEEP_T=file\n')
    monkeypatch.setenv("KEEP_T", "env")
    monkeypatch.delenv("FOO_T", raising=False)
    load_dotenv(str(tmp_path / ".env"))
    import os
    assert os.environ["FOO_T"] == "1" and os.environ["BAR_T"] == "two"
    assert "EMPTY_T" not in os.environ and os.environ["KEEP_T"] == "env"
    for k in ("FOO_T", "BAR_T"):
        monkeypatch.delenv(k)



def test_send_refuses_value_over_cap_or_foreign_destination():
    w = wallet()
    with pytest.raises(SignerError):
        w.send(w.build_wrap(0.001) | {"value": 10**18})
    with pytest.raises(SignerError):
        w.send(w.build_wrap(0.001) | {"to": "0x000000000000000000000000000000000000dEaD"})
    assert w.w3.eth.sent == []

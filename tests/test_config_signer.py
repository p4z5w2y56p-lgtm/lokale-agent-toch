import pytest

from agent_trader.config import UNLOCK_PHRASE, ConfigError, Stage, load_config
from agent_trader.signer import SignerError, TestnetSettings, require_testnet


def test_default_is_replay_and_locked_to_paper():
    assert load_config({}).stage is Stage.REPLAY


@pytest.mark.parametrize("stage", ["3", "4"])
def test_real_money_stages_refused_by_default(stage):
    with pytest.raises(ConfigError):
        load_config({"AGENT_STAGE": stage})


def test_real_money_stage_needs_exact_unlock_phrase():
    with pytest.raises(ConfigError):
        load_config({"AGENT_STAGE": "3", "ALLOW_MAINNET": "yes"})
    assert load_config({"AGENT_STAGE": "3", "ALLOW_MAINNET": UNLOCK_PHRASE}).mainnet_unlocked


def test_garbage_stage_is_refused():
    with pytest.raises(ConfigError):
        load_config({"AGENT_STAGE": "banana"})
    with pytest.raises(ConfigError):
        load_config({"AGENT_STAGE": "9"})


@pytest.mark.parametrize("chain_id", [1, 10, 8453, 42161])
def test_mainnet_chains_always_refused(chain_id):
    with pytest.raises(SignerError, match="MAINNET"):
        require_testnet(chain_id, Stage.TESTNET)


def test_unknown_chain_refused():
    with pytest.raises(SignerError):
        require_testnet(999999, Stage.TESTNET)


def test_testnet_needs_testnet_stage():
    with pytest.raises(SignerError):
        require_testnet(84532, Stage.REPLAY)
    require_testnet(84532, Stage.TESTNET)  # does not raise


def test_settings_from_env():
    env = {"CHAIN_ID": "84532", "RPC_URL": "https://example.invalid/rpc"}
    s = TestnetSettings.from_env(env, Stage.TESTNET)
    assert s.chain_id == 84532
    with pytest.raises(SignerError):
        TestnetSettings.from_env({"CHAIN_ID": "84532", "RPC_URL": "http://insecure"}, Stage.TESTNET)
    with pytest.raises(SignerError):
        TestnetSettings.from_env({"CHAIN_ID": "8453", "RPC_URL": "https://x"}, Stage.TESTNET)

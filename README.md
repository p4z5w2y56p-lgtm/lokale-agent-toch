# agent-trader

A crypto trading agent that has to **earn** its way to a wallet. It starts with fake money
on old data, graduates through stricter stages, and can never override its own safety rules.

> **Status:** stages 0-1 (replay and paper trading engine) work. Stage 2 (testnet) has only
> its safety guard. Stages 3-4 (real money) are locked. Nothing here can spend real money.
> Nothing here is a claim that the agent makes profit.

## How it works (in one picture)

```
market data -> agent (proposes) -> policy engine (code, can say NO) -> broker (fake money)
                                                                          |
                                         journal + scorecard + promotion gates <-+
```

The agent only *proposes* trades. A separate piece of plain code decides if they're allowed
(max size, allowed tokens, daily loss limit, kill switch...). Even a tricked or confused agent
can't get past it. Details: `docs/superpowers/specs/2026-10-05-trading-agent-design.md`.

## Stages

| Stage | What | Real money |
|---|---|---|
| 0 Replay | old/synthetic data | no |
| 1 Live paper | real prices, fake money | no |
| 2 Testnet | real transactions, worthless tokens | no |
| 3 Mainnet, tiny cap, you approve each trade | locked | yes |
| 4 Autonomous within hard limits | locked | yes |

Promotion is **advice only**: `promotion.py` tells you whether the scorecard earned it. You decide.

## Setup

```bash
pip install -e ".[dev]"
python -m pytest            # should be all green
```

Keys go in the environment's **secrets settings** (or a local `.env`, which is git-ignored).
See `.env.example` for the names. **Never** put a seed phrase or a mainnet private key anywhere.

## Run it

```bash
python -m agent_trader replay                      # rule-based baseline on synthetic data
python -m agent_trader replay --agent claude       # needs ANTHROPIC_API_KEY
python -m agent_trader replay --agent gemini       # needs GEMINI_API_KEY
python -m agent_trader replay --csv eth.csv,btc.csv --symbols ETH,WBTC
python -m agent_trader kill      # emergency stop: nothing can trade
python -m agent_trader unkill
python -m agent_trader status
```

CSV format: `ts,open,high,low,close,volume`.

Every run writes `runs/journal.jsonl`: each trade, each refusal and why, and the scorecard.

## How the agent gets "trained"

No model weights change. Training = run -> read the scorecard and journal -> edit
`agent_trader/playbook.md` (its rules and lessons) -> run again -> compare. The rule-based
baseline is the control: the LLM has to beat it, and beat plain buy-and-hold, to pass a gate.

## What's next

1. Real historical data (hourly candles for ETH/WBTC) to replace synthetic data.
2. Live paper trading feed (stage 1).
3. Testnet swaps on Base Sepolia (stage 2): needs an RPC URL and a throwaway wallet with faucet funds.
4. Only after weeks of clean results: a human decision about stage 3.

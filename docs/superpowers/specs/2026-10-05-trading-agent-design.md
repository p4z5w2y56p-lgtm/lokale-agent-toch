# Trading Agent: Design

Date: 2026-10-05
Status: Draft for owner review (owner delegated the build; stages 3-4 stay locked)

## Goal

An LLM-powered agent that is *trained* (evaluated, tuned, and gated) well enough
that it can eventually hold a wallet and trade crypto tokens on an EVM L2
(Base/Arbitrum), **safely**. "Safely" is the primary requirement; profit is secondary.

## Decisions made with the owner

| Topic | Decision |
|---|---|
| Market | Crypto tokens |
| Venue | EVM L2 (Base / Arbitrum), self-custody wallet, DEX swaps |
| "Training" | LLM agent + playbook + replay/paper trading. Prompts, rules and memory are tuned from scorecards. No model weights are trained. |
| Language | Python 3.11 |
| Money in this build | **None.** Only stages 0-2 are implemented. Stages 3-4 are designed and hard-locked. |

## Core principle

**The agent proposes. Code disposes.** The LLM is never trusted to enforce limits.
Limits live in a policy engine the agent cannot modify, and keys live in a signer the
agent cannot read. Anything the agent reads from the outside world (token names, news,
websites) is untrusted data and may contain prompt injection; the policy engine is the
backstop that makes injection survivable.

## Architecture

```
market data --> Observation --> Agent --(TradeProposal)--> PolicyEngine --(approved)--> Broker
                                   |                           |                        |
                                   +------- Journal <----------+---- Scorecard <--------+
```

| Unit | Responsibility | Depends on |
|---|---|---|
| `models` | Plain dataclasses: Candle, TradeProposal, Fill, Decision | nothing |
| `config` | Stage, limits, chain allowlist. Mainnet is locked unless an explicit unlock is set | models |
| `killswitch` | Global stop: a `KILL` file or in-memory flag. Checked before every trade | nothing |
| `policy` | Pure function-like engine: proposal + portfolio -> Decision with reasons | models, config, killswitch |
| `data` | Candle sources: CSV loader, seeded synthetic generator | models |
| `broker` | PaperBroker: fees, slippage, portfolio, equity | models |
| `agent` | `Agent` protocol; `MomentumBaseline` (rule-based control) and `LLMAgent` (provider-agnostic LLM agent) | models |
| `llm` | Provider adapters `(system, user) -> text`: `AnthropicLLM`, `GeminiLLM` (plain HTTPS, key in header) | nothing |
| `journal` | Append-only JSONL of every observation summary, proposal, decision, fill | models |
| `scorecard` | Return, max drawdown, win rate, rule-violation count, vs buy-and-hold | models |
| `promotion` | Stage gates: scorecard -> may promote or not (advisory, never auto) | scorecard, config |
| `replay` | Walks candles with no look-ahead, wires everything together | all above |
| `signer` | Signer interface + chain-ID guard. Testnet only. No key material in the repo | config |
| `cli` | `replay`, `status`, `kill`, `unkill` | replay, killswitch |

## Policy rules (v1)

A proposal is rejected (with reasons) if any of these fail:

1. Kill switch is engaged.
2. Stage does not allow trading (e.g. a locked stage).
3. Token is not on the allowlist.
4. Trade notional exceeds `max_trade_pct` of current equity.
5. Position in the token would exceed `max_position_pct` of equity.
6. Daily trade count exceeds `max_trades_per_day`.
7. Daily realized+unrealized loss exceeds `max_daily_loss_pct` (trading halts for the day).
8. Quoted slippage exceeds `max_slippage_bps`.
9. A buy would spend more cash than is available, or a sell more than is held (no leverage, no shorting).
10. Quantity or price is not a finite positive number.

Trusted fields (price, quoted slippage) are overwritten by the replay engine from market data;
the agent's own values for them are ignored. Cash checks budget for fee and slippage.

Policy checks are collected, not short-circuited, so the journal shows every reason.

## Stages

| Stage | Name | Money | Implemented here |
|---|---|---|---|
| 0 | Replay (historical) | none | yes |
| 1 | Live paper | none | engine yes, live feed later |
| 2 | Testnet | worthless tokens | signer guard + interface only |
| 3 | Mainnet, tiny cap, human approves each trade | real | locked |
| 4 | Mainnet autonomous within limits (+ on-chain limits via smart-contract wallet) | real | locked |

Promotion is **advisory**: `promotion` reports whether gates are met; a human moves the stage.
Moving into stage 3+ additionally requires `ALLOW_MAINNET=I_UNDERSTAND_THIS_USES_REAL_MONEY`
and a mainnet chain ID in the allowlist. Neither is set by default and neither can be set by the agent.

## Secrets

- API keys and RPC URLs come from environment variables only (`ANTHROPIC_API_KEY` or `GEMINI_API_KEY`, `RPC_URL`).
- Wallet private keys are never read by the agent process. The signer is the only component
  that may read `SIGNER_PRIVATE_KEY`, and only for testnet chain IDs.
- `.env` is git-ignored. `.env.example` documents names only.

## Testing

- Policy engine: one test per rule, plus a "hostile proposals" test file (NaN, negative, huge,
  unknown token, sell more than held).
- Replay: no look-ahead test, deterministic with a seed.
- Broker: fee/slippage math, cash and holdings never go negative.
- Config: mainnet is refused by default.
- LLMAgent: response parsing is tested with fake providers (Anthropic client and Gemini HTTP layer); malformed or injected output
  becomes "no trade", never an exception that skips the policy engine.

## Out of scope for this build

Live market feed, testnet swap execution, mainnet, smart-contract wallet limits, any
claim of profitability. These are the next milestones and need the owner's RPC key and
testnet faucet funds.

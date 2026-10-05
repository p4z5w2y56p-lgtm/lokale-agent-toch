"""Rule-based strategy research for OMIN: no LLM, no network, no PolicyEngine.

Harness rules (see backtest.py): a decision at bar t sees data up to close[t] only and fills at
open[t+1] with 5 bps slippage and a 10 bps fee; one continuous run per strategy from bar 0;
warm-up, dev and holdout periods are fixed bar ranges; tuning uses dev data only.

Run:  python -m agent_trader.research --data data/binance --out runs/research_results.md
"""

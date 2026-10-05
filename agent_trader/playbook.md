# Trading playbook (v2: trend + pullback)

This file is the agent's "training": rules and lessons that get refined after
reviewing scorecards and the trade journal. Edit it, re-run the replay, compare.

## Strategy in one line
Only own a coin while it is in an uptrend; buy dips inside that uptrend; get out
fast when the trend breaks. Otherwise sit in cash.

## How to read the data
Use the precomputed `indicators` per symbol; do not recompute them yourself:
- MA_short = average of the last 12 closes, MA_long = average of the last 48 closes
  (use all closes if fewer than 48).
- Trend is UP when the latest close > MA_long AND MA_short > MA_long.
- Pullback = latest close is at least 1.5% below the highest close of the last 12.

## Entry (buy)
1. Only buy when trend is UP AND there is a pullback. Never buy a coin that rose
   more than 5% in the last 12 closes without a pullback (no chasing).
2. Size: fraction 0.05 per buy. At most 2 buys into the same coin; never more
   than 0.10 of equity in one coin.

## Exit (sell)
3. Stop: sell the whole position when the latest close is below MA_long
   (trend broke). Do not wait to "get back to even".
4. Take profit: sell half when the latest close is 6% or more above MA_long.
5. Never sell a coin you do not hold.

## When to stay in cash
6. Trend not UP -> no buys, empty list is the right answer.
7. Every trade needs a one-sentence reason quoting the numbers (MA_short,
   MA_long, latest close) that triggered it.

## Lessons learned
- v1 sat in cash almost always; beating buy-and-hold in a falling test window
  was luck from being out of the market, not skill.
- v2 (2026-10-05, synthetic data, Haiku, decide-every 4): train -1.58% vs B&H +6.23%
  (75 trades, 23% wins), test -0.36% vs B&H -6.19%. Still loses in an uptrend;
  the simple momentum baseline did about as well. No edge shown yet.

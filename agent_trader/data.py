"""Candle sources: a CSV loader and a seeded synthetic generator."""
from __future__ import annotations

import csv
import math
import random
from pathlib import Path

from .models import Candle


def load_csv(path: str | Path, symbol: str) -> list[Candle]:
    """Load candles from a CSV with columns: ts,open,high,low,close,volume."""
    candles: list[Candle] = []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            candle = Candle(
                ts=int(row["ts"]),
                symbol=symbol,
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row.get("volume") or 0.0),
            )
            if not all(math.isfinite(x) and x > 0 for x in (candle.open, candle.high, candle.low, candle.close)):
                raise ValueError(f"Bad price in {path} at ts={candle.ts}")
            candles.append(candle)
    candles.sort(key=lambda c: c.ts)
    return candles


def synthetic(
    symbol: str,
    n: int,
    seed: int = 1,
    start_price: float = 2000.0,
    drift: float = 0.0002,
    vol: float = 0.01,
    step_seconds: int = 3600,
) -> list[Candle]:
    """Seeded geometric random walk. Deterministic: same seed, same candles."""
    rng = random.Random(seed)
    price = start_price
    out: list[Candle] = []
    for i in range(n):
        ret = rng.gauss(drift, vol)
        new = max(price * math.exp(ret), 0.01)
        hi = max(price, new) * (1 + abs(rng.gauss(0, vol / 4)))
        lo = min(price, new) * (1 - abs(rng.gauss(0, vol / 4)))
        out.append(Candle(i * step_seconds, symbol, price, hi, lo, new, rng.uniform(100, 1000)))
        price = new
    return out

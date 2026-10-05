"""Candle sources: a CSV loader and a seeded synthetic generator."""
from __future__ import annotations

import os
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


BINANCE_PAIRS = {"ETH": "ETHUSDT", "WBTC": "BTCUSDT", "BTC": "BTCUSDT"}
# data-api.binance.vision is Binance's public market-data mirror; api.binance.com answers 451 from US cloud IPs.
BINANCE_URL = os.environ.get("BINANCE_URL", "https://data-api.binance.vision/api/v3/klines")


def fetch_binance(
    symbol: str,
    start_ms: int,
    end_ms: int,
    interval: str = "1h",
    get_json=None,
) -> list[Candle]:
    """Download public candles from Binance (no API key, read-only). `get_json` is
    injectable for tests; it takes a URL and returns parsed JSON."""
    if get_json is None:
        get_json = _http_get_json
    pair = BINANCE_PAIRS.get(symbol.upper(), symbol.upper() + "USDT")
    out: list[Candle] = []
    cursor = start_ms
    while cursor < end_ms:
        url = f"{BINANCE_URL}?symbol={pair}&interval={interval}&startTime={cursor}&endTime={end_ms}&limit=1000"
        rows = get_json(url)
        if not rows:
            break
        for r in rows:
            out.append(Candle(int(r[0]) // 1000, symbol, float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5])))
        next_cursor = int(rows[-1][0]) + 1
        if next_cursor <= cursor:
            break
        cursor = next_cursor
    return out


def _http_get_json(url: str):
    import json
    import urllib.request

    with urllib.request.urlopen(url, timeout=30) as resp:
        return json.load(resp)


def save_csv(candles: list[Candle], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["ts", "open", "high", "low", "close", "volume"])
        for c in candles:
            w.writerow([c.ts, c.open, c.high, c.low, c.close, c.volume])


def align(candles: dict[str, list[Candle]]) -> dict[str, list[Candle]]:
    """Keep only timestamps every symbol has, so replay sees the same bars for all."""
    common = set.intersection(*({c.ts for c in v} for v in candles.values()))
    return {s: [c for c in v if c.ts in common] for s, v in candles.items()}


def split(candles: dict[str, list[Candle]], test_fraction: float) -> tuple[dict, dict]:
    """Chronological train/test split: tune on the past, judge on data the tuning never saw."""
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1")
    n = len(next(iter(candles.values())))
    cut = int(n * (1 - test_fraction))
    return {s: v[:cut] for s, v in candles.items()}, {s: v[cut:] for s, v in candles.items()}

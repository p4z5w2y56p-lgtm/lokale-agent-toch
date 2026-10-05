import pytest

from agent_trader.cli import main
from agent_trader.data import align, fetch_binance, load_csv, save_csv, split, synthetic


def _rows(start_ms, n):
    return [[start_ms + i * 3_600_000, "1", "2", "0.5", "1.5", "10"] for i in range(n)]


def test_fetch_binance_paginates_until_end():
    calls = []

    def fake(url):
        calls.append(url)
        start = int(url.split("startTime=")[1].split("&")[0])
        return {1: _rows(start, 1000), 2: _rows(start, 5)}.get(len(calls), [])

    out = fetch_binance("ETH", 0, 10**13, get_json=fake)
    assert len(out) == 1005
    assert "symbol=ETHUSDT" in calls[0]
    assert out[1].ts - out[0].ts == 3600
    assert [c.ts for c in out] == sorted(c.ts for c in out)


def test_fetch_maps_wbtc_to_btc_and_stops_on_empty():
    seen = []
    out = fetch_binance("WBTC", 0, 10, get_json=lambda u: seen.append(u) or [])
    assert out == [] and "BTCUSDT" in seen[0]


def test_csv_roundtrip(tmp_path):
    candles = synthetic("ETH", 20)
    save_csv(candles, tmp_path / "x" / "eth.csv")
    assert load_csv(tmp_path / "x" / "eth.csv", "ETH") == candles


def test_align_drops_missing_bars():
    a, b = synthetic("ETH", 10), synthetic("WBTC", 10, seed=2)
    out = align({"ETH": a, "WBTC": b[:3] + b[4:]})
    assert len(out["ETH"]) == len(out["WBTC"]) == 9


def test_split_is_chronological_and_disjoint():
    c = {"ETH": synthetic("ETH", 100)}
    train, test = split(c, 0.3)
    assert len(train["ETH"]) == 70 and len(test["ETH"]) == 30
    assert train["ETH"][-1].ts < test["ETH"][0].ts
    with pytest.raises(ValueError):
        split(c, 1.0)


def test_replay_holdout_runs(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert main(["replay", "--holdout", "0.3", "--journal", str(tmp_path / "j.jsonl")]) == 0
    out = capsys.readouterr().out
    assert "TRAIN" in out and "TEST" in out

from datetime import datetime, timezone
from pathlib import Path

import pytest

from agent_trader.data import synthetic
from agent_trader.research import stress
from agent_trader.research.backtest import HOUR, Market
from agent_trader.research.stress import (HOLDOUT_START_TS, assert_pre_holdout, chain_decisions, folds,
                                          join_markets, load_pre_holdout, matched_static, static_decisions,
                                          tune_fold, walk_forward, window)
from agent_trader.research.strategies import (drawdown_brake, dual_momentum, trend_ladder,
                                              vol_target_ladder)

REAL_OLD = Path("/mnt/project-files/data/binance_early")
REAL_DEV = Path("/mnt/project-files/data/binance")

# Small grids with short lookbacks so a whole walk-forward runs in a second on synthetic data.
SMALL_GRIDS = {
    "S1": {"b": (3, 5), "cadence": ("daily", "weekly")},
    "S2": {"vol_target": (0.4, 0.8), "n": (5, 10)},
    "S3": {"lookback": (5, 10), "tilt": (0.5, 1.0)},
    "S4": {"max_dd": (0.05, 0.10), "window": (5, 10)},
}
SMALL = dict(train_months=2, test_months=1, warmup=720, grids=SMALL_GRIDS, levels=(0.5,))


def _market(days: int = 300, t0: int = 0) -> Market:
    eth = synthetic("ETH", days * 24, seed=1, start_price=1500, vol=0.012)
    btc = synthetic("BTC", days * 24, seed=2, start_price=20000, vol=0.008)
    ts = [t0 + i * HOUR for i in range(days * 24)]
    return Market(ts, {"ETH": [c.open for c in eth], "BTC": [c.open for c in btc]},
                  {"ETH": [c.close for c in eth], "BTC": [c.close for c in btc]})


def _write(path: Path, rows: list[tuple]) -> None:
    path.write_text("ts,open,high,low,close,volume\n" + "".join(",".join(map(str, r)) + "\n" for r in rows))


def test_holdout_start_is_2025_07_12_19_utc():
    assert HOLDOUT_START_TS == int(datetime(2025, 7, 12, 19, tzinfo=timezone.utc).timestamp()) == 1752346800


# ----------------------------------------------------------------------------- no holdout bars anywhere


def test_loader_stops_before_the_holdout_and_never_parses_holdout_rows(tmp_path):
    cut = 10 * HOUR
    for sym, px in (("eth", 100.0), ("btc", 200.0)):
        rows = [(i * HOUR, px, px, px, px, 1) for i in range(10)]
        rows += [(i * HOUR, "garbage", "x", "x", "garbage", "x") for i in range(10, 15)]  # would fail to parse
        _write(tmp_path / f"{sym}_1h.csv", rows)
    m = load_pre_holdout(tmp_path, cutoff=cut)
    assert m.n == 10 and max(m.ts) == cut - HOUR


def test_every_round2_entry_point_refuses_holdout_bars():
    m = _market(days=60, t0=HOLDOUT_START_TS - 60 * 24 * HOUR + HOUR)  # last bar = holdout start
    assert m.ts[-1] == HOLDOUT_START_TS
    with pytest.raises(ValueError, match="holdout"):
        assert_pre_holdout(m)
    with pytest.raises(ValueError, match="holdout"):
        stress.one_shot(m)
    with pytest.raises(ValueError, match="holdout"):
        stress.exposure_study(m, 720, 1439)
    with pytest.raises(ValueError, match="holdout"):
        walk_forward(m, **SMALL)
    assert_pre_holdout(window(m, 0, m.n - 2))  # one bar earlier is fine


@pytest.mark.skipif(not (REAL_DEV / "eth_1h.csv").exists(), reason="real data not available")
def test_real_dev_data_is_cut_exactly_before_the_holdout():
    m = load_pre_holdout(REAL_DEV)
    assert m.n == 25_199                       # bars 0..25198: warm-up + dev, as in round 1
    assert m.ts[-1] == HOLDOUT_START_TS - HOUR  # 2025-07-12 18:00, the last dev bar


# ----------------------------------------------------------------------------- joining older and dev data


def test_join_keeps_order_and_rejects_duplicates_overlaps_and_seams():
    m = _market(days=20)
    a, b = window(m, 0, 239), window(m, 240, m.n - 1)
    j = join_markets(a, b)
    assert j.ts == m.ts and j.close == m.close and j.open == m.open
    assert all(y - x == HOUR for x, y in zip(j.ts, j.ts[1:]))
    with pytest.raises(ValueError, match="strictly increasing"):
        join_markets(a, window(m, 239, m.n - 1))           # duplicate bar at the seam
    with pytest.raises(ValueError, match="strictly increasing"):
        join_markets(a, window(m, 100, m.n - 1))           # overlap
    with pytest.raises(ValueError, match="strictly increasing"):
        join_markets(b, a)                                 # wrong order
    swapped = Market(b.ts, {"ETH": b.open["BTC"], "BTC": b.open["ETH"]},
                     {"ETH": b.close["BTC"], "BTC": b.close["ETH"]})
    with pytest.raises(ValueError, match="do not line up"):
        join_markets(a, swapped)                           # ETH/BTC columns swapped in one file
    off = Market([t + 60 for t in b.ts], b.open, b.close)
    with pytest.raises(ValueError, match="whole UTC hours"):
        join_markets(a, off)


def test_loader_rejects_misaligned_symbols_and_unsorted_rows(tmp_path):
    _write(tmp_path / "eth_1h.csv", [(i * HOUR, 1, 1, 1, 1, 1) for i in range(5)])
    _write(tmp_path / "btc_1h.csv", [(i * HOUR, 1, 1, 1, 1, 1) for i in (0, 1, 2, 4, 5)])
    with pytest.raises(ValueError, match="differ"):
        load_pre_holdout(tmp_path)
    _write(tmp_path / "btc_1h.csv", [(i * HOUR, 1, 1, 1, 1, 1) for i in (0, 1, 1, 2, 3)])
    with pytest.raises(ValueError, match="strictly increasing"):
        load_pre_holdout(tmp_path)


@pytest.mark.skipif(not (REAL_OLD / "eth_1h.csv").exists() or not (REAL_DEV / "eth_1h.csv").exists(),
                    reason="real data not available")
def test_real_joined_data_has_no_duplicate_or_misaligned_bars():
    old, dev = load_pre_holdout(REAL_OLD), load_pre_holdout(REAL_DEV)
    j = join_markets(old, dev)
    assert j.n == old.n + dev.n == 69_151
    assert len(set(j.ts)) == j.n and all(b > a for a, b in zip(j.ts, j.ts[1:]))
    assert dev.ts[0] - old.ts[-1] == HOUR
    assert max(j.ts) < HOLDOUT_START_TS
    for s in j.symbols:  # each symbol's prices follow its own timestamps through the seam
        assert j.close[s][old.n - 1] == old.close[s][-1] and j.open[s][old.n] == dev.open[s][0]


# ----------------------------------------------------------------------------- train/test windows


def test_folds_never_overlap_and_tests_chain():
    for n, tr, te, w in ((69_151, 17_280, 4_320, 3_599), (25_199, 8_640, 2_160, 3_599), (7_200, 1_440, 720, 720)):
        fs = folds(n, tr, te, w)
        assert fs[-1].test[1] == n - 1 and fs[0].warm >= 0
        for i, f in enumerate(fs):
            assert f.train[1] - f.train[0] + 1 == tr and f.test[1] - f.test[0] + 1 == te
            assert f.train[1] < f.test[0] and f.train[1] + 1 == f.test[0]
            assert f.warm == f.train[0] - w
            if i:
                assert fs[i - 1].test[1] + 1 == f.test[0]  # test windows chain without overlap or gap
    assert len(folds(69_151, 17_280, 4_320, 3_599)) == 11


def test_tuning_sees_no_test_window_bar():
    m = _market()
    seen: list[int] = []

    def spy(fn):
        def wrapped(mk, **kw):
            seen.append(mk.ts[-1])
            return fn(mk, **kw)
        return wrapped

    build = {"S1": spy(trend_ladder), "S2": spy(vol_target_ladder), "S3": spy(dual_momentum),
             "S4": spy(drawdown_brake)}
    for f in folds(m.n, 1440, 720, 720):
        seen.clear()
        tune_fold(m, f, SMALL_GRIDS, build)
        assert seen and max(seen) == m.ts[f.train[1]] < m.ts[f.test[0]]


def test_walk_forward_picks_ignore_prices_from_their_test_window_on():
    m = _market()
    base = walk_forward(m, **SMALL)
    k = 3
    shocked = walk_forward(m.scaled_after(base.folds[k].test[0] - 1, 10.0), **SMALL)
    assert shocked.picks[: k + 1] == base.picks[: k + 1]
    # the chained curve before fold k's test window is untouched as well
    upto = base.folds[k].test[0] - 1
    for key in ("S1", "S2", "S3", "S4", "REF"):
        assert shocked.rows[key].run.equity[: upto + 1] == base.rows[key].run.equity[: upto + 1]


def test_chained_decisions_come_from_each_folds_own_pick():
    m = _market()
    fs = folds(m.n, 1440, 720, 720)
    picks = [{"b": 3 if i % 2 else 5, "cadence": "daily"} for i in range(len(fs))]
    dec = chain_decisions(m, fs, "S1", picks)
    assert min(dec) == fs[0].test[0] - 1           # cash until the switch at the first train end
    for f, p in zip(fs, picks):
        own = trend_ladder(window(m, 0, f.test[1]), **p)
        # the last bar of a test window is the next fold's switch bar, where the new pick takes over
        last = f.test[1] if f is fs[-1] else f.test[1] - 1
        inside = [t for t in own if f.test[0] <= t <= last]
        assert inside and all(dec[t] == own[t] for t in inside)
        assert dec[f.test[0] - 1] == own[max(t for t in own if t < f.test[0])]
    assert all(fs[0].test[0] - 1 <= t <= fs[-1].test[1] for t in dec)


# ----------------------------------------------------------------------------- statics


def test_static_rebalances_on_sundays_and_matches_exposure():
    m = _market()
    dec = static_decisions(m, 0.4)
    assert dec and all(w == {"ETH": 0.2, "BTC": 0.2} for w in dec.values())
    days = {m.day_bars.index(t) for t in dec}
    assert all(m.is_sunday(d) for d in days) and len(days) == sum(m.is_sunday(d) for d in range(len(m.day_bars)))
    level, run = matched_static(m, 0.37, 720, m.n - 1)
    got = sum(run.exposure[720:]) / (m.n - 720)
    assert abs(got - 0.37) <= 0.002

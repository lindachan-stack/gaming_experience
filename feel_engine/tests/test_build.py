# -*- coding: utf-8 -*-
"""SPEC 13.1：只驗證程式邏輯（合成資料，不讀快取）。"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import numpy as np
import polars as pl
import pytest

from feel_engine.pipeline.build import (
    DATASET_COLUMNS,
    FEATURES,
    GP_FEATURES,
    INDEX_COLUMNS,
    NGP_FEATURES,
    SY_FEATURES,
    SEG_COLUMNS,
    TAG_COLUMNS,
    BuildParams,
    add_features,
    add_stats,
    add_weights,
    attach_tags,
    build_spin_table,
    decision_table,
    merge_free_spins,
    select_decisions,
)
from feel_engine.pipeline.data import (
    BUCKET_COLUMNS,
    ET,
    TZ,
    DataCheckError,
    outcome_expr,
    plan_read,
    pre_range_last_paid,
    prepare_day,
    write_buckets,
)

VMAP = {"v_umb": "保護傘", "v_base": "原模組", "v_high": "高倍數", "v_high2": "高倍數"}
T0 = datetime(2026, 8, 18, 10, 0, tzinfo=ET)
DAY = T0.date()


def make_raw(rows: list[dict]) -> pl.DataFrame:
    """rows：member_account、ts（aware）、bet_points、prize、points、last_points、game_version、is_buy、is_free；
    seq 未給時依列序。"""
    data = {
        "member_account": [r.get("member_account", "p1") for r in rows],
        "seq": [r.get("seq", i) for i, r in enumerate(rows)],
        "ts": [r["ts"].astimezone(timezone.utc).replace(tzinfo=None) for r in rows],
        "bet_points": [r.get("bet_points", 100) for r in rows],
        "prize": [r.get("prize", 0) for r in rows],
        "points": [r.get("points", 10_000) for r in rows],
        "last_points": [r.get("last_points", 10_000) for r in rows],
        "game_version": [r.get("game_version", "v_base") for r in rows],
        "is_buy": [r.get("is_buy", False) for r in rows],
        "is_free": [r.get("is_free", False) for r in rows],
    }
    df = pl.DataFrame(data, schema_overrides={"seq": pl.Int64, "bet_points": pl.Int64, "prize": pl.Int64,
                                              "points": pl.Int64, "last_points": pl.Int64})
    return df.with_columns(pl.col("ts").dt.replace_time_zone("UTC").dt.convert_time_zone(TZ)).select(BUCKET_COLUMNS)


def params(**kw) -> BuildParams:
    base = dict(
        window=20,
        session_gap=timedelta(minutes=30),
        return_window=timedelta(hours=24),
        days=(DAY,),
        data_end=datetime(2100, 1, 1, tzinfo=ET),
        arm_dominance=0.8,
        player_day_cap=50,
        dry_min_spins=100,
        cold_nowin_rate=0.8,
        loss_ratio=0.3,
        seed=42,
        version_map=VMAP,
        provisional_arm=False,
    )
    base.update(kw)
    return BuildParams(**base)


def spin_series(times: list[datetime], **kw) -> list[dict]:
    return [{"ts": t, **kw} for t in times]


def features_at(raw: pl.DataFrame, i: int, **kw) -> dict:
    p = params(**kw)
    spins, _ = build_spin_table(raw, p)
    return add_features(spins, p).row(i, named=True)


# ---------------------------------------------------------------------------
# 逐手序列
# ---------------------------------------------------------------------------
def test_merge_free_spins_into_previous_paid_spin():
    raw = make_raw([
        {"ts": T0, "bet_points": 100, "prize": 0, "points": 1000, "last_points": 900},
        {"ts": T0 + timedelta(seconds=1), "bet_points": 0, "prize": 300, "points": 900, "last_points": 1200},
        {"ts": T0 + timedelta(seconds=2), "bet_points": 0, "prize": 200, "points": 1200, "last_points": 1400},
    ])
    spins, stats = merge_free_spins(raw)
    assert stats["r6_orphan_free_rows"] == 0 and stats["free_merged"] == 2 and stats["free_late"] == 0
    row = spins.row(0, named=True)
    assert spins.height == 1 and row["prize"] == 500 and row["mult"] == pytest.approx(5.0)
    assert row["outcome"] == "SMALL_WIN"
    assert row["balance_before"] == 1000 and row["balance_after"] == 1400


def test_orphan_free_rows_are_excluded():
    raw = make_raw([{"ts": T0, "bet_points": 0, "prize": 50}, {"ts": T0 + timedelta(seconds=1), "bet_points": 100}])
    spins, stats = merge_free_spins(raw)
    assert stats["r6_orphan_free_rows"] == 1 and spins.height == 1 and spins["prize"][0] == 0


def test_late_free_rows_are_merged_but_counted():
    raw = make_raw([
        {"ts": T0, "bet_points": 100},
        {"ts": T0 + timedelta(hours=5), "bet_points": 0, "prize": 70, "game_version": "22071_card_1.0.0_4"},
    ])
    spins, stats = merge_free_spins(raw)
    assert spins["prize"].to_list() == [70]          # SPEC：仍併入前一個付費手
    assert stats["free_late"] == 1 and stats["free_late_prize"] == 70


def test_outcome_boundaries():
    df = pl.DataFrame({"mult": [0.0, 0.5, 1.0, 7.0, 30.0]}).with_columns(outcome=outcome_expr(pl.col("mult")))
    assert df["outcome"].to_list() == ["NO_WIN", "TINY_WIN", "SMALL_WIN", "MEDIUM_WIN", "BIG_WIN"]


def test_same_second_keeps_file_order():
    raw = make_raw([{"ts": T0, "bet_points": 200, "seq": 2}, {"ts": T0, "bet_points": 100, "seq": 1}])
    spins, _ = merge_free_spins(raw)
    assert spins["bet"].to_list() == [100, 200]


def test_free_spin_version_does_not_change_arm():
    raw = make_raw([
        {"ts": T0, "bet_points": 100, "game_version": "v_umb"},
        {"ts": T0 + timedelta(seconds=1), "bet_points": 0, "prize": 300, "game_version": "22071_card_1.0.0_4"},
    ])
    spins, _ = build_spin_table(raw, params())
    assert spins["arm"].to_list() == ["保護傘"]


# ---------------------------------------------------------------------------
# 特徵
# ---------------------------------------------------------------------------
POOL_ROWS = [  # (bet, prize, balance_before)
    (100, 0, 10_000), (100, 50, 9_900), (100, 0, 9_850), (100, 800, 9_750),
    (100, 0, 10_450), (100, 0, 10_350), (200, 3_500, 10_250),
]


def pool_raw(last_prize: int | None = None, buy_at: int | None = None, free_at: tuple[int, ...] = ()) -> pl.DataFrame:
    rows = []
    for i, (bet, prize, bal) in enumerate(POOL_ROWS):
        if i == 6 and last_prize is not None:
            prize = last_prize
        rows.append({"ts": T0 + timedelta(minutes=i), "bet_points": bet, "prize": prize,
                     "points": bal, "last_points": bal - bet + prize, "is_buy": i == buy_at, "is_free": i in free_at})
    return make_raw(rows)


def test_pool_features_example():
    f = features_at(pool_raw(), 6)
    expected = {
        "GP_dailySpinCount": 6, "GP_dailyRate_NO_WIN": 4 / 6, "GP_dailyRate_TINY_WIN": 1 / 6,
        "GP_dailyRate_SMALL_WIN": 0.0, "GP_dailyRate_MEDIUM_WIN": 1 / 6, "GP_dailyRate_BIG_WIN": 0.0, "GP_dailyRtp": 850 / 600,
        "GP_dailyRate_MEDIUM_PLUS": 1 / 6, "GP_dailyNoWin_x_spins": 1.2973,
        "GP_flag_dryLong": 0, "NGP_flag_coldLoss": 0, "GP_dailyRate_FREE_GAME": 0.0,
        "SY_bet": 200, "SY_isBuyFeature": 0, "SY_first_balance": 10_000, "SY_balance": 10_250, "SY_balanceToBet": 51.25,
        "NGP_dailyTotalWinToBet": 4.25,
    }
    for k, v in expected.items():
        assert f[k] == pytest.approx(v, abs=1e-4), k
    assert set(expected) == set(FEATURES)


def test_free_game_rate_uses_previous_spins_only():
    """第 3 手中 FreeGame → 第 6 手 1/6；當手（第 6 手）中 FreeGame 不改變任何特徵。"""
    keys = list(FEATURES)
    f = features_at(pool_raw(free_at=(3,)), 6)
    assert f["GP_dailyRate_FREE_GAME"] == pytest.approx(1 / 6)
    assert features_at(pool_raw(free_at=(3,)), 3)["GP_dailyRate_FREE_GAME"] == 0.0
    base, current = features_at(pool_raw(), 6), features_at(pool_raw(free_at=(6,)), 6)
    assert {k: base[k] for k in keys} == {k: current[k] for k in keys}
    assert features_at(pool_raw(free_at=(0,)), 0)["GP_dailyRate_FREE_GAME"] is None      # 今日首手：0 ÷ 0


@pytest.mark.parametrize("last_prize", [0, 100_000])
def test_features_only_use_previous_spins(last_prize):
    base, changed = features_at(pool_raw(), 6), features_at(pool_raw(last_prize), 6)
    assert {k: base[k] for k in FEATURES} == {k: changed[k] for k in FEATURES}


def test_gp_ignores_current_spin_ngp_uses_it():
    """GP_ 只用當手之前（與當手時間）：改當手下注額、下注前餘額、BUY，GP_ 全部不變；SY_、NGP_ 會跟著變。"""
    current = pl.int_range(pl.len()) == 6
    alt = pool_raw().with_columns(
        bet_points=pl.when(current).then(500).otherwise(pl.col("bet_points")),
        points=pl.when(current).then(99_999).otherwise(pl.col("points")),
        is_buy=current,
    )
    f0, f1 = features_at(pool_raw(), 6), features_at(alt, 6)
    assert {k: f0[k] for k in GP_FEATURES} == {k: f1[k] for k in GP_FEATURES}
    assert all(f0[k] != f1[k] for k in ("SY_bet", "SY_isBuyFeature", "SY_balance", "SY_balanceToBet",
                                         "NGP_dailyTotalWinToBet"))


@pytest.mark.parametrize("first, second, expected", [
    (datetime(2026, 8, 10, 3, 59), datetime(2026, 8, 10, 4, 1), 0),    # 8 月 EDT：UTC 04:00 為界
    (datetime(2026, 1, 15, 4, 1), datetime(2026, 1, 15, 4, 59), 1),    # 1 月 EST：04:00 不是界
    (datetime(2026, 1, 15, 4, 59), datetime(2026, 1, 15, 5, 1), 0),    # 1 月 EST：UTC 05:00 為界
])
def test_et_day_boundary(first, second, expected):
    raw = make_raw([{"ts": first.replace(tzinfo=timezone.utc)}, {"ts": second.replace(tzinfo=timezone.utc)}])
    assert features_at(raw, 1)["GP_dailySpinCount"] == expected


def test_multi_condition_flags():
    rows = [{"ts": T0 + timedelta(minutes=i), "prize": 0, "points": 10_000 if i == 0 else (6_000 if i == 120 else 9_000)}
            for i in range(121)]
    f = features_at(make_raw(rows), 120)
    assert f["GP_dailySpinCount"] == 120
    assert f["GP_flag_dryLong"] == 1 and f["NGP_flag_coldLoss"] == 1


def test_buy_spin_counts_in_pool_and_is_flagged():
    base, buy = features_at(pool_raw(), 6), features_at(pool_raw(buy_at=3), 6)
    pool = ["GP_dailySpinCount", *(f for f in FEATURES if f.startswith("GP_dailyRate_")), "GP_dailyRtp"]
    assert {k: base[k] for k in pool} == {k: buy[k] for k in pool}
    assert features_at(pool_raw(buy_at=3), 3)["SY_isBuyFeature"] == 1


def test_new_session_entry_balance():
    first = [{"ts": T0 + timedelta(minutes=i), "points": 10_000 - 100 * i} for i in range(5)]
    second = [{"ts": T0 + timedelta(hours=1, minutes=4 + i), "points": 8_000 - 100 * i} for i in range(3)]
    p = params()
    spins, _ = build_spin_table(make_raw(first + second), p)
    f = add_features(spins, p).filter(pl.col("session_id") == 2)
    assert f["SY_first_balance"].to_list() == [8_000] * 3
    assert f["session_spin_idx"].to_list() == [0, 1, 2]


# ---------------------------------------------------------------------------
# Y 與決策點
# ---------------------------------------------------------------------------
def decisions(times: list[datetime], versions: list[str] | None = None, **kw) -> pl.DataFrame:
    versions = versions or ["v_base"] * len(times)
    raw = make_raw([{"ts": t, "game_version": v} for t, v in zip(times, versions)])
    kw.setdefault("data_end", times[-1] + timedelta(days=3))
    p = params(**kw)
    spins, _ = build_spin_table(raw, p)
    return select_decisions(spins, p)[0]


def run_of(n: int, start: datetime = T0) -> list[datetime]:
    return [start + timedelta(minutes=i) for i in range(n)]


def y_by_idx(dp: pl.DataFrame, session_id: int = 1) -> dict[int, int]:
    s = dp.filter(pl.col("session_id") == session_id)
    return dict(zip(s["session_spin_idx"].to_list(), s["churnNext20"].to_list()))


def test_churn_after_two_day_break():
    times = run_of(25) + [T0 + timedelta(minutes=24, days=2)]
    assert y_by_idx(decisions(times)) == {0: 0, 20: 1}


def test_no_churn_when_back_within_window():
    times = run_of(25) + [T0 + timedelta(minutes=24, hours=2)]
    assert y_by_idx(decisions(times)) == {0: 0, 20: 0}


def test_fifty_spin_session_has_three_decisions_only_last_can_churn():
    times = run_of(50) + [T0 + timedelta(days=3)]
    dp = decisions(times).filter(pl.col("session_id") == 1)
    assert dp["session_spin_idx"].to_list() == [0, 20, 40]
    assert dp["is_last_dp"].to_list() == [False, False, True]
    assert dp["churnNext20"].to_list() == [0, 0, 1]


@pytest.mark.parametrize("n_umb, n_high, kept", [(17, 3, True), (15, 5, False)])
def test_dominant_arm_sampling(n_umb, n_high, kept):
    dp = decisions(run_of(20), ["v_umb"] * n_umb + ["v_high"] * n_high)
    assert (dp.height == 1) is kept
    if kept:
        assert dp["arm"][0] == "保護傘" and dp["arm_ratio"][0] == pytest.approx(n_umb / 20)


def test_dominance_uses_arm_not_version():
    dp = decisions(run_of(20), ["v_high"] * 10 + ["v_high2"] * 10)
    assert dp["arm"].to_list() == ["高倍數"]


def test_censored_session_is_removed_entirely():
    times = run_of(45)
    assert decisions(times, data_end=times[-1] + timedelta(hours=10)).height == 0     # 看不完 24h：整段 session 不抽
    assert decisions(times, data_end=times[-1] + timedelta(hours=30))["session_spin_idx"].to_list() == [0, 20, 40]


def test_censoring_uses_each_session_start_day():
    """截尾以起點日為準：session 須在隔天 0 點 + post_hours（12h）前結束；與讀取範圍終點多晚無關。"""
    late = datetime(2026, 8, 18, 23, 0, tzinfo=ET)
    long_run = [late + timedelta(minutes=30 * i) for i in range(28)]          # 23:00 起每 30 分鐘一手，到隔天 12:30
    assert decisions(long_run, data_end=long_run[-1] + timedelta(days=30)).height == 0
    short_run = long_run[:26]                                                  # 到隔天 11:30 結束
    assert decisions(short_run, data_end=short_run[-1] + timedelta(days=30))["session_spin_idx"].to_list() == [0, 20]


def test_a_day_is_the_same_whether_built_alone_or_with_the_previous_day():
    """逐日 dataset：同一天單獨建、或和前一天一起建，決策點（含 session_id）完全相同。"""
    prev_day = DAY - timedelta(days=1)
    rows = [{"member_account": "p1", "ts": datetime(2026, 8, 17, 9, 0, tzinfo=ET) + timedelta(minutes=i)} for i in range(25)]
    rows += [{"member_account": "p1", "ts": datetime(2026, 8, 17, 23, 50, tzinfo=ET) + timedelta(minutes=i)} for i in range(30)]
    rows += [{"member_account": "p1", "ts": T0 + timedelta(hours=h, minutes=i)} for h in (0, 3) for i in range(41)]
    rows += [{"member_account": "p2", "ts": T0 + timedelta(minutes=i)} for i in range(21)]
    data_end = T0 + timedelta(days=3)

    def built(days, start):
        p = params(days=days, data_end=data_end)
        raw = make_raw([r for r in rows if r["ts"] >= start])
        dp, _ = decision_table(build_spin_table(raw, p)[0], p)
        return dp.filter(pl.col("session_date") == DAY).select(
            "member_account", "spin_ts", "session_id", "session_spin_idx", "churnNext20", "w_cap").sort("member_account", "spin_ts")

    alone = built((DAY,), datetime(2026, 8, 17, 18, 0, tzinfo=ET))           # 範圍起點 = DAY 0 點 − 6h
    together = built((prev_day, DAY), datetime(2026, 8, 16, 18, 0, tzinfo=ET))
    assert alone.height == 3 + 3 + 2 and alone.equals(together)
    # session_id = yyyymmdd × 1000 ＋ 當天第幾個 session：p1 前一天 23:50 起的 session 歸 08-17，08-18 第 1、2 個是 10:00、13:00
    assert sorted(set(alone.filter(pl.col("member_account") == "p1")["session_id"].to_list())) == [20260818001, 20260818002]


def test_stats_by_day_add_up_to_totals():
    nxt = DAY + timedelta(days=1)
    times = run_of(45) + run_of(25, datetime(2026, 8, 19, 9, 0, tzinfo=ET))
    raw = make_raw([{"member_account": "p1", "ts": t} for t in times])
    p = params(days=(DAY, nxt), data_end=times[-1] + timedelta(days=3))
    dp, stats = decision_table(build_spin_table(raw, p)[0], p)
    by_day = stats["by_day"]
    assert set(by_day) == {str(DAY), str(nxt)}
    total = add_stats(by_day[str(DAY)], by_day[str(nxt)])
    assert total["funnel"]["after_dominance"] == stats["funnel"]["after_dominance"] == dp.height == 5
    assert total["weights"]["n"] == stats["weights"]["n"] and total["arm_counts"] == {"原模組": 5}
    assert by_day[str(DAY)]["funnel"] == {"on_days": 3, "after_censor": 3, "after_session_start": 3, "after_dominance": 3}


def test_decisions_belong_to_the_day_the_session_started():
    """決策點歸屬日 = session 起點的美東日期：跨午夜的 session 整段歸起點那天。"""
    from_prev = datetime(2026, 8, 17, 23, 50, tzinfo=ET)    # 前一天 23:50 開始、跨過午夜 → 整段歸 08-17
    assert decisions(run_of(45, from_prev)).height == 0
    into_next = datetime(2026, 8, 18, 23, 50, tzinfo=ET)    # 當天 23:50 開始、跨過午夜 → 午夜後的決策點也算當天
    dp = decisions(run_of(45, into_next))
    assert dp["session_spin_idx"].to_list() == [0, 20, 40]
    assert dp["spin_ts"][-1].date() == date(2026, 8, 19) and set(dp["session_date"].to_list()) == {DAY}


def test_train_and_valid_days_are_cut_at_session_start():
    """兩天一起處理：session_date = session 起點日；交界的 session 整段歸前一天（訓練日），兩天沒有共同的 session。"""
    nxt = DAY + timedelta(days=1)
    cross = run_of(45, datetime(2026, 8, 18, 23, 50, tzinfo=ET))        # 訓練日 23:50 開始、跨午夜
    later = run_of(25, datetime(2026, 8, 19, 9, 0, tzinfo=ET))           # 驗證日 09:00 開始
    raw = make_raw([{"member_account": "p1", "ts": t} for t in cross + later])
    p = params(days=(DAY, nxt), data_end=later[-1] + timedelta(days=3))
    spins, _ = build_spin_table(raw, p)
    dp, _ = select_decisions(spins, p)
    by_day = {d: g["session_id"].unique().to_list() for (d,), g in dp.group_by("session_date")}
    assert by_day == {DAY: [1], nxt: [2]}                                 # 交界的 session 只在訓練日
    assert dp.filter(pl.col("session_id") == 1)["spin_ts"].dt.date().to_list() == [DAY, nxt, nxt]


def test_session_cut_by_read_range_start_has_no_decisions():
    """讀取範圍起點前 30 分鐘內有付費手 → 範圍內第一個 session 的起點是被切出來的，不抽決策點；之後的 session 照常。"""
    utc = lambda t: t.astimezone(timezone.utc).replace(tzinfo=None)  # noqa: E731
    rows = [{"member_account": "p1", "ts": T0 + timedelta(minutes=i)} for i in range(25)]
    rows += [{"member_account": "p1", "ts": T0 + timedelta(hours=2, minutes=i)} for i in range(5)]   # p1 第二個 session
    rows += [{"member_account": a, "ts": T0 + timedelta(minutes=i)} for a in ("p2", "p3") for i in range(25)]
    pre = pl.DataFrame({"member_account": ["p1", "p2"],
                        "last_paid": [utc(T0 - timedelta(minutes=10)), utc(T0 - timedelta(minutes=40))]})
    p = params(data_end=T0 + timedelta(days=3))
    spins, _ = build_spin_table(make_raw(rows), p, pre_last=pre)
    dp, stats = select_decisions(spins, p)
    got = set(zip(dp["member_account"].to_list(), dp["session_id"].to_list()))
    assert ("p1", 1) not in got                                          # 10 分鐘前還在玩 → 起點不完整
    assert {("p1", 2), ("p2", 1), ("p3", 1)} <= got                     # 40 分鐘前、或範圍前沒玩 → 完整
    assert stats["funnel"]["after_censor"] - stats["funnel"]["after_session_start"] == 2   # p1 第一個 session 的第 0、20 手


def test_pre_range_last_paid_only_paid_spins_in_window(tmp_path):
    start = datetime(2026, 8, 28, 21, 30)                                # UTC naive
    df = pl.DataFrame({"member_account": ["a", "a", "b", "c"],
                       "create_at": [start + timedelta(minutes=5), start + timedelta(minutes=20),
                                     start + timedelta(minutes=10), start - timedelta(minutes=1)],
                       "bet_points": [100, 0, 100, 100]})                 # a 的第二筆是免費列；c 在窗口外
    df.write_parquet(tmp_path / "d.parquet")
    got = pre_range_last_paid([tmp_path / "d.parquet"], start.replace(tzinfo=timezone.utc), start.replace(tzinfo=timezone.utc) + timedelta(minutes=30))
    assert dict(zip(got["member_account"].to_list(), got["last_paid"].to_list())) == {
        "a": start + timedelta(minutes=5), "b": start + timedelta(minutes=10)}


def test_session_table_matches_forward_scan():
    rng = np.random.default_rng(0)
    gaps = rng.choice([5, 60, 40 * 60, 3 * 3600, 30 * 3600], size=999, p=[0.6, 0.3, 0.05, 0.03, 0.02])
    secs = np.concatenate([[0], np.cumsum(gaps)])
    times = [T0 + timedelta(seconds=int(s)) for s in secs]
    p = params(data_end=times[-1] + timedelta(days=3))
    spins, _ = build_spin_table(make_raw(spin_series(times)), p)
    got = spins.filter(pl.col("is_dp"))["churnNext20"].to_list()

    session_gap, ret = 30 * 60, 24 * 3600
    expected, idx = [], 0
    for k in range(len(secs)):
        idx = 0 if k == 0 or secs[k] - secs[k - 1] > session_gap else idx + 1
        if idx % 20:
            continue
        y = 0
        for j in range(k, min(k + 20, len(secs))):
            if j == len(secs) - 1 or secs[j + 1] - secs[j] > session_gap:
                y = int(j == len(secs) - 1 or secs[j + 1] - secs[j] > ret)
                break
        expected.append(y)
    assert sum(expected) > 10 and got == expected


# ---------------------------------------------------------------------------
# 讀取範圍（R7）
# ---------------------------------------------------------------------------
AUG = [date(2026, 8, 1) + timedelta(days=i) for i in range(31)]


def test_read_plan_for_0829():
    plan = plan_read(date(2026, 8, 29), date(2026, 8, 29), 6, 24, AUG, post_hours=12)
    assert plan.range_start == datetime(2026, 8, 28, 18, 0, tzinfo=ET)
    assert plan.range_end == datetime(2026, 8, 31, 12, 0, tzinfo=ET)       # 24:00 ＋ 跨午夜 12h ＋ Y 的 24h
    assert plan.range_start.astimezone(timezone.utc).hour == 22
    assert plan.range_days == tuple(date(2026, 8, d) for d in (28, 29, 30, 31))
    assert not plan.back_short


def test_read_plan_covers_train_and_valid_days_in_one_range():
    plan = plan_read(date(2026, 8, 28), date(2026, 8, 29), 6, 24, AUG, post_hours=12)
    assert (plan.range_start, plan.range_end) == (datetime(2026, 8, 27, 18, 0, tzinfo=ET), datetime(2026, 8, 31, 12, 0, tzinfo=ET))
    assert plan.range_days == tuple(date(2026, 8, d) for d in (27, 28, 29, 30, 31))


def test_read_plan_front_only_needs_the_read_range():
    """R7 前段：快取只要涵蓋讀取範圍起點（不必從當月 1 日起）。"""
    late = AUG[20:]                                                       # 快取從 8/21 起
    plan = plan_read(date(2026, 8, 29), date(2026, 8, 29), 6, 24, late, post_hours=12)
    assert plan.range_days == tuple(date(2026, 8, d) for d in (28, 29, 30, 31))
    with pytest.raises(DataCheckError, match="前段"):
        plan_read(date(2026, 8, 21), date(2026, 8, 21), 6, 24, late, post_hours=12)   # 範圍起點 8/20 18:00 不在快取


def test_read_plan_0831_aborts():
    with pytest.raises(DataCheckError, match="分析日"):
        plan_read(date(2026, 8, 30), date(2026, 8, 31), 6, 24, AUG)


def test_read_plan_0830_back_short_is_censored():
    plan = plan_read(date(2026, 8, 30), date(2026, 8, 30), 6, 24, AUG)
    assert plan.back_short and plan.data_end == datetime(2026, 8, 31, 20, 0, tzinfo=ET)


def test_read_plan_0801_front_short_aborts():
    with pytest.raises(DataCheckError, match="前段"):
        plan_read(date(2026, 8, 1), date(2026, 8, 2), 6, 24, AUG)


# ---------------------------------------------------------------------------
# 權重、標籤、分桶
# ---------------------------------------------------------------------------
def dp_frame(n_day1: int, n_day2: int) -> pl.DataFrame:
    times = [T0 + timedelta(minutes=i) for i in range(n_day1)] + \
            [T0 + timedelta(days=1, minutes=i) for i in range(n_day2)]
    df = make_raw(spin_series(times)).select("member_account", pl.col("ts").alias("spin_ts"))
    return df.with_columns(
        session_date=pl.col("spin_ts").dt.date(),
        churnNext20=(pl.int_range(pl.len()) % 7 == 0).cast(pl.Int8),
        feature=pl.int_range(pl.len()).cast(pl.Float64) * 1.5,
    )


def test_player_day_weights_keep_all_rows():
    dp = dp_frame(100, 30)
    out, stats = add_weights(dp, 50)
    assert out.height == dp.height
    by_day = out.group_by("session_date").agg(
        lo=pl.col("w_cap").min(), hi=pl.col("w_cap").max(), total=pl.col("w_cap").sum()).sort("session_date")
    assert by_day.row(0)[1:] == (0.5, 0.5, 50.0)
    assert by_day.row(1)[1:] == (1.0, 1.0, 30.0)
    assert out.select("member_account", "spin_ts", "churnNext20", "feature").equals(dp.select("member_account", "spin_ts", "churnNext20", "feature"))
    assert stats["player_days"] == 2 and stats["player_days_capped"] == 1


def test_latest_tag_per_player_before_decision_week():
    rows = pl.DataFrame({
        "member_account": ["a", "b", "c", "c", "d", "d", "d"],
        "week_index": [202634, 202625, 202630, 202629, 202633, 202635, 202636],
        "is_zero_fill": [0, 0, 1, 0, 0, 0, 0],
        "lifecycle_tag": ["活躍", "活躍", "流失", "一般", "一般", "活躍", "下滑"],
        "value_combined_tag": ["高", "中", "本週無下注", "中", "低", "高", "高"],
    })
    w = date(2026, 8, 24)                                   # ISO 202635 的週一
    ds = pl.DataFrame({"member_account": ["a", "b", "c", "d", "e"], "week_start": [w] * 5})
    got = {r["member_account"]: r for r in attach_tags(ds, {w: rows}).iter_rows(named=True)}
    assert (got["a"]["TAG_lifecycle"], got["a"]["TAG_value_combined"], got["a"]["TAG_source"]) == ("活躍", "高", "W-1")
    assert (got["b"]["TAG_lifecycle"], got["b"]["TAG_week_index"], got["b"]["TAG_source"]) == ("活躍", 202625, "更早")
    assert (got["c"]["TAG_lifecycle"], got["c"]["TAG_is_zero_fill"]) == ("流失", 1)
    assert (got["d"]["TAG_lifecycle"], got["d"]["TAG_week_index"]) == ("一般", 202633)   # 202635 以後不得使用
    assert (got["e"]["TAG_lifecycle"], got["e"]["TAG_value_combined"], got["e"]["TAG_source"]) == ("新手", "無紀錄", "無紀錄")
    assert {r["week_index"] for r in got.values()} == {202635}  # 決策週 W；TAG_week_index 才是取到的標籤週
    # 分群（8.1）：SEG_ = 取到的 TAG_，含無紀錄的預設值；d 不得用 202635、202636 的「高」
    assert {k: (r["SEG_lifecycle"], r["SEG_value"]) for k, r in got.items()} == {
        "a": ("活躍", "高"), "b": ("活躍", "中"), "c": ("流失", "本週無下注"), "d": ("一般", "低"), "e": ("新手", "無紀錄")}


def test_prepare_day_dedups_and_flags_buy_and_free():
    df = pl.DataFrame({
        "id": ["a", "b", "a", "c"], "member_account": ["p1"] * 4,
        "create_at": [datetime(2026, 8, 18, 4, 0, s) for s in range(4)],
        "bet_points": [100] * 4, "prize": [0, 0, 999, 0], "points": [1000] * 4, "last_points": [900] * 4,
        "game_version": ["v_base"] * 4, "game_mode": ["1", "19", "1", "3"],
    })
    out, stats = prepare_day(df, day_ordinal=3, version_map=VMAP)
    assert stats["r2_dup_ids"] == 1 and out.height == 3
    assert out["prize"].to_list() == [0, 0, 0]
    assert out["seq"].to_list() == [3_000_000_000, 3_000_000_001, 3_000_000_003]
    assert out["is_buy"].to_list() == [False, True, False]
    assert out["is_free"].to_list() == [False, True, True]          # 19 = Base＋Free＋BUY、3 = Base＋Free
    assert str(out["ts"].dtype.time_zone) == TZ


def test_bucketing_keeps_each_player_in_one_bucket(tmp_path):
    rng = np.random.default_rng(1)
    raw = make_raw([{"ts": T0 + timedelta(seconds=s), "member_account": f"m{rng.integers(200)}"} for s in range(2_000)])
    write_buckets(raw, 8, tmp_path, "2026-08-18")
    back = pl.read_parquet(tmp_path / "bucket=*" / "*.parquet", hive_partitioning=True)
    assert back.height == raw.height
    assert back.group_by("member_account").agg(pl.col("bucket").n_unique())["bucket"].max() == 1


def test_dataset_column_naming_and_order():
    """索引在前；快層 GP_ → SY_ → NGP_、慢層 TAG_、分群 SEG_；欄位不重複。"""
    assert DATASET_COLUMNS[:len(INDEX_COLUMNS)] == INDEX_COLUMNS
    assert len(set(DATASET_COLUMNS)) == len(DATASET_COLUMNS)
    assert all(c.startswith("GP_") for c in GP_FEATURES) and all(c.startswith("SY_") for c in SY_FEATURES)
    assert all(c.startswith("NGP_") for c in NGP_FEATURES)
    assert set(FEATURES) == set(GP_FEATURES) | set(SY_FEATURES) | set(NGP_FEATURES)
    assert all(c.startswith("TAG_") for c in TAG_COLUMNS) and all(c.startswith("SEG_") for c in SEG_COLUMNS)
    groups = [p if (p := c.split("_")[0]) in ("GP", "SY", "NGP", "TAG", "SEG") else "-" for c in DATASET_COLUMNS]
    runs = [g for i, g in enumerate(groups) if i == 0 or g != groups[i - 1]]
    assert [g for g in runs if g != "-"] == ["GP", "SY", "NGP", "TAG", "SEG"]   # 每個前綴各自連續、依此順序
    assert [c for c, g in zip(DATASET_COLUMNS, groups) if g in ("GP", "SY", "NGP")] == list(FEATURES)   # 快層只有 FEATURES
    assert "dom_share" not in DATASET_COLUMNS and "arm_ratio" in DATASET_COLUMNS

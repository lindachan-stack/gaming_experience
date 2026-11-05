# -*- coding: utf-8 -*-
"""逐日 dataset（pipeline/store.py）：缺／過期的判斷、讀取、build 紀錄清理（合成資料，不讀快取）。"""
from __future__ import annotations

import copy
from datetime import date, timedelta

import polars as pl
import pytest
import yaml

import feel_engine.paths as paths
from feel_engine.paths import CONFIG_PATH, StorePaths
from feel_engine.pipeline import store as st
from feel_engine.pipeline.data import contiguous_ranges

D1, D2, D3 = date(2026, 9, 21), date(2026, 9, 22), date(2026, 9, 23)


@pytest.fixture
def env(tmp_path, monkeypatch):
    """把 data／output 指到 tmp_path，建好注單與標籤 manifest；回傳（cfg, store）。"""
    monkeypatch.setattr(paths, "OUTPUT_ROOT", tmp_path / "output")
    monkeypatch.setattr(paths, "BET_RECORDS_ROOT", tmp_path / "bets")
    monkeypatch.setattr(paths, "TAG_ROOT", tmp_path / "tags")
    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    cfg["data"]["game"] = str(cfg["data"]["game"])
    bet = paths.bet_records_dir("TCG", "22071")
    bet.mkdir(parents=True)
    pl.DataFrame({"currency": ["PHP"] * 3, "bet_date": ["2026-09-21", "2026-09-22", "2026-09-23"],
                  "actual_rows": [10, 20, 30]}).write_csv(bet / "_manifest.csv")
    tags = paths.tag_dir("TCG", "PHP", "22071")
    tags.mkdir(parents=True)
    pl.DataFrame({"apply_week_start": ["2026-09-21"], "actual_rows": [100],
                  "max_etl_time": ["2026-10-07 05:37:02"]}).write_csv(tags / "_manifest.csv")
    return cfg, StorePaths("TCG_PHP_22071")


def write_day(cfg: dict, store: StorePaths, d: date, build_id: str, raw_rows: dict | None = None) -> None:
    store.dataset_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"member_account": ["a"], "session_date": [d]}).write_parquet(store.day(d))
    st.write_json(store.day_meta(d), {"session_date": str(d), "build_id": build_id, "fingerprint": st.fingerprint(cfg),
                                      "provisional_arm": True, "rows": 1, "tag_week": "2026-09-21", "stats": {}})
    st.write_json(store.build_meta(build_id), {
        "build_id": build_id, "stage1": {"raw_rows": raw_rows or {"2026-09-21": 10, "2026-09-22": 20}},
        "tag_weeks": {"2026-09-21": {"rows": 100, "max_etl_time": "2026-10-07 05:37:02"}}})


def test_contiguous_ranges():
    assert contiguous_ranges([D3, D1, D2, D1 + timedelta(days=7)]) == [(D1, D3), (D1 + timedelta(days=7),) * 2]
    assert contiguous_ranges([]) == []


def test_fingerprint_changes_with_build_settings_only():
    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    fp = st.fingerprint(cfg)
    other = copy.deepcopy(cfg)
    other["data"]["valid_start"], other["compute"]["n_buckets"], other["tree"]["max_depth"] = "2026-09-22", 7, 9
    assert st.fingerprint(other) == fp                       # 期間、運算、建模設定不影響決策點
    other["labels"]["return_window_hours"] = 48
    assert st.fingerprint(other) != fp


def test_day_status_ok_missing_and_stale(env):
    cfg, store = env
    write_day(cfg, store, D1, "b1")
    write_day(cfg, store, D2, "b2", raw_rows={"2026-09-22": 999})          # 建的時候注單日檔列數不同 → 快取已重抓
    status = st.day_status(cfg, store, [D1, D2, D3])
    assert status == {D1: st.OK, D2: "注單快取已重抓", D3: "還沒建"}
    changed = copy.deepcopy(cfg)
    changed["labels"]["arm_dominance"] = 0.9
    assert st.day_status(changed, store, [D1])[D1] == "build 設定或程式已改"
    assert st.check_days(cfg, store, [D1]) == []
    assert any("還沒建" in p and "2026-09-23" in p for p in st.check_days(cfg, store, [D1, D3]))


def test_tag_refetch_makes_day_stale(env):
    cfg, store = env
    write_day(cfg, store, D1, "b1")
    tags = paths.tag_dir("TCG", "PHP", "22071")
    pl.DataFrame({"apply_week_start": ["2026-09-21"], "actual_rows": [101],
                  "max_etl_time": ["2026-10-20 00:00:00"]}).write_csv(tags / "_manifest.csv")
    assert st.day_status(cfg, store, [D1])[D1] == "標籤快取已重抓"


def test_load_dataset_and_prune_builds(env):
    cfg, store = env
    write_day(cfg, store, D1, "b1")
    write_day(cfg, store, D2, "b1")
    write_day(cfg, store, D2, "b2")                                         # D2 重建 → b1 只剩 D1 用
    assert st.prune_builds(store) == []
    write_day(cfg, store, D1, "b2")                                         # D1 也重建 → b1 沒人用
    assert st.prune_builds(store) == ["b1"] and not store.build_meta("b1").exists()
    assert st.load_dataset(store, [D2, D1])["session_date"].to_list() == [D1, D2]
    with pytest.raises(FileNotFoundError):
        st.load_dataset(store, [D3])
    assert st.provisional_arm(store, [D1, D2]) is True

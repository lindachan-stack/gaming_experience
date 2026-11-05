# -*- coding: utf-8 -*-
"""實驗比較（只比 AUC）：加權 AUC、bootstrap、共同樣本、與基準差。"""
from __future__ import annotations

import numpy as np
import pytest
import pandas as pd
from sklearn.metrics import roc_auc_score

from feel_engine.html.compare import bucket_auc, compare_auc, verdict


def test_bucket_auc_matches_sklearn_with_ties_and_weights():
    rng = np.random.default_rng(0)
    n = 2000
    score = rng.choice([0.1, 0.2, 0.35, 0.6], size=n)                 # 決策樹預測值只有少數幾種 → 大量同分
    y = (rng.uniform(size=n) < score).astype(float)
    w = rng.uniform(0.2, 1.0, size=n)
    player = rng.integers(0, 300, size=n)
    got = bucket_auc(score, y, w, player, np.ones((300, 1)))[0]
    assert abs(got - roc_auc_score(y, score, sample_weight=w)) < 1e-12


def test_bootstrap_reweights_players():
    score, y, w = np.array([0.1, 0.9, 0.2, 0.8]), np.array([0.0, 1.0, 1.0, 0.0]), np.ones(4)
    player = np.array([0, 0, 1, 1])
    reps = np.array([[1.0, 1.0, 0.0], [1.0, 0.0, 1.0]])                 # 第 2 次只留玩家 0、第 3 次只留玩家 1
    # 全部：正例 0.9、0.2 對負例 0.1、0.8 → 4 組中贏 3 組 = 0.75
    assert np.allclose(bucket_auc(score, y, w, player, reps), [0.75, 1.0, 0.0])


def test_verdict():
    assert (verdict(0.01, 0.02), verdict(-0.02, -0.01), verdict(-0.01, 0.01)) == ("較佳", "較差", "無顯著差異")


def frame(n, seed, p_shift=0.0, keep=None):
    rng = np.random.default_rng(seed)
    base = rng.uniform(size=n)
    df = pd.DataFrame({"member_account": [f"p{i % 50}" for i in range(n)],
                       "spin_ts": pd.date_range("2026-08-29", periods=n, freq="min", tz="America/New_York"),
                       "y": (np.random.default_rng(99).uniform(size=n) < base).astype(int),
                       "w": 1.0, "p": np.clip(base + p_shift, 0, 1), "樹": "all"})
    df["_listed"] = True
    return df if keep is None else df.iloc[keep]


def test_compare_uses_common_rows_and_identical_models_have_zero_diff():
    full = frame(400, 1)
    subset = frame(400, 1, keep=slice(0, 200))                           # 例如 value 只建模部分群
    preds = {("all", "none"): full, ("all", "value"): frame(400, 1, p_shift=0.3), ("value", "none"): subset}
    res = compare_auc(preds, n_reps=50, seed=0)
    m = res["metrics"].set_index(["split", "tags"])
    assert res["n_common"] == 200
    assert m.loc[("all", "none"), "判定"] == "基準" and np.isnan(m.loc[("all", "none"), "− 基準"])
    assert m.loc[("value", "none"), "− 基準"] == 0 and m.loc[("value", "none"), "判定"] == "無顯著差異"
    assert m.loc[("all", "none"), "AUC_lo"] <= m.loc[("all", "none"), "AUC"] <= m.loc[("all", "none"), "AUC_hi"]


def test_load_predictions_follows_experiment_order(tmp_path):
    from feel_engine.html.compare import load_predictions
    from feel_engine.paths import RunPaths

    class Fake(RunPaths):
        @property
        def output_dir(self):
            return tmp_path

    rp = Fake("2026-08-29")
    for split, tags in [("value", "none"), ("all", "none"), ("all", "extra")]:
        path = rp.model_predictions(split, tags)
        path.parent.mkdir(parents=True)
        frame(10, 0).to_parquet(path)
    got = load_predictions(rp, [("all", "none"), ("value", "none")])
    assert list(got) == [("all", "none"), ("value", "none"), ("all", "extra")]   # 清單外的附在最後
    assert bool(got[("all", "extra")]["_listed"].iloc[0]) is False


def test_confidence_summary_scores_high_full_and_medium_half():
    from feel_engine.html.compare import confidence_summary

    rows = pd.DataFrame({"葉子": ["t/L1", "t/L2", "t/L3", "t/L4"], "w": [1.0, 1.0, 1.0, 1.0]})
    conf = {"orig": {"t/L1": "高", "t/L2": "中", "t/L3": "低"},          # L4 沒有信心 → 當無法比較
            "merged": {"t/L1": "高", "t/L2–L4": "中"},
            "merge_map": {"t/L1": "t/L1", "t/L2": "t/L2–L4", "t/L3": "t/L2–L4", "t/L4": "t/L2–L4"}}
    got = confidence_summary(rows, conf)
    assert got["orig"]["share"] == {"高": 0.25, "中": 0.25, "低": 0.25, "無法比較": 0.25}
    assert got["orig"]["score"] == pytest.approx(37.5) and got["merged"]["score"] == pytest.approx(25 + 0.5 * 75)


def test_keep_current_leaves_are_left_out_of_the_score():
    from feel_engine.html.compare import confidence_summary

    rows = pd.DataFrame({"葉子": ["t/L1", "t/L2"], "w": [1.0, 3.0]})
    conf = {"orig": {"t/L1": "高", "t/L2": "維持現狀"}, "merged": {"t/L1": "高", "t/L2": "維持現狀"},
            "merge_map": {"t/L1": "t/L1", "t/L2": "t/L2"}}
    got = confidence_summary(rows, conf)["orig"]
    assert got["share"] == {"高": 1.0, "中": 0.0, "低": 0.0, "無法比較": 0.0} and got["score"] == pytest.approx(100)   # 沿用現行不計入
    assert np.isnan(confidence_summary(rows.iloc[[1]], conf)["orig"]["score"])      # 全部維持現狀 → 不給分

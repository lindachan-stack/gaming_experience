# -*- coding: utf-8 -*-
"""SPEC 第 9 節：葉子 × arm 評估（合成資料，不讀快取）。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from collections import Counter

from feel_engine.evaluation.leaf_eval import (best_arms, bootstrap_retention, choose_common_range, common_range, comparable_fn,
                                             leaf_arm_stats)
from feel_engine.evaluation.merge import (collapse_names, merge_incomparable, merge_recommendations,
                                          merge_same_recommendation, recommended_version)
from feel_engine.evaluation.propensity import design_matrix

# SPEC 9.3 驗收範例：一個葉子分兩個子狀態，子狀態內 arm 佔比 = 傾向分數
CELLS = {  # 子狀態 → arm → (n, 流失率)
    "A": {"保護傘": (1600, 0.32), "原模組": (100, 0.40), "深模組": (100, 0.38), "高倍數": (2200, 0.30)},
    "B": {"保護傘": (400, 0.20), "原模組": (400, 0.28), "深模組": (700, 0.24), "高倍數": (4500, 0.22)},
}
ARMS = sorted(CELLS["A"])
LE_ALL = {"min_n_per_arm": 0, "overlap_min": 0.0, "min_overlap_share": 0.0}   # 不修剪：全部 arm、整片葉子


def acceptance_frame() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """回傳（df, Z, P）；P = 子狀態內 arm 佔比（正確的傾向分數）。"""
    rows = []
    for state, arms in CELLS.items():
        total = sum(n for n, _ in arms.values())
        p = [arms[a][0] / total for a in ARMS]
        for arm, (n, churn) in arms.items():
            n_churn = int(round(n * churn))
            rows += [{"葉子": "L1", "arm": arm, "state": state, "churnNext20": int(i < n_churn), "p": p} for i in range(n)]
    df = pd.DataFrame(rows)
    df["member_account"] = [f"p{i}" for i in range(len(df))]
    df["w_cap"] = 1.0
    df["_a"] = df["arm"].map(ARMS.index)
    df["_z"] = np.arange(len(df))
    P = np.array(df.pop("p").tolist())
    Z = (df[["state"]] == "A").to_numpy(dtype=float)                  # 共變數：子狀態
    return df, Z, P


def evaluate_frame(df, Z, P, M, le=LE_ALL, smd_max=10):
    df, info = common_range(df, P, le)
    return df, info, leaf_arm_stats(df, Z, M, info, smd_max=smd_max).set_index("arm")


def test_spec_9_3_acceptance_example():
    df, Z, P = acceptance_frame()
    _, _, stats = evaluate_frame(df, Z, P, np.zeros_like(P))           # 結果模型 = 0 → DR 就是 IPW
    churn_raw, churn_ipw = 1 - stats["原始留存率"], 1 - stats["加權留存率"]
    assert churn_raw["保護傘"] == pytest.approx(0.296, abs=1e-3) and churn_ipw["保護傘"] == pytest.approx(0.248, abs=1e-3)
    assert churn_raw["原模組"] == pytest.approx(0.304, abs=1e-3) and churn_ipw["原模組"] == pytest.approx(0.328, abs=1e-3)
    assert churn_raw["深模組"] == pytest.approx(0.2575, abs=1e-3) and churn_ipw["深模組"] == pytest.approx(0.296, abs=1e-3)
    assert stats["max_smd"].max() < 1e-9                               # IPW 後各 arm 的子狀態組成 = 整片葉子


def test_doubly_robust_right_if_either_model_is_right():
    df, Z, P = acceptance_frame()
    truth = {a: 1 - (0.4 * CELLS["A"][a][1] + 0.6 * CELLS["B"][a][1]) for a in ARMS}   # 子狀態標準化後的留存率
    cell_mean = df.assign(r=1 - df["churnNext20"]).groupby(["state", "arm"])["r"].mean()
    M_right = np.column_stack([cell_mean.loc[list(zip(df["state"], [a] * len(df)))].to_numpy() for a in ARMS])
    M_wrong = np.full_like(P, 0.5)
    _, _, a = evaluate_frame(df, Z, P, M_wrong)                        # 傾向分數對、結果模型錯
    _, _, b = evaluate_frame(df, Z, np.full_like(P, 0.25), M_right)    # 傾向分數錯、結果模型對
    for arm in ARMS:
        assert a.loc[arm, "加權留存率"] == pytest.approx(truth[arm], abs=1e-9)
        assert b.loc[arm, "加權留存率"] == pytest.approx(truth[arm], abs=1e-9)


def test_bootstrap_point_equals_direct_estimate():
    df, Z, P = acceptance_frame()
    M = np.random.default_rng(0).uniform(0.5, 0.9, size=P.shape)
    df, info, stats = evaluate_frame(df, Z, P, M)
    boot = bootstrap_retention(df, M, info, ARMS, n_reps=20, seed=0)
    for arm in stats.index:
        assert boot[("L1", arm)][0] == pytest.approx(stats.loc[arm, "加權留存率"], abs=1e-12)
        assert boot[("L1", arm)].shape == (21,)


# ---------------------------------------------------------------------------
# 9.1a 共同範圍
# ---------------------------------------------------------------------------
def overlap_leaf():
    """狀態 X（80%）：a、b 各半，c 幾乎不給（1%）；狀態 Y（20%）：a、b、c 都常給。"""
    px, py = [0.5, 0.49, 0.01], [0.3, 0.3, 0.4]
    arm = [0] * 2000 + [1] * 1960 + [2] * 40 + [0] * 300 + [1] * 300 + [2] * 400
    P = np.array([px] * 4000 + [py] * 1000)
    return np.array(arm), P


def test_common_range_keeps_coverage_before_adding_rare_arm():
    arm, P = overlap_leaf()
    C, S = choose_common_range(arm, P, min_n=300, overlap_min=0.05, min_share=0.8)
    assert C == (0, 1) and S.all()                                      # 加 c 只剩 Y（20%）→ 不比 c
    C, S = choose_common_range(arm, P, min_n=300, overlap_min=0.05, min_share=0.1)
    assert C == (0, 1, 2) and S.sum() == 1000                           # 門檻放寬 → 三個都比，只在 Y
    C, S = choose_common_range(arm, P, min_n=3000, overlap_min=0.05, min_share=0.8)
    assert C == () and S.all()                                          # 沒有可行的兩兩組合 → 不比較、不修剪


def test_common_range_falls_back_to_largest_coverage():
    arm, P = overlap_leaf()
    C, S = choose_common_range(arm, P, min_n=300, overlap_min=0.35, min_share=0.8)
    assert C == (0, 1) and S.sum() == 4000                              # Y 的 a、b 機率 0.3 < 0.35 → 只剩 X（80%）
    C, _ = choose_common_range(arm, P, min_n=300, overlap_min=0.35, min_share=0.9)
    assert C == (0, 1)                                                  # 都不到 90% → 選涵蓋最多的


def test_comparable_fn_recomputes_common_range_for_merged_leaves():
    P = np.array([[0.9, 0.1]] * 1000)
    df = pd.DataFrame({"葉子": ["L1"] * 500 + ["L2"] * 500, "_a": [0] * 500 + [1] * 500, "_z": np.arange(1000)})
    le = {"min_n_per_arm": 300, "overlap_min": 0.05, "min_overlap_share": 0.8, "dominant_share": 0.8}
    comparable = comparable_fn(df, P, le)
    assert comparable(["L1"]) and comparable(["L2"])                    # 單一版本 100% → 維持現狀（不合併）
    assert comparable(["L1", "L2"])                                     # 合起來兩個版本都夠、機率都 ≥ 5%
    le["dominant_share"] = 1.0
    assert not comparable_fn(df, P, le)(["L1"])


def stats_frame(rows):
    base = {"葉子": "L1", "葉子決策點": 1000, "葉子玩家數": 300, "葉子留存率": 0.9, "決策點": 400, "刪除": 0, "原始留存率": 0.9}
    out = pd.DataFrame([{**base, **r} for r in rows])
    out["樣本足夠"] = out["n"] >= 300
    out["平衡"] = out["max_smd"] <= 0.1
    out["合格"] = out["樣本足夠"] & out["平衡"]
    return out


def boot_of(values: dict[str, tuple[float, float]], reps=200, seed=0):
    rng = np.random.default_rng(seed)
    return {("L1", a): np.concatenate([[m], rng.normal(m, sd, reps)]) for a, (m, sd) in values.items()}


def test_best_arm_high_confidence_needs_balance_and_significance():
    stats = stats_frame([{"arm": "a", "n": 400, "加權留存率": 0.95, "max_smd": 0.05},
                         {"arm": "b", "n": 400, "加權留存率": 0.90, "max_smd": 0.05},
                         {"arm": "c", "n": 100, "加權留存率": 0.99, "max_smd": 0.05}])   # c 樣本不足，不當候選
    best = best_arms(stats, boot_of({"a": (0.95, 0.002), "b": (0.90, 0.002), "c": (0.99, 0.002)}), ["L1"])
    r = best.iloc[0]
    assert (r["最高留存 arm"], r["次高 arm"], r["信心"]) == ("a", "b", "高") and r["− 次高"] == pytest.approx(0.05)


def test_best_arm_unbalanced_or_single_is_low_confidence():
    unbalanced = stats_frame([{"arm": "a", "n": 400, "加權留存率": 0.95, "max_smd": 0.8},
                              {"arm": "b", "n": 400, "加權留存率": 0.90, "max_smd": 0.05}])
    r = best_arms(unbalanced, boot_of({"a": (0.95, 0.002), "b": (0.90, 0.002)}), ["L1"]).iloc[0]
    assert r["信心"] == "低" and "未通過平衡" in r["判讀"]
    single = stats_frame([{"arm": "a", "n": 400, "加權留存率": 0.95, "max_smd": 0.05},
                          {"arm": "b", "n": 50, "加權留存率": 0.99, "max_smd": 0.05}])
    r = best_arms(single, boot_of({"a": (0.95, 0.002), "b": (0.99, 0.002)}), ["L1"]).iloc[0]
    assert (r["最高留存 arm"], r["信心"]) == ("a", "低") and "只有一個" in r["判讀"]
    none = stats_frame([{"arm": "a", "n": 50, "加權留存率": 0.95, "max_smd": 0.05}])
    assert best_arms(none, boot_of({"a": (0.95, 0.002)}), ["L1"]).iloc[0]["最高留存 arm"] == "—"


def test_design_matrix_keeps_rare_binary_features():
    n = 2000
    x = np.zeros(n)
    x[:10] = 1                                                          # 0.5% 為 1（如 SY_isBuyFeature）
    df = pd.DataFrame({"GP_flag_rare": x, "GP_cont": np.arange(n, dtype=float), "TAG_x": ["a", "b"] * (n // 2)})
    Z, names = design_matrix(df, ["GP_flag_rare", "GP_cont"], ["TAG_x"])
    assert "GP_flag_rare" in names and Z[:, names.index("GP_flag_rare")].std() > 0



# ---------------------------------------------------------------------------
# 合併無法比較的葉子
# ---------------------------------------------------------------------------
def by_counts(n, min_n, every=None, dominant_share=None):
    """測試用的 comparable：合起來後 n ≥ min_n 的 arm ≥ 2，或最多的 arm 佔 every 超過 dominant_share。"""
    def comparable(leaves):
        c = sum((Counter(n.get(l, {})) for l in leaves), Counter())
        if sum(v >= min_n for v in c.values()) >= 2:
            return True
        e = sum((Counter(every.get(l, {})) for l in leaves), Counter()) if every else Counter()
        return bool(e and dominant_share is not None and max(e.values()) / sum(e.values()) > dominant_share)
    return comparable


#        0
#      /   \
#     1     4
#    / \   / \
#   2   3 5   6        葉子 L1–L4 = 節點 2、3、5、6
NODES = [{"id": 0, "left": 1, "right": 4}, {"id": 1, "left": 2, "right": 3}, {"id": 2, "left": -1, "right": -1, "leaf": "t/L1"},
         {"id": 3, "left": -1, "right": -1, "leaf": "t/L2"}, {"id": 4, "left": 5, "right": 6},
         {"id": 5, "left": -1, "right": -1, "leaf": "t/L3"}, {"id": 6, "left": -1, "right": -1, "leaf": "t/L4"}]


def test_merge_incomparable_collapses_parent_of_incomparable_leaf():
    n = {"t/L1": {"a": 500, "b": 400}, "t/L2": {"a": 500}, "t/L3": {"a": 400, "b": 300}, "t/L4": {"a": 900, "b": 350}}
    collapsed, owner, names = merge_incomparable(NODES, by_counts(n, 300))
    assert collapsed == {1}                                                # L2 只有一個版本足夠 → 與 L1 合併
    assert owner == {"t/L1": 1, "t/L2": 1, "t/L3": 5, "t/L4": 6}
    assert names == {1: "t/L1–L2", 5: "t/L3", 6: "t/L4"}


def test_merge_incomparable_repeats_up_to_root_and_keeps_comparable_trees():
    n = {"t/L1": {"a": 100}, "t/L2": {"a": 100}, "t/L3": {"a": 200}, "t/L4": {"b": 100}}
    collapsed, owner, names = merge_incomparable(NODES, by_counts(n, 300))
    assert 0 in collapsed and set(owner.values()) == {0} and names == {0: "t/L1–L4"}   # 合到根仍只有 a 足夠
    ok = {leaf: {"a": 400, "b": 400} for leaf in ("t/L1", "t/L2", "t/L3", "t/L4")}
    assert merge_incomparable(NODES, by_counts(ok, 300))[0] == set()


# ---------------------------------------------------------------------------
# 9.1b 葉子內校準
# ---------------------------------------------------------------------------
def test_balance_weights_align_means_within_tolerance():
    from feel_engine.evaluation.leaf_eval import balance_weights

    rng = np.random.default_rng(0)
    X = rng.normal(0.6, 1.0, size=(2000, 3)) - 0.0           # 該版本平均比葉子高 0.6 個標準差（目標 = 0）
    w = balance_weights(X, np.ones(len(X)), tol=0.05)
    assert abs(w.sum() - 1) < 1e-9
    assert np.abs((w[:, None] * X).sum(axis=0)).max() <= 0.05 + 1e-3


def test_calibration_improves_leaf_balance_and_keeps_unalignable_arm_unbalanced():
    from feel_engine.evaluation.leaf_eval import calibrate_leaf_weights

    rng = np.random.default_rng(1)
    n = 3000
    x = np.concatenate([rng.normal(0.5, 1, n), rng.normal(-0.5, 1, n), rng.normal(5, 0.1, 400)])   # c：完全不重疊
    arm = ["a"] * n + ["b"] * n + ["c"] * 400
    df = pd.DataFrame({"葉子": "L1", "arm": arm, "_a": ["abc".index(a) for a in arm],
                       "member_account": [f"p{i}" for i in range(len(x))], "churnNext20": 0, "w_cap": 1.0, "W": 1.0,
                       "共同範圍": True, "比較": True, "_z": np.arange(len(x))})
    Z, M, info = x[:, None], np.zeros((len(x), 3)), {"L1": {"比較": (0, 1, 2), "可比較比例": 1.0}}
    before = leaf_arm_stats(df, Z, M, info, smd_max=0.2).set_index("arm")["max_smd"]
    after = leaf_arm_stats(df.assign(W=calibrate_leaf_weights(df, Z, tol=0.05)), Z, M, info, smd_max=0.2).set_index("arm")
    assert after.loc["a", "max_smd"] < 0.1 < before.loc["a"] and after.loc["a", "平衡"]
    assert not after.loc["c", "平衡"]                                   # 對不齊 → 照實判未平衡
    assert after.loc["a", "有效樣本"] < n                                # 權重不均 → 有效樣本變少



def test_dominant_version_leaf_is_keep_current_not_incomparable():
    stats = stats_frame([{"arm": "a", "n": 900, "決策點": 900, "加權留存率": 0.95, "max_smd": 0.05},
                         {"arm": "b", "n": 100, "決策點": 100, "加權留存率": 0.90, "max_smd": 0.05}])
    boot = boot_of({"a": (0.95, 0.002), "b": (0.90, 0.002)})
    r = best_arms(stats, boot, ["L1"], dominant_share=0.8).iloc[0]
    assert (r["信心"], r["最多版本"], round(r["最多版本佔比"], 2)) == ("維持現狀", "a", 0.9)
    assert best_arms(stats, boot, ["L1"], dominant_share=0.9).iloc[0]["信心"] == "低"   # 要「超過」門檻：剛好 90% 不算


def test_keep_current_leaf_is_not_merged():
    n = {"t/L1": {"a": 500, "b": 400}, "t/L2": {"a": 900}, "t/L3": {"a": 400, "b": 300}, "t/L4": {"a": 900, "b": 350}}
    every = {**n, "t/L2": {"a": 900, "b": 50}}                              # L2：a 佔 95%
    assert merge_incomparable(NODES, by_counts(n, 300, every, 0.8))[0] == set()
    assert merge_incomparable(NODES, by_counts(n, 300))[0] == {1}                       # 不考慮維持現狀時會合併



def test_confidence_only_needs_best_version_balanced():
    second_unbalanced = stats_frame([{"arm": "a", "n": 400, "加權留存率": 0.95, "max_smd": 0.05},
                                     {"arm": "b", "n": 400, "加權留存率": 0.90, "max_smd": 0.8}])
    r = best_arms(second_unbalanced, boot_of({"a": (0.95, 0.002), "b": (0.90, 0.002)}), ["L1"]).iloc[0]
    assert r["信心"] == "高"                                               # 次高未平衡不影響
    close = boot_of({"a": (0.95, 0.03), "b": (0.94, 0.03)})
    assert best_arms(second_unbalanced, close, ["L1"]).iloc[0]["信心"] == "中"



def test_merge_same_recommendation_collapses_siblings_and_cascades():
    rec = {2: "a", 3: "a", 5: "b", 6: "c"}                                # L1、L2 都建議 a；L3、L4 不同
    collapsed = merge_same_recommendation(NODES, set(), rec)
    assert collapsed == {1}
    owner, names = collapse_names(NODES, collapsed)
    assert names == {1: "t/L1–L2", 5: "t/L3", 6: "t/L4"}
    assert merge_same_recommendation(NODES, set(), {2: "a", 3: "a", 5: "a", 6: "a"}) == {0, 1, 4}   # 一路併到根
    assert merge_same_recommendation(NODES, set(), {2: "a", 3: None, 5: "b", 6: "b"}) == {4}        # 無法比較不併


def test_small_common_range_caps_confidence_at_low():
    stats = stats_frame([{"arm": "a", "n": 400, "加權留存率": 0.95, "max_smd": 0.05, "可比較比例": 0.6},
                         {"arm": "b", "n": 400, "加權留存率": 0.90, "max_smd": 0.05, "可比較比例": 0.6}])
    boot = boot_of({"a": (0.95, 0.002), "b": (0.90, 0.002)})
    r = best_arms(stats, boot, ["L1"], min_overlap_share=0.8).iloc[0]
    assert r["信心"] == "低" and "共同範圍只涵蓋葉子 60%" in r["判讀"]
    assert best_arms(stats, boot, ["L1"], min_overlap_share=0.5).iloc[0]["信心"] == "高"


def test_keep_current_merges_with_leaf_recommending_same_version():
    keep = {"信心": "維持現狀", "樣本足夠 arm 數": 1, "最多版本": "a", "最高留存 arm": "—"}
    confident = {"信心": "中", "樣本足夠 arm 數": 2, "最多版本": "b", "最高留存 arm": "a"}
    other = {"信心": "高", "樣本足夠 arm 數": 2, "最多版本": "b", "最高留存 arm": "b"}
    assert recommended_version(keep) == recommended_version(confident) == "a"
    rec = {2: recommended_version(keep), 3: recommended_version(confident), 5: recommended_version(other), 6: "c"}
    assert merge_same_recommendation(NODES, set(), rec) == {1}             # 維持現狀_a 和建議 a 的葉子併成一片


def test_merge_similar_recommendation_merges_siblings_without_strong_evidence():
    rec = {2: "a", 3: "b", 5: "a", 6: "a"}
    # L1（a，中）＋ L2（b，中）都不是高信心 → 併成節點 1（建議要重算，這一輪不再往上）；L3、L4 同為 a → 併成節點 4
    assert merge_same_recommendation(NODES, set(), rec, {2: False, 3: False, 5: True, 6: True}) == {1, 4}
    assert merge_same_recommendation(NODES, set(), rec, {2: True, 3: False, 5: True, 6: True}) == {4}   # 有一片是高 → 不併
    assert merge_same_recommendation(NODES, set(), rec) == {4}                                         # 不給 strong：只併相同建議


def test_merge_recommendations_merges_similar_only_when_tree_is_bushy():
    rec, weak = {2: "a", 3: "b", 5: "a", 6: "a"}, {2: False, 3: False, 5: False, 6: False}
    # 先只併相同建議：L3、L4 → 節點 4，剩 3 片（L1、L2、L3–L4）
    assert merge_recommendations(NODES, set(), rec, weak, over_leaves=3) == {4}        # 不超過 3 片 → 照原本只併相同
    assert merge_recommendations(NODES, set(), rec, weak, over_leaves=2) == {1, 4}     # 超過 2 片（龐大）→ 再併相近


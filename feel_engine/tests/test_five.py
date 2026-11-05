# -*- coding: utf-8 -*-
"""SPEC 9.4：五個體感補位（合成資料，不讀快取）。"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from feel_engine.evaluation.five import (CARVED, ELSEWHERE, LOW_RANGE, NO_RANGE, OVER_LIMIT, PLACED, PLACED_OUTSIDE, TAKEN, UNPLACED_TAKEN, augment_tree,
                                         choose_placement, compare_pair, fill_versions, mine_sub_rule, FiveResult,
                                         place_every_version,
                                         recommendations, tree_shortfall, used_versions)
from feel_engine.evaluation.propensity import Propensity
from feel_engine.html import pages


def test_used_versions_counts_keep_current_and_skips_incomparable():
    best = pd.DataFrame([
        {"葉子": "低/L1", "樣本足夠 arm 數": 2, "信心": "中", "最高留存 arm": "a_0", "最多版本": "a_2"},
        {"葉子": "低/L2", "樣本足夠 arm 數": 1, "信心": "維持現狀", "最高留存 arm": "a_5", "最多版本": "a_5"},
        {"葉子": "高/L1", "樣本足夠 arm 數": 1, "信心": "低", "最高留存 arm": "a_1", "最多版本": "a_1"},   # 無法比較
        {"葉子": "高/L2", "樣本足夠 arm 數": 3, "信心": "高", "最高留存 arm": "a_2", "最多版本": "a_0"},
    ])
    assert recommendations(best) == {"低/L1": "a_0", "低/L2": "a_5", "高/L2": "a_2"}
    assert used_versions(best) == {"a_0", "a_2", "a_5"}                  # 兩棵樹合起來算


def test_mine_sub_rule_finds_region_where_rare_version_is_given():
    rng = np.random.default_rng(0)
    n = 6000
    x, other = rng.uniform(0, 1, n), rng.uniform(0, 1, n)
    in_o = x < 0.3                                                    # 可比區：只有 x < 0.3 會發 v（1）
    arm = np.where(in_o, rng.integers(0, 2, n), 0)
    X = np.column_stack([x, other])
    conds, rows = mine_sub_rule(X, in_o, arm, v=1, r=0, w=np.ones(n), depth=2, min_n=300, seed=0)
    assert len(conds) == 1 and conds[0][:2] == (0, "le") and conds[0][2] == pytest.approx(0.3, abs=0.02)
    assert in_o[rows].mean() > 0.99 and rows.sum() >= 0.25 * n
    assert mine_sub_rule(X, in_o, np.zeros(n, dtype=int), v=1, r=0, w=np.ones(n), depth=2, min_n=300, seed=0) is None


LE = {"overlap_min": 0.05, "min_n_per_arm": 300, "smd_threshold": 0.2, "bootstrap_reps": 50,
      "balance_calibration": True, "balance_tolerance": 0.1}


def make_ps(P: np.ndarray, Z: np.ndarray, arms: list[str]) -> Propensity:
    """結果模型 = 0 → 加權留存率就是 IPW（傾向分數給對時不偏）。"""
    return Propensity(table=pd.DataFrame(), Z=Z, names=[f"z{i}" for i in range(Z.shape[1])], arms=arms, P=P,
                      M=np.zeros_like(P), covariates=[])


def test_compare_pair_recovers_known_gap():
    rng = np.random.default_rng(1)
    n = 20000                                                         # 每個版本約 1 萬筆：差距的標準誤約 0.4 pp
    a = rng.integers(0, 2, n)                                         # 可比區內 v（1）與 r（0）各半
    kept = rng.uniform(size=n) < np.where(a == 1, 0.90, 0.93)         # 真實留存：v 90%、r 93% → 差距 3 pp
    rows = pd.DataFrame({"member_account": [f"p{i // 4}" for i in range(n)], "arm": np.where(a == 1, "a_1", "a_0"),
                         "churnNext20": (~kept).astype(int), "w_cap": 1.0, "_a": a, "_z": np.arange(n)})
    ps = make_ps(np.full((n, 2), 0.5), rng.normal(size=(n, 1)), ["a_0", "a_1"])
    res = compare_pair(rows, np.ones(n, dtype=bool), ps, LE, v=1, r=0, name="t/L1·_1", seed=0)
    assert res["gap_pp"] == pytest.approx(3.0, abs=1.5)
    assert res["gap_ci"][0] < res["gap_pp"] < res["gap_ci"][1]
    assert res["retention_v"] == pytest.approx(0.90, abs=0.015) and res["retention_base"] == pytest.approx(0.93, abs=0.015)
    assert res["comparable_share"] == 1.0 and res["balanced"] and res["ess"] > 5000


def cand(version: str, leaf: str, gap: float, share: float = 0.01, cs: float = 1.0) -> dict:
    return {"version": version, "leaf": leaf, "base_version": "a_0", "gap_pp": gap, "impact_pp": share * gap,
            "comparable_share": cs}


def test_choose_placement_minimizes_total_loss_one_per_leaf():
    cands = [cand("a_1", "L1", 1.0, 0.02), cand("a_1", "L2", 1.5, 0.01),
             cand("a_3", "L2", 0.5, 0.01), cand("a_3", "L3", 1.9, 0.05)]
    placed, unplaced = choose_placement(cands, ["a_1", "a_3"], 2.0, 0.5)
    assert [(p["version"], p["leaf"]) for p in placed] == [("a_1", "L1"), ("a_3", "L2")] and unplaced == []
    assert [c["status"] for c in cands] == [PLACED, ELSEWHERE, PLACED, ELSEWHERE]   # 總損失 0.025 最小


def test_choose_placement_reports_conflict_limit_and_no_range():
    cands = [cand("a_1", "L1", 0.5), cand("a_3", "L1", 0.8),             # 搶同一片葉子
             cand("a_4", "L2", 3.4), cand("a_4", "L3", 2.6),             # 都超過上限
             {"version": "a_5", "leaf": "L1", "base_version": "a_0", "gap_pp": None}]   # 沒有可比區
    placed, unplaced = choose_placement(cands, ["a_1", "a_3", "a_4", "a_5"], 2.0, 0.5)
    assert [(p["version"], p["leaf"]) for p in placed] == [("a_1", "L1")]
    reasons = {u["version"]: u["reason"] for u in unplaced}
    assert reasons == {"a_3": UNPLACED_TAKEN, "a_4": "最小差距 2.6 pp（L3）", "a_5": NO_RANGE}
    assert [c["status"] for c in cands] == [PLACED, TAKEN, OVER_LIMIT, OVER_LIMIT, NO_RANGE]


def test_choose_placement_prefers_version_that_is_better():
    cands = [cand("a_1", "L1", 0.3, 0.01), cand("a_1", "L2", -0.5, 0.10)]   # L2：v 反而較好 → 整體影響為負
    placed, _ = choose_placement(cands, ["a_1"], 2.0, 0.5)
    assert placed[0]["leaf"] == "L2"


def test_choose_placement_skips_low_comparable_share():
    cands = [cand("a_3", "L1", -4.6, 0.09, cs=0.01),                  # 看起來好很多，但子規則內只有 1% 可比
             cand("a_3", "L2", 0.3, 0.01, cs=0.7),
             cand("a_1", "L3", 0.5, 0.01, cs=0.2)]
    placed, unplaced = choose_placement(cands, ["a_3", "a_1"], 2.0, 0.5)
    assert [(p["version"], p["leaf"]) for p in placed] == [("a_3", "L2")]
    assert unplaced == [{"version": "a_1", "reason": "可比較比例都不到 50%（最高 20.0%）"}]
    assert [c["status"] for c in cands] == [LOW_RANGE, PLACED, LOW_RANGE]


def experiment_frame(seed: int = 2):
    """兩片葉子、三個版本：t/L1 建議 a_0，只有 x < 0.3 時 a_0／a_1 各半（a_1 低 1 pp）；t/L2 建議 a_2、從不發 a_1。"""
    rng = np.random.default_rng(seed)
    n = 20000                                                         # 每片葉子；可比區內每個版本約 3,000 筆
    x = rng.uniform(0, 1, 2 * n)
    leaf = np.repeat(["t/L1", "t/L2"], n)
    P = np.zeros((2 * n, 3))
    in_l1 = leaf == "t/L1"
    region = in_l1 & (x < 0.3)
    P[region] = [0.5, 0.5, 0.0]
    P[in_l1 & ~region] = [1.0, 0.0, 0.0]
    P[~in_l1] = [0.0, 0.0, 1.0]
    a = np.where(region, rng.integers(0, 2, 2 * n), np.where(in_l1, 0, 2))   # 依 P 分派
    kept = rng.uniform(size=2 * n) < np.array([0.93, 0.92, 0.97])[a]
    df = pd.DataFrame({"葉子": leaf, "member_account": [f"p{i // 4}" for i in range(2 * n)],
                       "arm": np.array(["a_0", "a_1", "a_2"])[a], "churnNext20": (~kept).astype(int), "w_cap": 1.0,
                       "_a": a, "_z": np.arange(2 * n)})
    best = pd.DataFrame([
        {"葉子": "t/L1", "樣本足夠 arm 數": 2, "信心": "中", "最高留存 arm": "a_0", "最多版本": "a_0"},
        {"葉子": "t/L2", "樣本足夠 arm 數": 1, "信心": "維持現狀", "最高留存 arm": "a_2", "最多版本": "a_2"},
    ])
    rules = {"features": ["GP_x"], "labels": {"GP_x": "測試特徵"}, "missing": {"fill": -1, "add_isna": False},
             "params": {"categories": {}}}
    ps = make_ps(P, rng.normal(size=(2 * n, 1)), ["a_0", "a_1", "a_2"])
    return df, pd.DataFrame({"GP_x": x}), best, rules, ps


def test_fill_versions_places_rare_version_in_its_comparable_region():
    df, feat, best, rules, ps = experiment_frame()
    res = fill_versions(df, feat, np.arange(len(feat)), best, rules, ps, LE, {"max_loss_pp": 2.0, "sub_rule_depth": 2},
                        seed=0)
    assert res.used == ["a_0", "a_2"] and res.missing == ["a_1"]
    (p,) = res.placed
    assert (p["leaf"], p["base_version"], p["tree"]) == ("t/L1", "a_0", "t")
    assert len(p["conditions"]) == 1 and p["conditions"][0].startswith("測試特徵 ≤ ")
    assert float(p["conditions"][0].split("≤ ")[1]) == pytest.approx(0.3, abs=0.02)
    assert p["gap_pp"] == pytest.approx(1.0, abs=2.0) and p["comparable_share"] > 0.99
    assert p["share"] == pytest.approx(0.15, abs=0.02) and p["impact_pp"] == pytest.approx(p["share"] * p["gap_pp"])
    assert {c["leaf"]: c["status"] for c in res.candidates} == {"t/L1": PLACED, "t/L2": NO_RANGE}
    out = res.to_json(2.0)
    assert out["used"] == ["a_0", "a_2"] and out["unplaced"] == [] and out["placed"][0]["conditions"] == p["conditions"]


def test_fill_versions_skips_when_every_version_is_used():
    df, feat, best, rules, ps = experiment_frame()
    best = pd.concat([best, pd.DataFrame([{"葉子": "t/L3", "樣本足夠 arm 數": 2, "信心": "低", "最高留存 arm": "a_1",
                                           "最多版本": "a_1"}])])
    res = fill_versions(df, feat, np.arange(len(feat)), best, rules, ps, LE, {"max_loss_pp": 2.0}, seed=0)
    assert res.missing == [] and res.placed == [] and res.unplaced == [] and res.candidates == []


def test_fill_versions_uses_feat_pos_and_z_mapping():
    """ps 列順序、特徵表多出的列、df 列序與索引都不同時，結果要與恆等對應一致（防 feat.iloc[idx] 之類的退化）。"""
    df, feat, best, rules, ps = experiment_frame()
    ff = {"max_loss_pp": 2.0, "sub_rule_depth": 2}
    base = fill_versions(df, feat, np.arange(len(feat)), best, rules, ps, LE, ff, seed=0)
    perm = np.random.default_rng(3).permutation(len(df))
    ps2 = make_ps(ps.P[perm], ps.Z[perm], ps.arms)
    inv = np.empty_like(perm)
    inv[perm] = np.arange(len(perm))
    df2 = df.assign(_z=inv[df["_z"].to_numpy()])
    feat2 = pd.concat([pd.DataFrame({"GP_x": np.full(5000, 99.0)}), feat], ignore_index=True)
    feat_pos2 = 5000 + perm                                           # 新 ps 列 j → feat2 第 5000 + perm[j] 列
    df2 = df2.sample(frac=1, random_state=4).set_axis(np.arange(len(df2)) * 7 + 100)
    res = fill_versions(df2, feat2, feat_pos2, best, rules, ps2, LE, ff, seed=0)
    (p0,), (p1,) = base.placed, res.placed
    assert (p1["leaf"], p1["conditions"], p1["decision_points"]) == (p0["leaf"], p0["conditions"], p0["decision_points"])
    assert p1["share"] == pytest.approx(p0["share"], abs=1e-9)
    assert p1["gap_pp"] == pytest.approx(p0["gap_pp"], abs=1e-9)      # 只比點估計；bootstrap 的抽樣順序依列序
    json.dumps(res.to_json(2.0), allow_nan=False, ensure_ascii=False)


TREE = [{"id": 0, "left": 1, "right": 2, "n": 1000, "churn": 0.1, "text": ["今日已玩手數", "≤ 35？"],
         "left_label": "是", "right_label": "否"},
        {"id": 1, "left": -1, "right": -1, "n": 400, "churn": 0.2, "leaf": "t/L1"},
        {"id": 2, "left": -1, "right": -1, "n": 600, "churn": 0.05, "leaf": "t/L2"}]
INFO = {1: {"name": "L1", "best": "_0", "color": "#0072B2", "conf": "中", "retention": 0.93},
        2: {"name": "L2", "best": "_2", "color": "#009E73", "conf": "高", "retention": 0.97}}
PLACE = {"version": "a_1", "leaf": "t/L1", "base_version": "a_0", "conditions": ["餘額÷下注額 ≤ 40", "今日已玩手數 ≤ 20"],
         "decision_points": 120, "leaf_decision_points": 400, "retention_v": 0.92, "gap_pp": 1.0}


def test_augment_tree_adds_sub_rule_question_and_fill_leaf():
    nodes, collapsed, info = augment_tree(TREE, set(), {1: "t/L1", 2: "t/L2"}, INFO, [PLACE], {"a_1": "#E69F00"})
    svg = pages.tree_svg(nodes, info, collapsed)
    for text in ("餘額÷下注額 ≤ 40", "且 今日已玩手數 ≤ 20？", "L1 補位", "補位 _1 · 92.0%", "比 _0 低 1.0 pp",
                 "驗證 120 筆", "L1 其餘", "驗證 280 筆", "最佳 _0 · 93.0%", "最佳 _2"):
        assert text in svg
    assert TREE[1]["left"] == -1 and 1 in INFO                       # 不改動傳入的節點與 info


def test_augment_tree_whole_leaf_and_collapsed_leaf():
    _, _, info = augment_tree(TREE, set(), {1: "t/L1", 2: "t/L2"}, INFO, [dict(PLACE, conditions=[])], {"a_1": "#E69F00"})
    assert info[1]["conf"] == pages.FILL and info[1]["name"] == "L1 補位"   # 子規則 = 整片葉子：直接改標
    merged = {0: {"name": "L1–L2", "best": "_0", "color": "#0072B2", "conf": "低", "retention": 0.93}}
    nodes, collapsed, info = augment_tree(TREE, {0}, {0: "t/L1–L2"}, merged,
                                          [dict(PLACE, leaf="t/L1–L2", leaf_decision_points=1000)], {"a_1": "#E69F00"})
    svg = pages.tree_svg(nodes, info, collapsed)
    assert collapsed == set() and "≤ 35？" not in svg and "L1–L2 補位" in svg and "驗證 880 筆" in svg


def tcand(version: str, tree: str, leaf: str, gap: float, share: float = 0.01) -> dict:
    return {**cand(version, f"{tree}/{leaf}", gap, share), "tree": tree}


def test_choose_placement_two_per_version_in_different_trees():
    cands = [tcand("a_3", "新手", "L1", 0.1), tcand("a_3", "新手", "L2", 0.2),   # 同一棵樹的兩片只能取一片
             tcand("a_3", "活躍", "L1", 0.5), tcand("a_1", "活躍", "L1", 0.3),   # 活躍/L1 兩個版本搶
             tcand("a_1", "一般", "L1", 0.4)]
    placed, short = choose_placement(cands, ["a_3", "a_1"], 2.0, 0.5, per_version=2)
    # 片數相同（共 3 片）時挑整體損失小的：活躍/L1 給 a_1（0.003）比給 a_3（0.005）好
    assert [(p["version"], p["leaf"]) for p in placed] == [("a_3", "新手/L1"), ("a_1", "活躍/L1"), ("a_1", "一般/L1")]
    assert short == [{"version": "a_3", "reason": UNPLACED_TAKEN, "placed_n": 1}]
    assert [c["status"] for c in cands] == [PLACED, ELSEWHERE, TAKEN, PLACED, PLACED]


def test_fill_versions_asks_two_placements_when_many_trees():
    df, feat, best, rules, ps = experiment_frame()
    rules = {**rules, "trees": {k: {} for k in "abcd"}}                  # 4 棵樹 > many_trees 3
    ff = {"max_loss_pp": 2.0, "many_trees": 3, "placements_many_trees": 2}
    res = fill_versions(df, feat, np.arange(len(feat)), best, rules, ps, LE, ff, seed=0)
    assert res.per_version == 2 and [p["leaf"] for p in res.placed] == ["t/L1"]
    assert res.unplaced == [{"version": "a_1", "reason": NO_RANGE, "placed_n": 1}]   # 其他樹沒有可比區
    assert res.to_json(2.0)["placements_per_version"] == 2


def test_choose_placement_min_per_tree_covers_every_tree_first():
    cands = [tcand("a_3", "新手", "L1", 0.1), tcand("a_3", "回流", "L1", 0.2), tcand("a_3", "活躍", "L1", 0.3),
             tcand("a_1", "新手", "L2", 0.05),
             tcand("a_1", "一般", "L1", 2.6)]                                  # 超過上限：一般樹補不到
    placed, short = choose_placement(cands, ["a_3", "a_1"], 2.0, 0.5, per_version=2, min_per_tree=1)
    # 三棵樹各一片；新手樹給 a_1（a_1 也需要補）比再多補一片 a_3 好；片數不超過需要
    assert [(p["version"], p["leaf"]) for p in placed] == [("a_3", "回流/L1"), ("a_3", "活躍/L1"), ("a_1", "新手/L2")]
    assert short == [{"version": "a_1", "reason": "最小差距 2.6 pp（一般/L1）", "placed_n": 1}]
    unfilled = tree_shortfall(cands, placed, ["新手", "回流", "活躍", "一般", "下滑"], 1, 2.0, 0.5)
    assert unfilled == [{"tree": "一般", "reason": "最小差距 2.6 pp（_1，一般/L1）"}, {"tree": "下滑", "reason": NO_RANGE}]


def test_choose_placement_min_per_tree_allows_more_than_per_version():
    cands = [tcand("a_3", t, "L1", 0.1) for t in ("新手", "回流", "活躍")]
    placed, short = choose_placement(cands, ["a_3"], 2.0, 0.5, per_version=2, min_per_tree=1)
    assert len(placed) == 3 and short == []                                 # 只剩 a_3 可補時，每棵樹都補 a_3
    placed, _ = choose_placement([dict(c) for c in cands], ["a_3"], 2.0, 0.5, per_version=2)
    assert len(placed) == 2                                                 # 不要求每棵樹時維持每個版本 2 片


def test_fill_versions_reports_unfilled_trees_in_json():
    df, feat, best, rules, ps = experiment_frame()
    rules = {**rules, "trees": {"t": {}, "u": {}}}                          # u 樹沒有葉子 → 沒有可比區
    ff = {"max_loss_pp": 2.0, "min_per_tree": 1}
    res = fill_versions(df, feat, np.arange(len(feat)), best, rules, ps, LE, ff, seed=0)
    assert [p["leaf"] for p in res.placed] == ["t/L1"] and res.unfilled_trees == [{"tree": "u", "reason": NO_RANGE}]
    js = res.to_json(2.0)
    assert js["min_per_tree"] == 1 and js["unfilled_trees"] == res.unfilled_trees
    assert "min_per_tree" not in fill_versions(df, feat, np.arange(len(feat)), best, {**rules, "trees": {"t": {}}}, ps, LE,
                                               {"max_loss_pp": 2.0}, seed=0).to_json(2.0)


def test_choose_placement_force_covers_every_tree_and_version_with_fewest_outside():
    cands = [tcand("a_3", "新手", "L1", 0.1), tcand("a_3", "回流", "L1", 0.2),
             {**tcand("a_3", "活躍", "L1", -0.1), "comparable_share": 0.49},     # 可比較比例差一點
             tcand("a_1", "活躍", "L2", 3.0), tcand("a_1", "一般", "L1", 5.8, 0.02), tcand("a_1", "一般", "L2", 4.0, 0.01),
             {"version": "a_3", "tree": "下滑", "leaf": "下滑/L1", "base_version": "a_0", "gap_pp": None}]   # 沒有可比區
    placed, short = choose_placement(cands, ["a_3", "a_1"], 2.0, 0.5, per_version=1, min_per_tree=1, force=True)
    # 新手、回流用門檻內；活躍、一般只有門檻外的候選 → 門檻外 2 片（一般的 a_1 同時讓 a_1 補到）；
    # 同樣 2 片時挑整體影響小的：活躍 a_3（−0.001）勝 a_1（0.03），一般 L2（0.04）勝 L1（0.116）
    assert [(p["version"], p["leaf"]) for p in placed] == [
        ("a_3", "新手/L1"), ("a_3", "回流/L1"), ("a_3", "活躍/L1"), ("a_1", "一般/L2")]
    assert short == []
    assert [p.get("outside") for p in placed] == [None, None, "可比較比例 49.0% < 50%", "差距 4.0 pp > 2 pp"]
    assert [c["status"] for c in cands] == [PLACED, PLACED, PLACED_OUTSIDE, OVER_LIMIT, OVER_LIMIT, PLACED_OUTSIDE, NO_RANGE]
    unfilled = tree_shortfall(cands, placed, ["新手", "回流", "活躍", "一般", "下滑"], 1, 2.0, 0.5, force=True)
    assert unfilled == [{"tree": "下滑", "reason": NO_RANGE}]


def test_choose_placement_force_prefers_within_thresholds():
    cands = [tcand("a_1", "t", "L1", 0.5), tcand("a_1", "u", "L1", 2.5, 0.001)]   # u 的損失其實較小，但超過門檻
    placed, _ = choose_placement(cands, ["a_1"], 2.0, 0.5, per_version=1, force=True)
    assert [p["leaf"] for p in placed] == ["t/L1"] and "outside" not in placed[0]


def test_place_every_version_searches_leaf_and_order():
    """新手樹要補 a_1、a_2、a_3：a_1 只能在 L1（可比區不足）、a_3 只能在 L2，a_2 兩片都行。L1 先給 a_2（門檻內）的話
    L1 其餘就沒有 a_1 的可比區 → 要讓 a_1 拿 L1（門檻外）、a_2 補在 L2 扣掉 a_3 之後的其餘部分，三個都補到。"""
    table = {("新手/L1", (), "a_1"): (1.5, 0.3), ("新手/L1", (), "a_2"): (-7.0, 1.0), ("新手/L2", (), "a_2"): (0.1, 1.0),
             ("新手/L2", (), "a_3"): (-0.9, 1.0), ("新手/L2", ("a_3",), "a_2"): (0.2, 1.0)}  # 其他（其餘部分）→ 沒有可比區
    rows_of = {}

    def run(table):
        def mine_slot(leaf, idx, r_arm, versions, after):
            out = []
            for v in versions:
                hit = table.get((leaf, tuple(after), v))
                c = {"version": v, "tree": "新手", "leaf": leaf, "base_version": r_arm, "gap_pp": None,
                     **({"after": after} if after else {})}
                if hit is not None:
                    c.update(gap_pp=hit[0], impact_pp=0.01 * hit[0], comparable_share=hit[1])
                    rows_of[id(c)] = idx[:10]
                out.append(c)
            return out

        res = FiveResult(["a_0"], ["a_1", "a_2", "a_3"], force=True, per_tree=True, need={"新手": ["a_1", "a_2", "a_3"]})
        for leaf in rec:
            res.candidates += mine_slot(leaf, leaf_rows[leaf], "a_0", ["a_1", "a_2", "a_3"], [])
        place_every_version(res, rec, leaf_rows, rows_of, mine_slot, ["新手"], 2.0, 0.5)
        return res

    rec = {"新手/L1": "a_0", "新手/L2": "a_0"}
    leaf_rows = {"新手/L1": np.arange(100), "新手/L2": np.arange(100, 200)}
    res = run(table)
    assert [(p["version"], p["leaf"], p.get("after")) for p in res.placed] == [
        ("a_1", "新手/L1", None), ("a_2", "新手/L2", ["a_3"]), ("a_3", "新手/L2", None)]
    assert res.unplaced == [] and res.placed[0]["outside"] == "可比較比例 30.0% < 50%"
    status = {(c["version"], c["leaf"], tuple(c.get("after") or ())): c["status"] for c in res.candidates}
    assert status == {("a_1", "新手/L1", ()): PLACED_OUTSIDE, ("a_2", "新手/L1", ()): ELSEWHERE,
                      ("a_3", "新手/L1", ()): NO_RANGE, ("a_1", "新手/L2", ()): NO_RANGE, ("a_2", "新手/L2", ()): CARVED,
                      ("a_2", "新手/L2", ("a_3",)): PLACED, ("a_3", "新手/L2", ()): PLACED}
    # force 也不放寬差距：a_1 唯一的位置差 2.02 pp > max_loss_pp 2 → a_1 不補，L1 改給 a_2
    res2 = run({**table, ("新手/L1", (), "a_1"): (2.02, 1.0)})
    assert sorted((p["version"], p["leaf"]) for p in res2.placed) == [("a_2", "新手/L1"), ("a_3", "新手/L2")]
    assert res2.unplaced == [{"version": "a_1", "tree": "新手", "reason": "最小差距 2.02 pp（新手/L1） > 2 pp"}]
    assert [c["status"] for c in res2.candidates if c["version"] == "a_1" and c["leaf"] == "新手/L1"] == [OVER_LIMIT]


def test_fill_versions_every_version_per_tree_uses_tree_own_recommendations():
    df, feat, best, rules, ps = experiment_frame()
    rules = {**rules, "trees": {"t": {}, "u": {}}}                          # u 樹沒有葉子
    ff = {"max_loss_pp": 2.0, "every_version_per_tree": True, "many_trees": 1}
    res = fill_versions(df, feat, np.arange(len(feat)), best, rules, ps, LE, ff, seed=0)
    assert res.need == {"t": ["a_1"], "u": ["a_0", "a_1", "a_2"]} and res.per_version == 1
    assert [p["leaf"] for p in res.placed] == ["t/L1"]
    assert res.unplaced == [{"version": v, "tree": "u", "reason": NO_RANGE} for v in ("a_0", "a_1", "a_2")]
    js = res.to_json(2.0)
    assert js["every_version_per_tree"] and js["need"] == res.need and js["placements_per_version"] == 1


def test_fill_versions_every_version_per_tree_carves_rest_of_leaf():
    """只有一片葉子 t/L1（建議 a_0）：x < 0.3 時 a_0／a_1 各半、x > 0.7 時 a_0／a_2 各半 → 兩個版本都要補在 t/L1，
    後補的在扣掉先補子規則後的其餘部分。"""
    rng = np.random.default_rng(5)
    n = 30000
    x = rng.uniform(0, 1, n)
    P = np.tile([1.0, 0.0, 0.0], (n, 1))
    P[x < 0.3] = [0.5, 0.5, 0.0]
    P[x > 0.7] = [0.5, 0.0, 0.5]
    a = np.where(x < 0.3, rng.integers(0, 2, n), np.where(x > 0.7, 2 * rng.integers(0, 2, n), 0))
    kept = rng.uniform(size=n) < np.array([0.93, 0.92, 0.925])[a]
    df = pd.DataFrame({"葉子": "t/L1", "member_account": [f"p{i // 4}" for i in range(n)],
                       "arm": np.array(["a_0", "a_1", "a_2"])[a], "churnNext20": (~kept).astype(int), "w_cap": 1.0,
                       "_a": a, "_z": np.arange(n)})
    best = pd.DataFrame([{"葉子": "t/L1", "樣本足夠 arm 數": 3, "信心": "中", "最高留存 arm": "a_0", "最多版本": "a_0"}])
    rules = {"features": ["GP_x"], "labels": {"GP_x": "測試特徵"}, "missing": {"fill": -1, "add_isna": False},
             "params": {"categories": {}}, "trees": {"t": {}}}
    ps = make_ps(P, rng.normal(size=(n, 1)), ["a_0", "a_1", "a_2"])
    ff = {"max_loss_pp": 2.0, "every_version_per_tree": True, "force_coverage": True}
    res = fill_versions(df, pd.DataFrame({"GP_x": x}), np.arange(n), best, rules, ps, LE, ff, seed=0)
    assert res.need == {"t": ["a_1", "a_2"]} and res.unplaced == []
    first, second = sorted(res.placed, key=lambda p: len(p.get("after") or ()))
    assert {first["version"], second["version"]} == {"a_1", "a_2"} and first["leaf"] == second["leaf"] == "t/L1"
    assert "after" not in first and second["after"] == [first["version"]]
    assert second["leaf_decision_points"] == n - first["decision_points"]
    taken = [c for c in res.candidates if c["version"] == second["version"] and "after" not in c]
    assert [c["status"] for c in taken] == [CARVED]
    assert res.to_json(2.0)["placed"][[p["version"] for p in res.placed].index(second["version"])]["after"] == second["after"]


def test_augment_tree_chains_second_placement_on_rest():
    second = dict(PLACE, version="a_2", conditions=["今日已玩手數 > 50"], decision_points=100, leaf_decision_points=280,
                  after=["a_1"])
    nodes, collapsed, info = augment_tree(TREE, set(), {1: "t/L1", 2: "t/L2"}, INFO, [second, PLACE],
                                          {"a_1": "#E69F00", "a_2": "#CC79A7"})
    svg = pages.tree_svg(nodes, info, collapsed)
    for text in ("補位 _1 · 92.0%", "補位 _2 · 92.0%", "今日已玩手數 &gt; 50？", "驗證 280 筆", "驗證 100 筆", "驗證 180 筆"):
        assert text in svg
    assert sum(1 for n in nodes if n["left"] == -1) == 4                # L2、_1 補位、_2 補位、L1 其餘

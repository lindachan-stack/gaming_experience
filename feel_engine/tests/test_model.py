# -*- coding: utf-8 -*-
"""SPEC 第 8 節：決策樹（合成資料，不讀快取）。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from feel_engine.modeling.model import (
    ModelConfigError,
    feature_columns,
    fit_encoder,
    fit_split,
    fit_tree,
    load_model_config,
    node_test,
    validate_features,
    weighted_auc,
)
from feel_engine.paths import MODEL_CONFIG_PATH

MCFG = {
    "fast": ["GP_a", "NGP_b"],
    "tags": {"none": [], "x": ["TAG_x"], "y": ["TAG_y"]},
    "missing": {"fill": -1, "add_isna": True},
}
CFG = {
    "seed": 42,
    "data": {"merchant": "TCG", "currency": "PHP", "game": "22071",
             "train_start": "2026-08-27", "train_end": "2026-08-28", "valid_start": "2026-08-29", "valid_end": "2026-08-29"},
    "tree": {"max_depth": 3, "min_samples_leaf": 20},
    "splits": {"min_segment_train_n": 300},
}


# ---------------------------------------------------------------------------
# 設定檔
# ---------------------------------------------------------------------------
def test_feature_columns_are_fast_plus_named_tags_not_cumulative():
    assert feature_columns(MCFG, "none") == ["GP_a", "NGP_b"]
    assert feature_columns(MCFG, "x") == ["GP_a", "NGP_b", "TAG_x"]
    assert feature_columns(MCFG, "y") == ["GP_a", "NGP_b", "TAG_y"]          # 不累加：value 不含 lifecycle
    with pytest.raises(ModelConfigError, match="tags"):
        feature_columns(MCFG, "nope")


@pytest.mark.parametrize("col", ["churnNext20", "w_cap", "arm", "arm_ratio", "SEG_value", "spin_ts", "bucket"])
def test_leaky_or_non_feature_columns_are_rejected(col):
    with pytest.raises(ModelConfigError, match=col):
        validate_features(["GP_a", col], available=["GP_a", col])


def test_unknown_and_duplicate_columns_are_rejected():
    with pytest.raises(ModelConfigError, match="GP_nope"):
        validate_features(["GP_a", "GP_nope"], available=["GP_a"])
    with pytest.raises(ModelConfigError, match="重複"):
        validate_features(["GP_a", "GP_a"], available=["GP_a"])


def test_shipped_model_config_matches_dataset_features_and_experiments():
    """model.yaml 的快層都是 dataset 有的欄位（不含 GP_rtp）且都有中文名稱；tags 的標籤組合；實驗清單 = 使用者定義的 8 組。"""
    from feel_engine.pipeline.build import FEATURES
    mcfg = load_model_config(MODEL_CONFIG_PATH)
    fast = feature_columns(mcfg, "none")
    assert set(fast) <= set(FEATURES) and "GP_rtp" not in fast and len(set(fast)) == len(fast)
    assert feature_columns(mcfg, "lifecycle") == [*fast, "TAG_lifecycle"]
    assert feature_columns(mcfg, "value") == [*fast, "TAG_value_combined"]
    assert feature_columns(mcfg, "lifecycle_value") == [*fast, "TAG_lifecycle", "TAG_value_combined"]
    assert all(mcfg["labels"].get(c) for c in [*fast, "TAG_lifecycle", "TAG_value_combined"])   # 樹圖用的中文名稱
    assert [(e["split"], e["tags"]) for e in mcfg["experiments"]] == [
        ("all", "none"), ("all", "lifecycle"), ("all", "value"), ("all", "lifecycle_value"),
        ("lifecycle", "none"), ("lifecycle", "value"),
        ("value", "none"), ("value", "lifecycle"),
    ]


def test_config_entries_can_carry_chinese_labels_used_in_rules(tmp_path):
    (tmp_path / "m.yaml").write_text("fast:\n  - GP_a: 今日手數\n  - NGP_b\ntags:\n  none: []\n", encoding="utf-8")
    mcfg = load_model_config(tmp_path / "m.yaml")
    assert mcfg["fast"] == ["GP_a", "NGP_b"] and mcfg["labels"] == {"GP_a": "今日手數"}
    gt = fit_tree(toy_frame(), ["GP_a"], "all", fill=-1, add_isna=True, max_depth=1, min_samples_leaf=20, seed=42,
                  labels=mcfg["labels"])
    assert all(r[0].startswith("今日手數 ") for r in gt.rules.values())          # 規則用中文名稱
    assert gt.encoder.display("GP_a") == "今日手數"


def test_config_rejects_old_slow_and_unknown_experiment(tmp_path):
    (tmp_path / "old.yaml").write_text("fast: [GP_a]\nslow: {1: [TAG_lifecycle]}\n", encoding="utf-8")
    with pytest.raises(ModelConfigError, match="tags"):
        load_model_config(tmp_path / "old.yaml")
    (tmp_path / "bad.yaml").write_text("fast: [GP_a]\ntags: {none: []}\nexperiments: [{split: all, tags: lifecycle}]\n",
                                       encoding="utf-8")
    with pytest.raises(ModelConfigError, match="experiments"):
        load_model_config(tmp_path / "bad.yaml")


# ---------------------------------------------------------------------------
# 編碼：缺值、one-hot、常數欄
# ---------------------------------------------------------------------------
def test_encoder_fills_missing_adds_isna_and_one_hot():
    train = pd.DataFrame({"GP_a": [1.0, np.nan, 3.0], "TAG_x": ["高", "低", "高"], "GP_c": [5, 5, 5]})
    enc = fit_encoder(train, ["GP_a", "TAG_x", "GP_c"], fill=-1, add_isna=True)
    assert enc.columns == ["GP_a", "GP_a_isna", "TAG_x=低", "TAG_x=高"]
    assert enc.constant_removed == ["GP_c"]
    X = enc.transform(pd.DataFrame({"GP_a": [np.nan, 2.0], "TAG_x": ["高", "沒看過"], "GP_c": [5, 5]}))
    assert X.to_numpy().tolist() == [[-1, 1, 0, 1], [2, 0, 0, 0]]   # 沒看過的類別 → 全 0


def test_encoder_without_isna():
    enc = fit_encoder(pd.DataFrame({"GP_a": [1.0, np.nan]}), ["GP_a"], fill=-1, add_isna=False)
    assert enc.columns == ["GP_a"]


# ---------------------------------------------------------------------------
# 樹與葉子規則
# ---------------------------------------------------------------------------
def toy_frame(n: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 10, n)
    return pd.DataFrame({
        "GP_a": x,
        "NGP_b": rng.uniform(0, 1, n),
        "TAG_x": np.where(x > 5, "高", "低"),
        "TAG_y": "中",
        "churnNext20": (x <= 5).astype(int),          # 規則：GP_a ≤ 5 → 流失
        "w_cap": 1.0,
    })


def test_leaf_rules_are_readable_and_prefixed_by_group():
    gt = fit_tree(toy_frame(), ["GP_a"], "新手", fill=-1, add_isna=True, max_depth=1, min_samples_leaf=20, seed=42)
    assert list(gt.leaf_names.values()) == ["新手/L1", "新手/L2"]
    rules = [gt.rules[node] for node in gt.leaf_names]
    assert rules[0][0].startswith("GP_a ≤ ") and rules[1][0].startswith("GP_a > ")
    proba, leaves = gt.apply(pd.DataFrame({"GP_a": [1.0, 9.0]}))
    assert leaves.tolist() == ["新手/L1", "新手/L2"] and proba[0] > 0.9 > 0.1 > proba[1]


def test_one_hot_rules_read_as_equal_not_equal():
    gt = fit_tree(toy_frame(), ["TAG_x"], "all", fill=-1, add_isna=True, max_depth=1, min_samples_leaf=20, seed=42)
    flat = [r for node in gt.leaf_names for r in gt.rules[node]]
    assert set(flat) <= {"TAG_x ≠ 低", "TAG_x = 低", "TAG_x ≠ 高", "TAG_x = 高"} and len(flat) == 2


def test_sample_weight_changes_leaf_churn():
    df = toy_frame()
    df.loc[df["GP_a"] <= 5, "w_cap"] = 3.0                # 流失的列加權 → 根節點加權流失率上升
    flat = fit_tree(toy_frame(), ["NGP_b"], "all", fill=-1, add_isna=True, max_depth=1, min_samples_leaf=1000, seed=42)
    heavy = fit_tree(df, ["NGP_b"], "all", fill=-1, add_isna=True, max_depth=1, min_samples_leaf=1000, seed=42)
    p_flat, _ = flat.apply(df.head(1))
    p_heavy, _ = heavy.apply(df.head(1))
    assert p_heavy[0] > p_flat[0]


def test_weighted_auc():
    assert weighted_auc(np.array([0, 1]), np.array([0.1, 0.9]), np.array([1.0, 1.0])) == 1.0
    assert weighted_auc(np.array([1, 1]), np.array([0.1, 0.9]), np.array([1.0, 1.0])) is None


# ---------------------------------------------------------------------------
# 三種切分（8.1）
# ---------------------------------------------------------------------------
def split_frame() -> pd.DataFrame:
    big = toy_frame(800, 1).assign(SEG_value="高", day=np.tile(["train", "valid"], 400))
    small = toy_frame(200, 2).assign(SEG_value="低", day=np.tile(["train", "valid"], 100))
    df = pd.concat([big, small], ignore_index=True)
    df["TAG_value_combined"] = df["SEG_value"]
    df["SEG_lifecycle"] = "活躍"
    df["member_account"] = [f"p{i}" for i in range(len(df))]
    df["spin_ts"] = pd.date_range("2026-08-28", periods=len(df), freq="min", tz="America/New_York")
    return df


def days(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """（訓練日, 驗證日）。"""
    return df[df["day"] == "train"].drop(columns="day"), df[df["day"] == "valid"].drop(columns="day")


def test_split_all_trains_one_tree_on_train_day_and_predicts_valid_day():
    res = fit_split(*days(split_frame()), "all", "none", CFG, {**MCFG, "fast": ["GP_a", "NGP_b"]})
    assert list(res.trees) == ["all"]
    g = res.groups.set_index("群")
    assert g.loc["all", "訓練"] == 500 and g.loc["all", "驗證"] == 500 and g.loc["all", "策略"] == "自建樹"
    assert len(res.predictions) == 500 and res.auc_valid > 0.95
    assert res.dates == {"merchant": "TCG", "currency": "PHP", "game": "22071",
                         "train_start": "2026-08-27", "train_end": "2026-08-28", "valid_start": "2026-08-29",
                         "valid_end": "2026-08-29"}


def test_split_value_removes_own_tag_and_small_group_uses_all():
    mcfg = {**MCFG, "fast": ["GP_a"], "tags": {"value": ["TAG_value_combined"]}}
    res = fit_split(*days(split_frame()), "value", "value", CFG, mcfg)
    g = res.groups.set_index("群")
    assert g.loc["高", "策略"] == "自建樹" and g.loc["低", "策略"] == "沿用 all"     # 低：訓練日 100 < 300
    assert set(res.trees) == {"高", "all"}
    assert "TAG_value_combined" not in " ".join(res.trees["高"].encoder.columns)   # 分群欄位對應的 TAG 移除
    assert any(c.startswith("TAG_value_combined=") for c in res.trees["all"].encoder.columns)
    leaves_b = res.predictions.groupby("群")["葉子"].first()
    assert leaves_b["高"].startswith("高/L") and leaves_b["低"].startswith("all/L")


def test_config_groups_model_only_listed_groups_and_still_need_enough_samples():
    df = split_frame()
    df.loc[df["SEG_value"] == "高", "SEG_value"] = np.where(np.arange((df["SEG_value"] == "高").sum()) % 4 < 2, "高", "無紀錄")
    cfg = {**CFG, "splits": {"min_segment_train_n": 150, "groups": {"value": ["低", "高"]}}}
    res = fit_split(*days(df), "value", "none", cfg, {**MCFG, "fast": ["GP_a"]})
    g = res.groups.set_index("群")
    assert g.loc["低", "策略"] == "沿用 all"                                      # 訓練 100 < 150：指定的群也沿用 all
    assert g.loc["高", "策略"] == "自建樹" and g.loc["無紀錄", "策略"] == "不建模"   # 高 200 ≥ 150
    assert set(res.trees) == {"高", "all"}
    assert set(res.predictions["群"]) == {"低", "高"}                                       # 不建模的群不評估
    assert len(res.predictions) == int(g.loc[["低", "高"], "驗證"].sum())


def test_training_sample_threshold_also_applies_to_all_tree():
    train, valid = days(split_frame())
    with pytest.raises(ModelConfigError, match="min_segment_train_n"):
        fit_split(train, valid, "all", "none", {**CFG, "splits": {"min_segment_train_n": 10_000}}, {**MCFG, "fast": ["GP_a"]})


def test_category_whitelist_keeps_other_values_out_of_nodes():
    train = pd.DataFrame({"TAG_value_combined": ["高", "中", "低", "無紀錄", "本週無下注"]})
    enc = fit_encoder(train, ["TAG_value_combined"], fill=-1, add_isna=True, allowed={"TAG_value_combined": ["低", "中", "高"]})
    assert enc.columns == ["TAG_value_combined=低", "TAG_value_combined=中", "TAG_value_combined=高"]
    X = enc.transform(pd.DataFrame({"TAG_value_combined": ["無紀錄", "本週無下注", "高"]}))
    assert X.to_numpy().tolist() == [[0, 0, 0], [0, 0, 0], [0, 0, 1]]


def test_rules_merge_repeated_conditions_on_same_column():
    rng = np.random.default_rng(3)
    x = rng.uniform(0, 10, 3000)
    y = np.where(x <= 2, 1, np.where(x <= 5, rng.integers(0, 2, 3000), 0))
    df = pd.DataFrame({"GP_a": x, "churnNext20": y, "w_cap": 1.0})
    gt = fit_tree(df, ["GP_a"], "all", fill=-1, add_isna=True, max_depth=3, min_samples_leaf=50, seed=42)
    for node, rules in gt.rules.items():
        assert len(rules) == 1, rules                     # 每片葉子只剩一條（合併成 ≤、> 或區間）
    assert any(" < GP_a ≤ " in r[0] for r in gt.rules.values())



# ---------------------------------------------------------------------------
# 樹圖用的節點（tree_rules.json 的 nodes）
# ---------------------------------------------------------------------------
def test_tree_nodes_carry_structure_counts_and_leaf_names():
    gt = fit_tree(toy_frame(), ["GP_a"], "高", fill=-1, add_isna=False, max_depth=2, min_samples_leaf=20, seed=42)
    nodes = {n["id"]: n for n in gt.nodes}
    root = nodes[0]
    assert root["n"] == len(toy_frame()) and root["left"] != -1
    assert nodes[root["left"]]["n"] + nodes[root["right"]]["n"] == root["n"]
    assert sorted(n["leaf"] for n in gt.nodes if n["left"] == -1) == sorted(gt.leaf_names.values())
    assert root["text"][0] == "GP_a" and (root["left_label"], root["right_label"]) == ("是", "否")


def test_node_test_reads_missing_onehot_binary_and_integer_thresholds():
    train = pd.DataFrame({"GP_rate": [np.nan, 0.2, 0.5, 0.9], "GP_flag": [0, 1, 0, 1], "GP_count": [0, 3, 10, 40],
                          "TAG_v": ["低", "高", "低", "高"]})
    enc = fit_encoder(train, ["GP_rate", "GP_flag", "GP_count", "TAG_v"], fill=-1, add_isna=False,
                      labels={"GP_rate": "今日未中比例", "GP_flag": "今日久旱", "GP_count": "今日已玩手數", "TAG_v": "價值等級"})
    col = {c: i for i, c in enumerate(enc.columns)}
    assert node_test(enc, col["GP_rate"], -0.5) == {"text": ["今日未中比例", "缺值？"], "left_label": "是", "right_label": "否"}
    assert node_test(enc, col["GP_rate"], 0.7)["text"] == ["今日未中比例", "≤ 0.7（含缺值）？"]
    assert node_test(enc, col["GP_rate"], 0.35)["text"] == ["今日未中比例", "= 0.2（含缺值）？"]   # 最小與次小值之間
    assert node_test(enc, col["GP_flag"], 0.5) == {"text": ["今日久旱", "= 1？"], "left_label": "否", "right_label": "是"}
    assert node_test(enc, col["GP_count"], 35.5)["text"] == ["今日已玩手數", "≤ 35？"]      # 整數欄位：≤ 35.5 = ≤ 35
    assert node_test(enc, col["GP_count"], 1.5)["text"] == ["今日已玩手數", "= 0？"]
    assert node_test(enc, col["TAG_v=高"], 0.5) == {"text": ["價值等級", "= 高？"], "left_label": "否", "right_label": "是"}

# -*- coding: utf-8 -*-
"""
第 8 節：流失風險決策樹（訓練期間訓練、驗證期間評估；見 config.yaml 的 data.train_start／train_end／valid_start／valid_end）。

  X     = model.yaml 的 fast ＋ tags[實驗的 tags]；數值缺值 → fill（＋ <欄位>_isna），文字欄位 one-hot
  Y     = churnNext20；sample_weight = w_cap；不用 class_weight
  split = all：一棵樹
          lifecycle／value：依 SEG_lifecycle／SEG_value 分群各訓練一棵；該群對應的 TAG_ 欄位與群內常數欄移除；
          群的訓練期間樣本 < min_segment_train_n → 不訓練，沿用 all 的樹（splits.groups 指定的群也一樣）
類別合併：讀入 dataset 時套用 config.yaml 的 regroup（apply_regroup；流失併入回流），TAG_ 與 SEG_ 一起改。
實驗 = split × tags，清單與順序在 model.yaml 的 experiments。
輸出（write_split_outputs）output/{run}/model/{split}/tag_{tags}/：
  tree_rules.json（樹結構、葉子、設定）、valid_predictions.parquet（驗證期間預測）；
  給人看的 HTML 由 evaluate（每個實驗一頁）與 compare（summary.html）產生。
"""
from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yaml

from feel_engine.paths import MODEL_CONFIG_PATH

log = logging.getLogger("feel_engine.modeling.model")

PLAYER = "member_account"
LABEL, WEIGHT = "churnNext20", "w_cap"
ALLOWED_PREFIXES = ("GP_", "SY_", "NGP_", "TAG_")
SPLIT_SEGMENTS = {"all": None, "lifecycle": "SEG_lifecycle", "value": "SEG_value"}
SEG_TO_TAG = {"SEG_lifecycle": "TAG_lifecycle", "SEG_value": "TAG_value_combined"}
ALL_TREE = "all"
TRAINED, USES_ALL, NOT_MODELED = "自建樹", "沿用 all", "不建模"


class ModelConfigError(ValueError):
    """model.yaml 或 run 參數不合法。"""


def apply_regroup(df: pd.DataFrame, regroup: dict[str, dict[str, str]] | None) -> pd.DataFrame:
    """config.yaml 的 regroup：TAG_ 欄位的類別併入另一類，對應的 SEG_ 分群一起改（例如流失併入回流）。
    讀入 dataset 時套用，dataset 檔案不動；沒有設定 → 原樣回傳。"""
    if not regroup:
        return df
    tag_to_seg = {tag: seg for seg, tag in SEG_TO_TAG.items()}
    out = {}
    for col, mapping in regroup.items():
        if not col.startswith("TAG_") or col not in df.columns:
            raise ModelConfigError(f"regroup 只能指定 dataset 中的 TAG_ 欄位（分群跟著改）：{col}")
        for c in (col, tag_to_seg.get(col)):
            if c in df.columns:
                out[c] = df[c].replace(mapping)
    return df.assign(**out)


# ---------------------------------------------------------------------------
# 設定檔
# ---------------------------------------------------------------------------
def load_model_config(path: Path = MODEL_CONFIG_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        mcfg = yaml.safe_load(f) or {}
    labels: dict[str, str] = {}
    mcfg["fast"] = _names(mcfg.get("fast"), labels)
    if "slow" in mcfg:
        raise ModelConfigError("model.yaml 的 slow（累加 stage）已改為 tags（具名的 TAG_ 組合）＋ experiments")
    mcfg["tags"] = {str(k): _names(v, labels) for k, v in (mcfg.get("tags") or {"none": []}).items()}
    mcfg["labels"] = labels
    mcfg["experiments"] = [{"split": str(e["split"]), "tags": str(e["tags"])} for e in mcfg.get("experiments") or []]
    for e in mcfg["experiments"]:
        if e["split"] not in SPLIT_SEGMENTS or e["tags"] not in mcfg["tags"]:
            raise ModelConfigError(f"model.yaml experiments 有不認得的組合：{e}（split 只能是 {list(SPLIT_SEGMENTS)}，"
                                   f"tags 只能是 {list(mcfg['tags'])}）")
    mcfg["missing"] = {"fill": -1, "add_isna": True, **(mcfg.get("missing") or {})}
    mcfg["categories"] = {k: [str(c) for c in v] for k, v in (mcfg.get("categories") or {}).items()}
    return mcfg


def _names(items, labels: dict[str, str]) -> list[str]:
    """model.yaml 的欄位清單：每項可寫「欄位」或「欄位: 中文名稱」；中文名稱收進 labels（樹圖與規則用）。"""
    names = []
    for item in items or []:
        if isinstance(item, dict):
            for col, label in item.items():
                names.append(str(col))
                if label:
                    labels[str(col)] = str(label)
        else:
            names.append(str(item))
    return names


def feature_columns(mcfg: dict, tags: str) -> list[str]:
    """實驗的 X 欄位 = fast ＋ tags[tags]（依設定檔順序）。"""
    if tags not in mcfg["tags"]:
        raise ModelConfigError(f"tags {tags} 不在 model.yaml 的 tags：{list(mcfg['tags'])}")
    return [*mcfg["fast"], *mcfg["tags"][tags]]


def validate_features(cols: list[str], available: Iterable[str]) -> None:
    """只允許 dataset 中 GP_／SY_／NGP_／TAG_ 開頭的欄位，不得重複。"""
    available = set(available)
    dup = sorted({c for c in cols if cols.count(c) > 1})
    if dup:
        raise ModelConfigError(f"model.yaml 欄位重複：{dup}")
    bad_prefix = [c for c in cols if not c.startswith(ALLOWED_PREFIXES)]
    if bad_prefix:
        raise ModelConfigError(f"只能放 GP_／SY_／NGP_／TAG_ 開頭的欄位，不得放入：{bad_prefix}")
    unknown = [c for c in cols if c not in available]
    if unknown:
        raise ModelConfigError(f"dataset 沒有這些欄位：{unknown}")


# ---------------------------------------------------------------------------
# 編碼：缺值、one-hot、常數欄
# ---------------------------------------------------------------------------
@dataclass
class Encoder:
    order: list[str]                                  # 原欄位（設定檔順序）
    categories: dict[str, list[str]]                  # 文字欄位 → 訓練資料中的類別
    isna: list[str]                                   # 加 _isna 的數值欄位
    fill: float
    columns: list[str]                                # X 欄位（已移除常數欄）
    constant_removed: list[str]                       # 整個被移除的原欄位
    kinds: dict[str, tuple[str, str, str]]            # X 欄位 → (numeric／isna／onehot, 原欄位, 類別)
    labels: dict[str, str] = field(default_factory=dict)   # 原欄位 → 中文名稱（model.yaml）
    observed: dict[str, dict] = field(default_factory=dict)  # 數值原欄位 → 訓練資料非缺值的最小、次小值、是否整數／0-1、有無缺值

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return _expand(df, self.order, self.categories, self.isna, self.fill)[self.columns]

    def label(self, src: str) -> str:
        return self.labels.get(src, src)

    def display(self, col: str) -> str:
        """X 欄位的中文顯示名稱。"""
        kind, src, cat = self.kinds[col]
        return {"onehot": f"{self.label(src)}={cat}", "isna": f"{self.label(src)} 缺值"}.get(kind, self.label(src))


def _expand(df: pd.DataFrame, order: list[str], categories: dict[str, list[str]], isna: list[str],
            fill: float) -> pd.DataFrame:
    parts = {}
    for c in order:
        if c in categories:
            values = df[c].astype(object).to_numpy()
            for cat in categories[c]:
                parts[f"{c}={cat}"] = (values == cat).astype(float)
        else:
            x = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
            parts[c] = np.where(np.isnan(x), fill, x)
            if c in isna:
                parts[f"{c}_isna"] = np.isnan(x).astype(float)
    return pd.DataFrame(parts, index=df.index)


def fit_encoder(train: pd.DataFrame, cols: list[str], fill: float, add_isna: bool,
                allowed: dict[str, list[str]] | None = None, labels: dict[str, str] | None = None) -> Encoder:
    """數值欄位：缺值填 fill（訓練資料有缺值才加 _isna）；文字欄位：依訓練資料的類別 one-hot；移除訓練資料中為常數的欄。

    allowed：文字欄位 → 只對這些類別做 one-hot（model.yaml 的 categories）；其他類別全為 0，不會單獨成為分支條件。
    """
    allowed = allowed or {}
    numeric = [c for c in cols if pd.api.types.is_numeric_dtype(train[c]) or pd.api.types.is_bool_dtype(train[c])]
    categories = {c: list(allowed[c]) if c in allowed else sorted(train[c].dropna().astype(str).unique().tolist())
                  for c in cols if c not in numeric}
    isna = [c for c in numeric if add_isna and bool(train[c].isna().any())]
    full = _expand(train, cols, categories, isna, fill)
    kinds = {}
    for c in cols:
        if c in categories:
            kinds.update({f"{c}={cat}": ("onehot", c, cat) for cat in categories[c]})
        else:
            kinds[c] = ("numeric", c, "")
            if c in isna:
                kinds[f"{c}_isna"] = ("isna", c, "")
    constant = {c for c in full.columns if full[c].nunique() <= 1}
    columns = [c for c in full.columns if c not in constant]
    kept_sources = {kinds[c][1] for c in columns}
    removed = [c for c in cols if c not in kept_sources]
    observed = {}
    for c in numeric:
        x = pd.to_numeric(train[c], errors="coerce").to_numpy(dtype=float)
        v = np.unique(x[~np.isnan(x)])
        observed[c] = {"min": float(v[0]) if len(v) else None, "min2": float(v[1]) if len(v) > 1 else None,
                       "has_missing": bool(np.isnan(x).any()),
                       "integer": bool(len(v) and np.all(v == np.round(v))), "binary": bool(len(v) and np.isin(v, (0, 1)).all())}
    return Encoder(list(cols), categories, isna, fill, columns, removed, kinds, dict(labels or {}), observed)


# ---------------------------------------------------------------------------
# 單棵樹
# ---------------------------------------------------------------------------
def weighted_auc(y: np.ndarray, p: np.ndarray, w: np.ndarray) -> float | None:
    """加權 AUC；只有單一類別時回傳 None。"""
    from sklearn.metrics import roc_auc_score          # 延後載入：只讀設定檔（如 test.ipynb）時不需要 scikit-learn

    if len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, p, sample_weight=w))


def _churn_proba(tree: DecisionTreeClassifier, X: np.ndarray) -> np.ndarray:
    proba = tree.predict_proba(X)
    classes = list(tree.classes_)
    return proba[:, classes.index(1)] if 1 in classes else np.zeros(len(X))


def _leaf_paths(tree: DecisionTreeClassifier) -> dict[int, list[tuple[int, str, float]]]:
    """葉節點 → 從根到葉的條件 (特徵序號, le／gt, 門檻)；依由左到右的順序。"""
    t = tree.tree_
    out: dict[int, list[tuple[int, str, float]]] = {}

    def walk(node: int, conds: list[tuple[int, str, float]]) -> None:
        if t.children_left[node] == -1:
            out[node] = conds
            return
        f, thr = int(t.feature[node]), float(t.threshold[node])
        walk(t.children_left[node], [*conds, (f, "le", thr)])
        walk(t.children_right[node], [*conds, (f, "gt", thr)])

    walk(0, [])
    return out


def _render(enc: Encoder, f: int, op: str, thr: float) -> str:
    kind, src, cat = enc.kinds[enc.columns[f]]
    name = enc.label(src)
    if kind == "onehot":
        return f"{name} ≠ {cat}" if op == "le" else f"{name} = {cat}"
    if kind == "isna":
        return f"{name} 非缺值" if op == "le" else f"{name} 缺值"
    return f"{name} ≤ {thr:.6g}" if op == "le" else f"{name} > {thr:.6g}"


def _render_path(enc: Encoder, conds: list[tuple[int, str, float]]) -> list[str]:
    """同一欄位同方向只留最嚴格的門檻；數值欄位兩個方向都有時寫成區間「a < x ≤ b」。"""
    tight: dict[tuple[int, str], float] = {}
    for f, op, thr in conds:
        key = (f, op)
        tight[key] = thr if key not in tight else (min(tight[key], thr) if op == "le" else max(tight[key], thr))
    out = []
    for f in dict.fromkeys(f for f, _, _ in conds):
        lo, hi = tight.get((f, "gt")), tight.get((f, "le"))
        if lo is not None and hi is not None and enc.kinds[enc.columns[f]][0] == "numeric":
            out.append(f"{lo:.6g} < {enc.label(enc.kinds[enc.columns[f]][1])} ≤ {hi:.6g}")
        else:
            out += [_render(enc, f, op, tight[(f, op)]) for op in ("le", "gt") if (f, op) in tight]
    return out


def _threshold_text(info: dict, thr: float) -> str:
    """整數欄位：x ≤ 35.5 等同 x ≤ 35；其他取 4 位有效數字。"""
    return f"{int(np.floor(thr)):,}" if info.get("integer") else f"{thr:.4g}"


def node_test(enc: Encoder, f: int, thr: float) -> dict:
    """內部節點的提問（樹圖用，最多兩行）與左右分支標籤；sklearn 的左分支 = X 欄位 ≤ 門檻。

    one-hot、_isna、0／1 欄位改問「= 類別？」「缺值？」「= 1？」（左分支 = 否）；缺值填 fill 且 fill 小於所有觀測值時，
    門檻落在 fill 與最小觀測值之間 = 問「缺值？」（左 = 是），否則左分支註明「含缺值」；
    門檻落在最小與次小觀測值之間 = 問「= 最小值？」（例如比例 ≤ 0.0007 其實是 = 0）。
    """
    kind, src, cat = enc.kinds[enc.columns[f]]
    name = enc.label(src)
    if kind == "onehot":
        return {"text": [name, f"= {cat}？"], "left_label": "否", "right_label": "是"}
    if kind == "isna":
        return {"text": [name, "缺值？"], "left_label": "否", "right_label": "是"}
    info = enc.observed.get(src, {})
    missing_below = info.get("has_missing") and info.get("min") is not None and enc.fill < info["min"]
    if missing_below and thr < info["min"]:
        return {"text": [name, "缺值？"], "left_label": "是", "right_label": "否"}
    if info.get("binary") and not missing_below:
        return {"text": [name, "= 1？"], "left_label": "否", "right_label": "是"}
    if info.get("min2") is not None and info["min"] <= thr < info["min2"]:
        return {"text": [name, f"= {info['min']:,.4g}{'（含缺值）' if missing_below else ''}？"],
                "left_label": "是", "right_label": "否"}
    return {"text": [name, f"≤ {_threshold_text(info, thr)}{'（含缺值）' if missing_below else ''}？"],
            "left_label": "是", "right_label": "否"}


@dataclass
class GroupTree:
    name: str
    encoder: Encoder
    tree: DecisionTreeClassifier
    leaf_names: dict[int, str]                        # 葉節點 → 「樹/L1」，由左到右
    rules: dict[int, list[str]]                       # 葉節點 → 條件
    train_leaves: pd.DataFrame                        # 訓練期間每片葉子的決策點數、加權流失率
    nodes: list[dict] = field(default_factory=list)  # 每個節點：左右子節點、提問、訓練期間 n 與加權流失率（樹圖用）

    def apply(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """回傳（預測流失機率, 葉子名稱）。"""
        X = self.encoder.transform(df).to_numpy(dtype=float)
        leaves = pd.Series(self.tree.apply(X)).map(self.leaf_names).to_numpy()
        return _churn_proba(self.tree, X), leaves


def _tree_nodes(tree: DecisionTreeClassifier, enc: Encoder, X: np.ndarray, y: np.ndarray, w: np.ndarray,
                leaf_names: dict[int, str]) -> list[dict]:
    t = tree.tree_
    path = tree.decision_path(X)                      # n × 節點，經過 = 1
    n = np.asarray(path.sum(axis=0)).ravel()
    sw, swy = path.T @ w, path.T @ (w * y)
    nodes = []
    for i in range(t.node_count):
        left, right = int(t.children_left[i]), int(t.children_right[i])
        node = {"id": i, "left": left, "right": right, "n": int(n[i]), "w": float(sw[i]),
                "churn": float(swy[i] / sw[i]) if sw[i] > 0 else None}
        if left == -1:
            node["leaf"] = leaf_names[i]
        else:
            node.update(node_test(enc, int(t.feature[i]), float(t.threshold[i])))
        nodes.append(node)
    return nodes


def fit_tree(train: pd.DataFrame, cols: list[str], name: str, *, fill: float, add_isna: bool,
             max_depth: int, min_samples_leaf: int, seed: int, min_impurity_decrease: float = 0.0,
             allowed: dict[str, list[str]] | None = None, labels: dict[str, str] | None = None) -> GroupTree:
    from sklearn.tree import DecisionTreeClassifier

    enc = fit_encoder(train, cols, fill, add_isna, allowed, labels)
    if not enc.columns:
        raise ModelConfigError(f"樹 {name}：所有欄位在訓練資料中都是常數，沒有可用的 X")
    X = enc.transform(train).to_numpy(dtype=float)
    y = train[LABEL].to_numpy(dtype=int)
    w = train[WEIGHT].to_numpy(dtype=float)
    tree = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf,
                                  min_impurity_decrease=min_impurity_decrease, random_state=seed)
    tree.fit(X, y, sample_weight=w)
    paths = _leaf_paths(tree)
    leaf_names = {node: f"{name}/L{i + 1}" for i, node in enumerate(paths)}
    rules = {node: _render_path(enc, conds) for node, conds in paths.items()}
    stats = (pd.DataFrame({"葉子": pd.Series(tree.apply(X)).map(leaf_names).to_numpy(), "w": w, "wy": w * y})
             .groupby("葉子").agg(n=("w", "size"), w=("w", "sum"), wy=("wy", "sum")))
    stats["churn_w"] = stats["wy"] / stats["w"]
    return GroupTree(name, enc, tree, leaf_names, rules, stats, _tree_nodes(tree, enc, X, y, w, leaf_names))


# ---------------------------------------------------------------------------
# 三種切分（8.1）
# ---------------------------------------------------------------------------
@dataclass
class SplitResult:
    split: str
    tags: str                                         # model.yaml 的 tags 名稱
    features: list[str]                               # 設定檔解析出的 X 原欄位
    params: dict
    dates: dict                                       # 商戶、幣別、遊戲、train_start、train_end、valid_start、valid_end
    groups: pd.DataFrame                              # 群、策略、樹、訓練、驗證、驗證加權流失率、驗證 AUC
    trees: dict[str, GroupTree]
    predictions: pd.DataFrame                         # 驗證期間：群、樹、葉子、p、y、w
    leaves: pd.DataFrame
    auc_valid: float | None


def fit_split(train: pd.DataFrame, valid: pd.DataFrame, split: str, tags: str, cfg: dict, mcfg: dict) -> SplitResult:
    """train：訓練期間 dataset（訓練決策樹）；valid：驗證期間 dataset（預測、AUC）。"""
    if split not in SPLIT_SEGMENTS:
        raise ModelConfigError(f"split 只能是 {list(SPLIT_SEGMENTS)}，收到 {split}")
    if "min_leaf_share" in cfg["tree"]:
        raise ModelConfigError("config.yaml 的 tree.min_leaf_share 已拿掉：每片葉子只看 tree.min_samples_leaf")
    if "segment_min_samples_leaf" in cfg["splits"]:
        raise ModelConfigError("config.yaml 的 splits.segment_min_samples_leaf 已拿掉：整體與分群樹都用 tree.min_samples_leaf")
    cols = feature_columns(mcfg, tags)
    validate_features(cols, train.columns)
    seg = SPLIT_SEGMENTS[split]
    tcfg, scfg = cfg["tree"], cfg["splits"]
    params = {
        "max_depth": int(tcfg["max_depth"]), "min_samples_leaf": int(tcfg["min_samples_leaf"]),
        "min_impurity_decrease": float(tcfg.get("min_impurity_decrease", 0.0)),
        "min_segment_train_n": int(scfg["min_segment_train_n"]),
        "groups": [str(g) for g in (scfg.get("groups") or {}).get(split) or []],
        "redundant_tag": SEG_TO_TAG[seg] if seg and SEG_TO_TAG[seg] in cols else None,   # 分群用的標籤又放進 X
        "categories": mcfg.get("categories") or {},
        "fill": mcfg["missing"]["fill"], "add_isna": bool(mcfg["missing"]["add_isna"]), "seed": int(cfg["seed"]),
    }
    dates = {k: str(cfg["data"][k]) for k in ("merchant", "currency", "game", "train_start", "train_end", "valid_start", "valid_end")}

    def fit(rows: pd.DataFrame, use_cols: list[str], name: str, leaf_min: int) -> GroupTree:
        return fit_tree(rows, use_cols, name, fill=params["fill"], add_isna=params["add_isna"],
                        max_depth=params["max_depth"], min_samples_leaf=leaf_min, seed=params["seed"],
                        min_impurity_decrease=params["min_impurity_decrease"],
                        allowed=params["categories"], labels=mcfg.get("labels") or {})

    trees: dict[str, GroupTree] = {}
    if len(train) < params["min_segment_train_n"]:                 # 訓練樣本門檻：all 樹也適用
        raise ModelConfigError(f"訓練期間只有 {len(train):,} 筆決策點，少於 min_segment_train_n {params['min_segment_train_n']:,}")
    if seg is None:
        group_v = pd.Series(ALL_TREE, index=valid.index)
        plan = [{"群": ALL_TREE, "策略": TRAINED, "原因": "", "樹": ALL_TREE, "訓練": len(train), "驗證": len(valid)}]
        trees[ALL_TREE] = fit(train, cols, ALL_TREE, params["min_samples_leaf"])
    else:
        group_v = valid[seg].astype(str)
        seg_cols = [c for c in cols if c != SEG_TO_TAG[seg]]     # 群內為常數，不放入 X
        n_t = train[seg].astype(str).value_counts()
        n_v = group_v.value_counts()
        chosen, min_n = params["groups"], params["min_segment_train_n"]
        absent = [g for g in chosen if n_t.get(g, 0) == 0]
        if absent:
            raise ModelConfigError(f"config splits.groups.{split} 的群在訓練期間沒有資料：{absent}")
        others = sorted((set(n_t.index) | set(n_v.index)) - set(chosen), key=lambda v: (-n_t.get(v, 0), v))
        plan = []
        for v in [*chosen, *others]:
            if chosen and v not in chosen:                # 有指定群：其他群不建模、不評估
                strategy, reason, tree_name = NOT_MODELED, f"不在 splits.groups.{split}", "—"
            elif n_t.get(v, 0) < min_n:                    # 訓練樣本不足（指定的群也一樣）→ 沿用 all
                strategy, reason, tree_name = USES_ALL, f"訓練 < {min_n:,}", ALL_TREE
            else:
                strategy, reason, tree_name = TRAINED, "", v
            if strategy == TRAINED:
                trees[v] = fit(train[train[seg].astype(str) == v], seg_cols, v, params["min_samples_leaf"])
            plan.append({"群": v, "策略": strategy, "原因": reason, "樹": tree_name,
                         "訓練": int(n_t.get(v, 0)), "驗證": int(n_v.get(v, 0))})
        if any(r["策略"] == USES_ALL for r in plan):
            trees[ALL_TREE] = fit(train, cols, ALL_TREE, params["min_samples_leaf"])

    # 驗證期間預測：每一群用自己的樹或 all 的樹；不建模的群沒有預測
    tree_of = {r["群"]: r["樹"] for r in plan if r["策略"] != NOT_MODELED}
    parts = []
    for name, gt in trees.items():
        mask = group_v.map(tree_of) == name
        if not mask.any():                                # 這棵樹的群在驗證期間沒有決策點：沒有可預測的列
            continue
        rows = valid[mask]
        p, leaf = gt.apply(rows)
        parts.append(pd.DataFrame({PLAYER: rows[PLAYER].to_numpy(), "spin_ts": rows["spin_ts"].to_numpy(),
                                   "群": group_v[mask].to_numpy(), "樹": name, "葉子": leaf, "p": p,
                                   "y": rows[LABEL].to_numpy(dtype=int), "w": rows[WEIGHT].to_numpy(dtype=float)},
                                  index=rows.index))
    if not parts:
        raise ModelConfigError("驗證期間沒有任何屬於建模群的決策點，無法評估")
    pred = pd.concat(parts).sort_index()
    pred_wy = pred["w"] * pred["y"]

    groups = pd.DataFrame(plan)
    by_group = pred.assign(wy=pred_wy).groupby("群")
    groups["驗證加權流失率"] = groups["群"].map(by_group["wy"].sum() / by_group["w"].sum())
    groups["驗證 AUC"] = groups["群"].map({g: weighted_auc(d["y"].to_numpy(), d["p"].to_numpy(), d["w"].to_numpy())
                                         for g, d in by_group})

    leaf_v = pred.assign(wy=pred_wy).groupby("葉子").agg(n=("w", "size"), w=("w", "sum"), wy=("wy", "sum"))
    leaves = pd.DataFrame([
        {"樹": name, "葉子": gt.leaf_names[node], "規則": " 且 ".join(gt.rules[node]) or "（全部）",
         "訓練決策點": int(gt.train_leaves.loc[gt.leaf_names[node], "n"]),
         "訓練加權流失率": float(gt.train_leaves.loc[gt.leaf_names[node], "churn_w"])}
        for name, gt in trees.items() for node in gt.leaf_names
    ])
    leaves["驗證決策點"] = leaves["葉子"].map(leaf_v["n"]).fillna(0).astype(int)
    leaves["驗證加權流失率"] = leaves["葉子"].map(leaf_v["wy"] / leaf_v["w"])

    return SplitResult(split, tags, cols, params, dates, groups, trees, pred, leaves,
                       weighted_auc(pred["y"].to_numpy(), pred["p"].to_numpy(), pred["w"].to_numpy()))


# ---------------------------------------------------------------------------
# 輸出
# ---------------------------------------------------------------------------
def _jsonable(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError(type(o))


def write_split_outputs(res: SplitResult, model_dir: Path, mcfg: dict, pred_path: Path,
                        config_path: Path | None = None, dataset_signature: str | None = None,
                        regroup: dict | None = None) -> Path:
    """model_dir/tree_rules.json（樹結構、葉子、設定）與 pred_path（驗證期間預測，供 evaluate、compare 使用）。

    tree_rules.json 另記本次用的 model.yaml 路徑、內容指紋（sha256 前 8 碼）、執行時間，以及 run 當時的 dataset 簽章
    （store.dataset_signature）與 config.yaml 的 regroup；evaluate 據此判斷 dataset 或類別合併之後有沒有改過。
    """
    model_dir.mkdir(parents=True, exist_ok=True)
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    res.predictions.assign(split=res.split, tags=res.tags).to_parquet(pred_path, index=False)
    rules = {
        "split": res.split, "tags": res.tags, **res.dates, "auc_valid": res.auc_valid,
        "run_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_config_file": str(config_path) if config_path else None,
        "model_config_sha": hashlib.sha256(Path(config_path).read_bytes()).hexdigest()[:8] if config_path else None,
        "dataset_signature": dataset_signature,
        "regroup": regroup or {},
        "features": res.features, "labels": {c: mcfg["labels"][c] for c in res.features if c in mcfg.get("labels", {})},
        "params": res.params, "missing": mcfg["missing"],
        "groups": res.groups.to_dict(orient="records"),
        "trees": {
            name: {
                "X_columns": gt.encoder.columns,
                "constant_removed": gt.encoder.constant_removed,
                "leaves": res.leaves[res.leaves["樹"] == name].to_dict(orient="records"),
                "nodes": gt.nodes,
            }
            for name, gt in res.trees.items()
        },
    }
    path = model_dir / "tree_rules.json"
    path.write_text(json.dumps(rules, ensure_ascii=False, indent=1, default=_jsonable), encoding="utf-8")
    return path

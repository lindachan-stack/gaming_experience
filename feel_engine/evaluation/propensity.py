# -*- coding: utf-8 -*-
"""
第 9.1 節：傾向分數與結果模型（只用驗證期間；evaluate 主流程配適一次、所有實驗共用）。

  9.1 傾向分數（所有實驗共用、只配適一次）：
      p(arm | x)，x = 快層 ＋ 所有 TAG_ 標籤（含分群用的標籤）；依玩家 5 折交叉配適；sample_weight = w_cap。
      模型（config leaf_eval.propensity_model）：logistic = 多項式邏輯迴歸（預設，用 design_matrix 的共變數）；
      gbm = 梯度提升樹，從資料自動學版本分派的門檻與交互作用（數值保留原值與缺值、文字 one-hot）。
      2026-10-12 實測 gbm 的權重較極端、有效樣本較少，配合 9.1b 校準後信心不如 logistic，所以預設 logistic。
      結果模型（同樣只配適一次、依玩家交叉配適）：每個 arm 一個邏輯迴歸，用狀態 x 預測「拿到該 arm 時」的留存率 m_a(x)。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

PLAYER = "member_account"
KEY = [PLAYER, "spin_ts"]
ARM, LABEL, WEIGHT = "arm", "churnNext20", "w_cap"
N_FOLDS = 5
CLIP = (0.01, 0.99)
PROPENSITY_MODELS = ("logistic", "gbm")
GBM_PARAMS = dict(max_iter=300, learning_rate=0.1, max_leaf_nodes=31, min_samples_leaf=100, l2_regularization=1.0,
                  early_stopping=True, validation_fraction=0.1, n_iter_no_change=10)


# ---------------------------------------------------------------------------
# 9.1 傾向分數
# ---------------------------------------------------------------------------
def design_matrix(df: pd.DataFrame, num_cols: list[str], cat_cols: list[str]) -> tuple[np.ndarray, list[str]]:
    """SMD 平衡檢查（與 logistic 傾向分數）用的共變數：數值 → 中位數補值（＋_isna）、截斷頭尾 1%、非負取 log1p、標準化；文字 → one-hot。

    0／1 欄位（旗標、SY_isBuyFeature）不截斷、不取 log：它們多半很稀少，截斷頭尾 1% 會變成常數而被移除。
    """
    parts, names = [], []
    for c in num_cols:
        x = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
        na = np.isnan(x)
        if na.all():
            continue
        if na.any():
            parts.append(na.astype(float))
            names.append(f"{c}_isna")
            x = np.where(na, np.nanmedian(x), x)
        if not np.isin(x, (0.0, 1.0)).all():
            lo, hi = np.quantile(x, CLIP)
            x = np.clip(x, lo, hi)
            if x.min() >= 0:
                x = np.log1p(x)
        sd = x.std()
        parts.append((x - x.mean()) / sd if sd > 0 else np.zeros_like(x))
        names.append(c)
    for c in cat_cols:
        v = df[c].astype(str).to_numpy()
        for cat in sorted(set(v)):
            parts.append((v == cat).astype(float))
            names.append(f"{c}={cat}")
    Z = np.column_stack(parts)
    keep = Z.std(axis=0) > 0
    return Z[:, keep], [n for n, k in zip(names, keep) if k]


def raw_matrix(df: pd.DataFrame, num_cols: list[str], cat_cols: list[str]) -> np.ndarray:
    """梯度提升樹用：數值保留原值（不截斷、不取 log，門檻可能落在極端值，例如高下注額）與缺值（模型自己處理），文字 one-hot。"""
    parts = [pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float) for c in num_cols]
    for c in cat_cols:
        v = df[c].astype(str).to_numpy()
        parts += [(v == cat).astype(float) for cat in sorted(set(v))]
    return np.column_stack(parts)


def fit_propensity(X: np.ndarray, arm_code: np.ndarray, n_arms: int, w_cap: np.ndarray, groups: np.ndarray,
                   seed: int, model: str = "logistic") -> np.ndarray:
    """依玩家分 N_FOLDS 折交叉配適（每筆的機率來自沒看過這位玩家的模型），回傳每筆對每個 arm 的機率（n × n_arms）。"""
    if model not in PROPENSITY_MODELS:
        raise ValueError(f"leaf_eval.propensity_model 只能是 {PROPENSITY_MODELS}，收到 {model}")
    P = np.zeros((len(X), n_arms))
    for train, test in GroupKFold(n_splits=N_FOLDS).split(X, arm_code, groups):
        m = (HistGradientBoostingClassifier(**GBM_PARAMS, random_state=seed) if model == "gbm"
             else LogisticRegression(max_iter=2000, random_state=seed))
        m.fit(X[train], arm_code[train], sample_weight=w_cap[train])
        P[np.ix_(test, m.classes_)] = m.predict_proba(X[test])
    return P


def fit_outcome(Z: np.ndarray, retained: np.ndarray, arm_code: np.ndarray, n_arms: int, w_cap: np.ndarray,
                groups: np.ndarray, seed: int) -> np.ndarray:
    """結果模型：依玩家分 N_FOLDS 折交叉配適，每個 arm 一個邏輯迴歸（只用拿到該 arm 的決策點），
    回傳每筆「若拿到各 arm」的預測留存率（n × n_arms）。某折某 arm 只有一種結果時用加權平均。"""
    M = np.zeros((len(Z), n_arms))
    for train, test in GroupKFold(n_splits=N_FOLDS).split(Z, arm_code, groups):
        for a in range(n_arms):
            rows = train[arm_code[train] == a]
            y, w = retained[rows], w_cap[rows]
            if len(rows) == 0:
                M[test, a] = np.average(retained[train], weights=w_cap[train])
            elif len(np.unique(y)) < 2:
                M[test, a] = float(y[0])
            else:
                m = LogisticRegression(max_iter=2000, random_state=seed).fit(Z[rows], y, sample_weight=w)
                M[test, a] = m.predict_proba(Z[test])[:, list(m.classes_).index(1)]
    return M


@dataclass
class Propensity:
    table: pd.DataFrame                  # KEY、arm、y、w_cap、_a（arm 編號，對齊 arms）（驗證期間）
    Z: np.ndarray                        # 共變數（列對齊 table）
    names: list[str]
    arms: list[str]
    P: np.ndarray                        # 傾向分數 p(arm | x)（n × arms）
    M: np.ndarray                        # 結果模型 m_arm(x) = 預測留存率（n × arms）
    covariates: list[str]
    model: str = "logistic"
    rows: np.ndarray | None = None       # table 第 i 列 = 輸入 V 的第 rows[i] 列（位置；補位找子規則的特徵用）


def build_propensity(V: pd.DataFrame, num_cols: list[str], cat_cols: list[str], cfg: dict) -> Propensity:
    rows = np.flatnonzero(V[ARM].notna().to_numpy())             # arm 不明（R4）不評估
    B = V.iloc[rows]
    Z, names = design_matrix(B, num_cols, cat_cols)
    arms = sorted(B[ARM].astype(str).unique())
    code = pd.Categorical(B[ARM].astype(str), categories=arms).codes.astype(int)
    w_cap = B[WEIGHT].to_numpy(dtype=float)
    model = str(cfg["leaf_eval"].get("propensity_model", "logistic"))
    X = raw_matrix(B, num_cols, cat_cols) if model == "gbm" else Z
    groups = pd.factorize(B[PLAYER])[0]
    P = fit_propensity(X, code, len(arms), w_cap, groups, int(cfg["seed"]), model)
    retained = 1.0 - B[LABEL].to_numpy(dtype=float)
    M = fit_outcome(Z, retained, code, len(arms), w_cap, groups, int(cfg["seed"]))
    table = B[[*KEY, ARM, LABEL, WEIGHT]].copy()
    table[ARM] = table[ARM].astype(str)
    table["_a"] = code
    return Propensity(table.reset_index(drop=True), Z, names, arms, P, M, [*num_cols, *cat_cols], model, rows)

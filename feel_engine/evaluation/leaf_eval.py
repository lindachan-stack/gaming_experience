# -*- coding: utf-8 -*-
"""
第 9 節：每片葉子 × arm 的評估（共同範圍、葉子內校準、加權留存率、bootstrap、最高留存 arm）。

  9.1a 共同範圍（2026-10-15 起，取代 propensity_min）：每片葉子挑要比較的 arm 組合 C，共同範圍 S = 葉子內
      「C 的每個 arm 機率都 ≥ overlap_min」的決策點，C 的每個 arm 在 S 內都要 ≥ min_n_per_arm 筆。
      盡量多比幾個 arm，但 S 要涵蓋葉子 ≥ min_overlap_share；沒有組合做得到 → 選涵蓋最多的（信心最多「低」）。
      現行規則幾乎只給某版本的狀態（機率極低的那一邊）不拿來比，避免把「誰拿到」當成「版本效果」。
  9.1b 葉子內校準（leaf_eval.balance_calibration）：S 內每個比較 arm，以 W = w_cap / p(實際 arm) 為起點調整權重，
      讓該 arm 各共變數的加權平均與 S 相差 ≤ balance_tolerance 個標準差（近似 entropy balancing）；比 W 更平衡才採用。
      對不齊時平衡檢查照實判未平衡。只用資料，不用遊戲端規則。
  9.2 每片葉子 × arm：加權留存率 = 雙重穩健（DR）估計
        = S 內 m_a 的 w_cap 加權平均 ＋ S 內實際拿到 a 者「y − m_a」的校準權重平均（y = 1 − churnNext20）；
      max_smd（對 S）、有效樣本。最高留存 arm = 比較 arm 中加權留存率最高者；與次高、與臨時基準（n 最大的 arm）比較，
      差距用玩家層級 Poisson(1) bootstrap（先聚合成「玩家 × 葉子 × arm」再以稀疏矩陣運算）。
"""
from __future__ import annotations

import warnings
from collections.abc import Callable
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import minimize

from feel_engine.evaluation.propensity import ARM, LABEL, PLAYER, WEIGHT, Propensity

CALIBRATION_MIN_ROWS = 20                   # 葉子 × arm 保留樣本少於此數不校準
CALIBRATION_BOUND = 20.0                    # 對偶變數上限（避免對不齊時數值溢位）


# ---------------------------------------------------------------------------
# 9.1a 共同範圍
# ---------------------------------------------------------------------------
def choose_common_range(arm_code: np.ndarray, P: np.ndarray, min_n: int, overlap_min: float,
                        min_share: float) -> tuple[tuple[int, ...], np.ndarray]:
    """一片葉子：挑要比較的 arm 組合 C 與共同範圍 S（C 的每個 arm 機率都 ≥ overlap_min 的決策點）。

    arm_code：葉子內每筆的 arm 編號；P：每筆對每個 arm 的傾向分數。C 的每個 arm 在 S 內都要 ≥ min_n 筆。
    S 涵蓋葉子 ≥ min_share 的組合中挑 arm 最多的（同數量挑涵蓋最多）；都不到 → 挑涵蓋最多的（同涵蓋挑 arm 多）。
    沒有任何兩個 arm 的組合可行 → C = ()、S = 整片葉子（不比較，不修剪）。
    """
    n_arms = P.shape[1]
    counts = np.bincount(arm_code, minlength=n_arms)
    base = [a for a in range(n_arms) if counts[a] >= min_n]
    ok = P >= overlap_min
    best, best_key = ((), np.ones(len(arm_code), dtype=bool)), None
    for k in range(len(base), 1, -1):
        for C in combinations(base, k):
            S = ok[:, list(C)].all(axis=1)
            if (np.bincount(arm_code[S], minlength=n_arms)[list(C)] < min_n).any():
                continue
            share = float(S.mean())
            key = (1, k, share) if share >= min_share else (0, share, k)
            if best_key is None or key > best_key:
                best, best_key = (C, S), key
    return best


def common_range(df: pd.DataFrame, P: np.ndarray, le: dict) -> tuple[pd.DataFrame, dict[str, dict]]:
    """每片葉子挑共同範圍。df 需有 葉子、_a（arm 編號）、_z（P 的列號）、w_cap。

    回傳（df 加上 共同範圍、比較（在共同範圍內且 arm ∈ C）、W（比較列 = w_cap ÷ p(實際 arm)，其餘 0）,
    {葉子: {"比較": C 的 arm 編號, "可比較比例": 共同範圍佔葉子的比例（C 為空時 NaN）}}）。
    """
    min_n, lo, share = int(le["min_n_per_arm"]), float(le["overlap_min"]), float(le["min_overlap_share"])
    a, z = df["_a"].to_numpy(), df["_z"].to_numpy()
    in_s, cmp_ = np.zeros(len(df), dtype=bool), np.zeros(len(df), dtype=bool)
    info = {}
    for leaf, idx in df.groupby("葉子", sort=False).indices.items():
        C, S = choose_common_range(a[idx], P[z[idx]], min_n, lo, share)
        in_s[idx] = S
        cmp_[idx] = S & np.isin(a[idx], C)
        info[leaf] = {"比較": C, "可比較比例": float(S.mean()) if C else np.nan}
    p = np.where(cmp_, P[z, a], 1.0)
    W = np.where(cmp_, df[WEIGHT].to_numpy(dtype=float) / p, 0.0)
    return df.assign(共同範圍=in_s, 比較=cmp_, W=W), info


def comparable_fn(df: pd.DataFrame, P: np.ndarray, le: dict) -> Callable[[list[str]], bool]:
    """merge_incomparable 用：一組原葉子合在一起、重挑共同範圍後，比較 arm ≥ 2，或是維持現狀（不合併）。"""
    idx_of = df.groupby("葉子", sort=False).indices
    a, z = df["_a"].to_numpy(), df["_z"].to_numpy()
    min_n, lo, share = int(le["min_n_per_arm"]), float(le["overlap_min"]), float(le["min_overlap_share"])
    dominant = float(le.get("dominant_share", 0.8))
    memo: dict[tuple[str, ...], bool] = {}

    def comparable(leaves: list[str]) -> bool:
        key = tuple(leaves)
        if key not in memo:
            parts = [idx_of[l] for l in leaves if l in idx_of]
            if not parts:
                memo[key] = False
            else:
                idx = np.concatenate(parts)
                C, _ = choose_common_range(a[idx], P[z[idx]], min_n, lo, share)
                counts = np.bincount(a[idx])
                memo[key] = len(C) >= 2 or counts.max() / counts.sum() > dominant
        return memo[key]

    return comparable


# ---------------------------------------------------------------------------
# 9.2 葉子 × arm
# ---------------------------------------------------------------------------
def leaf_arm_stats(df: pd.DataFrame, Z: np.ndarray, M: np.ndarray, info: dict[str, dict],
                   smd_max: float) -> pd.DataFrame:
    """df：common_range（與校準）後的驗證期間列（葉子、arm、_a、_z、y、w_cap、共同範圍、比較、W）。
    回傳每片葉子 × arm 的統計：n、範圍內玩家數 = 該 arm 在共同範圍內的筆數、玩家數；樣本足夠 = 納入比較（arm ∈ C，C 的每個 arm
    在共同範圍內都 ≥ min_n_per_arm）；加權留存率（DR）、max_smd（對共同範圍）、有效樣本只算比較 arm。"""
    rows = []
    for leaf, g in df.groupby("葉子", sort=False):
        C = set(info[leaf]["比較"])
        s = g[g["共同範圍"]]
        zs, vs = Z[s["_z"].to_numpy()], s[WEIGHT].to_numpy(dtype=float)
        mu = (vs[:, None] * zs).sum(axis=0) / vs.sum()
        sd = np.sqrt((vs[:, None] * (zs - mu) ** 2).sum(axis=0) / vs.sum())
        ok = sd > 1e-9
        wl, yl = g[WEIGHT].to_numpy(), g[LABEL].to_numpy(dtype=float)
        for arm, ga in g.groupby(ARM, sort=True):
            code = int(ga["_a"].iloc[0])
            wa, ya = ga[WEIGHT].to_numpy(), ga[LABEL].to_numpy(dtype=float)
            n = int(ga["共同範圍"].sum())
            row = {"葉子": leaf, "arm": arm, "決策點": len(ga), "玩家數": int(ga[PLAYER].nunique()),
                   "n": n, "範圍內玩家數": int(ga.loc[ga["共同範圍"], PLAYER].nunique()), "範圍外": len(ga) - n,
                   "原始留存率": 1 - float((wa * ya).sum() / wa.sum()),
                   "葉子留存率": 1 - float((wl * yl).sum() / wl.sum()), "葉子決策點": len(g),
                   "葉子玩家數": int(g[PLAYER].nunique()), "可比較比例": info[leaf]["可比較比例"],
                   "加權留存率": np.nan, "max_smd": np.nan, "有效樣本": np.nan}
            if code in C:
                k = ga[ga["比較"]]
                zk, W, y = k["_z"].to_numpy(), k["W"].to_numpy(), 1 - k[LABEL].to_numpy(dtype=float)
                regression = (vs * M[s["_z"].to_numpy(), code]).sum() / vs.sum()
                row["加權留存率"] = float(regression + (W * (y - M[zk, code])).sum() / W.sum())
                mu_a = (W[:, None] * Z[zk]).sum(axis=0) / W.sum()
                row["有效樣本"] = float(W.sum() ** 2 / (W ** 2).sum())        # Kish：權重越不均，越少
                row["max_smd"] = float(np.max(np.abs(mu_a - mu)[ok] / sd[ok])) if ok.any() else 0.0
            row["樣本足夠"] = code in C
            row["平衡"] = bool(row["max_smd"] <= smd_max)
            row["合格"] = row["樣本足夠"] and row["平衡"]                    # SPEC 9.2 eligible
            rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_retention(df: pd.DataFrame, M: np.ndarray, info: dict[str, dict], arms: list[str], n_reps: int,
                        seed: int) -> dict[tuple[str, str], np.ndarray]:
    """每個（葉子, 比較 arm）的 DR 加權留存率：第 0 欄 = 點估計，其餘 = 玩家層級 Poisson(1) 重抽（權重 W 固定）。"""
    s = df[df["共同範圍"]]
    player, players = pd.factorize(s[PLAYER])
    z, a, cmp_ = s["_z"].to_numpy(), s["_a"].to_numpy(), s["比較"].to_numpy()
    v, W, y = s[WEIGHT].to_numpy(dtype=float), s["W"].to_numpy(dtype=float), 1 - s[LABEL].to_numpy(dtype=float)
    cells, reg, cor = [], [], []
    for leaf, idx in s.groupby("葉子", sort=False).indices.items():
        for code in info[leaf]["比較"]:
            c = len(cells)
            cells.append((leaf, arms[code]))
            reg.append((np.full(len(idx), c), player[idx], v[idx], v[idx] * M[z[idx], code]))
            k = idx[cmp_[idx] & (a[idx] == code)]
            cor.append((np.full(len(k), c), player[k], W[k], W[k] * (y[k] - M[z[k], code])))
    if not cells:
        return {}

    def agg(parts: list[tuple], j: int) -> sparse.csr_matrix:
        cell, who, val = (np.concatenate([p[i] for p in parts]) for i in (0, 1, j))
        return sparse.csr_matrix((val, (cell, who)), shape=(len(cells), len(players)))

    R = np.column_stack([np.ones(len(players)), np.random.default_rng(seed).poisson(1.0, size=(len(players), n_reps))])
    with np.errstate(invalid="ignore", divide="ignore"):
        ret = (agg(reg, 3) @ R) / (agg(reg, 2) @ R) + (agg(cor, 3) @ R) / (agg(cor, 2) @ R)
    return {c: ret[i] for i, c in enumerate(cells)}


def _ci(x: np.ndarray) -> tuple[float, float]:
    return float(np.nanpercentile(x[1:], 2.5)), float(np.nanpercentile(x[1:], 97.5))


KEEP_CURRENT = "維持現狀"


def best_arms(stats: pd.DataFrame, boot: dict[tuple[str, str], np.ndarray], leaf_order: list[str],
              dominant_share: float = 0.8, min_overlap_share: float = 0.8) -> pd.DataFrame:
    """每片葉子：比較 arm（樣本足夠 = 共同範圍內 ≥ min_n_per_arm 且納入比較）中加權留存率最高者，附平衡檢查與顯著性判讀。

    維持現狀：樣本足夠的 arm 不到兩個，且最多的 arm 佔葉子決策點超過 dominant_share（現行分派讓大家幾乎都是同一版本）
    → 不是資料不足，而是沒有其他版本可比；信心標「維持現狀_X」，不計入信心分數。

    SPEC 9.2 的「合格」還要求 max_smd ≤ smd_threshold；單片葉子很少達到，所以這裡只用樣本數挑候選，
    再依「最高者是否通過平衡檢查」「是否顯著高於次高」給判讀與信心（次高者是否平衡不影響信心）。
    共同範圍不到葉子的 min_overlap_share → 信心最多「低」（建議只對部分決策點有依據）。
    """
    rows = []
    for leaf in [l for l in leaf_order if l in set(stats["葉子"])]:
        g = stats[stats["葉子"] == leaf]
        cand = g[g["樣本足夠"]].sort_values("加權留存率", ascending=False)
        row = {"葉子": leaf, "葉子決策點": int(g["葉子決策點"].iloc[0]), "葉子玩家數": int(g["葉子玩家數"].iloc[0]),
               "葉子留存率": float(g["葉子留存率"].iloc[0]), "樣本足夠 arm 數": len(cand), "arm 數": len(g),
               "合格 arm 數": int(g["合格"].sum()),
               "最高留存 arm": "—", "加權留存率": np.nan, "留存_lo": np.nan, "留存_hi": np.nan, "最高者 max_smd": np.nan,
               "次高 arm": "—", "次高者 max_smd": np.nan, "− 次高": np.nan, "− 次高_lo": np.nan, "− 次高_hi": np.nan,
               "臨時基準 arm": "—", "− 基準": np.nan, "− 基準_lo": np.nan, "− 基準_hi": np.nan,
               "可比較比例": float(g["可比較比例"].iloc[0]) if "可比較比例" in g else np.nan,
               "判讀": "共同範圍內樣本足夠的版本不到兩個", "信心": "—"}
        top = g.sort_values(["決策點", "arm"], ascending=[False, True]).iloc[0]
        row["最多版本"], row["最多版本佔比"] = top["arm"], float(top["決策點"] / g["決策點"].sum())
        keep_current = len(cand) < 2 and row["最多版本佔比"] > dominant_share
        if keep_current:
            row["判讀"] = f"現行分派讓 {row['最多版本佔比']:.0%} 的決策點都是同一版本，沒有其他版本可比 → 維持現狀"
            row["信心"] = KEEP_CURRENT
        if cand.empty:
            rows.append(row)
            continue
        best = cand.iloc[0]
        b = boot[(leaf, best["arm"])]
        row.update({"最高留存 arm": best["arm"], "加權留存率": best["加權留存率"], "最高者 max_smd": best["max_smd"]})
        row["留存_lo"], row["留存_hi"] = _ci(b)
        base = cand.sort_values(["n", "arm"], ascending=[False, True]).iloc[0]
        row["臨時基準 arm"] = base["arm"]
        if base["arm"] != best["arm"]:
            d = b - boot[(leaf, base["arm"])]
            row["− 基準"] = d[0]
            row["− 基準_lo"], row["− 基準_hi"] = _ci(d)
        else:
            row["− 基準"] = 0.0
        if keep_current:
            pass
        elif len(cand) == 1:
            row["判讀"], row["信心"] = "只有一個 arm 樣本足夠，無從比較", "低"
        else:
            second = cand.iloc[1]
            d = b - boot[(leaf, second["arm"])]
            row.update({"次高 arm": second["arm"], "次高者 max_smd": second["max_smd"], "− 次高": d[0]})
            row["− 次高_lo"], row["− 次高_hi"] = _ci(d)
            significant, balanced = row["− 次高_lo"] > 0, bool(best["平衡"])        # 平衡只看最佳版本
            if significant and balanced:
                row["判讀"], row["信心"] = "可信：最佳版本平衡，且顯著高於次高", "高"
            elif significant:
                row["判讀"], row["信心"] = "顯著高於次高，但最佳版本未通過平衡檢查（差距可能來自玩家狀態）", "低"
            elif balanced:
                row["判讀"], row["信心"] = "最佳版本平衡，但與次高無顯著差異", "中"
            else:
                row["判讀"], row["信心"] = "與次高無顯著差異，且最佳版本未通過平衡檢查", "低"
        if len(cand) >= 2 and not keep_current and row["可比較比例"] < min_overlap_share:
            row["判讀"] += f"；共同範圍只涵蓋葉子 {row['可比較比例']:.0%}"
            row["信心"] = "低"
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 9.1b 葉子內校準
# ---------------------------------------------------------------------------
def balance_weights(X: np.ndarray, q: np.ndarray, tol: float) -> np.ndarray:
    """近似 entropy balancing：在 q（起始權重）附近找權重 w，使 X 各欄的加權平均都在 ±tol 內（X 已減去目標、除以標準差）。

    解 L1 懲罰的對偶：min log Σ q·exp(λ·x) + tol·‖λ‖₁（λ 拆成正負兩部分、有上限）；回傳加總為 1 的權重。
    """
    q = q / q.sum()
    k = X.shape[1]

    def dual(v):
        lam = v[:k] - v[k:]
        e = X @ lam
        m = e.max()
        s = q * np.exp(e - m)
        g = (s[:, None] * X).sum(axis=0) / s.sum()
        return np.log(s.sum()) + m + tol * v.sum(), np.concatenate([g + tol, tol - g])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = minimize(dual, np.zeros(2 * k), jac=True, method="L-BFGS-B", bounds=[(0, CALIBRATION_BOUND)] * (2 * k))
    e = X @ (res.x[:k] - res.x[k:])
    w = q * np.exp(e - e.max())
    return w / w.sum()


def calibrate_leaf_weights(df: pd.DataFrame, Z: np.ndarray, tol: float) -> np.ndarray:
    """9.1b：共同範圍內每個比較 arm 校準權重，讓 arm 的加權平均狀態對齊共同範圍；比原本的 W 更平衡才採用。回傳新的 W。"""
    W = df["W"].to_numpy(dtype=float).copy()
    pos = pd.Series(np.arange(len(df)), index=df.index)
    for _, g in df.groupby("葉子", sort=False):
        s = g[g["共同範圍"]]
        zl, wl = Z[s["_z"].to_numpy()], s[WEIGHT].to_numpy()
        mu = (wl[:, None] * zl).sum(axis=0) / wl.sum()
        sd = np.sqrt((wl[:, None] * (zl - mu) ** 2).sum(axis=0) / wl.sum())
        ok = sd > 1e-9
        if not ok.any():
            continue
        for _, ga in g[g["比較"]].groupby(ARM, sort=False):
            if len(ga) < CALIBRATION_MIN_ROWS:
                continue
            X = (Z[ga["_z"].to_numpy()][:, ok] - mu[ok]) / sd[ok]
            q = ga["W"].to_numpy(dtype=float)
            w = balance_weights(X, q, tol)
            before = np.abs(((q / q.sum())[:, None] * X).sum(axis=0)).max()
            if np.abs((w[:, None] * X).sum(axis=0)).max() < before:
                W[pos[ga.index].to_numpy()] = w * q.sum()
    return W


# ---------------------------------------------------------------------------
# 9.1a–9.2 一次做完
# ---------------------------------------------------------------------------
def evaluate_leaves(df: pd.DataFrame, ps: Propensity, le: dict, seed: int, leaf_order: list[str]):
    df, info = common_range(df, ps.P, le)
    if le.get("balance_calibration", True):
        df = df.assign(W=calibrate_leaf_weights(df, ps.Z, float(le.get("balance_tolerance", 0.05))))
    stats = leaf_arm_stats(df, ps.Z, ps.M, info, float(le["smd_threshold"]))
    order = {l: j for j, l in enumerate(leaf_order)}
    stats = stats.sort_values(["葉子", "arm"], key=lambda c: c.map(order) if c.name == "葉子" else c, kind="stable")
    boot = bootstrap_retention(df, ps.M, info, ps.arms, int(le["bootstrap_reps"]), seed)
    best = best_arms(stats, boot, leaf_order, float(le.get("dominant_share", 0.8)), float(le["min_overlap_share"]))
    return stats, best, {k: _ci(v) for k, v in boot.items()}

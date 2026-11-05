# -*- coding: utf-8 -*-
"""
五個體感補位（SPEC 9.4）。

以「合併相近建議」的結果為準，找出沒被任何葉子建議的版本 v。在每片有建議（r）的葉子內，用小樹描述
「v 與 r 的傾向分數都 ≥ overlap_min」的可比區（只看特徵，不看留存），在子規則內用 leaf_eval 的雙重穩健估計
比較 v 與 r；差距 ≤ max_loss_pp 的候選中，每個 v 一條（決策樹超過 many_trees 棵時 placements_many_trees 條、分在不同的樹）、
每片葉子最多一個版本，以整數規劃挑整體留存減少最少的組合。min_per_tree > 0 時，先讓每棵樹至少補到 min_per_tree 片；
force_coverage 時每棵樹、每個版本一定要補到，門檻內補不到就用門檻外的候選（標「超過門檻」）。
every_version_per_tree 時改以每棵樹計：該樹的葉子沒建議到的版本，都要在該樹內補一片（每棵樹都有每個版本）；
一片葉子可補多個版本：後補的在扣掉前面子規則後的其餘部分再挖子規則；窮舉位置與先後，補到最多者優先；
差距一律 ≤ max_loss_pp（force_coverage 只放寬可比區佔比）。
決策樹、AUC、前三個分頁不受影響。
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from feel_engine.evaluation.leaf_eval import (KEEP_CURRENT, _ci, bootstrap_retention, calibrate_leaf_weights,
                                              leaf_arm_stats)
from feel_engine.evaluation.propensity import ARM, WEIGHT, Propensity
from feel_engine.html import pages
from feel_engine.modeling.model import _render_path, fit_encoder

# 候選的結果（其他候選表的最後一欄）
PLACED, ELSEWHERE, TAKEN, OVER_LIMIT, NO_RANGE = "選中", "已補在其他葉子", "葉子已分給其他版本", "差距超過上限", "沒有可比區"
PLACED_OUTSIDE = "選中（超過門檻）"               # force_coverage：門檻內補不到，改用門檻外的候選
CARVED = "改補在這片葉子的其餘部分"               # every_version_per_tree：葉子先補了別的版本，扣掉後在其餘部分補到
UNPLACED_TAKEN = "可行的葉子已分給其他版本"      # 放不進去的原因
LOW_RANGE = "可比較比例不足"                      # 子規則內可比區佔比 < min_comparable_share：差距多半反映誰拿到，不採用
PLACE_REWARD = 1e3                                # 整數規劃中每補一片的獎勵：先求片數最多，再求損失最小
JSON_KEYS = ("version", "tree", "leaf", "base_version", "conditions", "decision_points", "leaf_decision_points", "share",
             "retention_v", "retention_base", "gap_pp", "gap_ci", "impact_pp", "impact_ci", "comparable_share",
             "max_smd", "balanced", "ess")


def recommendations(best: pd.DataFrame) -> dict[str, str]:
    """「合併相近建議」每片葉子建議的版本：高／中／低 → 最高留存 arm；維持現狀_X → X；無法比較不列。"""
    out = {}
    for r in best.to_dict(orient="records"):
        conf = pages.display_conf(r)
        if conf == pages.NOT_COMPARABLE:
            continue
        out[r["葉子"]] = r["最多版本"] if conf == KEEP_CURRENT else r["最高留存 arm"]
    return out


def used_versions(best: pd.DataFrame) -> set[str]:
    """有用到的版本 = 任一片葉子的建議（以整個實驗計，分群實驗的多棵樹合起來）。"""
    return set(recommendations(best).values())


def _node_paths(tree) -> dict[int, list[tuple[int, str, float]]]:
    """小樹每個節點（含根）→ 從根到該節點的條件 (特徵序號, le／gt, 門檻)；格式同 modeling/model.py 的 _leaf_paths。"""
    t = tree.tree_
    out: dict[int, list[tuple[int, str, float]]] = {}

    def walk(node: int, conds: list[tuple[int, str, float]]) -> None:
        out[node] = conds
        if t.children_left[node] != -1:
            f, thr = int(t.feature[node]), float(t.threshold[node])
            walk(t.children_left[node], [*conds, (f, "le", thr)])
            walk(t.children_right[node], [*conds, (f, "gt", thr)])

    walk(0, [])
    return out


def mine_sub_rule(X: np.ndarray, in_o: np.ndarray, arm_code: np.ndarray, v: int, r: int, w: np.ndarray,
                  depth: int, min_n: int, seed: int) -> tuple[list[tuple[int, str, float]], np.ndarray] | None:
    """葉子內：深度 ≤ depth 的小樹描述可比區 in_o（只用特徵與傾向分數，不看留存）。

    候選範圍 = 小樹每個節點（含根 = 整片葉子）；v、r 在「節點 ∩ 可比區」內各要 ≥ min_n 筆。
    挑可比區佔比（不加權筆數）最高者，同佔比挑筆數多者。回傳（條件, 節點內的列），沒有可行節點 → None。
    """
    from sklearn.tree import DecisionTreeClassifier

    tree = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=max(1, min_n), random_state=seed)
    tree.fit(X, in_o.astype(int), sample_weight=w)
    member = tree.decision_path(X).tocsc()
    best, best_key = None, None
    for node, conds in _node_paths(tree).items():
        rows = member[:, node].toarray().ravel().astype(bool)
        o = rows & in_o
        if (arm_code[o] == v).sum() < min_n or (arm_code[o] == r).sum() < min_n:
            continue
        key = (float(in_o[rows].mean()), int(rows.sum()))
        if best_key is None or key > best_key:
            best, best_key = (conds, rows), key
    return best


def compare_pair(rows: pd.DataFrame, in_o: np.ndarray, ps: Propensity, le: dict, v: int, r: int, name: str,
                 seed: int) -> dict:
    """子規則內只比 v 與 r：共同範圍 = 可比區 in_o、比較組合固定 {v, r}，沿用 leaf_eval 的權重（w_cap ÷ 傾向分數）、
    葉子內校準、雙重穩健加權留存率、max_smd 與玩家層級 bootstrap。

    rows：子規則內的驗證期間列（member_account、arm、churnNext20、w_cap、_a、_z）；name：當成葉子名稱用。
    回傳 v 與 r 的加權留存率、差距（r − v，百分點）與 95% 區間、可比較比例、v 的 max_smd、平衡與有效樣本。
    """
    rows = rows.reset_index(drop=True)
    a, z = rows["_a"].to_numpy(), rows["_z"].to_numpy()
    cmp_ = in_o & np.isin(a, [v, r])
    p = np.where(cmp_, ps.P[z, a], 1.0)
    df = rows.assign(葉子=name, 共同範圍=in_o, 比較=cmp_,
                     W=np.where(cmp_, rows[WEIGHT].to_numpy(dtype=float) / p, 0.0))
    if le.get("balance_calibration", True):
        df = df.assign(W=calibrate_leaf_weights(df, ps.Z, float(le.get("balance_tolerance", 0.05))))
    info = {name: {"比較": (v, r), "可比較比例": float(in_o.mean())}}
    stats = leaf_arm_stats(df, ps.Z, ps.M, info, float(le["smd_threshold"])).set_index(ARM)
    boot = bootstrap_retention(df, ps.M, info, ps.arms, int(le["bootstrap_reps"]), seed)
    gap = (boot[(name, ps.arms[r])] - boot[(name, ps.arms[v])]) * 100
    lo, hi = _ci(gap)
    sv = stats.loc[ps.arms[v]]
    return {"retention_v": float(sv["加權留存率"]), "retention_base": float(stats.loc[ps.arms[r], "加權留存率"]),
            "gap_pp": float(gap[0]), "gap_ci": [lo, hi], "comparable_share": float(in_o.mean()),
            "max_smd": float(sv["max_smd"]), "balanced": bool(sv["平衡"]), "ess": float(sv["有效樣本"])}


def _tree(c: dict) -> str:
    return c.get("tree", c["leaf"])


def _shortfall_reason(rest: list[dict], max_loss_pp: float, min_share: float, show_version: bool = False) -> str:
    """沒補滿時的原因（rest = 該版本在還沒用到的樹裡、有比出差距的候選；樹沒補滿時 = 該樹還沒補的候選，show_version）。"""
    ok = [c for c in rest if c["comparable_share"] >= min_share]
    if not rest:
        return NO_RANGE
    if not ok:
        return f"可比較比例都不到 {min_share:.0%}（最高 {max(c['comparable_share'] for c in rest):.1%}）"
    if all(c["gap_pp"] > max_loss_pp for c in ok):
        m = min(ok, key=lambda c: c["gap_pp"])
        return f"最小差距 {m['gap_pp']:.1f} pp（{pages.arm_short(m['version']) + '，' if show_version else ''}{m['leaf']}）"
    return UNPLACED_TAKEN


def _unplaced_reason(rest: list[dict], max_loss_pp: float, min_share: float, force: bool, gap_capped: bool = False,
                     show_version: bool = False) -> str:
    """補不到的原因；force 時門檻外也能補，補不到只會是沒有可比區、差距都超過 max_loss_pp（gap_capped：
    every_version_per_tree 時差距仍設限）或葉子已分給其他版本。"""
    if not force:
        return _shortfall_reason(rest, max_loss_pp, min_share, show_version)
    if rest and gap_capped and all(c["gap_pp"] > max_loss_pp for c in rest):
        m = min(rest, key=lambda c: c["gap_pp"])
        return (f"最小差距 {m['gap_pp']:.2f} pp（{pages.arm_short(m['version']) + '，' if show_version else ''}{m['leaf']}）"
                f" > {max_loss_pp:g} pp")
    return UNPLACED_TAKEN if rest else NO_RANGE


def _solve_placement(feasible: list[dict], per_version: int) -> list[dict]:
    """整數規劃（scipy milp）：每個候選選或不選；每個版本 ≤ per_version 片、每個（版本, 樹）≤ 1 片、每片葉子 ≤ 1 個版本。
    目標先讓片數最多（每片獎勵 PLACE_REWARD，遠大於 |impact_pp|），再讓 impact_pp 總和最小。"""
    from scipy.optimize import Bounds, LinearConstraint, milp

    n = len(feasible)
    rows, caps = [], []
    for key, cap in ((lambda c: c["version"], per_version), (lambda c: (c["version"], _tree(c)), 1),
                     (lambda c: c["leaf"], 1)):
        groups: dict = {}
        for i, c in enumerate(feasible):
            groups.setdefault(key(c), []).append(i)
        for idx in groups.values():
            row = np.zeros(n)
            row[idx] = 1.0
            rows.append(row)
            caps.append(cap)
    res = milp(np.array([c["impact_pp"] - PLACE_REWARD for c in feasible]),
               constraints=LinearConstraint(np.array(rows), -np.inf, np.array(caps, dtype=float)),
               integrality=np.ones(n), bounds=Bounds(0, 1))
    if not res.success:
        raise RuntimeError(f"補位的整數規劃沒有解：{res.message}")
    return [c for c, x in zip(feasible, res.x) if x > 0.5]


def _solve_placement_per_tree(pool: list[dict], per_version: int, min_per_tree: int,
                              outside: frozenset[int] = frozenset()) -> list[dict]:
    """min_per_tree > 0 或 force_coverage 的整數規劃：每個（版本, 樹）≤ 1 片、每片葉子 ≤ 1 個版本（版本片數不設上限）。
    目標依序（逐層求解，每層鎖住上一層的最佳值）：每棵樹補到 min_per_tree 片 → 每個版本補到 per_version 片 →
    門檻外的片數最少（outside = 門檻外候選的 id）→ 片數最少 → impact_pp 總和最小。
    變數 = 候選 x ＋ 每棵樹、每個版本「算數的片數」（≤ 實際片數、≤ 目標）。"""
    from scipy.optimize import Bounds, LinearConstraint, milp

    n = len(pool)
    trees = list(dict.fromkeys(_tree(c) for c in pool))
    versions = list(dict.fromkeys(c["version"] for c in pool))
    nt, nv = len(trees), len(versions)
    k = n + nt + nv
    rows, caps = [], []
    for key in (lambda c: (c["version"], _tree(c)), lambda c: c["leaf"]):
        groups: dict = {}
        for i, c in enumerate(pool):
            groups.setdefault(key(c), []).append(i)
        for idx in groups.values():
            row = np.zeros(k)
            row[idx] = 1.0
            rows.append(row)
            caps.append(1.0)
    for offset, key, names in ((n, _tree, trees), (n + nt, lambda c: c["version"], versions)):
        for j, name in enumerate(names):
            row = np.zeros(k)
            row[offset + j] = 1.0
            row[[i for i, c in enumerate(pool) if key(c) == name]] = -1.0
            rows.append(row)
            caps.append(0.0)
    upper = np.concatenate([np.ones(n), np.full(nt, min_per_tree), np.full(nv, per_version)])
    on_x = lambda v: np.concatenate([v, np.zeros(nt + nv)])            # noqa: E731
    goals = [np.concatenate([np.zeros(n), -np.ones(nt), np.zeros(nv)]),          # 樹補到的片數最多
             np.concatenate([np.zeros(n + nt), -np.ones(nv)]),                   # 版本補到的片數最多
             on_x(np.array([1.0 if id(c) in outside else 0.0 for c in pool])),  # 門檻外的片數最少
             on_x(np.ones(n)),                                                   # 片數最少
             on_x(np.array([c["impact_pp"] for c in pool]))]                     # 整體影響最小
    for i, goal in enumerate(goals):
        res = milp(goal, constraints=LinearConstraint(np.array(rows), -np.inf, np.array(caps)),
                   integrality=np.ones(k), bounds=Bounds(0, upper), options={"mip_rel_gap": 0})
        if not res.success:
            raise RuntimeError(f"補位的整數規劃沒有解：{res.message}")
        if i < len(goals) - 1:                    # 前幾層的目標都是整數：鎖住最佳值（+0.5 容許浮點誤差）
            rows.append(goal)
            caps.append(res.fun + 0.5)
    return [c for c, x in zip(pool, res.x[:n]) if x > 0.5]


def _outside_text(c: dict, max_loss_pp: float, min_share: float) -> str:
    """門檻外的候選超過哪些門檻。"""
    return "、".join([*([f"差距 {c['gap_pp']:.1f} pp > {max_loss_pp:g} pp"] if c["gap_pp"] > max_loss_pp else []),
                     *([f"可比較比例 {c['comparable_share']:.1%} < {min_share:.0%}"]
                       if c["comparable_share"] < min_share else [])])


def tree_shortfall(cands: list[dict], placed: list[dict], trees: list[str], min_per_tree: int, max_loss_pp: float,
                   min_share: float, force: bool = False) -> list[dict]:
    """min_per_tree：補不到 min_per_tree 片的樹 [{tree, reason[, placed_n]}]；reason 看該樹還沒補的版本的候選
    （force 時門檻外也能補，補不到只會是沒有可比區或葉子已分給其他版本）。"""
    n = Counter(_tree(p) for p in placed)
    out = []
    for t in trees:
        if n[t] >= min_per_tree:
            continue
        done = {p["version"] for p in placed if _tree(p) == t}
        rest = [c for c in cands if _tree(c) == t and c.get("gap_pp") is not None and c["version"] not in done]
        reason = _unplaced_reason(rest, max_loss_pp, min_share, force, show_version=True)
        out.append({"tree": t, "reason": reason, **({"placed_n": n[t]} if n[t] else {})})
    return out


def choose_placement(cands: list[dict], missing: list[str], max_loss_pp: float, min_share: float,
                     per_version: int = 1, min_per_tree: int = 0, force: bool = False) -> tuple[list[dict], list[dict]]:
    """差距點估計 ≤ max_loss_pp、且子規則內可比區佔比（comparable_share）≥ min_share 的候選中，
    每個待補版本最多 per_version 片、同一版本的各片在不同的樹，每片葉子最多一個版本；
    先讓補到的片數最多，再讓整體影響（impact_pp）總和最小（v 較好時為負，自然優先）。
    min_per_tree > 0：先讓每棵樹至少補到 min_per_tree 片，版本片數不設上限（_solve_placement_per_tree）。
    force：每棵樹（min_per_tree）與每個版本（per_version）一定要補到；門檻內補不到時改用門檻外有比出差距的候選，
    門檻外的片數最少、其次整體影響最小；選中的寫 outside（超過哪些門檻）、status = PLACED_OUTSIDE。
    可比區佔比太低時（例如切不出子規則、整片葉子只有 1% 可比），差距多半反映「誰拿到」而不是版本效果，不採用（force 除外）。

    cands：每個（版本, 葉子）一筆（tree = 葉子所屬的樹），沒有可比區時 gap_pp = None；會就地寫入 status（其他候選表用）。
    回傳（選中的候選，依 missing 順序；沒補滿 per_version 片的 [{version, reason[, placed_n]}]，
    placed_n = 已補到的片數（0 片時不寫），reason 說明其餘的樹為什麼補不到）。
    """
    mined = [c for c in cands if c.get("gap_pp") is not None]
    within = lambda c: c["comparable_share"] >= min_share and c["gap_pp"] <= max_loss_pp   # noqa: E731
    feasible = [c for c in mined if within(c)]
    if force:
        outside = frozenset(id(c) for c in mined if not within(c))
        picked = _solve_placement_per_tree(mined, per_version, min_per_tree, outside) if mined else []
    else:
        outside = frozenset()
        solve = (lambda f: _solve_placement_per_tree(f, per_version, min_per_tree)) if min_per_tree else (
            lambda f: _solve_placement(f, per_version))
        picked = solve(feasible) if feasible else []
    chosen, n_placed = {id(c) for c in picked}, Counter(c["version"] for c in picked)
    used_trees = {(c["version"], _tree(c)) for c in picked}
    for c in cands:
        if c.get("gap_pp") is None:
            c["status"] = NO_RANGE
        elif id(c) in chosen:
            c["status"] = PLACED_OUTSIDE if id(c) in outside else PLACED
            if id(c) in outside:
                c["outside"] = _outside_text(c, max_loss_pp, min_share)
        elif c["comparable_share"] < min_share:
            c["status"] = LOW_RANGE
        elif c["gap_pp"] > max_loss_pp:
            c["status"] = OVER_LIMIT
        elif n_placed[c["version"]] >= per_version or (c["version"], _tree(c)) in used_trees:
            c["status"] = ELSEWHERE
        else:
            c["status"] = TAKEN
    short = []
    for v in missing:
        n = n_placed[v]
        if n >= per_version:
            continue
        rest = [c for c in mined if c["version"] == v and (v, _tree(c)) not in used_trees]
        short.append({"version": v, "reason": _unplaced_reason(rest, max_loss_pp, min_share, force),
                      **({"placed_n": n} if n else {})})
    return [c for v in missing for c in picked if c["version"] == v], short


@dataclass
class FiveResult:
    used: list[str]                                   # 有用到的版本（完整 arm 名稱）
    missing: list[str]                                # 待補版本
    placed: list[dict] = field(default_factory=list)      # 選中的候選（欄位見 JSON_KEYS）
    unplaced: list[dict] = field(default_factory=list)    # [{version, reason}]
    candidates: list[dict] = field(default_factory=list)  # 每個（葉子, 待補版本）一筆，含 status
    per_version: int = 1                              # 每個待補版本要補幾片（決策樹超過 many_trees 棵時 > 1）
    min_per_tree: int = 0                             # 每棵樹至少補幾片（0 = 不要求）
    unfilled_trees: list[dict] = field(default_factory=list)  # [{tree, reason[, placed_n]}]（min_per_tree > 0 時）
    force: bool = False                               # force_coverage：門檻內補不到時改用門檻外的候選
    per_tree: bool = False                            # every_version_per_tree：每棵樹都要有每個版本
    need: dict[str, list[str]] = field(default_factory=dict)  # per_tree：{樹: 該樹的葉子沒建議到、要補的版本}

    def to_json(self, max_loss_pp: float) -> dict:
        """five_feelings.json 的內容（程式讀取用）。"""
        out = {"max_loss_pp": max_loss_pp, "placements_per_version": self.per_version, "used": self.used,
               "placed": [{**{k: p[k] for k in JSON_KEYS}, **{k: p[k] for k in ("outside", "after") if p.get(k)}}
                          for p in self.placed], "unplaced": self.unplaced}
        if self.min_per_tree:
            out.update(min_per_tree=self.min_per_tree, unfilled_trees=self.unfilled_trees)
        if self.force:
            out["force_coverage"] = True
        if self.per_tree:
            out.update(every_version_per_tree=True, need=self.need)
        return out


def fill_versions(df: pd.DataFrame, feat: pd.DataFrame, feat_pos: np.ndarray, best: pd.DataFrame, rules: dict,
                  ps: Propensity, le: dict, ff: dict, seed: int) -> FiveResult:
    """一個實驗的補位（只用驗證期間）。

    df：該實驗的驗證期間列（葉子 = 「合併相近建議」後的名稱；member_account、arm、churnNext20、w_cap、_a、_z）；
    feat、feat_pos：特徵表與「ps.table 第 i 列（= _z）→ feat 第 feat_pos[i] 列」；best：「合併相近建議」的 best_arms；
    rules：tree_rules.json（features、labels、missing、params.categories：子規則與決策樹同樣的特徵、編碼與寫法）；
    ff：config.yaml 的 five_feelings。
    """
    rec = recommendations(best)
    used = sorted(set(rec.values()))
    many = len(rules.get("trees") or {}) > int(ff.get("many_trees", 3))
    per_tree = bool(ff.get("every_version_per_tree", False))
    need = {}
    if per_tree:                                  # 每棵樹要補的版本 = 該樹的葉子沒建議到的版本
        by_tree: dict[str, set[str]] = {t: set() for t in rules.get("trees") or {}}
        for leaf, arm in rec.items():
            by_tree.setdefault(leaf.split("/", 1)[0], set()).add(arm)
        need = {t: [a for a in ps.arms if a not in s] for t, s in by_tree.items() if set(ps.arms) - s}
    result = FiveResult(used, [a for a in ps.arms if (any(a in vs for vs in need.values()) if per_tree else a not in used)],
                        per_version=int(ff.get("placements_many_trees", 2)) if many and not per_tree else 1,
                        min_per_tree=int(ff.get("min_per_tree", 0)), force=bool(ff.get("force_coverage", False)),
                        per_tree=per_tree, need=need)
    if not result.missing:
        return result
    df = df.reset_index(drop=True)
    z, a_all, w_all = df["_z"].to_numpy(), df["_a"].to_numpy(), df[WEIGHT].to_numpy(dtype=float)
    total_w = float(w_all.sum())
    cols = list(rules["features"])
    col_pos = [feat.columns.get_loc(c) for c in cols]
    miss = rules["missing"]
    allowed, labels = (rules.get("params") or {}).get("categories") or {}, rules.get("labels") or {}
    lo, min_n, depth = float(le["overlap_min"]), int(le["min_n_per_arm"]), int(ff.get("sub_rule_depth", 2))
    leaf_rows = df.groupby("葉子", sort=False).indices
    rows_of: dict[int, np.ndarray] = {}           # id(候選) → 子規則內的列（同一片葉子再補其他版本時扣掉）

    def mine_slot(leaf: str, idx: np.ndarray, r_arm: str, versions: list[str], after: list[str]) -> list[dict]:
        """葉子（after 非空時 = 扣掉已補版本 after 的子規則後的其餘部分）內，每個 v 挖子規則並與 r 比較。"""
        leaf_feat = feat.iloc[feat_pos[z[idx]], col_pos]                 # 只取這片葉子的特徵欄，省記憶體
        enc = fit_encoder(leaf_feat, cols, miss["fill"], bool(miss["add_isna"]), allowed, labels)
        X = enc.transform(leaf_feat).to_numpy(dtype=float) if enc.columns else np.zeros((len(idx), 1))
        r = ps.arms.index(r_arm)
        Pl = ps.P[z[idx]]
        out = []
        for v_arm in versions:
            v = ps.arms.index(v_arm)
            cand = {"version": v_arm, "tree": leaf.split("/", 1)[0], "leaf": leaf, "base_version": r_arm,
                    "leaf_decision_points": int(len(idx)), "gap_pp": None, **({"after": after} if after else {})}
            out.append(cand)
            in_o = (Pl[:, v] >= lo) & (Pl[:, r] >= lo)
            mined = mine_sub_rule(X, in_o, a_all[idx], v, r, w_all[idx], depth, min_n, seed)
            if mined is None:
                continue
            conds, inside = mined
            sub = idx[inside]
            rows_of[id(cand)] = sub
            res = compare_pair(df.iloc[sub], in_o[inside], ps, le, v, r, f"{leaf}·{pages.arm_short(v_arm)}", seed)
            share = float(w_all[sub].sum() / total_w)
            cand.update(res, conditions=_render_path(enc, conds), decision_points=int(len(sub)), share=share,
                        impact_pp=share * res["gap_pp"], impact_ci=[share * res["gap_ci"][0], share * res["gap_ci"][1]])
        return out

    for leaf, r_arm in rec.items():
        idx = leaf_rows.get(leaf)
        if idx is not None:
            result.candidates += mine_slot(leaf, idx, r_arm,
                                           result.need.get(leaf.split("/", 1)[0], []) if per_tree else result.missing, [])
    max_loss, min_share = float(ff.get("max_loss_pp", 2.0)), float(ff.get("min_comparable_share", 0.5))
    if per_tree:
        place_every_version(result, rec, leaf_rows, rows_of, mine_slot, list(rules.get("trees") or {}), max_loss, min_share)
    else:
        result.placed, result.unplaced = choose_placement(result.candidates, result.missing, max_loss, min_share,
                                                          result.per_version, result.min_per_tree, result.force)
    if result.min_per_tree:
        result.unfilled_trees = tree_shortfall(result.candidates, result.placed, list(rules.get("trees") or {}),
                                               result.min_per_tree, max_loss, min_share, result.force)
    return result


def place_every_version(result: FiveResult, rec: dict[str, str], leaf_rows: dict, rows_of: dict[int, np.ndarray],
                        mine_slot, trees: list[str], max_loss_pp: float, min_share: float) -> None:
    """every_version_per_tree：每棵樹分開找（樹之間不共用葉子），一片葉子可補多個版本：依序補，後補的在扣掉前面
    子規則後的其餘部分重新挖子規則、比較（mine_slot(葉子, 列, r, 版本們, after)；after = 先補的版本）。
    窮舉每個版本補在哪片葉子、同一片葉子內的先後，依序取：補到的版本最多 → 門檻外的片數最少 → 整體影響總和最小。
    差距一律 ≤ max_loss_pp；force_coverage 只放寬可比區佔比（可比區不足的也能用，標門檻外），否則只用門檻內。
    可行的葉子 = 整片葉子挖得出子規則者（其餘部分是子集，整片挖不出就不可能）。
    rows_of：id(候選) → 子規則內的列；就地寫入 result 的 placed、unplaced、status，並把選中的其餘部分候選加進 candidates。"""
    within = lambda c: c["comparable_share"] >= min_share and c["gap_pp"] <= max_loss_pp   # noqa: E731
    usable = lambda c: (c is not None and c.get("gap_pp") is not None and c["gap_pp"] <= max_loss_pp   # noqa: E731
                        and (result.force or within(c)))
    whole = {(c["version"], c["leaf"]): c for c in result.candidates}
    slots: dict[tuple[str, tuple[str, ...]], dict[str, dict]] = {}

    def slot(leaf: str, prefix: tuple[str, ...]) -> dict[str, dict]:
        """葉子依序補了 prefix 之後，其餘部分的候選 {版本: 候選}（快取）。"""
        if (leaf, prefix) not in slots:
            if not prefix:
                slots[leaf, prefix] = {v: c for (v, lf), c in whole.items() if lf == leaf}
            else:
                done = [slot(leaf, prefix[:i])[prefix[i]] for i in range(len(prefix))]
                idx = np.setdiff1d(leaf_rows[leaf], np.concatenate([rows_of[id(c)] for c in done]))
                vs = [v for v in result.need[leaf.split("/", 1)[0]]
                      if v not in prefix and whole.get((v, leaf), {}).get("gap_pp") is not None]
                slots[leaf, prefix] = ({c["version"]: c for c in mine_slot(leaf, idx, rec[leaf], vs, list(prefix))}
                                       if len(idx) and vs else {})
        return slots[leaf, prefix]

    def best_for(vs: list[str], leaves: list[str]) -> list[dict]:
        viable = {v: [lf for lf in leaves if whole.get((v, lf), {}).get("gap_pp") is not None] for v in vs}
        best: list = [None, []]
        seen = set()

        def dfs(left: tuple[str, ...], seqs: dict[str, tuple[str, ...]]) -> None:
            state = frozenset(seqs.items())
            if state in seen:
                return
            seen.add(state)
            chosen = [slot(lf, seq[:i])[seq[i]] for lf, seq in seqs.items() for i in range(len(seq))]
            key = (-len(chosen), sum(not within(c) for c in chosen), sum(c["impact_pp"] for c in chosen))
            if best[0] is None or key < best[0]:
                best[:] = [key, chosen]
            for v in left:
                for lf in viable[v]:
                    seq = seqs.get(lf, ())
                    if usable(slot(lf, seq).get(v)):
                        dfs(tuple(x for x in left if x != v), {**seqs, lf: (*seq, v)})

        dfs(tuple(vs), {})
        return best[1]

    placed = [c for t, vs in result.need.items() for c in best_for(vs, [lf for lf in rec if lf.split("/", 1)[0] == t])]
    ids, at = {id(c) for c in placed}, {(c["version"], c["tree"]): c["leaf"] for c in placed}
    result.candidates = [x for c in result.candidates
                         for x in (c, *(p for p in placed if p.get("after") and (p["version"], p["leaf"]) == (c["version"], c["leaf"])))]
    for c in result.candidates:
        if c.get("gap_pp") is None:
            c["status"] = NO_RANGE
        elif id(c) in ids:
            c["status"] = PLACED if within(c) else PLACED_OUTSIDE
            if not within(c):
                c["outside"] = _outside_text(c, max_loss_pp, min_share)
        elif c["comparable_share"] < min_share:
            c["status"] = LOW_RANGE
        elif c["gap_pp"] > max_loss_pp:
            c["status"] = OVER_LIMIT
        elif (c["version"], c["tree"]) in at:
            c["status"] = CARVED if at[c["version"], c["tree"]] == c["leaf"] else ELSEWHERE
        else:
            c["status"] = TAKEN
    order = {v: i for i, v in enumerate(result.missing)}
    result.placed = sorted(placed, key=lambda p: (order[p["version"]], trees.index(p["tree"]) if p["tree"] in trees
                                                  else len(trees), len(p.get("after") or ())))
    result.unplaced = [{"version": v, "tree": t, "reason": _unplaced_reason(
        [c for (cv, lf), c in whole.items() if cv == v and lf.split("/", 1)[0] == t and c.get("gap_pp") is not None],
        max_loss_pp, min_share, result.force, gap_capped=True)} for t, vs in result.need.items() for v in vs if (v, t) not in at]


def augment_tree(nodes: list[dict], collapsed: set[int], names: dict[int, str], info: dict[int, dict],
                 placed: list[dict], colors: dict[str, str]) -> tuple[list[dict], set[int], dict[int, dict]]:
    """「五個體感」樹圖：被選中的葉子改畫成提問節點（子規則的條件），「是」→ 補位葉子、「否」→ 原葉子其餘部分；
    子規則就是整片葉子時，直接把那片葉子改標補位。同一片葉子補多個版本（after）時，後補的子規則接在「其餘」上。
    names（節點 → 葉子名稱）、info、collapsed 同 pages.tree_svg 的輸入；不改動傳入的物件。
    新增的葉子用 stat 顯示驗證期間筆數（原樹節點的筆數是訓練期間）。"""
    by_id = {n["id"]: dict(n) for n in nodes}
    collapsed, info = set(collapsed), dict(info)
    node_of = {leaf: i for i, leaf in names.items()}
    next_id = max(by_id) + 1
    rest_of: dict[str, int] = {}                  # 葉子 → 目前的「其餘」節點
    for p in sorted(placed, key=lambda p: len(p.get("after") or ())):
        k = rest_of.get(p["leaf"], node_of[p["leaf"]])
        short = pages.short_leaf(p["leaf"])
        fill = {"name": f"{short} {pages.FILL}", "best": pages.arm_short(p["version"]), "color": colors.get(p["version"]),
                "conf": pages.FILL, "retention": p["retention_v"],
                "detail": f"比 {pages.arm_short(p['base_version'])} {'低' if p['gap_pp'] >= 0 else '高'} "
                          f"{abs(p['gap_pp']):.1f} pp" + ("，超過門檻" if p.get("outside") else "")}
        conds = p["conditions"]
        if not conds:
            info[k] = fill
            continue
        text = [conds[0], f"且 {' 且 '.join(conds[1:])}？"] if len(conds) > 1 else [f"{pages.FILL}條件", f"{conds[0]}？"]
        yes, no = next_id, next_id + 1
        next_id += 2
        by_id[k].update(left=yes, right=no, text=text, left_label="是", right_label="否")
        collapsed.discard(k)
        rest = p["leaf_decision_points"] - p["decision_points"]
        by_id[yes] = {"id": yes, "left": -1, "right": -1, "leaf": p["leaf"], "stat": f"驗證 {p['decision_points']:,} 筆"}
        by_id[no] = {"id": no, "left": -1, "right": -1, "leaf": p["leaf"], "stat": f"驗證 {rest:,} 筆"}
        info[yes] = fill
        info[no] = {**info.pop(k), "name": f"{short} 其餘"}
        rest_of[p["leaf"]] = no
    return list(by_id.values()), collapsed, info

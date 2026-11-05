# -*- coding: utf-8 -*-
"""
決策樹的實驗比較（跨實驗 = split × tags）→ result/{run}/summary.html：AUC ＋ 信心分數。

- 讀 run 產生的 output/{run}/model/{split}/tag_{tags}/valid_predictions.parquet；
  排序依 model.yaml 的 experiments，清單外的結果附在最後並標示。
- 共同樣本：每個實驗都有預測的驗證期間決策點（value 只建模低／中／高時，共同樣本就是這三群），在同一批決策點上比較。
- AUC 以 w_cap 加權；95% 區間與「與基準（split = all、TAG_ = 無）差」用玩家層級 Poisson(1) bootstrap，
  所有實驗共用同一組重抽權重；差的區間不含 0 才寫「較佳／較差」，否則「無顯著差異」（SPEC 11.2）。
- 信心分數（evaluate 之後才有）：共同樣本中（維持現狀的葉子不計入），落在信心高／中／低／無法比較葉子的決策點佔比
  （w_cap 加權），分數 = 100 × (高佔比 + 0.5 × 中佔比)；原始葉子與合併後各算一次（CONF_POINTS），summary 只顯示合併後。
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from feel_engine.html import pages
from feel_engine.paths import RunPaths

log = logging.getLogger("feel_engine.html.compare")

PLAYER = "member_account"
KEY = [PLAYER, "spin_ts"]
BASELINE = ("all", "none")
CONF_POINTS = {"高": 1.0, "中": 0.5, "低": 0.0, "無法比較": 0.0}   # 信心分數的給分；維持現狀的葉子不計入
CONF_LEVELS = tuple(CONF_POINTS)
                                                                    # 評分依據見 summary.html、SPEC 8.2


def tag_label(tags: str) -> str:
    return "無" if tags == "none" else tags.replace("_", "＋")


def exp_label(key: tuple[str, str]) -> str:
    return f"{key[0]}／{tag_label(key[1])}"


# ---------------------------------------------------------------------------
# 指標
# ---------------------------------------------------------------------------
def bucket_auc(score: np.ndarray, y: np.ndarray, w: np.ndarray, player: np.ndarray, reps: np.ndarray) -> np.ndarray:
    """加權 AUC（同分各給一半），每一欄 reps[:, r] 是一組玩家權重；回傳長度 = reps 欄數。

    先把 w × y、w × (1 − y) 依（預測值, 玩家）加總，再乘上玩家權重矩陣，一次算完所有重抽。
    """
    levels, k = np.unique(score, return_inverse=True)               # 由低到高
    n_players = reps.shape[0]
    pos = np.zeros((len(levels), n_players))
    neg = np.zeros((len(levels), n_players))
    np.add.at(pos, (k, player), w * y)
    np.add.at(neg, (k, player), w * (1 - y))
    P, N = pos @ reps, neg @ reps                                     # 每個預測值 × 每次重抽
    below = np.cumsum(N, axis=0) - N
    num = (P * (below + 0.5 * N)).sum(axis=0)
    den = P.sum(axis=0) * N.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def verdict(lo: float, hi: float) -> str:
    if np.isnan(lo) or np.isnan(hi):
        return "—"
    return "較佳" if lo > 0 else ("較差" if hi < 0 else "無顯著差異")


# ---------------------------------------------------------------------------
# 讀取、比較
# ---------------------------------------------------------------------------
def load_predictions(rp: RunPaths, order: list[tuple[str, str]]) -> dict[tuple[str, str], pd.DataFrame]:
    """依實驗清單排序；清單外但已跑過的結果附在最後（_listed = False）。"""
    found = {}
    for path in (rp.output_dir / "model").glob("*/tag_*/valid_predictions.parquet"):
        key = (path.parent.parent.name, path.parent.name.removeprefix("tag_"))
        found[key] = pd.read_parquet(path).assign(_listed=key in order)
    rank = {k: i for i, k in enumerate(order)}
    return dict(sorted(found.items(), key=lambda kv: (rank.get(kv[0], len(rank)), kv[0])))


def common_keys(preds: dict[tuple[str, str], pd.DataFrame]) -> pd.MultiIndex:
    """共同樣本：每個實驗都有預測的驗證期間決策點（KEY）。"""
    common = None
    for df in preds.values():
        idx = pd.MultiIndex.from_frame(df[KEY])
        common = idx if common is None else common.intersection(idx)
    return common.sort_values()


def compare_auc(preds: dict[tuple[str, str], pd.DataFrame], n_reps: int, seed: int) -> dict:
    """在共同樣本上算每個實驗的 AUC（95% 區間）與相對基準的差距。"""
    keys = list(preds)
    common = common_keys(preds)
    aligned = {k: df.set_index(KEY).loc[common] for k, df in preds.items()}
    first = aligned[keys[0]]
    y, w = first["y"].to_numpy(dtype=float), first["w"].to_numpy(dtype=float)
    player, players = pd.factorize(common.get_level_values(PLAYER))
    rng = np.random.default_rng(seed)
    reps = np.column_stack([np.ones(len(players)), rng.poisson(1.0, size=(len(players), n_reps))])
    auc = {k: bucket_auc(aligned[k]["p"].to_numpy(), y, w, player, reps) for k in keys}
    rows = []
    for i, k in enumerate(keys, 1):
        row = {"#": i, "split": k[0], "tags": k[1], "實驗": exp_label(k), "在清單": bool(preds[k]["_listed"].iloc[0]),
               "AUC": auc[k][0], "AUC_lo": np.nanpercentile(auc[k][1:], 2.5), "AUC_hi": np.nanpercentile(auc[k][1:], 97.5),
               "− 基準": np.nan, "− 基準_lo": np.nan, "− 基準_hi": np.nan, "判定": "基準" if k == BASELINE else "—"}
        if BASELINE in auc and k != BASELINE:
            d = auc[k] - auc[BASELINE]
            lo, hi = np.nanpercentile(d[1:], 2.5), np.nanpercentile(d[1:], 97.5)
            row.update({"− 基準": d[0], "− 基準_lo": lo, "− 基準_hi": hi, "判定": verdict(lo, hi)})
        rows.append(row)
    return {"metrics": pd.DataFrame(rows), "n_common": len(common), "n_players": len(players), "n_reps": n_reps,
            "common": common}


def confidence_summary(rows: pd.DataFrame, conf: dict) -> dict:
    """rows：某實驗在共同樣本上的預測（葉子、w）；conf：leaf_confidence.json。
    回傳 {orig／merged: {share: 高／中／低／無法比較的決策點佔比（w 加權，合計 1）, score: 0–100}}；找不到信心的葉子當無法比較。
    維持現狀（現行分派讓大家幾乎都是同一版本）的葉子不計入；全部都是維持現狀時 share 為空、score 為 NaN。"""
    w = rows["w"].to_numpy(dtype=float)
    views = {"orig": rows["葉子"].map(conf["orig"]),
             "merged": rows["葉子"].map(conf["merge_map"]).map(conf["merged"])}
    out = {}
    for view, level in views.items():
        level = level.fillna("無法比較").to_numpy()
        scored = np.isin(level, CONF_LEVELS)
        total = w[scored].sum()
        share = {lv: float(w[level == lv].sum() / total) if total > 0 else float("nan") for lv in CONF_LEVELS}
        out[view] = {"share": share,
                     "score": 100 * sum(CONF_POINTS[lv] * share[lv] for lv in CONF_LEVELS) if total > 0 else float("nan")}
    return out


def load_confidence(rp: RunPaths, key: tuple[str, str]) -> dict | None:
    """evaluate 寫的 leaf_confidence.json；比 valid_predictions.parquet 舊（run 之後還沒 evaluate）→ None。"""
    path, pred = rp.leaf_confidence(*key), rp.model_predictions(*key)
    if not path.exists() or path.stat().st_mtime < pred.stat().st_mtime:
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_summary(rp: RunPaths, cfg: dict, mcfg: dict) -> Path | None:
    order = [(e["split"], e["tags"]) for e in mcfg.get("experiments") or []]
    preds = load_predictions(rp, order)
    if not preds:
        log.warning("沒有任何 valid_predictions.parquet，請先執行 run")
        return None
    res = compare_auc(preds, int(cfg["leaf_eval"]["bootstrap_reps"]), int(cfg["seed"]))
    m = res["metrics"]
    first = rp.tree_rules(*next(iter(preds)))
    dates = json.loads(first.read_text(encoding="utf-8")) if first.exists() else {}
    missing = [exp_label(k) for k in order if k not in preds]
    extra = [r["實驗"] for r in m.to_dict(orient="records") if not r["在清單"]]
    nan = lambda v: None if v != v else float(v)                    # noqa: E731
    conf = {}
    for k, df in preds.items():
        c = load_confidence(rp, k)
        conf[k] = None if c is None else confidence_summary(df.set_index(KEY).loc[res["common"]].reset_index(), c)
    rows = [{"label": f"#{r['#']} {r['實驗']}" + ("" if r["在清單"] else "（清單外）"),
             "href": rp.experiment_html(r["split"], r["tags"]).name,
             "auc": r["AUC"], "lo": r["AUC_lo"], "hi": r["AUC_hi"], "diff": nan(r["− 基準"]),
             "diff_lo": nan(r["− 基準_lo"]), "diff_hi": nan(r["− 基準_hi"]), "verdict": r["判定"],
             "conf": conf[(r["split"], r["tags"])]}
            for r in m.to_dict(orient="records")]
    base = m.loc[(m["split"] == BASELINE[0]) & (m["tags"] == BASELINE[1]), "AUC"]
    meta = [
        *([f"⚠️ 實驗清單中尚未跑：{pages.esc('、'.join(missing))}"] if missing else []),
        *([f"⚠️ 不在實驗清單、但已跑過（附在最後）：{pages.esc('、'.join(extra))}"] if extra else []),
    ]
    facts = [
        f"商戶／幣別／遊戲：{pages.esc(dates.get('merchant', '—'))}／{pages.esc(dates.get('currency', '—'))}／"
        f"{pages.esc(dates.get('game', '—'))}",
        f"訓練期間：美東 {pages.esc(dates.get('train_start', '—'))}～{pages.esc(dates.get('train_end', '—'))}（用來長決策樹）",
        f"驗證期間：{pages.esc(pages.period(dates.get('valid_start', '—'), dates.get('valid_end', '—')))}（用來算 AUC 與信心）",
        f"共同樣本：{res['n_common']:,} 個決策點、{res['n_players']:,} 位玩家（每個實驗都有預測；value 只含低／中／高三群）",
        "實驗順序：依 model.yaml 的 experiments",
    ]
    le = cfg["leaf_eval"]
    auc_notes = [
        "<b>AUC</b>：預測 20 手內離場（churnNext20）的排序能力，0.5 = 亂猜、1 = 完美（w_cap 加權）。",
        "<b>判定</b>：與 all／無 的差距，95% 區間不含 0 才算較佳／較差。",
    ]
    conf_notes = [
        "<b>信心分數</b> = 100 ×（高佔比 ＋ 0.5 × 中佔比）；佔比以共同樣本的決策點計（w_cap 加權），維持現狀不計入。",
        f"<b>信心</b>：高 = 最佳版本平衡，且顯著勝過次佳；中 = 平衡但不顯著；低 = 未平衡，或共同範圍不到葉子 "
        f"{float(le['min_overlap_share']):.0%}；無法比較 = 樣本足夠的版本不到兩個；"
        f"維持現狀_X = 同一版本佔葉子超過 {float(le.get('dominant_share', 0.8)):.0%}。",
        "信心以合併後的葉子計算：無法比較的葉子併回上一層後重算。",
    ]
    rp.result_dir.mkdir(parents=True, exist_ok=True)
    rp.summary_html.write_text(pages.summary_page("決策樹實驗比較（AUC ＋ 信心）", meta, rows,
                                                  float(base.iloc[0]) if len(base) else None, auc_notes, conf_notes,
                                                  facts=facts),
                               encoding="utf-8")
    return rp.summary_html

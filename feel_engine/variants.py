# -*- coding: utf-8 -*-
"""
實驗的變體：改補位規則或加標籤後，另外產生一頁 HTML 與正式實驗並排比較
（不在 model.yaml 清單、不進 summary.html、不覆蓋正式結果；在專案根目錄執行）：

  python -m feel_engine.variants new                       # → result/{run}/lifecycle_tag_none_new.html
  python -m feel_engine.variants alltags                   # → result/{run}/all_tag_alltags.html
  python -m feel_engine.variants alltags --config my.yaml  # 換訓練期間／驗證期間（run 依 config 的 data）

流程與正式實驗相同（讀逐日 dataset、套用 config.yaml 的 regroup → fit_split → 第 9 節評估），變體只改：
  extra_tags    ：標籤表的其他類別欄位，依 build 同樣的規則加進 dataset（決策週之前最新一筆週標籤；沒有紀錄 → 無紀錄）
  tag_set       ：這個變體的 TAG_ 組合（X = 快層 ＋ tag_set）；傾向分數仍用 model.yaml 的標籤，與正式實驗相同
  five_feelings ：覆蓋 config.yaml 的 five_feelings
頁首：與 compare 列出的正式實驗在同一批決策點上比較 AUC（玩家層級 bootstrap 95% 區間，同 summary.html）。
輸出：output/{run}/variants/{頁名}/（tree_rules.json、valid_predictions.parquet、leaf_confidence.json、five_feelings.json）、
      result/{run}/{頁名}.html；頁名 = {split}_tag_{tags}[_{suffix}]（paths.VariantPaths）。
注意：run --clean 會清掉 result/{run}/*_tag_*.html（含變體頁）；重跑本指令即可產生。
"""
from __future__ import annotations

import argparse
import copy
import logging
import sys
from datetime import date

import numpy as np
import pandas as pd
import polars as pl

from feel_engine.paths import CONFIG_PATH, MODEL_CONFIG_PATH, RunPaths, VariantPaths, rel, run_paths, tag_dir
from feel_engine.pipeline.data import load_config, run_name

log = logging.getLogger("feel_engine.variants")

EXTRA_TAGS = {                                     # 標籤表欄位 → (dataset 欄位, 中文名稱)
    "bet_points_tag": ("TAG_bet_points", "總下注額等級"),
    "avg_bet_points_tag": ("TAG_avg_bet_points", "平均下注額等級"),
    "bet_counts_tag": ("TAG_bet_counts", "下注手數等級"),
    "fund_level_tag": ("TAG_fund_level", "資金水位"),
    "buy_feature_tag": ("TAG_buy_feature", "購買特色"),
    "bet_intensity_tag": ("TAG_bet_intensity", "下注強度"),
}

VARIANTS = {
    "new": {                                       # 實驗 5，補位改為每棵樹都要有每個版本
        "split": "lifecycle", "tags": "none", "suffix": "new",
        "five_feelings": {"every_version_per_tree": True, "force_coverage": True},
        "title": "實驗 5 新版：lifecycle／無（每棵樹都有每個版本、差距 ≤ 2 pp）",
        "changes": "五個體感改為每棵樹都要有每個版本："
                   "這棵樹的葉子沒建議到的版本，都在這棵樹內補一片子規則，門檻內補不到時可用可比區不足的候選（標「超過門檻」），"
                   "但補位版本比原建議版本留存率低超過 2 pp 的位置一律不用（寧可那棵樹缺這個版本）",
        "compare": [("lifecycle", "none")],
    },
    "alltags": {                                   # 全部玩家一棵樹，X = 快層 ＋ 標籤表所有有資料的類別標籤
        "split": "all", "tags": "alltags", "suffix": "",
        "extra_tags": list(EXTRA_TAGS),
        "tag_set": ["TAG_lifecycle", "TAG_value_combined", *(col for col, _ in EXTRA_TAGS.values())],
        "title": "測試：all／全部標籤",
        "changes": "全部玩家一棵樹；X = 快層 ＋ 標籤表所有有資料的類別標籤：生命週期、價值，以及總下注額、平均下注額、"
                   "下注手數、資金水位、購買特色、下注強度（決策週之前最新一筆週標籤；沒有紀錄 → 無紀錄，"
                   "上週沒下注 → 本週無下注；價值只取低／中／高，其他標籤每個類別都 one-hot）",
        "compare": [("all", "none"), ("all", "lifecycle_value")],
    },
}


def attach_tags(df: pd.DataFrame, cfg: dict, columns: list[str]) -> pd.DataFrame:
    """標籤表的其他類別欄位依 build 同樣的規則加進 dataset：決策週 W（week_index）讀 as-of 檔 apply_week=W，
    取 week_index < W 的最新一筆；沒有紀錄 → 無紀錄。"""
    from feel_engine.pipeline.build import NO_RECORD_VALUE

    d = cfg["data"]
    folder = tag_dir(d["merchant"], d["currency"], d["game"])
    parts = []
    for w in sorted(int(x) for x in df["week_index"].unique()):
        start = date.fromisocalendar(w // 100, w % 100, 1)
        rows = pl.read_parquet(folder / f"apply_week={start:%Y-%m-%d}.parquet",
                               columns=["member_account", "week_index", *columns])
        latest = (rows.filter(pl.col("week_index") < w)
                  .sort(["member_account", "week_index"], descending=[False, True])
                  .unique("member_account", keep="first", maintain_order=True))
        parts.append(latest.with_columns(week_index=pl.lit(w, dtype=pl.Int64)))
    names = {c: EXTRA_TAGS[c][0] for c in columns}
    tags = pl.concat(parts).rename(names).to_pandas()
    out = df.merge(tags.astype({"week_index": df["week_index"].dtype}), on=["member_account", "week_index"], how="left")
    return out.fillna({col: NO_RECORD_VALUE for col in names.values()})


def compare_official(rp: RunPaths, cfg: dict, pred: pd.DataFrame, others: list[tuple[str, str]],
                     order: list[tuple[str, str]]) -> list[str]:
    """與正式實驗在同一批決策點（兩邊都有預測）上比較 AUC：玩家層級 Poisson(1) bootstrap 的 95% 區間（同 summary.html）。"""
    from feel_engine.html.compare import KEY, PLAYER, bucket_auc, exp_label, verdict

    mine = pred.set_index(KEY)
    lines = []
    for key in others:
        path = rp.model_predictions(*key)
        if not path.exists():
            continue
        off = pd.read_parquet(path, columns=[*KEY, "p", "y"]).set_index(KEY)
        common = mine.index.intersection(off.index)
        a, b = mine.loc[common], off.loc[common]
        y, w = a["y"].to_numpy(dtype=float), a["w"].to_numpy(dtype=float)
        player, players = pd.factorize(common.get_level_values(PLAYER))
        rng = np.random.default_rng(int(cfg["seed"]))
        reps = np.column_stack([np.ones(len(players)),
                                rng.poisson(1.0, size=(len(players), int(cfg["leaf_eval"]["bootstrap_reps"])))])
        auc_v = bucket_auc(a["p"].to_numpy(), y, w, player, reps)
        auc_o = bucket_auc(b["p"].to_numpy(), y, w, player, reps)
        diff = auc_v - auc_o
        lo, hi = np.nanpercentile(diff[1:], [2.5, 97.5])
        label = (f"實驗 {order.index(key) + 1}：" if key in order else "") + exp_label(key)
        mismatch = int((a["y"].to_numpy() != b["y"].to_numpy()).sum())
        lines.append(f'對照 <a href="{rp.experiment_html(*key).name}">{label}</a>（同一批 {len(common):,} 個決策點'
                     + (f"，其中 {mismatch:,} 筆 Y 不同、以本頁為準" if mismatch else "")
                     + f"）：AUC {auc_o[0]:.4f} → 本頁 {auc_v[0]:.4f}，差 {diff[0]:+.4f} [{lo:+.4f}, {hi:+.4f}]（{verdict(lo, hi)}）")
    return lines


def run_variant(name: str, config=CONFIG_PATH, model_config=MODEL_CONFIG_PATH) -> int:
    from feel_engine.cli import _load_train_valid
    from feel_engine.evaluation.evaluate import evaluate_experiment, prepare_evaluation
    from feel_engine.modeling.model import fit_split, load_model_config, write_split_outputs

    spec = VARIANTS[name]
    split, tags = spec["split"], spec["tags"]
    cfg, mcfg = load_config(config), load_model_config(model_config)
    rp = run_paths(run_name(cfg))
    vp = VariantPaths(rp.name, variant=spec.get("suffix", name))
    loaded = _load_train_valid(cfg)
    if loaded is None:
        return 1
    train, valid = loaded
    if spec.get("extra_tags"):
        train, valid = (attach_tags(d, cfg, spec["extra_tags"]) for d in (train, valid))
    fit_mcfg = copy.deepcopy(mcfg)                 # 傾向分數仍用 mcfg（與正式實驗相同的共變數）
    if spec.get("tag_set"):
        fit_mcfg["tags"][tags] = list(spec["tag_set"])
        fit_mcfg["labels"].update({col: label for col, label in EXTRA_TAGS.values()})

    res = fit_split(train, valid, split, tags, cfg, fit_mcfg)
    write_split_outputs(res, vp.model_dir(split, tags), fit_mcfg, vp.model_predictions(split, tags), model_config,
                        regroup=cfg.get("regroup"))
    log.info("變體 %s：%d 棵樹（%s），驗證期間加權 AUC %.4f", name, len(res.trees), "、".join(res.trees), res.auc_valid)

    vcfg = copy.deepcopy(cfg)
    vcfg["five_feelings"] = {**(cfg.get("five_feelings") or {}), **spec.get("five_feelings", {})}
    order = [(e["split"], e["tags"]) for e in mcfg["experiments"]]
    facts = (f"變更：{spec['changes']}", f"驗證期間加權 AUC（全部決策點）：{res.auc_valid:.4f}",
             *compare_official(rp, cfg, res.predictions, spec.get("compare", [(split, tags)]), order))
    for line in facts[2:]:
        log.info("%s", line)
    V, ps, base, colors = prepare_evaluation(valid, vcfg, mcfg)
    path = evaluate_experiment(vp, vcfg, split, tags, spec["title"], V, ps, base, colors, facts)
    log.info("變體 %s → %s", name, rel(path))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="feel_engine.variants", description="實驗的變體：另外產生一頁 HTML，不進 summary.html")
    parser.add_argument("variant", choices=sorted(VARIANTS))
    parser.add_argument("--config", default=CONFIG_PATH)
    parser.add_argument("--model-config", default=MODEL_CONFIG_PATH)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    return run_variant(args.variant, args.config, args.model_config)


if __name__ == "__main__":
    sys.exit(main())

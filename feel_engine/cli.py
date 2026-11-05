# -*- coding: utf-8 -*-
"""
指令入口（在專案根目錄執行）；商戶／幣別／遊戲、訓練期間、驗證期間見 config.yaml 的 data：

  python -m feel_engine.cli build                     # 建本 run 還沒有（或已過期）的日子：資料檢查 + 逐日 dataset；已建好的日子直接沿用
  python -m feel_engine.cli build --provisional-arm   # version_map 未填完時，以 game_version 暫代 arm
  python -m feel_engine.cli build --rebuild           # 本 run 的日子全部重建
  python -m feel_engine.cli run                                  # 跑 model.yaml 的 experiments 全部（訓練期間訓練、驗證期間預測）
  python -m feel_engine.cli run --split all --tags lifecycle     # 只跑一組（可逗號分隔，跑所有組合）
  python -m feel_engine.cli run --model-config my.yaml           # 換一份特徵與實驗設定檔
  python -m feel_engine.cli compare                              # 實驗比較 summary.html（AUC ＋ 信心分數；run、evaluate 結束時也會做）
  python -m feel_engine.cli evaluate                             # 第 9 節：每個實驗一頁 HTML（決策樹、葉子 × 版本留存率、信心）
"""
from __future__ import annotations

import argparse
import logging
import shutil
import sys
from datetime import date, datetime
from pathlib import Path

from feel_engine.pipeline.data import (
    DataCheckError,
    contiguous_ranges,
    load_config,
    run_name,
    train_valid_days,
    run_stage1,
    version_map_missing,
)
from feel_engine.pipeline.store import OK, build_metas, check_days, day_metas, day_status, fingerprint, prune_builds
from feel_engine.paths import CONFIG_PATH, MODEL_CONFIG_PATH, rel, run_paths

log = logging.getLogger("feel_engine")

BUILD_NOTES = [
    "dataset 一天一檔（`session_date` = session 起點的美東日期），所有 run 共用；build 只建還沒有的日子，"
    "一段連續的日子一起讀、建一份 session 表。跨午夜的 session 整段歸起點那天。",
    "截尾以 session 起點日為準（隔天 0 點 + `post_hours` 前要結束），所以一天的決策點與 build 一次建幾天無關。",
    "dataset 欄位依序：索引（`member_account`、`spin_ts`、`session_date`、`session_id`、`session_spin_idx`、`week_index`）→ "
    "遊戲版本（`game_version`、`arm`、`arm_ratio`）→ 快層（`GP_` 只用當手之前／`SY_` 系統端／`NGP_` 含當手開獎前資訊）→ "
    "慢層 `TAG_` → 分群 `SEG_` → 流失與權重（`churnNext20`、`w_cap`）。`session_id` = `session_date`（yyyymmdd）× 1000 "
    "＋ 該玩家當天第幾個 session。",
    "樣本版本（`arm`）= 之後 20 手的主版本；決策點當手的版本見 `game_version`。BUY 手以 `game_mode` 位元 16 判定、"
    "含 FreeGame 的手以位元 2 判定（只用來算當手之前的 `GP_dailyRate_FREE_GAME`）。",
    "慢層標籤（`TAG_`）取決策週之前最新一筆（無紀錄 → 新手／無紀錄）；分群（`SEG_`）與標籤相同。",
    "特徵除以 0 存為缺值（null）；建模時再依 model.yaml 填補。",
]


def _run_days(cfg: dict) -> tuple[list[date], list[date]]:
    train, valid = train_valid_days(cfg)
    return [date.fromisoformat(d) for d in train], [date.fromisoformat(d) for d in valid]


def cmd_build(args: argparse.Namespace) -> int:
    from feel_engine.pipeline.build import run_stage2
    from feel_engine.pipeline.report import write_build_report, write_data_quality_report
    from feel_engine.pipeline.store import load_dataset

    cfg = load_config(args.config)
    version_map = cfg["modules"]["version_map"]
    rp = run_paths(run_name(cfg))
    store = rp.store
    train, valid = _run_days(cfg)
    days = [*train, *valid]

    status = day_status(cfg, store, days)
    todo = days if args.rebuild else [d for d in days if status[d] != OK]
    reused = [d for d in days if d not in todo]
    log.info("本 run %d 天：沿用已建好的 %d 天、要建 %d 天 %s", len(days), len(reused), len(todo),
             "、".join(f"{a}～{b}" if a != b else str(a) for a, b in contiguous_ranges(todo)) or "")
    for reason in sorted({status[d] for d in todo} - {OK}):
        log.info("  %s：%s", reason, ", ".join(str(d) for d in todo if status[d] == reason))

    fp = fingerprint(cfg)
    if todo and rp.state_debug.exists():
        shutil.rmtree(rp.state_debug)  # 只清本程式自己產生的除錯輸出
    for first, last in contiguous_ranges(todo):
        build_id = f"{datetime.now():%Y%m%dT%H%M%S}_{first:%Y%m%d}-{last:%Y%m%d}"
        try:
            meta = run_stage1(cfg, first, last, store)
            missing = version_map_missing(meta["version_counts"], version_map)
            if missing and not args.provisional_arm:
                write_data_quality_report(rp, {build_id: {"build_id": build_id, "first_day": str(first),
                                                          "last_day": str(last), "stage1": meta}}, version_map, None)
                log.error("version_map 未填完：%s。已輸出版本分佈至 %s 後停止；"
                          "填好 config.yaml 後重跑 build（或加 --provisional-arm 先以 game_version 暫代）",
                          missing, rel(rp.dq_report))
                return 2
            run_stage2(cfg, meta, bool(missing), store, rp, build_id, fp)
        except DataCheckError:
            log.exception("資料檢查未通過，中止（%s～%s）", first, last)
            return 1
        finally:
            if store.tmp_dir.exists():
                shutil.rmtree(store.tmp_dir)  # 分桶等暫存：build 結束即刪，不留第二份注單
    removed = prune_builds(store)
    if removed:
        log.info("已刪除不再被任何一天使用的 build 紀錄：%s", removed)

    metas = {str(d): m for d, m in day_metas(store, days).items()}
    builds = build_metas(store, {m["build_id"] for m in metas.values()})
    write_data_quality_report(rp, builds, version_map, {d: m["stats"] for d, m in metas.items()})
    ds = load_dataset(store, days)
    write_build_report(rp, cfg, [str(d) for d in train], [str(d) for d in valid], metas, builds, ds, BUILD_NOTES)
    log.info("dataset：%s（本 run %d 天，%s 筆）；報告：%s、%s", rel(store.dataset_dir), len(days), f"{ds.height:,}",
             rel(rp.dq_report), rel(rp.build_report))
    return 0


def _load_train_valid(cfg: dict):
    """（訓練期間 dataset, 驗證期間 dataset）：從逐日 dataset 讀訓練期間與驗證期間的日子（pandas）。
    有缺或過期的日子 → 回傳 None（請先 build）。"""
    from feel_engine.pipeline.store import load_dataset

    train, valid = _run_days(cfg)
    store = run_paths(run_name(cfg)).store
    problems = check_days(cfg, store, [*train, *valid])
    if problems:
        for p in problems:
            log.error(p)
        log.error("請先執行 build（只會建缺的或過期的日子）")
        return None
    return load_dataset(store, train).to_pandas(), load_dataset(store, valid).to_pandas()


def _summary(rp, cfg: dict, model_config) -> None:
    from feel_engine.html.compare import write_summary
    from feel_engine.modeling.model import load_model_config

    path = write_summary(rp, cfg, load_model_config(model_config))
    if path:
        log.info("實驗比較（AUC，依 model.yaml 的 experiments 排序）→ %s", rel(path))


def _clean_model_outputs(rp) -> None:
    """清掉上次 run／evaluate 產生的結果（只限本程式產生的檔案與資料夾），避免殘留舊設定或已移出清單的實驗。"""
    import shutil

    r = rp.result_dir
    targets = [*(r / s for s in ("all", "lifecycle", "value")),          # 舊版的實驗資料夾（report.md、樹圖…）
               r / "compare", r / "propensity", rp.summary_html, *sorted(r.glob("*_tag_*.html")),
               rp.output_dir / "model"]
    removed = [t for t in targets if t.exists()]
    for t in removed:
        if t.is_dir():
            shutil.rmtree(t)
        else:
            t.unlink()
    log.info("已清除舊的實驗結果 %d 處（%s 的 HTML 與舊報告、%s）", len(removed), rel(rp.result_dir), rel(rp.output_dir / "model"))


def cmd_run(args: argparse.Namespace) -> int:
    """第 8 節：訓練期間訓練決策樹、驗證期間預測，最後更新實驗比較（summary.html）。

    不給 --split／--tags：跑 model.yaml 的 experiments 全部；有給：跑 split × tags 的所有組合。
    """
    from feel_engine.modeling.model import NOT_MODELED, TRAINED, ModelConfigError, fit_split, load_model_config, write_split_outputs

    cfg = load_config(args.config)
    try:
        mcfg = load_model_config(args.model_config)
    except ModelConfigError:
        log.exception("model.yaml 不合法，中止")
        return 1
    if args.split is None and args.tags is None:
        runs = [(e["split"], e["tags"]) for e in mcfg["experiments"]]
        if not runs:
            log.error("model.yaml 沒有 experiments；請列出實驗，或用 --split、--tags 指定")
            return 1
    elif args.clean:
        log.error("--clean 只能在跑整份實驗清單時使用（不要同時給 --split／--tags），以免清掉其他實驗的結果")
        return 1
    else:
        splits = [s.strip() for s in (args.split or "all").split(",") if s.strip()]
        tag_names = [t.strip() for t in (args.tags or "none").split(",") if t.strip()]
        runs = [(s, t) for s in splits for t in tag_names]
    loaded = _load_train_valid(cfg)
    if loaded is None:
        return 1
    train, valid = loaded
    rp = run_paths(run_name(cfg))
    if args.clean:
        _clean_model_outputs(rp)
    for split, tags in runs:
        try:
            res = fit_split(train, valid, split, tags, cfg, mcfg)
        except ModelConfigError:
            log.exception("設定不合法，中止")
            return 1
        write_split_outputs(res, rp.model_dir(split, tags), mcfg, rp.model_predictions(split, tags), args.model_config)
        n = res.groups["策略"].value_counts()
        log.info("split=%s TAG_=%s：%d 棵樹（自建 %d 群、沿用 all %d 群、不建模 %d 群），驗證期間加權 AUC %s",
                 split, tags, len(res.trees), n.get(TRAINED, 0), n.get("沿用 all", 0), n.get(NOT_MODELED, 0),
                 "—" if res.auc_valid is None else f"{res.auc_valid:.4f}")
    _summary(rp, cfg, args.model_config)
    log.info("每個實驗的 HTML（決策樹、葉子 × 版本留存率）由 evaluate 產生")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    """第 9 節：已跑過決策樹的實驗，每片葉子 × arm 的加權留存率（共同範圍內的雙重穩健估計）與最高留存 arm，每個實驗寫一頁 HTML。"""
    from feel_engine.evaluation.evaluate import evaluate_experiments
    from feel_engine.modeling.model import ModelConfigError, load_model_config

    cfg = load_config(args.config)
    try:
        mcfg = load_model_config(args.model_config)
    except ModelConfigError:
        log.exception("model.yaml 不合法，中止")
        return 1
    from feel_engine.pipeline.store import provisional_arm

    rp = run_paths(run_name(cfg))
    loaded = _load_train_valid(cfg)
    if loaded is None:
        return 1
    valid = loaded[1]
    provisional = provisional_arm(rp.store, [*_run_days(cfg)[0], *_run_days(cfg)[1]])
    paths = evaluate_experiments(rp, cfg, mcfg, valid, provisional)
    if paths:
        log.info("實驗頁面 %d 個 → %s；總覽 %s", len(paths), rel(rp.result_dir), rel(rp.summary_html))
    log.warning("第 10–11 節（換算體感模組 M1–M7、離線評估）尚未實作")
    return 0 if paths else 1


def cmd_compare(args: argparse.Namespace) -> int:
    """實驗比較：所有已跑過的實驗，依 model.yaml 的 experiments 排序，在共同樣本上比較 AUC 與信心分數。"""
    cfg = load_config(args.config)
    _summary(run_paths(run_name(cfg)), cfg, args.model_config)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="feel_engine")
    parser.add_argument("--config", default=CONFIG_PATH)
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build", help="建本 run 還沒有（或已過期）的日子：資料檢查 + 逐日 dataset；已建好的日子沿用")
    b.add_argument("--provisional-arm", action="store_true", help="version_map 未填完時以 game_version 暫代 arm")
    b.add_argument("--rebuild", action="store_true", help="本 run 的日子全部重建（不管是否已建好）")
    b.set_defaults(func=cmd_build)
    r = sub.add_parser("run", help="決策樹（第 8 節，訓練期間訓練、驗證期間預測）；不給 --split／--tags 時跑 model.yaml 的 experiments 全部")
    r.add_argument("--split", help="all／lifecycle／value，可逗號分隔")
    r.add_argument("--tags", help="model.yaml 的 tags 名稱（none／lifecycle／value…），可逗號分隔")
    r.add_argument("--model-config", type=Path, default=MODEL_CONFIG_PATH, help="特徵與實驗設定檔")
    r.add_argument("--clean", action="store_true", help="先清掉舊的實驗結果（只能搭配整份實驗清單）")
    r.set_defaults(func=cmd_run)
    e = sub.add_parser("evaluate", help="第 9 節：每個實驗一頁 HTML（決策樹、葉子 × 版本留存率、信心；需先 run）")
    e.add_argument("--model-config", type=Path, default=MODEL_CONFIG_PATH, help="特徵與實驗設定檔")
    e.set_defaults(func=cmd_evaluate)
    c = sub.add_parser("compare", help="實驗比較 summary.html：AUC ＋ 信心分數（依 model.yaml 的 experiments 排序；run、evaluate 結束時也會做）")
    c.add_argument("--model-config", type=Path, default=MODEL_CONFIG_PATH, help="特徵與實驗設定檔（取實驗順序）")
    c.set_defaults(func=cmd_compare)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

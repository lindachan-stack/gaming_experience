# -*- coding: utf-8 -*-
"""
逐日 dataset（output/{商戶}_{幣別}_{遊戲}/dataset/）：每個 session_date 一檔，同一天只存一份、所有 run 共用（SPEC 2、5.0）。

  {美東日}.parquet          該日的決策點（DATASET_COLUMNS）
  {美東日}.json             指紋、列數、產生它的 build、逐日統計（build 報告彙總用）
  builds/{build_id}.json    一次 build：讀取範圍、R1–R7、用到的注單日檔與標籤週（判斷過期、資料檢查報告用）

一天的決策點只由「該日的資料＋前後固定的範圍」決定（5.3 截尾也以該日為準），與 build 一次建幾天無關，
所以不同 run 可以共用同一天。下列任一項改變時該日視為過期，build 會重建：
  指紋 = build 相關設定（data.pre_hours、post_hours、labels、weighting、features.derived、modules.version_map）
         ＋ pipeline/data.py、build.py 的程式內容；
  注單快取中用到的日檔列數、或標籤快取中用到的週有變（重抓過）；
  建置時注單快取尾端不足（R7 後段不足，該日有 session 因看不完 Y 被截掉），而現在的快取已經補上。
"""
from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, time as dtime, timedelta, timezone
from pathlib import Path

import polars as pl

from feel_engine.paths import PACKAGE_DIR, StorePaths, bet_records_dir, rel, tag_dir
from feel_engine.pipeline.data import et_midnight

OK = "ok"
TAIL_FILLED = "建置時注單快取尾端不足，現已補上"
FINGERPRINT_CODE = (PACKAGE_DIR / "pipeline" / "data.py", PACKAGE_DIR / "pipeline" / "build.py")


def build_settings(cfg: dict) -> dict:
    """會影響決策點內容的設定（不含 compute、seed：只影響速度與除錯抽樣）。"""
    return {
        "pre_hours": cfg["data"]["pre_hours"],
        "post_hours": cfg["data"].get("post_hours", 0),
        "labels": cfg["labels"],
        "weighting": cfg["weighting"],
        "derived": cfg["features"]["derived"],
        "version_map": cfg["modules"]["version_map"],
    }


def fingerprint(cfg: dict) -> str:
    h = hashlib.sha256(json.dumps(build_settings(cfg), sort_keys=True, ensure_ascii=False, default=str).encode("utf-8"))
    for p in FINGERPRINT_CODE:
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def day_metas(store: StorePaths, days: list[date]) -> dict[date, dict]:
    """已建好的日子的 meta（缺的日子不在結果中）。"""
    return {d: _read_json(store.day_meta(d)) for d in days if store.day(d).exists() and store.day_meta(d).exists()}


def build_metas(store: StorePaths, build_ids: set[str]) -> dict[str, dict]:
    return {b: _read_json(store.build_meta(b)) for b in sorted(build_ids) if store.build_meta(b).exists()}


def current_raw_rows(cfg: dict) -> dict[str, int]:
    path = bet_records_dir(cfg["data"]["merchant"], cfg["data"]["game"]) / "_manifest.csv"
    if not path.exists():
        return {}
    m = pl.read_csv(path, try_parse_dates=True).filter(pl.col("currency").cast(pl.String) == cfg["data"]["currency"])
    return {str(d): int(n) for d, n in zip(m["bet_date"].to_list(), m["actual_rows"].to_list()) if n is not None}


def current_tag_weeks(cfg: dict) -> dict[str, dict]:
    """標籤快取 manifest：apply_week → {rows, max_etl_time}。"""
    path = tag_dir(cfg["data"]["merchant"], cfg["data"]["currency"], cfg["data"]["game"]) / "_manifest.csv"
    if not path.exists():
        return {}
    m = pl.read_csv(path, infer_schema=False)
    return {r["apply_week_start"]: {"rows": int(float(r["actual_rows"])), "max_etl_time": r["max_etl_time"]}
            for r in m.iter_rows(named=True) if r["actual_rows"] not in (None, "")}


def day_status(cfg: dict, store: StorePaths, days: list[date]) -> dict[date, str]:
    """每一天：OK，或缺／過期的原因。"""
    fp = fingerprint(cfg)
    metas = day_metas(store, days)
    builds = build_metas(store, {m["build_id"] for m in metas.values()})
    raw_now, tags_now = current_raw_rows(cfg), current_tag_weeks(cfg)
    cache_end = (datetime.combine(max(date.fromisoformat(k) for k in raw_now) + timedelta(days=1), dtime.min, tzinfo=timezone.utc)
                 if raw_now else None)                    # 注單快取目前的終點（UTC，不含）
    out = {}
    for d in days:
        m = metas.get(d)
        if m is None:
            out[d] = "還沒建"
            continue
        b = builds.get(m["build_id"])
        if m["fingerprint"] != fp:
            out[d] = "build 設定或程式已改"
        elif b is None:
            out[d] = f"找不到 build 紀錄 {m['build_id']}"
        elif any(raw_now.get(k) != v for k, v in b["stage1"]["raw_rows"].items()):
            out[d] = "注單快取已重抓"
        elif any(tags_now.get(w) != v for w, v in b["tag_weeks"].items() if w == m.get("tag_week")):
            out[d] = "標籤快取已重抓"
        elif _tail_now_available(cfg, d, b, cache_end):
            out[d] = TAIL_FILLED
        else:
            out[d] = OK
    return out


def _tail_now_available(cfg: dict, d: date, build: dict, cache_end: datetime | None) -> bool:
    """該日建置時注單快取的尾端不足（看不完 Y 的 session 整段不抽，5.3），而現在的快取比當時長 → 重建會多出決策點。

    該日需要讀到「隔天美東 0 點 ＋ post_hours ＋ return_window」；build 的 data_end 比它早才算被截尾。
    快取沒有變長時重建結果相同，不判過期。"""
    plan = build["stage1"].get("read_plan") or {}
    if not plan.get("back_short") or cache_end is None:
        return False
    data_end = datetime.fromisoformat(plan["data_end"])
    need = et_midnight(d + timedelta(days=1)) + timedelta(
        hours=float(cfg["data"].get("post_hours", 0)) + float(cfg["labels"]["return_window_hours"]))
    return data_end < need and cache_end > data_end


def dataset_signature(store: StorePaths, days: list[date]) -> str:
    """這些日子目前的 dataset 是哪一份：每天的 build 與列數的指紋。任一天重建（即使內容相同）就會變；
    run 記在 tree_rules.json，evaluate 比對，避免拿舊的樹與預測配上重建後的 dataset。"""
    metas = day_metas(store, days)
    parts = [(str(d), metas[d]["build_id"], metas[d]["rows"]) if d in metas else (str(d), None, None) for d in sorted(days)]
    return hashlib.sha256(json.dumps(parts).encode("utf-8")).hexdigest()[:16]


def check_days(cfg: dict, store: StorePaths, days: list[date]) -> list[str]:
    """run／evaluate 前檢查：回傳問題清單（空 = 可用）。"""
    status = day_status(cfg, store, days)
    bad = {d: s for d, s in status.items() if s != OK}
    if not bad:
        return []
    by_reason: dict[str, list[str]] = {}
    for d, s in bad.items():
        by_reason.setdefault(s, []).append(str(d))
    return [f"{rel(store.dataset_dir)} {reason}：{', '.join(ds)}" for reason, ds in by_reason.items()]


def load_dataset(store: StorePaths, days: list[date], columns: list[str] | None = None) -> pl.DataFrame:
    """讀指定日子的決策點，依 session_date、member_account、spin_ts 排序。"""
    paths = [store.day(d) for d in sorted(days)]
    missing = [rel(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"缺 {missing}，請先執行 build")
    return pl.read_parquet(paths, columns=columns, hive_partitioning=False)


def provisional_arm(store: StorePaths, days: list[date]) -> bool:
    flags = {m["provisional_arm"] for m in day_metas(store, days).values()}
    if len(flags) > 1:
        raise ValueError("各天的 arm 對應方式不同（有的以 game_version 暫代、有的依 version_map），請重新 build")
    return flags.pop() if flags else False


def prune_builds(store: StorePaths) -> list[str]:
    """刪掉已沒有任何一天引用的 build 紀錄（那些天都已被重建）。只刪本程式產生的 builds/*.json。"""
    if not store.builds_dir.exists():
        return []
    used = {_read_json(p)["build_id"] for p in store.dataset_dir.glob("*.json")}
    removed = []
    for p in store.builds_dir.glob("*.json"):
        if p.stem not in used:
            p.unlink()
            removed.append(p.stem)
    return removed

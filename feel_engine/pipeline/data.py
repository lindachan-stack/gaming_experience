# -*- coding: utf-8 -*-
"""
第一段：日檔 → 玩家分桶（SPEC 5.0、5.1、第 12 節 R1–R4、R7）。

  build 只建還沒有（或已過期）的日子：缺的日子依連續區段分組，每段一次讀完。
  R1 manifest 對帳（只看設定的幣別）→ R7 讀取範圍（區段第一天 0 點 − pre_hours 至最後一天 24:00 + post_hours + return_window）
  → 範圍內日檔讀 9 欄，只留範圍內的列 → R2 去重、R3、R4 → 轉美東 → 依 member_account 雜湊寫入分桶（暫存，build 結束即刪）

實作細節：
  - game_version → arm 的對應在第二段（build.py）做，桶內保留 game_version。
  - 每列加 seq = 日序 × SEQ_STRIDE + 檔內列號，作為「同秒保留檔內 id 順序」的穩定排序鍵。
  - BUY 手以 game_mode 位元 16 判定（位元遮罩不得用 = 比對）；實測付費手的 game_mode 只有 1、3、19，等同 game_mode = 19。
  - 含 FreeGame 的手以 game_mode 位元 2 判定 → is_free：3 = 一般手中 FreeGame、19 = BUY（必含）。
    是當手開獎結果，只能用來算「當手之前」的水池特徵。
"""
from __future__ import annotations

import hashlib
import logging
import platform
import resource
import shutil
import time
from dataclasses import asdict, dataclass
from datetime import date, datetime, time as dtime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl
import yaml

from feel_engine.paths import CONFIG_PATH, StorePaths, bet_records_dir

# ---------------------------------------------------------------------------
# 0.1 定義
# ---------------------------------------------------------------------------
TZ = "America/New_York"
ET = ZoneInfo(TZ)
OUTCOMES = ("NO_WIN", "TINY_WIN", "SMALL_WIN", "MEDIUM_WIN", "BIG_WIN")
TINY_UPPER = 1.0      # TINY_WIN：0 < x < 1
SMALL_UPPER = 7.0     # SMALL_WIN：1 ≤ x < 7
MEDIUM_UPPER = 30.0   # MEDIUM_WIN：7 ≤ x < 30；BIG_WIN：x ≥ 30
BUY_BIT = 16          # game_mode 位元：BUY
FREE_BIT = 2          # game_mode 位元：Free（含 FreeGame）
READ_COLUMNS = ("id", "member_account", "create_at", "bet_points", "prize", "points", "last_points",
                "game_version", "game_mode")
BUCKET_COLUMNS = ("member_account", "seq", "ts", "bet_points", "prize", "points", "last_points",
                  "game_version", "is_buy", "is_free")
SEQ_STRIDE = 1_000_000_000
R4_WARN_RATIO = 0.01

log = logging.getLogger("feel_engine.pipeline.data")


class DataCheckError(RuntimeError):
    """資料檢查不通過（R1、R7），必須中止。"""


# ---------------------------------------------------------------------------
# 共用工具
# ---------------------------------------------------------------------------
def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    d = cfg["data"]
    for old in ("analysis_date", "train_date", "valid_date"):
        if old in d:
            raise ValueError(f"config.yaml 的 data.{old} 已改為 train_start、train_end（訓練期間）與 valid_start、valid_end（驗證期間）")
    for old in ("bet_records_dir", "tag_dir"):
        if old in d:
            raise ValueError(f"config.yaml 的 data.{old} 已拿掉：快取資料夾改由 merchant／currency／game 決定（不含日期），見 feel_engine/paths.py")
    start, end, v_start, v_end = (date.fromisoformat(str(d[k])) for k in ("train_start", "train_end", "valid_start", "valid_end"))
    if not start <= end < v_start <= v_end:
        raise ValueError(f"須 train_start（{start}）≤ train_end（{end}）< valid_start（{v_start}）≤ valid_end（{v_end}）")
    for k in ("merchant", "currency", "game"):
        d[k] = str(d[k])
    return cfg


def _days(start, end) -> list[str]:
    s, e = date.fromisoformat(str(start)), date.fromisoformat(str(end))
    return [str(s + timedelta(days=i)) for i in range((e - s).days + 1)]


def train_valid_days(cfg: dict) -> tuple[list[str], list[str]]:
    """（訓練期間每一天, 驗證期間每一天），美東 YYYY-MM-DD。"""
    d = cfg["data"]
    return _days(d["train_start"], d["train_end"]), _days(d["valid_start"], d["valid_end"])


def store_key(cfg: dict) -> str:
    """{商戶}_{幣別}_{遊戲}：逐日 dataset（output/{key}/dataset/）與各 run 的上層資料夾。"""
    d = cfg["data"]
    return f"{d['merchant']}_{d['currency']}_{d['game']}"


def run_name(cfg: dict) -> str:
    """output／result 底下的資料夾：{商戶}_{幣別}_{遊戲}/train_{起}-{迄}_valid_{起}[-{迄}]（驗證只有一天時只寫一個日期）。"""
    d = cfg["data"]
    s, e, vs, ve = (str(d[k]).replace("-", "") for k in ("train_start", "train_end", "valid_start", "valid_end"))
    return f"{store_key(cfg)}/train_{s}-{e}_valid_{vs if vs == ve else f'{vs}-{ve}'}"


def contiguous_ranges(days: list[date]) -> list[tuple[date, date]]:
    """日期依連續區段分組：[(第一天, 最後一天), …]。"""
    out: list[tuple[date, date]] = []
    for d in sorted(set(days)):
        if out and (d - out[-1][1]).days == 1:
            out[-1] = (out[-1][0], d)
        else:
            out.append((d, d))
    return out


def et_midnight(d: date) -> datetime:
    """美東某日 0 點（tz-aware，夏令／冬令自動處理）。"""
    return datetime.combine(d, dtime.min, tzinfo=ET)


def lit_et(dt: datetime) -> pl.Expr:
    """tz-aware datetime → 美東時區的 polars 常數（避免不同時區比較）。"""
    utc_naive = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return pl.lit(utc_naive).cast(pl.Datetime("us")).dt.replace_time_zone("UTC").dt.convert_time_zone(TZ)


def utc_naive(dt: datetime) -> datetime:
    return dt.astimezone(timezone.utc).replace(tzinfo=None)


def peak_rss_mb() -> float:
    """本行程至今的峰值記憶體（MB）；macOS 單位為 bytes、Linux 為 KB。"""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / 1024 / 1024 if platform.system() == "Darwin" else rss / 1024


def outcome_expr(mult: pl.Expr) -> pl.Expr:
    """倍數區間（0.1，左閉右開）。"""
    return (
        pl.when(mult <= 0).then(pl.lit("NO_WIN"))
        .when(mult < TINY_UPPER).then(pl.lit("TINY_WIN"))
        .when(mult < SMALL_UPPER).then(pl.lit("SMALL_WIN"))
        .when(mult < MEDIUM_UPPER).then(pl.lit("MEDIUM_WIN"))
        .otherwise(pl.lit("BIG_WIN"))
    )


def bucket_of(accounts: pl.Series, n_buckets: int) -> pl.DataFrame:
    """member_account → 桶號。用 md5 而非 polars hash，跨版本、跨機器都穩定。"""
    uniq = accounts.unique()
    ids = [int.from_bytes(hashlib.md5(a.encode("utf-8")).digest()[:8], "little") % n_buckets for a in uniq.to_list()]
    return pl.DataFrame({"member_account": uniq, "bucket": pl.Series(ids, dtype=pl.Int32)})


def utc_days_overlapping(start: datetime, end: datetime) -> list[date]:
    """與 [start, end) 有重疊的 UTC 日期。"""
    s, e = start.astimezone(timezone.utc), end.astimezone(timezone.utc)
    if e <= s:
        return []
    last = (e - timedelta(microseconds=1)).date()
    return [s.date() + timedelta(days=i) for i in range((last - s.date()).days + 1)]


# ---------------------------------------------------------------------------
# R1：manifest 對帳
# ---------------------------------------------------------------------------
def check_manifest(bet_dir: Path, currency: str) -> dict[date, Path]:
    """R1：該幣別的 manifest 全部對帳一致、日期連續、檔案列數 = manifest。不符丟 DataCheckError。"""
    manifest_path = bet_dir / "_manifest.csv"
    if not manifest_path.exists():
        raise DataCheckError(f"找不到 {manifest_path}")
    manifest = pl.read_csv(manifest_path, try_parse_dates=True)
    manifest = manifest.filter(pl.col("currency").cast(pl.String) == currency)
    if manifest.height == 0:
        raise DataCheckError(f"R1 manifest 沒有幣別 {currency} 的資料")
    bad = manifest.filter(~pl.col("match").cast(pl.Boolean) | (pl.col("status") == "failed"))
    if bad.height:
        raise DataCheckError(f"R1 manifest 對帳不一致：\n{bad}")

    days = sorted(manifest["bet_date"].to_list())
    gaps = [d for d, nxt in zip(days, days[1:]) if (nxt - d).days != 1]
    if gaps:
        raise DataCheckError(f"R1 日期不連續，缺口在 {gaps} 之後")

    files = {}
    expected = dict(zip(manifest["bet_date"].to_list(), manifest["actual_rows"].to_list()))
    for d in days:
        matches = list(bet_dir.glob(f"currency={currency}/{d:%Y-%m-%d}.parquet"))
        if len(matches) != 1:
            raise DataCheckError(f"R1 {d} 日檔數量 = {len(matches)}（應為 1）")
        n = pl.scan_parquet(matches[0], hive_partitioning=False).select(pl.len()).collect().item()
        if n != expected[d]:
            raise DataCheckError(f"R1 {d} 檔案列數 {n} ≠ manifest {expected[d]}")
        files[d] = matches[0]
    return files


# ---------------------------------------------------------------------------
# R7：讀取範圍
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ReadPlan:
    first_day: date                # 美東；本次 build 的第一天
    last_day: date                 # 美東；本次 build 的最後一天
    range_start: datetime          # 美東 first_day 0 點 − pre_hours
    analysis_end: datetime         # 美東 last_day 24 點
    range_end: datetime            # analysis_end + post_hours（接完跨午夜的 session）+ return_window
    data_end: datetime             # min(range_end, 快取終點)；5.3 截尾用
    range_days: tuple[date, ...]   # 範圍內要讀的 UTC 日檔
    back_short: bool               # 後段不足（已截尾）

    def to_json(self) -> dict:
        out = {k: (v.isoformat() if isinstance(v, (date, datetime)) else v) for k, v in asdict(self).items()}
        out["range_days"] = [d.isoformat() for d in self.range_days]
        return out


def plan_read(first_day: date, last_day: date, pre_hours: float, return_hours: float, cache_days: list[date],
              post_hours: float = 0.0) -> ReadPlan:
    """R7：快取是否涵蓋讀取範圍。前段或分析日不足 → 中止；只有後段不足 → 警告並截尾。

    讀取範圍一次涵蓋 first_day 到 last_day（本次 build 的連續日子）；post_hours：起點在 last_day 的 session 可能跨過午夜，
    往後多讀這麼久再加 return_hours 才看得到它的 Y。
    """
    range_start = et_midnight(first_day) - timedelta(hours=pre_hours)
    analysis_end = et_midnight(last_day + timedelta(days=1))
    range_end = analysis_end + timedelta(hours=post_hours + return_hours)

    cache_start = datetime.combine(min(cache_days), dtime.min, tzinfo=timezone.utc)
    cache_end = datetime.combine(max(cache_days) + timedelta(days=1), dtime.min, tzinfo=timezone.utc)

    missing_front = utc_days_overlapping(range_start, cache_start)
    if missing_front:
        raise DataCheckError(f"R7 前段不足（讀取範圍起點早於快取），缺 UTC 日檔：{missing_front}")
    if cache_end < analysis_end:
        missing = utc_days_overlapping(cache_end, analysis_end)
        raise DataCheckError(f"R7 分析日 {last_day} 超出快取（快取到美東 {cache_end.astimezone(ET)}），缺 UTC 日檔：{missing}")
    back_short = cache_end < range_end
    data_end = min(range_end, cache_end).astimezone(ET)
    if back_short:
        log.warning("R7 後段不足：讀取範圍到 %s，快取只到 %s；缺 UTC 日檔 %s，將截尾",
                    range_end, data_end, utc_days_overlapping(cache_end, range_end))
    return ReadPlan(
        first_day=first_day,
        last_day=last_day,
        range_start=range_start,
        analysis_end=analysis_end,
        range_end=range_end,
        data_end=data_end,
        range_days=tuple(utc_days_overlapping(range_start, data_end)),
        back_short=back_short,
    )


# ---------------------------------------------------------------------------
# 單日處理、分桶
# ---------------------------------------------------------------------------
def pre_range_last_paid(paths: list[Path], start: datetime, end: datetime) -> pl.DataFrame:
    """[start, end) 內每位玩家最後一個付費手的時間（UTC naive）。

    讀取範圍從 start = 範圍起點 − session_gap 之後才逐筆讀；這段若有付費手，玩家在範圍內的第一個 session
    其實更早就開始了（起點是被讀取範圍切出來的），第二段據此不從該 session 抽決策點（SPEC 5.4）。
    """
    parts = [
        pl.read_parquet(p, columns=["member_account", "create_at", "bet_points"], hive_partitioning=False)
        .filter((pl.col("bet_points") > 0) & (pl.col("create_at") >= utc_naive(start)) & (pl.col("create_at") < utc_naive(end)))
        .group_by("member_account").agg(last_paid=pl.col("create_at").max())
        for p in paths
    ]
    if not parts:
        return pl.DataFrame(schema={"member_account": pl.String, "last_paid": pl.Datetime("ns")})
    return pl.concat(parts).group_by("member_account").agg(pl.col("last_paid").max()).sort("member_account")


def prepare_day(df: pl.DataFrame, day_ordinal: int, version_map: dict) -> tuple[pl.DataFrame, dict]:
    """R2 去重、R3、R4 統計，轉美東、加 seq、is_buy 與 is_free。回傳（桶欄位的 DataFrame, 統計）。"""
    n_raw = df.height
    df = df.with_row_index("row_idx")

    # R2：日檔內 id 重複 → 保留第一筆
    df = df.unique(subset="id", keep="first", maintain_order=True)
    n_dup = n_raw - df.height

    # R3：last_points = points − bet_points + prize
    r3_ok = df.select(
        (pl.col("last_points") == pl.col("points") - pl.col("bet_points") + pl.col("prize")).sum()
    ).item()

    # R4：付費手 game_version 為空或不在 version_map
    paid = df.filter(pl.col("bet_points") > 0)
    version_counts = paid.group_by("game_version").len().sort("game_version")
    n_unknown = paid.filter(
        (pl.col("game_version") == "") | pl.col("game_version").is_null()
        | ~pl.col("game_version").is_in(list(version_map))
    ).height

    mode = pl.col("game_mode").cast(pl.Int64, strict=False).fill_null(0)
    out = df.with_columns(
        seq=pl.lit(day_ordinal, dtype=pl.Int64) * SEQ_STRIDE + pl.col("row_idx").cast(pl.Int64),
        ts=pl.col("create_at").dt.replace_time_zone("UTC").dt.convert_time_zone(TZ),
        is_buy=(mode & BUY_BIT) > 0,
        is_free=(mode & FREE_BIT) > 0,
    ).select(BUCKET_COLUMNS)

    stats = {
        "rows": n_raw,
        "r2_dup_ids": n_dup,
        "r3_match": int(r3_ok),
        "rows_after_dedup": out.height,
        "paid_rows": paid.height,
        "buy_paid_rows": int(out.filter(pl.col("bet_points") > 0)["is_buy"].sum()),
        "free_paid_rows": int(out.filter(pl.col("bet_points") > 0)["is_free"].sum()),
        "r4_unknown": n_unknown,
        "version_counts": dict(zip(version_counts["game_version"].to_list(), version_counts["len"].to_list())),
    }
    return out, stats


def write_buckets(df: pl.DataFrame, n_buckets: int, out_dir: Path, day_label: str) -> None:
    """依 member_account 雜湊寫入各桶：out_dir/bucket=XX/{day_label}.parquet。"""
    df = df.join(bucket_of(df["member_account"], n_buckets), on="member_account", how="left")
    for (bucket,), part in df.partition_by("bucket", as_dict=True).items():
        path = out_dir / f"bucket={bucket:02d}" / f"{day_label}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        part.select(BUCKET_COLUMNS).write_parquet(path, compression="zstd")


# ---------------------------------------------------------------------------
# 第一段主流程
# ---------------------------------------------------------------------------
def run_stage1(cfg: dict, first_day: date, last_day: date, store: StorePaths) -> dict:
    """執行第一段：讀 first_day～last_day（美東，連續）所需的日檔，分桶寫入 store 的暫存資料夾，回傳 meta。
    R1、R7 不符時丟 DataCheckError。"""
    t0 = time.time()
    bet_dir = bet_records_dir(cfg["data"]["merchant"], cfg["data"]["game"])
    n_buckets = int(cfg["compute"]["n_buckets"])
    version_map = cfg["modules"]["version_map"]

    files = check_manifest(bet_dir, cfg["data"]["currency"])
    cache_days = sorted(files)
    log.info("R1 通過：%d 個日檔 %s ～ %s", len(files), cache_days[0], cache_days[-1])

    plan = plan_read(
        first_day,
        last_day,
        float(cfg["data"]["pre_hours"]),
        float(cfg["labels"]["return_window_hours"]),
        cache_days,
        post_hours=float(cfg["data"].get("post_hours", 0)),
    )
    log.info("R7 通過：讀取範圍美東 %s 至 %s（UTC 日檔 %s）",
             plan.range_start, plan.data_end, [str(d) for d in plan.range_days])

    if store.tmp_dir.exists():
        shutil.rmtree(store.tmp_dir)  # 只清本程式自己產生的暫存（上次中斷殘留的分桶）
    store.buckets.mkdir(parents=True)

    gap = timedelta(minutes=float(cfg["labels"]["session_gap_minutes"]))
    pre_days = utc_days_overlapping(plan.range_start - gap, plan.range_start)
    pre = pre_range_last_paid([files[d] for d in pre_days if d in files], plan.range_start - gap, plan.range_start)
    pre.write_parquet(store.pre_range_last, compression="zstd")
    log.info("讀取範圍起點前 %s 分鐘內有付費手：%s 位玩家（範圍內第一手與它間隔不超過此值者，第一個 session 起點不完整）",
             f"{gap.total_seconds() / 60:.0f}", f"{pre.height:,}")

    per_day = []
    for d in plan.range_days:
        day_t = time.time()
        df = (pl.read_parquet(files[d], columns=list(READ_COLUMNS), hive_partitioning=False)
              .filter((pl.col("create_at") >= utc_naive(plan.range_start)) & (pl.col("create_at") < utc_naive(plan.data_end))))
        out, stats = prepare_day(df, (d - cache_days[0]).days, version_map)
        write_buckets(out, n_buckets, store.buckets, f"{d:%Y-%m-%d}")
        stats.update(date=str(d), seconds=round(time.time() - day_t, 1))
        per_day.append(stats)
        log.info("UTC %s 範圍內 rows=%s dup=%s R4 unknown=%s（%.1fs）",
                 d, f"{stats['rows']:,}", stats["r2_dup_ids"], stats["r4_unknown"], stats["seconds"])

    version_counts: dict[str, int] = {}
    for s in per_day:
        for k, v in s["version_counts"].items():
            version_counts[k] = version_counts.get(k, 0) + v

    used = sorted({*pre_days, *plan.range_days} & set(files))
    meta = {
        "first_day": str(first_day), "last_day": str(last_day),
        "merchant": cfg["data"]["merchant"], "currency": cfg["data"]["currency"], "game": cfg["data"]["game"],
        "bet_records_dir": bet_dir.name,
        "n_buckets": n_buckets,
        "cache_start_utc": str(cache_days[0]),
        "cache_end_utc": str(cache_days[-1] + timedelta(days=1)),
        "raw_rows": raw_rows(bet_dir, cfg["data"]["currency"], used),   # 用到的日檔列數；之後快取重抓時據此判斷 dataset 過期
        "read_plan": plan.to_json(),
        "pre_range_players": pre.height,
        "days": per_day,
        "version_counts": dict(sorted(version_counts.items())),
        "seconds": round(time.time() - t0, 1),
        "peak_rss_mb": round(peak_rss_mb(), 1),
    }
    log.info("第一段完成：%.1fs，峰值記憶體 %.0f MB", meta["seconds"], meta["peak_rss_mb"])
    return meta


def raw_rows(bet_dir: Path, currency: str, days: list[date]) -> dict[str, int]:
    """manifest 記載的各 UTC 日檔列數（只取 days）。"""
    manifest = pl.read_csv(bet_dir / "_manifest.csv", try_parse_dates=True).filter(pl.col("currency").cast(pl.String) == currency)
    rows = dict(zip(manifest["bet_date"].to_list(), manifest["actual_rows"].to_list()))
    return {str(d): int(rows[d]) for d in days if d in rows}


def version_map_missing(version_counts: dict[str, int], version_map: dict) -> list[str]:
    """付費手出現過、但 version_map 沒有對應（不存在或值為 null）的 game_version。"""
    return sorted(v for v in version_counts if v and version_map.get(v) is None)

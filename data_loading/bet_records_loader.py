# -*- coding: utf-8 -*-
"""
抓取 tableau.bet_records 單筆注單明細，落地 parquet 以利重現。

預設任務：2026-08-01 ～ 2026-09-29（UTC，半開區間 [08-01, 09-30)）× agent_code = 'TCG' × game_id = '22071' × currency = 'PHP'。
快取資料夾不含日期：同一個 agent_code × game_id × 時區只有一份，延伸期間時給新的 --start／--end，
已在快取且列數相符的日子跳過、只抓缺的，_manifest.csv 併入（不同期間不再各存一份）。

流程：
  1. 規劃：直接對 bet_records 做「幣別 × 小時」count()，決定每次查詢切多大時段。
     ⚠️ 不用 bet_record_pre_aggregations 規劃：實測 pre_agg 會整小時缺漏或少算
     （例：2026-08-01 UTC 18–23 時整段沒有、08-14 缺 1 小時），用它規劃會讓那些
     小時從頭到尾沒被查詢、靜默漏抓，且「抓回列數 = pre_agg」也照樣對得上。
     pre_agg 只拿來對照，差異寫進 _manifest.csv 的 pre_agg_rows。
  2. 抓取：每個「幣別 × 日期」寫一個 parquet；單日注數過大時依小時切成多段查詢
     （每段不超過 --max-rows 筆），用 ParquetWriter 逐段寫入，避免一次吃爆記憶體。
  3. 對帳：每個檔案的實際列數與 bet_records count() 比對，結果併入 _manifest.csv
     （本次區間內的日子以新結果取代，區間外的列保留）；不一致只警告、不刪檔。
  4. 快取：檔案已存在、欄位齊全且列數 = 應有列數才跳過（USE_QUERY_CACHE），
     中斷後重跑會從斷點續抓；舊版快取缺欄位（REQUIRED_COLUMNS）或列數不符時自動重抓。

game_version：
  取自 wagers_detail 最上層的 gameVersion 鍵，在 SQL 端用 JSONExtractString 抽出，
  不把整個 wagers_detail 抓下來。實測值：付費手 22071_1.0.0_{0,1,2,3,5}、
  免費局 22071_card_1.0.0_4；Jackpot 紀錄沒有此鍵 → 空字串。

輸出（feel_engine/paths.py 的 bet_records_dir 用同一個名稱）：
  data/bet_records/bet_records_{AG}_{GAME}_tz{OFFSET}/
      currency={CUR}/{YYYY-MM-DD}.parquet
      _manifest.csv

用法：
  python data_loading/bet_records_loader.py              # 預設全量（在專案根目錄執行；已抓過的日子跳過）
  python data_loading/bet_records_loader.py --plan-only  # 只看規劃與預估筆數，不抓明細
  python data_loading/bet_records_loader.py --start 2026-09-26 --end 2026-09-30   # 只補新的日子
  python data_loading/bet_records_loader.py --include-wagers-detail  # 連同最寬欄位 wagers_detail 一起抓

讀回：
  from data_loading.bet_records_loader import load_bet_records
  df = load_bet_records(start="2026-09-21", end="2026-09-22", columns=["member_account", "bet_points", "create_at"])
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:  # 直接執行本檔時也找得到根目錄的 db_manager
    sys.path.insert(0, str(PROJECT_ROOT))

from db_manager import DatabaseManager  # noqa: E402

# ---------------------------------------------------------------------------
# 設定
# ---------------------------------------------------------------------------
AGENT_CODE = "TCG"          # 精確比對；TCGS2 是另一個 agent_code，需要時另跑
GAME_ID = "22071"           # game_id 是 String
CURRENCIES = ("PHP",)       # 必濾；tuple 會展開成 IN (...)
START_DATE = "2026-08-01"   # 半開區間 [START_DATE, END_DATE)，指「位移後」的日期
END_DATE = "2026-09-30"
TZ_OFFSET_HOURS = 0         # 0 = UTC；位移只在 Python 端換算邊界與 SELECT 的 bet_date
MAX_ROWS_PER_QUERY = 500_000
REQUIRED_COLUMNS = ("game_version",)  # 快取檔缺這些欄位就視為過期、重抓
USE_QUERY_CACHE = True
CACHE_ROOT = PROJECT_ROOT / "data" / "bet_records"

log = logging.getLogger("data_loader")


# ---------------------------------------------------------------------------
# SQL（全部為靜態字串 + %(name)s 參數化）
# ---------------------------------------------------------------------------
# 規劃用：bet_records 本身的每小時列數（權威來源，見檔首說明）
PLAN_SQL = """
SELECT
    currency,
    toStartOfHour(create_at) AS start_at,
    count() AS expected_rows
FROM bet_records
WHERE agent_code = %(agent_code)s
  AND currency IN %(currencies)s
  AND game_id = %(game_id)s
  AND create_at >= %(start_utc)s
  AND create_at < %(end_utc)s
GROUP BY currency, start_at
ORDER BY currency, start_at
SETTINGS max_bytes_before_external_group_by = 4000000000, max_bytes_before_external_sort = 4000000000
"""

# 對照用：pre_agg 每小時注數（可能缺漏，只寫進 manifest 供參考）
PRE_AGG_SQL = """
SELECT
    currency,
    start_at,
    sum(game_count) AS pre_agg_rows
FROM bet_record_pre_aggregations
WHERE agent_code = %(agent_code)s
  AND currency IN %(currencies)s
  AND game_id = %(game_id)s
  AND start_at >= %(start_utc)s
  AND start_at < %(end_utc)s
GROUP BY currency, start_at
ORDER BY currency, start_at
SETTINGS max_bytes_before_external_group_by = 4000000000, max_bytes_before_external_sort = 4000000000
"""

# wagers_detail 是全表最寬欄位，預設不抓，只在 SQL 端抽出 game_version
DETAIL_SQL = """
SELECT
    id, agent_code, agent_account, member_account, game_id, game_mode,
    prize, remote, sub_code_name, points, bet_points, win_points,
    last_points, currency, event, create_at, jackpot_bet, jackpot_prize,
    valid_bet, refund, commission,
    JSONExtractString(wagers_detail, 'gameVersion') AS game_version,
    toDate(create_at + toIntervalHour(%(tz_offset)s)) AS bet_date
FROM bet_records
WHERE agent_code = %(agent_code)s
  AND currency = %(currency)s
  AND game_id = %(game_id)s
  AND create_at >= %(start_utc)s
  AND create_at < %(end_utc)s
ORDER BY create_at, id
SETTINGS max_bytes_before_external_group_by = 4000000000, max_bytes_before_external_sort = 4000000000
"""

# 同 DETAIL_SQL，另含 wagers_detail（--include-wagers-detail）
DETAIL_WITH_WAGERS_SQL = """
SELECT
    id, agent_code, agent_account, member_account, game_id, game_mode,
    prize, remote, sub_code_name, points, bet_points, win_points,
    last_points, currency, event, create_at, wagers_detail, jackpot_bet, jackpot_prize,
    valid_bet, refund, commission,
    JSONExtractString(wagers_detail, 'gameVersion') AS game_version,
    toDate(create_at + toIntervalHour(%(tz_offset)s)) AS bet_date
FROM bet_records
WHERE agent_code = %(agent_code)s
  AND currency = %(currency)s
  AND game_id = %(game_id)s
  AND create_at >= %(start_utc)s
  AND create_at < %(end_utc)s
ORDER BY create_at, id
SETTINGS max_bytes_before_external_group_by = 4000000000, max_bytes_before_external_sort = 4000000000
"""


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------
def local_to_utc(date_str: str, tz_offset: int) -> datetime:
    """位移後日期的 00:00 → UTC naive datetime（WHERE 只放裸欄位 + UTC 常數）。"""
    return datetime.strptime(date_str, "%Y-%m-%d") - timedelta(hours=tz_offset)


def cache_dir(agent_code: str, game_id: str, tz_offset: int) -> Path:
    """不含日期：同一個 agent_code × game_id × 時區只有一個快取資料夾，逐日檔累積在裡面。"""
    return CACHE_ROOT / f"bet_records_{agent_code}_{game_id}_tz{tz_offset:+d}"


def merge_manifest(path: Path, new_rows: pd.DataFrame) -> pd.DataFrame:
    """既有 manifest 中與本次相同（currency, bet_date）的列以本次結果取代，其餘保留。"""
    if path.exists():
        old = pd.read_csv(path, dtype={"currency": str})
        old["bet_date"] = pd.to_datetime(old["bet_date"]).dt.date
        keys = set(zip(new_rows["currency"], new_rows["bet_date"]))
        old = old[[k not in keys for k in zip(old["currency"], old["bet_date"])]]
        new_rows = pd.concat([old, new_rows], ignore_index=True)
    return new_rows.sort_values(["currency", "bet_date"], ignore_index=True)


def build_plan(db: DatabaseManager, params: dict, tz_offset: int, max_rows: int) -> pd.DataFrame:
    """回傳每個查詢切片一列：currency / bet_date / slice_id / slice_start / slice_end /
    expected_rows（bet_records count）/ pre_agg_rows（對照用）。"""
    hourly = db.fetch_query_with_params(PLAN_SQL, params)
    if hourly.empty:
        return hourly

    hourly["start_at"] = pd.to_datetime(hourly["start_at"])
    pre_agg = db.fetch_query_with_params(PRE_AGG_SQL, params)
    pre_agg["start_at"] = pd.to_datetime(pre_agg["start_at"])
    hourly = hourly.merge(pre_agg, on=["currency", "start_at"], how="left")
    hourly["pre_agg_rows"] = hourly["pre_agg_rows"].fillna(0).astype("int64")
    hourly["expected_rows"] = hourly["expected_rows"].astype("int64")
    hourly["bet_date"] = (hourly["start_at"] + pd.Timedelta(hours=tz_offset)).dt.normalize()
    hourly = hourly.sort_values(["currency", "start_at"], ignore_index=True)

    # 同一「幣別 × 日期」內依小時累積注數，每 max_rows 筆切一段（以小時為最小單位）
    rows_before = hourly.groupby(["currency", "bet_date"])["expected_rows"].cumsum() - hourly["expected_rows"]
    hourly["slice_id"] = rows_before // max_rows

    plan = (
        hourly.groupby(["currency", "bet_date", "slice_id"], as_index=False)
        .agg(slice_start=("start_at", "min"), last_hour=("start_at", "max"),
             expected_rows=("expected_rows", "sum"), pre_agg_rows=("pre_agg_rows", "sum"))
    )
    plan["slice_end"] = plan["last_hour"] + pd.Timedelta(hours=1)
    plan = plan[["currency", "bet_date", "slice_id", "slice_start", "slice_end", "expected_rows", "pre_agg_rows"]]
    return plan.sort_values(["currency", "bet_date", "slice_id"], ignore_index=True)


def cast_types(df: pd.DataFrame) -> pd.DataFrame:
    """型態對齊：create_at 為 UTC naive datetime、bet_date 為日期。"""
    df["create_at"] = pd.to_datetime(df["create_at"])
    df["bet_date"] = pd.to_datetime(df["bet_date"]).dt.date
    return df


def fetch_one_file(
    db: DatabaseManager,
    detail_sql: str,
    slices: pd.DataFrame,
    base_params: dict,
    out_path: Path,
) -> int:
    """抓一個「幣別 × 日期」的所有切片，逐段寫入同一個 parquet，回傳實際列數。"""
    tmp_path = out_path.with_suffix(".parquet.tmp")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer: pq.ParquetWriter | None = None
    schema: pa.Schema | None = None
    total = 0
    try:
        for s in slices.itertuples(index=False):
            params = {
                **base_params,
                "currency": s.currency,
                "start_utc": s.slice_start.to_pydatetime(),
                "end_utc": s.slice_end.to_pydatetime(),
            }
            df = cast_types(db.fetch_query_with_params(detail_sql, params))
            if df.empty:
                continue
            table = pa.Table.from_pandas(df, schema=schema, preserve_index=False)
            if writer is None:
                schema = table.schema
                writer = pq.ParquetWriter(tmp_path, schema, compression="zstd")
            writer.write_table(table)
            total += len(df)
        if writer is not None:
            writer.close()
            writer = None
            tmp_path.replace(out_path)
        return total
    except Exception:
        if writer is not None:
            writer.close()
        tmp_path.unlink(missing_ok=True)
        raise


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def run(args: argparse.Namespace) -> int:
    tz = args.tz_offset
    start_utc = local_to_utc(args.start, tz)
    end_utc = local_to_utc(args.end, tz)
    out_dir = cache_dir(args.agent_code, args.game_id, tz)
    detail_sql = DETAIL_WITH_WAGERS_SQL if args.include_wagers_detail else DETAIL_SQL

    try:
        db = DatabaseManager("tableau")
    except Exception:
        log.exception("ClickHouse 連線失敗，請確認 .env 的 CH_HOST / CH_USER / CH_PASSWORD")
        return 1

    base_params = {
        "agent_code": args.agent_code,
        "game_id": args.game_id,
        "currencies": tuple(args.currencies),
        "tz_offset": tz,
        "start_utc": start_utc,
        "end_utc": end_utc,
    }
    log.info("區間 [%s, %s) tz%+d → UTC [%s, %s)", args.start, args.end, tz, start_utc, end_utc)

    plan = build_plan(db, base_params, tz, args.max_rows)
    if plan.empty:
        log.warning("bet_records 查無資料：agent_code=%s currency=%s game_id=%s", args.agent_code, args.currencies, args.game_id)
        return 0

    summary = (
        plan.groupby("currency", as_index=False)
        .agg(days=("bet_date", "nunique"), queries=("slice_id", "size"), expected_rows=("expected_rows", "sum"))
        .sort_values("expected_rows", ascending=False)
    )
    log.info("規劃（%d 個查詢，預估 %s 筆）：\n%s",
             len(plan), f"{summary['expected_rows'].sum():,}", summary.to_string(index=False))
    if args.plan_only:
        return 0

    manifest_rows = []
    failures = 0
    files = plan.groupby(["currency", "bet_date"], sort=True)
    for n, ((currency, bet_date), slices) in enumerate(files, start=1):
        out_path = out_dir / f"currency={currency}" / f"{bet_date:%Y-%m-%d}.parquet"
        expected = int(slices["expected_rows"].sum())
        pre_agg_rows = int(slices["pre_agg_rows"].sum())

        if (USE_QUERY_CACHE and not args.no_cache and out_path.exists()
                and set(REQUIRED_COLUMNS) <= set(pq.read_schema(out_path).names)
                and pq.ParquetFile(out_path).metadata.num_rows == expected):
            actual = pq.ParquetFile(out_path).metadata.num_rows
            status = "cached"
        else:
            try:
                actual = fetch_one_file(db, detail_sql, slices, base_params, out_path)
                status = "fetched"
            except Exception:
                log.exception("抓取失敗 %s %s（重跑會自動續抓）", currency, bet_date.date())
                failures += 1
                manifest_rows.append((currency, bet_date.date(), expected, None, pre_agg_rows, "failed"))
                continue

        if actual != expected:
            log.warning("對帳不符 %s %s：抓回=%s bet_records 應有=%s", currency, bet_date.date(), actual, expected)
        if pre_agg_rows != expected:
            log.warning("pre_agg 與 bet_records 不一致 %s %s：pre_agg=%s bet_records=%s（僅記錄）",
                        currency, bet_date.date(), pre_agg_rows, expected)
        log.info("[%d/%d] %s %s %s rows=%s", n, files.ngroups, status, currency, bet_date.date(), f"{actual:,}")
        manifest_rows.append((currency, bet_date.date(), expected, actual, pre_agg_rows, status))

    manifest = pd.DataFrame(
        manifest_rows,
        columns=["currency", "bet_date", "expected_rows", "actual_rows", "pre_agg_rows", "status"],
    )
    manifest["match"] = manifest["expected_rows"] == manifest["actual_rows"]
    out_dir.mkdir(parents=True, exist_ok=True)
    merged = merge_manifest(out_dir / "_manifest.csv", manifest)
    merged.to_csv(out_dir / "_manifest.csv", index=False)

    log.info("完成：本次 %d 檔（新抓 %d），失敗 %d，對帳不符 %d；快取共 %d 個日檔 %s ～ %s，輸出 %s",
             len(manifest), int((manifest["status"] == "fetched").sum()), failures, int((~manifest["match"]).sum()),
             len(merged), merged["bet_date"].min(), merged["bet_date"].max(), out_dir)
    return 1 if failures else 0


def load_bet_records(
    agent_code: str = AGENT_CODE,
    game_id: str = GAME_ID,
    start: str = START_DATE,
    end: str = END_DATE,
    tz_offset: int = TZ_OFFSET_HOURS,
    currencies: tuple[str, ...] | None = CURRENCIES,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """讀回已落地的 parquet（日檔落在 [start, end) 者），依 currency、create_at、id 排序。

    PHP 整月約 2 億筆，務必用 start／end、currencies／columns 縮小範圍再讀進記憶體。
    """
    root = cache_dir(agent_code, game_id, tz_offset)
    # 檔內已有 currency 欄位，不用 hive partitioning（會與資料夾名稱推出的欄位型態衝突）
    paths = sorted(
        p for p in root.glob("currency=*/*.parquet")
        if (not currencies or p.parent.name.split("=", 1)[1] in currencies) and start <= p.stem < end
    )
    if not paths:
        raise FileNotFoundError(f"{root} 下沒有符合的 parquet，請先執行 python data_loading/bet_records_loader.py")
    read_cols = None
    if columns is not None:
        read_cols = list(dict.fromkeys([*columns, "currency", "create_at", "id"]))
    df = pq.read_table([str(p) for p in paths], columns=read_cols, partitioning=None).to_pandas()
    df["create_at"] = pd.to_datetime(df["create_at"])
    return df.sort_values(["currency", "create_at", "id"], ignore_index=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="抓取 bet_records 單筆明細並落地 parquet")
    p.add_argument("--agent-code", default=AGENT_CODE)
    p.add_argument("--game-id", default=GAME_ID)
    p.add_argument("--start", default=START_DATE, help="含，YYYY-MM-DD（位移後日期）")
    p.add_argument("--end", default=END_DATE, help="不含，YYYY-MM-DD（位移後日期）")
    p.add_argument("--tz-offset", type=int, default=TZ_OFFSET_HOURS, help="小時，預設 0 = UTC")
    p.add_argument("--currencies", nargs="+", default=list(CURRENCIES), help="幣別，預設 PHP")
    p.add_argument("--max-rows", type=int, default=MAX_ROWS_PER_QUERY, help="單次查詢筆數上限")
    p.add_argument("--include-wagers-detail", action="store_true", help="一併抓 wagers_detail（很寬）")
    p.add_argument("--no-cache", action="store_true", help="忽略既有 parquet，重新抓取")
    p.add_argument("--plan-only", action="store_true", help="只列規劃與預估筆數")
    return p.parse_args(argv)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    sys.exit(run(parse_args()))

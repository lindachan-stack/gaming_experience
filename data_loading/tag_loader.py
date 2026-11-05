# -*- coding: utf-8 -*-
"""
抓取 for_math_aggregations.dwt_player_game_tag_wi_test（玩家 × 遊戲 × 週 標籤表），
落地 parquet 以利重現；--start / --end 給的是「注單」日期，與 bet_records_loader.py 相同。

預設任務：agent_code = 'TCG' × currency = 'PHP' × game_id = '22071'，
注單日期區間 [2026-05-11, 2026-09-30)。
快取資料夾不含日期：同一個商戶 × 幣別 × 遊戲 × 延遲週數只有一份，延伸期間時給新的 --start／--end，
已在快取且與 DB 一致的注單週跳過、只抓缺的，_manifest.csv 併入（不同期間不再各存一份）。

對應規則（延遲一週＋往回找最新一筆，TAG_LAG_WEEKS = 1）：
  標籤表一週一列（start_date = 週一，week_index = ISO 年週，如 202634）。
  對區間內每個注單週 W（週一），每位玩家取 start_date ≤ W − 1 週 的標籤中「日期最新」的那一筆：
    - 前一週（W−1）有列的玩家 → 就是 W−1 那一筆（含補零列 is_zero_fill = 1）；
    - 前一週沒有列的玩家（例：久未回來）→ 往回取他最後一筆標籤；
    - 從未出現在標籤表的玩家（新玩家）→ 沒有列，合併後為「無標籤」。
  不會用到 W−1 週之後的標籤，延遲一週的規則不變。
  輸出欄位：
    apply_week_start  套用的注單週週一（合併鍵：member_account、game_id、注單週週一 = apply_week_start）
    tag_age_weeks     (apply_week_start − start_date) ÷ 7；1 = 前一週，> 1 = 更早的舊標籤
  每個注單週一個檔案，約 110～200 萬位玩家（3 月以來出現過的都會有一筆）。

版本規則（不管 tag_version）：
  同一 (agent_code, currency, game_id, member_account, week_index) 會因重跑 ETL
  而有多個 tag_version 並存。本程式只用商戶、貨幣、遊戲篩選，以 FINAL 取每個 key
  write_seq 最大（最後寫入）的那一筆，不指定、也不比對 tag_version。
  實際取到的 tag_version 與 etl_time 記在 _manifest.csv。

  ⚠️ 「最新」會隨 ETL 重跑而變：本機快取與 DB 當下結果（筆數、最後 etl_time）不同時會重抓；
     要強制重抓請加 --no-cache。

輸出（feel_engine/paths.py 的 tag_dir 用同一個名稱）：
  data/tags/dwt_player_game_tag_wi_test_{AG}_{CUR}_{GAME}_lag{N}w_asof/
      apply_week={YYYY-MM-DD}.parquet
      _manifest.csv（本次區間內的注單週以新結果取代，其餘保留）

用法（在專案根目錄執行）：
  python data_loading/tag_loader.py
  python data_loading/tag_loader.py --plan-only      # 只列每個注單週的預估玩家數
  python data_loading/tag_loader.py --no-cache       # 重新抓 DB 當下最新資料
  python data_loading/tag_loader.py --start 2026-09-28 --end 2026-10-05   # 只補新的注單週

讀回：
  from data_loading.tag_loader import load_tags
  tags = load_tags(start="2026-09-21", end="2026-09-28")
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:  # 直接執行本檔時也找得到根目錄的 db_manager
    sys.path.insert(0, str(PROJECT_ROOT))

from db_manager import DatabaseManager  # noqa: E402

# ---------------------------------------------------------------------------
# 設定
# ---------------------------------------------------------------------------
DATABASE = "for_math_aggregations"
TABLE = "dwt_player_game_tag_wi_test"
AGENT_CODE = "TCG"
CURRENCY = "PHP"
GAME_ID = "22071"            # game_id 是 String
START_DATE = "2026-05-11"    # 與 bet_records_loader.py 相同的半開區間 [START_DATE, END_DATE)
END_DATE = "2026-09-30"
TAG_LAG_WEEKS = 1            # 注單週 W 使用 start_date ≤ W − TAG_LAG_WEEKS 週的最新標籤
N_SLICES = 4                 # 每個注單週依 member_account 雜湊分批抓，控制記憶體
USE_QUERY_CACHE = True
CACHE_ROOT = PROJECT_ROOT / "data" / "tags"

log = logging.getLogger("tag_loader")


# ---------------------------------------------------------------------------
# SQL（全部為靜態字串 + %(name)s 參數化；start_date 為分區鍵來源，WHERE 只放裸欄位）
# ---------------------------------------------------------------------------
# 每個注單週的預期結果：玩家數與所選列的最後 etl_time（快取是否仍為 DB 最新）
PLAN_SQL = """
SELECT
    count() AS n_players,
    max(etl_time) AS max_etl_time,
    countIf(start_date = %(cutoff)s) AS n_cutoff_week
FROM
(
    SELECT member_account, start_date, etl_time
    FROM dwt_player_game_tag_wi_test FINAL
    WHERE agent_code = %(agent_code)s
      AND currency = %(currency)s
      AND game_id = %(game_id)s
      AND start_date <= %(cutoff)s
    ORDER BY member_account, start_date DESC
    LIMIT 1 BY member_account
)
SETTINGS max_bytes_before_external_group_by = 4000000000, max_bytes_before_external_sort = 4000000000
"""

DETAIL_SQL = """
SELECT
    tag_version, etl_time, write_seq,
    agent_code, currency, member_account, game_id, week_index, start_date,
    is_zero_fill, play_days,
    bet_points_total, bet_counts_total, BF_bet_points_ratio, in_out_wallet_amt_mean,
    bet_amt_cv, bet_amt_peak_ratio, volatility_score, chase_ratio, high_mult_ratio,
    bet_intensity_ratio, loyalty_score, similarity_score,
    value_combined_score, similarity_combined_score,
    bet_points_tag, avg_bet_points_tag, bet_counts_tag, fund_level_tag, buy_feature_tag,
    strategy_tag, return_pref_tag, emotion_tag, bet_intensity_tag, risk_pref_tag,
    loyalty_tag, similarity_score_tag, lifecycle_tag, value_combined_tag,
    similarity_combined_tag, need_prev_tag_fill
FROM dwt_player_game_tag_wi_test FINAL
WHERE agent_code = %(agent_code)s
  AND currency = %(currency)s
  AND game_id = %(game_id)s
  AND start_date <= %(cutoff)s
  AND modulo(cityHash64(member_account), %(n_slices)s) = %(slice)s
ORDER BY member_account, start_date DESC
LIMIT 1 BY member_account
SETTINGS max_bytes_before_external_group_by = 4000000000, max_bytes_before_external_sort = 4000000000
"""


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------
def bet_weeks(start: str, end: str) -> list[date]:
    """注單區間 [start, end) 涵蓋的各週週一。"""
    d = datetime.strptime(start, "%Y-%m-%d").date()
    first = d - timedelta(days=d.weekday())
    end_date = datetime.strptime(end, "%Y-%m-%d").date()
    n_weeks = -(-(end_date - first).days // 7)   # 無條件進位：最後一週只要有一天落在區間內就算
    return [first + timedelta(weeks=i) for i in range(n_weeks)]


def cache_dir(agent_code: str, currency: str, game_id: str, lag_weeks: int) -> Path:
    """不含日期：同一個商戶 × 幣別 × 遊戲 × 延遲週數只有一個快取資料夾，注單週檔累積在裡面。"""
    return CACHE_ROOT / f"{TABLE}_{agent_code}_{currency}_{game_id}_lag{lag_weeks}w_asof"


def merge_manifest(path: Path, new_rows: pd.DataFrame) -> pd.DataFrame:
    """既有 manifest 中與本次相同 apply_week_start 的列以本次結果取代，其餘保留。"""
    if path.exists():
        old = pd.read_csv(path)
        old["apply_week_start"] = pd.to_datetime(old["apply_week_start"]).dt.date
        old = old[~old["apply_week_start"].isin(set(new_rows["apply_week_start"]))]
        new_rows = pd.concat([old, new_rows], ignore_index=True)
    return new_rows.sort_values("apply_week_start", ignore_index=True)


def cast_types(df: pd.DataFrame, apply_week: date) -> pd.DataFrame:
    """型態對齊：etl_time 為 UTC naive datetime、start_date 為日期；補上 apply_week_start、tag_age_weeks。"""
    df["etl_time"] = pd.to_datetime(df["etl_time"])
    start = pd.to_datetime(df["start_date"])
    df["start_date"] = start.dt.date
    df["apply_week_start"] = apply_week
    df["tag_age_weeks"] = ((pd.Timestamp(apply_week) - start).dt.days // 7).astype("int16")
    return df


def fetch_week(db: DatabaseManager, params: dict, apply_week: date, n_slices: int) -> pa.Table:
    """分 n_slices 批抓一個注單週的 as-of 標籤，合併成一張 arrow 表（依 member_account 排序）。"""
    tables = []
    for s in range(n_slices):
        df = db.fetch_query_with_params(DETAIL_SQL, {**params, "n_slices": n_slices, "slice": s})
        tables.append(pa.Table.from_pandas(cast_types(df, apply_week), preserve_index=False))
    # 某批若整欄皆空會推成 null 型別，promote 成其他批的型別後再合併
    table = pa.concat_tables(tables, promote_options="default")
    return table.sort_by("member_account")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def run(args: argparse.Namespace) -> int:
    try:
        db = DatabaseManager(DATABASE)
    except Exception:
        log.exception("ClickHouse 連線失敗，請確認 .env 的 CH_HOST / CH_USER / CH_PASSWORD")
        return 1

    base_params = {"agent_code": args.agent_code, "currency": args.currency, "game_id": args.game_id}
    weeks = bet_weeks(args.start, args.end)
    plan_rows = []
    for w in weeks:
        cutoff = w - timedelta(weeks=args.lag_weeks)
        r = db.fetch_query_with_params(PLAN_SQL, {**base_params, "cutoff": cutoff}).iloc[0]
        plan_rows.append({"apply_week_start": w, "cutoff_week": cutoff, "n_players": int(r["n_players"]),
                          "n_cutoff_week": int(r["n_cutoff_week"]), "max_etl_time": pd.Timestamp(r["max_etl_time"])})
    plan = pd.DataFrame(plan_rows)
    log.info("每個注單週取 start_date ≤ 截止週 的最新一筆（延遲 %d 週，FINAL、不管 tag_version）：\n%s",
             args.lag_weeks, plan.to_string(index=False))
    if args.plan_only:
        return 0

    out_dir = cache_dir(args.agent_code, args.currency, args.game_id, args.lag_weeks)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    failures = 0
    for w in plan.itertuples(index=False):
        out_path = out_dir / f"apply_week={w.apply_week_start:%Y-%m-%d}.parquet"
        cached_ok = False
        if USE_QUERY_CACHE and not args.no_cache and out_path.exists():
            cached = pq.read_table(out_path, columns=["etl_time"])
            cached_ok = (cached.num_rows == w.n_players
                         and pd.Timestamp(pc.max(cached["etl_time"]).as_py()) == w.max_etl_time)
            if not cached_ok:
                log.warning("apply_week=%s 本機快取與 DB 當下最新結果不同，重抓", w.apply_week_start)
        try:
            if cached_ok:
                table = pq.read_table(out_path)
                status = "cached"
            else:
                params = {**base_params, "cutoff": w.cutoff_week}
                table = fetch_week(db, params, w.apply_week_start, args.slices)
                tmp_path = out_path.with_suffix(".parquet.tmp")
                pq.write_table(table, tmp_path, compression="zstd")
                tmp_path.replace(out_path)
                status = "fetched"
        except Exception:
            log.exception("抓取失敗 apply_week=%s（重跑會自動續抓）", w.apply_week_start)
            failures += 1
            manifest_rows.append({"apply_week_start": w.apply_week_start, "cutoff_week": w.cutoff_week,
                                  "expected_players": w.n_players, "status": "failed"})
            continue

        df = table.select(["member_account", "tag_age_weeks", "is_zero_fill", "tag_version"]).to_pandas()
        dup = int(df["member_account"].duplicated().sum())
        age = df["tag_age_weeks"]
        row = {
            "apply_week_start": w.apply_week_start,
            "cutoff_week": w.cutoff_week,
            "expected_players": w.n_players,
            "actual_rows": len(df),
            "duplicated_players": dup,
            "from_cutoff_week": int((age == args.lag_weeks).sum()),
            "from_older_weeks": int((age > args.lag_weeks).sum()),
            "zero_fill_rows": int((df["is_zero_fill"] == 1).sum()),
            "max_tag_age_weeks": int(age.max()),
            "tag_versions": ",".join(sorted(df["tag_version"].astype(str).unique())),
            "max_etl_time": w.max_etl_time,
            "status": status,
        }
        if row["actual_rows"] != w.n_players or dup or row["from_cutoff_week"] != w.n_cutoff_week:
            log.warning("對帳不符 apply_week=%s：%s", w.apply_week_start, row)
        log.info("%s apply_week=%s rows=%s（前一週 %s、更早 %s）", status, w.apply_week_start,
                 f"{row['actual_rows']:,}", f"{row['from_cutoff_week']:,}", f"{row['from_older_weeks']:,}")
        manifest_rows.append(row)

    manifest = pd.DataFrame(manifest_rows)
    manifest["match"] = (
        (manifest["expected_players"] == manifest.get("actual_rows"))
        & (manifest.get("duplicated_players") == 0)
    )
    merged = merge_manifest(out_dir / "_manifest.csv", manifest)
    merged.to_csv(out_dir / "_manifest.csv", index=False)

    log.info("完成：本次 %d 個注單週，失敗 %d，對帳不符 %d；快取共 %d 週 %s ～ %s，輸出 %s",
             len(manifest), failures, int((~manifest["match"]).sum()), len(merged),
             merged["apply_week_start"].min(), merged["apply_week_start"].max(), out_dir)
    return 1 if failures else 0


def load_tags(
    agent_code: str = AGENT_CODE,
    currency: str = CURRENCY,
    game_id: str = GAME_ID,
    start: str = START_DATE,
    end: str = END_DATE,
    lag_weeks: int = TAG_LAG_WEEKS,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """讀回已落地的 parquet（[start, end) 涵蓋的注單週），依 apply_week_start、member_account 排序。

    start / end 是注單日期；與注單合併時用 apply_week_start 對注單所在週的週一。
    """
    root = cache_dir(agent_code, currency, game_id, lag_weeks)
    weeks = {f"apply_week={w:%Y-%m-%d}" for w in bet_weeks(start, end)}
    paths = sorted(p for p in root.glob("apply_week=*.parquet") if p.stem in weeks)
    if not paths:
        raise FileNotFoundError(f"{root} 下沒有 parquet，請先執行 python data_loading/tag_loader.py")
    read_cols = None
    if columns is not None:
        read_cols = list(dict.fromkeys([*columns, "apply_week_start", "start_date", "tag_age_weeks", "member_account"]))
    tables = [pq.read_table(p, columns=read_cols) for p in paths]
    df = pa.concat_tables(tables, promote_options="default").to_pandas()
    df["start_date"] = pd.to_datetime(df["start_date"])
    df["apply_week_start"] = pd.to_datetime(df["apply_week_start"])
    return df.sort_values(["apply_week_start", "member_account"], ignore_index=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=f"抓取 {TABLE} 各注單週的 as-of 最新標籤並落地 parquet")
    p.add_argument("--agent-code", default=AGENT_CODE)
    p.add_argument("--currency", default=CURRENCY)
    p.add_argument("--game-id", default=GAME_ID)
    p.add_argument("--start", default=START_DATE, help="注單日期，含，YYYY-MM-DD；會往前對齊到該週週一")
    p.add_argument("--end", default=END_DATE, help="注單日期，不含，YYYY-MM-DD")
    p.add_argument("--lag-weeks", type=int, default=TAG_LAG_WEEKS, help="注單週 W 使用 start_date ≤ W − N 週的最新標籤，預設 1")
    p.add_argument("--slices", type=int, default=N_SLICES, help="每個注單週分幾批抓")
    p.add_argument("--no-cache", action="store_true", help="忽略既有 parquet，重新抓 DB 當下最新資料")
    p.add_argument("--plan-only", action="store_true", help="只列每個注單週的預估玩家數")
    return p.parse_args(argv)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    sys.exit(run(parse_args()))

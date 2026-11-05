# -*- coding: utf-8 -*-
"""
第二段：逐桶建立逐手序列、狀態、Y、決策點、權重、特徵（SPEC 5.1–6），合併後附加慢層標籤與分群欄位（7、8.1）。

每桶依序：
  ① merge_free_spins      逐手序列：排序、併入免費局（5.1）
  ② add_pool_state        逐手狀態：日水池、當月（含月初基準）、session（5.2）   ┐ 在完整序列上計算
  ③ add_sessions          session 表與 Y、截尾、窗口主版本（5.3、5.4）           ┘
  ④ select_decisions      抽決策點：session 起點在訓練期間或驗證期間（session_date）、未截尾、主版本 ≥ arm_dominance（5.4）
  ⑤ add_weights           附加 w_cap，不刪列（5.5）
  ⑥ add_features          快層特徵 + 當月 RTP GP_rtp（目前不使用；第 6 節）
合併後：attach_tags        每個決策點取該玩家 week_index < W 的最新一筆標籤、分群欄位（7、8.1）
→ 依 session_date 一天一檔寫入逐日 dataset（store.py），同一天只存一份、所有 run 共用。
③ 之後只篩選列或附加欄位，不平移、不重算（5.0）。

一次 build 處理一段連續的日子（一份 session 表），決策點的 session_date（session 起點的美東日期）決定它屬於哪一天；
跨午夜的 session 整段歸起點那天。一天的決策點與 build 一次建幾天無關（截尾以該日為準、session_id 含日期），
run 再依 train_start～train_end → 訓練、valid_start～valid_end → 驗證 取用。
dataset 欄位命名：索引在前；快層 GP_（只用當手之前）／NGP_（含當手開獎前資訊）、慢層 TAG_、分群 SEG_；
遊戲版本 game_version、arm、arm_ratio；Y／權重 churnNext20、w_cap。順序見 DATASET_COLUMNS。
"""
from __future__ import annotations

import logging
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from feel_engine.pipeline.data import (
    OUTCOMES,
    TZ,
    DataCheckError,
    lit_et,
    outcome_expr,
    peak_rss_mb,
)
from feel_engine.pipeline.store import current_tag_weeks, write_json
from feel_engine.paths import RunPaths, StorePaths, rel, tag_dir

N_DEBUG_PLAYERS = 5
PLAYER = "member_account"
NA_ARM = "<NA>"

# 快層（第 6 節）
#   GP_ ：水池類，只用當手之前的手（含 session 已玩時間）
#   SY_ ：系統端狀態：當手下注額、是否 BUY、下注前餘額、餘額 ÷ 下注額、session 起始餘額
#   NGP_：由水池與系統端組合的衍生特徵（用到當手開獎前資訊）
GP_FEATURES = (
    "GP_dailySpinCount",                              # 6.2 日水池
    *(f"GP_dailyRate_{o}" for o in OUTCOMES),
    "GP_dailyRtp",
    "GP_dailyRate_FREE_GAME",                         # 今日含 FreeGame 的手數比例（當手之前）
    "GP_dailyRate_MEDIUM_PLUS",                       # 6.3 交互作用與多條件
    "GP_dailyNoWin_x_spins",
    "GP_flag_dryLong",
    "GP_sessionMinutes",                              # 6.4 本 session 已玩分鐘數
)
SY_FEATURES = (
    "SY_bet",                                         # 當手下注額
    "SY_isBuyFeature",                                # 當手是否購買特色
    "SY_balance",                                     # 當手下注前餘額
    "SY_balanceToBet",                                # 當手下注前餘額 ÷ 當手下注額
    "SY_first_balance",                               # session 第一手下注前餘額（整個 session 固定）
)
NGP_FEATURES = (
    "NGP_dailyTotalWinToBet",                         # 今日總贏分（gamePotSave totalWin 的日版本，到當手之前）÷ 當手下注額
    "NGP_flag_coldLoss",
    "NGP_dailyNoWin_x_sessionLoss",                   # 今日 NO_WIN 比例 × session 跌幅
)
FEATURES = (*GP_FEATURES, *SY_FEATURES, *NGP_FEATURES)   # 可放進決策樹的快層欄位（實際放哪些見 model.yaml）
GUARD_FEATURES = ("GP_rtp",)
LABEL_COLUMNS = ("churnNext20", "w_cap")
# 分桶中間檔；week_start 只供 attach_tags 合併標籤，不進 dataset
PART_COLUMNS = (
    PLAYER, "spin_ts", "session_date", "session_id", "session_spin_idx", "week_start",
    "game_version", "arm", "arm_ratio", *FEATURES, *GUARD_FEATURES, *LABEL_COLUMNS,
)
SESSION_ID_STRIDE = 1000      # dataset 的 session_id = session_date（yyyymmdd）× 1000 ＋ 該玩家當天第幾個 session

# 7、8.1
TAG_READ_COLUMNS = ("member_account", "week_index", "is_zero_fill", "lifecycle_tag", "value_combined_tag")
NEW_PLAYER_LIFECYCLE = "新手"
NO_RECORD_VALUE = "無紀錄"
TAG_SOURCE_PREV, TAG_SOURCE_OLDER, TAG_SOURCE_NONE = "W-1", "更早", "無紀錄"
TAG_FEATURES = ("TAG_lifecycle", "TAG_value_combined")    # 慢層：stage 1 加前者、stage 2 再加後者（第 7 節）
TAG_COLUMNS = (*TAG_FEATURES, "TAG_source", "TAG_week_index", "TAG_is_zero_fill")
SEG_COLUMNS = ("SEG_lifecycle", "SEG_value")              # split = lifecycle／value（8.1）

# dataset 欄位順序：索引 → 遊戲版本 → 快層（GP_ → SY_ → NGP_）→ 慢層（TAG_）→ 分群（SEG_）→ Y／權重
INDEX_COLUMNS = (PLAYER, "spin_ts", "session_date", "session_id", "session_spin_idx", "week_index")
VERSION_COLUMNS = ("game_version", "arm", "arm_ratio")
DATASET_COLUMNS = (*INDEX_COLUMNS, *VERSION_COLUMNS, *GP_FEATURES, *GUARD_FEATURES, *SY_FEATURES, *NGP_FEATURES,
                   *TAG_COLUMNS, *SEG_COLUMNS, *LABEL_COLUMNS)

log = logging.getLogger("feel_engine.pipeline.build")


@dataclass(frozen=True)
class BuildParams:
    window: int                       # churn_window_spins（同時是決策間隔）
    session_gap: timedelta
    return_window: timedelta
    days: tuple[date, ...]            # 美東；只抽 session 起點在這些日期的決策點（本次 build 的連續日子）
    data_end: datetime                # 美東，tz-aware；讀取範圍終點（後段不足時為快取終點）
    arm_dominance: float
    player_day_cap: float
    dry_min_spins: int
    cold_nowin_rate: float
    loss_ratio: float
    seed: int
    version_map: dict
    provisional_arm: bool = False     # True：version_map 未填完時以 game_version 暫代 arm
    post_hours: timedelta = timedelta(hours=12)   # 截尾以 session 起點日為準：session 須在隔天 0 點 ＋ 此值前結束（5.3）

    @property
    def arms(self) -> tuple[str, ...]:
        if self.provisional_arm:
            return tuple(sorted(self.version_map))
        return tuple(sorted({v for v in self.version_map.values() if v is not None}))

    @property
    def base_month(self) -> int:
        """月初基準加在哪個月（= 第一天所在月份；讀取範圍內跨月時，下個月從 1 日 0 點在範圍內重新累計）。"""
        first = min(self.days)
        return first.year * 100 + first.month


def params_from_config(cfg: dict, meta: dict, provisional_arm: bool) -> BuildParams:
    """meta：第一段的回傳（本次 build 的連續日子 first_day～last_day 與讀取範圍）。"""
    derived = cfg["features"]["derived"]
    first, last = date.fromisoformat(meta["first_day"]), date.fromisoformat(meta["last_day"])
    return BuildParams(
        window=int(cfg["labels"]["churn_window_spins"]),
        session_gap=timedelta(minutes=cfg["labels"]["session_gap_minutes"]),
        return_window=timedelta(hours=cfg["labels"]["return_window_hours"]),
        days=tuple(first + timedelta(days=i) for i in range((last - first).days + 1)),
        data_end=datetime.fromisoformat(meta["read_plan"]["data_end"]),
        arm_dominance=float(cfg["labels"]["arm_dominance"]),
        player_day_cap=float(cfg["weighting"]["player_day_cap"]),
        dry_min_spins=int(derived["dry_min_spins"]),
        cold_nowin_rate=float(derived["cold_nowin_rate"]),
        loss_ratio=float(derived["loss_ratio"]),
        seed=int(cfg["seed"]),
        version_map=dict(cfg["modules"]["version_map"]),
        provisional_arm=provisional_arm,
        post_hours=timedelta(hours=float(cfg["data"].get("post_hours", 0))),
    )


def _ratio(num: pl.Expr, den: pl.Expr) -> pl.Expr:
    """除以 0（或分母缺值）→ 缺值（6.1）。"""
    return pl.when(den > 0).then(num.cast(pl.Float64) / den).otherwise(None)


def rows_by_day(df: pl.DataFrame, day_col: str) -> dict[str, dict]:
    """group_by 的結果 → {日期字串: {欄: 值}}（逐日統計）。"""
    return {str(r.pop(day_col)): r for r in df.iter_rows(named=True)}


def count_by_day(df: pl.DataFrame, day_col: str = "session_date") -> dict[str, int]:
    g = df.group_by(day_col).len()
    return {str(d): n for d, n in zip(g[day_col].to_list(), g["len"].to_list())}


def add_stats(a: dict, b: dict) -> dict:
    """逐鍵相加（巢狀 dict 遞迴）：合併各桶的逐日統計、或彙總多天。"""
    out = dict(a)
    for k, v in b.items():
        out[k] = add_stats(out.get(k, {}), v) if isinstance(v, dict) else out.get(k, 0) + (v or 0)
    return out


# ---------------------------------------------------------------------------
# ① 逐手序列
# ---------------------------------------------------------------------------
def merge_free_spins(raw: pl.DataFrame, late_after: timedelta = timedelta(minutes=30)) -> tuple[pl.DataFrame, dict]:
    """依 member_account、seq 穩定排序，把 bet_points = 0 的列併入同玩家前一個付費手。

    回傳（付費手表, 統計）。統計：
      r6_orphan_free_rows  找不到前一個付費手而排除的 bet_points = 0 列（R6）
      free_merged          併入的 bet_points = 0 列
      free_late / free_late_prize  其中晚於付費手 late_after 以上才出現的列數與贏分（待確認事項）
      by_day               以上各項依該列的美東日期分開（逐日 dataset 的報告用）
    """
    raw = (
        raw.sort([PLAYER, "seq"])
        .with_columns(is_paid=pl.col("bet_points") > 0)
        .with_columns(spin_no=pl.col("is_paid").cast(pl.Int64).cum_sum().over(PLAYER))
    )
    # 間隔要在篩選前算：組內第一列才是付費手
    raw = raw.with_columns(late=(pl.col("ts") - pl.col("ts").first().over([PLAYER, "spin_no"])) > late_after)
    free = raw.filter(~pl.col("is_paid") & (pl.col("spin_no") > 0))
    stats = {
        "r6_orphan_free_rows": raw.filter(pl.col("spin_no") == 0).height,
        "free_merged": free.height,
        "free_late": int(free["late"].sum()),
        "free_late_prize": int(free.filter(pl.col("late"))["prize"].sum()),
        "free_prize": int(free["prize"].sum()),
    }
    merged, late = pl.col("spin_no") > 0, (pl.col("spin_no") > 0) & pl.col("late")
    per_day = raw.filter(~pl.col("is_paid")).group_by(et_date=pl.col("ts").dt.date()).agg(
        r6_orphan_free_rows=(~merged).sum(), free_merged=merged.sum(), free_late=late.sum(),
        free_late_prize=pl.col("prize").filter(late).sum(), free_prize=pl.col("prize").filter(merged).sum(),
    )
    stats["by_day"] = rows_by_day(per_day, "et_date")
    spins = (
        raw.filter(pl.col("spin_no") > 0)
        .group_by([PLAYER, "spin_no"], maintain_order=True)
        .agg(
            spin_ts=pl.col("ts").first(),
            bet=pl.col("bet_points").first(),
            balance_before=pl.col("points").first(),
            game_version=pl.col("game_version").first(),   # 付費手的版本；免費局列不影響
            is_buy=pl.col("is_buy").first(),               # 付費手的 game_mode
            is_free=pl.col("is_free").first(),             # 付費手含 FreeGame（game_mode 位元 2；開獎結果）
            prize=pl.col("prize").sum(),
            balance_after=pl.col("last_points").last(),
            n_free=pl.len() - 1,
        )
    )
    spins = spins.with_columns(mult=pl.col("prize").cast(pl.Float64) / pl.col("bet"))
    return spins.with_columns(outcome=outcome_expr(pl.col("mult"))), stats


def attach_arm(spins: pl.DataFrame, version_map: dict, provisional: bool) -> pl.DataFrame:
    """game_version → arm（第 4 節）。對應不到 → 缺值（R4，不進第 9–11 節）。"""
    if provisional:
        arm = pl.when(pl.col("game_version").is_in(list(version_map))).then(pl.col("game_version")).otherwise(None)
    else:
        mapping = {k: v for k, v in version_map.items() if v is not None}
        arm = pl.col("game_version").replace_strict(mapping, default=None, return_dtype=pl.String)
    return spins.with_columns(arm=arm)


# ---------------------------------------------------------------------------
# ② 逐手狀態
# ---------------------------------------------------------------------------
def add_pool_state(spins: pl.DataFrame, base: pl.DataFrame | None, base_month: int) -> pl.DataFrame:
    """第 i 手的狀態只累計同週期內第 i − 1 手（含）以前的付費手（5.2）；含 FreeGame 手數同樣不含當手。

    「累計到本手 − 本手」等同「shift(1) 後依週期分組累計」，單次掃描。
    當月 Σbet、Σprize = 月初基準（只加在 base_month）＋ 範圍內累計。
    """
    day = [PLAYER, "et_date"]
    month = [PLAYER, "et_month"]
    spins = spins.with_columns(
        et_date=pl.col("spin_ts").dt.date(),
        et_month=pl.col("spin_ts").dt.year() * 100 + pl.col("spin_ts").dt.month(),
        **{f"is_{o}": (pl.col("outcome") == o).cast(pl.Int64) for o in OUTCOMES},
        is_FREE_GAME=pl.col("is_free").cast(pl.Int64),
    )
    before = lambda col, keys: pl.col(col).cum_sum().over(keys) - pl.col(col)  # noqa: E731
    spins = spins.with_columns(
        st_d_count=pl.int_range(pl.len()).over(day),
        st_d_bet=before("bet", day),
        st_d_prize=before("prize", day),
        st_m_bet=before("bet", month),
        st_m_prize=before("prize", month),
        **{f"st_d_{o}": before(f"is_{o}", day) for o in OUTCOMES},
        st_d_FREE_GAME=before("is_FREE_GAME", day),
    )
    if base is None or base.height == 0:
        return spins
    in_month = pl.col("et_month") == base_month
    return (
        spins.join(base, on=PLAYER, how="left", maintain_order="left")
        .with_columns(
            st_m_bet=pl.col("st_m_bet") + pl.when(in_month).then(pl.col("base_bet").fill_null(0)).otherwise(0),
            st_m_prize=pl.col("st_m_prize") + pl.when(in_month).then(pl.col("base_prize").fill_null(0)).otherwise(0),
        )
        .select(pl.exclude("base_bet", "base_prize"))
    )


# ---------------------------------------------------------------------------
# ③ session、Y、截尾、窗口主版本
# ---------------------------------------------------------------------------
def add_window_dominance(spins: pl.DataFrame, arms: tuple[str, ...], window: int) -> pl.DataFrame:
    """窗口（第 i 至 i + window − 1 手或至 session 結束）內手數最多的 arm 與其佔比（5.4）。

    每個 arm 的窗口手數 = session 內累計次數在窗口尾 − 窗口頭前一手，單次掃描。
    """
    sess = [PLAYER, "session_id"]
    keys = (*arms, NA_ARM)
    arm_key = pl.col("arm").fill_null(NA_ARM)
    spins = spins.with_columns(**{f"_c{i}": (arm_key == a).cast(pl.Int32).cum_sum().over(sess) for i, a in enumerate(keys)})
    spins = spins.with_columns(**{
        f"_w{i}": (
            pl.col(f"_c{i}").shift(-(window - 1)).over(sess).fill_null(pl.col(f"_c{i}").last().over(sess))
            - pl.col(f"_c{i}") + (arm_key == a).cast(pl.Int32)
        )
        for i, a in enumerate(keys)
    })
    w_cols = [f"_w{i}" for i in range(len(keys))]
    dom_count = pl.max_horizontal(w_cols)
    dom_arm = pl.when(pl.col(w_cols[0]) == dom_count).then(pl.lit(keys[0]))
    for i in range(1, len(keys)):
        dom_arm = dom_arm.when(pl.col(w_cols[i]) == dom_count).then(pl.lit(keys[i]))
    win_len = pl.min_horizontal(pl.lit(window), pl.col("session_n") - pl.col("session_spin_idx"))
    spins = spins.with_columns(
        arm_ratio=dom_count.cast(pl.Float64) / win_len,
        dom_arm=pl.when(dom_arm == NA_ARM).then(None).otherwise(dom_arm),
    )
    return spins.select(pl.exclude([f"_c{i}" for i in range(len(keys))] + w_cols))


def add_sessions(spins: pl.DataFrame, p: BuildParams, pre_last: pl.DataFrame | None = None) -> pl.DataFrame:
    """建立 session 表並在每手標記決策點、Y、截尾、session 起點是否完整與窗口主版本（5.3、5.4）。

    pre_last：讀取範圍起點前 session_gap 內每位玩家最後一個付費手（UTC naive）。玩家在範圍內第一手與它的間隔
    ≤ session_gap 時，範圍內第一個 session 其實更早開始（起點是被讀取範圍切出來的）→ session_truncated。
    """
    sess_keys = [PLAYER, "session_id"]
    gap = pl.col("spin_ts").diff().over(PLAYER)
    spins = spins.with_columns(new_session=gap.is_null() | (gap > p.session_gap)) \
        .with_columns(session_id=pl.col("new_session").cast(pl.Int64).cum_sum().over(PLAYER))
    spins = spins.with_columns(
        session_spin_idx=pl.int_range(pl.len()).over(sess_keys),
        st_session_start_balance=pl.col("balance_before").first().over(sess_keys),
        # 本 session 已玩分鐘數 = 當手時間 − session 第一手時間（遊戲端 gamePotSave 有時間可算）
        st_session_minutes=(pl.col("spin_ts") - pl.col("spin_ts").first().over(sess_keys)).dt.total_seconds() / 60,
    )
    cut_players = pl.Series(PLAYER, [], dtype=pl.String)
    if pre_last is not None and pre_last.height:
        first_in = spins.group_by(PLAYER).agg(first_ts=pl.col("spin_ts").min())
        last_pre = pre_last.with_columns(pl.col("last_paid").dt.replace_time_zone("UTC").dt.convert_time_zone(TZ))
        cut_players = (first_in.join(last_pre, on=PLAYER)
                       .filter(pl.col("first_ts") - pl.col("last_paid") <= p.session_gap)[PLAYER])
    spins = spins.with_columns(
        session_truncated=(pl.col("session_id") == 1) & pl.col(PLAYER).is_in(cut_players.implode()),
    )

    # 截尾以 session 起點日為準（與 build 一次建幾天無關）：須在隔天 0 點 ＋ post_hours 前結束、且之後 return_window 都讀得到
    start_date = pl.col("session_start").dt.date()
    day_horizon = (start_date.cast(pl.Datetime("us")).dt.offset_by("1d").dt.replace_time_zone(TZ)
                   + pl.duration(seconds=p.post_hours.total_seconds()))
    sessions = (
        spins.group_by(sess_keys, maintain_order=True)
        .agg(session_end=pl.col("spin_ts").last(), session_start=pl.col("spin_ts").first(), session_n=pl.len())
        .with_columns(next_session_start=pl.col("session_start").shift(-1).over(PLAYER))
        .with_columns(
            session_churn=pl.col("next_session_start").is_null()
            | (pl.col("next_session_start") > pl.col("session_end") + p.return_window),
            session_censored=(pl.col("session_end") > day_horizon)
            | ((pl.col("session_end") + p.return_window) > lit_et(p.data_end)),
            session_day_no=pl.int_range(1, pl.len() + 1).over([PLAYER, start_date]),   # 該玩家當天第幾個 session
        )
    )
    spins = spins.join(
        sessions.select([*sess_keys, "session_start", "session_end", "session_n", "session_churn", "session_censored",
                         "session_day_no"]),
        on=sess_keys, how="left", maintain_order="left",
    )
    w = p.window
    spins = spins.with_columns(
        is_dp=pl.col("session_spin_idx") % w == 0,
        is_last_dp=pl.col("session_spin_idx") == (pl.col("session_n") - 1) // w * w,
    ).with_columns(
        churnNext20=(pl.col("is_last_dp") & pl.col("session_churn")).cast(pl.Int8),
        censored=pl.col("session_censored"),          # 看不完 Y 的 session 整段不抽（只刪最後一筆會讓流失率偏低）
    )
    return add_window_dominance(spins, p.arms, w)


def build_spin_table(raw: pl.DataFrame, p: BuildParams, base: pl.DataFrame | None = None,
                     pre_last: pl.DataFrame | None = None) -> tuple[pl.DataFrame, dict]:
    """①②③：回傳（含狀態與 Y 的完整逐手表, 併入免費局的統計）。"""
    spins, merge_stats = merge_free_spins(raw, p.session_gap)
    spins = attach_arm(spins, p.version_map, p.provisional_arm)
    spins = add_pool_state(spins, with_orphan_prize(base, raw), p.base_month)
    return add_sessions(spins, p, pre_last), merge_stats


def with_orphan_prize(base: pl.DataFrame | None, raw: pl.DataFrame) -> pl.DataFrame | None:
    """月初基準加上 R6 免費列的贏分：讀取範圍內、玩家第一個付費手之前的免費列，併入的是範圍起點前最後一個付費手，
    範圍內併不進任何一手；讀取範圍較早開始時它們會計入那一手的月份。那一手在基準月份（玩家在基準窗口有付費手，
    base_bet > 0）才加回，GP_rtp 才與範圍起點無關；否則那一手在上個月，不加。"""
    if base is None or base.height == 0:
        return base
    first_paid = raw.filter(pl.col("bet_points") > 0).group_by(PLAYER).agg(first_seq=pl.col("seq").min())
    orphan = (raw.join(first_paid, on=PLAYER, how="inner")
              .filter(pl.col("seq") < pl.col("first_seq"))
              .group_by(PLAYER).agg(orphan_prize=pl.col("prize").sum()))
    return (base.join(orphan, on=PLAYER, how="left")
            .with_columns(base_prize=pl.col("base_prize") + pl.when(pl.col("base_bet") > 0)
                          .then(pl.col("orphan_prize").fill_null(0)).otherwise(0))
            .select(PLAYER, "base_bet", "base_prize"))


# ---------------------------------------------------------------------------
# ④ 決策點、⑤ 權重、⑥ 特徵
# ---------------------------------------------------------------------------
def add_features(df: pl.DataFrame, p: BuildParams) -> pl.DataFrame:
    """⑥ 由狀態算快層特徵與當月 RTP GP_rtp（目前不使用；第 6 節）；只用開獎前已知的欄位，不用當手開獎結果。"""
    df = df.with_columns(
        GP_dailySpinCount=pl.col("st_d_count"),
        **{f"GP_dailyRate_{o}": _ratio(pl.col(f"st_d_{o}"), pl.col("st_d_count")) for o in OUTCOMES},
        GP_dailyRtp=_ratio(pl.col("st_d_prize"), pl.col("st_d_bet")),
        GP_dailyRate_FREE_GAME=_ratio(pl.col("st_d_FREE_GAME"), pl.col("st_d_count")),
        GP_rtp=_ratio(pl.col("st_m_prize"), pl.col("st_m_bet")),
        GP_sessionMinutes=pl.col("st_session_minutes"),
        SY_bet=pl.col("bet"),
        SY_isBuyFeature=pl.col("is_buy").cast(pl.Int8),
        SY_balance=pl.col("balance_before"),
        SY_balanceToBet=_ratio(pl.col("balance_before"), pl.col("bet")),
        SY_first_balance=pl.col("st_session_start_balance"),
        NGP_dailyTotalWinToBet=_ratio(pl.col("st_d_prize"), pl.col("bet")),
    )
    session_loss = 1 - _ratio(pl.col("SY_balance"), pl.col("SY_first_balance"))
    df = df.with_columns(
        GP_dailyRate_MEDIUM_PLUS=pl.col("GP_dailyRate_MEDIUM_WIN") + pl.col("GP_dailyRate_BIG_WIN"),
        GP_dailyNoWin_x_spins=pl.col("GP_dailyRate_NO_WIN") * pl.col("GP_dailySpinCount").cast(pl.Float64).log1p(),
        NGP_dailyNoWin_x_sessionLoss=pl.col("GP_dailyRate_NO_WIN")
        * pl.when(session_loss < 0).then(0.0).otherwise(session_loss),
    )
    return df.with_columns(
        GP_flag_dryLong=((pl.col("GP_dailySpinCount") >= p.dry_min_spins) & (pl.col("GP_dailyRate_MEDIUM_PLUS") == 0))
        .fill_null(False).cast(pl.Int8),
        NGP_flag_coldLoss=((pl.col("GP_dailyRate_NO_WIN") >= p.cold_nowin_rate)
                          & (pl.col("SY_balance") <= pl.col("SY_first_balance") * (1 - p.loss_ratio)))
        .fill_null(False).cast(pl.Int8),
    )


def select_decisions(spins: pl.DataFrame, p: BuildParams) -> tuple[pl.DataFrame, dict]:
    """④ 只篩選列並附加特徵；回傳（決策點, 各步驟統計）。樣本的 arm = 窗口主版本。統計的 by_day 依 session_date 分開。"""
    funnel, counts = {}, {}
    dp = spins.filter(pl.col("is_dp"))
    funnel["dp_all"] = dp.height
    # 決策點歸屬日 session_date = session 起點的美東日期：跨午夜的 session 整段歸起點那天（訓練期間／驗證期間交界的 session 歸訓練日）
    dp = dp.with_columns(session_date=pl.col("session_start").dt.date()).filter(pl.col("session_date").is_in(list(p.days)))
    funnel["on_days"], counts["on_days"] = dp.height, count_by_day(dp)
    dp = dp.filter(~pl.col("censored"))
    funnel["after_censor"], counts["after_censor"] = dp.height, count_by_day(dp)
    dp = dp.filter(~pl.col("session_truncated"))                  # session 起點被讀取範圍切到 → 不抽
    funnel["after_session_start"], counts["after_session_start"] = dp.height, count_by_day(dp)

    dp = add_features(dp, p).with_columns(meet=pl.col("arm_ratio") >= p.arm_dominance)
    dom = dp.group_by("session_date", "meet").agg(
        n=pl.len(), churn=pl.col("churnNext20").cast(pl.Int64).sum(),
        rtp_sum=pl.col("GP_dailyRtp").sum(), rtp_n=pl.col("GP_dailyRtp").count(),
    )
    by_day: dict[str, dict] = {d: {"funnel": {}} for d in counts["on_days"]}
    for r in dom.iter_rows(named=True):
        by_day[str(r["session_date"])].setdefault("dominance", {})["meet" if r["meet"] else "miss"] = {
            k: r[k] for k in ("n", "churn", "rtp_sum", "rtp_n")}
    dominance = {}
    for day in by_day.values():
        dominance = add_stats(dominance, day.get("dominance", {}))

    dp = dp.filter(pl.col("meet")).with_columns(
        arm=pl.col("dom_arm"),
    )
    funnel["after_dominance"], counts["after_dominance"] = dp.height, count_by_day(dp)
    for step, c in counts.items():
        for d, n in c.items():
            by_day[d]["funnel"][step] = n
    return dp, {"funnel": funnel, "dominance": dominance, "by_day": by_day}


def add_weights(dp: pl.DataFrame, cap: float) -> tuple[pl.DataFrame, dict]:
    """⑤ w_cap = min(1, cap ÷ 玩家當天決策點數)；全部保留（5.5）。當天 = session_date（session 起點日）。"""
    keys = [PLAYER, "session_date"]
    dp = dp.with_columns(_n=pl.len().over(keys)).with_columns(
        w_cap=pl.min_horizontal(pl.lit(1.0), pl.lit(cap) / pl.col("_n"))
    )
    per_day = dp.group_by(keys).agg(pl.col("_n").first())
    stats = {
        "player_days": per_day.height,
        "player_days_capped": int((per_day["_n"] > cap).sum()),
        "n": dp.height,
        "players": dp[PLAYER].n_unique(),
        "churn_sum": int(dp["churnNext20"].cast(pl.Int64).sum()),
        "w_sum": float(dp["w_cap"].sum()),
        "wy_sum": float((dp["w_cap"] * dp["churnNext20"]).sum()),
    }
    days = per_day.group_by("session_date").agg(player_days=pl.len(), player_days_capped=(pl.col("_n") > cap).sum())
    rows = dp.group_by("session_date").agg(
        n=pl.len(), churn_sum=pl.col("churnNext20").cast(pl.Int64).sum(),
        w_sum=pl.col("w_cap").sum(), wy_sum=(pl.col("w_cap") * pl.col("churnNext20")).sum(),
    )
    stats["by_day"] = rows_by_day(days.join(rows, on="session_date"), "session_date")
    return dp.select(pl.exclude("_n")), stats


def dataset_session_id() -> pl.Expr:
    """dataset 的 session_id = session_date（yyyymmdd）× 1000 ＋ 該玩家當天第幾個 session（起點依序）。
    只由當天的資料決定，與 build 一次建幾天無關；（玩家, session_id）跨天也唯一。"""
    d = pl.col("session_date")
    ymd = d.dt.year().cast(pl.Int64) * 10_000 + d.dt.month().cast(pl.Int64) * 100 + d.dt.day().cast(pl.Int64)
    return ymd * SESSION_ID_STRIDE + pl.col("session_day_no").cast(pl.Int64)


def decision_table(spins: pl.DataFrame, p: BuildParams) -> tuple[pl.DataFrame, dict]:
    """④⑤⑥：回傳（決策點樣本, 統計）；session_id 換成 dataset 的編號（dataset_session_id）。"""
    dp, stats = select_decisions(spins, p)
    dp, wstats = add_weights(dp, p.player_day_cap)
    dp = dp.with_columns(
        week_start=pl.col("session_date") - pl.duration(days=pl.col("session_date").dt.weekday() - 1),
        session_id=dataset_session_id(),
    )
    by_day = stats.pop("by_day")
    for d, w in wstats.pop("by_day").items():
        by_day[d]["weights"] = w
    for r in dp.group_by("session_date", "arm").len().iter_rows(named=True):
        by_day[str(r["session_date"])].setdefault("arm_counts", {})[str(r["arm"])] = r["len"]
    return dp, {**stats, "weights": wstats, "by_day": by_day}


# ---------------------------------------------------------------------------
# 7、8.1：慢層標籤與分群欄位
# ---------------------------------------------------------------------------
def iso_week_index(d: date) -> int:
    y, w, _ = d.isocalendar()
    return y * 100 + w


def load_tag_rows(tag_dir: Path, week_starts: list[date]) -> dict[date, pl.DataFrame]:
    """讀需要的標籤列。as-of 快取（apply_week=W.parquet）只讀該週檔；否則讀整個週標籤快取。"""
    out = {}
    weekly = sorted(tag_dir.glob("week_index=*.parquet"))
    for w in week_starts:
        asof = tag_dir / f"apply_week={w:%Y-%m-%d}.parquet"
        if asof.exists():
            out[w] = pl.read_parquet(asof, columns=list(TAG_READ_COLUMNS))
        elif weekly:
            out[w] = pl.concat([pl.read_parquet(f, columns=list(TAG_READ_COLUMNS)) for f in weekly], how="diagonal_relaxed")
        else:
            raise DataCheckError(f"標籤快取 {tag_dir} 沒有 apply_week={w} 的檔案，請先執行 data_loading/tag_loader.py")
    return out


def latest_tags(rows: pl.DataFrame, week_start: date) -> pl.DataFrame:
    """7：每位玩家取 week_index < W 的最新一筆（含補零列）。"""
    w_idx = iso_week_index(week_start)
    latest = (rows.filter(pl.col("week_index") < w_idx)
              .sort([PLAYER, "week_index"], descending=[False, True])
              .unique(PLAYER, keep="first", maintain_order=True))
    return latest.select(
        PLAYER,
        TAG_week_index=pl.col("week_index"),
        TAG_is_zero_fill=pl.col("is_zero_fill"),
        TAG_lifecycle=pl.col("lifecycle_tag"),
        TAG_value_combined=pl.col("value_combined_tag"),
    )


def attach_tags(ds: pl.DataFrame, tag_rows: dict[date, pl.DataFrame]) -> pl.DataFrame:
    """依決策點所在美東週 W 附加 week_index（= W）、TAG_ 慢層欄位與 SEG_ 分群欄位（只附加欄位）。

    week_index 是決策點自己的週；TAG_week_index 是實際取到的那筆標籤的週（< W 的最新一筆，無紀錄為空）。
    分群（8.1）：SEG_lifecycle = TAG_lifecycle、SEG_value = TAG_value_combined。
    """
    tags = pl.concat([latest_tags(rows, w).with_columns(week_start=pl.lit(w)) for w, rows in tag_rows.items()])
    week_idx = {w: iso_week_index(w) for w in tag_rows}
    prev_idx = {w: iso_week_index(w - timedelta(weeks=1)) for w in tag_rows}

    ds = ds.join(tags, on=[PLAYER, "week_start"], how="left", maintain_order="left").with_columns(
        week_index=pl.col("week_start").replace_strict(week_idx, return_dtype=pl.Int64),
        _prev_idx=pl.col("week_start").replace_strict(prev_idx, return_dtype=pl.Int64),
    )
    no_record = pl.col("TAG_week_index").is_null()
    ds = ds.with_columns(
        TAG_source=pl.when(no_record).then(pl.lit(TAG_SOURCE_NONE))
        .when(pl.col("TAG_week_index") == pl.col("_prev_idx")).then(pl.lit(TAG_SOURCE_PREV))
        .otherwise(pl.lit(TAG_SOURCE_OLDER)),
        TAG_lifecycle=pl.when(no_record).then(pl.lit(NEW_PLAYER_LIFECYCLE)).otherwise(pl.col("TAG_lifecycle")),
        TAG_value_combined=pl.when(no_record).then(pl.lit(NO_RECORD_VALUE)).otherwise(pl.col("TAG_value_combined")),
    ).with_columns(
        SEG_lifecycle=pl.col("TAG_lifecycle"),
        SEG_value=pl.col("TAG_value_combined"),
    )
    return ds.select(pl.exclude("_prev_idx"))


# ---------------------------------------------------------------------------
# 逐桶處理（可平行）
# ---------------------------------------------------------------------------
def spin_stats(spins: pl.DataFrame, days: tuple[date, ...]) -> dict[str, dict]:
    """逐日（每手的美東日期，只算 days）：手數、倍數區間分佈與 RTP、R5 balance 不相接、同 session 相鄰付費手換版本、
    session 數（起點在該日）、起點被讀取範圍切到的 session 數。"""
    prev = lambda c: pl.col(c).shift(1).over(PLAYER)  # noqa: E731
    within = ~pl.col("new_session")
    flags = spins.select(
        PLAYER, "et_date", "outcome", "is_buy", "bet", "prize", "new_session",
        within=within,
        has_prev=prev("balance_after").is_not_null(),
        mismatch=prev("balance_after").is_not_null() & (pl.col("balance_before") != prev("balance_after")),
        switched=within & (pl.col("game_version") != prev("game_version")),
        bet_changed=within & (pl.col("bet") != prev("bet")),
    ).filter(pl.col("et_date").is_in(list(days)))
    g = flags.group_by("et_date").agg(
        spins=pl.len(), players=pl.col(PLAYER).n_unique(), buy_spins=pl.col("is_buy").sum(),
        sum_bet=pl.col("bet").sum(), sum_prize=pl.col("prize").sum(), sessions=pl.col("new_session").sum(),
        within_total=(pl.col("has_prev") & pl.col("within")).sum(),
        within_mismatch=(pl.col("mismatch") & pl.col("within")).sum(),
        between_total=(pl.col("has_prev") & ~pl.col("within")).sum(),
        between_mismatch=(pl.col("mismatch") & ~pl.col("within")).sum(),
        within_spins=pl.col("within").sum(), switched=pl.col("switched").sum(),
        switched_bet_changed=(pl.col("switched") & pl.col("bet_changed")).sum(),
        kept_bet_changed=(pl.col("within") & ~pl.col("switched") & pl.col("bet_changed")).sum(),
    )
    outcome = flags.group_by("et_date", "outcome").len()
    truncated = count_by_day(spins.filter(pl.col("session_truncated") & (pl.col("session_spin_idx") == 0))
                             .with_columns(start_date=pl.col("session_start").dt.date()), "start_date")
    out = {}
    for d, r in rows_by_day(g, "et_date").items():
        out[d] = {
            **{k: r[k] for k in ("spins", "players", "buy_spins", "sum_bet", "sum_prize", "sessions")},
            "outcome_counts": {},
            "r5": {k: r[k] for k in ("within_total", "within_mismatch", "between_total", "between_mismatch")},
            "version_switch": {k: r[k] for k in ("within_spins", "switched", "switched_bet_changed", "kept_bet_changed")},
            "truncated_sessions": truncated.get(d, 0),
        }
    for r in outcome.iter_rows(named=True):
        out[str(r["et_date"])]["outcome_counts"][r["outcome"]] = r["len"]
    return out


def process_bucket(bucket: int, p: BuildParams, debug: bool, store: StorePaths, rp: RunPaths) -> dict:
    """處理一個桶：寫出暫存的 dataset_parts/bucket=XX.parquet，回傳統計（by_day：本次各天的逐日統計）。"""
    t0 = time.time()
    raw = pl.read_parquet(store.buckets / f"bucket={bucket:02d}" / "*.parquet", hive_partitioning=False)
    base = None
    if store.month_base.exists():
        base = pl.read_parquet(store.month_base).join(raw.select(PLAYER).unique(), on=PLAYER, how="semi")
    n_raw = raw.height
    if not store.pre_range_last.exists():
        raise DataCheckError(f"找不到 {rel(store.pre_range_last)}：第一段沒有完成，請重跑 build")
    pre_last = pl.read_parquet(store.pre_range_last).join(raw.select(PLAYER).unique(), on=PLAYER, how="semi")
    spins, merge_stats = build_spin_table(raw, p, base, pre_last)
    del raw
    dp, dstats = decision_table(spins, p)
    dp.select(PART_COLUMNS).write_parquet(store.dataset_parts / f"bucket={bucket:02d}.parquet", compression="zstd")

    if debug and dp.height:
        rng = np.random.default_rng([p.seed, bucket])
        players = dp[PLAYER].unique().sort()
        player = players[int(rng.integers(players.len()))]
        rp.state_debug.mkdir(parents=True, exist_ok=True)
        debug_cols = [c for c in spins.columns if not c.startswith("is_") or c in ("is_dp", "is_last_dp", "is_buy")]
        (spins.filter(pl.col(PLAYER) == player).select(debug_cols)
         .with_columns(pl.col(pl.Datetime).dt.to_string("%Y-%m-%d %H:%M:%S%z"))
         .write_csv(rp.state_debug / f"{p.days[0]}_bucket{bucket:02d}_{player}.csv"))

    per_spin_day = spin_stats(spins, p.days)
    by_day = {}
    for d in (str(x) for x in p.days):
        by_day[d] = {**per_spin_day.get(d, {}), **merge_stats["by_day"].get(d, {}), **dstats["by_day"].get(d, {})}
    return {
        "bucket": bucket,
        "raw_rows": n_raw,
        "spins": spins.height,
        "players_spins": spins[PLAYER].n_unique(),
        "decisions": dp.height,
        "by_day": by_day,
        "seconds": round(time.time() - t0, 1),
        "peak_rss_mb": round(peak_rss_mb(), 1),
    }


def run_stage2(cfg: dict, meta: dict, provisional_arm: bool, store: StorePaths, rp: RunPaths,
               build_id: str, fingerprint: str) -> dict:
    """逐桶處理 → 合併 → 附加標籤與分群欄位 → 依 session_date 寫入逐日 dataset（每天一個 parquet ＋ json）；
    回傳並落地這次 build 的紀錄（builds/{build_id}.json）。"""
    from feel_engine.pipeline.store import build_settings

    t0 = time.time()
    p = params_from_config(cfg, meta, provisional_arm)
    n_buckets = int(meta["n_buckets"])
    workers = int(cfg["compute"]["workers"])
    n_debug = min(N_DEBUG_PLAYERS, n_buckets)
    debug_buckets = set(np.random.default_rng(p.seed).choice(n_buckets, n_debug, replace=False).tolist())

    if store.dataset_parts.exists():
        shutil.rmtree(store.dataset_parts)  # 只清本程式自己產生的暫存
    store.dataset_parts.mkdir(parents=True)

    results = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(process_bucket, b, p, b in debug_buckets, store, rp) for b in range(n_buckets)]
        for f in futures:
            r = f.result()
            results.append(r)
            log.info("bucket %02d：%s 手 → %s 決策點（%.1fs，峰值 %.0f MB）",
                     r["bucket"], f"{r['spins']:,}", f"{r['decisions']:,}", r["seconds"], r["peak_rss_mb"])

    tag_t = time.time()
    ds = pl.read_parquet(store.dataset_parts / "*.parquet", hive_partitioning=False)
    weeks = sorted(ds["week_start"].unique().to_list())
    tdir = tag_dir(cfg["data"]["merchant"], cfg["data"]["currency"], cfg["data"]["game"])
    ds = attach_tags(ds, load_tag_rows(tdir, weeks))
    log.info("附加標籤與分群欄位：%.1fs", time.time() - tag_t)

    tags_now = current_tag_weeks(cfg)
    day_stats: dict[str, dict] = {str(d): {} for d in p.days}
    for r in results:
        for d, st in r.pop("by_day").items():
            day_stats[d] = add_stats(day_stats[d], st)
    store.dataset_dir.mkdir(parents=True, exist_ok=True)
    for d in p.days:
        part = ds.filter(pl.col("session_date") == d).sort([PLAYER, "spin_ts"]).select(DATASET_COLUMNS)
        tmp = store.day(d).with_suffix(".parquet.tmp")
        part.write_parquet(tmp, compression="zstd")
        tmp.replace(store.day(d))
        write_json(store.day_meta(d), {
            "session_date": str(d), "build_id": build_id, "fingerprint": fingerprint,
            "provisional_arm": p.provisional_arm, "arms": list(p.arms), "rows": part.height,
            "tag_week": str(d - timedelta(days=d.weekday())), "stats": day_stats[str(d)],
        })
    shutil.rmtree(store.dataset_parts)

    build = {
        "build_id": build_id,
        "built_at": datetime.now().isoformat(timespec="seconds"),
        "run_name": rp.name,
        "first_day": meta["first_day"], "last_day": meta["last_day"],
        "fingerprint": fingerprint,
        "settings": build_settings(cfg),
        "provisional_arm": p.provisional_arm, "arms": list(p.arms),
        "stage1": meta,
        "tag_dir": tdir.name,
        "tag_weeks": {str(w): tags_now.get(str(w)) for w in weeks},
        "stage2": {
            "data_end": p.data_end.isoformat(),
            "buckets": results,
            "seconds": round(time.time() - t0, 1),
            "peak_rss_mb_main": round(peak_rss_mb(), 1),
        },
    }
    write_json(store.build_meta(build_id), build)
    log.info("第二段完成：%.1fs → %s（%s～%s，%s 筆）", build["stage2"]["seconds"], rel(store.dataset_dir),
             meta["first_day"], meta["last_day"], f"{ds.height:,}")
    return build

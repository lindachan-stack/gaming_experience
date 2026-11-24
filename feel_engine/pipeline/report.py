# -*- coding: utf-8 -*-
"""build 階段的報告：result/{run}/build/ 下的 data_quality_report.md（第 12 節）與 build_report.md（5.4、5.5、6、7、8.1）。"""
from __future__ import annotations

from collections import Counter
from datetime import date

import polars as pl

from feel_engine.pipeline.build import FEATURES, SEG_COLUMNS, TAG_SOURCE_NONE, TAG_SOURCE_OLDER, TAG_SOURCE_PREV
from feel_engine.pipeline.data import OUTCOMES, R4_WARN_RATIO
from feel_engine.paths import rel



def _pct(num: float, den: float, digits: int = 2) -> str:
    return f"{num / den * 100:.{digits}f}%" if den else "—"


def _table(header: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def _sum(buckets: list[dict], *path: str) -> float:
    total = 0
    for b in buckets:
        v = b
        for k in path:
            v = v.get(k, 0) if isinstance(v, dict) else 0
        total += v or 0
    return total


# ---------------------------------------------------------------------------
# 資料檢查
# ---------------------------------------------------------------------------
def _span(days: list[str]) -> str:
    return days[0] if len(days) == 1 else f"{days[0]}～{days[-1]}"


def write_data_quality_report(rp, builds: dict[str, dict], version_map: dict, day_stats: dict[str, dict] | None,
                              roles: dict[str, str] | None = None) -> str:
    """builds：本 run 各天來自的 build 紀錄（store builds/*.json）；day_stats：本 run 各天的逐日統計（None = 第二段沒跑）。"""
    metas = [b["stage1"] for b in builds.values()]
    days = [d for m in metas for d in m["days"]]
    paid = sum(d["paid_rows"] for d in days)
    unknown = sum(d["r4_unknown"] for d in days)
    r3 = sum(d["r3_match"] for d in days)
    dedup = sum(d["rows_after_dedup"] for d in days)
    version_counts = Counter()
    for m in metas:
        version_counts.update(m["version_counts"])
    cache = metas[-1]

    parts = [
        "# 資料檢查報告（SPEC 第 12 節）",
        "",
        f"- 快取：`{cache['bet_records_dir']}`（UTC {cache['cache_start_utc']} 至 {cache['cache_end_utc']}，不含終點）",
        "- R1 manifest 對帳：**通過**（逐日列數 = bet_records count，日期連續）",
        f"- dataset 逐日存放（`{rel(rp.store.dataset_dir)}`），本 run 的日子來自 {len(builds)} 次 build；"
        "每次 build 讀一段連續日子所需的範圍（下表），已建好的日子不重建",
        "",
        _table(
            ["build", "建置日子", "讀取範圍（美東）", "UTC 日檔", "R7"],
            [[f"`{b['build_id']}`", _span(sorted({b["first_day"], b["last_day"]})),
              f"{b['stage1']['read_plan']['range_start']} 至 {b['stage1']['read_plan']['data_end']}",
              _span(b["stage1"]["read_plan"]["range_days"]),
              ("⚠️ 後段不足，已截尾" if b["stage1"]["read_plan"]["back_short"] else "**通過**")]
             for b in builds.values()],
        ),
        "",
        f"- R2 日檔內 `id` 重複：{sum(d['r2_dup_ids'] for d in days):,} 筆（已去重）",
        f"- R3 `last_points = points − bet_points + prize`：{_pct(r3, dedup)} 相符（只報告）",
        f"- R4 讀取範圍內付費手 `game_version` 為空或不在 `version_map`：{unknown:,} / {paid:,} = {_pct(unknown, paid, 4)}"
        + ("　⚠️ 超過 1%" if paid and unknown / paid > R4_WARN_RATIO else ""),
        "",
        "## R4 讀取範圍內付費手版本分佈",
        "",
        _table(
            ["game_version", "付費手數", "佔比", "version_map"],
            [[f"`{v or '（空）'}`", f"{n:,}", _pct(n, paid), version_map.get(v) or "**未填**"]
             for v, n in sorted(version_counts.items(), key=lambda kv: -kv[1])],
        ),
    ]
    if day_stats is not None:
        b = list(day_stats.values())
        r5 = {k: _sum(b, "r5", k) for k in ("within_total", "within_mismatch", "between_total", "between_mismatch")}
        gap = float(list(builds.values())[-1]["settings"]["labels"]["session_gap_minutes"])
        parts += [
            "",
            "## 第二段檢查（本 run 的日子，依每手的美東日期）",
            "",
            f"- R5 相鄰付費手 balance 不相接：session 內 {r5['within_mismatch']:,} / {r5['within_total']:,}"
            f" = {_pct(r5['within_mismatch'], r5['within_total'])}；session 間 {r5['between_mismatch']:,} / "
            f"{r5['between_total']:,} = {_pct(r5['between_mismatch'], r5['between_total'])}（只報告）",
            f"- R6 找不到前一個付費手的免費局列：{_sum(b, 'r6_orphan_free_rows'):,} 筆（已排除）",
            f"- 併入前一個付費手的免費局列：{_sum(b, 'free_merged'):,} 筆；其中晚於付費手 "
            f"{gap:.0f} 分鐘以上才出現 {_sum(b, 'free_late'):,} 筆"
            f"（{_pct(_sum(b, 'free_late'), _sum(b, 'free_merged'))}）",
        ]
    parts += [
        "",
        "## 逐日明細（各次 build 讀取範圍內的列）",
        "",
        _table(
            ["build", "UTC 日期", "列數", "R2 重複 id", "R3 相符", "付費手", "BUY 付費手", "含 FreeGame 付費手", "R4 不明", "耗時(s)"],
            [[f"`{bid}`", d["date"], f"{d['rows']:,}", d["r2_dup_ids"], _pct(d["r3_match"], d["rows_after_dedup"]),
              f"{d['paid_rows']:,}", f"{d['buy_paid_rows']:,}", f"{d.get('free_paid_rows', 0):,}",
              d["r4_unknown"], d["seconds"]] for bid, bd in builds.items() for d in bd["stage1"]["days"]],
        ),
    ]
    text = "\n".join(parts) + "\n"
    rp.build_dir.mkdir(parents=True, exist_ok=True)
    rp.dq_report.write_text(text, encoding="utf-8")
    return text


# ---------------------------------------------------------------------------
# build 報告
# ---------------------------------------------------------------------------
def _churn_table(ds: pl.DataFrame, by: str, label: str) -> str:
    g = ds.group_by(by).agg(
        players=pl.col("member_account").n_unique(),
        n=pl.len(),
        churn=pl.col("churnNext20").mean(),
        w_churn=(pl.col("churnNext20") * pl.col("w_cap")).sum() / pl.col("w_cap").sum(),
    ).sort("n", descending=True)
    total = g["n"].sum()
    return _table([label, "玩家數", "決策點", "佔比", "流失率", "加權流失率"],
                  [[r[by], f"{r['players']:,}", f"{r['n']:,}", _pct(r["n"], total),
                    f"{r['churn'] * 100:.2f}%", f"{r['w_churn'] * 100:.2f}%"] for r in g.iter_rows(named=True)])


def _segment_sizes(ds: pl.DataFrame, col: str) -> pl.DataFrame:
    return ds.group_by(col).agg(n=pl.len()).sort("n", descending=True)


def _small_segments(ds: pl.DataFrame, min_n: int) -> dict[str, list[str]]:
    """各 SEG_ 欄位中訓練期間樣本不足、會沿用 all 策略的群。"""
    return {col: _segment_sizes(ds, col).filter(pl.col("n") < min_n)[col].to_list() for col in SEG_COLUMNS}


def _segment_table(ds: pl.DataFrame, col: str, min_n: int) -> str:
    g = _segment_sizes(ds, col)
    return _table([col, "訓練期間決策點", "策略"],
                  [[r[col], f"{r['n']:,}", "自建樹" if r["n"] >= min_n else "沿用 all"] for r in g.iter_rows(named=True)])


def pending_items(cfg: dict, provisional: bool, b: list[dict], back_short: bool, ds: pl.DataFrame,
                  train_ds: pl.DataFrame) -> list[list[str]]:
    """SPEC 第 14 節待確認事項 + 本次實作發現，附目前處理方式與數據。b：本 run 各天的逐日統計。"""
    sw = {k: _sum(b, "version_switch", k) for k in ("within_spins", "switched", "switched_bet_changed", "kept_bet_changed")}
    gap = float(cfg["labels"]["session_gap_minutes"])
    min_n = int(cfg["splits"]["min_segment_train_n"])
    no_record = ds.filter(pl.col("TAG_source") == TAG_SOURCE_NONE)
    return [
        ["現行 `gameVersion` 分派規則", "未知；`GP_dailyRtp`、`GP_dailySpinCount` 已放入特徵。"
         f"同 session 相鄰付費手 {_pct(sw['switched'], sw['within_spins'])} 換版本，換版本時下注額同時改變 "
         f"{_pct(sw['switched_bet_changed'], sw['switched'])}（未換版本 {_pct(sw['kept_bet_changed'], sw['within_spins'] - sw['switched'])}）"],
        ["`version_map` 未填", "⚠️ 暫代模式：以 `game_version` 當 arm（5 個值）；主版本判定也以 game_version 計，"
         "填好後重跑 build（設定改了，各天會自動重建）" if provisional else "已依 version_map 對應"],
        ["水池倍數區間是否計入免費局贏分", f"併入前一個付費手；但 {_pct(_sum(b, 'free_late'), _sum(b, 'free_merged'))} 的併入列"
         f"（幾乎都是 `22071_card_1.0.0_4` 免費卡）晚於付費手 {gap:.0f} 分鐘以上，疑為獨立發放"],
        ["BUY 手判定", "以 `game_mode` 位元 16 判定；付費手的 game_mode 只有 1、3、19"],
        ["讀取範圍", "後段不足，已截尾" if back_short else "完整（R7 通過）"],
        ["標籤涵蓋週數", f"標籤可回溯到 2026-03-02；"
         f"無紀錄（歸為新手）{no_record['member_account'].n_unique():,} 位玩家、{no_record.height:,} 個決策點"],
        ["分群樣本（8.1）", "訓練期間樣本 < `min_segment_train_n` 的群沿用 all 的策略："
         + "；".join(f"`{col}` {'、'.join(small) or '無'}" for col, small in _small_segments(train_ds, min_n).items())],
        ["樣本是否足夠", "葉子 × 版本樣本是否足夠見各實驗頁面"],
    ]


def write_build_report(rp, cfg: dict, train_days: list[str], valid_days: list[str], day_metas: dict[str, dict],
                       builds: dict[str, dict], ds: pl.DataFrame, notes: list[str]) -> str:
    """本 run 的 build 報告：逐日統計彙總本 run 的日子（day_metas），表格依 ds（本 run 的決策點）。"""
    days = [*train_days, *valid_days]
    role = {**{d: "訓練" for d in train_days}, **{d: "驗證" for d in valid_days}}
    stats = {d: day_metas[d]["stats"] for d in days}
    b = list(stats.values())
    provisional = any(m["provisional_arm"] for m in day_metas.values())
    back_short = any(x["stage1"]["read_plan"]["back_short"] for x in builds.values())
    labels_cfg = cfg["labels"]
    valid_text = _span(valid_days)
    day = {d: {k: stats[d].get(k, 0) for k in ("spins", "players", "buy_spins", "sum_bet", "sum_prize")} for d in days}
    outcomes = {d: Counter(stats[d].get("outcome_counts", {})) for d in days}
    arms = Counter()
    for st in b:
        arms.update(st.get("arm_counts", {}))
    funnel_keys = ["on_days", "after_censor", "after_session_start", "after_dominance"]
    funnel = {k: _sum(b, "funnel", k) for k in funnel_keys}
    dom = {s: {k: _sum(b, "dominance", s, k) for k in ("n", "churn", "rtp_sum", "rtp_n")} for s in ("meet", "miss")}
    w = {k: _sum(b, "weights", k) for k in ("player_days", "player_days_capped", "n", "churn_sum", "w_sum", "wy_sum")}

    train_ds = ds.filter(pl.col("session_date").is_in([date.fromisoformat(d) for d in train_days]))
    n_ds = ds.height
    w_idx = ds["week_index"].max()
    w_date = date.fromisocalendar(w_idx // 100, w_idx % 100, 1)
    age = ds.filter(pl.col("TAG_week_index").is_not_null()).with_columns(
        age=pl.col("TAG_week_index").map_elements(
            lambda i: (w_date - date.fromisocalendar(i // 100, i % 100, 1)).days // 7, return_dtype=pl.Int64)
    ).group_by("age").agg(players=pl.col("member_account").n_unique(), n=pl.len()).sort("age")
    nulls = ds.select([pl.col(f).null_count().alias(f) for f in FEATURES]).row(0, named=True)
    flags = {f: float(ds[f].mean()) for f in ("GP_flag_dryLong", "NGP_flag_coldLoss", "SY_isBuyFeature")}
    min_n = int(cfg["splits"]["min_segment_train_n"])
    built_by = Counter(day_metas[d]["build_id"] for d in days)

    labels = {
        "on_days": f"session 起點在訓練期間 {_span(train_days)} 或驗證期間 {valid_text}（跨午夜的 session 整段歸起點那天）",
        "after_censor": f"未截尾（session 在起點隔天 0 點 + {float(cfg['data'].get('post_hours', 0)):.0f}h 前結束，"
                        f"且結束 + {float(labels_cfg['return_window_hours']):.0f}h 都在讀取範圍內；看不完的 session 整段不抽）",
        "after_session_start": f"session 起點完整（讀取範圍起點前 {float(labels_cfg['session_gap_minutes']):.0f} 分鐘內沒有付費手；"
                               f"起點被切到的 session {_sum(b, 'truncated_sessions'):,.0f} 個）",
        "after_dominance": f"主版本佔比 ≥ {labels_cfg['arm_dominance']}",
    }
    rtp = lambda s: dom[s]["rtp_sum"] / dom[s]["rtp_n"] if dom[s]["rtp_n"] else float("nan")  # noqa: E731
    cand = dom["meet"]["n"] + dom["miss"]["n"]

    parts = [
        "# build 報告（建模前資料）",
        "",
        f"- 商戶 **{cfg['data']['merchant']}**、幣別 **{cfg['data']['currency']}**、遊戲 **{cfg['data']['game']}**",
        f"- 訓練期間 美東 **{_span(train_days)}**、驗證期間 **{valid_text}**；決策點依 `session_date`（session 起點的美東日期）歸屬"
        "，跨午夜的 session 整段歸起點那天",
        f"- dataset：`{rel(rp.store.dataset_dir)}/{{日期}}.parquet`（一天一檔、所有 run 共用），本 run {len(days)} 天、"
        f"{n_ds:,} 列 × {ds.width} 欄；來自 build " + "、".join(f"`{k}`（{n} 天）" for k, n in built_by.items()),
        f"- arm：{'⚠️ **暫代模式**（version_map 未填，以 game_version 當 arm）' if provisional else '依 version_map 對應'}",
        f"- 標籤：`{list(builds.values())[-1]['tag_dir']}`",
        "",
        "## 資料說明",
        "",
        *[f"- {n}" for n in notes],
        "",
        "## 各日概況（依每手的美東日期）",
        "",
        _table(["", "日期", "付費手", "玩家", "BUY 手", "RTP"],
               [[role[d], d, f"{day[d]['spins']:,}", f"{day[d]['players']:,}",
                 f"{day[d]['buy_spins']:,}（{_pct(day[d]['buy_spins'], day[d]['spins'])}）",
                 f"{day[d]['sum_prize'] / day[d]['sum_bet']:.4f}" if day[d]["sum_bet"] else "—"] for d in days]),
        "",
        _table(["倍數區間", *[f"{role[d]} {d[5:]}" for d in days]],
               [[o, *[_pct(outcomes[d].get(o, 0), day[d]["spins"]) for d in days]] for o in OUTCOMES]),
        "",
        "## 抽決策點（5.4）",
        "",
        _table(["步驟", "剩餘筆數"], [[labels[k], f"{funnel[k]:,}"] for k in funnel_keys]),
        "",
        f"截尾刪除 {funnel['on_days'] - funnel['after_censor']:,} 筆"
        f"（{_pct(funnel['on_days'] - funnel['after_censor'], funnel['on_days'])}）。",
        "",
        _churn_table(ds, "session_date", "session_date（訓練期間 → 訓練；驗證期間 → 驗證）"),
        "",
        "### 主版本條件：符合與不符合者比較",
        "",
        _table(["", "決策點", "佔候選", "流失率", "平均今日 RTP（GP_dailyRtp）"],
               [["符合（抽入）", f"{dom['meet']['n']:,}", _pct(dom["meet"]["n"], cand), _pct(dom["meet"]["churn"], dom["meet"]["n"]), f"{rtp('meet'):.4f}"],
                ["不符合", f"{dom['miss']['n']:,}", _pct(dom["miss"]["n"], cand), _pct(dom["miss"]["churn"], dom["miss"]["n"]), f"{rtp('miss'):.4f}"]]),
        "",
        "## 玩家 × 日權重（5.5）",
        "",
        f"- 樣本 {w['n']:,}、玩家 {ds['member_account'].n_unique():,}；流失率 未加權 {_pct(w['churn_sum'], w['n'])}、"
        f"以 `w_cap` 加權 {_pct(w['wy_sum'], w['w_sum'])}",
        f"- `w_cap` < 1 的玩家 × 日：{w['player_days_capped']:,} / {w['player_days']:,} = {_pct(w['player_days_capped'], w['player_days'])}；"
        f"權重總和 {w['w_sum']:,.0f}（= 有效樣本上限）",
        "",
        "### arm 分佈（決策點，arm = 窗口主版本）",
        "",
        _table(["arm", "決策點", "佔比"],
               [[f"`{a}`", f"{arms[a]:,}", _pct(arms[a], sum(arms.values()))]
                for a in sorted(arms, key=lambda x: (x == "None", x))]),
        "",
        "## 慢層標籤（第 7 節）",
        "",
        _churn_table(ds, "TAG_source", "標籤來源（TAG_source）"),
        "",
        f"最新一筆距決策週的週數（1 = W − 1 週；{TAG_SOURCE_PREV}／{TAG_SOURCE_OLDER}／{TAG_SOURCE_NONE}）：",
        "",
        _table(["週數", "玩家數", "決策點"], [[r["age"], f"{r['players']:,}", f"{r['n']:,}"] for r in age.iter_rows(named=True)]),
        "",
        _churn_table(ds, "TAG_lifecycle", "TAG_lifecycle（= SEG_lifecycle）"),
        "",
        _churn_table(ds, "TAG_value_combined", "TAG_value_combined"),
        "",
        "## 分群（8.1）",
        "",
        f"split = lifecycle／value；訓練期間樣本 < {min_n:,} 的群不訓練，沿用 all 的策略"
        "（value 依 `splits.groups` 只建模低／中／高，不受此限）。",
        "",
        *[line for col in SEG_COLUMNS for line in (_segment_table(train_ds, col, min_n), "")],
        "",
        "## 特徵（第 6 節）",
        "",
        f"旗標觸發比例：`GP_flag_dryLong` {flags['GP_flag_dryLong'] * 100:.2f}%、`NGP_flag_coldLoss` "
        f"{flags['NGP_flag_coldLoss'] * 100:.2f}%、`SY_isBuyFeature` {flags['SY_isBuyFeature'] * 100:.2f}%",
        "",
        _table(["特徵", "缺值", "佔比"], [[f"`{f}`", f"{nulls[f]:,}", _pct(nulls[f], n_ds)] for f in FEATURES]),
        "",
        "## 待確認事項（SPEC 第 14 節＋本次發現）",
        "",
        _table(["問題", "目前處理"], pending_items(cfg, provisional, b, back_short, ds, train_ds)),
        "",
        "## 耗時與記憶體（各次 build）",
        "",
        _table(["build", "建置日子", "第一段", "第二段", "單桶最長", "worker 峰值記憶體"],
               [[f"`{x['build_id']}`", _span(sorted({x["first_day"], x["last_day"]})),
                 f"{x['stage1']['seconds']}s（峰值 {x['stage1']['peak_rss_mb']:.0f} MB）",
                 f"{x['stage2']['seconds']}s（{len(x['stage2']['buckets'])} 桶）",
                 f"{max(r['seconds'] for r in x['stage2']['buckets'])}s",
                 f"{max(r['peak_rss_mb'] for r in x['stage2']['buckets']):.0f} MB"] for x in builds.values()]),
    ]
    text = "\n".join(parts) + "\n"
    rp.build_dir.mkdir(parents=True, exist_ok=True)
    rp.build_report.write_text(text, encoding="utf-8")
    return text

# -*- coding: utf-8 -*-
"""HTML 頁面：決策樹 SVG、葉子 × 版本熱圖、訓練日／驗證日設定（合成資料）。"""
from __future__ import annotations

import pandas as pd
import pytest
import yaml

from feel_engine.html import pages
from feel_engine.pipeline.data import load_config
from feel_engine import paths
from feel_engine.paths import CONFIG_PATH

NODES = [
    {"id": 0, "left": 1, "right": 2, "n": 1000, "churn": 0.1, "text": ["今日已玩手數", "≤ 35？"], "left_label": "是", "right_label": "否"},
    {"id": 1, "left": -1, "right": -1, "n": 400, "churn": 0.2, "leaf": "高/L1"},
    {"id": 2, "left": -1, "right": -1, "n": 600, "churn": 0.05, "leaf": "高/L2"},
]


def test_tree_svg_shows_conditions_leaf_ids_and_best_version():
    leaves = {1: {"name": "L1", "best": "_2", "color": "#009E73", "conf": "中"},
              2: {"name": "L2", "best": None, "color": None, "conf": pages.NOT_COMPARABLE}}
    svg = pages.tree_svg(NODES, leaves)
    for text in ("今日已玩手數", "≤ 35？", "L1", "最佳 _2", "信心 中", "L2", pages.NOT_COMPARABLE, "是", "否", "400 筆｜留存 80.0%"):
        assert text in svg
    merged = pages.tree_svg(NODES, {0: {"name": "L1–L2", "best": "_0", "color": "#0072B2", "conf": "低"}}, collapsed={0})
    assert "L1–L2" in merged and "今日已玩手數" not in merged                # 合併後根節點畫成葉子


def stats_best():
    stats = pd.DataFrame([
        {"葉子": "高/L1", "arm": "a_0", "決策點": 500, "n": 480, "範圍外": 20, "原始留存率": 0.9, "加權留存率": 0.91,
         "max_smd": 0.1, "樣本足夠": True, "平衡": True},
        {"葉子": "高/L1", "arm": "a_2", "決策點": 100, "n": 100, "範圍外": 0, "原始留存率": 0.95, "加權留存率": 0.96,
         "max_smd": 0.3, "樣本足夠": False, "平衡": False},
    ])
    best = pd.DataFrame([{"葉子": "高/L1", "葉子決策點": 600, "葉子玩家數": 87, "葉子留存率": 0.915, "可比較比例": 0.96, "樣本足夠 arm 數": 1, "最高留存 arm": "a_0", "信心": "低",
                          "判讀": "只有一個 arm 樣本足夠，無從比較"}])
    return stats, best


def test_leaf_arm_table_grays_insufficient_and_marks_incomparable():
    stats, best = stats_best()
    colors = pages.arm_colors(["a_0", "a_2"])
    html = pages.leaf_arm_table(best, stats, {}, ["a_0", "a_2"], colors, min_n=300, smd_max=0.2)
    assert "樣本不足" in html and "100 筆" in html and "91.0%" in html
    assert "以玩家為單位重抽" in html
    assert "★" not in html.split("<tbody>")[1].split("</tbody>")[0]                           # 只有一個版本足夠 → 不標最高
    assert pages.NOT_COMPARABLE in html and ">L1<" in html                   # 只顯示葉子編號，不寫規則
    assert "整體留存" in html and "91.5%" in html
    assert "驗證玩家數" in html and ">87<" in html
    assert "可比較比例" not in html


def test_config_requires_train_period_before_valid_day_and_matching_codes(tmp_path):
    from feel_engine.pipeline.data import run_name, train_valid_days

    base = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    path = tmp_path / "c.yaml"

    def load(**data):
        cfg = {**base, "data": {**base["data"], **data}}
        path.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
        return load_config(path)

    cfg = load(train_start="2026-08-25", train_end="2026-08-28", valid_start="2026-08-29", valid_end="2026-08-29")
    assert train_valid_days(cfg) == (["2026-08-25", "2026-08-26", "2026-08-27", "2026-08-28"], ["2026-08-29"])
    assert run_name(cfg) == "TCG_PHP_22071/train_20260825-20260828_valid_20260829"          # 驗證一天：只寫一個日期
    cfg = load(train_start="2026-08-25", train_end="2026-08-28", valid_start="2026-08-29", valid_end="2026-08-30")
    assert train_valid_days(cfg)[1] == ["2026-08-29", "2026-08-30"]
    assert run_name(cfg) == "TCG_PHP_22071/train_20260825-20260828_valid_20260829-20260830"
    for start, end, vs, ve in (("2026-08-25", "2026-08-29", "2026-08-29", "2026-08-29"),
                               ("2026-08-28", "2026-08-25", "2026-08-29", "2026-08-29"),
                               ("2026-08-25", "2026-08-28", "2026-08-30", "2026-08-29")):
        with pytest.raises(ValueError, match="train_end"):
            load(train_start=start, train_end=end, valid_start=vs, valid_end=ve)
    from feel_engine.paths import bet_records_dir, run_paths, tag_dir
    cfg = load(currency="USD", game="99999")                               # 快取與輸出資料夾由代號決定，不含日期
    assert run_name(cfg).startswith("TCG_USD_99999/") and run_paths(run_name(cfg)).store.key == "TCG_USD_99999"
    assert bet_records_dir("TCG", "22071").name == "bet_records_TCG_22071_tz+0"
    assert tag_dir("TCG", "PHP", "22071").name == "dwt_player_game_tag_wi_test_TCG_PHP_22071_lag1w_asof"
    with pytest.raises(ValueError, match="bet_records_dir"):
        load(bet_records_dir="bet_records_TCG_22071_20260801_20260926_tz+0")   # 舊設定：資料夾名稱含日期
    with pytest.raises(ValueError, match="tag_dir"):
        load(tag_dir="dwt_player_game_tag_wi_test_TCG_PHP_22071_20260511_20260926_lag1w_asof")
    with pytest.raises(ValueError, match="train_start"):
        load(train_date="2026-08-28")                                      # 舊設定
    with pytest.raises(ValueError, match="valid_start"):
        load(valid_date="2026-08-29")


def test_summary_page_shows_auc_and_confidence_with_scoring_basis():
    rows = [{"label": "#1 all／無", "href": "all_tag_none.html", "auc": 0.80, "lo": 0.79, "hi": 0.81, "diff": None,
             "diff_lo": None, "diff_hi": None, "verdict": "基準",
             "conf": {v: {"share": {"高": 0.2, "中": 0.4, "低": 0.3, "無法比較": 0.1}, "score": 40.0}
                      for v in ("orig", "merged")}},
            {"label": "#2 value／無", "href": "value_tag_none.html", "auc": 0.82, "lo": 0.81, "hi": 0.83, "diff": 0.02,
             "diff_lo": 0.01, "diff_hi": 0.03, "verdict": "較佳", "conf": None}]
    html = pages.summary_page("比較", [], rows, 0.80, ["AUC 說明"], ["信心分數 = 100 × (高佔比 ＋ 0.5 × 中佔比)"])
    assert "<th>信心分數</th>" in html and "原始" not in html and ">40<" in html and "尚未 evaluate" in html
    assert "評分依據" in html and "0.5 × 中佔比" in html


def test_keep_current_leaf_shows_version_without_star():
    stats, best = stats_best()
    best = best.assign(信心="維持現狀", 最多版本="a_0")
    html = pages.leaf_arm_table(best, stats, {}, ["a_0", "a_2"], pages.arm_colors(["a_0", "a_2"]), min_n=300, smd_max=0.2)
    body = html.split("<tbody>")[1].split("</tbody>")[0]
    assert "維持現狀_0" in body and "c-維持現狀" in body and "★" not in body
    svg = pages.tree_svg(NODES, {1: {"name": "L1", "best": "_0", "color": "#0072B2", "conf": "維持現狀", "retention": 0.95},
                                 2: {"name": "L2", "best": None, "color": None, "conf": pages.NOT_COMPARABLE}})
    assert "維持現狀_0" in svg and "整體留存 95.0%" in svg


def test_experiment_page_tree_fits_screen_with_zoom_buttons():
    svg = pages.tree_svg(NODES, {1: {"name": "L1", "best": None, "color": None, "conf": pages.NOT_COMPARABLE},
                                 2: {"name": "L2", "best": None, "color": None, "conf": pages.NOT_COMPARABLE}})
    assert 'data-w="' in svg and 'data-h="' in svg                         # 原始大小，縮放用
    view = [{"title": "決策樹", "sub": "", "svg": svg, "table": ""}]
    html = pages.experiment_page("實驗", [], {"orig": view, "merged": view}, [])
    for text in ('data-act="fit"', 'data-act="in"', 'data-act="out"', 'data-act="full"', "fitScale", "<script>",
                 "合併相近建議", 'class="view simple"'):
        assert text in html


def test_experiment_page_switches_trees_without_all_or_experiment_links():
    view = [{"title": f"樹：{g}", "sub": "", "svg": "<svg data-w='1' data-h='1'></svg>", "table": ""} for g in ("活躍", "新手")]
    html = pages.experiment_page("實驗", [], {"orig": view, "merged": view, "simple": view}, [])
    assert 'data-tree="活躍"' in html and 'data-tree="新手"' in html and "切換決策樹" in html
    assert "__all__" not in html and "切換實驗" not in html
    single = pages.experiment_page("實驗", [], {"orig": view[:1], "merged": view[:1], "simple": view[:1]}, [])
    assert "切換決策樹：" not in single                                          # 只有一棵樹不顯示按鈕



FIVE_PLACED = {"version": "a_1", "tree": "t", "leaf": "t/L1", "base_version": "a_0",
               "conditions": ["餘額÷下注額 ≤ 40", "今日已玩手數 ≤ 20"], "decision_points": 120, "leaf_decision_points": 400,
               "share": 0.012, "retention_v": 0.92, "retention_base": 0.93, "gap_pp": 1.0, "gap_ci": [0.2, 1.8],
               "impact_pp": 0.012, "impact_ci": [0.002, 0.022], "comparable_share": 0.8, "max_smd": 0.12,
               "balanced": True, "ess": 850.0, "status": "選中"}


def test_five_section_lists_placed_unplaced_and_candidates():
    colors = pages.arm_colors(["a_0", "a_1", "a_3"])
    no_range = {"version": "a_3", "tree": "t", "leaf": "t/L2", "base_version": "a_0", "gap_pp": None, "status": "沒有可比區"}
    html = pages.five_section([FIVE_PLACED], [{"version": "a_3", "reason": "最小差距 3.4 pp（t/L2）"}],
                              [FIVE_PLACED, no_range], colors)
    for text in ("_1 → t/L1", "_3 找不到合適位置（最小差距 3.4 pp（t/L2））", "t/L1：餘額÷下注額 ≤ 40 且 今日已玩手數 ≤ 20",
                 "120（1.2%）", "92.00%", "93.00%", "1.00 [0.20, 1.80]", "-0.012 [-0.022, -0.002]", "其他候選（2 個）",
                 "沒有可比區", "可比較比例 80%", "max_smd 0.12（平衡）", "整體留存變化 = 佔比 × 差距"):
        assert text in html
    assert "不需補位" in pages.five_section([], [], [], colors)


def test_tree_svg_shows_fill_leaf_and_validation_stat():
    nodes = [*NODES[:2], {"id": 2, "left": -1, "right": -1, "leaf": "高/L2", "stat": "驗證 280 筆"}]
    svg = pages.tree_svg(nodes, {1: {"name": "L1", "best": "_0", "color": "#0072B2", "conf": "中", "retention": 0.93},
                                 2: {"name": "L2 補位", "best": "_1", "color": "#E69F00", "conf": pages.FILL,
                                     "retention": 0.92, "detail": "比 _0 低 1.0 pp"}})
    for text in ("L2 補位", "補位 _1 · 92.0%", "比 _0 低 1.0 pp", "驗證 280 筆", "最佳 _0 · 93.0%"):
        assert text in svg


def test_experiment_page_adds_five_tab_only_when_given():
    view = [{"title": "決策樹", "sub": "", "svg": "<svg data-w='1' data-h='1'></svg>", "table": ""}]
    base = {"orig": view, "merged": view, "simple": view}
    assert "五個體感" not in pages.experiment_page("實驗", [], base, [])
    html = pages.experiment_page("實驗", [], {**base, "five": view}, [], five_results="<p>補位摘要</p>")
    assert 'id="v-five"' in html and 'class="view five"' in html and "補位摘要" in html
    assert "#v-five:checked ~ .view.five" in html
    assert html.index("補位摘要") > html.rindex('<section class="tree-sec"')


def test_pages_show_experiment_facts_as_bullets():
    view = [{"title": "決策樹", "sub": "", "svg": "<svg data-w='1' data-h='1'></svg>", "table": ""}]
    html = pages.experiment_page("實驗", ["<a href='s.html'>←</a>"], {"orig": view, "merged": view, "simple": view}, [],
                                 facts=["商戶／幣別／遊戲：TCG／PHP／22071", "驗證期間：2026-09-14～2026-09-20"])
    assert '<ul class="meta">' in html and "<li>商戶／幣別／遊戲：TCG／PHP／22071</li>" in html
    assert html.index("<a href='s.html'>") < html.index('<ul class="meta">') < html.index('<div class="tabs">')
    rows = [{"label": "#1 all／無", "href": "all_tag_none.html", "auc": 0.80, "lo": 0.79, "hi": 0.81, "diff": None,
             "diff_lo": None, "diff_hi": None, "verdict": "基準", "conf": None}]
    summary = pages.summary_page("比較", [], rows, 0.80, [], [], facts=["共同樣本：10 個決策點"])
    assert "<li>共同樣本：10 個決策點</li>" in summary


def test_five_section_marks_partial_placement():
    colors = pages.arm_colors(["a_0", "a_1"])
    html = pages.five_section([FIVE_PLACED], [{"version": "a_1", "reason": "沒有可比區", "placed_n": 1}], [FIVE_PLACED], colors)
    assert "_1 只補到 1 片（其他樹：沒有可比區）" in html and "找不到合適位置" not in html


def test_leaf_arm_table_shows_players_and_decision_points():
    stats, best = stats_best()
    html = pages.leaf_arm_table(best, stats.assign(範圍內玩家數=[50, 20]), {}, ["a_0", "a_2"],
                                pages.arm_colors(["a_0", "a_2"]), min_n=300, smd_max=0.2)
    assert "50 人／480 筆" in html and "20 人／100 筆" in html          # 顯著性以玩家為單位，同時列人數與筆數


def test_five_section_lists_trees_without_placement():
    colors = pages.arm_colors(["a_0", "a_1"])
    html = pages.five_section([FIVE_PLACED], [], [FIVE_PLACED], colors,
                              [{"tree": "下滑", "reason": "最小差距 2.6 pp（_1，下滑/L3）"}, {"tree": "一般", "reason": "沒有可比區"}])
    assert "樹 下滑 沒有補位（最小差距 2.6 pp（_1，下滑/L3））" in html and "樹 一般 沒有補位（沒有可比區）" in html
    assert "不需補位" in pages.five_section([], [], [], colors, [])


def test_variant_paths_keep_official_results_apart():
    rp = paths.run_paths("TCG_PHP_22071/train_a_valid_b")
    vp = paths.VariantPaths(rp.name, variant="new")
    assert vp.experiment_html("lifecycle", "none").name == "lifecycle_tag_none_new.html"
    assert vp.model_predictions("lifecycle", "none").parent == rp.output_dir / "variants" / "lifecycle_tag_none_new"
    assert vp.five_feelings("lifecycle", "none").parent.parent.name == "variants" and vp.store == rp.store
    assert rp.experiment_html("lifecycle", "none").name == "lifecycle_tag_none.html"   # 正式實驗不受影響


def test_five_section_marks_placement_outside_thresholds():
    colors = pages.arm_colors(["a_0", "a_1"])
    html = pages.five_section([dict(FIVE_PLACED, outside="差距 4.0 pp > 2 pp")], [], [FIVE_PLACED], colors)
    assert "_1 → t/L1（超過門檻）" in html and "⚠️ 超過門檻：差距 4.0 pp &gt; 2 pp" in html


def test_experiment_page_two_five_feelings_tabs():
    view = [{"title": "決策樹", "sub": "", "svg": "<svg data-w='1' data-h='1'></svg>", "table": ""}]
    base = {"orig": view, "merged": view, "simple": view}
    html = pages.experiment_page("實驗", [], {**base, "five": view, "five2": view}, [],
                                 five_results={"five": "<p>補位一</p>", "five2": "<p>補位二</p>"},
                                 five_labels={"five": "五個體感1", "five2": "五個體感2"})
    assert 'id="v-five"' in html and 'id="v-five2"' in html and "五個體感1" in html and "五個體感2" in html
    assert '<div class="view five2">' in html and "補位一" in html and "補位二" in html
    assert html.count("#v-five2:checked") == 1                              # 分頁樣式只放一次

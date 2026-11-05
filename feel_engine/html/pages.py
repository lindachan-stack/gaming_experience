# -*- coding: utf-8 -*-
"""
給人看的 HTML 頁面（單檔、不連外部資源，瀏覽器直接開）：

  experiment_page：每個實驗一頁 result/{run}/{split}_tag_{tags}.html
                   每棵樹：決策樹（大字；葉子標最佳版本與信心）＋ 葉子 × 版本留存率熱圖；
                   分「原始葉子」「合併無法比較的葉子」「合併相近建議」（與「五個體感」）分頁（evaluate 產生）
  summary_page   ：實驗比較 result/{run}/summary.html：AUC ＋ 信心分數與評分依據（compare／evaluate 產生）
"""
from __future__ import annotations

import html
import math

import pandas as pd

ARM_COLORS = ("#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#8C6D31")   # Okabe-Ito 色盤
NOT_COMPARABLE = "無法比較"
KEEP_CURRENT = "維持現狀"                     # 同一版本佔葉子超過 dominant_share，沒有別的版本可比；畫面顯示「維持現狀_0」
FILL = "補位"                                 # 五個體感補位的葉子（evaluation/five.py）
FIVE_CSS = ("<style>#v-five:checked ~ .view.five, #v-five2:checked ~ .view.five2 { display:block; } details { margin:10px 0 18px; } "
            "details summary { cursor:pointer; font-weight:600; }</style>")   # 只放在有第 4 個分頁的頁面，其他頁面不變
GRAY = "#9CA3AF"

CSS = """
:root { --bg:#ffffff; --surface:#f8fafc; --text:#111827; --muted:#6b7280; --border:#d1d5db; --gray:#e5e7eb;
        --heat:37,99,235; --good:#15803d; --mid:#b45309; --bad:#b91c1c; --na:#6b7280; --keep:#0e7490; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
        --bg:#0f172a; --surface:#1e293b; --text:#e5e7eb; --muted:#9ca3af; --border:#475569; --gray:#334155;
        --heat:96,165,250; --good:#4ade80; --mid:#fbbf24; --bad:#f87171; --na:#9ca3af; --keep:#22d3ee; } }
:root[data-theme="dark"] { --bg:#0f172a; --surface:#1e293b; --text:#e5e7eb; --muted:#9ca3af; --border:#475569;
        --gray:#334155; --heat:96,165,250; --good:#4ade80; --mid:#fbbf24; --bad:#f87171; --na:#9ca3af; --keep:#22d3ee; }
* { box-sizing: border-box; }
body { margin:0; padding:24px 16px 48px; background:var(--bg); color:var(--text); font-size:16px; line-height:1.55;
       font-family:"PingFang TC","Noto Sans TC","Microsoft JhengHei","Heiti TC",system-ui,sans-serif; }
main { max-width:1400px; margin:0 auto; }
h1 { font-size:26px; margin:0 0 6px; } h2 { font-size:21px; margin:36px 0 10px; } h3 { font-size:18px; margin:22px 0 8px; }
a { color:rgb(var(--heat)); }
.meta { color:var(--muted); margin:0 0 4px; }
.note { color:var(--muted); font-size:14px; margin:6px 0; }
.figure { overflow-x:auto; border:1px solid var(--border); border-radius:10px; background:var(--surface); padding:8px; margin:10px 0; }
.figure svg { display:block; }
.zoom { display:flex; gap:6px; align-items:center; margin:6px 0 0; flex-wrap:wrap; }
.zoom button { font:inherit; font-size:14px; padding:3px 12px; border:1px solid var(--border); border-radius:8px;
               background:var(--surface); color:var(--text); cursor:pointer; }
.zoom button:hover { border-color:rgb(var(--heat)); }
.zoom .zoom-pct { color:var(--muted); font-size:14px; min-width:3.5em; }
.tree-switch { display:flex; gap:6px; align-items:center; flex-wrap:wrap; margin:14px 0 4px; }
.tree-switch button { font:inherit; font-size:15px; padding:4px 14px; border:1px solid var(--border); border-radius:999px;
                      background:var(--surface); color:var(--text); cursor:pointer; }
.tree-switch button.on { background:rgb(var(--heat)); border-color:rgb(var(--heat)); color:#fff; font-weight:700; }
svg text { fill:var(--text); }
svg .edge { stroke:var(--muted); stroke-width:1.6; fill:none; }
svg .edge-label { font-size:14px; fill:var(--muted); paint-order:stroke; stroke:var(--surface); stroke-width:5px; }
svg .box-int { fill:var(--bg); stroke:var(--border); stroke-width:1.5; }
svg .small { fill:var(--muted); }
.table-wrap { overflow-x:auto; }
table { border-collapse:collapse; font-size:16px; margin:8px 0; }
th, td { border:1px solid var(--border); padding:6px 12px; text-align:center; white-space:nowrap; }
th { background:var(--surface); font-weight:600; }
td.leaf { font-weight:700; text-align:left; }
td.cell .v { font-size:17px; font-weight:600; }
td.cell .n { display:block; font-size:12px; color:var(--muted); }
td.na { background:var(--gray); color:var(--muted); }
td.best { outline:3px solid var(--text); outline-offset:-3px; }
.badge { display:inline-block; padding:1px 10px; border-radius:999px; font-weight:700; font-size:15px; color:#fff; }
.c-高 { background:var(--good); } .c-中 { background:var(--mid); } .c-低 { background:var(--bad); } .c-無法比較 { background:var(--na); } .c-維持現狀 { background:var(--keep); }
.swatch { display:inline-block; width:12px; height:12px; border-radius:3px; vertical-align:-1px; margin-right:4px; }
.legend { font-size:14px; color:var(--muted); margin:4px 0 0; }
.legend span { margin-right:14px; white-space:nowrap; }
.tabs { margin:18px 0 0; }
.tabs input { position:absolute; opacity:0; }
.tabs label { display:inline-block; padding:8px 18px; border:1px solid var(--border); border-bottom:none; border-radius:10px 10px 0 0;
              cursor:pointer; color:var(--muted); font-weight:600; margin-right:4px; background:var(--surface); }
.tabs input:checked + label { color:var(--text); background:var(--bg); box-shadow:inset 0 3px 0 rgb(var(--heat)); }
.view { display:none; border-top:1px solid var(--border); padding-top:4px; }
#v-orig:checked ~ .view.orig, #v-merged:checked ~ .view.merged, #v-simple:checked ~ .view.simple { display:block; }
.verdict-better { color:var(--good); font-weight:700; } .verdict-worse { color:var(--bad); font-weight:700; }
"""


def esc(v) -> str:
    return html.escape(str(v), quote=True)


def arm_short(arm) -> str:
    """22071_1.0.0_2 → _2。"""
    return "_" + arm.rsplit("_", 1)[-1] if isinstance(arm, str) and "_" in arm else str(arm)


def arm_colors(arms: list[str]) -> dict[str, str]:
    return {a: ARM_COLORS[i % len(ARM_COLORS)] for i, a in enumerate(sorted(arms))}


def short_leaf(name: str) -> str:
    """「高/L3」→「L3」（樹名已在小標題）。"""
    return str(name).split("/", 1)[-1]


def period(start, end) -> str:
    """日期區間的顯示：同一天只寫一次。"""
    return str(start) if str(start) == str(end) else f"{start}～{end}"


def _pct(v, digits: int = 1) -> str:
    return "—" if v is None or v != v else f"{v * 100:.{digits}f}%"


def page(title: str, body: str, script: str = "") -> str:
    return (f'<!doctype html>\n<html lang="zh-Hant">\n<head>\n<meta charset="utf-8">\n'
            f'<meta name="viewport" content="width=device-width, initial-scale=1">\n<title>{esc(title)}</title>\n'
            f"<style>{CSS}</style>\n</head>\n<body>\n<main>\n{body}\n</main>\n"
            + (f"<script>{script}</script>\n" if script else "") + "</body>\n</html>\n")


# 決策樹縮放：預設縮到一個畫面內（寬、高都放得下）；按鈕可放大、縮小、回原始大小；切換分頁或調整視窗時重新套用
ZOOM_JS = """
(function () {
  var STEP = 1.25;
  function size(fig) { var svg = fig.querySelector('svg'); return [+svg.dataset.w, +svg.dataset.h, svg]; }
  function fitScale(fig) {
    var d = size(fig), main = document.querySelector('main');
    var availW = main.clientWidth - 24, availH = window.innerHeight * 0.85;
    return Math.min(1, availW / d[0], availH / d[1]);
  }
  function apply(fig, s) {
    var d = size(fig);
    d[2].style.width = (d[0] * s) + 'px';
    d[2].style.height = (d[1] * s) + 'px';
    fig.dataset.scale = s;
    fig.querySelector('.zoom-pct').textContent = Math.round(s * 100) + '%';
  }
  var figs = document.querySelectorAll('.tree-fig');
  figs.forEach(function (fig) {
    fig.dataset.fit = '1';
    apply(fig, fitScale(fig));
    fig.querySelectorAll('button').forEach(function (b) {
      b.addEventListener('click', function () {
        var act = b.dataset.act, s = +fig.dataset.scale;
        fig.dataset.fit = act === 'fit' ? '1' : '0';
        apply(fig, act === 'fit' ? fitScale(fig) : act === 'full' ? 1 : act === 'in' ? Math.min(1, s * STEP) : s / STEP);
      });
    });
  });
  function refit() { figs.forEach(function (fig) { if (fig.dataset.fit === '1') apply(fig, fitScale(fig)); }); }
  window.refitTrees = refit;
  window.addEventListener('resize', refit);
  document.querySelectorAll('.tabs input').forEach(function (r) { r.addEventListener('change', refit); });
})();
"""


# ---------------------------------------------------------------------------
# 決策樹（SVG）
# ---------------------------------------------------------------------------
def text_width(s: str, size: float) -> float:
    """估計字寬：中日韓全形字 ≈ 1 em，其他 ≈ 0.6 em。"""
    return sum(size * (1.0 if ord(ch) >= 0x2E80 else 0.6) for ch in str(s))


def tree_svg(nodes: list[dict], leaves: dict[int, dict], collapsed: set[int] | None = None) -> str:
    """nodes：tree_rules.json 的節點（id、left、right、text、left_label、right_label、n、churn）；
    leaves：畫成葉子的節點 id → {name, best, color, conf}；collapsed：合併後當成葉子的內部節點。"""
    collapsed = collapsed or set()
    by_id = {n["id"]: n for n in nodes}
    is_leaf = lambda i: by_id[i]["left"] == -1 or i in collapsed      # noqa: E731
    PAD, GAP, MARGIN = 12, 18, 16
    pos: dict[int, tuple[float, int]] = {}
    count = [0]

    def place(i: int, depth: int) -> float:
        if is_leaf(i):
            x = float(count[0])
            count[0] += 1
        else:
            x = (place(by_id[i]["left"], depth + 1) + place(by_id[i]["right"], depth + 1)) / 2
        pos[i] = (x, depth)
        return x

    place(0, 0)

    def lines(i: int) -> list[tuple[str, int, str, str | None]]:     # (文字, 字級, 粗細, 顏色)
        n = by_id[i]
        churn = n.get("churn")
        stat = n["stat"] if "stat" in n else f"{n['n']:,} 筆｜留存 {_pct(None if churn is None else 1 - churn)}"
        if not is_leaf(i):
            return [(n["text"][0], 17, "700", None), (n["text"][1], 17, "700", None), (stat, 13, "400", "small")]
        info = leaves[i]
        ret = info.get("retention")
        if info["conf"] == FILL:                                   # 五個體感補位：子規則內給這個版本
            return [(info["name"], 19, "700", None),
                    (f"{FILL} {info['best']}" + ("" if ret is None or ret != ret else f" · {_pct(ret)}"), 18, "700",
                     info["color"]),
                    (info.get("detail", ""), 15, "600", None), (stat, 13, "400", "small")]
        if info["conf"] == KEEP_CURRENT and info.get("best"):     # 單一版本獨大：維持現狀，不是比出來的最佳
            return [(info["name"], 19, "700", None), (f"{KEEP_CURRENT}{info['best']}", 17, "700", info["color"]),
                    ("" if ret is None or ret != ret else f"整體留存 {_pct(ret)}", 15, "600", None),
                    (stat, 13, "400", "small")]
        if info.get("best"):
            return [(info["name"], 19, "700", None),
                    (f"最佳 {info['best']}" + ("" if ret is None or ret != ret else f" · {_pct(ret)}"), 18, "700", info["color"]),
                    (f"信心 {info['conf']}", 15, "600", None),
                    (stat, 13, "400", "small")]
        return [(info["name"], 19, "700", None), (NOT_COMPARABLE, 17, "700", "small"), ("", 15, "400", None),
                (stat, 13, "400", "small")]

    content = {i: lines(i) for i in pos}
    size = {i: (max(text_width(t, s) for t, s, _, _ in ls) + 2 * PAD, sum(s * 1.35 for _, s, _, _ in ls) + 2 * PAD - 4)
            for i, ls in content.items()}
    box_h = max(h for _, h in size.values())
    level_h = box_h + 58
    unit = 1.0                                                        # 相鄰兩個同層方框不重疊所需的水平單位
    for depth in {d for _, d in pos.values()}:
        row = sorted((x, i) for i, (x, d) in pos.items() if d == depth)
        for (xa, a), (xb, b) in zip(row, row[1:]):
            unit = max(unit, ((size[a][0] + size[b][0]) / 2 + GAP) / (xb - xa))
    left = min(pos[i][0] * unit - size[i][0] / 2 for i in pos)
    right = max(pos[i][0] * unit + size[i][0] / 2 for i in pos)
    width = right - left + 2 * MARGIN
    height = (max(d for _, d in pos.values()) + 1) * level_h - (level_h - box_h) + 2 * MARGIN
    cx = {i: pos[i][0] * unit - left + MARGIN for i in pos}
    top = {i: pos[i][1] * level_h + MARGIN for i in pos}

    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" height="{height:.0f}" '
           f'viewBox="0 0 {width:.0f} {height:.0f}" data-w="{width:.0f}" data-h="{height:.0f}" role="img">']
    for i in pos:                                                     # 連線與「是／否」
        if is_leaf(i):
            continue
        n = by_id[i]
        for child, label in ((n["left"], n["left_label"]), (n["right"], n["right_label"])):
            x1, y1, x2, y2 = cx[i], top[i] + box_h, cx[child], top[child]
            out.append(f'<line class="edge" x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}"/>')
            lx, ly = x1 + (x2 - x1) * 0.42, y1 + (y2 - y1) * 0.42 + 5
            out.append(f'<text class="edge-label" x="{lx:.1f}" y="{ly:.1f}" text-anchor="middle">{esc(label)}</text>')
    for i in pos:                                                     # 方框
        w, x, y = size[i][0], cx[i] - size[i][0] / 2, top[i]
        if is_leaf(i):
            color = leaves[i].get("color") or GRAY
            out.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{box_h:.1f}" rx="10" fill="{color}" '
                       f'fill-opacity="0.16" stroke="{color}" stroke-width="2.5"/>')
        else:
            out.append(f'<rect class="box-int" x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{box_h:.1f}" rx="10"/>')
        ty = y + PAD
        for text, s, weight, color in content[i]:
            ty += s * 1.15
            if text:
                style = f' class="small"' if color == "small" else (f' style="fill:{color}"' if color else "")
                out.append(f'<text x="{cx[i]:.1f}" y="{ty:.1f}" text-anchor="middle" font-size="{s}" '
                           f'font-weight="{weight}"{style}>{esc(text)}</text>')
            ty += s * 0.2
    out.append("</svg>")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# 葉子 × 版本留存率（表格熱圖）
# ---------------------------------------------------------------------------
def conf_text(conf: str, keep_arm: str | None = None) -> str:
    """畫面上的信心文字：單一版本獨大 → 「維持現狀_0」；其他照原等級（高／中／低／無法比較）。"""
    return f"{KEEP_CURRENT}{arm_short(keep_arm)}" if conf == KEEP_CURRENT and keep_arm else conf


def display_conf(row: dict) -> str:
    if row.get("信心") == KEEP_CURRENT:
        return KEEP_CURRENT
    return NOT_COMPARABLE if row["樣本足夠 arm 數"] < 2 else row["信心"]


def _people_rows(c: dict) -> str:
    """版本格子下方的小字：共同範圍內的玩家數／決策點筆數（顯著性以玩家為單位重抽，同樣筆數的人數可能差很多）。"""
    people = c.get("範圍內玩家數")
    return f"{people:,} 人／{c['n']:,} 筆" if people is not None and people == people else f"{c['n']:,} 筆"


def leaf_arm_table(best: pd.DataFrame, stats: pd.DataFrame, cis: dict, arms: list[str], colors: dict[str, str],
                   min_n: int, smd_max: float) -> str:
    """列 = 葉子、欄 = 整體留存率＋各版本加權留存率（未納入比較 → 灰色只標 n；★ = 最高）；最後兩欄 = 最佳版本、信心。"""
    cell = {(r["葉子"], r["arm"]): r for r in stats.to_dict(orient="records")}
    ok = stats[stats["樣本足夠"] & stats["加權留存率"].notna()]["加權留存率"]
    lo, hi = (float(ok.min()), float(ok.max())) if len(ok) else (0.0, 1.0)
    head = "".join(f'<th><span class="swatch" style="background:{colors[a]}"></span>{esc(arm_short(a))}</th>' for a in arms)
    rows = []
    for r in best.to_dict(orient="records"):
        leaf = r["葉子"]
        conf = display_conf(r)
        tds = [f'<td class="leaf">{esc(short_leaf(leaf))}</td>', f'<td>{r["葉子決策點"]:,}</td>',
               f'<td>{r["葉子玩家數"]:,}</td>' if "葉子玩家數" in r else '<td class="na">—</td>',
               f'<td><b>{_pct(r["葉子留存率"])}</b></td>']
        for a in arms:
            c = cell.get((leaf, a))
            if c is None:
                tds.append('<td class="na">—</td>')
                continue
            tip = (f"{arm_short(a)}｜決策點 {c['決策點']:,}、玩家 {c.get('玩家數', 0):,}（共同範圍外 {c['範圍外']:,}）｜原始留存 {_pct(c['原始留存率'], 2)}"
                   f"｜加權留存 {_pct(c['加權留存率'], 2)}")
            if (leaf, a) in cis:
                tip += f" [95% {_pct(cis[(leaf, a)][0], 2)}, {_pct(cis[(leaf, a)][1], 2)}]"
            tip += f"｜max_smd {c['max_smd']:.2f}" if c["max_smd"] == c["max_smd"] else ""
            tip += f"｜有效樣本 {c['有效樣本']:,.0f}" if c.get("有效樣本") == c.get("有效樣本") and c.get("有效樣本") is not None else ""
            if not c["樣本足夠"]:
                label = "樣本不足" if c["n"] < min_n else "未納入比較"
                tds.append(f'<td class="cell na" title="{esc(tip)}">{label}<span class="n">{_people_rows(c)}</span></td>')
                continue
            t = 0.0 if hi <= lo else (c["加權留存率"] - lo) / (hi - lo)
            is_best = a == r["最高留存 arm"] and conf not in (NOT_COMPARABLE, KEEP_CURRENT)
            mark = ("★ " if is_best else "") + _pct(c["加權留存率"])
            tds.append(f'<td class="cell{" best" if is_best else ""}" style="background:rgba(var(--heat),{0.06 + 0.5 * t:.2f})" '
                       f'title="{esc(tip)}"><span class="v">{esc(mark)}</span><span class="n">{_people_rows(c)}</span></td>')
        best_arm = r["最高留存 arm"] if r["最高留存 arm"] != "—" else None
        tds.append(f'<td><span class="swatch" style="background:{colors[best_arm]}"></span><b>{esc(arm_short(best_arm))}</b></td>'
                   if best_arm and conf not in (NOT_COMPARABLE, KEEP_CURRENT) else '<td class="na">—</td>')
        text = conf_text(conf, r.get("最多版本"))
        tds.append(f'<td title="{esc(r["判讀"])}"><span class="badge c-{esc(conf)}">{esc(text)}</span></td>')
        rows.append("<tr>" + "".join(tds) + "</tr>")
    legend = ('<p class="legend"><span>數字 = 加權留存率；下方 = 共同範圍內的玩家數／決策點筆數</span><span>★ = 最高</span>'
              '<span>灰色 = 樣本不足或未比較</span><span>滑鼠停在格子上看 95% 區間</span>'
              '<span>顯著與否看最佳與次佳差距的 95% 區間，以玩家為單位重抽：同一位玩家的決策點一起抽，人數越少、區間越寬</span></p>')
    return (f'<div class="table-wrap"><table><thead><tr><th>葉子</th><th>驗證決策點</th><th>驗證玩家數</th><th>整體留存</th>'
            f'{head}<th>最佳版本</th>'
            f'<th>信心</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div>{legend}')


def five_section(placed: list[dict], unplaced: list[dict], candidates: list[dict], colors: dict[str, str],
                 unfilled_trees: list[dict] | None = None) -> str:
    """「五個體感」分頁在各棵樹之後：摘要、補位結果表、其他候選（收合）。placed／candidates 的欄位見 evaluation/five.py；
    unfilled_trees：每棵樹至少補幾片（min_per_tree）時沒補滿的樹 [{tree, reason[, placed_n]}]。"""
    unfilled_trees = unfilled_trees or []
    if not placed and not unplaced and not unfilled_trees:
        return '<p class="note"><b>五個體感都已使用，不需補位。</b></p>'

    def arm(a: str) -> str:
        return f'<span class="swatch" style="background:{colors.get(a, GRAY)}"></span>{esc(arm_short(a))}'

    def rule(c: dict) -> str:
        return " 且 ".join(c["conditions"]) or ("其餘全部" if c.get("after") else "整片葉子")

    def where(c: dict) -> str:                         # 同一片葉子補多個版本：後補的在扣掉前面子規則後的其餘部分
        return c["leaf"] + (f"（扣掉 {'、'.join(arm_short(a) for a in c['after'])} 的補位後）" if c.get("after") else "")

    def gap(c: dict) -> str:
        return f"{c['gap_pp']:.2f} [{c['gap_ci'][0]:.2f}, {c['gap_ci'][1]:.2f}]"

    def change(c: dict) -> str:                       # 整體留存變化 = −整體影響
        return f"{-c['impact_pp']:+.3f} [{-c['impact_ci'][1]:+.3f}, {-c['impact_ci'][0]:+.3f}]"

    def short(u: dict) -> str:                         # 沒補滿：只補到幾片，或一片都找不到（每棵樹都要補時標樹）
        return (f"只補到 {u['placed_n']} 片（其他樹：{u['reason']}）" if u.get("placed_n")
                else (f"在樹 {u['tree']} " if u.get("tree") else "") + f"找不到合適位置（{u['reason']}）")

    def tree_short(u: dict) -> str:
        return (f"樹 {u['tree']} 只補到 {u['placed_n']} 片（{u['reason']}）" if u.get("placed_n")
                else f"樹 {u['tree']} 沒有補位（{u['reason']}）")

    summary = "；".join([f"{arm_short(p['version'])} → {where(p)}" + ("（超過門檻）" if p.get("outside") else "")
                        for p in placed]
                       + [f"{arm_short(u['version'])} {short(u)}" for u in unplaced]
                       + [tree_short(u) for u in unfilled_trees])
    rows = []
    for p in placed:
        tip = (f"可比較比例 {p['comparable_share']:.0%}｜{arm_short(p['version'])} max_smd {p['max_smd']:.2f}"
               f"（{'平衡' if p['balanced'] else '未平衡'}）｜有效樣本 {p['ess']:,.0f}")
        over = (f'<br><span class="na">⚠️ 超過門檻：{esc(p["outside"])}（每棵樹、每個版本都要補到，門檻內沒有可用的位置）</span>'
                if p.get("outside") else "")
        rows.append(f'<tr title="{esc(tip)}"><td>{arm(p["version"])}</td>'
                    f'<td style="text-align:left">{esc(where(p))}：{esc(rule(p))}{over}</td>'
                    f'<td>{p["decision_points"]:,}（{p["share"]:.1%}）</td><td>{_pct(p["retention_v"], 2)}</td>'
                    f'<td>{arm(p["base_version"])} · {_pct(p["retention_base"], 2)}</td>'
                    f'<td>{esc(gap(p))}</td><td>{esc(change(p))}</td></tr>')
    for u in unplaced:
        rows.append(f'<tr><td>{arm(u["version"])}</td><td class="na" colspan="6">{esc(short(u))}</td></tr>')
    for u in unfilled_trees:
        rows.append(f'<tr><td>—</td><td class="na" colspan="6">{esc(tree_short(u))}</td></tr>')
    table = ('<div class="table-wrap"><table><thead><tr><th>版本</th><th>位置（葉子：子規則）</th><th>涵蓋驗證決策點（佔全部，w_cap 加權）</th>'
             '<th>補位版本加權留存</th><th>原建議版本 · 加權留存</th><th>差距 pp（原建議 − 補位）[95%]</th>'
             f'<th>整體留存變化 pp [95%]</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div>')
    legend = ('<p class="legend"><span>差距 &gt; 0 = 補位版本較差</span><span>整體留存變化 = 佔比 × 差距</span>'
              '<span>滑鼠停在列上看可比較比例</span></p>')
    crow = []
    for c in candidates:
        cells = (["—"] * 4 if c.get("gap_pp") is None else
                 [esc(rule(c)), f"{c['decision_points']:,}", esc(gap(c)), esc(change(c))])
        tip = "" if c.get("gap_pp") is None else f' title="可比較比例 {c["comparable_share"]:.0%}"'
        crow.append(f'<tr{tip}><td>{arm(c["version"])}</td><td>{esc(where(c))}</td><td>{arm(c["base_version"])}</td>'
                    f'<td style="text-align:left">{cells[0]}</td><td>{cells[1]}</td><td>{cells[2]}</td><td>{cells[3]}</td>'
                    f'<td>{esc(c.get("status", ""))}</td></tr>')
    details = (f'<details><summary>其他候選（{len(candidates)} 個）</summary><div class="table-wrap"><table><thead><tr>'
               '<th>版本</th><th>葉子</th><th>原建議</th><th>子規則</th><th>涵蓋驗證決策點</th><th>差距 pp [95%]</th>'
               f'<th>整體留存變化 pp [95%]</th><th>結果</th></tr></thead><tbody>{"".join(crow)}</tbody></table></div></details>')
    return f'<h3>補位結果</h3><p class="note">{esc(summary)}</p>{table}{legend}{details}'


TREE_SWITCH_JS = """
(function () {
  var bars = document.querySelectorAll('.tree-switch');
  if (!bars.length) return;
  function show(name) {
    document.querySelectorAll('.tree-sec').forEach(function (sec) {
      sec.style.display = sec.dataset.tree === name ? '' : 'none';
    });
    document.querySelectorAll('.tree-switch button').forEach(function (b) {
      b.classList.toggle('on', b.dataset.tree === name);
    });
    if (window.refitTrees) window.refitTrees();
  }
  document.querySelectorAll('.tree-switch button').forEach(function (b) {
    b.addEventListener('click', function () { show(b.dataset.tree); });
  });
  show(bars[0].querySelector('button').dataset.tree);
})();
"""


def _facts(items: list[str] | None) -> list[str]:
    """頁首的實驗資料：條列（項目內可含 HTML）。"""
    return ['<ul class="meta">', *[f"<li>{x}</li>" for x in items], "</ul>"] if items else []


FIVE_TABS = ("five", "five2")                  # 五個體感分頁的 views 鍵（依序）


def experiment_page(title: str, meta_lines: list[str], views: dict[str, list[dict]], notes: list[str],
                    facts: list[str] | None = None, five_results: str | dict[str, str] | None = None,
                    five_labels: dict[str, str] | None = None) -> str:
    """views：{"orig": [...], "merged": [...], "simple": [...]}（原始、合併無法比較、合併相近建議）；
    另有 "five"（與 "five2"）時多一（兩）個五個體感分頁，各棵樹之後接 five_results[鍵]（補位結果，SPEC 9.4）；
    five_labels[鍵] = 分頁名稱（預設「五個體感」「五個體感2」）；five_results 給字串時視為 "five" 的結果；
    meta_lines = 頁首單行（連結、警示），facts = 頁首條列的實驗資料；
    每個元素 = 一棵樹 {title, sub, svg, table, note}（table 為空字串時不顯示）；多棵樹時上方有按鈕切換（各分頁同步）。"""
    def tree_name(t: dict) -> str:
        return t["title"].removeprefix("樹：")

    def render(trees: list[dict]) -> str:
        parts = []
        if len(trees) > 1:
            parts.append('<div class="tree-switch"><span class="meta">切換決策樹：</span>'
                         + "".join(f'<button type="button" data-tree="{esc(tree_name(t))}">{esc(tree_name(t))}</button>'
                                   for t in trees)
                         + '</div>')
        for t in trees:
            parts.append(f'<section class="tree-sec" data-tree="{esc(tree_name(t))}">')
            parts.append(f'<h2>{esc(t["title"])}</h2>')
            if t.get("sub"):
                parts.append(f'<p class="meta">{esc(t["sub"])}</p>')
            if t.get("note"):
                parts.append(f'<p class="note">{esc(t["note"])}</p>')
            parts.append('<h3>決策樹</h3><div class="tree-fig"><div class="zoom">'
                         '<button type="button" data-act="fit">符合畫面</button><button type="button" data-act="out">−</button>'
                         '<button type="button" data-act="in">＋</button><button type="button" data-act="full">原始大小</button>'
                         f'<span class="zoom-pct"></span></div><div class="figure">{t["svg"]}</div></div>')
            parts.append('<p class="legend">最下行 = 訓練期間筆數與留存；百分比 = 驗證期間加權留存；灰色 = 無法比較；'
                         '答「是」走左、「否」走右。</p>')
            if t.get("table"):
                parts.append(f'<h3>每片葉子在各版本的留存率</h3>{t["table"]}')
            parts.append("</section>")
        return "\n".join(parts)

    tabs = [k for k in FIVE_TABS if k in views]
    labels = {"five": "五個體感", "five2": "五個體感2", **(five_labels or {})}
    results = five_results if isinstance(five_results, dict) else {"five": five_results or ""}
    body = [f"<h1>{esc(title)}</h1>", *[f'<p class="meta">{m}</p>' for m in meta_lines], *_facts(facts),
            '<div class="tabs">',
            '<input type="radio" name="view" id="v-orig" checked><label for="v-orig">原始葉子</label>',
            '<input type="radio" name="view" id="v-merged"><label for="v-merged">合併無法比較的葉子</label>',
            '<input type="radio" name="view" id="v-simple"><label for="v-simple">合併相近建議</label>',
            *[f'<input type="radio" name="view" id="v-{k}"><label for="v-{k}">{esc(labels[k])}</label>' for k in tabs],
            f'<div class="view orig">{render(views["orig"])}</div>',
            f'<div class="view merged">{render(views["merged"])}</div>',
            f'<div class="view simple">{render(views.get("simple", views["merged"]))}</div>',
            *[f'<div class="view {k}">{FIVE_CSS if i == 0 else ""}{render(views[k])}{results.get(k) or ""}</div>'
              for i, k in enumerate(tabs)],
            "</div>",
            "<h2>怎麼讀</h2>", "<ul>", *[f"<li>{n}</li>" for n in notes], "</ul>"]
    return page(title, "\n".join(body), ZOOM_JS + TREE_SWITCH_JS)


# ---------------------------------------------------------------------------
# 實驗比較：AUC
# ---------------------------------------------------------------------------
def auc_chart(rows: list[dict], baseline: float | None) -> str:
    """rows：{label, auc, lo, hi, verdict}；點 = AUC、橫線 = 95% 區間、虛線 = 基準。"""
    W, LABEL_W, VALUE_W, ROW_H, TOP, BOTTOM = 880, 230, 90, 44, 34, 40
    lo = min(r["lo"] for r in rows)
    hi = max(r["hi"] for r in rows)
    pad = (hi - lo) * 0.08 or 0.01
    lo, hi = lo - pad, hi + pad
    step = 0.005 if hi - lo < 0.06 else 0.01
    plot_w = W - LABEL_W - VALUE_W
    X = lambda v: LABEL_W + (v - lo) / (hi - lo) * plot_w           # noqa: E731
    H = TOP + ROW_H * len(rows) + BOTTOM
    color = {"較佳": "var(--good)", "較差": "var(--bad)", "基準": "var(--text)"}
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="100%" style="max-width:{W}px" role="img">']
    tick = math.ceil(lo / step) * step
    while tick <= hi + 1e-12:
        out.append(f'<line x1="{X(tick):.1f}" y1="{TOP - 8}" x2="{X(tick):.1f}" y2="{H - BOTTOM + 4}" stroke="var(--gray)"/>')
        out.append(f'<text x="{X(tick):.1f}" y="{H - BOTTOM + 24}" text-anchor="middle" font-size="14" class="small">{tick:.3f}</text>')
        tick += step
    if baseline is not None:
        out.append(f'<line x1="{X(baseline):.1f}" y1="{TOP - 14}" x2="{X(baseline):.1f}" y2="{H - BOTTOM + 4}" '
                   f'stroke="var(--muted)" stroke-dasharray="6 4" stroke-width="1.5"/>')
        out.append(f'<text x="{X(baseline):.1f}" y="{TOP - 18}" text-anchor="middle" font-size="13" class="small">基準</text>')
    for k, r in enumerate(rows):
        y = TOP + ROW_H * k + ROW_H / 2
        c = color.get(r["verdict"], "var(--muted)")
        out.append(f'<text x="{LABEL_W - 14}" y="{y + 6:.1f}" text-anchor="end" font-size="17" font-weight="600">{esc(r["label"])}</text>')
        out.append(f'<line x1="{X(r["lo"]):.1f}" y1="{y:.1f}" x2="{X(r["hi"]):.1f}" y2="{y:.1f}" stroke="{c}" stroke-width="3"/>')
        out.append(f'<circle cx="{X(r["auc"]):.1f}" cy="{y:.1f}" r="7" fill="{c}"/>')
        out.append(f'<text x="{W - VALUE_W + 12}" y="{y + 6:.1f}" font-size="17" font-weight="700">{r["auc"]:.4f}</text>')
    out.append("</svg>")
    return "\n".join(out)


CONF_LEVELS = ("高", "中", "低", "無法比較")                 # 比較頁的信心分佈（維持現狀的葉子不計入）


def _score(v) -> str:
    return "—" if v is None or v != v else f"{v:.0f}"


def _conf_cells(c: dict | None) -> str:
    if c is None:
        return f'<td class="na" colspan="{1 + len(CONF_LEVELS)}">尚未 evaluate</td>'
    return (f'<td><b>{_score(c["score"])}</b></td>'
            + "".join(f'<td>{c["share"][lv]:.0%}</td>' for lv in CONF_LEVELS))


def summary_page(title: str, meta_lines: list[str], rows: list[dict], baseline: float | None,
                 auc_notes: list[str], conf_notes: list[str], facts: list[str] | None = None) -> str:
    """頁首另有 facts 條列。rows：{label, href, auc, lo, hi, diff, diff_lo, diff_hi, verdict, conf}，依實驗清單順序；
    conf = {orig／merged: {share, score}}（evaluate 之前為 None）；信心只顯示 merged（合併無法比較的葉子後）。
    內容：總覽表、① AUC（圖與表）、② 信心（表與評分依據）。"""
    def fmt_diff(r):
        if r["diff"] is None:
            return "—", "—"
        return f"{r['diff']:+.4f}", f"[{r['diff_lo']:+.4f}, {r['diff_hi']:+.4f}]"

    def name(r):
        return f'<a href="{esc(r["href"])}">{esc(r["label"])}</a>' if r.get("href") else esc(r["label"])

    def score(r, view):
        c = (r.get("conf") or {}).get(view)
        return "—" if c is None else _score(c["score"])

    verdict_cls = {"較佳": "verdict-better", "較差": "verdict-worse"}
    overview = "".join(
        f'<tr><td class="leaf">{name(r)}</td><td><b>{r["auc"]:.4f}</b></td>'
        f'<td class="{verdict_cls.get(r["verdict"], "")}">{esc(r["verdict"])}</td>'
        f'<td><b>{score(r, "merged")}</b></td></tr>' for r in rows)
    auc_rows = []
    for r in rows:
        d, ci = fmt_diff(r)
        auc_rows.append(f'<tr><td class="leaf">{name(r)}</td><td><b>{r["auc"]:.4f}</b></td><td>[{r["lo"]:.4f}, {r["hi"]:.4f}]</td>'
                        f'<td>{d}</td><td>{ci}</td><td class="{verdict_cls.get(r["verdict"], "")}">{esc(r["verdict"])}</td></tr>')
    conf_rows = "".join(f'<tr><td class="leaf">{name(r)}</td>{_conf_cells((r.get("conf") or {}).get("merged"))}</tr>'
                        for r in rows)
    level_head = "".join(f"<th>{lv}</th>" for lv in CONF_LEVELS)
    body = [f"<h1>{esc(title)}</h1>", *[f'<p class="meta">{m}</p>' for m in meta_lines], *_facts(facts),
            "<h2>總覽</h2>",
            '<div class="table-wrap"><table><thead><tr><th>實驗（點名稱看每片葉子）</th><th>AUC</th><th>與基準 all／無</th>'
            '<th>信心分數</th></tr></thead>',
            f"<tbody>{overview}</tbody></table></div>",
            '<p class="legend"><span>AUC：分不分得出誰會離場（churnNext20），越高越好</span>'
            '<span>信心分數 0–100：能拿到可信版本建議的決策點比例，越高越好</span></p>',
            "<h2>① AUC：分不分得出誰會離開</h2>",
            f'<div class="figure">{auc_chart(rows, baseline)}</div>',
            '<p class="legend"><span>點 = AUC</span><span>橫線 = 95% 區間</span><span>虛線 = 基準 all／無</span>'
            '<span style="color:var(--good)">綠 = 顯著優於基準</span><span style="color:var(--bad)">紅 = 顯著較差</span>'
            '<span>灰 = 與基準無顯著差異</span></p>',
            '<div class="table-wrap"><table><thead><tr><th>實驗</th><th>AUC</th><th>95% 區間</th>'
            '<th>與基準差</th><th>95% 區間</th><th>判定</th></tr></thead>',
            f'<tbody>{"".join(auc_rows)}</tbody></table></div>',
            "<ul>", *[f"<li>{n}</li>" for n in auc_notes], "</ul>",
            "<h2>② 信心：能不能可信地為每片葉子挑版本</h2>",
            f'<div class="table-wrap"><table><thead><tr><th>實驗</th><th>分數</th>{level_head}</tr></thead>',
            f"<tbody>{conf_rows}</tbody></table></div>",
            "<h3>評分依據</h3>", "<ul>", *[f"<li>{n}</li>" for n in conf_notes], "</ul>"]
    return page(title, "\n".join(body))

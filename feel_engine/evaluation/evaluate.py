# -*- coding: utf-8 -*-
"""
第 9 節：葉子 × arm 評估（只用驗證期間）→ 每片葉子留存率最高的 arm（體感模組版本）。

  9.1 傾向分數（所有實驗共用、只配適一次）：
      p(arm | x)，x = 快層 ＋ 所有 TAG_ 標籤（含分群用的標籤）；依玩家 5 折交叉配適；sample_weight = w_cap。
      模型（config leaf_eval.propensity_model）：logistic = 多項式邏輯迴歸（預設，用 design_matrix 的共變數）；
      gbm = 梯度提升樹，從資料自動學版本分派的門檻與交互作用（數值保留原值與缺值、文字 one-hot）。
      2026-10-12 實測 gbm 的權重較極端、有效樣本較少，配合 9.1b 校準後信心不如 logistic，所以預設 logistic。
      結果模型（同樣只配適一次、依玩家交叉配適）：每個 arm 一個邏輯迴歸，用狀態 x 預測「拿到該 arm 時」的留存率 m_a(x)。
  9.1a 共同範圍（2026-10-15 起，取代 propensity_min）：每片葉子挑要比較的 arm 組合 C，共同範圍 S = 葉子內
      「C 的每個 arm 機率都 ≥ overlap_min」的決策點，C 的每個 arm 在 S 內都要 ≥ min_n_per_arm 筆。
      盡量多比幾個 arm，但 S 要涵蓋葉子 ≥ min_overlap_share；沒有組合做得到 → 選涵蓋最多的（信心最多「低」）。
      現行規則幾乎只給某版本的狀態（機率極低的那一邊）不拿來比，避免把「誰拿到」當成「版本效果」。
  9.1b 葉子內校準（leaf_eval.balance_calibration）：S 內每個比較 arm，以 W = w_cap / p(實際 arm) 為起點調整權重，
      讓該 arm 各共變數的加權平均與 S 相差 ≤ balance_tolerance 個標準差（近似 entropy balancing）；比 W 更平衡才採用。
      對不齊時平衡檢查照實判未平衡。只用資料，不用遊戲端規則。
  9.2 每片葉子 × arm：加權留存率 = 雙重穩健（DR）估計
        = S 內 m_a 的 w_cap 加權平均 ＋ S 內實際拿到 a 者「y − m_a」的校準權重平均（y = 1 − churnNext20）；
      max_smd（對 S）、有效樣本。最高留存 arm = 比較 arm 中加權留存率最高者；與次高、與臨時基準（n 最大的 arm）比較，
      差距用玩家層級 Poisson(1) bootstrap（先聚合成「玩家 × 葉子 × arm」再以稀疏矩陣運算）。
  合併：比較 arm 不到兩個（無法比較）的葉子，由下往上把父節點整個合併成一片葉子（合併後重挑共同範圍），重算 9.2，
        直到每片葉子都可比較或合併到根（merge_incomparable）。
  程式：9.1 在 propensity.py；9.1a–9.2 在 leaf_eval.py；合併在 merge.py；本檔為主流程。

暫代模式（version_map 未填）：arm = game_version；不換算 M1–M7（第 10 節）。
輸出：每個實驗一頁 result/{run}/{split}_tag_{tags}.html（原始葉子、合併無法比較、合併相近建議三個分頁；
      config five_feelings.enabled 時再加五個體感分頁，9.4、evaluation/five.py；pages.experiment_page）：
      正式實驗兩個分頁「五個體感1」（five_feelings）與「五個體感2」（five_feelings 再覆蓋 five_feelings_2，每棵樹都有每個版本）；
      變體頁一個「五個體感」分頁（變體自己的設定）。
      output/.../leaf_confidence.json（每片葉子的信心）、five_feelings.json／five_feelings_2.json（補位結果），
      最後重做 summary.html（AUC ＋ 信心分數）。
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from feel_engine.evaluation.five import FiveResult, augment_tree, fill_versions
from feel_engine.evaluation.leaf_eval import KEEP_CURRENT, comparable_fn, evaluate_leaves
from feel_engine.evaluation.merge import (collapse_names, merge_incomparable, merge_recommendations,
                                          recommended_version)
from feel_engine.evaluation.propensity import KEY, Propensity, build_propensity
from feel_engine.html import pages
from feel_engine.paths import RunPaths, rel

log = logging.getLogger("feel_engine.evaluation.evaluate")
SIMPLIFY_ROUNDS = 10                              # 合併相近建議：合併 → 重算 → 再合併，最多幾輪


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def _tag_label(tags: str) -> str:
    return "無" if tags == "none" else tags.replace("_", "＋")


def _tree_view(name: str, tree: dict, rules: dict, collapsed: set[int], names: dict[int, str], best: pd.DataFrame,
               stats: pd.DataFrame, cis: dict, arms: list[str], colors: dict[str, str], le: dict, merged: bool,
               base_collapsed: set[int] | None = None, five: FiveResult | None = None) -> dict:
    by_leaf = {r["葉子"]: r for r in best.to_dict(orient="records")}
    est = stats.set_index(["葉子", "arm"])["加權留存率"].to_dict()
    info = {}
    for node, leaf in names.items():
        r = by_leaf.get(leaf)
        conf = pages.display_conf(r) if r else pages.NOT_COMPARABLE
        arm = None if r is None else {pages.NOT_COMPARABLE: None, KEEP_CURRENT: r.get("最多版本")}.get(conf, r["最高留存 arm"])
        retention = r["葉子留存率"] if conf == KEEP_CURRENT else est.get((leaf, arm))    # 維持現狀 = 現行分派下的整體留存
        info[node] = {"name": pages.short_leaf(leaf), "best": pages.arm_short(arm) if arm else None,
                      "color": colors.get(arm), "conf": conf, "retention": retention}
    shown = list(names.values())
    best, stats = best[best["葉子"].isin(shown)], stats[stats["葉子"].isin(shown)]
    root = tree["nodes"][0]
    users = [g["群"] for g in rules["groups"] if g["樹"] == name and g["群"] != name]
    sub = f"訓練期間 {root['n']:,} 筆 → 驗證期間 {int(best['葉子決策點'].sum()):,} 筆"
    if users:
        sub += f"；{'、'.join(users)} 樣本不足，沿用這棵樹"
    note = None
    if base_collapsed is not None:                # 合併相近建議：只列這一步新併的
        new = [names[i] for i in sorted(collapsed) if i in names and i not in base_collapsed]
        note = ("沒有建議相同或差異不顯著的相鄰葉子，與「合併無法比較的葉子」相同。" if not new else
                f"再合併了 {len(new)} 處建議相同或差異不顯著的葉子：{'、'.join(pages.short_leaf(t) for t in new)}。")
    elif merged:
        tops = [names[i] for i in sorted(collapsed) if i in names]
        note = ("所有葉子都至少有兩個版本樣本足夠，與原始葉子相同。" if not tops else
                f"合併了 {len(tops)} 處：{'、'.join(pages.short_leaf(t) for t in tops)}。"
                + ("整棵樹合併後仍無法比較。" if 0 in collapsed and info.get(0, {}).get("conf") == pages.NOT_COMPARABLE else ""))
    title = "決策樹" if name == "all" and rules["split"] == "all" else f"樹：{name}"
    table = pages.leaf_arm_table(best, stats, cis, arms, colors, int(le["min_n_per_arm"]), float(le["smd_threshold"]))
    if five is not None:                          # 五個體感：在「合併相近建議」的樹上接補位子規則（SPEC 9.4）
        here = [p for p in five.placed if p["leaf"] in set(names.values())]
        nodes, drawn, info = augment_tree(tree["nodes"], collapsed, names, info, here, colors)
        why = {u["tree"]: u["reason"] for u in five.unfilled_trees}.get(name)      # min_per_tree 沒補滿的原因
        done = "、".join(f"{pages.arm_short(p['version'])}（{pages.short_leaf(p['leaf'])}）" for p in here)
        note = ("這棵樹補位：" + done + (f"（未達每棵樹 {five.min_per_tree} 片：{why}）" if why else "")
                if here else "這棵樹沒有補位" + (f"（{why}）" if why else "") + "。")
        if five.per_tree:                         # every_version_per_tree：原本建議的版本 ＋ 補位 = 全部版本
            own = "、".join(pages.arm_short(a) for a in arms if a not in five.need.get(name, []))
            lack = "、".join(f"{pages.arm_short(u['version'])}（{u['reason']}）" for u in five.unplaced
                             if u.get("tree") == name)
            note = (f"這棵樹的葉子建議 {own}；" + (f"補位：{done}" if here else "不需補位")
                    + (f"；補不到：{lack}" if lack else "") + "。")
        return {"title": title, "sub": sub, "note": note, "svg": pages.tree_svg(nodes, info, drawn), "table": table}
    return {"title": title, "sub": sub, "note": note, "svg": pages.tree_svg(tree["nodes"], info, collapsed), "table": table}


def _five_note(label: str, five: FiveResult, ff: dict, min_share: float, max_loss: float) -> str:
    """「怎麼讀」裡五個體感分頁的說明（依補位設定）。"""
    return (f"<b>{label}</b>：{'每棵樹的葉子沒建議到的版本，在同一棵樹的' if five.per_tree else '沒被建議的版本，在'}"
            f"葉子內找子規則補位：可比區至少佔 {min_share:.0%}（min_comparable_share）、"
            f"比原建議低不超過 {max_loss:g} pp（max_loss_pp）；"
            + ("每棵樹內每個版本各一條（every_version_per_tree：每棵樹都要有每個版本）" if five.per_tree
               else "每個版本在整個實驗至少一條")
            + "，挑整體留存損失最小的位置"
            + (f"；這個實驗的決策樹超過 {int(ff.get('many_trees', 3))} 棵，每個版本補 {five.per_version} 片、分在不同的樹"
               if five.per_version > 1 else "")
            + (f"；每棵樹至少補 {five.min_per_tree} 片（min_per_tree，優先於每個版本的片數）" if five.min_per_tree else "")
            + (("；門檻內沒有可用的位置時，可比區不足的候選也可以用（force_coverage），標「超過門檻」；"
                f"比原建議低超過 {max_loss:g} pp 的位置一律不用" if five.per_tree else
                "；每棵樹、每個版本都一定要補到（force_coverage）：門檻內沒有可用的位置時，改用門檻外的候選"
                "（先求門檻外的片數最少，再求整體留存損失最小），標「超過門檻」")
               if five.force else "")
            + "。")


def prepare_evaluation(valid: pd.DataFrame, cfg: dict, mcfg: dict) -> tuple[pd.DataFrame, Propensity, pd.DataFrame,
                                                                             dict[str, str]]:
    """所有實驗共用：（驗證期間資料（重設索引）, 傾向分數與結果模型（9.1，只配適一次）, ps.table 加列號 _z, 版本顏色）。"""
    tag_cols = list(dict.fromkeys(c for cols in mcfg["tags"].values() for c in cols))
    V = valid.reset_index(drop=True)
    num_cols = [c for c in mcfg["fast"] if pd.api.types.is_numeric_dtype(V[c])]
    cat_cols = [c for c in [*mcfg["fast"], *tag_cols] if c not in num_cols]
    ps = build_propensity(V, num_cols, cat_cols, cfg)
    log.info("傾向分數與結果模型：驗證期間 %s 筆、%d 個版本", f"{len(ps.table):,}", len(ps.arms))
    return V, ps, ps.table.assign(_z=np.arange(len(ps.table))), pages.arm_colors(ps.arms)


def evaluate_experiments(rp: RunPaths, cfg: dict, mcfg: dict, valid: pd.DataFrame, provisional: bool) -> list[Path]:
    """對 model.yaml experiments 中已跑過決策樹的實驗，做第 9 節的葉子 × arm 評估，每個實驗寫一頁 HTML。"""
    if provisional:
        log.info("版本 = game_version（暫代，version_map 未填），尚未對應到體感模組 M1–M7")
    experiments = [(e["split"], e["tags"]) for e in mcfg["experiments"]]
    present = [k for k in experiments if rp.model_predictions(*k).exists() and rp.tree_rules(*k).exists()]
    if not present:
        log.warning("沒有任何實驗的 valid_predictions.parquet，請先執行 run")
        return []
    V, ps, base, colors = prepare_evaluation(valid, cfg, mcfg)
    paths = []
    for split, tags in present:
        title = f"實驗 {experiments.index((split, tags)) + 1}：{split}／{_tag_label(tags)}"
        paths.append(evaluate_experiment(rp, cfg, split, tags, title, V, ps, base, colors, two_five_tabs=True))
    from feel_engine.html.compare import write_summary            # 信心更新後重做 summary.html（AUC ＋ 信心分數）

    write_summary(rp, cfg, mcfg)
    return paths


def evaluate_experiment(rp: RunPaths, cfg: dict, split: str, tags: str, title: str, V: pd.DataFrame, ps: Propensity,
                        base: pd.DataFrame, colors: dict[str, str], extra_facts: tuple[str, ...] = (),
                        two_five_tabs: bool = False) -> Path:
    """一個實驗（rp.model_predictions、rp.tree_rules 已存在）：9.1a–9.2、合併、五個體感；寫 leaf_confidence.json、
    five_feelings.json 與實驗頁 HTML，回傳 HTML 路徑。V、ps、base、colors 來自 prepare_evaluation；
    extra_facts：頁首多加的說明（HTML，排在最後）；two_five_tabs（正式實驗）且 config 有 five_feelings_2 →
    兩個分頁「五個體感1」「五個體感2」（後者另寫 five_feelings_2.json），否則一個「五個體感」分頁。"""
    le, seed = cfg["leaf_eval"], int(cfg["seed"])
    min_n = int(le["min_n_per_arm"])
    share_min = float(le["min_overlap_share"])
    ff = cfg.get("five_feelings") or {}
    five_on, max_loss = bool(ff.get("enabled", False)), float(ff.get("max_loss_pp", 2.0))
    min_share = float(ff.get("min_comparable_share", 0.5))

    pred = pd.read_parquet(rp.model_predictions(split, tags), columns=[*KEY, "群", "樹", "葉子"])
    df = pred.merge(base, on=KEY, how="inner")
    rules = json.loads(rp.tree_rules(split, tags).read_text(encoding="utf-8"))
    leaf_order = [l["葉子"] for t in rules["trees"].values() for l in t["leaves"]]
    stats_o, best_o, cis_o = evaluate_leaves(df, ps, le, seed, leaf_order)
    rename = {l: l for l in leaf_order}

    comparable = comparable_fn(df, ps.P, le)
    merge = {name: merge_incomparable(t["nodes"], comparable) for name, t in rules["trees"].items()}
    if any(m[0] for m in merge.values()):
        rename = {leaf: m[2][node] for m in merge.values() for leaf, node in m[1].items()}
        stats_m, best_m, cis_m = evaluate_leaves(df.assign(葉子=df["葉子"].map(rename)), ps, le, seed,
                                                  list(dict.fromkeys(rename[l] for l in leaf_order)))
    else:
        stats_m, best_m, cis_m = stats_o, best_o, cis_o

    # 合併相近建議：建議相同的相鄰葉子併成一片；樹枝過於龐大（仍超過 similar_merge_over_leaves 片）的樹，
    # 再併建議不同但都不是高信心的相鄰葉子 → 重算 → 再合併，直到不再變化
    over_leaves = int(le.get("similar_merge_over_leaves", 8))
    simple = {name: (merge[name][0], merge[name][2]) for name in rules["trees"]}
    rename_s, stats_s, best_s, cis_s = rename, stats_m, best_m, cis_m
    for _ in range(SIMPLIFY_ROUNDS):
        by_s = {r["葉子"]: r for r in best_s.to_dict(orient="records")}
        nxt, rename_n = {}, {}
        for name, t in rules["trees"].items():
            rows = {node: by_s.get(leaf) for node, leaf in simple[name][1].items()}
            rec = {node: recommended_version(r) for node, r in rows.items()}
            strong = {node: r is not None and pages.display_conf(r) == "高" for node, r in rows.items()}
            collapsed_s = merge_recommendations(t["nodes"], simple[name][0], rec, strong, over_leaves)
            owner_s, names_s = collapse_names(t["nodes"], collapsed_s)
            nxt[name] = (collapsed_s, names_s)
            rename_n.update({leaf: names_s[node] for leaf, node in owner_s.items()})
        if rename_n == rename_s:
            break
        simple, rename_s = nxt, rename_n
        stats_s, best_s, cis_s = evaluate_leaves(df.assign(葉子=df["葉子"].map(rename_s)), ps, le, seed,
                                                  list(dict.fromkeys(rename_s[l] for l in leaf_order)))

    modes = []                                # （分頁鍵, 分頁名稱, 補位設定, 補位結果檔後綴）
    if five_on:
        ff2 = cfg.get("five_feelings_2")
        modes = ([("five", "五個體感1", ff, ""), ("five2", "五個體感2", {**ff, **ff2}, "2")] if two_five_tabs and ff2
                 else [("five", "五個體感", ff, "")])
    fives: dict[str, tuple[str, dict, FiveResult]] = {}
    for key, label, f, suffix in modes:       # SPEC 9.4：以「合併相近建議」為準補位
        res = fill_versions(df.assign(葉子=df["葉子"].map(rename_s)), V, ps.rows, best_s, rules, ps, le, f, seed)
        rp.five_feelings(split, tags, suffix).write_text(json.dumps(res.to_json(max_loss), ensure_ascii=False, indent=1),
                                                         encoding="utf-8")
        fives[key] = (label, f, res)
    views = {"orig": [], "merged": [], "simple": [], **{key: [] for key in fives}}
    for name, t in rules["trees"].items():
        leaf_names = {n["id"]: n["leaf"] for n in t["nodes"] if n["left"] == -1}
        views["orig"].append(_tree_view(name, t, rules, set(), leaf_names, best_o, stats_o, cis_o, ps.arms, colors, le,
                                        merged=False))
        views["merged"].append(_tree_view(name, t, rules, merge[name][0], merge[name][2], best_m, stats_m, cis_m,
                                          ps.arms, colors, le, merged=True))
        views["simple"].append(_tree_view(name, t, rules, simple[name][0], simple[name][1], best_s, stats_s, cis_s,
                                          ps.arms, colors, le, merged=True, base_collapsed=merge[name][0]))
        for key, (_, _, res) in fives.items():
            views[key].append(_tree_view(name, t, rules, simple[name][0], simple[name][1], best_s, stats_s, cis_s,
                                         ps.arms, colors, le, merged=True, five=res))
    conf = lambda best: {r["葉子"]: pages.display_conf(r) for r in best.to_dict(orient="records")}   # noqa: E731
    rp.leaf_confidence(split, tags).write_text(json.dumps({
        "propensity_model": ps.model, "balance_calibration": bool(le.get("balance_calibration", True)),
        "orig": conf(best_o), "merged": conf(best_m), "merge_map": rename,
        "simple": conf(best_s), "simple_map": rename_s}, ensure_ascii=False, indent=1), encoding="utf-8")
    skipped = [g["群"] for g in rules["groups"] if g["策略"] == "不建模"]
    n_conf = best_m.assign(c=[pages.display_conf(r) for r in best_m.to_dict(orient="records")])["c"].value_counts()
    cmp_m = best_m[best_m["樣本足夠 arm 數"] >= 2]                # 頁首只報合併無法比較後的葉子
    covered = float((cmp_m["可比較比例"] * cmp_m["葉子決策點"]).sum() / max(int(best_m["葉子決策點"].sum()), 1))
    meta = [f'<a href="{pages.esc(rp.summary_html.name)}">← 實驗比較（AUC）</a>']
    conf_order = ["高", "中", "低", pages.NOT_COMPARABLE, KEEP_CURRENT]
    conf_text = "、".join(f"{k} {int(n_conf[k])}" for k in [*conf_order, *[k for k in n_conf.index if k not in conf_order]]
                         if k in n_conf.index)
    facts = [
        f"商戶／幣別／遊戲：{pages.esc(rules['merchant'])}／{pages.esc(rules['currency'])}／{pages.esc(rules['game'])}",
        f"訓練期間：美東 {pages.esc(rules['train_start'])}～{pages.esc(rules['train_end'])}（用來長決策樹）",
        f"驗證期間：{pages.esc(pages.period(rules['valid_start'], rules['valid_end']))}（用來算留存率與信心），"
        f"共 {len(df):,} 個決策點",
        f"合併後葉子的信心：{conf_text}",
        f"在共同範圍內比較的決策點：{covered:.0%}",
        *([f"不建模的群：{pages.esc('、'.join(skipped))}"] if skipped else []),
        *extra_facts,
    ]
    dominant = float(le.get("dominant_share", 0.8))
    notes = [
        "<b>留存率</b>：20 手內沒離場，或離場後 24 小時內回來（1 − churnNext20）。",
        "<b>加權留存率</b>：依玩家狀態校正後的留存率，可以在版本間直接比較（雙重穩健估計，w_cap 加權）。",
        f"<b>共同範圍</b>：葉子內每個比較版本都有機會被發到（機率 ≥ {float(le['overlap_min']):.0%}，overlap_min）的決策點，只在這裡比較。",
        f"<b>樣本足夠</b>：共同範圍內至少 {min_n} 筆（min_n_per_arm）。",
        f"<b>信心</b>：高 = 最佳版本平衡，且顯著勝過次佳；中 = 平衡但不顯著；低 = 未平衡，或共同範圍不到葉子 {share_min:.0%}；"
        f"無法比較 = 樣本足夠的版本不到兩個；維持現狀_X = 同一版本佔葉子超過 {dominant:.0%}。",
        f"<b>平衡／顯著</b>：平衡 = 加權後各特徵相差不到 {le['smd_threshold']} 個標準差（smd_threshold）；"
        "顯著 = 最佳減次佳的 95% 區間不含 0。",
        "<b>合併</b>：無法比較的葉子併回上一層；再把建議相同的相鄰葉子併成一片。樹枝過於龐大（仍超過 "
        f"{int(le.get('similar_merge_over_leaves', 8))} 片，similar_merge_over_leaves）的樹，建議不同但兩片都不是高信心的也併；"
        "併完重算、直到不再變化。只影響建議，不影響決策樹與 AUC。",
        *[_five_note(label, res, f, min_share, max_loss) for label, f, res in fives.values()],
        "<b>注意</b>：信心低或可比區小時，差距可能來自玩家狀態，不一定是版本本身的效果。",
    ]
    path = rp.experiment_html(split, tags)
    path.parent.mkdir(parents=True, exist_ok=True)
    results = {key: pages.five_section(res.placed, res.unplaced, res.candidates, colors, res.unfilled_trees)
               for key, (_, _, res) in fives.items()}
    labels = {key: label for key, (label, _, _) in fives.items()}
    path.write_text(pages.experiment_page(title, meta, views, notes, facts=facts, five_results=results,
                                          five_labels=labels), encoding="utf-8")
    for label, _, five in fives.values():
        log.info("%s %s 補位：%s", title, label, "；".join(
            [f"{pages.arm_short(p['version'])} → {p['leaf']}（差距 {p['gap_pp']:.2f} pp）" for p in five.placed]
            + [f"{pages.arm_short(u['version'])} " + (f"只補到 {u['placed_n']} 片（其他樹：{u['reason']}）" if u.get("placed_n")
                                                      else (f"在樹 {u['tree']} " if u.get("tree") else "")
                                                      + f"找不到（{u['reason']}）") for u in five.unplaced]
            + [f"樹 {u['tree']} 沒補滿（{u['reason']}）" for u in five.unfilled_trees]) or "不需補位")
    log.info("%s → %s（原始 %d 片葉子、合併無法比較後 %d 片、合併相近建議後 %d 片；合併後信心高 %d）", title, rel(path),
             len(best_o), len(best_m), len(best_s), int(n_conf.get("高", 0)))
    return path

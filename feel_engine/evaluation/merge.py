# -*- coding: utf-8 -*-
"""
第 9 節：合併葉子（只影響版本建議，決策樹與 AUC 不變）。

  合併：比較 arm 不到兩個（無法比較）的葉子，由下往上把父節點整個合併成一片葉子（合併後重挑共同範圍），重算 9.2，
        直到每片葉子都可比較或合併到根（merge_incomparable）。
  合併相近建議：在合併無法比較之後，同一個上層節點底下建議版本相同的兩片葉子併成一片；併完仍超過 similar_merge_over_leaves 片的樹，
  再併建議不同但兩片都不是高信心的葉子（merge_recommendations）。
"""
from __future__ import annotations

from collections.abc import Callable

from feel_engine.evaluation.leaf_eval import KEEP_CURRENT
from feel_engine.html import pages


# ---------------------------------------------------------------------------
# 合併無法比較的葉子
# ---------------------------------------------------------------------------
def merge_incomparable(nodes: list[dict], comparable: Callable[[list[str]], bool]
                       ) -> tuple[set[int], dict[str, int], dict[int, str]]:
    """由下往上：子節點中有「無法比較」的葉子 → 父節點整個合併成一片葉子；重複到根。

    nodes：tree_rules.json 的節點；comparable(原葉子名稱清單) → 這些葉子合在一起時可不可以比較
    （comparable_fn：重挑共同範圍後比較 arm ≥ 2；或最多的 arm 佔決策點超過 dominant_share 的「維持現狀」，不合併）。
    回傳（合併成葉子的內部節點, 原葉子名稱 → 合併後所屬節點, 合併後的葉子節點 → 名稱）；
    合併後的名稱 = 涵蓋的原葉子編號範圍（例如 高/L3–L6；葉子由左到右編號，子樹內必連續）。
    """
    by_id = {n["id"]: n for n in nodes}
    under: dict[int, list[str]] = {}

    def walk(i: int) -> list[str]:
        n = by_id[i]
        under[i] = [n["leaf"]] if n["left"] == -1 else walk(n["left"]) + walk(n["right"])
        return under[i]

    walk(0)
    collapsed: set[int] = set()

    def visit(i: int) -> bool:                    # 回傳：合併後是否為葉子
        n = by_id[i]
        if n["left"] == -1:
            return True
        left, right = visit(n["left"]), visit(n["right"])
        if (left and not comparable(under[n["left"]])) or (right and not comparable(under[n["right"]])):
            collapsed.add(i)
            return True
        return False

    visit(0)
    return (collapsed, *collapse_names(nodes, collapsed))


def collapse_names(nodes: list[dict], collapsed: set[int]) -> tuple[dict[str, int], dict[int, str]]:
    """collapsed（當成葉子的內部節點）→（原葉子名稱 → 所屬的最上層葉子節點, 葉子節點 → 名稱）。
    合併後的名稱 = 涵蓋的原葉子編號範圍（例如 高/L3–L6；葉子由左到右編號，子樹內必連續）。"""
    by_id = {n["id"]: n for n in nodes}
    owner: dict[str, int] = {}
    names: dict[int, str] = {}

    def assign(i: int, top: int | None) -> list[str]:
        n = by_id[i]
        top = i if top is None and i in collapsed else top
        if n["left"] == -1:
            owner[n["leaf"]] = i if top is None else top
            if top is None:
                names[i] = n["leaf"]
            return [n["leaf"]]
        covered = assign(n["left"], top) + assign(n["right"], top)
        if i == top:
            first, last = covered[0], covered[-1]
            names[i] = first if first == last else f"{first}–{last.rsplit('/', 1)[-1]}"
        return covered

    assign(0, None)
    return owner, names


def merge_same_recommendation(nodes: list[dict], collapsed: set[int], rec: dict[int, str | None],
                              strong: dict[int, bool] | None = None) -> set[int]:
    """精簡規則（合併相近建議）：在 collapsed 之上由下往上，兩個子節點都是葉子，且
    建議的版本相同 → 父節點合併成一片葉子，沿用該建議、可一路往上；
    建議不同、但兩片都不是高信心（strong = False，最佳版本沒有可信地勝出）→ 也合併，合併後的建議要重算，這一輪不再往上。

    rec：目前的葉子節點 → 建議版本（None = 無法比較，不合併）；strong：葉子節點 → 是否高信心（None = 只併相同建議）。
    回傳新的 collapsed。決策樹與 AUC 不受影響。
    """
    by_id = {n["id"]: n for n in nodes}
    out, rec, strong = set(collapsed), dict(rec), dict(strong or {})

    def visit(i: int) -> bool:                    # 回傳：是否為葉子
        n = by_id[i]
        if n["left"] == -1 or i in out:
            return True
        left, right = visit(n["left"]), visit(n["right"])
        a, b = rec.get(n["left"]), rec.get(n["right"])
        if not (left and right) or a is None or b is None:
            return False
        if a == b:
            out.add(i)
            rec[i], strong[i] = a, strong.get(n["left"], True) or strong.get(n["right"], True)
            return True
        if strong and not strong.get(n["left"], True) and not strong.get(n["right"], True):
            out.add(i)
            rec[i] = None                         # 建議不同：合併後要重算，這一輪不再往上
            return True
        return False

    visit(0)
    return out


def merge_recommendations(nodes: list[dict], collapsed: set[int], rec: dict[int, str | None], strong: dict[int, bool],
                          over_leaves: int) -> set[int]:
    """合併相近建議的一輪：先只併相同建議；併完這棵樹仍超過 over_leaves 片（樹枝過於龐大）才再併
    建議不同、但兩片都不是高信心的相鄰葉子。其他樹維持只併相同建議，保留多元的版本建議。"""
    same = merge_same_recommendation(nodes, collapsed, rec)
    if len(collapse_names(nodes, same)[1]) <= over_leaves:
        return same
    return merge_same_recommendation(nodes, collapsed, rec, strong)


def recommended_version(row: dict | None) -> str | None:
    """葉子建議的版本：高／中／低 → 最高留存版本；維持現狀_X → X（可和建議 X 的葉子合併，2026-10-20）；無法比較 → None。"""
    if row is None:
        return None
    conf = pages.display_conf(row)
    if conf == pages.NOT_COMPARABLE:
        return None
    return row.get("最多版本") if conf == KEEP_CURRENT else row["最高留存 arm"]

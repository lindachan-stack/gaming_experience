#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Claude Code PreToolUse hook：Frozen／核心模組保護

用途：
  攔截 Edit／Write／MultiEdit／NotebookEdit，若目標檔案被標記為 frozen，
  直接擋下修改，落實 CLAUDE.md「不得直接修改明確標示為 frozen 的模組」
  與「先設計、後實作＋TDD」的規定。

判定 frozen 的兩種方式（符合其一即視為 frozen）：
  1. 檔案的專案相對路徑列在 .claude/hooks/frozen_paths.txt（支援 glob，
     例如 production/etl/steps/w6_*.py）。
  2. 檔案現存內容的前 20 行內，含有形如「# STATUS: FROZEN」或「# FROZEN」
     的標記（大小寫不拘，中文冒號亦可）。
輸入：
  stdin 傳入的 PreToolUse JSON。
輸出：
  命中 frozen 時印出 permissionDecision=deny 的 JSON；否則安靜地 exit 0。

解除方式：
  這是刻意的流程關卡，不提供程式內建的略過標記。要修改 frozen 模組，
  請使用者自行從 frozen_paths.txt 移除該路徑、或拿掉檔案內的 FROZEN
  標記後，再讓 Claude 進行修改（呼應 CLAUDE.md「先設計、後實作」的要求：
  這一步應該是先跟使用者確認完設計，才解除保護）。
"""
from __future__ import annotations

import fnmatch
import json
import os
import re
import sys
from itertools import islice
from pathlib import Path

FROZEN_MARKER_RE = re.compile(r"^\s*#.*\bSTATUS\s*[:：]\s*FROZEN\b", re.IGNORECASE | re.MULTILINE)
FROZEN_TAG_RE = re.compile(r"^\s*#\s*FROZEN\b", re.IGNORECASE | re.MULTILINE)
MARKER_SCAN_LINES = 20

HOOK_DIR = Path(__file__).resolve().parent
FROZEN_LIST_PATH = HOOK_DIR / "frozen_paths.txt"


def _load_frozen_patterns() -> list[str]:
    """讀取 frozen_paths.txt，回傳非註解、非空白的 glob pattern 清單。"""
    if not FROZEN_LIST_PATH.exists():
        return []
    patterns = []
    for line in FROZEN_LIST_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        patterns.append(line)
    return patterns


def _extract_file_path(tool_name: str, tool_input: dict) -> str | None:
    if tool_name == "NotebookEdit":
        return tool_input.get("notebook_path")
    return tool_input.get("file_path")


def _matches_frozen_list(rel_path: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(rel_path, pattern) for pattern in patterns)


def _read_head(file_path: str, n_lines: int) -> str:
    """讀取檔案前 n_lines 行；檔案不存在或讀取失敗時回傳空字串。"""
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return "".join(islice(f, n_lines))
    except OSError:
        return ""


def _has_inline_marker(file_path: str) -> bool:
    head = _read_head(file_path, MARKER_SCAN_LINES)
    if not head:
        return False
    return bool(FROZEN_MARKER_RE.search(head) or FROZEN_TAG_RE.search(head))


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        return 0

    tool_name = payload.get("tool_name", "")
    if tool_name not in {"Edit", "Write", "MultiEdit", "NotebookEdit"}:
        return 0

    tool_input = payload.get("tool_input", {}) or {}
    file_path = _extract_file_path(tool_name, tool_input)
    if not file_path:
        return 0

    project_dir = payload.get("cwd") or os.environ.get("CLAUDE_PROJECT_DIR", "")
    try:
        rel_path = str(Path(file_path).resolve().relative_to(Path(project_dir).resolve()))
    except (ValueError, OSError):
        rel_path = file_path

    patterns = _load_frozen_patterns()
    frozen_by_list = _matches_frozen_list(rel_path, patterns)
    frozen_by_marker = os.path.isfile(file_path) and _has_inline_marker(file_path)

    if not (frozen_by_list or frozen_by_marker):
        return 0

    reason_bits = []
    if frozen_by_list:
        reason_bits.append("路徑列在 .claude/hooks/frozen_paths.txt")
    if frozen_by_marker:
        reason_bits.append("檔案開頭含 FROZEN 標記")

    reason = (
        f"Frozen／核心模組保護擋下這次修改：{rel_path}（{'；'.join(reason_bits)}）。\n"
        "CLAUDE.md 規定 frozen／核心模組須先設計、後實作（TDD），不得直接修改。"
        "請先跟使用者確認設計方向；若確認可以解除保護，"
        "請使用者自行從 frozen_paths.txt 移除此路徑或拿掉檔案內的 FROZEN 標記。"
    )

    print(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": reason,
        }
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())

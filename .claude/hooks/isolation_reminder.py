#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Claude Code PreToolUse hook：production／experiments 隔離提醒

用途：
  沿用 player_label 專案「production 和 experiments 的檔案不互通，不擅自去參考」
  的規定，在編輯 production/ 底下的檔案時，偵測是否引用了
  experiments/（反之亦然），若命中就以 additionalContext 提醒 Claude，
  但不擋下這次編輯（這是提醒，不是紅線）。
輸入：
  stdin 傳入的 PreToolUse JSON。
輸出：
  命中時印出 permissionDecision=allow + additionalContext 的 JSON
  （直接放行，但把提醒文字帶回給 Claude；同時附上 systemMessage 讓使用者
  在 transcript 也看得到）；未命中則安靜地 exit 0。
"""
from __future__ import annotations

import json
import re
import sys

PRODUCTION_REF_RE = re.compile(
    r"(\b(from|import)\s+experiments\b|['\"]\.\./experiments/|['\"]experiments/)",
    re.IGNORECASE,
)
EXPERIMENTS_REF_RE = re.compile(
    r"(\b(from|import)\s+production\b|['\"]\.\./production/|['\"]production/)",
    re.IGNORECASE,
)


def _extract_edit_texts(tool_name: str, tool_input: dict) -> tuple[str | None, str]:
    if tool_name == "Write":
        return tool_input.get("file_path"), tool_input.get("content", "")
    if tool_name == "Edit":
        return tool_input.get("file_path"), tool_input.get("new_string", "")
    if tool_name == "MultiEdit":
        edits = tool_input.get("edits", []) or []
        text = "\n".join(e.get("new_string", "") for e in edits)
        return tool_input.get("file_path"), text
    if tool_name == "NotebookEdit":
        return tool_input.get("notebook_path"), tool_input.get("new_source", "")
    return None, ""


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        return 0

    tool_name = payload.get("tool_name", "")
    tool_input = payload.get("tool_input", {}) or {}
    file_path, text = _extract_edit_texts(tool_name, tool_input)
    if not file_path or not text:
        return 0

    norm_path = file_path.replace("\\", "/")
    in_production = "/production/" in f"/{norm_path}" or norm_path.startswith("production/")
    in_experiments = "/experiments/" in f"/{norm_path}" or norm_path.startswith("experiments/")

    warning = None
    if in_production and PRODUCTION_REF_RE.search(text):
        warning = (
            f"production／experiments 隔離提醒：正在編輯的 production 檔案"
            f"（{file_path}）疑似引用了 experiments/。CLAUDE.md 規定兩邊不互通，"
            "不可互相引用或假設一致，請確認這是否為刻意的引用。"
        )
    elif in_experiments and EXPERIMENTS_REF_RE.search(text):
        warning = (
            f"production／experiments 隔離提醒：正在編輯的 experiments 檔案"
            f"（{file_path}）疑似引用了 production/。CLAUDE.md 規定兩邊不互通，"
            "不可互相引用或假設一致，請確認這是否為刻意的引用。"
        )

    if not warning:
        return 0

    print(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "allow",
            "additionalContext": warning,
        },
        "systemMessage": warning,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())

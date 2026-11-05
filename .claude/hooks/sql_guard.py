#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Claude Code PreToolUse hook：SQL 資料操作紅線防護

用途：
  攔截 Edit／Write／MultiEdit／NotebookEdit／Bash 呼叫，偵測即將寫入或執行的
  內容是否違反 .claude/agents/sql.md 的資料操作紅線：
    1. DROP／DELETE／TRUNCATE／ALTER 破壞性語句（AI 一律不得執行或寫入）
    2. allow_destructive=True（.claude/agents/sql.md 明文禁止出現在程式中）
    3. SELECT *（.claude/agents/sql.md 明文禁止）
    4. 用 f-string 拼接 SQL 語句（.claude/agents/sql.md 規定一律參數化）
  命中任何一條，直接擋下該次工具呼叫，並把原因回傳給 Claude 自行修正。

比對方式：
  只認 SQL 語句的形狀，而不是單字本身：例如 DROP 後面要接 TABLE／DATABASE／VIEW…、
  DELETE 要接 FROM、ALTER／TRUNCATE 要接 TABLE…。pandas 的 df.drop(columns=…)、
  reset_index(drop=True)、說明文字裡的 drop／alter 等字不會被誤擋。
  f-string 也要長得像 SQL（SELECT … FROM、INSERT INTO、UPDATE … SET、WHERE 欄位 = …）才算。
輸入：
  stdin 傳入的 PreToolUse JSON（tool_name／tool_input）。
輸出：
  命中違規時印出 permissionDecision=deny 的 JSON 並以 exit 0 結束
  （JSON 決定優先於 exit code）；否則安靜地 exit 0。

逃生閥：若新增內容中含有 `claude-hook-allow: sql-guard` 標記字串，本次呼叫略過
檢查（僅供極少數已知的誤判情境使用，用完建議移除標記）。
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ESCAPE_HATCH = "claude-hook-allow: sql-guard"

CODE_EXTENSIONS = {".py", ".sql", ".ipynb"}

DESTRUCTIVE_RE = re.compile(
    r"\bDROP\s+(?:TEMPORARY\s+)?(?:TABLE|DATABASE|VIEW|DICTIONARY|COLUMN|PARTITION|INDEX|PROJECTION|"
    r"USER|ROLE|QUOTA|FUNCTION|ROW\s+POLICY|SETTINGS\s+PROFILE|NAMED\s+COLLECTION)\b"
    r"|\bDELETE\s+FROM\b"
    r"|\bALTER\s+(?:TABLE|DATABASE|USER|ROLE|QUOTA|ROW\s+POLICY|SETTINGS\s+PROFILE)\b"
    r"|\bTRUNCATE\s+(?:TABLE|DATABASE|ALL\s+TABLES)\b",
    re.IGNORECASE,
)
ALLOW_DESTRUCTIVE_RE = re.compile(r"allow_destructive\s*=\s*True", re.IGNORECASE)
SELECT_STAR_RE = re.compile(r"SELECT\s+\*", re.IGNORECASE)
FSTRING_RE = re.compile(
    r"f(?P<q>'''|\"\"\"|'|\")(?P<body>.*?)(?P=q)", re.IGNORECASE | re.DOTALL
)
SQL_STATEMENT_RE = re.compile(
    r"\bSELECT\b[\s\S]*?\bFROM\b"
    r"|\bINSERT\s+INTO\b"
    r"|\bDELETE\s+FROM\b"
    r"|\bUPDATE\s+[\w.`\"]+\s+SET\b"
    r"|\bALTER\s+TABLE\b"
    r"|\bCREATE\s+(?:TABLE|VIEW|DATABASE)\b"
    r"|\bWHERE\s+[\w.`\"()]+\s*(?:=|!=|<>|>|<|\bIN\b|\bLIKE\b|\bBETWEEN\b)",
    re.IGNORECASE,
)


def _extract_edit_texts(tool_name: str, tool_input: dict) -> tuple[str | None, str]:
    """
    依工具類型取出「檔案路徑」與「即將寫入／執行的文字」。

    輸入：tool_name、tool_input（PreToolUse JSON 的對應欄位）
    輸出：(file_path 或 None, 待檢查文字)
    """
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
    if tool_name == "Bash":
        return None, tool_input.get("command", "")
    return None, ""


def _is_code_target(tool_name: str, file_path: str | None) -> bool:
    """Bash 一律檢查；檔案類工具只檢查 .py／.sql／.ipynb（避免誤判 .md 文件說明文字）。"""
    if tool_name == "Bash":
        return True
    if not file_path:
        return False
    return Path(file_path).suffix.lower() in CODE_EXTENSIONS


def _find_violations(text: str) -> list[str]:
    """回傳命中的紅線說明列表；空列表代表沒有違規。"""
    violations = []

    m = DESTRUCTIVE_RE.search(text)
    if m:
        stmt = " ".join(m.group(0).upper().split())
        violations.append(
            f"偵測到破壞性 SQL 語句「{stmt}」"
            "（.claude/agents/sql.md：DROP／DELETE／TRUNCATE／ALTER 只能由人工手動執行，AI 一律不得寫入或執行）"
        )

    if ALLOW_DESTRUCTIVE_RE.search(text):
        violations.append("偵測到 allow_destructive=True（.claude/agents/sql.md 明文規定程式中不應出現）")

    if SELECT_STAR_RE.search(text):
        violations.append("偵測到 SELECT *（.claude/agents/sql.md 禁止 SELECT *，請只列實際需要的欄位）")

    for m in FSTRING_RE.finditer(text):
        body = m.group("body")
        if "{" in body and SQL_STATEMENT_RE.search(body):
            violations.append(
                "偵測到疑似用 f-string 拼接 SQL（.claude/agents/sql.md 規定一律用 %(name)s 參數化 + "
                "fetch_query_with_params，禁止 f-string 拼 SQL）"
            )
            break

    return violations


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        return 0

    tool_name = payload.get("tool_name", "")
    tool_input = payload.get("tool_input", {}) or {}

    file_path, text = _extract_edit_texts(tool_name, tool_input)
    if not text or not _is_code_target(tool_name, file_path):
        return 0

    if ESCAPE_HATCH in text:
        return 0

    violations = _find_violations(text)
    if not violations:
        return 0

    reason = "SQL 資料操作紅線防護擋下這次操作：\n- " + "\n- ".join(violations)
    if file_path:
        reason += f"\n（檔案：{file_path}）"
    reason += (
        "\n請依 .claude/agents/sql.md 修正後再試一次；如確定是誤判，可在內容中加註 "
        f"`{ESCAPE_HATCH}` 略過本次檢查。"
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

#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Claude Code PreToolUse hook：專案防護單一入口
#
# 用途：
#   settings.json 只掛這一個 command，由本腳本依序執行三道檢查，
#   第一個有輸出的檢查即為最終決定（拒絕優先於提醒）：
#     1. sql_guard.py          擋破壞性 DDL／DML 關鍵字、allow_destructive 旗標、
#                              全欄位查詢、字串插值拼 SQL           → 拒絕
#     2. frozen_guard.py       擋修改 frozen／核心模組                → 拒絕
#     3. isolation_reminder.py production／experiments 互相引用      → 放行 + 提醒
# 輸入：
#   stdin 傳入的 PreToolUse JSON（本腳本只轉發，不解析）。
# 輸出：
#   命中的那道檢查所產生的 hook JSON；全部沒命中則無輸出。
#
# 各檢查自己會依 tool_name 決定要不要動作，所以 matcher 可以只掛一組。
# 一律 exit 0：權限決定走 JSON，不靠 exit code；某道檢查掛掉時 fail-open，
# 不因防護腳本自身錯誤而卡住工作。
# ---------------------------------------------------------------------------
set -uo pipefail

HOOK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PAYLOAD="$(cat)"

for guard in sql_guard.py frozen_guard.py isolation_reminder.py; do
    [ -f "$HOOK_DIR/$guard" ] || continue
    output="$(printf '%s' "$PAYLOAD" | python3 "$HOOK_DIR/$guard" 2>/dev/null)"
    if [ -n "$output" ]; then
        printf '%s\n' "$output"
        exit 0
    fi
done

exit 0

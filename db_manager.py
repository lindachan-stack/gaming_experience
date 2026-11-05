# -*- coding: utf-8 -*-
"""
D1～W6 共用的 ClickHouse 連線管理

用途：
  --target test／prod 只切換各步驟寫入的表名（見各 etl/steps 的 TABLES
  對照），連線本身共用同一個集群、同一套 .env 憑證；上線時不需要改這裡，
  只需把呼叫端的 --target 從 test 換成 prod。介面與安全機制比照
  experiments/db_manager.py（該檔給 experiments/*.py 自己的腳本用，兩邊
  各自獨立、不互相 import，見 CLAUDE.md「production 與 experiments 兩分支」）。

claude-hook-allow: sql-guard（下方 DESTRUCTIVE_KEYWORDS 只是列出要攔截的
關鍵字字串，guard() 用它們來拒絕執行，本檔沒有任何一句真正的刪除類語句）
"""
from __future__ import annotations

import os
import re

import pandas as pd
from clickhouse_driver import Client
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

# 帳號 tableau_readonly 的實際授權（SHOW GRANTS 查證）：
# for_math_aggregations 可 INSERT／CREATE／OPTIMIZE 及刪除類語句，
# tableau／default／mochi／xinbonus_mochi 只有 SELECT。
WRITABLE_DATABASES = frozenset({"for_math_aggregations"})
DEFAULT_DATABASE = "tableau"

# 不改動資料的語句，任何 database 都允許
READ_KEYWORDS = frozenset(
    {"SELECT", "WITH", "SHOW", "DESCRIBE", "DESC", "EXPLAIN", "EXISTS", "CHECK"}
)
# 刪除類語句：即使 database 可寫，也必須先向使用者確認
DESTRUCTIVE_KEYWORDS = frozenset({"DROP", "DELETE", "TRUNCATE", "ALTER"})


def first_keyword(sql: str) -> str:
    """取出 SQL 的第一個關鍵字（大寫）。

    先移除 `--` 行註解與 `/* */` 區塊註解，因此檔首帶說明註解的 .sql 也能
    正確判定語句類型。
    """
    without_block = re.sub(r"/\*.*?\*/", " ", sql, flags=re.DOTALL)
    without_line = re.sub(r"--[^\n]*", " ", without_block)
    match = re.search(r"[A-Za-z]+", without_line)
    return match.group(0).upper() if match else ""


class DatabaseManager:
    """D1～W6 共用的 ClickHouse 連線管理（介面與安全機制比照 experiments/db_manager.py）。"""

    def __init__(self, db_name: str | None = None):
        """建立連線。

        db_name 省略時取環境變數 CH_DATABASE，未設定則用 DEFAULT_DATABASE。
        """
        self._db_name = db_name or os.getenv("CH_DATABASE") or DEFAULT_DATABASE
        self.client = self._get_client(self._db_name)

    @property
    def db_name(self) -> str:
        """目前連線的 database 名稱。"""
        return self._db_name

    @property
    def writable(self) -> bool:
        """目前 database 是否允許寫入。"""
        return self._db_name in WRITABLE_DATABASES

    def _get_client(self, db_name: str) -> Client:
        # 900 秒：W6 大批次 INSERT 的往返時間本身就長，非單次網路抖動、
        # 靠重試就能解決的量級，逾時門檻需相應放寬。
        #
        # secure 預設 True（正式連線走 TLS，行為不變）；CH_SECURE=0 只給 CI
        # 用的臨時 ClickHouse 容器（無 TLS 的本機連線）用，不影響本機／正式環境。
        return Client(
            host=os.getenv("CH_HOST"),
            port=int(os.getenv("CH_PORT", 9440)),
            database=db_name,
            user=os.getenv("CH_USER"),
            password=os.getenv("CH_PASSWORD"),
            secure=os.getenv("CH_SECURE", "1") != "0",
            verify=False,
            send_receive_timeout=900,
            compression="lz4",
            # 伺服器端 receive_timeout／send_timeout 預設 300 秒，資源吃緊
            # 或查詢較重（JOIN／quantile 運算）時可能等不到客戶端送完資料就
            # 主動斷線；跟上面 send_receive_timeout（客戶端等伺服器）方向相反，
            # 兩邊都要放寬才夠。
            settings={"receive_timeout": 600, "send_timeout": 600},
        )

    def guard(self, sql: str) -> str:
        """檢查 sql 是否允許執行，回傳判定出的關鍵字。

        刪除類語句一律拒絕，不論在哪個 database，也沒有參數可以繞過——
        刪除只能由使用者人工在 client 下指令。唯讀 database 另外只放行
        READ_KEYWORDS。不符合時丟 PermissionError，不會送到 DB。
        """
        keyword = first_keyword(sql)
        if keyword in READ_KEYWORDS:
            return keyword
        if keyword in DESTRUCTIVE_KEYWORDS:
            raise PermissionError(
                f"{keyword} 屬刪除類語句，程式一律不得執行；"
                f"如確定要刪除，請由使用者人工在 client 下指令"
            )
        if not self.writable:
            raise PermissionError(
                f"database `{self._db_name}` 為唯讀，不可執行 {keyword or 'unknown'}；"
                f"可寫入的只有 {sorted(WRITABLE_DATABASES)}"
            )
        return keyword

    def _execute(self, sql: str, params: dict | None = None, retry_timeout: bool = False):
        """Execute with one automatic reconnect on connection reset or (if retry_timeout) read timeout.

        retry_timeout 只在呼叫端確認「同一批資料重送一次也不會產生壞資料」時才傳
        True——ReplacingMergeTree／AggregatingMergeTree 目標表的重複寫入會被
        merge 自然吸收；純 MergeTree 目標表重試等於可能產生真正的重複列，
        故預設 False，逾時就照舊往上拋，交回呼叫端手動重跑。SELECT 類查詢
        沒有副作用，一律視為可重試。
        """
        self.guard(sql)
        kwargs = dict(with_column_types=True)
        if params:
            kwargs["params"] = params
        try:
            return self.client.execute(sql, **kwargs)
        except Exception as e:
            is_reset = "Connection reset" in str(e) or "ConnectionReset" in type(e).__name__
            is_retryable_timeout = retry_timeout and isinstance(e, TimeoutError)
            if is_reset or is_retryable_timeout:
                self.client = self._get_client(self._db_name)
                return self.client.execute(sql, **kwargs)
            raise

    def fetch_query(self, sql: str) -> pd.DataFrame:
        """執行唯讀查詢，回傳 DataFrame。"""
        res, cols = self._execute(sql, retry_timeout=True)
        return pd.DataFrame(res, columns=[col[0] for col in cols])

    def fetch_query_with_params(self, sql: str, params: dict) -> pd.DataFrame:
        """執行帶參數的唯讀查詢，回傳 DataFrame。"""
        res, cols = self._execute(sql, params, retry_timeout=True)
        return pd.DataFrame(res, columns=[col[0] for col in cols])

    def execute_write(self, sql: str, params: dict | None = None, retryable: bool = False) -> None:
        """在可寫 database 執行寫入語句（INSERT／CREATE／OPTIMIZE …）。

        guard() 一律擋下刪除類語句並丟 PermissionError，沒有參數可以放行——
        刪除只能由使用者人工下指令。

        retryable：目標表是 ReplacingMergeTree／AggregatingMergeTree 時才傳
        True，讓逾時也自動重連重試一次；純 MergeTree 表維持預設 False，
        避免重試造成無法自動消除的重複列（見 _execute() docstring）。
        """
        self._execute(sql, params, retry_timeout=retryable)

    def load_sql(self, path: str) -> str:
        """讀取 SQL 檔案內容。"""
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

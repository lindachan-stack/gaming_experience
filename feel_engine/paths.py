# -*- coding: utf-8 -*-
"""
專案目錄（SPEC 第 2 節）。每份資料只存一份：原始快取逐日累積、dataset 逐日存，期間不同的 run 共用。

  data/bet_records/bet_records_{商戶}_{遊戲}_tz+0/   注單逐日快取（UTC 日檔；資料夾不含日期，延伸期間只補新的日子）
  data/tags/dwt_player_game_tag_wi_test_{商戶}_{幣別}_{遊戲}_lag1w_asof/   週標籤（apply_week=W.parquet）
  {key} = {商戶}_{幣別}_{遊戲}（例：TCG_PHP_22071）
  output/{key}/dataset/{美東日}.parquet       決策點，依 session_date 一天一檔（所有 run 共用，可重跑產生）
  output/{key}/dataset/{美東日}.json          該日的建置指紋、列數、逐日統計（build 報告用）
  output/{key}/dataset/builds/{build_id}.json 一次 build 讀了哪些日檔、R1–R7 檢查結果（資料檢查報告用）
  output/{key}/_build_tmp/                    build 中的暫存（分桶、月初基準）；build 結束即刪
  {run} = {key}/train_{起}-{迄}_valid_{起}[-{迄}]（例：TCG_PHP_22071/train_20260831-20260913_valid_20260921-20260927）
  output/{run}/model/              各實驗的樹結構（tree_rules.json）與驗證期間預測，供 evaluate、compare 讀取
  output/{run}/state_debug/        build 時隨機 5 位玩家的完整逐手狀態
  result/{run}/build/              build 報告
  result/{run}/                    實驗結果：summary.html（AUC ＋ 信心）、{split}_tag_{tags}.html（每個實驗一頁）
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent
CONFIG_PATH = PACKAGE_DIR / "config" / "config.yaml"
MODEL_CONFIG_PATH = PACKAGE_DIR / "config" / "model.yaml"     # 決策樹要放進 X 的欄位

DATA_DIR = PROJECT_ROOT / "data"
BET_RECORDS_ROOT = DATA_DIR / "bet_records"
TAG_ROOT = DATA_DIR / "tags"
OUTPUT_ROOT = PROJECT_ROOT / "output"
RESULT_ROOT = PROJECT_ROOT / "result"


def bet_records_dir(merchant: str, game: str) -> Path:
    """注單快取（UTC 日檔）；與 data_loading/bet_records_loader.py 的 cache_dir 同名。"""
    return BET_RECORDS_ROOT / f"bet_records_{merchant}_{game}_tz+0"


def tag_dir(merchant: str, currency: str, game: str) -> Path:
    """週標籤快取（延遲一週、往回取最新）；與 data_loading/tag_loader.py 的 cache_dir 同名。"""
    return TAG_ROOT / f"dwt_player_game_tag_wi_test_{merchant}_{currency}_{game}_lag1w_asof"


@dataclass(frozen=True)
class StorePaths:
    """output/{key}/：同一商戶 × 幣別 × 遊戲共用的逐日 dataset 與 build 暫存。可 pickle，傳給 worker 行程使用。"""

    key: str

    @property
    def dataset_dir(self) -> Path:
        return OUTPUT_ROOT / self.key / "dataset"

    def day(self, d: date | str) -> Path:
        return self.dataset_dir / f"{d}.parquet"

    def day_meta(self, d: date | str) -> Path:
        return self.dataset_dir / f"{d}.json"

    @property
    def builds_dir(self) -> Path:
        return self.dataset_dir / "builds"

    def build_meta(self, build_id: str) -> Path:
        return self.builds_dir / f"{build_id}.json"

    # output/{key}/_build_tmp/：build 中的暫存，結束即刪
    @property
    def tmp_dir(self) -> Path:
        return OUTPUT_ROOT / self.key / "_build_tmp"

    @property
    def buckets(self) -> Path:
        return self.tmp_dir / "buckets"

    @property
    def month_base(self) -> Path:
        return self.tmp_dir / "month_base.parquet"

    @property
    def pre_range_last(self) -> Path:
        """讀取範圍起點前 session_gap 內，每位玩家最後一個付費手的時間（判斷 session 起點是否被切到）。"""
        return self.tmp_dir / "pre_range_last.parquet"

    @property
    def dataset_parts(self) -> Path:
        return self.tmp_dir / "dataset_parts"


@dataclass(frozen=True)
class RunPaths:
    """一次實驗的輸出路徑；name = data.run_name(cfg)（商戶_幣別_遊戲/訓練期間_驗證期間）。可 pickle，傳給 worker 行程使用。"""

    name: str

    @property
    def store(self) -> StorePaths:
        return StorePaths(self.name.split("/", 1)[0])

    @property
    def output_dir(self) -> Path:
        return OUTPUT_ROOT / self.name

    @property
    def result_dir(self) -> Path:
        return RESULT_ROOT / self.name

    @property
    def state_debug(self) -> Path:
        return self.output_dir / "state_debug"

    # result/{run}/build/
    @property
    def build_dir(self) -> Path:
        return self.result_dir / "build"

    @property
    def dq_report(self) -> Path:
        return self.build_dir / "data_quality_report.md"

    @property
    def build_report(self) -> Path:
        return self.build_dir / "build_report.md"

    # 實驗 = split × tags（model.yaml）
    def model_dir(self, split: str, tags: str) -> Path:
        """output/{run}/model/{split}/tag_{tags}/：程式讀取用（樹結構、驗證期間預測）。"""
        return self.output_dir / "model" / split / f"tag_{tags}"

    def tree_rules(self, split: str, tags: str) -> Path:
        return self.model_dir(split, tags) / "tree_rules.json"

    def model_predictions(self, split: str, tags: str) -> Path:
        return self.model_dir(split, tags) / "valid_predictions.parquet"

    def leaf_confidence(self, split: str, tags: str) -> Path:
        """evaluate 寫入：每片葉子的信心（原始、合併後）與合併對照，供 summary.html 算信心分數。"""
        return self.model_dir(split, tags) / "leaf_confidence.json"

    def five_feelings(self, split: str, tags: str, mode: str = "") -> Path:
        """evaluate 寫入：五個體感補位選中的子規則與放不進去的版本（SPEC 9.4）；mode = "2" → 五個體感2 的 five_feelings_2.json。"""
        return self.model_dir(split, tags) / f"five_feelings{'_' + mode if mode else ''}.json"

    # result/{run}/：給人看的實驗結果
    def experiment_html(self, split: str, tags: str) -> Path:
        return self.result_dir / f"{split}_tag_{tags}.html"

    @property
    def summary_html(self) -> Path:
        return self.result_dir / "summary.html"


@dataclass(frozen=True)
class VariantPaths(RunPaths):
    """實驗的變體（feel_engine/variants.py；不在 model.yaml 清單、不進 summary.html）：
    output/{run}/variants/{split}_tag_{tags}[_{variant}]/（程式讀取用）、result/{run}/{split}_tag_{tags}[_{variant}].html；
    variant 為空時不加後綴（tags 不是正式的 tags 名稱時用，不會與正式實驗撞名）。"""

    variant: str = ""

    def _stem(self, split: str, tags: str) -> str:
        return f"{split}_tag_{tags}" + (f"_{self.variant}" if self.variant else "")

    def model_dir(self, split: str, tags: str) -> Path:
        return self.output_dir / "variants" / self._stem(split, tags)

    def experiment_html(self, split: str, tags: str) -> Path:
        return self.result_dir / f"{self._stem(split, tags)}.html"


def run_paths(name: str) -> RunPaths:
    return RunPaths(str(name))


def rel(path: Path) -> str:
    """報告與 log 中顯示相對專案根目錄的路徑。"""
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)

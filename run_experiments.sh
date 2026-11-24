#!/usr/bin/env bash
# =============================================================================
# 體感模組實驗：跑 model.yaml 的 experiments 全部（run --clean → evaluate）
#
#   bash run_experiments.sh            # 跑實驗清單
#   bash run_experiments.sh --build    # 先 build（只建還沒有或已過期的日子）再跑
#   MODEL_CONFIG=feel_engine/config/my_model.yaml bash run_experiments.sh   # 換一份特徵與實驗設定檔
#
# 換訓練期間／驗證期間：改 config.yaml 的 data 再加 --build；不同期間的結果寫到不同資料夾。
# 結果：result/<run>/summary.html（實驗比較）、result/<run>/<split>_tag_<tags>.html（每個實驗一頁）。
# 注意：不同設定檔的結果會寫到同一個 split／tag 位置（後跑的覆蓋先跑的）；--clean 也會清掉變體頁。
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"

PY="${PY:-/Users/linda_chan/miniconda3/envs/py311/bin/python}"   # 需要 polars、scikit-learn
MODEL_CONFIG="${MODEL_CONFIG:-feel_engine/config/model.yaml}"

if [[ "${1:-}" == "--build" ]]; then
  # version_map 填好後拿掉 --provisional-arm
  "$PY" -m feel_engine.cli build --provisional-arm
fi

"$PY" -m feel_engine.cli run --model-config "$MODEL_CONFIG" --clean   # 先清掉舊結果；訓練 → 驗證期間預測 → summary.html
"$PY" -m feel_engine.cli evaluate --model-config "$MODEL_CONFIG"     # 第 9 節：每個實驗一頁 HTML
echo "完成：用瀏覽器打開 result/<run>/summary.html（點實驗名稱看每個實驗）"

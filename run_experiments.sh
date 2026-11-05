#!/usr/bin/env bash
# =============================================================================
# 體感模組實驗：跑 feel_engine/config/model.yaml 的 experiments 全部，最後產生綜合比較
#
#   bash run_experiments.sh            # 跑實驗清單
#   bash run_experiments.sh --build    # 先 build（config.yaml 的 data：只建訓練期間＋驗證期間還沒有的日子）再跑
#   換訓練期間／驗證期間：直接改 config.yaml 的 data 再加 --build；dataset 一天一檔、所有期間共用（不重複存），
#   結果寫到不同的 result 資料夾，彼此不覆蓋
#
# 時間外驗證：訓練期間的決策點長樹，驗證日的決策點算 AUC 與每片葉子各版本的留存率
#
# 實驗清單與順序在 model.yaml 的 experiments（每個實驗 = split × tags）：
#   split：all（一棵樹）／lifecycle（依生命週期分群）／value（依價值分群，只建模 config.yaml splits.groups.value）
#   tags ：放進 X 的 TAG_ 組合，定義在 model.yaml 的 tags（none = 無、lifecycle、value）
#
# 結果（瀏覽器打開）：result/<run>/summary.html（實驗比較：AUC ＋ 信心分數，依清單順序）
#                     result/<run>/<split>_tag_<tags>.html（每個實驗：決策樹、葉子 × 版本留存率與信心）
# 想試另一份設定：複製 feel_engine/config/model.yaml 改名後
#   MODEL_CONFIG=feel_engine/config/model_no_flags.yaml bash run_experiments.sh
# 注意：不同設定檔跑出的結果會寫到同一個 split／tag 位置（後跑的覆蓋先跑的）；
#       各實驗用的設定記在 output/<run>/model/<split>/tag_<tags>/tree_rules.json（model_config_sha）。
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"

PY="${PY:-/Users/linda_chan/miniconda3/envs/py311/bin/python}"   # 需要 polars、scikit-learn
MODEL_CONFIG="${MODEL_CONFIG:-feel_engine/config/model.yaml}"

if [[ "${1:-}" == "--build" ]]; then
  # version_map 填好後拿掉 --provisional-arm；已建好的日子沿用，只建缺的或過期的
  "$PY" -m feel_engine.cli build --provisional-arm
fi

"$PY" -m feel_engine.cli run --model-config "$MODEL_CONFIG" --clean   # 先清掉舊結果；訓練 → 驗證日預測 → summary.html
"$PY" -m feel_engine.cli evaluate --model-config "$MODEL_CONFIG"     # 第 9 節：每個實驗一頁 HTML
echo "完成：用瀏覽器打開 result/<run>/summary.html（點實驗名稱看每個實驗）"

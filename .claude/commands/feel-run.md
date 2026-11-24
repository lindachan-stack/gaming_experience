---
description: 體感模組流程：重建 dataset、資料檢查、訓練決策樹、實驗比較，並摘要結果
argument-hint: [check | build | run [<split> <tags>] | compare]（不給參數 = check）
---

依 `.claude/agents/feel-trainer.md`（體感訓練大師）的規範執行 $ARGUMENTS，在專案根目錄、用 conda `py311`（`/Users/linda_chan/miniconda3/envs/py311/bin/python`）：

| 參數 | 執行 |
|---|---|
| `check`（預設） | `python -m pytest -q feel_engine/tests`，再執行 `test.ipynb`（全部 cell），回報測試數與 ✅／❌ 檢查數 |
| `build` | `python -m feel_engine.cli build --provisional-arm`（只建本 run 缺的或過期的日子，已建好的沿用；version_map 已填則不加 `--provisional-arm`），接著做 `check` |
| `run` | `bash run_experiments.sh`：跑 `model.yaml` 的 experiments 全部（訓練期間長樹、驗證期間評估）＋ 第 9 節 evaluate |
| `run <split> <tags>` | `python -m feel_engine.cli run --split <split> --tags <tags>`（可逗號分隔，跑所有組合） |
| `compare` | `python -m feel_engine.cli compare` |
| `evaluate` | `python -m feel_engine.cli evaluate`：每個實驗一頁 HTML（決策樹、葉子 × 版本留存率、信心） |

執行後回報：

1. 實際跑了哪些指令、是否成功；失敗時貼出錯誤並判斷原因，不要自行改設定重試。
2. `run`／`compare`：讀 `result/{run}/summary.html`，依實驗編號摘要共同樣本、各實驗 AUC 與 95% 區間、相對基準 all／無的差距與判定。只有區間不含 0 才說「較佳」。`evaluate`：看 log 的各實驗葉子數、合併後葉子數與信心高的片數，必要時打開 `result/{run}/{split}_tag_{tags}.html`；提醒信心低的最高留存版本不能直接當版本效果。
3. 若 `model.yaml`、`config.yaml` 或逐日 dataset（`output/{商戶}_{幣別}_{遊戲}/dataset/`）在這次之前被改過，提醒哪些既有結果是舊設定產生的（看 HTML 與 `output/{run}/model/*/tag_*/tree_rules.json` 的 `run_at`、各日 `{日期}.json` 的 `build_id`）。
4. 不改任何程式或設定；需要調整時列出建議，交由使用者決定。

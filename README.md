# 體感模組（feel_engine）

依 `SPEC.md` 從注單（bet_records）與週標籤建立決策點樣本，訓練「20 手內是否離場」的決策樹，並評估每片葉子在各遊戲版本下的留存。

- **時間外驗證**：訓練期間（`config.yaml` 的 `data.train_start`～`train_end`）長樹，驗證期間（`valid_start`～`valid_end`）算 AUC 與葉子 × 版本留存率。決策點依 `session_date`（session 起點的美東日期）歸屬，跨午夜的 session 整段歸起點那天。
- **每份資料只存一份**：快取資料夾不含日期，延伸期間只補抓新的日子；dataset 一天一檔（`output/TCG_PHP_22071/dataset/{日期}.parquet`），所有 run 共用，build 只建還沒有或已過期的日子。
- **結果位置**：`{商戶}_{幣別}_{遊戲}/train_{起}-{迄}_valid_{起}[-{迄}]/`，例如 `TCG_PHP_22071/train_20260928-20261011_valid_20261012-20261018/`（下文簡寫為 `{run}`）。
- **測試規則**（SPEC 第 2 節）：訓練 09-28～10-11、驗證只用下一週 10-12～10-18；只跑實驗、不產生變體頁；生命週期的流失併入回流（`config.yaml` 的 `regroup`）；舊結果移到 `result／output/TCG_PHP_22071/delete/`。

執行方式：

- 所有指令都在**專案根目錄**執行。
- Python 用 conda `py311`（需要 polars、scikit-learn）：`/Users/linda_chan/miniconda3/envs/py311/bin/python`。下文的 `python` 都指它。

## 最常用：一次跑完實驗

```bash
bash run_experiments.sh            # 清掉舊結果 → 跑 model.yaml 的實驗清單 → 每個實驗一頁 HTML ＋ summary.html
bash run_experiments.sh --build    # 先 build 再跑（改了 config.yaml 的日期、Y、權重、特徵設定時）
MODEL_CONFIG=feel_engine/config/my_model.yaml bash run_experiments.sh   # 換一份特徵與實驗設定檔
```

換訓練期間／驗證期間：改 `config.yaml` 的 `data` 再跑 `--build`；不同期間的結果放在不同資料夾，不會覆蓋。

## 補抓資料（只抓缺的日子）

```bash
python data_loading/bet_records_loader.py --start 2026-09-26 --end 2026-09-30   # 注單：UTC 日期 [start, end)
python data_loading/tag_loader.py --start 2026-09-28 --end 2026-10-05          # 週標籤：注單日期涵蓋的注單週
```

驗證期間最後一天是 D（美東）時，注單要抓到 UTC D + 2 天（隔天 12:00 前結束的 session ＋ 24 小時的 Y）。

## 指令

| 指令 | 做什麼 |
|---|---|
| `python -m feel_engine.cli build --provisional-arm` | 建本 run 還沒有或已過期的日子：資料檢查（R1–R7）→ 逐日 dataset → build 報告。`version_map` 未填時要加 `--provisional-arm`（以 `game_version` 暫代版本）；`--rebuild` 全部重建 |
| `python -m feel_engine.cli run --clean` | 清掉舊結果後，跑 `model.yaml` 的 `experiments` 全部（訓練期間長樹、驗證期間預測），並更新 summary.html |
| `python -m feel_engine.cli run --split value --tags none` | 只跑指定的實驗（可逗號分隔，跑所有組合）；不清其他結果 |
| `python -m feel_engine.cli run --model-config my.yaml` | 用另一份特徵與實驗設定檔 |
| `python -m feel_engine.cli evaluate` | 每片葉子 × 版本的留存率與信心，每個實驗寫一頁 HTML，並在 summary.html 加入信心分數（需先 `run`；`run` 之後 dataset 重建過、`regroup` 或 `model.yaml` 的欄位改過會要求重跑 `run`） |
| `python -m feel_engine.cli compare` | 只重做 `summary.html` |
| `python -m feel_engine.variants new` | 變體：lifecycle／無；補位改為每棵樹都要有每個版本（目前與實驗 5 的「五個體感2」分頁相同）。另寫一頁 `lifecycle_tag_none_new.html`，不進 summary.html |
| `python -m feel_engine.variants alltags --config my.yaml` | 變體：all，X = 快層 ＋ 標籤表所有類別標籤。另寫一頁 `all_tag_alltags.html`；`--config` 可換期間 |
| `python -m feel_engine.cli --help`、`run --help` | 列出子指令與參數 |

## 設定檔

| 檔案 | 改什麼 | 改完要跑 |
|---|---|---|
| `feel_engine/config/model.yaml` | 放進模型的特徵（`欄位: 中文名稱`）、TAG_ 組合、實驗清單與順序、one-hot 類別限定 | `bash run_experiments.sh` |
| `config.yaml` 的 `tree`、`splits`、`regroup`、`leaf_eval`、`five_feelings` | 樹的超參數、分群規則、類別合併（流失併入回流）、葉子 × 版本的門檻、補位設定 | `bash run_experiments.sh` |
| `config.yaml` 的 `data`、`labels`、`weighting`、`features`、`modules` | 期間、session／Y 定義、權重上限、旗標門檻、`version_map` | `bash run_experiments.sh --build` |

## 結果在哪

用瀏覽器打開 HTML 即可（單檔、不需網路）：

| 路徑 | 內容 |
|---|---|
| `result/{run}/summary.html` | **實驗比較**：各實驗的 AUC（95% 區間、與基準 all／無的差距）與信心分數，附評分依據；點實驗名稱進入該實驗頁 |
| `result/{run}/{split}_tag_{tags}.html` | **每個實驗一頁**：決策樹（葉子標最佳版本、留存率與信心）＋ 葉子 × 版本留存率熱圖。分頁：「原始葉子」、「合併無法比較的葉子」、「合併相近建議」、「五個體感1／2」（替沒被建議的版本在葉子內找子規則補位，SPEC 9.4） |
| `result/{run}/{split}_tag_{tags}_{變體}.html` | 變體頁；`run --clean` 會清掉，重跑變體指令即可 |
| `result/{run}/build/` | 資料檢查與 build 報告 |
| `output/TCG_PHP_22071/dataset/` | 逐日 dataset：`{日期}.parquet`（決策點）、`{日期}.json`（指紋、逐日統計）、`builds/`（每次 build 的紀錄） |
| `output/{run}/model/{split}/tag_{tags}/` | 程式讀取用：`tree_rules.json`、`valid_predictions.parquet`、`leaf_confidence.json`、`five_feelings.json`、`five_feelings_2.json`（變體在 `output/{run}/variants/`） |

## 檢查程式與資料

```bash
python -m pytest -q feel_engine/tests   # 單元測試（合成資料）
```

`test.ipynb`：dataset 長相與資料檢查，包含不經過 build 程式、從注單快取獨立重算抽樣玩家的逐欄比對（kernel 用 `py311` 或 `player_label`）。

## 目錄

```
SPEC.md                 規格（唯一依據）
run_experiments.sh      一次跑完實驗
data_loading/           唯一連 DB 的程式，抓資料到 data/
data/                   原始快取（注單逐日檔、週標籤）
feel_engine/            程式與設定
  cli.py、variants.py     指令入口、實驗的變體
  paths.py                路徑
  config/                 config.yaml、model.yaml
  pipeline/               建資料集（build）：data.py 分桶、build.py 狀態／Y／特徵、store.py 逐日 dataset、report.py 報告
  modeling/               決策樹（run）：model.py
  evaluation/             葉子 × 版本評估（evaluate）：propensity.py、leaf_eval.py、merge.py、five.py、evaluate.py（主流程）
  html/                   pages.py（頁面元件）、compare.py（summary.html）
  tests/                  單元測試
output/                 中間產物：逐日 dataset、各實驗的樹與預測
result/{run}/           build 報告、summary.html、每個實驗的 HTML
test.ipynb              資料檢查
```

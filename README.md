# 體感模組（feel_engine）

依 `SPEC.md` 從注單（bet_records）與週標籤建立決策點樣本，訓練「20 手內是否離場」的決策樹，並評估每片葉子在各遊戲版本下的留存。

- **時間外驗證**：訓練期間（`config.yaml` 的 `data.train_start`～`train_end`）長樹，驗證期間（`data.valid_start`～`valid_end`）算 AUC 與葉子 × 版本留存率。決策點依 `session_date`（session 起點的美東日期）歸屬：跨午夜的 session 整段歸起點那天；session 要在起點隔天 12:00 前結束、且之後 24 小時都有資料（看得到 Y），否則整段不用。
- **每份資料只存一份**：注單與標籤快取的資料夾不含日期，延伸期間只補抓新的日子；dataset 一天一檔（`output/TCG_PHP_22071/dataset/{日期}.parquet`），所有 run 共用，build 只建還沒有的日子。換訓練期間／驗證期間不會再存一份重複的資料。
- **商戶／幣別／遊戲**：`data.merchant`、`currency`、`game`（目前 TCG、PHP、22071）。實驗結果放在 `{商戶}_{幣別}_{遊戲}/train_{起}-{迄}_valid_{起}[-{迄}]/` 底下，目前是 `TCG_PHP_22071/train_20260928-20261011_valid_20261012-20261018/`（下表簡寫為 `{run}`）。

- 所有指令都在**專案根目錄**執行。
- Python 用 conda `py311`（需要 polars、scikit-learn）：`/Users/linda_chan/miniconda3/envs/py311/bin/python`。下文的 `python` 都指它。
- `python -m feel_engine.cli <子指令>` = 執行 `feel_engine/cli.py`（指令入口，依子指令呼叫其他檔案）。

## 最常用：一次跑完實驗

```bash
bash run_experiments.sh            # 清掉舊結果 → 跑 model.yaml 的實驗清單 → AUC 比較 → 每個實驗一頁 HTML
bash run_experiments.sh --build    # 先 build（只建還沒有或已過期的日子）再跑（改了 config.yaml 的日期、Y、權重設定時）
MODEL_CONFIG=feel_engine/config/my_model.yaml bash run_experiments.sh   # 換一份特徵與實驗設定檔
# 換訓練期間／驗證期間：直接改 config.yaml 的 data，再跑 --build；不同期間的結果放在不同資料夾，不會覆蓋
```

## 補抓資料（只抓缺的日子）

```bash
python data_loading/bet_records_loader.py --start 2026-09-26 --end 2026-09-30   # 注單：UTC 日期 [start, end)，已抓過的日子跳過
python data_loading/tag_loader.py --start 2026-09-28 --end 2026-10-05          # 週標籤：注單日期涵蓋的注單週
```

驗證期間最後一天是 D（美東）時，注單要抓到 UTC D + 2 天（隔天 12:00 前結束的 session ＋ 24 小時的 Y）。
```

## feel_engine.cli 指令

| 指令 | 做什麼 | 主要檔案 |
|---|---|---|
| `python -m feel_engine.cli build --provisional-arm` | 建本 run 還沒有的日子（訓練期間＋驗證期間）：資料檢查（R1–R7）→ 讀檔分桶（暫存，結束即刪）→ 寫入逐日 dataset，並產生本 run 的 build 報告。已建好的日子直接沿用；改了 build 相關設定或程式、或快取重抓過的日子會自動重建。`version_map` 未填時要加 `--provisional-arm`（以 `game_version` 暫代版本） | `pipeline/data.py`、`build.py`、`store.py`、`report.py` |
| `python -m feel_engine.cli build --rebuild --provisional-arm` | 本 run 的日子全部重建 | 同上 |
| `python -m feel_engine.cli run --clean` | 清掉舊結果後，跑 `model.yaml` 的 `experiments` 全部（訓練期間長樹、驗證期間預測），並更新 AUC 比較 | `modeling/model.py`、`html/compare.py`、`html/pages.py` |
| `python -m feel_engine.cli run --split value --tags none` | 只跑指定的實驗（可逗號分隔，跑所有組合）；不清其他結果 | 同上 |
| `python -m feel_engine.cli run --model-config my.yaml` | 用另一份特徵與實驗設定檔 | 同上 |
| `python -m feel_engine.cli evaluate` | 第 9 節：傾向分數 ＋ 葉子內校準 ＋ 每片葉子 × 版本的留存率與信心，每個實驗寫一頁 HTML，並在 summary.html 加入信心分數（需先 `run`） | `evaluation/`、`html/pages.py`、`html/compare.py` |
| `python -m feel_engine.cli compare` | 只重做 `summary.html`（`run`、`evaluate` 結束時已自動做） | `html/compare.py`、`html/pages.py` |
| `python -m feel_engine.variants new` | 實驗的變體：lifecycle／無，流失併入回流（同一棵樹）；補位條件：每棵樹都要有每個版本（該樹的葉子沒建議到的版本都在該樹內補一片，門檻內補不到可用可比區不足的位置），但比原建議留存率低超過 2 pp（max_loss_pp）的位置一律不用（2026-11-02 起取代原本的「每棵樹至少一片」，原 new2 併入 new）；另外寫一頁 `lifecycle_tag_none_new.html`，不進 summary.html、不覆蓋正式結果（需先 `run`、`evaluate` 才有對照的正式版） | `variants.py`、`evaluation/`、`modeling/model.py` |
| `python -m feel_engine.variants alltags --config my.yaml` | 實驗的變體：all，X = 快層 ＋ 標籤表所有有資料的類別標籤（8 個）；另外寫一頁 `all_tag_alltags.html`，頁首與實驗 1、4 比較 AUC；`--config` 可換訓練期間／驗證期間 | `variants.py` |
| `python -m feel_engine.cli --help`、`run --help` | 列出子指令與參數 | |

## 設定檔

| 檔案 | 改什麼 | 改完要跑 |
|---|---|---|
| `feel_engine/config/model.yaml` | 放進模型的特徵（`欄位: 中文名稱`，中文名稱用在樹圖與規則）、TAG_ 組合、實驗清單與順序、one-hot 類別限定 | `bash run_experiments.sh` |
| `feel_engine/config/config.yaml` 的 `tree`、`splits`、`leaf_eval` | 樹的深度、葉子筆數（`min_samples_leaf`）與切分門檻（`min_impurity_decrease`）、分群規則（value 只建模低／中／高）、葉子 × 版本門檻、傾向分數模型（`propensity_model`）與葉子內校準（`balance_calibration`） | `bash run_experiments.sh` |
| `feel_engine/config/config.yaml` 的 `data`、`labels`、`weighting`、`features`、`modules` | 訓練期間／驗證期間、session／Y 定義、權重上限、旗標門檻、`version_map` | `bash run_experiments.sh --build` |

## 結果在哪

用瀏覽器打開 HTML 即可（單檔、不需網路）：

| 路徑 | 內容 |
|---|---|
| `result/{run}/summary.html` | **實驗比較（AUC ＋ 信心）**：8 組實驗的 AUC（圖與表：95% 區間、與基準 all／無的差距）與信心分數表（合併後的葉子、信心分佈），附評分依據；點實驗名稱進入該實驗頁 |
| `result/{run}/{split}_tag_{tags}.html` | **每個實驗一頁**：每棵樹的決策樹（開啟時縮到一個畫面內，按鈕可放大到原始大字；葉子標最佳版本、留存率與信心）＋ 葉子 × 版本留存率熱圖（只標葉子編號）。分頁「原始葉子」把樣本不足的格子標灰；「合併無法比較的葉子」把無法比較的葉子併回父節點後重算；「合併相近建議」再把建議版本相同的相鄰葉子併成一片，規則最少；「五個體感」替沒被建議的版本在葉子內找子規則補位（SPEC 9.4，`config.yaml` 的 `five_feelings`） |
| `result/{run}/build/` | 資料檢查與 build 報告（`build_report.md`、`data_quality_report.md`，本 run 的日子並列） |
| `output/TCG_PHP_22071/dataset/` | 逐日 dataset：`{日期}.parquet`（決策點）、`{日期}.json`（指紋、逐日統計）、`builds/`（每次 build 讀了哪些日檔），所有 run 共用，可重跑產生 |
| `output/{run}/model/{split}/tag_{tags}/` | 程式讀取用：`tree_rules.json`（樹結構、葉子、設定）、`valid_predictions.parquet`（驗證期間預測）、`leaf_confidence.json`（信心）、`five_feelings.json`（補位結果） |
| `result/{run}/{split}_tag_{tags}_{變體}.html`、`output/{run}/variants/` | 實驗的變體（`python -m feel_engine.variants`）：頁面與程式讀取用的檔案，格式同上；`run --clean` 會清掉變體頁，重跑變體指令即可 |

## 檢查程式與資料

```bash
python -m pytest -q feel_engine/tests   # 單元測試（合成資料）
```

`test.ipynb`：dataset 長相與資料檢查（訓練期間、驗證期間一起檢查），包含不經過 build 程式、從注單快取獨立重算抽樣玩家的逐欄比對（kernel 用 `py311` 或 `player_label`）。

## 目錄

```
SPEC.md                 規格（唯一依據）
run_experiments.sh      一次跑完實驗
data_loading/           唯一連 DB 的程式，抓資料到 data/
data/                   原始快取（注單逐日檔、週標籤；資料夾不含日期，只補新的日子）
feel_engine/            程式與設定（依流程分資料夾）
  cli.py、paths.py        指令入口、路徑
  config/                 config.yaml、model.yaml
  pipeline/               建資料集（build）：data.py 分桶、build.py 狀態／Y／特徵、store.py 逐日 dataset、report.py build 報告
  modeling/               決策樹（run）：model.py
  evaluation/             葉子 × 版本評估（evaluate）：propensity.py、leaf_eval.py、merge.py、evaluate.py（主流程）
  html/                   HTML：pages.py（頁面元件）、compare.py（summary.html）
  tests/                  單元測試
output/TCG_PHP_22071/dataset/   逐日 dataset（所有 run 共用）
output/{run}/           各實驗的樹與驗證期間預測（model/）
result/{run}/           build 報告、summary.html、每個實驗的 HTML
test.ipynb              資料檢查
```

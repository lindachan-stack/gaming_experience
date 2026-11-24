# 體感模組決策樹：架構規格

> 輸入：`tableau.bet_records` 逐筆注單（含 `game_version`）＋ `dwt_player_game_tag_wi_test` 週標籤，皆用已落地的快取。
> 本文件只定義架構、資料流、特徵算法與驗收標準，不含程式碼。

---

## 0. 實作規則

- 依第 1–13 節順序實作，每完成一節就跑第 13 節對應測試。
- 可調參數放 `config.yaml`（第 3 節）；決策樹要放進 X 的欄位放 `model.yaml`（第 8 節）；0.1 的常數寫成程式中的具名常數。
- 只讀快取，本專案不新增連 DB 的程式。bet_records 以 polars 逐日檔讀取；標籤讀 `tag_loader` 落地的快取。
- 特徵只能用開獎前已知的資訊（6.1）。
- 逐手狀態與 Y 必須在完整逐手序列上算完，之後才能挑決策點、加權；之後只能篩選列或附加欄位，不得平移或重算（5.0）。
- 不計算模組賠率表，模組 RTP 只用設計值；不使用現有分派規則。
- 第 14 節的待確認事項先用預設值實作，並在報告中標示。
- Python 3.11、polars、scikit-learn、pyyaml、pyarrow；固定隨機種子。給人看的結果是單檔 HTML（不連外部資源）。

### 0.1 定義

| 項目 | 定義 |
|---|---|
| 時區 | 美東 `America/New_York`（自動處理夏令／冬令）；快取 `create_at` 為 UTC，讀入即轉換 |
| 玩家鍵／排序 | `member_account`；依 `member_account`、`create_at` 穩定排序（同秒保留檔內 `id` 順序） |
| 付費手 | `bet_points > 0` |
| 免費局／Jackpot | `bet_points = 0`，併入同玩家前一個付費手 |
| BUY 手 | `game_mode = 19`；與一般付費手一樣計算，另以 `SY_isBuyFeature` 標示 |
| 含 FreeGame 的手 | 付費手 `game_mode` 位元 2（Free）：3 = 一般手中 FreeGame、19 = BUY（必含）；是當手開獎結果，只用來算當手之前的 `GP_dailyRate_FREE_GAME` |
| 倍數區間（`prize / bet`） | NO_WIN：0；TINY：0–1；SMALL：1–7；MEDIUM：7–30；BIG：≥ 30（左閉右開） |
| 水池週期 | 日：美東 0 點重置（與遊戲端 gamePotSave 一致） |
| 決策點 | 每個 session 第 0、20、40… 手；Y 窗口 = 之後 20 手，不重疊 |
| 商戶／幣別／遊戲 | `config.yaml` 的 `data.merchant`、`currency`、`game`（目前 TCG、PHP、22071），決定快取與輸出的資料夾（`paths.py`）；只讀該幣別的注單 |
| 每份資料只存一份 | 注單與標籤快取的資料夾不含日期，延伸期間只補抓缺的日子；dataset 依 `session_date` 一天一檔，所有 run 共用，build 只建還沒有（或已過期）的日子；分桶是 build 中的暫存，結束即刪 |
| 訓練期間／驗證期間 | 時間外驗證：session 起點落在 `train_start`～`train_end` 的決策點訓練決策樹，`valid_start`～`valid_end`（須晚於 `train_end`，中間可以空幾天）的決策點評估 AUC 與葉子 × arm。同一玩家可同時出現在訓練與驗證，同一個 session 不會 |
| 決策點歸屬日 | session 起點的美東日期（`session_date`）。跨午夜的 session 整段歸起點那天，所以訓練與驗證交界的 session 歸訓練。午夜後決策點的日水池特徵仍依遊戲端 0 點重置 |
| 輸出路徑 `{run}` | `output／result` 底下的 `{商戶}_{幣別}_{遊戲}/train_{起}-{迄}_valid_{起}[-{迄}]/`（驗證只有一天時只寫一個日期），例如 `TCG_PHP_22071/train_20260928-20261011_valid_20261012-20261018/` |
| dataset 欄位順序 | 索引（`member_account`、`spin_ts`、`session_date`、`session_id`、`session_spin_idx`、`week_index`）→ 遊戲版本（`game_version`、`arm`、`arm_ratio`）→ 快層 `GP_` → 快層 `SY_` → 快層 `NGP_` → 慢層 `TAG_` → 分群 `SEG_` → Y／權重（`churnNext20`、`w_cap`）。`session_id` = `session_date` 的 yyyymmdd × 1000 ＋ 該玩家當天第幾個 session |

---

## 1. 目標與資料

### 1.1 目標

1. 由 bet_records 建立逐手序列，在決策點算出快層特徵（第 6 節），加入慢層標籤（第 7 節）。
2. 訓練流失風險決策樹，Y = 20 手內是否離場；依實驗清單（切分 × TAG_ 組合，共 8 組，第 2 節）各訓練一次（第 8 節）。
3. 每個葉子校正玩家狀態後比較各版本的留存率，選出模組（第 9、10 節）。
4. 離線評估新策略相對現況的留存，並比較各實驗（第 11 節）。

### 1.2 資料

| 資料 | 快取 | 範圍 |
|---|---|---|
| bet_records | `data/bet_records/bet_records_TCG_22071_tz+0/`（UTC 日檔） | UTC 2026-08-01 起，逐日累積 |
| 週標籤 | `data/tags/dwt_player_game_tag_wi_test_TCG_PHP_22071_lag1w_asof/`（每個套用週一檔） | 標籤自 202619（05-04 起），所有玩家 |

### 1.3 實測概況（2026-08-29）

| 項目 | 實測值 | 設計上的處理 |
|---|---|---|
| 規模 | 787 萬付費手、30,551 位玩家、41.4 萬個候選決策點 | 依玩家分桶處理 |
| 每人每日 | 付費手中位數 86、P99 2,453；決策點中位數 5、P99 125 | 用權重壓低重度玩家影響力，不刪資料（5.5） |
| `game_version` | 付費手 `22071_1.0.0_{0,1,2,3,5}`，佔比 `_2` 42%、`_0` 38%、`_5` 12%、`_1` 6%、`_3` 2% | 經 `version_map` 對應原版本 |
| 版本切換 | session 內每手 1.9% 換版本；20 手窗口有 22% 含多個版本 | 只抽主版本明確的決策點（5.4） |
| 版本依狀態分派 | 今日 RTP > 1.2 或同 session 前一手 BIG_WIN 時 `_2` 佔 60–68%；今日 RTP < 0.8 時 `_1` 佔 20%；今日前 20 手 `_0` 佔 75%；`_3` 依下注額分派 | 今日 RTP 與手數必須放入 X（6.2）；部分葉子只有單一版本（10.1） |
| 對不到前一週標籤 | 42% 玩家 | 取最新一筆標籤，完全沒有 → 新手（第 7 節） |
| 其他 | 相鄰手間隔中位數 2 秒；`refund` = `prize`、`valid_bet` = `bet_points`；`risk_pref_tag` 全空 | 不以出手速度排除玩家；不使用這三欄 |

---

## 2. 專案結構與命令

```
gaming_experience/
├── SPEC.md
├── README.md                         # 指令與結果位置速查
├── run_experiments.sh                # 一次跑完實驗（run --clean → evaluate）
├── db_manager.py                     # DB 連線
├── data_loading/                     # 抓資料：唯一連 DB 的程式，寫入 data/
│   ├── bet_records_loader.py
│   └── tag_loader.py
├── data/                             # 原始快取（刪了要重抓）
│   ├── bet_records/
│   ├── tags/
│   └── reference/                    # example_gamePotSave.json
├── feel_engine/                      # 程式與設定，不放產出
│   ├── cli.py                        # 指令入口（python -m feel_engine.cli）
│   ├── variants.py                   # 實驗的變體（python -m feel_engine.variants）
│   ├── paths.py                      # 所有路徑集中定義
│   ├── config/
│   │   ├── config.yaml
│   │   └── model.yaml                # 決策樹要放進 X 的欄位與實驗清單（第 8 節）
│   ├── pipeline/                     # 建資料集（build）
│   │   ├── data.py                   # 第一段：日檔 → 分桶（暫存）
│   │   ├── build.py                  # 第二段：逐手序列、狀態、Y、決策點、特徵 → 逐日 dataset
│   │   ├── store.py                  # 逐日 dataset：指紋、缺／過期的判斷、讀取
│   │   └── report.py                 # build 報告
│   ├── modeling/
│   │   └── model.py                  # 第 8 節：決策樹（run）
│   ├── evaluation/                   # 第 9 節：葉子 × arm 評估（evaluate）
│   │   ├── propensity.py             # 9.1：傾向分數、結果模型
│   │   ├── leaf_eval.py              # 9.1–9.2：共同範圍、葉子內校準、加權留存率、bootstrap、最高留存 arm
│   │   ├── merge.py                  # 合併無法比較的葉子、合併相近建議
│   │   ├── five.py                   # 9.4：五個體感補位
│   │   └── evaluate.py               # 主流程：每個實驗一頁 HTML
│   ├── html/
│   │   ├── pages.py                  # HTML 頁面（決策樹 SVG、留存率熱圖、AUC 圖）
│   │   └── compare.py                # 8.2：實驗比較 summary.html
│   └── tests/
├── output/{商戶}_{幣別}_{遊戲}/      # 中間產物（可重跑產生）
│   ├── dataset/                      # 逐日 dataset，所有 run 共用
│   │   ├── {日期}.parquet            # session_date = 該日的決策點
│   │   ├── {日期}.json               # 指紋、列數、產生它的 build、逐日統計
│   │   └── builds/{build_id}.json    # 一次 build：讀取範圍、R1–R7、用到的注單日檔與標籤週
│   ├── _build_tmp/                   # build 中的暫存（結束即刪）
│   └── train_…_valid_…/              # = output/{run}/
│       ├── model/{split}/tag_{tags}/ # tree_rules.json、valid_predictions.parquet、leaf_confidence.json、five_feelings*.json
│       ├── variants/{頁名}/          # 變體的同一組檔案
│       └── state_debug/              # build 時隨機 5 位玩家的完整逐手狀態
└── result/{run}/                     # 給人看的成果
    ├── build/                        # data_quality_report.md、build_report.md
    ├── summary.html                  # 實驗比較：AUC ＋ 信心（8.2）
    └── {split}_tag_{tags}.html       # 每個實驗一頁（第 9 節）；變體頁加後綴
```

```
python data_loading/bet_records_loader.py --start 2026-09-26 --end 2026-09-30   # 補抓注單（只抓缺的 UTC 日）
python data_loading/tag_loader.py --start 2026-05-11 --end 2026-09-30           # 補抓週標籤（只抓缺的注單週）
python -m feel_engine.cli build                             # 建本 run 還沒有或已過期的日子（--rebuild 全部重建）
bash run_experiments.sh                                     # run --clean → evaluate（--build 先 build）
python -m feel_engine.cli run                               # 跑 model.yaml 的 experiments 全部
python -m feel_engine.cli run --split lifecycle --tags value   # 只跑一組（可逗號分隔，跑所有組合）
python -m feel_engine.cli run --model-config my.yaml        # 換一份特徵與實驗設定檔
python -m feel_engine.cli evaluate                          # 第 9 節：每個實驗一頁 HTML
python -m feel_engine.cli compare                           # 只重做 summary.html（run、evaluate 結束時也會做）
python -m feel_engine.variants new                          # 實驗的變體
python -m feel_engine.variants alltags --config my.yaml     # 變體也可以換 config（例如另一個驗證期間）
```

- 所有實驗共用同一份逐日 dataset，結果可直接比較。
- 實驗 = split × tags，清單與順序在 `model.yaml` 的 `experiments`（實驗比較依此排序）：

| # | split | TAG_ | 目的 |
|---|---|---|---|
| 1 | all | 無 | 基準：不分群、只用快層 |
| 2 | all | lifecycle | 生命週期當特徵 |
| 3 | all | value | 價值當特徵 |
| 4 | all | lifecycle＋value | 生命週期與價值都當特徵（對照 #2、#3） |
| 5 | lifecycle | 無 | 依生命週期分群（對照 #2：分群 vs 當特徵） |
| 6 | lifecycle | value | 分群內再加價值（對照 #5） |
| 7 | value | 無 | 依價值分群（對照 #3） |
| 8 | value | lifecycle | 分群內再加生命週期（對照 #7） |

- 分群用的標籤不再放進同一個實驗的 X（群內為常數，放了也會被移除，只會影響沿用 all 的小群）。

**測試規則**：

- 訓練期間 2026-09-28～10-11，驗證期間只用下一週 10-12～10-18（一個訓練期間只驗證一週，不另跑其他驗證週），即 run `train_20260928-20261011_valid_20261012-20261018`。
- 只跑 `model.yaml` 的實驗（`bash run_experiments.sh`），不產生變體頁。每個實驗頁有五個分頁：原始葉子、合併無法比較的葉子、合併相近建議、五個體感1、五個體感2。
- 生命週期的流失併入回流（`config.yaml` 的 `regroup`，第 7 節）；lifecycle 分群為活躍、新手、一般、下滑、回流（含流失）五群。
- 不再使用的舊結果移到 `result/{商戶}_{幣別}_{遊戲}/delete/`、`output/{商戶}_{幣別}_{遊戲}/delete/`（只移動不刪除）。

**實驗的變體**（`feel_engine/variants.py`）：改分群的值（`SEG_` 改名）、加標籤表的其他類別欄位（規則同第 7 節）或覆蓋 `five_feelings` 設定，流程與正式實驗相同（傾向分數仍用 `model.yaml` 的標籤）。資料寫到 `output/{run}/variants/{頁名}/`、頁面 `result/{run}/{頁名}.html`（頁名 = `{split}_tag_{tags}[_{後綴}]`），不在 `model.yaml` 清單、不進 `summary.html`。頁首寫變更內容，並與指定的正式實驗在同一批決策點上比較 AUC（方法同 8.2）。`run --clean` 會清掉變體頁，重跑變體指令即可。

- `new`（`lifecycle_tag_none_new.html`）：lifecycle／無；補位改為每棵樹都要有每個版本（9.4 第 4b–4d 點）；對照實驗 5。流失併入回流已是預設，所以目前與實驗 5 的「五個體感2」分頁相同。
- `alltags`（`all_tag_alltags.html`）：all，X = 快層 ＋ 標籤表所有有資料的類別標籤（生命週期、價值、總下注額、平均下注額、下注手數、資金水位、購買特色、下注強度；價值只取低／中／高，其他每個類別都 one-hot）；對照實驗 1、4。

---

## 3. `config.yaml`

設定值與逐項說明以 `feel_engine/config/config.yaml` 為準，這裡只列各區塊的用途：

| 區塊 | 內容 | 相關章節 | 改了要重跑 |
|---|---|---|---|
| `seed` | 隨機種子 | — | run |
| `data` | 商戶／幣別／遊戲、訓練期間、驗證期間、`pre_hours`、`post_hours` | 0.1、5.1、5.3 | build |
| `compute` | 分桶數、worker 數（只影響速度與記憶體） | 5.0 | — |
| `modules` | `version_map`、模組設計值、基準、可上線模組 | 第 4、10 節 | build |
| `labels` | session 間隔、Y 窗口、回訪時間、`arm_dominance` | 5.3、5.4 | build |
| `weighting` | `player_day_cap` | 5.5 | build |
| `features` | 旗標門檻 `derived`（`GP_flag_dryLong`、`NGP_flag_coldLoss`） | 6.3 | build |
| `splits` | 一棵樹的訓練樣本下限、指定建模的群 | 8.1 | run |
| `regroup` | 讀入 dataset 時併類別（目前流失併入回流）；`TAG_` 與對應的 `SEG_` 一起改 | 7、8.1 | run |
| `tree` | `max_depth`、`min_samples_leaf`、`min_impurity_decrease` | 第 8 節 | run |
| `leaf_eval` | 共同範圍、校準、平衡與合併的門檻、bootstrap 次數 | 9.1、9.2 | evaluate |
| `five_feelings`、`five_feelings_2` | 五個體感補位（後者覆蓋前者，用在「五個體感2」分頁） | 9.4 | evaluate |

`config.yaml` 須滿足 `train_start` ≤ `train_end` < `valid_start` ≤ `valid_end`。

---

## 4. 模組

| 代碼 | 名稱 | 波動度 | 倍率 | RTP | 資料來源（原版本） | 目的（只供解讀） |
|---|---|---|---|---|---|---|
| M1 | 保護傘 | 中 | 中 | 高 | 保護傘 | 玩家撐得住 |
| M2 | 標準 | 中 | 中 | 中 | 原模組（基準） | — |
| M3 | 冷卻 | 中 | 中 | 低 | 無 | 剛中大獎、RTP 過高 |
| M4 | 驚喜 | 中 | 高 | 高 | 高倍數 | 回流、挽回、長期無大獎 |
| M5 | 高倍數 | 高 | 高 | 中 | 高倍數 | 大額下注 |
| M6 | 高波動 | 高 | 中 | 中 | 深模組 | 活躍、偏好風險 |
| M7 | 極限 | 高 | 高 | 低 | 無 | 高風險、高價值 |

- 離線資料只觀察得到原版本，因此評估單位 arm = 原版本（保護傘、原模組、高倍數、深模組），由 `game_version` 經 `version_map` 對應。
- 原版本與對應模組視為同一設計；高倍數同時對應 M4、M5，兩者共用 `uplift`、並列。
- M3、M7 沒有資料，列為 `untested_modules`。arm 不放入 X。
- `version_map` 未填完時：build 加 `--provisional-arm` 以 `game_version` 暫代 arm（暫代模式），第 9 節只輸出最高留存 arm、不換算 M1–M7。

---

## 5. 計算流程

### 5.0 處理順序

```
build：本 run 的日子中，逐日 dataset 還沒有（或已過期）的，依連續區段分組；每段：

第一段 pipeline/data.py（逐日檔）
  R7 → 讀 9 欄 → R1–R4 → 轉美東 → 依玩家雜湊寫入分桶（暫存）

第二段 pipeline/build.py（逐桶，可平行）
  ① 逐手序列              5.1
  ② 逐手狀態（水池）      5.2  ┐ 在完整序列上計算
  ③ session 與 Y          5.3  ┘
  ④ 抽決策點              5.4
  ⑤ 附加權重 w_cap        5.5
  ⑥ 計算特徵              第 6 節
  → 附加標籤（第 7 節）→ 依 session_date 一天一檔寫入逐日 dataset（store.py）→ 刪除暫存
```

錯誤做法：先挑決策點，再對剩下的列往後平移算 Y。此時「往後 20 列」是之後 20 個決策點，標籤會錯且不會報錯。

一天的決策點只由該日與前後固定的範圍決定（截尾以起點日為準、`session_id` 含日期），與一次 build 幾天無關，所以不同 run 可以共用同一天。

某天在下列情況視為過期、下次 build 自動重建；run／evaluate 遇到缺或過期的日子會停下、請先 build：

- build 相關設定（`data.pre_hours`、`post_hours`、`labels`、`weighting`、`features`、`modules.version_map`）或 `pipeline/data.py`、`build.py` 的內容改了（含註解）。
- 用到的注單日檔或標籤週重抓過（列數或最後 ETL 時間不同）。
- 建置時注單快取尾端不足（R7 後段不足，該日有 session 因看不完 Y 被截掉），而現在的快取已經補上。快取沒有變長時重建結果相同，不算過期。

### 5.1 讀取與逐手序列

- 讀 9 欄：`id`、`member_account`、`create_at`、`bet_points`、`prize`、`points`、`last_points`、`game_version`、`game_mode`。`id` 只用於去重。
- build 一次讀完一段連續的日子（訓練期間與驗證期間中間空了幾天 → 兩段各讀一次，中間不讀）。
- 讀取範圍（美東）：區段第一天 00:00 − `pre_hours` 至最後一天 24:00 ＋ `post_hours` ＋ `return_window_hours`，換算 UTC 決定日檔。
- 範圍前檢查：讀取範圍起點前 `session_gap_minutes` 內，每位玩家最後一個付費手的時間，供 5.4 條件 4 使用。
- 分桶：玩家雜湊 → 桶號，只為平行處理。
- 每桶排序後，`bet_points = 0` 的列併入前一個付費手：`prize` 相加，出場 balance 取最後一列 `last_points`。

| 付費手欄位 | 算法 | 開獎前可知 |
|---|---|---|
| `spin_ts` | `create_at`（美東） | 是 |
| `bet` | `bet_points` | 是 |
| `balance_before` | `points` | 是 |
| `arm` | `game_version` 對應的原版本 | 是 |
| `is_buy` | `game_mode` 位元 16（付費手實測只有 1、3、19，等同 `game_mode = 19`） | 是 |
| `is_free` | `game_mode` 位元 2（Free） | 否（BUY 手除外） |
| `prize`、`outcome` | 合併後總贏分、倍數區間 | 否 |
| `balance_after` | 合併後最後一列 `last_points` | 否 |

### 5.2 逐手狀態

第 i 手的狀態只累計同一週期內、第 i − 1 手以前的付費手。

| 週期 | 分組 | 累計內容 |
|---|---|---|
| 日 | 玩家 × 美東日期 | 手數、各倍數區間次數、含 FreeGame 的手數、Σbet、Σprize |
| session | 玩家 × session | 序號、第一手的 `balance_before` |

### 5.3 Session 與 Y

- 與同玩家前一個付費手間隔 > 30 分鐘即切 session。
- 因決策間隔 = 窗口 = 20 手，只有 session 的最後一個決策點，窗口會含 session 結束：
  - `churnNext20` = 1：最後一個決策點，且下一個 session 晚於本 session 結束 + 24 小時（或沒有下一個）。
  - 其他 = 0；`retained` = 1 − `churnNext20`（評估時計算，dataset 不另存）。
- 截尾：session 在起點隔天 0 點 ＋ `post_hours` 之後才結束，或結束 + 24 小時超過快取終點（看不完 Y）時，整個 session 的決策點都不抽入（只刪最後一個會讓流失率偏低）。每天都用同一條規則。

### 5.4 抽決策點

只抽同時符合以下條件的決策點：

1. `session_spin_idx % 20 == 0`。
2. session 起點的美東日期（`session_date`）落在本次 build 的日子。
3. 未截尾。
4. session 起點完整：玩家在讀取範圍內第一手與範圍起點前最後一個付費手的間隔 > `session_gap_minutes`；否則範圍內第一個 session 的起點是被讀取範圍切出來的（`session_spin_idx`、`SY_first_balance` 會錯），整個 session 不抽。跨午夜的 session 不受此條件影響。
5. 窗口內主版本（手數最多的 arm）佔比 `arm_ratio` ≥ `arm_dominance`；樣本的 `arm` = 主版本。

報告列出：條件 4 排除的 session 數；符合條件 5 的比例，以及符合與不符合者的流失率、今日 RTP 比較。

### 5.5 玩家 × 日權重

- 全部決策點保留。每位玩家當天（歸屬日）有 n 個決策點，每筆權重 `w_cap` = min(1, `player_day_cap` ÷ n)。
- 一般玩家權重 1；重度玩家資料全留，但一天的總影響力最多等於 `player_day_cap` 筆。
- `w_cap` 用於決策樹、傾向分數、葉子評估與離線評估。

---

## 6. 快層特徵（`GP_`／`SY_`／`NGP_`）

### 6.1 原則

- 在決策點「下注後、開獎前」計算：可用 5.2 的狀態與當手 `bet`、`balance_before`、`is_buy`；不可用當手 `prize`、`balance_after`、`is_free`。
- dataset 的快層欄位 = `model.yaml` `fast` 列出的欄位，含註解掉、暫不放進 X 的（目前 `GP_flag_dryLong`、`NGP_flag_coldLoss`）；`model.yaml` 沒列的不產生。
- 三種前綴：`GP_` = 水池類，只用當手之前的手；`SY_` = 系統端狀態（當手下注額、是否 BUY、下注前餘額、session 起始餘額）；`NGP_` = 水池與系統端組合的衍生特徵。
- 水池特徵只用日週期的 count 與 mean；不使用前一個月。
- 除以 0 → 缺值；建模時依 `model.yaml` 的 `missing` 填補（第 8 節）。

### 6.2 日水池（8 項）

| 特徵 | 算法 |
|---|---|
| `GP_dailySpinCount` | 今日手數 |
| `GP_dailyRate_{NO_WIN, TINY_WIN, SMALL_WIN, MEDIUM_WIN, BIG_WIN}` | 今日各區間次數 ÷ 今日手數 |
| `GP_dailyRtp` | 今日 Σprize ÷ Σbet（現行分派依此切換版本，必須放入 X） |
| `GP_dailyRate_FREE_GAME` | 今日含 FreeGame 的手數 ÷ 今日手數（不含當手；BUY 手也算） |

### 6.3 交互作用與多條件（4 項）

| 特徵 | 算法 |
|---|---|
| `GP_dailyRate_MEDIUM_PLUS` | MEDIUM ＋ BIG 比率 |
| `GP_dailyNoWin_x_spins` | NO_WIN 比率 × log1p(今日手數) |
| `GP_flag_dryLong` | 今日手數 ≥ `dry_min_spins` 且 MEDIUM_PLUS = 0 → 1 |
| `NGP_flag_coldLoss` | NO_WIN 比率 ≥ `cold_nowin_rate` 且 `SY_balance` ≤ `SY_first_balance` × (1 − `loss_ratio`) → 1 |

旗標任一條件為缺值時記為 0。這些是特徵，不是分派規則。

### 6.4 非水池

| 特徵 | 算法 |
|---|---|
| `SY_bet` | 當手下注金額 |
| `SY_isBuyFeature` | 當手是否購買特色 |
| `SY_balance` | 當手下注前的 balance |
| `SY_balanceToBet` | `SY_balance` ÷ `SY_bet`（以當手下注額還能玩幾手） |
| `SY_first_balance` | 本 session 第一手下注前的 balance（整個 session 固定） |
| `NGP_dailyTotalWinToBet` | 今日到當手之前的 Σprize（遊戲端 gamePotSave `totalWin` 的日版本）÷ `SY_bet` |

---

## 7. 慢層標籤

| `model.yaml` tags | 加入 X 的欄位 |
|---|---|
| `none` | 無 |
| `lifecycle` | `TAG_lifecycle`：一般、新手、下滑、活躍、回流（流失併入回流） |
| `value` | `TAG_value_combined`：本週無下注、低、中、高、手數過低、無紀錄（one-hot 只取低／中／高） |
| `lifecycle_value` | 以上兩欄 |

- 決策點所在美東週為 W，取該玩家 `week_index` < W 的**最新一筆**（含補零列）。
- 完全沒有紀錄 → `TAG_lifecycle` = 新手、`TAG_value_combined` = 無紀錄。
- 流失併入回流：dataset 與 build 報告保留標籤表原本的六類；run、evaluate、變體讀入 dataset 時依 `config.yaml` 的 `regroup` 把 `TAG_lifecycle` 的流失改成回流，`SEG_lifecycle` 一起改。X、傾向分數、lifecycle 分群都只看到五類。改 `regroup` 不用重新 build，要重跑 run。
- 只讀 `member_account`、`week_index`、`is_zero_fill`、`lifecycle_tag`、`value_combined_tag`。
- 寫入 dataset 的欄位：`TAG_lifecycle`、`TAG_value_combined`（X），以及 `TAG_source`（W-1／更早／無紀錄）、`TAG_week_index`（取到的那筆標籤的週；決策週 W 本身存為索引 `week_index`）、`TAG_is_zero_fill`。
- 報告列出：取到 W − 1 週、更早週、無紀錄的人數與流失率。

---

## 8. 流失風險決策樹

- 訓練期間訓練、驗證期間預測；`DecisionTreeClassifier`，不用 `class_weight`，`sample_weight` = `w_cap`，Y = `churnNext20`。超參數在 `config.yaml` 的 `tree`（all 樹與分群樹共用）；`min_impurity_decrease` 讓改善太小的切分不切。
- X 由 `model.yaml` 決定：`fast`（每個實驗都放）＋ `tags[實驗的 tags]`（不累加）。只能放 `GP_`／`SY_`／`NGP_`／`TAG_` 開頭的欄位；`arm`、`arm_ratio`（用到之後 20 手）、`SEG_`、Y、`w_cap`、索引放入會報錯。每個欄位寫成「欄位: 中文名稱」，中文名稱用在樹圖與規則。
- 編碼：數值缺值填 `missing.fill`（−1），`missing.add_isna` 為 true 時另加 `<欄位>_isna`；文字欄位依訓練資料的類別 one-hot（驗證期間未見過的類別 → 全 0）；訓練資料中為常數的欄移除。
- `model.yaml` 的 `categories` 限定文字欄位只對哪些類別做 one-hot，其他類別全為 0、不會單獨成為分支條件。目前 `TAG_value_combined: [低, 中, 高]`。
- 輸出 `output/{run}/model/{split}/tag_{tags}/`（供程式讀取）：`tree_rules.json`（節點的左右子節點、提問、訓練期間筆數與加權流失率；葉子規則；X 欄位、超參數、model.yaml 路徑與指紋、run 當時的 dataset 簽章與 `regroup`）、`valid_predictions.parquet`（驗證期間每筆的群、樹、葉子、預測流失機率）。給人看的決策樹畫在該實驗的 HTML（第 9 節）。
- `run --clean`（`run_experiments.sh` 預設）先清掉舊的實驗結果，避免殘留舊設定或已移出清單的實驗。
- 樹圖的提問：數值「X ≤ t？」（左 = 是；整數欄位寫成整數）；門檻落在缺值填補值與最小觀測值之間 →「X 缺值？」；門檻落在最小與次小觀測值之間 →「X = 最小值？」；左分支含缺值時註明「含缺值」；one-hot、0／1 欄位 →「X = 類別？」「X = 1？」（左 = 否）。
- 葉子規則：同一欄位同方向只留最嚴格的門檻；數值欄位兩個方向都有時寫成區間「a < x ≤ b」。

### 8.1 三種切分

| split | 群 | 做法 |
|---|---|---|
| `all` | 1 群 | 一棵樹 |
| `lifecycle` | `SEG_lifecycle`（= `TAG_lifecycle`）五類：一般、新手、下滑、活躍、回流（流失併入回流，第 7 節） | 每群各自訓練樹、評估葉子 |
| `value` | `SEG_value`（= `TAG_value_combined`）只建模 `splits.groups.value` 列出的低、中、高 | 三群各自訓練樹、評估葉子；無紀錄、本週無下注、手數過低不建模、不評估 |

- 分群只用決策週 W 之前的標籤，不得用當日下注分群（上線時決策前就要知道）。
- 分群欄位在群內為常數，對應的 `TAG_` 特徵自動從 X 移除。
- 群的訓練期間樣本 < `min_segment_train_n` → 不訓練，沿用 `all` 的樹，仍納入驗證期間評估（`splits.groups` 列出的群也一樣）。all 樹本身也要達到這個門檻。
- 自建樹的群在驗證期間沒有決策點 → 該群不預測，其他群照常。
- 葉子編號加群名（例如 `新手/L3`）。

### 8.2 實驗比較（AUC ＋ 信心）

`run` 結束時（或 `compare`）讀所有已跑過的 `valid_predictions.parquet`，依 `model.yaml` 的 `experiments` 排序（清單外的附在最後並標示），輸出 `result/{run}/summary.html`；`evaluate` 結束時再重做一次，加入信心分數。內容依序為總覽表、AUC（圖與表）、信心（表與評分依據）。不放留存率（避免被誤用）。

- 共同樣本：每個實驗都有預測的驗證期間決策點（value 只建模低／中／高 → 共同樣本 = 這三群），AUC 與信心都在同一批決策點上計算。
- **AUC**：以 `w_cap` 加權，只看分不分得出誰會離開。95% 區間與「與基準 all／無的差」用玩家層級 Poisson(1) bootstrap `bootstrap_reps` 次，所有實驗同一組重抽權重；差的 95% 區間不含 0 才寫「較佳／較差」，否則「無顯著差異」。
- **信心分數（0–100）** = 100 × (高佔比 ＋ 0.5 × 中佔比)。佔比 = 共同樣本中落在該信心等級葉子的決策點比例（`w_cap` 加權），維持現狀的葉子不計入；代表「可以選版本的決策點中，多少能拿到可信的版本建議」。以合併無法比較後的葉子計算（9.2）；`leaf_confidence.json` 比預測舊時顯示「尚未 evaluate」；全部都是維持現狀時不給分。
- 給分依據（信心等級的定義見 9.2）：

| 信心 | 給分 | 理由 |
|---|---|---|
| 高 | 1 | 最佳版本平衡、且顯著高於次佳：建議可直接採用 |
| 中 | 0.5 | 最佳版本平衡，但和次佳沒有顯著差距：建議可用但價值有限 |
| 低 | 0 | 最佳版本未平衡（差距可能來自玩家狀態）或共同範圍不到葉子的 `min_overlap_share` |
| 無法比較 | 0 | 共同範圍內樣本足夠的版本不到兩個 |
| 維持現狀 | 不計入 | 同一版本佔葉子超過 `dominant_share`：沒有別的版本可比，不算信心不足 |

- AUC 與信心衡量不同的事：AUC = 分不分得出誰會離開；信心 = 版本建議可不可信。兩者都高的實驗最值得採用。

---

## 9. 葉子 × arm 評估

只用驗證期間，排除 arm 不明（R4）的樣本。evaluate 只讀驗證期間的日子。

**開始前的檢查**：run 把當時的 dataset 簽章（每天的 build 與列數）與 `regroup` 記進 `tree_rules.json`。evaluate 逐個實驗比對：簽章與現在不同（run 之後有日子重建過）、`regroup` 與現在的 `config.yaml` 不同，或該實驗的 X 欄位、缺值填補、one-hot 類別與現在的 `model.yaml` 不同 → 全部不評估，請先重跑 run。另外，任何實驗的驗證期間預測只要有一筆在目前的 dataset 找不到就中止（不靜默丟列）。

### 9.1 傾向分數、共同範圍與葉子內校準

只用資料，不用遊戲端的分派規則。

- **傾向分數** p(arm | x)：只配適一次、所有實驗共用；x = `model.yaml` 的快層特徵 ＋ 所有 `TAG_` 標籤（含分群用的標籤）；依玩家分 5 折交叉配適（每筆的機率來自沒看過這位玩家的模型）；`sample_weight` = `w_cap`。
- 模型（`leaf_eval.propensity_model`）：`logistic` = 多項式邏輯迴歸（預設；特徵截斷頭尾 1%、非負取 `log1p`、標準化，0／1 欄位不截斷、不取 log；數值缺值填中位數並加 `_isna`）；`gbm` = 梯度提升樹（數值保留原值與缺值）。實測 gbm 的權重較極端、有效樣本較少、信心較差，所以預設 logistic。
- **結果模型**：同樣只配適一次、依玩家 5 折交叉配適；每個 arm 一個邏輯迴歸（只用拿到該 arm 的決策點），用同一組共變數預測「拿到該 arm 時」的留存率 m_arm(x)。
- **共同範圍**：每片葉子挑要比較的 arm 組合 C；共同範圍 S = 葉子內「C 的每個 arm 傾向分數都 ≥ `overlap_min`」的決策點；C 的每個 arm 在 S 內都要 ≥ `min_n_per_arm` 筆。S 涵蓋葉子 ≥ `min_overlap_share` 的組合中挑 arm 最多的（同數量挑涵蓋最多）；都做不到 → 挑涵蓋最多的（信心最多「低」）；沒有任何兩個 arm 可行 → 不比較。目的：現行規則幾乎只給某個版本的狀態不拿來比，避免把「誰拿到」當成「版本效果」。
- 權重：S 內拿到比較 arm 的決策點 W = `w_cap` ÷ p(實際 arm)。
- **葉子內校準**（`leaf_eval.balance_calibration`）：傾向分數是對全體配適的，不保證每片葉子內平衡。S 內每個比較 arm（樣本 ≥ 20）以 W 為起點，找最接近 W 的權重（近似 entropy balancing），讓該 arm 每個共變數的加權平均與 S 相差 ≤ `balance_tolerance` 個標準差；比原本的 W 更平衡才採用。某 arm 根本沒有某種狀態的玩家時對不齊，平衡檢查照實判未平衡。代價是權重變得不均、有效樣本（Kish：(ΣW)² ÷ ΣW²）變少、區間變寬。

### 9.2 每個葉子 × arm

| 欄位 | 算法 |
|---|---|
| `n`、原始留存率 | 共同範圍內的樣本數；整片葉子的 `w_cap` 加權留存率 |
| 加權留存率（DR） | S 內 m_arm 的 `w_cap` 加權平均 ＋ S 內拿到該 arm 者 (y − m_arm) 的校準權重平均（y = 1 − churnNext20） |
| `max_smd` | 加權後該 arm 與共同範圍 S 各共變數標準化平均差的最大值 |
| 樣本足夠／平衡 | 納入比較（arm ∈ C）；`max_smd` ≤ `smd_threshold` |
| `uplift`、`uplift_ci`（第 10 節用） | (基準流失率 − 該 arm 流失率) × 100；玩家層級 bootstrap 95% 區間。基準 arm 不合格時改用 `n` 最大的合格 arm 作臨時基準並標示 |

- **最高留存 arm**：每片葉子在比較 arm（C）中挑加權留存率最高者，與次高、與臨時基準（`n` 最大者）比較差距與 bootstrap 95% 區間。
- **信心**（平衡只看最高留存 arm）：高 = 平衡且顯著高於次高；中 = 平衡但不顯著；低 = 未平衡，或共同範圍不到葉子的 `min_overlap_share`；無法比較 = 比較 arm 不到兩個（不標最高留存 arm）；維持現狀 = 比較 arm 不到兩個，且最多的 arm 佔葉子決策點超過 `dominant_share`（沿用該版本）。未平衡時差距可能來自玩家狀態，不能當版本效果。
- **合併無法比較的葉子**：由下往上，某個節點的子節點中有無法比較的葉子 → 把該節點底下全部合併成一片葉子（名稱 = 涵蓋的原葉子編號，例如 `高/L3–L6`），重挑共同範圍並重算；重複到每片葉子都可比較，或已合併到根。維持現狀的葉子不合併。判斷只用各 arm 的樣本數與傾向分數（不看結果）。
- **合併相近建議**：在合併無法比較之後，由下往上，同一個上層節點底下的兩片葉子建議的版本相同（高／中／低 → 最高留存 arm；維持現狀 → 最多的 arm；無法比較不併）→ 合併、可一路往上。建議不同但兩片都不是高信心 → 只有在這棵樹併完相同建議後仍超過 `similar_merge_over_leaves` 片時才合併。合併後重算，再依新結果合併，直到不再變化（最多 10 輪）。
- 兩種合併都只影響版本建議，決策樹與 AUC 不變。
- bootstrap：先聚合成「玩家 × 葉子 × arm」的加總（稀疏矩陣），每次給每位玩家一個 Poisson(1) 權重，以矩陣運算算完（W 固定）。
- 輸出：每個實驗一頁 `result/{run}/{split}_tag_{tags}.html`，分頁「原始葉子」「合併無法比較的葉子」「合併相近建議」，`five_feelings.enabled` 時再加五個體感分頁（9.4）；多棵樹的實驗上方有「切換決策樹」按鈕。每棵樹：
  - 決策樹（SVG，開啟時縮到一個畫面內，可縮放）：內部節點寫提問與訓練期間筆數、流失率，線上標「是／否」；葉子寫編號、最佳 arm（以 arm 顏色填色）、信心，無法比較為灰色；維持現狀寫葉子的整體留存率。
  - 葉子 × arm 留存率熱圖：列 = 葉子編號（不寫規則），另列驗證決策點、驗證玩家數、整體留存；欄 = arm，格內加權留存率與共同範圍內的「玩家數／決策點筆數」，★ = 最高，樣本不足或未納入比較為灰色只標 n；最後兩欄 = 最佳 arm、信心；滑鼠停留顯示 95% 區間、有效樣本與明細。
- 另寫 `leaf_confidence.json`（每片葉子的信心與合併對照，供 summary.html 算信心分數）。

### 9.3 驗收範例

一個葉子分兩個子狀態，以子狀態內 arm 佔比為傾向分數：

| 子狀態 | 佔葉子 | 保護傘（n，流失率） | 原模組 | 深模組 | 高倍數 |
|---|---|---|---|---|---|
| A | 40% | 1,600，32% | 100，40% | 100，38% | 2,200，30% |
| B | 60% | 400，20% | 400，28% | 700，24% | 4,500，22% |

預期（±0.1 個百分點）：保護傘 raw 29.6% / ipw 24.8%；原模組 30.4% / 32.8%；深模組 25.75% / 29.6%。
以原模組為基準：M1 `uplift` = 8.0、M6 `uplift` = 3.2。
DR 檢查（同一份資料）：結果模型給錯（常數）但傾向分數正確，或傾向分數給錯（均勻）但結果模型正確，加權留存率都等於上面的 ipw（雙重穩健）。

### 9.4 五個體感補位

目標：每個體感都至少是一條規則的建議，且整體留存最高（`config.yaml` 的 `five_feelings`；程式 `evaluation/five.py`）。

1. **有用到的版本**：以「合併相近建議」為準；高／中／低 → 最高留存 arm，維持現狀_X → X，無法比較不算；以整個實驗計（分群實驗的多棵樹合起來）。沒用到的版本 v 才補位。
2. **候選**：每片有建議（r）的葉子，可比區 O = 「v 與 r 的傾向分數都 ≥ `overlap_min`」。葉子內用深度 ≤ `sub_rule_depth` 的小樹描述 O（X = 該實驗決策樹的特徵與編碼，y = 是否在 O，`min_samples_leaf` = `min_n_per_arm`，`sample_weight` = `w_cap`；不看留存）。小樹每個節點（含根 = 整片葉子）中，v、r 在「節點 ∩ O」內各 ≥ `min_n_per_arm` 者，取 O 佔比最高（同佔比取筆數多）的當子規則；沒有 → 沒有可比區。
3. **比較**：子規則內，共同範圍 = O、比較組合固定 {v, r}，沿用 9.1–9.2 的權重、校準、DR 加權留存率、`max_smd` 與 bootstrap。差距 = r − v（百分點）；整體影響 = 子規則佔全部驗證決策點的比例（`w_cap` 加權）× 差距。
4. **挑位置**：差距點估計 ≤ `max_loss_pp`、且可比較比例（子規則內 O 的佔比）≥ `min_comparable_share` 的候選中，每個 v 一條（實驗的決策樹超過 `many_trees` 棵時 `placements_many_trees` 條、分在不同的樹）、每片葉子最多一個版本，以整數規劃挑選：先求補到的片數最多，再求整體影響總和最小。可比較比例太低時差距多半是選擇偏差，所以不採用。沒補滿時寫原因：沒有可比區／可比較比例都不到門檻／最小差距 x pp／可行的葉子已分給其他版本。
   - 4a. **每棵樹至少補幾片**（`min_per_tree` > 0）：目標依序為「每棵樹補到 `min_per_tree` 片」→「每個 v 補到上面的片數」→ 片數最少 → 整體影響最小。版本片數不設上限，每個（v, 樹）仍 ≤ 1 片、每片葉子 ≤ 1 個版本，門檻不變。補不到的樹寫原因，也寫進 `five_feelings.json` 的 `unfilled_trees`。
   - 4b. **一定要補到**（`force_coverage`）：門檻內補不到時，改用門檻外、有比出差距的候選；目標在「每個 v 補到片數」之後加「門檻外的片數最少」。選中的門檻外候選標「超過門檻」並寫超過哪些門檻（`five_feelings.json` 的 `outside`）。仍補不到只會是沒有可比區或葉子已分給其他版本。
   - 4c. **每棵樹都有每個版本**（`every_version_per_tree`）：改以每棵樹計，該樹的葉子沒建議到的版本都要在該樹的葉子內補一片。一片葉子可補多個版本：依序補，後補的在扣掉前面子規則後的其餘部分重新挖子規則、比較（r 不變）。每棵樹分開窮舉每個 v 補在哪片葉子與先後，依序取：補到的 v 最多 → 門檻外的片數最少 → 整體影響最小。不看 `placements_many_trees`、`min_per_tree`。`five_feelings.json` 加 `need`（每棵樹要補的版本），`unplaced` 每筆帶 `tree`，後補的 `placed` 帶 `after`（先補的版本）。
   - 4d. **4c 的差距上限**：差距點估計一律 ≤ `max_loss_pp`；`force_coverage` 只放寬可比較比例。只剩差距超過的候選時寫「最小差距 x.xx pp > 上限」。
5. **輸出**：正式實驗頁兩個分頁：「五個體感1」= `five_feelings`（每個版本在整個實驗至少出現一次）；「五個體感2」= `five_feelings` 再覆蓋 `five_feelings_2`（每棵樹都要有每個版本）。變體頁一個「五個體感」分頁，用變體自己的設定。每個分頁：樹圖在被選中的葉子下接子規則（新增的葉子標驗證期間筆數）、葉子 × 版本留存率表、補位結果表、其他候選（收合）。結果另寫 `five_feelings.json`（五個體感2 為 `five_feelings_2.json`）。`summary.html`、`leaf_confidence.json`、前三個分頁不受影響。
6. **限制**：子規則的位置與挑選都在驗證期間，挑損失最小的候選會讓損失偏樂觀；子規則內 O 以外的決策點也給 v，但沒有證據（看可比較比例）。要在更多狀態評估少用的版本，需要遊戲端小比例隨機發放。

---

## 10. 策略表

尚未實作（需先填 `version_map` 才能換算 M1–M7）。

### 10.1 每個葉子的排序

1. 合格 arm 展開成模組（同一原版本對應的模組共用 `uplift`）；只保留可上線模組。目前不考慮 RTP 成本，只看留存。
2. 顯著 = `uplift_ci` 下限 > 0；M2 視為顯著、`uplift` = 0。
3. 排序：顯著模組依 `uplift` → M2 → 不顯著模組依 `uplift`；同一原版本對應的模組（M4、M5）並列。
4. 信心：高 = 第一名顯著且 `max_smd` 合格；中 = 第一名不顯著但樣本足夠；低 = 第一名未平衡、共同範圍不到 `min_overlap_share`，或用臨時基準。
5. 沒有資料的模組列為 `untested_modules`。
6. 單一版本葉子（合格 arm 只有一個）：不計 `uplift`，維持該 arm 對應模組，信心 = 低，標示 `single_arm`；報告列出這類葉子的數量與樣本佔比。

### 10.2 `choose_module(row, policy)`

1. 依 split 找出所屬的群（沿用 all 的群用 all 的策略），再找葉子。
2. 取該葉子排序清單的第一個模組。目前不考慮成本，不設 RTP 護欄。

輸出（每個 split）：`leaf_report.csv`、`policy.json`（群、葉子條件、排序清單、信心、`untested_modules`）。

---

## 11. 離線評估

尚未實作。

### 11.1 每個 split

用驗證期間：

1. π(x) = `choose_module`，換回 arm（M4、M5 → 高倍數）。
2. 預期留存（SNIPS）：V = Σ[1(arm = π) × W × retained] ÷ Σ[1(arm = π) × W]。
3. 與現況（以 `w_cap` 加權的平均 `retained`）比較：整體與每個葉子的 V、差距、bootstrap 95% 區間、覆蓋率、ESS = (ΣW)² ÷ ΣW²；ESS < `min_n_per_arm` 標示「評估不可靠」。
4. 另列：落在 `untested_modules` 的佔比、每 4 小時的差距。
5. 依價值、生命週期各分組列一次；另輸出下注額加權的留存（權重再乘當手 `SY_bet`）。

結果加進該實驗的 HTML：各葉子選定模組、離線評估、單一版本葉子。

### 11.2 實驗比較（策略）

輸出加進 `result/{run}/summary.html` 或另一頁（實驗與順序同 8.2；實作時再與使用者確認呈現內容）：

| 表 | 內容 |
|---|---|
| 整體 | 各實驗的 V、差距、95% 區間、ESS、覆蓋率、`untested_modules` 佔比 |
| 差距 | 基準 all／無、同資訊 all、同 split 無標籤（同一次重抽計算） |
| 依群 | 每個生命週期群、價值群在各實驗下的 V |
| 下注額加權 | 以上三表的加權版 |
| 複雜度 | 群數、葉子總數、沿用 all 的群 |

只有差距區間不含 0 才寫「較佳」，否則寫「無顯著差異，建議用 all」。

---

## 12. 資料檢查與成本

| 代碼 | 檢查 | 判定 |
|---|---|---|
| R1 | manifest 對帳（只看 `data.currency` 的幣別） | 不符則中止 |
| R2 | 日檔內 `id` 重複 | 去重 |
| R3 | `last_points = points − bet_points + prize` 比例 | 只報告 |
| R4 | `game_version` 為空或不在 `version_map` 的比例與各版本手數 | > 1% 警告；這些樣本不進第 9–11 節 |
| R5 | 相鄰付費手 balance 不相接比例 | 只報告 |
| R6 | 找不到前一個付費手的免費局列 | 排除 |
| R7 | 快取是否涵蓋讀取範圍 | 前段（讀取範圍起點）或分析日不足 → 中止；只有後段不足 → 警告並截尾（快取補上後該日視為過期，5.0） |

成本：第一段每次只載入一個日檔；第二段逐桶處理，逐手狀態不落地。實測（21 天、595 萬個決策點，48 桶、2 個 worker）：第一段約 30 秒、第二段約 85 秒。

---

## 13. 測試

### 13.1 `test_build.py`

**逐手序列**

- 付費手 bet 100、prize 0，之後兩筆免費局 prize 300、200 → 合併為 prize 500、SMALL_WIN。
- 倍數邊界：0 → NO_WIN；0.5 → TINY；1.0 → SMALL；7.0 → MEDIUM；30.0 → BIG。
- 同秒兩筆依檔內順序；免費局的 `game_version` 不影響合併後的 arm。

**特徵**：一位玩家、同 session、同美東日、間隔 1 分鐘：

| 手 | 0 | 1 | 2 | 3 | 4 | 5 | 6（測試點） |
|---|---|---|---|---|---|---|---|
| bet | 100 | 100 | 100 | 100 | 100 | 100 | 200 |
| prize | 0 | 50 | 0 | 800 | 0 | 0 | 3,500 |
| balance_before | 10,000 | 9,900 | 9,850 | 9,750 | 10,450 | 10,350 | 10,250 |

第 6 手預期：

| 特徵 | 預期 | 特徵 | 預期 |
|---|---|---|---|
| `GP_dailySpinCount` | 6 | `SY_bet` | 200 |
| `GP_dailyRate_NO_WIN` | 0.6667 | `SY_isBuyFeature` | 0 |
| `GP_dailyRate_TINY_WIN` | 0.1667 | `SY_first_balance` | 10,000 |
| `GP_dailyRate_MEDIUM_WIN` | 0.1667 | `SY_balance` | 10,250 |
| `GP_dailyRate_BIG_WIN` | 0 | `SY_balanceToBet` | 51.25（10,250 ÷ 200） |
| `GP_dailyRtp` | 1.4167 | `NGP_dailyTotalWinToBet` | 4.25（850 ÷ 200） |
| `GP_dailyRate_MEDIUM_PLUS` | 0.1667 | `GP_dailyNoWin_x_spins` | 1.2973 |
| `GP_dailyRate_FREE_GAME` | 0 | `GP_flag_dryLong`、`NGP_flag_coldLoss` | 0、0 |

另外驗證：

- 預期表的欄位 = dataset 的全部快層欄位。
- 第 6 手 prize 改為 0 或 100,000 → 第 6 手特徵全部不變。
- 美東日界：UTC 03:59 與 04:01（8 月）兩手 → 第二手 `GP_dailySpinCount` = 0；1 月以 UTC 05:00 為界。
- 多條件：連續 120 手 NO_WIN、`SY_first_balance` 10,000、`SY_balance` 6,000 → 兩個旗標皆 1。
- BUY 手：第 3 手改 `game_mode = 19` → 第 6 手水池特徵不變；第 3 手 `SY_isBuyFeature` = 1。
- FreeGame：第 3 手 `game_mode = 3` → 第 6 手 `GP_dailyRate_FREE_GAME` = 1/6、第 3 手 = 0；第 6 手（當手）`game_mode = 3` → 第 6 手特徵全部不變。
- 新 session 第一手 `balance_before` 8,000 → 該 session 各手 `SY_first_balance` = 8,000，`session_spin_idx` 從 0 起。

**Y 與決策點**

- 25 手後中斷 2 天：第 0 手 `churnNext20` = 0、第 20 手 = 1；中斷 2 小時：第 20 手 = 0。
- 50 手 session → 第 0、20、40 手三筆，只有第 40 手可能為 1。
- 窗口 17 手保護傘、3 手高倍數 → 抽入，arm = 保護傘；15 對 5 → 不抽入。
- session 結束 + 24 小時超過資料終點 → 整個 session 不抽入；資料夠長時三個決策點都抽入。當天 23:00 開始、隔天 12:30 才結束的 session → 不抽入（即使資料夠長）；11:30 結束 → 抽入。
- 同一天單獨建、或和前一天一起建 → 決策點與 `session_id` 完全相同；`session_id` = 20260818001、20260818002（前一天 23:50 開始的 session 歸 08-17）。各天的統計相加 = 整段統計。
- 前一天 23:50 開始、跨過午夜的 session 不出決策點（整段歸前一天）；當天 23:50 開始、跨過午夜的 session，午夜後的決策點也算當天（`session_date` = 當天）。`pre_hours` 與後段資料計入 session 序號與 Y。
- 範圍起點前 10 分鐘還有付費手 → 範圍內第一個 session 不抽決策點、之後的 session 照常；間隔 40 分鐘或範圍前沒玩 → 照常抽。範圍前檢查只取付費手、只取窗口內。
- 讀取範圍：訓練期間從 08-28 起、驗證期間 08-29、`post_hours` = 12 → 美東 08-27 18:00 至 08-31 12:00，讀 UTC 08-27 至 08-31，R7 通過。只建 08-29 → 08-28 18:00 至 08-31 12:00。驗證期間 08-31 → 分析日本身超出快取，R7 中止。快取從 08-21 起：只建 08-29 通過；只建 08-21 → 讀取範圍起點 08-20 18:00 早於快取，R7 中止。
- 一起處理：訓練期間最後一天 23:50 開始、跨午夜的 session 只出現在訓練；驗證期間 09:00 開始的 session 只出現在驗證。
- 1,000 手隨機序列上，session 表算出的 Y 與逐手往後檢查 20 手完全相同。

**權重、標籤、分桶**

- 同一天 100 個決策點、cap 50 → 全部保留，每筆 `w_cap` = 0.5；30 個 → 每筆 1；Y 與特徵不變。
- 標籤（W = 202635）：有 202634 → 取 202634；最新為 202625 → 取 202625 的值；最新為 202630 補零列「流失」→ 流失；202635 以後的列不得使用；完全無紀錄 → 新手、無紀錄。
- 同一玩家只在同一桶；分桶前後總列數相同。

### 13.1a `test_store.py`

- 日期依連續區段分組；指紋只隨 build 相關設定改變（期間、`compute`、`tree` 不影響；旗標門檻 `features.derived` 會）；`config.yaml` 的 `features` 只有兩個旗標的門檻。
- 某天：沒有檔案 → 還沒建；指紋不同 → 設定或程式已改；建置時用到的注單日檔列數、或標籤週的列數／最後 ETL 時間與現在的 manifest 不同 → 快取已重抓；都相同 → 可用。
- 建置時快取尾端不足而被截尾的日子：快取補上後 → 過期；同一次 build 中尾端看得完的日子、或快取沒有變長時 → 可用。
- dataset 簽章：同一批日子順序無關；任一天重建就不同。
- 讀指定日子的 dataset；缺日子 → 報錯。所有引用某 build 的日子都重建後，該 build 紀錄刪除。

### 13.2 `test_model.py`

- `model.yaml`：dataset 的快層 = `fast` 列出的欄位（含註解掉的）；實驗的 X = fast ＋ tags[名稱]（不累加）；experiments 預設 8 組且順序固定；舊的 `slow`、不認得的 split／tags、重複欄位、不存在的欄位、`churnNext20`／`w_cap`／`arm`／`arm_ratio`／`SEG_`／索引 → 報錯。
- 編碼：缺值 → −1 ＋ `_isna`；one-hot 依訓練類別，未見過的類別 → 全 0；常數欄移除。
- 葉子規則可讀（`≤`／`>`、`= 類別`／`≠ 類別`），同欄位條件合併；葉子編號 `群/Lk` 由左到右。
- `sample_weight` = `w_cap`：流失列加權後葉子流失率上升。
- 訓練期間長樹、驗證期間預測與 AUC；樹的節點（左右子節點、提問、筆數）完整寫進 `tree_rules.json`；提問的讀法（缺值、= 最小值、整數門檻、one-hot、0／1）；`tree_rules.json` 記下 dataset 簽章。
- split = value：群內 `TAG_value_combined` 不放入 X；訓練期間不足的群沿用 all（`splits.groups` 列出的群也一樣）；未列出的群不建模、不評估；all 樹也要達到訓練樣本門檻。
- 自建樹的群在驗證期間沒有決策點 → 該群不預測（驗證 0 筆），其他群照常。
- `categories` 白名單：未列出的類別 one-hot 全為 0。
- `regroup`：`TAG_` 與對應的 `SEG_` 一起改、其他欄位不動；只能指定 dataset 中的 `TAG_` 欄位；`config.yaml` 是流失併入回流；`tree_rules.json` 記下 `regroup`。

### 13.3 `test_compare.py`

- 加權 AUC（大量同分）= sklearn `roc_auc_score`；bootstrap 依玩家重抽權重。
- 共同樣本 = 每個實驗都有預測的決策點；相同模型的 AUC 差距 = 0、判定「無顯著差異」；基準本身標「基準」。
- 判定（較佳／較差／無顯著差異）。
- 信心分數：高 1 分、中 0.5 分、低與無法比較 0 分，依決策點權重加總；維持現狀的葉子不計入（全部維持現狀時不給分）；找不到信心的葉子當無法比較；合併後依合併對照算。

### 13.4 `test_evaluate.py`

- 9.3 驗收範例（raw／IPW 流失率；IPW 後 max_smd = 0）。
- 雙重穩健：結果模型錯（常數）但傾向分數對、或傾向分數錯（均勻）但結果模型對，加權留存率都等於子狀態標準化後的真值；bootstrap 點估計 = 直接計算的 DR。
- 共同範圍：加入只在少數狀態出現的版本會讓涵蓋 < `min_overlap_share` → 不比該版本；門檻放寬則多比；都不到門檻選涵蓋最多；沒有可行組合 → 不比較、不修剪；合併後重挑共同範圍；共同範圍不到門檻 → 信心最多低。
- 最高留存 arm：樣本不足的 arm 不當候選；信心高需最高留存 arm 平衡且顯著（次高未平衡也可）；最高留存 arm 未平衡或只有一個 arm → 信心低。
- 合併：子節點有無法比較的葉子 → 父節點合併（名稱 `L1–L2`）；一路合到根仍無法比較時停在根；全部可比較時不合併。
- 合併相近建議：兩個子葉子建議同一版本 → 合併，可一路併到根；建議不同但兩片都不是高信心 → 只在這棵樹併完相同建議後仍超過 `similar_merge_over_leaves` 片時合併（這一輪不再往上）；有一邊是高信心或無法比較就不併。
- 維持現狀：版本 a 佔 90% 且只有 a 樣本足夠 → `dominant_share` 0.8 時判維持現狀、0.9 時判低（要「超過」門檻）；維持現狀的葉子在「合併無法比較」時不合併；「合併相近建議」時當成版本 X，可和建議 X 的葉子合併。
- 葉子內校準：平均偏 0.6 個標準差的 arm 校準後對齊到 ±0.05 內、判平衡、有效樣本變少；完全不重疊的 arm 對不齊、仍判未平衡。
- evaluate 前的檢查：dataset 簽章不同、`regroup` 與 run 時不同（沒記的舊輸出 = 沒合併）、或 X 欄位／缺值設定與 `model.yaml` 不同 → 回報原因；舊輸出沒有簽章時不擋；預測在 dataset 找不到的筆數。
- 第 10–11 節實作時補：高倍數展開成 M4、M5（`uplift` 相同、並列）；單一版本葉子（只有保護傘合格 → 清單 [M1]、信心低、`single_arm`）；各實驗的 W 逐筆相同，差距用同一組 Poisson 權重，區間含 0 → 「無顯著差異」。

### 13.5 `test_pages.py`

- 決策樹 SVG 含提問、是／否、葉子編號、最佳 arm、信心；合併後的節點畫成葉子（名稱 `L1–L2`）。
- 熱圖：樣本不足的格子為灰色、只標 n；只有一個 arm 樣本足夠的葉子標「無法比較」、不標 ★；只寫葉子編號、不寫規則。
- 熱圖有整體留存欄；決策樹方框寫留存率（不是流失率），葉子寫最佳版本與它的加權留存率；維持現狀的葉子寫「維持現狀_0」、不標 ★。
- `config.yaml`：`train_start` ≤ `train_end` < `valid_start` ≤ `valid_end`；快取與輸出資料夾由商戶／幣別／遊戲決定（不含日期）；舊的 `analysis_date`、`train_date`、`bet_records_dir`、`tag_dir` 報錯；輸出路徑 = `{商戶}_{幣別}_{遊戲}/train_{起}-{迄}_valid_{起}[-{迄}]`。
- summary：總覽表有 AUC 與信心分數（合併後）；尚未 evaluate 的實驗標「尚未 evaluate」；附評分依據。

### 13.6 其他

- `test_five.py`：9.4 的子規則、挑位置與各補位選項。
- `test_variants.py`：加標籤的規則同第 7 節；變體的輸出路徑。
- `test_cli.py`：evaluate 只讀驗證期間的日子；run 讀訓練期間與驗證期間；讀入時套用 `regroup`。

---

## 14. 待確認事項

### 14.1 遊戲端規則

| 問題 | 預設處理 |
|---|---|
| 現行 `gameVersion` 分派規則 | 未知；以今日 RTP、手數等特徵配適傾向分數。取得規則後確認其變數都在 X 中（`safeModeMap` 可能相關） |
| `version_map`：5 個 `game_version` 對 4 個原版本 | 由使用者填寫，未填完時 build 停止（或以 `--provisional-arm` 暫代）。線索：`_0` 為每日開局主版本、`_1` 多在輸錢時、`_2` 多在 RTP 高或剛中 BIG_WIN 後 |
| 水池倍數區間是否計入免費局贏分 | 併入前一個付費手後計算 |
| 模組切換時機 | 遊戲端每手可切換；策略每 20 手決定一次 |
| 樣本 arm 的定義 | 目前 = 決策點之後 20 手的主版本（5.4）。但窗口內每手的版本會被窗口內前一手的開獎結果影響：2026-08-29 bucket 0 的抽入窗口中 7.9% 窗口內出現 BIG_WIN，這些窗口流失率 1.2%（其他 7.0%）、arm = `_2` 佔 58%（其他 40%）。決策後的開獎結果同時影響 arm 與 Y，傾向分數校正不了。替代方案：arm = 決策點當手的 `game_version`（只依決策當下資訊，可不再篩主版本）；待決定 |
| M3、M7 沒有原版本 | 列為 `untested_modules`；實作後加入 `deployable` 並需線上實驗 |

### 14.2 資料範圍

| 問題 | 預設處理 |
|---|---|
| 標籤只從 202619 起 | 更早玩過、之後都沒紀錄的玩家會被歸為新手；依「無紀錄」人數決定是否往前補抓 |
| 訓練期間與驗證期間的時間重疊 | 同一筆資料不會出現在兩邊（0.1）。仍有兩種重疊：① 訓練期間最後一天的 Y 要看之後 24 小時，會看到驗證期間的回訪；② 驗證期間第一天的日水池特徵從 0 點起算，會計入前一晚跨午夜 session 在午夜後的手（與遊戲端一致）。要完全隔開需在兩段之間空一天。上線時要等 Y 完整（訓練期間之後 24 小時）才能訓練 |

### 14.3 分群定義

| 問題 | 預設處理 |
|---|---|
| 價值分群 | 前一週沒來的玩家取更早的最新一筆（可能是「本週無下注」）；無紀錄、本週無下注、手數過低不建模，實驗比較的共同樣本因此只含低／中／高（8.1、8.2） |
| 分群後樣本 | 群在訓練期間 < `min_segment_train_n` 筆沿用 all（8.1） |

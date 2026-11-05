# 體感模組決策樹：架構規格

> 輸入：`tableau.bet_records` 逐筆注單（含 `game_version`）＋ `dwt_player_game_tag_wi_test` 週標籤，皆用已落地的快取。
> 本文件只定義架構、資料流、特徵算法與驗收標準，不含程式碼。

---

## 0. 實作規則

- 依第 1–13 節順序實作，每完成一節就跑第 13 節對應測試。
- 可調參數放 `config.yaml`（第 3 節）；決策樹要放進 X 的欄位放 `model.yaml`（第 8 節）；0.1 的常數寫成程式中的具名常數。
- 只讀快取，本專案不新增連 DB 的程式。bet_records 以 polars 逐日檔讀取；標籤用 `tag_loader.load_tags(start="2026-05-11", end="2026-08-31")`。
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
| 水池週期 | 日：美東 0 點重置（與遊戲端 gamePotSave 一致）；月：美東每月 1 日重置（只算 `GP_rtp`，目前不使用） |
| 決策點 | 每個 session 第 0、20、40… 手；Y 窗口 = 之後 20 手，不重疊 |
| 商戶／幣別／遊戲 | `config.yaml` 的 `data.merchant`、`currency`、`game`（目前 TCG、PHP、22071），決定快取與輸出的資料夾（不含日期，`paths.py`）；只讀該幣別的注單（`currency=PHP`）|
| 每份資料只存一份（2026-10-27） | 注單快取 `data/bet_records/bet_records_{商戶}_{遊戲}_tz+0/`、標籤快取 `data/tags/…_{商戶}_{幣別}_{遊戲}_lag1w_asof/` 不含日期，延伸期間只補抓缺的日子、manifest 併入；dataset 依 `session_date` 一天一檔（`output/{商戶}_{幣別}_{遊戲}/dataset/{日期}.parquet`），所有 run 共用，build 只建還沒有（或已過期）的日子；分桶是 build 中的暫存，結束即刪 |
| 訓練期間／驗證期間 | 時間外驗證：session 起點落在 `train_start`～`train_end` 的決策點訓練決策樹，`valid_start`～`valid_end`（必須晚於 `train_end`，中間可以空幾天）的決策點評估 AUC 與葉子 × arm。run 從逐日 dataset 取這兩段的日子；同一玩家可同時出現在訓練與驗證，同一個 session 不會 |
| 輸出路徑 | 實驗結果在 `output／result` 底下的 `{商戶}_{幣別}_{遊戲}/train_{起}-{迄}_valid_{起}[-{迄}]/`（驗證只有一天時只寫一個日期），例如 `TCG_PHP_22071/train_20260831-20260913_valid_20260921-20260927/`，以下簡寫為 `{run}` |
| 決策點歸屬日 | session 起點的美東日期。跨午夜的 session 整段（含午夜後的決策點）歸起點那天，所以訓練期間與驗證期間交界的 session 歸訓練，同一個 session 不會同時出現在兩邊。午夜後決策點的日水池特徵仍依遊戲端 0 點重置 |
| dataset 欄位命名 | 索引在前（`member_account`、`spin_ts`、`session_date`（session 起點的美東日期）、`session_id`（= `session_date` 的 yyyymmdd × 1000 ＋ 該玩家當天第幾個 session）、`session_spin_idx`、`week_index`）→ 遊戲版本（`game_version`、`arm`、`arm_ratio`）→ 快層 `GP_`（水池類，只用當手之前的手）→ 當月 RTP `GP_rtp`（目前不使用）→ 快層 `SY_`（系統端：當手下注額、是否 BUY、下注前餘額、session 起始餘額）→ 快層 `NGP_`（水池與系統端組合的衍生特徵）→ 慢層 `TAG_` → 分群 `SEG_` → Y／權重（`churnNext20`、`w_cap`） |

---

## 1. 目標與資料

### 1.1 目標

1. 由 bet_records 建立逐手序列，在決策點算出快層特徵（第 6 節），加入慢層標籤（第 7 節）。
2. 訓練流失風險決策樹，Y = 20 手內是否離場；依實驗清單（三種切分 × TAG_ 組合，共 7 組，第 2 節）各訓練一次（第 8 節）。
3. 每個葉子以傾向分數加權比較各版本流失率，選出模組（第 9、10 節）。
4. 離線評估新策略相對現況的留存，並比較各實驗（第 11 節）。

### 1.2 資料

| 資料 | 快取 | 範圍 |
|---|---|---|
| bet_records | `data/bet_records/bet_records_TCG_22071_20260801_20260901_tz+0/` | UTC 08-01 至 08-31（31 個日檔）＝美東 07-31 20:00 至 08-31 20:00 |
| 週標籤 | `data/tags/dwt_player_game_tag_wi_test_TCG_PHP_22071_20260511_20260831_lag1w_asof/` | 202619（05-04 起）至 202634，所有玩家 |

### 1.3 實測概況

| 項目 | 實測值 | 設計上的處理 |
|---|---|---|
| 規模（美東 08-29） | 787 萬付費手、30,551 位玩家、41.4 萬個候選決策點 | 依玩家分桶處理 |
| 每人每日 | 付費手中位數 86、P99 2,453；決策點中位數 5、P99 125 | 用權重壓低重度玩家影響力，不刪資料（5.5） |
| `game_version` | 付費手 `22071_1.0.0_{0,1,2,3,5}`，佔比 `_2` 42%、`_0` 38%、`_5` 12%、`_1` 6%、`_3` 2% | 經 `version_map` 對應原版本 |
| 版本切換 | session 內每手 1.9% 換版本；20 手窗口有 22% 含多個版本 | 只抽主版本明確的決策點（5.4） |
| 版本依狀態分派 | 今日 RTP > 1.2 或前一手 BIG_WIN 時 `_2` 佔 60–68%；今日 RTP < 0.8 時 `_1` 佔 20%；今日前 20 手 `_0` 佔 75% | 今日 RTP 與手數必須放入 X（6.2）；部分葉子只有單一版本（10.1）。實測：同 session 前一手 BIG_WIN 後 `_2` 由 41% 升到 67%、`_1` 降到 0%（跨 session 無此效果，曾試加 `GP_prevBigWin`，已移除，見 14.1）；`_3` 依下注額分派、`_1` 在手冷時分派 |
| 對不到前一週標籤 | 42% 玩家（多數 8/1–8/23 從未出現） | 取最新一筆標籤，完全沒有 → 新手（7.2） |
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
├── feel_engine/                      # 程式與設定，不放產出；依流程分資料夾（2026-10-19）
│   ├── cli.py                        # 指令入口（python -m feel_engine.cli）
│   ├── variants.py                   # 實驗的變體（python -m feel_engine.variants；另外一頁、不進 summary）
│   ├── paths.py                      # 所有路徑集中定義
│   ├── config/
│   │   ├── config.yaml
│   │   └── model.yaml                # 決策樹要放進 X 的欄位（第 8 節）
│   ├── pipeline/                     # 建資料集（build）
│   │   ├── data.py                   # 第一段：日檔 → 分桶（暫存）
│   │   ├── build.py                  # 第二段：逐手序列、狀態、Y、決策點、特徵 → 逐日 dataset
│   │   ├── store.py                  # 逐日 dataset：指紋、缺／過期的判斷、讀取
│   │   └── report.py                 # build 報告
│   ├── modeling/                     # 決策樹（run）
│   │   └── model.py                  # 第 8 節：決策樹（訓練期間訓練、驗證期間評估）
│   ├── evaluation/                   # 葉子 × arm 評估（evaluate，第 9 節）
│   │   ├── propensity.py             # 9.1：傾向分數、結果模型
│   │   ├── leaf_eval.py              # 9.1a–9.2：共同範圍、葉子內校準、加權留存率、bootstrap、最高留存 arm
│   │   ├── merge.py                  # 合併無法比較的葉子、合併相近建議
│   │   └── evaluate.py               # 主流程：每個實驗一頁 HTML
│   ├── html/
│   │   ├── pages.py                  # HTML 頁面（決策樹 SVG、留存率熱圖、AUC 圖）
│   │   └── compare.py                # 8.2：實驗比較（AUC）
│   └── tests/
├── output/{商戶}_{幣別}_{遊戲}/      # 中間產物（可重跑產生）
│   ├── dataset/                      # 逐日 dataset，所有 run 共用（同一天只存一份）
│   │   ├── {日期}.parquet            # session_date = 該日的決策點
│   │   ├── {日期}.json               # 指紋、列數、產生它的 build、逐日統計
│   │   └── builds/{build_id}.json    # 一次 build：讀取範圍、R1–R7、用到的注單日檔與標籤週
│   ├── _build_tmp/                   # build 中的分桶、月初基準（結束即刪）
│   └── train_…_valid_…/              # = output/{run}/
│       ├── model/{split}/tag_{tags}/ # tree_rules.json（樹結構、葉子、設定）、valid_predictions.parquet
│       └── state_debug/              # build 時隨機 5 位玩家的完整逐手狀態
└── result/{run}/                     # 給人看的成果
    ├── build/                        # data_quality_report.md、build_report.md（本 run 的日子）
    ├── summary.html                  # 實驗比較：總覽表、AUC（圖與表）、信心（表與評分依據）（8.2）
    └── {split}_tag_{tags}.html       # 每個實驗一頁（決策樹、葉子 × arm 留存率與信心，第 9 節）
```

```
python data_loading/bet_records_loader.py --start 2026-09-26 --end 2026-09-30   # 補抓注單（只抓缺的 UTC 日）
python data_loading/tag_loader.py --start 2026-05-11 --end 2026-09-30   # 補抓週標籤（只抓缺的注單週）
python -m feel_engine.cli build                             # 建本 run 還沒有的日子：資料檢查 + 逐日 dataset（--rebuild 全部重建）
bash run_experiments.sh                                     # 跑 model.yaml 的 experiments 全部並比較（--build 先重建）
python -m feel_engine.cli run                              # 同上（不含 build）
python -m feel_engine.cli run --split lifecycle --tags value   # 只跑一組（可逗號分隔，跑所有組合）
python -m feel_engine.cli run --model-config my.yaml       # 換一份特徵與實驗設定檔
python -m feel_engine.cli compare                           # 實驗比較 summary.html（run、evaluate 結束時也會自動更新）
python -m feel_engine.cli evaluate                          # 第 9 節：每個實驗一頁 HTML（run_experiments.sh 也會跑）
python -m feel_engine.variants new                          # 實驗的變體（feel_engine/variants.py）：另外一頁 {split}_tag_{tags}_{變體}.html，不進 summary.html
python -m feel_engine.variants alltags --config my.yaml     # 變體也可以換 config（例如另一個驗證期間）
```

實驗的變體（`feel_engine/variants.py`，2026-10-28）：改分群的值（`SEG_` 改名，例如流失併入回流）、加標籤表的其他類別欄位（與 7 節同樣取決策週之前最新一筆；沒有紀錄 → 無紀錄）或 `five_feelings` 設定，流程與正式實驗相同（傾向分數仍用 `model.yaml` 的標籤）；資料寫到 `output/{run}/variants/{頁名}/`、頁面 `result/{run}/{頁名}.html`（頁名 = `{split}_tag_{tags}[_{後綴}]`），不在 `model.yaml` 清單、不進 `summary.html`。頁首寫變更內容，並與指定的正式實驗在同一批決策點上比較 AUC（玩家層級 bootstrap 95% 區間，同 `summary.html`）。`run --clean` 會清掉變體頁（`*_tag_*.html`），重跑變體指令即可。目前的變體：
- `new`（`lifecycle_tag_none_new.html`）：lifecycle／無，流失併入回流（同一棵樹）；2026-11-02 起改用原 `new2`（2026-10-30）的補位條件，原本的「每棵樹至少一片、每個沒出現的版本至少一次」（9.4 第 4a 點）與 `new2` 指令移除：補位條件 = 每棵樹都要有每個版本（該樹的葉子沒建議到的版本都在該樹內補一片），門檻內補不到可用可比區不足的位置，但比原建議低超過 2 pp（max_loss_pp）的位置一律不用（9.4 第 4b、4c、4d 點）；對照實驗 5。
- `alltags`（`all_tag_alltags.html`）：all，X = 快層 ＋ 標籤表所有有資料的類別標籤（生命週期、價值、總下注額、平均下注額、下注手數、資金水位、購買特色、下注強度；價值只取低／中／高，其他每個類別都 one-hot，含無紀錄、本週無下注）；對照實驗 1、4。

- 所有實驗共用同一份逐日 dataset（依 `session_date` 取訓練／驗證的日子），結果可直接比較。
- 實驗 = split × tags，清單與順序在 `model.yaml` 的 `experiments`（實驗比較依此排序）：

| # | split | TAG_ | 目的 |
|---|---|---|---|
| 1 | all | 無 | 基準：不分群、只用快層 |
| 2 | all | lifecycle | 生命週期當特徵 |
| 3 | all | value | 價值當特徵 |
| 4 | all | lifecycle＋value | 生命週期與價值都當特徵（對照 #2、#3；2026-10-22 加） |
| 5 | lifecycle | 無 | 依生命週期分群（對照 #2：分群 vs 當特徵） |
| 6 | lifecycle | value | 分群內再加價值（對照 #5） |
| 7 | value | 無 | 依價值分群（對照 #3） |
| 8 | value | lifecycle | 分群內再加生命週期（對照 #7） |

- 分群用的標籤不再放進同一個實驗的 X（群內為常數，放了也會被移除，只會影響沿用 all 的小群）。

---

## 3. `config.yaml`

```yaml
seed: 42

data:
  merchant: TCG                      # 商戶
  currency: PHP                      # 幣別（只讀 currency=PHP）
  game: "22071"                      # 遊戲代號；三者決定快取與輸出的資料夾（不含日期）
  train_start: "2026-08-31"          # 美東；訓練期間第一天
  train_end: "2026-09-13"            # 美東；訓練期間最後一天（含）
  valid_start: "2026-09-21"          # 美東；驗證期間第一天，須晚於 train_end
  valid_end: "2026-09-27"            # 美東；驗證期間最後一天（含）；AUC、葉子 × arm 都在驗證期間計算
  pre_hours: 6                       # 往前多讀：判斷午夜後的手是否接續前一天的 session（是 → 整段歸前一天）
  post_hours: 12                     # 截尾：session 須在起點隔天 0 點 + 此值前結束（再加 return_window_hours 判斷 Y）

compute:
  n_buckets: 48                      # 一次 build 讀 5 天以上 16 桶、14 天以上 32 桶、20 天以上 48 桶，每桶約 3 百萬手
  workers: 2                         # 每個 worker 約需 1 GB

modules:
  version_map:                       # game_version → 原版本；未填完時 build 只輸出版本分佈後停止
    "22071_1.0.0_0": null
    "22071_1.0.0_1": null
    "22071_1.0.0_2": null
    "22071_1.0.0_3": null
    "22071_1.0.0_5": null
  rtp_level: {low: 96.0, mid: 96.5, high: 97.0}   # 設計值；low 為暫定
  design:                            # 波動度 × 倍率 × RTP；source = 提供離線資料的原版本
    M1: {name: 保護傘, volatility: mid,  multiplier: mid,  rtp: high, source: 保護傘}
    M2: {name: 標準,   volatility: mid,  multiplier: mid,  rtp: mid,  source: 原模組}
    M3: {name: 冷卻,   volatility: mid,  multiplier: mid,  rtp: low,  source: null}
    M4: {name: 驚喜,   volatility: mid,  multiplier: high, rtp: high, source: 高倍數}
    M5: {name: 高倍數, volatility: high, multiplier: high, rtp: mid,  source: 高倍數}
    M6: {name: 高波動, volatility: high, multiplier: mid,  rtp: mid,  source: 深模組}
    M7: {name: 極限,   volatility: high, multiplier: high, rtp: low,  source: null}
  baseline: 原模組
  deployable: [M1, M2, M4, M5, M6]   # M3、M7 實作後再加入

labels:
  session_gap_minutes: 30
  churn_window_spins: 20             # 同時是決策間隔
  return_window_hours: 24
  arm_dominance: 0.8                 # 窗口內主版本佔比 ≥ 此值的決策點才抽入樣本
                                     # 調整依據：符合比例，以及符合／不符合者的狀態差異

weighting:
  player_day_cap: 50                 # 每位玩家每天的權重總和上限
                                     # 調整依據：w_cap < 1 的比例與價值分層結果

splits:                              # split = all／lifecycle（SEG_lifecycle）／value（SEG_value）
  min_segment_train_n: 10000         # 一棵樹的訓練樣本至少幾筆（all 與分群都適用）；群少於此數 → 沿用 all 的策略
  groups:                            # 指定 split 只對這些群建模（樣本仍需 ≥ min_segment_train_n，否則沿用 all）；
    value: [低, 中, 高]              # 未列出的群（無紀錄、本週無下注、手數過低）在該 split 不建模、不評估

features:
  derived:                           # 調整依據：各旗標的觸發比例
    dry_min_spins: 100               # GP_flag_dryLong：今日手數下限
    cold_nowin_rate: 0.8             # NGP_flag_coldLoss：今日 NO_WIN 比例下限
    loss_ratio: 0.3                  # NGP_flag_coldLoss：balance 相對進場的跌幅

tree:
  max_depth: 4
  min_samples_leaf: 1000             # all 樹與分群樹共用：每片葉子至少幾筆訓練期間決策點
  min_impurity_decrease: 0.00005    # 每次切分至少要讓流失預測改善多少；太小的切分不切

leaf_eval:
  min_n_per_arm: 300
  dominant_share: 0.8                # 同一版本佔葉子超過此值、樣本足夠的版本不到兩個 → 維持現狀_X（9.2）
  propensity_model: logistic         # logistic = 多項式邏輯迴歸；gbm = 梯度提升樹（9.1）
  balance_calibration: true          # 葉子內校準權重（9.1）
  balance_tolerance: 0.1             # 校準目標：各共變數加權平均與共同範圍相差 ≤ 0.1 個標準差
  overlap_min: 0.05                  # 共同範圍：每個比較版本的傾向分數都 ≥ 此值的決策點（9.1a）
  min_overlap_share: 0.8             # 共同範圍至少涵蓋葉子此比例；做不到 → 信心最多「低」（9.1a、9.2）
  smd_threshold: 0.2                 # 單片葉子 n 小、共變數多，0.1 光抽樣誤差就會超過
  similar_merge_over_leaves: 8       # 合併相同建議後仍超過此數的樹才合併相近建議（9.2）
  bootstrap_reps: 300

five_feelings:                       # 五個體感補位（9.4）
  enabled: true
  max_loss_pp: 2.0                   # 比原建議版本低不超過此值（點估計，百分點）
  min_comparable_share: 0.5          # 子規則內可比區至少佔此比例（9.4）
  many_trees: 3                      # 決策樹超過此數 → 每個待補版本補 placements_many_trees 片
  placements_many_trees: 2           # 各片在不同的樹
  min_per_tree: 0                    # 每棵樹至少補幾片（9.4 第 4a 點；0 = 不要求，變體 new 用 1）
  force_coverage: false              # 每棵樹、每個版本一定要補到，門檻內補不到用門檻外（9.4 第 4b 點；變體 new 用 true）
  every_version_per_tree: false      # 以每棵樹計：該樹沒建議到的版本都在該樹內補一片（9.4 第 4c、4d 點；變體 new 用 true）
  sub_rule_depth: 2                  # 子規則最多幾個條件
```

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

---

## 5. 計算流程

### 5.0 處理順序

```
build：本 run 的日子中，逐日 dataset 還沒有（或已過期）的，依連續區段分組；每段：

第一段 pipeline/data.py（逐日檔）
  R7 → 月初基準（玩家 × Σbet、Σprize）
     → 讀 9 欄 → R1–R4 → 轉美東 → 對應 arm → 依玩家雜湊寫入分桶（暫存）

第二段 pipeline/build.py（逐桶，可平行）
  ① 逐手序列              5.1
  ② 逐手狀態（水池）      5.2  ┐ 在完整序列上計算
  ③ session 與 Y          5.3  ┘
  ④ 抽決策點              5.4
  ⑤ 附加權重 w_cap        5.5
  ⑥ 計算特徵              第 6 節
  → 附加標籤（第 7 節）→ 依 session_date 一天一檔寫入逐日 dataset（store.py）→ 刪除暫存
```

一天的決策點只由該日與前後固定的範圍決定（截尾以起點日為準、`session_id` 含日期），與一次 build 幾天無關，所以不同 run 可以共用同一天。
某天在下列情況視為過期、下次 build 自動重建：build 相關設定（`data.pre_hours`、`post_hours`、`labels`、`weighting`、`features.derived`、`modules.version_map`）或 `pipeline/data.py`、`build.py` 改了；用到的注單日檔或標籤週重抓過（列數或最後 ETL 時間不同）。run／evaluate 遇到缺或過期的日子會停下、請先 build。

錯誤做法：先挑決策點，再對剩下的列往後平移算 Y。此時「往後 20 列」是之後 20 個決策點，標籤會錯且不會報錯。

### 5.1 讀取與逐手序列

- 讀 9 欄：`id`、`member_account`、`create_at`、`bet_points`、`prize`、`points`、`last_points`、`game_version`、`game_mode`。`id` 只用於去重。
- build 一次讀完一段連續的日子（本 run 缺的日子依連續區段分組；例如訓練期間與驗證期間中間空了幾天 → 兩段各讀一次，中間不讀）。
- 讀取範圍（美東）：區段第一天 00:00 − `pre_hours` 至最後一天 24:00 ＋ `post_hours` ＋ `return_window_hours`，換算 UTC 決定日檔。
- 月初基準：當月 1 日美東 0 點至讀取範圍起點，逐檔只讀 4 欄，加總成每位玩家的 Σbet、Σprize，不保留逐手資料。免費列算在它併入的那一手（前一個付費手）的月份：窗口內第一個付費手之前的免費列屬於上個月那一手，不計入；讀取範圍內、玩家第一個付費手之前的免費列（R6）併入範圍起點前最後一個付費手，那一手在本月（玩家在基準窗口有付費手）時把贏分加回基準。如此 `GP_rtp` 與讀取範圍起點無關。
- 範圍前檢查：讀取範圍起點前 `session_gap_minutes` 內，每位玩家最後一個付費手的時間（暫存 `pre_range_last.parquet`），供 5.4 條件 4 使用。
- 分桶：玩家雜湊 → 桶號，只為平行處理（不再依桶分 A／B 半）。
- 每桶排序後，`bet_points = 0` 的列併入前一個付費手：`prize` 相加，出場 balance 取最後一列 `last_points`。

| 付費手欄位 | 算法 | 開獎前可知 |
|---|---|---|
| `spin_ts` | `create_at`（美東） | 是 |
| `bet` | `bet_points` | 是 |
| `balance_before` | `points` | 是 |
| `arm` | `game_version` 對應的原版本 | 是 |
| `is_buy` | `game_mode = 19` | 是 |
| `is_free` | 付費手 `game_mode` 位元 2（Free） | 否（BUY 手除外） |
| `prize`、`outcome` | 合併後總贏分、倍數區間 | 否 |
| `balance_after` | 合併後最後一列 `last_points` | 否 |

### 5.2 逐手狀態

第 i 手的狀態只累計同一週期內、第 i − 1 手以前的付費手（shift(1) 後分組累計）。

| 週期 | 分組 | 累計內容 |
|---|---|---|
| 日 | 玩家 × 美東日期 | 手數、各倍數區間次數、含 FreeGame 的手數、Σbet、Σprize |
| 月 | 玩家 × 美東年月 | 月初基準 ＋ 範圍內的 Σbet、Σprize（只算 `GP_rtp`，目前不使用） |
| session | 玩家 × session | 序號、第一手的時間與 `balance_before` |

### 5.3 Session 與 Y

- 與同玩家下一個付費手間隔 > 30 分鐘即切 session。
- 因決策間隔 = 窗口 = 20 手，只有 session 的最後一個決策點，窗口會含 session 結束：
  - `churnNext20` = 1：最後一個決策點，且下一個 session 晚於本 session 結束 + 24 小時（或沒有下一個）。
  - 其他 = 0；`retained` = 1 − `churnNext20`（評估時計算，dataset 不另存）。
- 截尾：session 在起點隔天 0 點 ＋ `post_hours`（12 小時）之後才結束，或結束 + 24 小時超過快取終點（看不完 Y）時，整個 session 的決策點都不抽入（只刪最後一個會讓流失率偏低，因為只有最後一個決策點可能流失）。每天都用同一條規則（2026-10-27 起；之前只有讀取範圍最後一天受 `post_hours` 限制），所以一天的決策點與 build 一次建幾天無關；08-31～09-13 依此規則重建後，與先前（舊規則）的 dataset 逐筆相同（`session_id` 以外 36 欄）。

### 5.4 抽決策點

只抽同時符合以下條件的決策點：

1. `session_spin_idx % 20 == 0`。
2. session 起點的美東日期落在訓練期間（`train_start`～`train_end`）或驗證期間（`valid_start`～`valid_end`），存為 `session_date`；跨午夜的 session 整段歸起點那天（0.1 決策點歸屬日）。
3. 未截尾。
4. session 起點完整：玩家在讀取範圍內第一手與範圍起點前最後一個付費手的間隔 > `session_gap_minutes` 才算真正的 session 起點；否則範圍內第一個 session 的起點是被讀取範圍切出來的（`session_spin_idx`、`GP_sessionMinutes`、`SY_first_balance` 會錯），整個 session 不抽決策點。跨午夜的 session 是正常的，不受此條件影響。
5. 窗口內主版本（手數最多的 arm）佔比 `arm_ratio` ≥ `arm_dominance`；樣本的 `arm` = 主版本。

報告列出：條件 4 排除的 session 數；符合條件 5 的比例，以及符合與不符合者的流失率、今日 RTP 比較。

### 5.5 玩家 × 日權重

- 全部決策點保留。每位玩家當天（歸屬日）有 n 個決策點，每筆權重 `w_cap` = min(1, `player_day_cap` ÷ n)。
- 一般玩家權重 1；重度玩家資料全留，但一天的總影響力最多等於 50 筆。
- `w_cap` 用於決策樹、傾向分數、葉子評估與離線評估。

---

## 6. 快層特徵（`GP_`／`NGP_`）

### 6.1 原則

- 在決策點「下注後、開獎前」計算：可用 5.2 的狀態與當手 `bet`、`balance_before`、`is_buy`；不可用當手 `prize`、`balance_after`、`is_free`。
- 由 bet_records 算出的快層特徵分三種前綴：`GP_` = 水池類，只用當手之前的手（`GP_sessionMinutes` 用到當手的下注時間）；`SY_` = 系統端狀態（當手下注額、是否 BUY、下注前餘額、餘額 ÷ 下注額、session 起始餘額）；`NGP_` = 水池與系統端組合的衍生特徵。任何特徵都不用當手的開獎結果。`model.yaml` 每個欄位寫成「欄位: 中文名稱」，中文名稱用在決策樹圖；水池特徵只用日週期的 count 與 mean；當月只算 `GP_rtp`（目前不使用），不放入 X；不使用前一個月。
- 除以 0 → NaN；決策樹前以 −1 填補並加 `<特徵>_isna`。

### 6.2 日水池（8 項）

| 特徵 | 算法 |
|---|---|
| `GP_dailySpinCount` | 今日手數 |
| `GP_dailyRate_{NO_WIN, TINY_WIN, SMALL_WIN, MEDIUM_WIN, BIG_WIN}` | 今日各區間次數 ÷ 今日手數 |
| `GP_dailyRtp` | 今日 Σprize ÷ Σbet（現行分派依此切換版本，必須放入 X） |
| `GP_dailyRate_FREE_GAME` | 今日含 FreeGame 的手數 ÷ 今日手數（不含當手；BUY 手也算）。當手是否中 FreeGame 是開獎結果，不放入 X |

目前不使用（不放入 X）：`GP_rtp` = 當月 Σprize ÷ Σbet（含月初基準），保留給之後考慮成本時使用。

### 6.3 交互作用與多條件（5 項）

| 特徵 | 算法 |
|---|---|
| `GP_dailyRate_MEDIUM_PLUS` | MEDIUM ＋ BIG 比率 |
| `GP_dailyNoWin_x_spins` | NO_WIN 比率 × log1p(今日手數) |
| `NGP_dailyNoWin_x_sessionLoss` | NO_WIN 比率 × max(0, 1 − `SY_balance` ÷ `SY_first_balance`) |
| `GP_flag_dryLong` | 今日手數 ≥ `dry_min_spins` 且 MEDIUM_PLUS = 0 → 1 |
| `NGP_flag_coldLoss` | NO_WIN 比率 ≥ `cold_nowin_rate` 且 `SY_balance` ≤ `SY_first_balance` × (1 − `loss_ratio`) → 1 |

旗標任一條件為 NaN 時記為 0。這些是特徵，不是分派規則。

### 6.4 非水池

| 特徵 | 算法 |
|---|---|
| `GP_sessionMinutes` | 本 session 已玩分鐘數 = 當手下注時間 − session 第一手時間（遊戲端 gamePotSave 有時間可算）；session 序號改為索引 `session_spin_idx`，不放入 X。2026-10-15 起 `model.yaml` 不放入 X（dataset 仍保留） |
| `SY_bet` | 當手下注金額 |
| `SY_isBuyFeature` | 當手是否購買特色 |
| `SY_balance` | 當手下注前的 balance |
| `SY_balanceToBet` | `SY_balance` ÷ `SY_bet`（以當手下注額還能玩幾手） |
| `SY_first_balance` | 本 session 第一手下注前的 balance（整個 session 固定） |
| `NGP_dailyTotalWinToBet` | 今日總贏分 ÷ 當手下注額：今日到當手之前的 Σprize（遊戲端 gamePotSave `totalWin` 的日版本，美東 0 點重置）÷ `SY_bet` |

---

## 7. 慢層標籤

| `model.yaml` tags | 加入 X 的欄位 |
|---|---|
| `none` | 無 |
| `lifecycle` | `TAG_lifecycle`：流失、一般、新手、下滑、活躍、回流 |
| `value` | `TAG_value_combined`：本週無下注、低、中、高、手數過低、無紀錄（one-hot 只取低／中／高） |

- 決策點所在美東週為 W，取該玩家 `week_index` < W 的**最新一筆**（含補零列）。
- 完全沒有紀錄 → `TAG_lifecycle` = 新手、`TAG_value_combined` = 無紀錄。
- 只讀 `member_account`、`week_index`、`is_zero_fill`、`lifecycle_tag`、`value_combined_tag`；類別欄位 one-hot。
- 寫入 dataset 時一律加 `TAG_` 前綴：`TAG_lifecycle`、`TAG_value_combined`（X），以及 `TAG_source`（W-1／更早／無紀錄）、`TAG_week_index`（取到的那筆標籤的週；決策週 W 本身存為索引 `week_index`）、`TAG_is_zero_fill`。
- 報告列出：取到 W − 1 週、更早週、無紀錄的人數與流失率。

---

## 8. 流失風險決策樹

- 訓練期間訓練、驗證期間預測；`DecisionTreeClassifier`，不用 `class_weight`，`sample_weight` = `w_cap`。超參數：`max_depth` 4、`min_samples_leaf` 1,000（all 樹與分群樹共用）、`min_impurity_decrease` 5×10⁻⁵（切分改善太小就不切，避免兩邊流失率只差一點也切開）。
- X 由 `feel_engine/config/model.yaml` 決定：`fast`（每個實驗都放）＋ `tags[實驗的 tags]`（不累加）；預設 = 快層特徵（`GP_`、`SY_`、`NGP_`，不含 `GP_rtp`）＋ 該實驗的慢層標籤（`TAG_`）。只能放 `GP_`／`SY_`／`NGP_`／`TAG_` 開頭的欄位（`arm`、`arm_ratio` 用到之後 20 手、`SEG_`、Y、`w_cap`、索引不得放入）。Y = `churnNext20`。
- 數值缺值填 `missing.fill`（−1），`missing.add_isna` 時另加 `<欄位>_isna`；文字欄位依訓練資料的類別 one-hot（驗證期間未見過的類別 → 全 0）；訓練資料中為常數的欄移除。
- `model.yaml` 的 `categories` 可限定文字欄位只對哪些類別做 one-hot，其他類別全為 0、不會單獨成為分支條件。預設 `TAG_value_combined: [低, 中, 高]`：無紀錄、本週無下注、手數過低不當節點。
- 輸出 `output/{run}/model/{split}/tag_{tags}/`（供程式讀取）：`tree_rules.json`（每個節點的左右子節點、提問、訓練期間筆數與加權流失率；葉子規則；本次放入 X 的欄位、超參數、model.yaml 路徑與內容指紋）、`valid_predictions.parquet`（驗證期間每筆的群、樹、葉子、預測流失機率）。給人看的決策樹畫在該實驗的 HTML（第 9 節）。`run --clean`（`run_experiments.sh` 預設）先清掉舊的實驗結果，避免殘留舊設定或已移出清單的實驗。
- 樹圖的提問（特徵以 model.yaml 的中文名稱顯示）：數值「X ≤ t？」（左 = 是；整數欄位寫成整數）；門檻落在缺值填補值與最小觀測值之間 →「X 缺值？」；門檻落在最小與次小觀測值之間 →「X = 最小值？」；左分支含缺值時註明「含缺值」；one-hot、0／1 欄位 →「X = 類別？」「X = 1？」（左 = 否）。
- 葉子規則（存在 `tree_rules.json`）：同一欄位同方向只留最嚴格的門檻；數值欄位兩個方向都有時寫成區間「a < x ≤ b」。

### 8.1 三種切分

| split | 群 | 做法 |
|---|---|---|
| `all` | 1 群 | 一棵樹、一份策略表 |
| `lifecycle` | `SEG_lifecycle`（= 7 節取得的 `TAG_lifecycle`）六類 | 每群各自訓練樹、評估葉子、產生策略表 |
| `value` | `SEG_value`（= 7 節取得的 `TAG_value_combined`）只建模低、中、高（`splits.groups.value`）；無紀錄、本週無下注、手數過低不建模、不評估 | 低、中、高各自訓練樹、評估葉子、產生策略表 |

- 分群只用決策週 W 之前的標籤，不得用當日下注分群（上線時決策前就要知道）。
- 分群欄位（`SEG_`）在群內為常數，對應的 `TAG_` 特徵自動從 X 移除。
- 有 `splits.groups.{split}` 時：只對列出的群建模，未列出的群不建模、不評估；列出的群訓練期間樣本仍需 ≥ `min_segment_train_n`，不足時沿用 `all` 的樹。
- 沒有指定時（lifecycle）：群的訓練期間樣本 < `min_segment_train_n` → 不訓練，沿用 `all` 的樹與策略，仍納入驗證期間評估。
- 分群樹與 all 樹用同一個 `min_samples_leaf`；葉子編號加群名（例如 `新手/L3`）。

### 8.2 實驗比較（AUC ＋ 信心）

`run` 結束時（或 `compare`）讀所有已跑過的 `output/{run}/model/{split}/tag_{tags}/valid_predictions.parquet`，依 `model.yaml` 的 `experiments` 排序（清單外的附在最後並標示），輸出 `result/{run}/summary.html`；`evaluate` 結束時再重做一次，加入信心分數。內容依序為總覽表（AUC、與基準的判定、信心分數原始／合併後）、① AUC（圖與表）、② 信心（表格與評分依據；2026-10-15 起不放信心長條圖）：

- 共同樣本：每個模型都有預測的驗證期間決策點（value 只建模低／中／高 → 共同樣本 = 這三群），AUC 與信心都在同一批決策點上計算。
- **AUC**：以 `w_cap` 加權，只看分不分得出誰會離開，不看體感模組；95% 區間與「與基準 all／無差」用玩家層級 Poisson(1) bootstrap `bootstrap_reps` 次、所有模型同一組重抽權重（先聚合成「預測值 × 玩家」再以矩陣運算）；差的 95% 區間不含 0 才寫「較佳／較差」，否則「無顯著差異」。
- **信心分數（0–100）** = 100 × (高佔比 ＋ 0.5 × 中佔比)。佔比 = 共同樣本中落在該信心等級葉子的決策點比例（`w_cap` 加權），維持現狀的葉子不計入；代表「可以選版本的決策點中，多少能拿到可信的版本建議」。原始葉子與合併後（9.2）各算一次，summary 與實驗頁頁首只顯示合併後（2026-10-20）；`evaluate` 寫的 `leaf_confidence.json` 比預測舊（run 之後還沒 evaluate）時顯示「尚未 evaluate」；全部都是維持現狀時不給分（—）。
- 給分依據（信心等級的定義見 9.2）：

| 信心 | 給分 | 理由 |
|---|---|---|
| 高 | 1 | 最佳版本平衡、且顯著高於次佳：建議可直接採用 |
| 中 | 0.5 | 最佳版本平衡，但和次佳沒有顯著差距：選哪個差不多，建議可用但價值有限 |
| 低 | 0 | 最佳版本未平衡（差距可能來自玩家狀態）或共同範圍不到葉子的 `min_overlap_share`（9.1） |
| 無法比較 | 0 | 共同範圍內樣本足夠的版本不到兩個（葉子太小，或各版本發給的玩家狀態不重疊） |
| 維持現狀 | 不計入 | 同一版本佔葉子超過 `dominant_share`（超過 80%）：現行分派讓大家幾乎都是同一版本，沒有別的版本可比，不算信心不足 |

- 照建議留存率（照每個實驗建議的版本分派、預估的留存率；玩家分兩半交叉估計）2026-10-15 做過，使用者擔心被誤用而移除，summary.html 不放留存率。
- AUC 與信心衡量不同的事：AUC = 分不分得出誰會離開；信心 = 版本建議可不可信。兩者都高的實驗最值得採用。

---

## 9. 葉子 × arm 評估

只用驗證期間，排除 arm 不明（R4）的樣本。

### 9.1 傾向分數與葉子內校準（只用資料，不用遊戲端的分派規則）

- p(arm | x)：只配適一次、所有實驗共用；X = 決策樹的快層特徵 ＋ 所有 `TAG_` 標籤（含分群用的標籤），不含 arm；依玩家分 5 折交叉配適（每筆的機率來自沒看過這位玩家的模型）；`sample_weight` = `w_cap`。
- 模型（`leaf_eval.propensity_model`）：`logistic` = 多項式邏輯迴歸（預設；特徵截斷頭尾 1%、非負取 `log1p`、標準化，0／1 欄位不截斷、不取 log；數值缺值填中位數並加 `_isna`）；`gbm` = 梯度提升樹，從資料自動學分派的門檻與交互作用（數值保留原值與缺值）。2026-10-12 實測：gbm 的權重較極端、有效樣本較少，配合下面的校準後信心不如 logistic，所以預設 logistic。
- 結果模型（2026-10-15 起）：同樣只配適一次、依玩家 5 折交叉配適；每個 arm 一個邏輯迴歸（只用拿到該 arm 的決策點，`sample_weight` = `w_cap`），用 9.1 的共變數預測「拿到該 arm 時」的留存率 m_arm(x)。
- **共同範圍**（9.1a，2026-10-15 起，取代 `propensity_min` 0.02 與 99 分位截斷）：每片葉子挑要比較的 arm 組合 C；共同範圍 S = 葉子內「C 的每個 arm 傾向分數都 ≥ `overlap_min`（0.05）」的決策點；C 的每個 arm 在 S 內都要 ≥ `min_n_per_arm` 筆。S 涵蓋葉子 ≥ `min_overlap_share`（0.8）的組合中挑 arm 最多的（同數量挑涵蓋最多）；都做不到 → 挑涵蓋最多的（信心最多「低」）；沒有任何兩個 arm 可行 → 不比較。目的：現行規則幾乎只給某個版本的狀態（傾向分數極低的那一邊）不拿來比，避免把「誰拿到」當成「版本效果」。S 內也包含拿到非比較 arm 的決策點（只用於結果模型的平均）。
- 權重：S 內拿到比較 arm 的決策點 W = `w_cap` ÷ p(實際 arm)（p ≥ 0.05，最大 20 倍，不再截斷）。
- 2026-10-15 診斷（C 組，舊方法）：信心「低」中 65–92% 是最佳版本未平衡；最佳版本多為全體只佔 1.9% 的 `_3`；未平衡者 max_smd 兩極（約 0.1 或 0.7–0.9）→ 放寬 `smd_threshold` 無效，改用共同範圍。
- **葉子內校準**（`leaf_eval.balance_calibration`，預設開）：傾向分數是對全體配適的，不保證每片葉子內平衡（實測未平衡多來自餘額、已玩分鐘、今日手數、session 淨輸贏、下注額）。S 內每個比較 arm（樣本 ≥ 20）以 W 為起點，找最接近 W 的權重（最小化 KL 距離，近似 entropy balancing），讓該 arm 每個共變數的加權平均與共同範圍 S 相差 ≤ `balance_tolerance`（0.1）個標準差；比原本的 W 更平衡才採用。某 arm 根本沒有某種狀態的玩家時對不齊，平衡檢查照實判未平衡。校準直接以平均對齊為目標，所以「平衡」代表平均已對齊；代價是權重變得不均、有效樣本（Kish：(ΣW)² ÷ ΣW²）變少、區間變寬。
- 2026-10-12 實測（共同樣本的信心分數，原始／合併後）：未校準 all 三組 15–18／24–25、分群四組 7–9／9–11；校準後 all 三組 9–23／26–27、分群四組 19–23／19–25。

### 9.2 每個葉子 × arm

| 欄位 | 算法 |
|---|---|
| `n`、`raw_churn` | 共同範圍內的樣本數、原始流失率（整片葉子） |
| 加權留存率（DR） | S 內 m_arm 的 `w_cap` 加權平均 ＋ S 內拿到該 arm 者 (y − m_arm) 的校準權重平均（y = 1 − churnNext20） |
| `max_smd` | 加權後該 arm 與共同範圍 S 各特徵標準化平均差的最大值 |
| `eligible` | 納入比較（arm ∈ C）且 `max_smd` ≤ `smd_threshold` |
| `uplift` | (基準 `ipw_churn` − 該 arm `ipw_churn`) × 100 |
| `uplift_ci` | 群集 bootstrap 95% 區間 |

- 基準 arm 不合格時，改用 `n` 最大的合格 arm 作臨時基準並標示。
- **最高留存 arm（目前實作）**：每片葉子在比較 arm（C）中挑加權留存率（DR）最高者，與次高、與臨時基準（`n` 最大者）比較差距與 bootstrap 95% 區間。信心（平衡只看最高留存 arm，次高是否平衡不影響）：高 = 最高留存 arm 平衡（校準後 `max_smd` ≤ `smd_threshold`）且顯著高於次高；中 = 平衡但不顯著；低 = 最高留存 arm 未平衡，或共同範圍不到葉子的 `min_overlap_share`；無法比較 = 比較 arm 不到兩個；維持現狀 = 比較 arm 不到兩個，且最多的 arm 佔葉子決策點超過 `dominant_share`（沿用該版本，不標最高）。未平衡時差距可能來自玩家狀態，不能當版本效果。熱圖的滑鼠提示另列有效樣本。
- 暫代模式（`version_map` 未填）：arm = `game_version`，只輸出最高留存 arm，不換算 M1–M7。
- 無法比較 = 葉子內找不到可比較的組合（比較 arm 不到兩個；不標最高留存 arm）。
- 合併無法比較的葉子：由下往上，某個節點的子節點中有無法比較的葉子 → 把該節點底下全部合併成一片葉子；維持現狀的葉子不合併（名稱 = 涵蓋的原葉子編號，例如 `高/L3–L6`），重挑共同範圍並重算本節；重複到每片葉子都可比較，或已合併到根。判斷只用各 arm 的樣本數與傾向分數（不看結果），用驗證期間資料。
- 合併相近建議（2026-10-22 由原「合併相同建議」擴充）：在合併無法比較之後，由下往上，同一個上層節點底下的兩片葉子建議的版本相同（高／中／低 → 最高留存 arm；維持現狀 → 最多的 arm；無法比較不併）→ 合併、可一路往上；建議不同但兩片都不是高信心（中／低／維持現狀，最佳版本沒有可信地勝出）→ 只有在這棵樹併完相同建議後仍超過 `similar_merge_over_leaves`（8）片（樹枝過於龐大）時才合併，其他樹保留多元建議。合併後重算本節，再依新結果合併，直到不再變化（最多 10 輪）。只影響版本建議，決策樹與 AUC 不變。
- 輸出：每個實驗一頁 `result/{run}/{split}_tag_{tags}.html`，分「原始葉子」「合併無法比較的葉子」「合併相近建議」三個分頁（`five_feelings.enabled` 時再加五個體感分頁，見 9.4：正式實驗兩個「五個體感1」「五個體感2」，變體頁一個「五個體感」）；多棵樹的實驗上方有「切換決策樹」按鈕（每次顯示一棵）；每棵樹：
  - 決策樹（大字 SVG；開啟時縮到一個畫面內，可用「符合畫面／−／＋／原始大小」按鈕縮放，切換分頁或調整視窗時自動重新符合）：內部節點寫提問與訓練期間筆數、流失率，線上標「是／否」；葉子寫編號、最佳 arm（以 arm 顏色填色）、信心，無法比較為灰色；維持現狀寫葉子的整體留存率。
  - 葉子 × arm 留存率熱圖：列 = 葉子編號（不寫規則），另列驗證決策點、驗證玩家數、整體留存（2026-10-15 試過「可比較比例」欄，使用者要求刪除）、欄 = arm；格內加權留存率與「玩家數／決策點筆數」（共同範圍內；2026-10-27 起，因顯著性以玩家為單位重抽），★ = 最高、樣本不足或未納入比較為灰色只標 n（不標未平衡；平衡與否反映在信心）；最後兩欄 = 最佳 arm、信心；滑鼠停留顯示 95% 區間與明細。
- bootstrap：先聚合成「玩家 × 葉子 × arm」的 Σw_cap、Σw_cap·m、ΣW、ΣW·(y − m)（稀疏矩陣），每次給每位玩家一個 Poisson(1) 權重，以矩陣運算算完（W 固定）。第 11 節同。

### 9.3 驗收範例

一個葉子分兩個子狀態，以子狀態內 arm 佔比為傾向分數：

| 子狀態 | 佔葉子 | 保護傘（n，流失率） | 原模組 | 深模組 | 高倍數 |
|---|---|---|---|---|---|
| A | 40% | 1,600，32% | 100，40% | 100，38% | 2,200，30% |
| B | 60% | 400，20% | 400，28% | 700，24% | 4,500，22% |

預期（±0.1 個百分點）：保護傘 raw 29.6% / ipw 24.8%；原模組 30.4% / 32.8%；深模組 25.75% / 29.6%。
以原模組為基準：M1 `uplift` = 8.0、M6 `uplift` = 3.2。
DR 檢查（同一份資料）：結果模型給錯（常數）但傾向分數正確，或傾向分數給錯（均勻）但結果模型正確（子狀態 × arm 的平均），加權留存率都等於上面的 ipw（雙重穩健）。

### 9.4 五個體感補位（2026-10-19）

目標：每個體感都至少是一條規則的建議，且整體留存最高（`config.yaml` 的 `five_feelings`；程式 `evaluation/five.py`）。

1. 有用到的版本：以「合併相近建議」為準；高／中／低 → 最高留存 arm，維持現狀_X → X，無法比較不算；以整個實驗計（分群實驗的多棵樹合起來）。沒用到的版本 v 才補位。
2. 候選：每片有建議（r）的葉子，可比區 O = 「v 與 r 的傾向分數都 ≥ `overlap_min`」；葉子內用深度 ≤ `sub_rule_depth` 的小樹（X = 該實驗決策樹的特徵與編碼，y = 是否在 O，`min_samples_leaf` = `min_n_per_arm`，`sample_weight` = `w_cap`；不看留存）描述 O。小樹每個節點（含根 = 整片葉子）中，v、r 在「節點 ∩ O」內各 ≥ `min_n_per_arm` 者，取 O 佔比最高（同佔比取筆數多）的當子規則；沒有 → 沒有可比區。
3. 比較：子規則內，共同範圍 = O、比較組合固定 {v, r}，沿用 9.1–9.2 的權重、校準、DR 加權留存率、`max_smd` 與 bootstrap；差距 = r − v（百分點），整體影響 = 子規則佔全部驗證決策點的比例（`w_cap` 加權）× 差距。
4. 挑位置：差距點估計 ≤ `max_loss_pp`、且可比較比例（子規則內 O 的佔比）≥ `min_comparable_share`（0.5，2026-10-20 使用者決定；整片葉子只有 1% 可比時差距多半是選擇偏差）的候選中，每個 v 一條（實驗的決策樹超過 `many_trees` 棵時 `placements_many_trees` 條、且分在不同的樹，2026-10-22）、每片葉子最多一個版本，以整數規劃（`scipy.optimize.milp`：先求補到的片數最多）挑整體影響總和最小者（v 較好時為負，優先）。沒補滿時寫「只補到 n 片（其他樹：原因）」；放不進去的原因：沒有可比區／可比較比例都不到 50%（最高 x%）／最小差距 x pp（葉子）／可行的葉子已分給其他版本。
4a. 每棵樹至少補幾片（`min_per_tree`，預設 0 = 不要求；2026-10-28 加，變體 new 用 1）：> 0 時整數規劃的目標依序為「每棵樹補到 `min_per_tree` 片」→「每個 v 補到上面的片數」→ 片數最少 → 整體影響最小（逐層求解、每層鎖住上一層的最佳值）；版本片數不設上限，每個（v, 樹）仍 ≤ 1 片、每片葉子 ≤ 1 個版本、差距與可比較比例門檻不變。補不到的樹寫「樹 X 沒有補位（原因）」，原因同上並標版本（最小差距 x pp（_v，葉子））；也寫進 `five_feelings.json` 的 `unfilled_trees`。
4b. 一定要補到（`force_coverage`，預設 false；2026-10-28 使用者定的變體 new 補位條件：每棵樹至少一片補位葉子、每個沒出現的版本至少補到一次 → `min_per_tree` 1、`placements_many_trees` 1、`force_coverage` true）：門檻內補不到時，改用門檻外、有比出差距的候選；目標在 4a 的「每個 v 補到片數」之後加「門檻外的片數最少」，再求片數最少、整體影響最小。選中的門檻外候選標「超過門檻：差距 x pp > 2 pp／可比較比例 y% < 50%」（補位結果表、樹圖、其他候選表的「選中（超過門檻）」，`five_feelings.json` 的 `outside`）。仍補不到只會是沒有可比區（子規則挖不出 v 與 r 各 ≥ `min_n_per_arm` 筆）或葉子已分給其他版本。
4c. 每棵樹都有每個版本（`every_version_per_tree`，預設 false；2026-10-30 使用者定的變體 new2 補位條件，2026-11-02 起為變體 new）：改以每棵樹計，該樹的葉子（合併相近建議）沒建議到的版本 v，都要在該樹的葉子內補一片；候選只在該樹的葉子內找（第 2、3 點不變）。稀有版本的可比區常集中在同一片葉子，所以一片葉子可補多個版本：依序補，後補的在扣掉前面子規則後的其餘部分重新挖子規則、比較（r 不變）。每棵樹分開窮舉每個 v 補在哪片葉子、同一片葉子內的先後（其餘部分的結果依先後快取），依序取：補到的 v 最多 → 門檻外的片數最少（配合 4b）→ 整體影響最小；整片葉子挖不出 v 的子規則時，其餘部分也不可能（子集），不列入。不看 `placements_many_trees`、`min_per_tree`。補不到的寫「_v 在樹 X 找不到合適位置（原因）」；每棵樹的說明列出該樹原本建議的版本、補位與補不到的版本；樹圖把後補的子規則接在「其餘」上，表格寫「葉子（扣掉 _v 的補位後）」，其他候選表只多列選中的其餘部分候選，整片葉子的候選因先後改補在其餘部分時標「改補在這片葉子的其餘部分」。`five_feelings.json` 加 `every_version_per_tree`、`need`（每棵樹要補的版本），`unplaced` 每筆帶 `tree`，`placed` 後補的帶 `after`（先補的版本）。
4d. 每棵樹都有每個版本時的差距上限（2026-10-30 使用者定：補位版本比原建議留存率低超過 5 pp 的葉子不要，同日改為 2 pp，並只保留 2 pp 的參數 → 不另設參數，直接用 `max_loss_pp`）：`every_version_per_tree` 時差距點估計一律 ≤ `max_loss_pp`，`force_coverage` 只放寬可比區佔比（可比區不足的候選也能用，標「超過門檻：可比較比例 y% < 50%」）；差距超過的候選狀態「差距超過上限」，只剩這種候選時寫「最小差距 x.xx pp（葉子） > 2 pp」（兩位小數，避免 2.02 顯示成 2.0）。4b 的整數規劃（變體 new）不受影響，差距仍可超過。
5. 輸出：正式實驗頁兩個分頁（2026-11-04 使用者要求）：「五個體感1」= `five_feelings`（每個版本在整個實驗至少出現一次），「五個體感2」= `five_feelings` 再覆蓋 `five_feelings_2`（`every_version_per_tree` true、`force_coverage` true：每棵樹都要有每個版本，見 4c、4d）；變體頁（`variants.py`）維持一個「五個體感」分頁、用變體自己的設定。每個分頁：樹圖在被選中的葉子下接子規則，新增的兩片葉子標驗證期間筆數；每棵樹下附與「合併相近建議」相同的葉子 × 版本留存率表；補位結果表；其他候選收合）；`output/{run}/model/{split}/tag_{tags}/five_feelings.json`（五個體感2 另寫 `five_feelings_2.json`）。`summary.html`、`leaf_confidence.json`、前三個分頁不變。
6. 限制：子規則的位置與挑選都在驗證期間，挑損失最小的候選會讓損失偏樂觀；子規則內 O 以外的決策點也給 v，但沒有證據（看可比較比例）。要在更多狀態評估少用的版本，需要遊戲端小比例隨機發放。

---

## 10. 策略表

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
| R7 | 快取是否涵蓋讀取範圍 | 前段或分析日不足 → 中止；只有後段不足 → 警告並截尾 |

成本：第一段每次只載入一個日檔；第二段每桶約 90 萬列，峰值約 3–4 倍；逐手狀態不落地。build 約 30 秒（兩天一起；檔案已在本機快取時）；7 組決策樹約 15 秒；傾向分數 5 折約 30 秒 CPU；bootstrap 1 分鐘內。

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
| `GP_dailyRate_BIG_WIN` | 0 | `GP_sessionMinutes` | 6（每手間隔 1 分鐘） |
| `GP_dailyRtp` | 1.4167 | `NGP_dailyTotalWinToBet` | 4.25（850 ÷ 200） |
| `GP_dailyRate_MEDIUM_PLUS` | 0.1667 | `GP_dailyNoWin_x_spins` | 1.2973 |
| `NGP_dailyNoWin_x_sessionLoss` | 0 | `GP_flag_dryLong`、`NGP_flag_coldLoss` | 0、0 |
| `GP_dailyRate_FREE_GAME` | 0 | `SY_balanceToBet` | 51.25（10,250 ÷ 200） |
| `GP_rtp`（目前不使用） | 1.4167 | | |

另外驗證：

- 第 6 手 prize 改為 0 或 100,000 → 第 6 手特徵全部不變。
- 美東日界：UTC 03:59 與 04:01（8 月）兩手 → 第二手 `GP_dailySpinCount` = 0；1 月以 UTC 05:00 為界。
- 美東月界：8/1 美東 00:01 的第一手 → `GP_rtp` 為 NaN（7 月不計入）。
- 月初基準：8/1–8/27 Σbet 1,000、Σprize 1,200，分析日第一手前 Σbet 600、Σprize 850 → `GP_rtp` = 1.2813；`GP_dailyRtp` 只算當天。
- 多條件：連續 120 手 NO_WIN、`SY_first_balance` 10,000、`SY_balance` 6,000 → 兩個旗標皆 1，`NGP_dailyNoWin_x_sessionLoss` = 0.4。
- BUY 手：第 3 手改 `game_mode = 19` → 第 6 手水池特徵不變；第 3 手 `SY_isBuyFeature` = 1。
- FreeGame：第 3 手 `game_mode = 3` → 第 6 手 `GP_dailyRate_FREE_GAME` = 1/6、第 3 手 = 0；第 6 手（當手）`game_mode = 3` → 第 6 手特徵全部不變。
- 新 session 第一手 `balance_before` 8,000 → 該 session 各手 `SY_first_balance` = 8,000，`session_spin_idx` 與 `GP_sessionMinutes` 從 0 起；跨午夜的 session，已玩分鐘從前一天的 session 起點算。

**Y 與決策點**

- 25 手後中斷 2 天：第 0 手 `churnNext20` = 0、第 20 手 = 1；中斷 2 小時：第 20 手 = 0。
- 50 手 session → 第 0、20、40 手三筆，只有第 40 手可能為 1。
- 窗口 17 手保護傘、3 手高倍數 → 抽入，arm = 保護傘；15 對 5 → 不抽入。
- session 結束 + 24 小時超過資料終點 → 整個 session 不抽入；資料夠長時三個決策點都抽入。當天 23:00 開始、隔天 12:30 才結束的 session → 不抽入（即使資料夠長）；11:30 結束 → 抽入。
- 範圍起點前的付費手、其免費列落在範圍內 → 贏分加回月初基準，`GP_rtp` 與讀取範圍從更早開始時相同。
- 月初基準：9/1 第一個付費手之前的免費卡（8/31 那一手的）不計入，之後的免費列與隔天檔的列計入。
- 同一天單獨建、或和前一天一起建 → 決策點與 `session_id` 完全相同；`session_id` = 20260818001、20260818002（前一天 23:50 開始的 session 歸 08-17）。各天的統計相加 = 整段統計。
- 前一天 23:50 開始、跨過午夜的 session 不出決策點（整段歸前一天）；當天 23:50 開始、跨過午夜的 session，午夜後的決策點也算當天、`decision_date` = 當天。`pre_hours` 與後段資料計入 session 序號與 Y。
- 範圍起點前 10 分鐘還有付費手 → 範圍內第一個 session 不抽決策點、之後的 session 照常；間隔 40 分鐘或範圍前沒玩 → 照常抽。範圍前檢查只取付費手、只取窗口內。
- 讀取範圍：訓練期間從 08-28 起、驗證期間 08-29、`post_hours` = 12 → 美東 08-27 18:00 至 08-31 12:00，讀 UTC 08-27 至 08-31，R7 通過；月初基準讀 UTC 08-01 至 08-27。只建 08-29 → 08-28 18:00 至 08-31 12:00。驗證期間 08-31 → 分析日本身超出快取（快取只到美東 08-31 20:00），R7 中止。
- 一起處理：訓練期間最後一天 23:50 開始、跨午夜的 session 只出現在訓練；驗證期間 09:00 開始的 session 只出現在驗證。
- 1,000 手隨機序列上，session 表算出的 Y 與逐手往後檢查 20 手完全相同。

**權重、標籤、分桶**

- 同一天 100 個決策點、cap 50 → 全部保留，每筆 `w_cap` = 0.5；30 個 → 每筆 1；Y 與特徵不變。
- 標籤（W = 202635）：有 202634 → 取 202634；最新為 202625 → 取 202625 的值；最新為 202630 補零列「流失」→ 流失；202635 以後的列不得使用；完全無紀錄 → 新手、無紀錄。
- 同一玩家只在同一桶；分桶前後總列數相同。

### 13.1a `test_store.py`

- 日期依連續區段分組；指紋只隨 build 相關設定改變（期間、`compute`、`tree` 不影響）。
- 某天：沒有檔案 → 還沒建；指紋不同 → 設定或程式已改；建置時用到的注單日檔列數、或標籤週的列數／最後 ETL 時間與現在的 manifest 不同 → 快取已重抓；都相同 → 可用。
- 讀指定日子的 dataset；缺日子 → 報錯。所有引用某 build 的日子都重建後，該 build 紀錄刪除。

### 13.2 `test_model.py`

- `model.yaml`：實驗的 X = fast ＋ tags[名稱]（不累加）；experiments 預設 8 組且順序固定；舊的 `slow`、不認得的 split／tags、重複欄位、不存在的欄位、`churnNext20`／`w_cap`／`arm`／`arm_ratio`／`SEG_`／索引 → 報錯。
- 編碼：缺值 → −1 ＋ `_isna`；one-hot 依訓練類別，未見過的類別 → 全 0；常數欄移除。
- 葉子規則可讀（`≤`／`>`、`= 類別`／`≠ 類別`），同欄位條件合併；葉子編號 `群/Lk` 由左到右。
- `sample_weight` = `w_cap`：流失列加權後葉子流失率上升。
- 訓練期間長樹、驗證期間預測與 AUC；樹的節點（左右子節點、提問、筆數）完整寫進 `tree_rules.json`；提問的讀法（缺值、= 最小值、整數門檻、one-hot、0／1）。
- split = value：群內 `TAG_value_combined` 不放入 X；未指定 groups 時訓練期間不足的群沿用 all；`splits.groups` 指定時只建模列出的群（各自建樹、不足門檻也建），其他群不建模、不評估。
- `categories` 白名單：未列出的類別 one-hot 全為 0。

### 13.3 `test_compare.py`

- 加權 AUC（大量同分）= sklearn `roc_auc_score`；bootstrap 依玩家重抽權重。
- 共同樣本 = 每個模型都有預測的決策點；相同模型的 AUC 差距 = 0、判定「無顯著差異」；基準本身標「基準」。
- 判定（較佳／較差／無顯著差異）。
- 信心分數：高 1 分、中 0.5 分、低與無法比較 0 分，依決策點權重加總；維持現狀的葉子不計入（全部維持現狀時不給分）；找不到信心的葉子當無法比較；合併後依合併對照算。

### 13.4 `test_evaluate.py`

- 9.3 驗收範例（raw／IPW 流失率；IPW 後 max_smd = 0）。
- 雙重穩健：結果模型錯（常數）但傾向分數對、或傾向分數錯（均勻）但結果模型對，加權留存率都等於子狀態標準化後的真值；bootstrap 點估計 = 直接計算的 DR。
- 共同範圍：加入只在少數狀態出現的版本會讓涵蓋 < `min_overlap_share` → 不比該版本；門檻放寬則多比；都不到門檻選涵蓋最多；沒有可行組合 → 不比較、不修剪；合併後重挑共同範圍；共同範圍不到門檻 → 信心最多低。
- 最高留存 arm：樣本不足的 arm 不當候選；信心高需最高留存 arm 平衡且顯著（次高未平衡也可）；最高留存 arm 未平衡或只有一個 arm → 信心低。
- 合併：子節點有無法比較的葉子 → 父節點合併（名稱 `L1–L2`）；一路合到根仍無法比較時停在根；全部可比較時不合併。
- 合併相近建議：兩個子葉子建議同一版本 → 合併，可一路併到根；建議不同但兩片都不是高信心 → 只在這棵樹併完相同建議後仍超過 `similar_merge_over_leaves` 片時合併（這一輪不再往上）；有一邊是高信心或無法比較就不併。
- 維持現狀：版本 a 佔 90% 且只有 a 樣本足夠 → `dominant_share` 0.8 時判維持現狀、0.9 時判低（要「超過」門檻）；維持現狀的葉子在「合併無法比較」時不合併；「合併相近建議」時當成版本 X，可和建議 X 的葉子（維持現狀或高／中／低）合併（2026-10-20）。
- 葉子內校準：平均偏 0.6 個標準差的 arm 校準後對齊到 ±0.05 內、判平衡、有效樣本變少；完全不重疊的 arm 對不齊、仍判未平衡。
- 高倍數展開成 M4、M5：`uplift` 相同、並列。
- bootstrap 聚合後的點估計 = 直接計算的 `ipw_churn`（含 `w_cap`）。
- 單一版本葉子：只有保護傘合格 → 清單 [M1]、信心低、`single_arm`。
- 價值分群：`SEG_value` = 決策週 W 之前最新一筆的 `value_combined_tag`；完全無紀錄 → 無紀錄群；不用當日下注。
- 群在訓練期間 15,000 筆 → 沿用 all。
- 各實驗的 W 逐筆相同；差距用同一組 Poisson 權重，區間含 0 → 「無顯著差異」。

### 13.5 `test_pages.py`

- 決策樹 SVG 含提問、是／否、葉子編號、最佳 arm、信心；合併後的節點畫成葉子（名稱 `L1–L2`）。
- 熱圖：樣本不足的格子為灰色、只標 n；只有一個 arm 樣本足夠的葉子標「無法比較」、不標 ★；只寫葉子編號、不寫規則。
- 熱圖有整體留存欄；決策樹方框寫留存率（不是流失率），葉子寫最佳版本與它的加權留存率；維持現狀的葉子寫「維持現狀_0」、不標 ★。
- `config.yaml`：`train_start` ≤ `train_end` < `valid_start` ≤ `valid_end`；快取與輸出資料夾由商戶／幣別／遊戲決定（不含日期）；舊的 `analysis_date`、`train_date`、`bet_records_dir`、`tag_dir` 報錯；輸出路徑 = `{商戶}_{幣別}_{遊戲}/train_{起}-{迄}_valid_{起}[-{迄}]`。
- summary：總覽表有 AUC 與信心分數（合併後）；尚未 evaluate 的實驗標「尚未 evaluate」；附評分依據。

---

## 14. 待確認事項

### 14.1 遊戲端規則

| 問題 | 預設處理 |
|---|---|
| 現行 `gameVersion` 分派規則 | 未知；以今日 RTP、手數等特徵配適傾向分數。取得規則後確認其變數都在 X 中（`safeModeMap` 可能相關） |
| `version_map`：5 個 `game_version` 對 4 個原版本 | 由使用者填寫，未填完時 build 停止。線索：`_0` 為每日開局主版本、`_1` 多在輸錢時、`_2` 多在 RTP 高或剛中 BIG_WIN 後 |
| 水池倍數區間是否計入免費局贏分 | 併入前一個付費手後計算 |
| 模組切換時機 | 遊戲端每手可切換；策略每 20 手決定一次 |
| 樣本 arm 的定義 | 目前 = 決策點之後 20 手的主版本（5.4）。但窗口內每手的版本會被窗口內前一手的開獎結果影響：2026-08-29 bucket 0 的抽入窗口中 7.9% 窗口內出現 BIG_WIN，這些窗口流失率 1.2%（其他 7.0%）、arm = `_2` 佔 58%（其他 40%），BIG_WIN 後 `_2` 持續約 66%。決策後的開獎結果同時影響 arm 與 Y，傾向分數校正不了。替代方案：arm = 決策點當手的 `game_version`（只依決策當下資訊分派，可不再篩主版本）；待決定，`GP_prevBigWin` 等分派變數先不加 |
| M3、M7 沒有原版本 | 列為 `untested_modules`；實作後加入 `deployable` 並需線上實驗 |

### 14.2 資料範圍

| 問題 | 預設處理 |
|---|---|
| bet_records 只涵蓋美東 07-31 20:00 至 08-31 20:00 | 不補抓可分析 8/2–8/30（8/30 後段差 4 小時，少量樣本截尾）；8/1 需補 UTC 07-31，8/31 需補 UTC 09-01、09-02 |
| 標籤只從 202619 起 | 更早玩過、之後都沒紀錄的玩家會被歸為新手；依「無紀錄」人數決定是否往前補抓 |
| 單日樣本是否足夠 | 訓練、驗證各用一天（2026-08-28 → 08-29，星期五 → 星期六）；依 `min_n_per_arm` 判定，不足時改用多天訓練（需支援多日合併） |
| 訓練期間與驗證期間的重疊 | 依使用者決定（2026-10-12）重疊歸訓練期間：跨午夜的 session 整段歸起點那天（0.1），驗證期間的決策點只來自驗證期間開始的 session。仍保留的時間重疊只有兩種，都不是同一筆資料出現在兩邊：① 訓練期間的 Y 要看之後 24 小時，會看到驗證期間的回訪；② 驗證期間的日水池特徵從 0 點起算，會計入前一晚跨午夜 session 在午夜後的手（與遊戲端一致）。完全隔開需在兩天之間空一天。上線時要等 Y 完整（訓練期間之後 24 小時）才能訓練 |

### 14.3 分群定義

| 問題 | 預設處理 |
|---|---|
| 價值分群 | 依 `value_combined_tag` 六類各自成群；前一週沒來的玩家取更早的最新一筆（可能是「本週無下注」） |
| 分群後樣本 | 群在訓練期間 < 10,000 筆沿用 all（2026-10-14 由 20,000 降低）。訓練期間 2026-08-28 實測：lifecycle 只有回流（11,166）沿用 all，其他五群自建樹；value 依 `splits.groups` 只建模低（7,663）、中（35,437）、高（138,463），三群各自建樹；無紀錄、本週無下注、手數過低不建模（實驗比較的共同樣本因此只含低／中／高）|

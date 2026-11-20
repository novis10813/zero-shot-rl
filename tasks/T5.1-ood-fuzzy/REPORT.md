# T5.1-ood-fuzzy：在 Walker OOD 動力學中以 ANFIS 尋找更好的 FB latent $z$

固定預訓練好的 FB 模型（Walker RND 660k，1 個 checkpoint），在 4 種偏移動力學中比較四種取得 task latent $z$ 的方法。$z$ 一律由 $z = \sqrt{d}\,\operatorname{normalize}(\mathbb{E}[\tilde{r}_\theta(s) B(s)])$ 計算，差別只在狀態權重 $\tilde{r}_\theta(s)$ 的來源與訓練環境。

## 1. 實驗設定

**OOD 動力學（各調低、調高一次）**

| 名稱 | 偏移 | 原值 |
|---|---|---|
| `friction_mild` | 地板與身體 geom 摩擦調低為 0.1 | 0.7 |
| `friction_high_mild` | 摩擦調高為 1.5 | 0.7 |
| `actuator_mild` | 所有關節 gear 調低為 76% | 100% |
| `actuator_high_mild` | 所有關節 gear 調高為 130% | 100% |

**四種方法**

| 方法 | 說明 |
|---|---|
| Frozen FB $z_r$ | 以原始 reward 推論 $z_r$，直接在 OOD 環境評估 |
| Clean ANFIS-z 直接遷移 | 在原環境以 REINFORCE 訓練 4 sensors 的 ANFIS-z，不再調整，直接在 OOD 環境評估 |
| OOD finetune，4 sensors | 同一個 4 sensors ANFIS（`torso_height`、`torso_upright`、`horizontal_velocity`、`angmomentum`，36 rules），在各 OOD 環境重新 finetune |
| OOD finetune，客製 sensors | 依偏移類型加入更多 sensors 後在各 OOD 環境 finetune。Friction：另加 `com_vz`、`qvel_right_ankle`、`qvel_left_ankle`（7 sensors，128 rules）。Actuator：另加 `qvel_right_hip`、`qvel_left_hip`、`qvel_right_knee`、`qvel_left_knee`（8 sensors，256 rules） |

**訓練與評估協定**

- Finetune：10 個 finetune seeds（`1000, 1010, ..., 1090`），每個 seed 使用互不重疊的訓練環境種子。每個任務 50 次 REINFORCE 迭代，每次 16 個成對擾動的 $z$（$\sigma=0.5$），learning rate $10^{-3}$，先以真實 reward warm-start 500 步。
- 評估：四個任務（`stand`、`walk`、`run`、`flip`）共用 20 個評估環境（種子 `5..24`），與 finetune 種子不重疊。每個任務分數為 20 個 episode 的 IQM，再對 10 個 finetune seeds 取平均。
- Finetune 方法每 10 次迭代評估一次（step 0、10、...、50）。依 Table 6 的做法，每個環境、每種方法各自取「十個 seeds 平均後 All-task IQM 最高」的 step 回報（「選定 step」欄）。挑選與回報用的是同一批評估環境，所以分數略偏樂觀。「第 50 步 All-task IQM」欄為最後一步、不挑選的分數。Clean ANFIS-z 直接遷移與 Frozen FB $z_r$ 沒有迭代，不挑選。
- All-task IQM 為四個任務分數的 25% trimmed mean（Table 6 of Jeen et al. 2024，`analysis.table6_aggregate`）。「相對 $z_r$」為該列 All-task IQM 減去同一環境 Frozen FB $z_r$ 的 All-task IQM。
- Frozen FB $z_r$ 不含隨機訓練，10 個 finetune seeds 的分數相同。

## 2. 結果

四環境平均列的四個任務欄位為四個環境該任務分數的平均，All-task IQM 欄位為四個環境各自 All-task IQM 的平均。資料來源：`data/summary.csv`（同內容的表格在 `data/summary.md`）。

| OOD 環境 | 方法 | 選定 step | stand | walk | run | flip | All-task IQM | 相對 $z_r$ | 第 50 步 All-task IQM |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `friction_mild` | Frozen FB $z_r$ | - | 189.8 | 321.0 | 118.1 | 187.3 | 188.5 | +0.0 | 188.5 |
| `friction_mild` | Clean ANFIS-z 直接遷移 | - | 270.9 | 380.3 | 146.1 | 279.1 | 275.0 | +86.5 | 275.0 |
| `friction_mild` | OOD finetune，4 sensors | 50 | **285.2** | 373.0 | 153.1 | 280.0 | **282.6** | **+94.0** | 282.6 |
| `friction_mild` | OOD finetune，客製 sensors | 50 | 242.7 | **405.6** | **156.4** | 296.9 | 269.8 | +81.2 | 269.8 |
| `friction_high_mild` | Frozen FB $z_r$ | - | 500.7 | 208.5 | 82.6 | 345.4 | 277.0 | +0.0 | 277.0 |
| `friction_high_mild` | Clean ANFIS-z 直接遷移 | - | 573.3 | 257.3 | **154.0** | **369.0** | 313.2 | +36.2 | 313.2 |
| `friction_high_mild` | OOD finetune，4 sensors | 40 | **573.6** | **332.8** | 141.5 | 363.4 | **348.1** | **+71.2** | 335.9 |
| `friction_high_mild` | OOD finetune，客製 sensors | 30 | 553.8 | 322.7 | 136.6 | 364.2 | 343.5 | +66.5 | 342.7 |
| `actuator_mild` | Frozen FB $z_r$ | - | 442.6 | 122.7 | 99.0 | 239.0 | 180.9 | +0.0 | 180.9 |
| `actuator_mild` | Clean ANFIS-z 直接遷移 | - | 444.4 | 173.9 | 116.3 | 304.4 | 239.2 | +58.3 | 239.2 |
| `actuator_mild` | OOD finetune，4 sensors | 40 | **448.2** | **186.1** | 136.0 | 302.2 | 244.2 | +63.3 | 242.9 |
| `actuator_mild` | OOD finetune，客製 sensors | 50 | 439.8 | 181.2 | **154.7** | 319.1 | **250.2** | **+69.3** | 250.2 |
| `actuator_high_mild` | Frozen FB $z_r$ | - | 681.9 | 617.2 | 251.0 | 350.3 | 483.8 | +0.0 | 483.8 |
| `actuator_high_mild` | Clean ANFIS-z 直接遷移 | - | 694.1 | 628.4 | 238.6 | 421.6 | 525.0 | +41.3 | 525.0 |
| `actuator_high_mild` | OOD finetune，4 sensors | 50 | 700.2 | 619.9 | 242.1 | 423.0 | 521.4 | +37.7 | 521.4 |
| `actuator_high_mild` | OOD finetune，客製 sensors | 50 | **711.3** | **637.0** | **252.2** | **435.1** | **536.0** | **+52.3** | 536.0 |
| 四環境平均 | Frozen FB $z_r$ | - | 453.7 | 317.4 | 137.7 | 280.5 | 282.5 | +0.0 | 282.5 |
| 四環境平均 | Clean ANFIS-z 直接遷移 | - | 495.7 | 360.0 | 163.7 | 343.5 | 338.1 | +55.6 | 338.1 |
| 四環境平均 | OOD finetune，4 sensors | - | **501.8** | 377.9 | 168.2 | 342.1 | 349.1 | +66.5 | 345.7 |
| 四環境平均 | OOD finetune，客製 sensors | - | 486.9 | **386.6** | **175.0** | **353.8** | **349.9** | **+67.3** | 349.7 |

粗體為同一環境、同一欄位中四種方法的最高值（相對 $z_r$ 欄不含 $z_r$ 自身）。

原環境參考值（同一 checkpoint、同樣 20 個評估環境，`data/inputs/frozen_fb/clean/` 與 `data/inputs/clean/baseline/`）：Frozen FB $z_r$ 的 All-task IQM 為 363.5，4 sensors ANFIS-z 在選定 step（40）為 415.6，第 50 步為 410.7。

## 3. 觀察

- **ANFIS-z 在四個 OOD 環境都高於 $z_r$。** 即使不在 OOD 環境重新訓練，原環境訓練的 ANFIS-z 也使 All-task IQM 提高 36.2 到 86.5。
- **OOD finetune 對 4 sensors ANFIS 的額外增益有限。** 與直接遷移相比，四個環境的差為 +7.6、+34.9、+5.1、−3.6，平均 +11.0。增益集中在 `friction_high_mild`，`actuator_high_mild` 略低於直接遷移。
- **客製 sensors 對 actuator 偏移有幫助，對 friction 偏移沒有。** 與 4 sensors finetune 相比，`actuator_mild`、`actuator_high_mild` 分別 +6.0、+14.7，`friction_mild`、`friction_high_mild` 分別 −12.8、−4.6。四環境平均幾乎相同（349.9 對 349.1），差距小於各環境間的差異。
- **`friction_mild` 的客製 sensors 在 `walk`、`run`、`flip` 都高於 4 sensors，`stand` 低 42.5。** 整體 All-task IQM 因 `stand` 下降而低於 4 sensors。
- **`run` 在 `friction_high_mild` 沒有受益於 OOD finetune。** 4 sensors 與客製 sensors finetune 的 `run` 分數（141.5、136.6）都低於直接遷移（154.0）。

## 4. 限制

- 結果為單一 FB checkpoint，只有一種偏移強度（mild）與單一 REINFORCE 設定，沒有跨 FB seed 的結果。
- 表中沒有信賴區間。客製 sensors 與 4 sensors 之間 4.6 到 14.7 分的差距，以及 OOD finetune 與直接遷移之間的差距，可能落在 finetune seeds 之間的變異內，這份資料尚未檢定。選定 step 的做法也會讓 finetune 方法略為有利。
- 客製 sensors 同時改變 sensor 數量與規則數（128 或 256 對 36），無法區分增益來自哪些 sensors 或來自較多規則。
- 客製 sensors 依偏移類型各選一組，沒有與其他 sensor 組合比較。
- 只測 Walker 與 friction、actuator gear 兩類偏移。

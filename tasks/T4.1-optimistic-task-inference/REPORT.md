# T4.1 結果：OpTI-BFM 在本 repo 的 FB 上，3 到 5 個 episode 內追上 Oracle；在 friction 改變的環境中表現超過 Oracle

本任務重現 Rupf et al. (2025), "Optimistic Task Inference for Behavior Foundation Models"（arXiv 2510.20264）。
這個方法在測試時不使用有 reward 標記的資料集，只靠與環境互動時拿到的 reward 推論 task。
做法是維護一個 reward 係數 $w$ 的最小平方估計與信賴橢球，每一步依此選擇 FB 的 latent。
評估對象是 T04 中標準 FB 的 3 個訓練 seed（Walker，RND 100k），四個 task 為 stand、walk、run、flip。
比較的方法：

- **Oracle**：使用離線推論的 $z_r$，即本 repo 的標準做法，用 10k 筆有 reward 的資料推出。
- **Random**：每一步從球面均勻抽一個 latent。
- **OpTI-BFM**：UCB 規則。
- **OpTI-BFM-TS**：Thompson sampling 規則。

設定：

- 每個 task 跑 10 個 run，每個 run 連續 10 個 1000 步的 episode。
- run 內的估計在 episode 之間保留，換 task 時重設。
- 同一個 run 在各方法下的初始狀態相同。

指標：

- 表中的 return 是 10 個 run 的平均，再對 3 個 seed 平均。all-task 分數是四個 task 平均值的 IQM，依 `analysis.table6_aggregate`。
- 圖中的相對表現是方法的 return 除以同一 seed、同一 task、同一環境下 Oracle 的平均 return，再對四個 task 平均（與論文相同），陰影為 3 個 seed 的最小值到最大值。

## 超參數：TS 用 σ = 0.1，UCB 用 β = 0.001

| 方法 | 參數 | 10 個 episode 的累積 return |
|---|---|---|
| OpTI-BFM | β = 1 | 2962 |
| OpTI-BFM | β = 0.1 | 3689 |
| OpTI-BFM | β = 0.001 | 3895 |
| OpTI-BFM-TS | σ = 0.1 | 4148 |
| OpTI-BFM-TS | σ = 0.01 | 3933 |
| OpTI-BFM-TS | σ = 0.001 | 4091 |
| OpTI-BFM-TS | σ = 0.0001 | 3966 |

累積 return 是一個 run 的 10 個 episode return 加總，再對四個 task 和三個 seed 平均。
依論文附錄 F，每個方法取累積 return 最高的參數。以下所有結果都用這兩個值。

- **TS 對 σ 不敏感。** 四個值的差距在 6% 以內。
- **UCB 的 β 越小越好。** β = 0.001 時信賴橢球的半徑只有 0.002，128 個候選點幾乎都等於目前的估計 $\hat z$，所以 UCB 實際上接近 greedy。作者 README 的預設值 β = 0.1 少 5%，β = 1 少 24%。

## 線上推論在 3 到 5 個 episode 內追上 Oracle

| Episode | Oracle | Random | OpTI-BFM | OpTI-BFM-TS |
|---|---|---|---|---|
| 1 | 409 | 91 | 209 | 320 |
| 2 | 385 | 94 | 323 | 415 |
| 3 | 427 | 95 | 386 | 421 |
| 5 | 404 | 95 | 410 | 435 |
| 10 | 382 | 92 | 407 | 439 |

數值為 all-task IQM。Oracle 每個 episode 都用同一個 $z_r$，episode 之間的差異來自初始狀態不同。

| Task | Oracle | OpTI-BFM ep. 1 | OpTI-BFM ep. 6–10 | OpTI-BFM-TS ep. 1 | OpTI-BFM-TS ep. 6–10 |
|---|---|---|---|---|---|
| stand | 628 | 431 | 653 | 515 | 660 |
| walk | 551 | 290 | 550 | 417 | 560 |
| run | 201 | 102 | 189 | 109 | 201 |
| flip | 251 | 129 | 317 | 223 | 306 |

Oracle 欄為 episode 6 到 10 的平均。

- **兩種規則都追上 Oracle。** OpTI-BFM-TS 在第 2 個 episode 就達到 Oracle 的水準，OpTI-BFM（β = 0.001）要到第 4 個 episode。
- **stand、walk、run 與 Oracle 相差 6% 以內。** flip 則高出 22% 到 26%，這表示從離線資料推出的 $z_r$ 對 flip 不是這個 FB 的最佳 latent。
- **線上推出的方向與 $z_r$ 不同。** 第 10 個 episode 結束時，估計 $\hat z$ 與 $z_r$ 的 cosine 在 stand 為 0.90 到 0.92，在 flip 為 0.64 到 0.72。
- **Random 只有 Oracle 的 0.27 倍左右。**

![相對表現](figures/main_relative.png)
![各 task 的 episode return](figures/main_per_task.png)

## 每個 episode 只換一次 latent 會慢 3 到 5 個 episode，但最後一樣追上

| Episode | 1 | 2 | 3 | 5 | 10 | 20 | 50 |
|---|---|---|---|---|---|---|---|
| OpTI-BFM | 209 | 323 | 386 | 410 | 407 | | |
| OpTI-BFM-EP | 57 | 88 | 144 | 369 | 405 | 359 | 440 |
| OpTI-BFM-TS | 320 | 415 | 421 | 435 | 439 | | |
| OpTI-BFM-TS-EP | 73 | 121 | 208 | 374 | 426 | 416 | 459 |

數值為 all-task IQM。-EP 變體只在 episode 開始時選一次 latent，跑 50 個 episode。每步更新的版本只跑 10 個 episode。

- **-EP 的前 3 個 episode 遠低於每步更新的版本。** 例如第 1 個 episode 為 57 對 209、73 對 320。
- **-EP 在第 5 到 10 個 episode 追上。** 第 10 到 50 個 episode，OpTI-BFM-EP 的相對表現在 0.94 到 1.13 之間，OpTI-BFM-TS-EP 在 1.01 到 1.16 之間。這和論文第 5.3 節的結論相同：每步更新收斂較快，episodic 版本最後達到相同水準。

![每步更新與 episodic 更新](figures/episodic.png)

## 推論資料的品質：TS 收集的資料與 RND 一樣有效，近乎 greedy 的 UCB 最差

| 來源 | n = 100 | 200 | 500 | 1000 | 2000 | 5000 | 10000 |
|---|---|---|---|---|---|---|---|
| OpTI-BFM | 0.29 | 0.40 | 0.51 | 0.63 | 0.82 | 1.00 | 1.07 |
| OpTI-BFM-TS | 0.38 | 0.56 | 0.88 | 0.95 | 0.98 | 1.08 | 1.06 |
| RND | 0.45 | 0.53 | 0.83 | 0.93 | 0.97 | 0.97 | 0.98 |
| Random | 0.33 | 0.49 | 0.63 | 0.77 | 0.86 | 0.94 | 0.93 |

對每個來源，取前 n 筆 transition，以無正則化的最小平方推一個固定的 $z$。
接著評估這個 $z$ 的 policy，表中數值是它的相對表現（每個 n 跑 20 個 episode，對四個 task 和三個 seed 平均）。
資料來源：

- OpTI-BFM、OpTI-BFM-TS、Random：取自上面主實驗的 run。
- RND：從 RND 資料集隨機抽整條 episode。

- **TS 與 RND 相當。** n ≥ 500 時兩者都達到 Oracle 的 0.83 倍以上，n = 5000 時 TS 為 1.08、RND 為 0.97。
- **OpTI-BFM 在 n ≤ 2000 時最差，甚至低於 Random。** 它要到 n = 5000 才達到 1.0。
- **與論文不同。** 論文第 5.2 節中 OpTI-BFM 與 TS 都比 RND 更省資料。可能的原因是 β = 0.001 使 UCB 幾乎不探索：前幾個 episode 的資料集中在少數行為上，對 50 維的無正則化最小平方不夠多樣。

![data efficiency](figures/data_efficiency.png)

## 動力學改變後，線上推論在 friction 改變時超過 Oracle，在 actuator gear 降低時沒有

| 動力學 | Oracle | OpTI-BFM | OpTI-BFM-TS | Random | OpTI-BFM ÷ Oracle | OpTI-BFM-TS ÷ Oracle |
|---|---|---|---|---|---|---|
| 原始 | 401 | 434 | 433 | 91 | 1.08 | 1.08 |
| friction 0.1 | 346 | 373 | 434 | 79 | 1.08 | 1.25 |
| friction 0.02 | 285 | 329 | 347 | 77 | 1.15 | 1.22 |
| actuator gear 76% | 323 | 334 | 348 | 61 | 1.03 | 1.08 |
| actuator gear 62% | 256 | 225 | 236 | 44 | 0.88 | 0.92 |

數值為 episode 6 到 10 平均 return 的 all-task IQM。
四種 shift 沿用 T3.2 的設定，reward 函數不變。
Oracle 在改變後的環境中執行，但 $z_r$ 仍是從原始資料推出的。OpTI-BFM 和 OpTI-BFM-TS 在改變後的環境中只靠 reward 重新推論。

| 動力學 | 方法 | stand | walk | run | flip |
|---|---|---|---|---|---|
| friction 0.1 | Oracle | 474 | 483 | 148 | 217 |
| friction 0.1 | OpTI-BFM-TS | 550 | 536 | 164 | 332 |
| friction 0.02 | Oracle | 475 | 367 | 103 | 203 |
| friction 0.02 | OpTI-BFM-TS | 517 | 358 | 136 | 337 |
| actuator gear 62% | Oracle | 372 | 342 | 112 | 169 |
| actuator gear 62% | OpTI-BFM-TS | 388 | 334 | 86 | 139 |

- **friction 改變時，OpTI-BFM-TS 超過 Oracle 22% 到 25%，大於原始環境的 8%。**
  - friction 0.1 時四個 task 都超過 Oracle。
  - friction 0.02 時 stand、run、flip 超過，walk 相差 3%。
  - flip 在三種環境都高出 Oracle 很多，是比值偏高的主因之一。
- **actuator gear 降到 62% 時，線上推論低於 Oracle。** run 和 flip 分別低 23% 和 18%。gear 76% 時大致持平。
- **改變動力學沒有讓推出的方向離 $z_r$ 更遠。** 第 10 個 episode 時，$\hat z$ 與 $z_r$ 的 cosine 在每種動力學下都和原始環境相近（stand 0.90 到 0.93，flip 0.63 到 0.78）。
  因此 friction 下的增益，看不出來自「為新的動力學找到不同的 latent」，還是單純延續原始環境中 flip 已有的優勢。
- **只有 friction 改變時，三個 seed 都不低於 Oracle。**

  | 動力學 | OpTI-BFM（seed 0 / 1 / 2） | OpTI-BFM-TS（seed 0 / 1 / 2） |
  |---|---|---|
  | 原始 | 1.10 / 0.90 / 1.25 | 1.10 / 0.93 / 1.23 |
  | friction 0.1 | 1.11 / 0.96 / 1.47 | 1.21 / 1.00 / 1.70 |
  | friction 0.02 | 1.17 / 1.00 / 1.56 | 1.30 / 1.07 / 1.54 |
  | actuator gear 76% | 1.10 / 0.77 / 1.19 | 1.24 / 0.80 / 1.16 |
  | actuator gear 62% | 1.11 / 0.68 / 1.12 | 1.12 / 0.70 / 1.10 |

  數值為 episode 6 到 10 的相對表現（四個 task 平均）。
  在 actuator 改變時，seed 1 低於 Oracle 20% 到 32%，seed 0、2 則高出 10% 到 24%。
  因此 actuator gear 62% 的 IQM 比值（0.88、0.92）主要反映 seed 1。

![動力學改變下的相對表現](figures/dynamics.png)

## 與論文的比較

論文的數字來自 Rupf et al. (2025) 第 5 節與附錄 C。論文的 FB 在完整的 RND 資料上訓練 2M 步，本 repo 的 FB 只用 100k 筆、1M 步，因此只比較結論，不比較 return 的絕對值。

| 論文的結論 | 本任務 |
|---|---|
| OpTI-BFM 在 5 個 episode 內於所有 task 追上 Oracle（第 5.1 節） | 成立：OpTI-BFM 在第 4 個 episode、OpTI-BFM-TS 在第 2 個 episode 追上 |
| UCB 與 TS 只在 Cheetah 有明顯差距 | 在 Walker 上，TS 前 3 個 episode 較快，之後相同 |
| OpTI-BFM 與 TS 收集的資料比 RND 和 Random 更省資料（第 5.2 節） | 只有 TS 與 RND 相當。β = 0.001 的 OpTI-BFM 在 n ≤ 2000 時最差 |
| -EP 變體較慢，最後達到相同水準（第 5.3 節） | 成立 |
| 對超參數穩健（附錄 F） | TS 成立。UCB 的 β 從 0.001 到 1 會差 24% |

## 限制

- 只有 Walker 一個 domain，也只有 3 個 FB seed，seed 間的範圍很寬。
- 超參數是在報告結果的同一批 run 上選的，與論文做法相同。
- Oracle 的 $z_r$ 用本 repo 的離線推論（10k 筆、$\mathbb E[B r]$），不是論文的 50k 筆最小平方。flip 上線上推論超過 Oracle，表示這個 $z_r$ 不是 flip 的最佳 latent。
- 取樣依論文的 $V^{-1}$ 實作，與作者程式不同（見 README）。
- data efficiency 的 RND 來源取自完整的 10M 資料，FB 只用其中 100k 筆訓練。
- 每個 task 開始時 belief 重設，所以每個 task 都要付前幾個 episode 的探索成本。沒有測試 warm-start、D-gap thresholding、LoLA 和 non-stationary reward。

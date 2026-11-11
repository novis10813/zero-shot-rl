# T4.1.3.0 結果：固定 latent 下對 FB 做線上價值修正

## 設定

在 frozen FB 上（F、B、actor、$z$ 全部凍結，T04 的 FB seed 0 checkpoint，walk 任務），用任務 reward 線上訓練一個修正 critic $Q_\phi$，並在 $a_0=\pi(s,z)$ 附近的 32 個候選動作（$a_0$ 加上標準差 0.2 的高斯雜訊，截到 $[-1,1]$）中挑 $Q_{\text{adapt}}$ 最高者。

- **Residual 版本**：$Q_{\text{adapt}}(s,a)=\text{scale}\cdot\min_k F_k^\top z + Q_\phi(s,a)$，其中 $\text{scale}=\lVert\mathbb{E}[rB(s)]\rVert/\sqrt d$ 把 $F^\top z$ 換算到 reward 單位。$Q_\phi$ 是 2 個 critic，每個是 $30\to64\to64\to1$ 的 MLP，最後一層初始化為 0，所以起點等於 FB 自己的價值。
- **訓練**：100 個 episode（每個 1000 步），每步一次更新（batch 512、學習率 $10^{-4}$、$\tau=0.01$、$\gamma=0.98$、緩衝區 1000 筆後開始）。訓練動作是分數最高的候選，另有 30% 機率改選均勻隨機的候選。目標值為 $r+\gamma\max_i\min_k Q_{\text{adapt},k}^{\text{target}}(s',a'_i)$。
- **候選半徑**：候選與 $a_0$ 的 L2 距離限制在 0.2 以內，或不限制。
- **評估**：訓練前與每 10 個 episode，固定 critic，以分數最高的候選執行，在 5 個測試環境（seed 0 到 4）各跑一個 episode，報告平均 return。
- **環境**：原本的 Walker（Clean）與 T3.2 的 Actuator Mild、Actuator Strong。
- **Frozen FB**：同一批測試 episode 上，FB actor 的確定性動作。
- **Search**：不訓練，只用 FB 自己的價值 $\text{scale}\cdot\min_kF_k^\top z$ 在同樣的 32 個候選中取最大者，用 20 個測試 episode 評估。

## Frozen FB 的 return

Frozen FB 的值隨評估方式不同。T04 是該報告自己的評估（IQM，z 的推論流程不同）。Actuator Mild 與 Strong 沒有 T04 的值。

| 環境 | 5 個測試 episode（與訓練曲線同一批） | 20 個測試 episode（與 Search 同一批） | T04 的評估 |
| --- | --- | --- | --- |
| Clean | 564.4 | 547.8 | 643.6 |
| Actuator Mild | 435.3 | 536.5 | — |
| Actuator Strong | 383.0 | 428.5 | — |

## 訓練曲線

每一格是該 checkpoint 的 5 個測試 episode 的平均 return，訓練前（0）是 critic 尚未訓練時的候選選擇。Frozen FB 是同一批 5 個測試 episode 的值。

| 訓練 episode 數 | Clean（frozen FB 564.4）半徑 0.2 | Clean（frozen FB 564.4）不限半徑 | Actuator Mild（frozen FB 435.3）半徑 0.2 | Actuator Mild（frozen FB 435.3）不限半徑 | Actuator Strong（frozen FB 383.0）半徑 0.2 | Actuator Strong（frozen FB 383.0）不限半徑 |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 595.5 | 577.5 | 580.5 | 491.2 | 411.9 | 355.7 |
| 10 | 575.8 | 615.6 | 535.3 | 543.7 | 376.4 | 299.8 |
| 20 | 545.2 | 562.0 | 524.6 | 555.8 | 442.8 | 381.5 |
| 30 | 615.6 | 596.8 | 550.2 | 559.8 | 380.1 | 493.2 |
| 40 | 568.5 | 686.7 | 550.3 | 513.1 | 482.9 | 499.2 |
| 50 | 652.3 | 560.4 | 460.2 | 578.8 | 397.0 | 521.1 |
| 60 | 608.1 | 633.6 | 581.0 | 607.2 | 474.5 | 432.3 |
| 70 | 617.0 | 633.6 | 523.7 | 583.1 | 282.6 | 513.8 |
| 80 | 641.9 | 541.1 | 551.4 | 597.3 | 522.7 | 480.5 |
| 90 | 540.5 | 593.7 | 558.2 | 531.2 | 489.7 | 507.2 |
| 100 | 650.5 | 669.4 | 523.5 | 650.0 | 412.0 | 508.7 |

訓練前期（ep0–20）與後期（ep70–100）的 checkpoint 平均如下，後期差是不限半徑減半徑 0.2 的整數差。

| 環境 | frozen FB | 半徑 0.2：ep0–20 / 70–100 | 不限半徑：ep0–20 / 70–100 | 後期差（不限 − 半徑） |
| --- | --- | --- | --- | --- |
| Clean | 564.4 | 572 / 612 | 585 / 609 | -3 |
| Actuator Mild | 435.3 | 547 / 539 | 530 / 590 | +51 |
| Actuator Strong | 383.0 | 410 / 427 | 346 / 503 | +76 |

## 只用 frozen FB 的動作搜尋

| 環境 | Frozen FB | Search，不限半徑 | Search，半徑 0.2 |
| --- | --- | --- | --- |
| Clean | 547.8 | 573.1 (+25.3) | 574.9 (+27.1) |
| Actuator Mild | 536.5 | 421.9 (-114.7) | 468.2 (-68.3) |
| Actuator Strong | 428.5 | 353.6 (-74.9) | 344.7 (-83.8) |

括號內是與同一表格中 frozen FB 的差。

## 觀察

- 在 Actuator Mild 與 Strong，訓練曲線後期的平均高於 5 個 episode 的 frozen FB（Mild 435.3、Strong 383.0），Clean 的後期平均也高於其 5 個 episode 的 frozen FB（564.4）。但 20 個 episode 的 frozen FB 在 Mild 是 536.5，比 5 個 episode 的值高約 101，Strong 高約 46，Clean 低約 17。所以曲線與 frozen FB 的差距取決於用哪個 frozen 值，在 Mild 幾乎消失。
- Search（不訓練）在 Clean 比 frozen FB 高約 25，在兩個偏移環境低 68 到 115。FB 自己的價值在偏移環境不能可靠地排序動作。
- 曲線的第 0 個 checkpoint 尚未訓練。Mild 與 Strong 在半徑 0.2 是 580.5 與 411.9，高於 5 個 episode 的 frozen FB（435.3、383.0）。不限半徑是 491.2 與 355.7，Mild 高於、Strong 低於 frozen FB。
- 半徑 0.2 與不限半徑的後期差在 Clean 是 −3、Mild 是 +51、Strong 是 +76。相鄰 checkpoint 的跳動與這個量級相當，例如 Strong 半徑 0.2 在第 70 與第 80 個 episode 是 282.6 與 522.7，Clean 不限半徑在第 40 與第 50 個 episode 是 686.7 與 560.4。

## 限制

- 只有 1 個 FB seed 與 1 次訓練，沒有 seed 間的變異。
- 訓練曲線每個點只有 5 個測試 episode。同一個 checkpoint 的 5 個 episode 的 return 可以差到 300 以上（例如某點的 5 個值在 362 到 729 之間）。
- 同一組 seed 的測試在不同的 episode 數（5 與 20）下得到不同的 frozen FB 值，表示 5 個與 20 個 episode 的結果不能直接互相比較。原因沒有查證。
- 曲線與 Search 兩組用的測試 episode 數不同，所以兩組的 frozen FB 不同。
- 評估只涵蓋 100 個訓練 episode、hidden 64 與兩個半徑設定，沒有調整其他超參數。

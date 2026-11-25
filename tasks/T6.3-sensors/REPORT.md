# T6.3-sensors：ANFIS 規則字典的秩與 sensor 數量

在固定的 FB 模型下，ANFIS 能產生的任務潛向量 $z$ 全部落在一個字典矩陣 $\mathbf{Z}_{\text{dict}}$ 的欄空間裡。本報告比較 4、8、37 個 sensors 時這個欄空間的大小與數值條件，判斷增加 sensors 會不會讓 ANFIS 表示更多任務。

結論：在 fuzzy 規則結構下，4 個 sensors 已經能表示所有 $z \in \mathbb{R}^{50}$。增加到 8 個 sensors 不擴大表示範圍，只把字典的條件數從 2137 降到 258。37 個 sensors 的線性 reward 則表示不完全，四個任務的 $z_r$ 殘差在 3.5% 到 17.4% 之間。

## 1. 方法

membership 參數固定、輸出層為 identity 時，first_order ANFIS 的 reward 對 consequent 參數是線性的：

$$r(s) = \sum_{k=1}^{K} \bar{w}_k(s)\,\big(W_k \tilde{x}(s) + b_k\big) = \Psi(s)^\top \theta$$

其中 $\bar{w}_k$ 是正規化後的規則觸發強度，$\tilde{x}$ 是 z-score 後的 sensors。每條規則貢獻 $D+1$ 個基底函數 $\bar{w}_k(s)$ 與 $\bar{w}_k(s)\tilde{x}_i(s)$。FB 的任務潛向量因此是

$$z \propto \mathbb{E}_\rho\big[B(s)\,r(s)\big] = \mathbf{Z}_{\text{dict}}\,\theta, \qquad \mathbf{Z}_{\text{dict}} = \frac{1}{N} B^\top \Psi \in \mathbb{R}^{50 \times K(D+1)}$$

最後的 $\sqrt{50}$ 長度正規化不改變方向，所以 $\mathbf{Z}_{\text{dict}}$ 的欄空間就是 ANFIS 能到達的 $z$ 方向集合。

**設定**

- FB 模型：Walker RND 660k（與 T6.1、T6.2 相同），$d_z = 50$。
- 狀態分布 $\rho$：從 Walker RND 離線資料集抽 100,000 筆 transitions，再取 10,000 個狀態（seed 42），與 T6.2 的 `cmaes_anfis` 設定相同。分析不做 rollout，結果與 OOD 動力學無關。
- membership 參數：`init_from_data` 的分位數初始化，與 T6.2 的 ANFIS 起點相同。
- 數值秩：$\sigma_i > 10^{-3}\,\sigma_1$ 的奇異值個數。
- 任務殘差：stand、walk、run、flip 的 $z_r$ 對 $\mathbf{Z}_{\text{dict}}$ 做最小平方擬合後的相對殘差 $\|\mathbf{Z}_{\text{dict}}\theta^{\ast} - z_r\| / \|z_r\|$。

**Sensor 組合**

| 組合 | Sensors | 每個 sensor 的 terms | 規則數 $K$ | 字典欄數 |
|---|---|---|---:|---:|
| 4 sensors | `torso_height`、`torso_upright`、`horizontal_velocity`、`angmomentum` | 3、2、3、2 | 36 | 180 |
| 8 sensors | 上述 4 個加左右 hip、knee 角速度（T6.1 actuator 組） | 2 | 256 | 2,304 |
| 37 sensors | Walker 全部 sensors | 1 | 1 | 38 |

37 個 sensors 每個只用 1 個 term，等於 sensors 的線性 reward。Cartesian grid 用 2 個 terms 會有 $2^{37}$ 條規則，無法建立。

## 2. 結果

資料來源：`data/zdict_rank.csv` 與 `data/zdict_spectra.npz`。

| 組合 | 數值秩 | 有效秩 | 條件數 | stand 殘差 | walk 殘差 | run 殘差 | flip 殘差 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 4 sensors | 45 | 14.3 | 2,137 | 0 | 0 | 0 | 0 |
| 8 sensors | 50 | 17.5 | 258 | 0 | 0 | 0 | 0 |
| 37 sensors（線性） | 36 | 20.3 | $4.4 \times 10^{16}$ | 0.035 | 0.174 | 0.094 | 0.130 |

殘差為 0 的格子實際值都小於 $10^{-14}$。有效秩是 $\exp(-\sum_i p_i \log p_i)$，其中 $p_i = \sigma_i / \sum_j \sigma_j$。

## 3. 解讀

**4 sensors 能表示所有 $z$，但有 5 個方向很難到達。** 50 個奇異值都不為 0，四個任務的殘差都是 0。數值秩只有 45，是因為最小的 5 個奇異值落在 $4.7 \times 10^{-4}\sigma_1$ 到 $9.1 \times 10^{-4}\sigma_1$ 之間。要在這些方向上移動相同距離，consequent 參數的變化量必須是最強方向的約 1100 到 2100 倍。

**8 sensors 改善數值條件，不擴大表示範圍。** 它和 4 sensors 一樣能表示整個 $\mathbb{R}^{50}$，條件數低約 8 倍，最小奇異值是 $3.9 \times 10^{-3}\sigma_1$。代價是參數從約 200 維增加到 2,300 維以上。

**37 sensors 的線性 reward 不足以表示任務。** 38 個欄位中有兩個奇異值低於 float32 精度（$1.6 \times 10^{-8}\sigma_1$ 與 $2.3 \times 10^{-17}\sigma_1$），表示有 sensors 是其他 sensors 的線性組合，欄空間只有 36 維。四個任務都有殘差，walk 最大（17.4%）。增加 sensors 不能取代 fuzzy 規則提供的非線性。

**對 sensors 搜尋實驗的意義。** 在 fuzzy 規則結構下，sensors 數量不影響能表示哪些 $z$。之後 CMA-ES 比較 4 和 8 sensors 時，結果反映的是「條件數改善」和「搜尋維度增加約 10 倍」之間的取捨。37 sensors 線性版的上限本來就比較低，它表現較差不能用來說明 sensors 越多越好。

## 4. 限制

- membership 參數取初始化值。T6.2 的 CMA-ES 使用 `param_scope: all`，會同時調整 membership，字典在搜尋過程中會改變。本分析只描述搜尋起點。
- 線性關係只在 identity 輸出層下成立。使用 softplus 等輸出層時，可到達的集合不再是欄空間。
- 能表示某個 $z$ 不代表搜尋找得到它。條件數只是搜尋難度的一個指標，實際效果要由 CMA-ES 實驗確認。
- 8 sensors 只測了 T6.1 actuator 組這一種組合，其他 8 個 sensors 的組合可能有不同的條件數。
- 只用了一個 FB 模型與一組 10,000 個狀態，沒有估計抽樣變異。

# T6.4-fuzzy-add-rule：ANFIS 的 sensors 之間的交互作用是必要的嗎？

## 背景

在前面的 Sensors, More Sensors 實驗（T6.2、T6.3）中，我們可以發現 8 sensor 的 fuzzy system 效果普遍稍微略高於 4 sensor 的 ANFIS。但是我們仍無法分出是**交互作用**的功勞，還是**只要基底函數夠多**就好。那這樣 sensors 之間的交互作用是不是必要的？

## 方法

因此在這個實驗裡面，我會使用基於 Additive 的 Fuzzy System，測試最後的結果。

**加法規則 ANFIS**：`rule_mode: additive`，每個 sensor $i$ 的第 $m$ 個 linguistic term 獨立成為一條規則，規則數 $K = \sum_i M_i$（Cartesian product 為 $\prod_i M_i$，37 個 sensors 無法建立）。`normalize_firing: false`，觸發權重為未正規化的 membership 值 $w_{i,m}(s) = \mu_{i,m}(\tilde{x}_i(s))$（$\tilde{x}$ 為 z-score 後的 sensors）。兩種 consequent：

- `zero_order`：$r(s) = \sum_{i,m} b_{i,m}\,w_{i,m}(s)$，只有各 sensor 的 1D 函數相加，沒有 sensors 之間的交互作用（等價單層 RBF-KAN）。
- `first_order`：$r(s) = \sum_{i,m} w_{i,m}(s)\,(W_{i,m}\tilde{x}(s) + b_{i,m})$，每條規則含 $\tilde{x}$ 的線性項，帶入二階交互作用 $\mu_{i,m}(x_i)\,x_j$。

**條件**

- `all37_first`：37 個 sensors、`first_order`（74 條規則，2,960 參數）。
- `8_first`：8 個 sensors、`first_order`（16 條規則，176 參數）。
- `all37_zero`：37 個 sensors、`zero_order`（74 條規則，222 參數）。
- 字典秩分析另外包含 4 sensors 的 `zero_order` / `first_order` 加法設定（設定檔見下表）。任務潛向量為 $z \propto \mathbf{Z}_{\text{dict}}\,\theta$，$\mathbf{Z}_{\text{dict}} = \frac{1}{N} B^\top \Psi \in \mathbb{R}^{50 \times P}$（$P$ 為欄數）。

| 設定檔 | Sensors | 每個 sensor 的 terms | 規則數 $K$ | Consequent | 字典欄數 $P$ |
|---|---|---|---:|---|---:|
| `walker4_additive_zero` | 4：`torso_height`、`torso_upright`、`horizontal_velocity`、`angmomentum` | 3、2、3、2 | 10 | `zero_order` | 10 |
| `walker4_additive_first` | 同上 4 個 | 3、2、3、2 | 10 | `first_order` | 50 |
| `walker8_additive_zero` | 8：上述 4 個加左右 hip、knee 角速度 | 2 × 8 | 16 | `zero_order` | 16 |
| `walker8_additive_first` | 同上 8 個 | 2 × 8 | 16 | `first_order` | 144 |
| `walker_all37_additive_zero` | 37：Walker 全部 scalar sensors | 2 × 37 | 74 | `zero_order` | 74 |
| `walker_all37_additive_first` | 同上 37 個 | 2 × 37 | 74 | `first_order` | 2,812 |

**設定**

- FB 模型：Walker RND 660k（$d_z = 50$）；字典秩分析的狀態分布為 10,000 個狀態（seed 42，discount 0.98），與 T6.3 對齊。
- OOD 環境：`friction_mild`、`friction_high_mild`、`actuator_mild`、`actuator_high_mild`（與 T6.1、T6.2 相同），各 10 個 finetune seeds（`1000..1090`）。
- 搜尋：`sep-CMA-ES`（對齊 T6.2 `cmaes_anfis.yaml`：`pop_size: 16`、$\sigma_0 = 0.5$、真實 reward warm-start 500 步），每個任務 800,000 環境步，每 40,000 步於 20 個共用評估環境（種子 `5..24`）評估。
- 指標：依 Table 6（Jeen et al. 2024，`analysis.table6_aggregate`）。`select IQM` 為所有評估點中 all-task IQM 最高者，`final IQM` 為第 800,000 步的 all-task IQM。

**預期結果**（實驗前寫下）

1. 使用 37 個 sensors 的 Additive ANFIS 會比 4 sensors 的 ANFIS 還好
2. zero order 的 additive ANFIS 會差於 first order 的 additive ANFIS

## 結論

- **預期 1 成立（平均上）**：`all37_first` 4 環境平均 `566.3 / 561.3`（select / final），高於 T6.2 Cartesian 4 sensors 的 `549.6 / 543.6`（`+16.7`）。但並非每個環境都成立：`friction_high_mild`（`585.8` vs `558.9`）、`actuator_mild`（`501.8` vs `489.7`）、`actuator_high_mild`（`678.4` vs `637.9`）較高，`friction_mild` 較低（`499.2` vs `511.9`）。
- **預期 2 成立**：`all37_zero` 4 環境平均 `478.0 / 471.2`，落後 `all37_first` `88.3` 分，且 4 個環境全部較低（例如 `friction_mild` `427.2` vs `499.2`）。
- **交互作用與基底函數數量**：只有 1D 項的 `all37_zero` 在 37 個 sensors（74 條規則）下仍明顯較差，所以單靠基底函數多並不夠，需要二階交互作用（$\mu_{i,m}(x_i)\,x_j$）。另一方面，8 個 sensors 的 `8_first`（176 參數）`554.6 / 551.9` 與 T6.2 Cartesian custom（128–256 條規則，最高 2,336 參數）`555.9 / 550.9` 持平，高階 Cartesian 乘積並不必要。
- **與直接搜 $z$ 的差距仍在**：`all37_first` 將 $\mathbf{Z}_{\text{dict}}$ 條件數降到 `57.8`（變異數有效秩 `11.47`），高出 T6.2 Cartesian custom `10.4` 分，仍落後 T6.2 直接搜 $z$（`644.2 / 639.3`）`77.9` 分。
- **限制**：
  - `8_first` 與 T6.2 Cartesian custom 的 8 個 sensors 在 actuator 環境相同，T6.2 在 friction 環境另用 7 個 sensors，兩者在 friction 並非同一組 sensors。
  - `all37_first / friction_mild`（10 個 jobs）與 `all37_zero` 的部分 seeds（12 個）在 CPU 執行，其餘 98 個在 CUDA 執行，單一任務可能受裝置的浮點微小誤差影響；`all37_zero` 落後約 80–90 分、直接搜 $z$ 領先約 78 分的差距遠大於此變異。
  - `select IQM` 在報告使用的同一組評估種子上挑選最佳評估點，數值偏樂觀，因此一併列出 `final IQM`。

## 結果

### ANFIS 條件的 4 環境平均（對應預期 1、2 與直接搜 $z$ 的差距）

資料來源：`data/summary.csv`（T6.2 對照列來自 `tasks/T6.2-faster-update-algo/data/inputs/`，皆為 4 環境 × 10 seeds 配對計算）。

| 條件 | select IQM | final IQM | stand | walk | run | flip |
|---|---:|---:|---:|---:|---:|---:|
| `all37_first`（T6.4） | **566.3** | **561.3** | **741.0** | 647.7 | **237.6** | **474.9** |
| `8_first`（T6.4） | 554.6 | 551.9 | 698.9 | 649.7 | 208.5 | 454.0 |
| `all37_zero`（T6.4） | 478.0 | 471.2 | 663.6 | 515.9 | 145.3 | 426.5 |
| Cartesian 4 sensors（T6.2） | 549.6 | 543.6 | 739.8 | 638.5 | 217.1 | 448.7 |
| Cartesian custom（T6.2） | 555.9 | 550.9 | 738.3 | **652.4** | 230.2 | 449.3 |
| 直接搜 $z$ full-cov（T6.2） | 644.2 | 639.3 | 867.3 | 742.7 | 284.6 | 535.9 |

- 各任務欄為第 800,000 步的分數（ANFIS 條件中粗體為最高）。
- `all37_first` 的 `select IQM` 與 `final IQM` 在 ANFIS 條件中最高。

### 各環境明細（對應預期 1、2 的逐環境比較）

資料來源：`data/summary.csv`；格式為 select IQM / final IQM。

| 環境 | `all37_first` | `8_first` | `all37_zero` | T6.2 Cartesian 4 sensors | T6.2 Cartesian custom | T6.2 直接搜 $z$ |
|---|---:|---:|---:|---:|---:|---:|
| `friction_mild` | 499.2 / 490.7 | 509.5 / 507.3 | 427.2 / 427.2 | 511.9 / 503.5 | 516.7 / 516.7 | 616.6 / 615.1 |
| `friction_high_mild` | **585.8 / 582.2** | 555.6 / 547.7 | 464.9 / 453.6 | 558.9 / 548.7 | 561.0 / 556.6 | 656.5 / 643.9 |
| `actuator_mild` | **501.8 / 501.8** | 488.9 / 488.1 | 412.1 / 401.7 | 489.7 / 489.7 | 491.0 / 475.4 | 549.3 / 543.7 |
| `actuator_high_mild` | **678.4 / 670.4** | 664.4 / 664.4 | 607.9 / 602.2 | 637.9 / 632.6 | 654.9 / 654.9 | 754.5 / 754.5 |

- 粗體標示 `all37_first` 為 ANFIS 條件最高的環境。

### 字典秩分析（對應 `all37_zero` 與 `all37_first` 的條件數與有效秩）

資料來源：`data/zdict_rank.csv` 與 `data/zdict_spectra.npz`；Cartesian 對照組數據取自 T6.3（`tasks/T6.3-sensors/data/zdict_rank.csv` 與 `zdict_spectra.npz`）。

| 結構 / 設定 | Sensors | 規則數 | 欄數 | 數值秩 | 有效秩 ($\sigma_i$) | 條件數 | stand 殘差 | walk 殘差 | run 殘差 | flip 殘差 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| T6.3 Cartesian 4 sensors | 4 | 36 | 180 | 45 / 50 | 14.33 | 1,982 | 0 | 0 | 0 | 0 |
| T6.3 Cartesian 8 sensors | 8 | 256 | 2,304 | 50 / 50 | 17.41 | 252 | 0 | 0 | 0 | 0 |
| `walker4_additive_zero` | 4 | 10 | 10 | 10 / 10 | 4.75 | 147.7 | 0.016 | 0.280 | 0.160 | 0.250 |
| `walker4_additive_first` | 4 | 10 | 50 | 38 / 50 | 11.79 | 76,024 | 0 | 0 | 0 | 0 |
| `walker8_additive_zero` | 8 | 16 | 16 | 16 / 16 | 4.20 | 623.2 | 0.025 | 0.278 | 0.151 | 0.233 |
| `walker8_additive_first` | 8 | 16 | 144 | 50 / 50 | 15.62 | 425.2 | 0 | 0 | 0 | 0 |
| `walker_all37_additive_zero` | 37 | 74 | 74 | 43 / 50 | 8.66 | 3,388 | 0 | 0 | 0 | 0 |
| `walker_all37_additive_first` | 37 | 74 | 2,812 | 50 / 50 | 27.45 | 57.8 | 0 | 0 | 0 | 0 |

數值秩欄的上限為 $\min(50, P)$；表中的 0 實際值皆小於 $10^{-13}$。

數值秩欄的上限為 $\min(50, P)$；表中的 0 實際值皆小於 $10^{-13}$。

### 探索變異數分布（對應有效秩 `1.36` 與 `11.47`）

資料來源：`data/zdict_spectra.npz`；Cartesian 對照組取自 T6.3。依 $\theta$ 均向擾動在 $z$ 空間引出的共變異數 $\mathbf{Z}_{\text{dict}}\mathbf{Z}_{\text{dict}}^\top$，特徵值正比於 $\sigma_i^2$。

| 設定 | 變異數有效秩 ($\sigma_i^2$) | 前 10 個方向變異數佔比 | 最弱 25 個方向變異數佔比 |
|---|---:|---:|---:|
| T6.3 Cartesian 4 sensors | 6.39 | 98.11% | 0.02% |
| T6.3 Cartesian 8 sensors | 6.06 | 97.67% | 0.17% |
| `walker4_additive_zero` | 1.85 | 100.00% | — |
| `walker4_additive_first` | 5.33 | 99.08% | 0.01% |
| `walker8_additive_zero` | 1.31 | 99.98% | — |
| `walker8_additive_first` | 5.74 | 98.35% | 0.10% |
| `walker_all37_additive_zero` | 1.36 | 99.52% | 0.02% |
| `walker_all37_additive_first` | 11.47 | 89.26% | 1.09% |

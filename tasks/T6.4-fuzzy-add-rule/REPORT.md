# T6.4-fuzzy-add-rule：加法規則 ANFIS 字典秩分析與 OOD 適應實驗

在 T6.3 中，Cartesian product 規則數隨 sensor 數量指數成長（37 個 sensors 需要 $2^{37}$ 條規則而無法建立），且 4 與 8 sensors 的 $\mathbf{Z}_{\text{dict}}$ 條件數偏高，使探索集中在少數 $z$ 方向。本任務改用加法規則（`rule_mode: additive`，規則數 $K = \sum_i M_i$）與未正規化觸發強度（`normalize_firing: false`），先分析 $\mathbf{Z}_{\text{dict}}$ 的譜性質，再於 Walker 四個 OOD 動力學環境（各 10 個 finetune seeds）以 `sep-CMA-ES` 檢驗高階 Cartesian 交互作用與二階交互作用是否必要。

結論：

- **高階 Cartesian 規則乘積不必要**：同為 8 個 sensors、`first_order` consequent 時，拿掉 8 維乘積的加法規則 `8_first`（16 條規則、176 參數）在 4 環境平均取得 `554.6 / 551.9`（select / final IQM），與 T6.2 的 Cartesian custom（128–256 條規則、最高 2,336 參數）`555.9 / 550.9` 持平。
- **二階交互作用是必要的**：純 1D 加法 `all37_zero`（無 $\mu_{i,m}(x_i)\,x_j$ 交叉項，等價單層 RBF-KAN）在 4 與 8 sensors 無法表達 Walker 任務；擴到 37 sensors 雖能歸零 $z_r$ 殘差，但條件數高達 `3,388`（變異數有效秩僅 `1.36`），OOD 4 環境平均僅 `478.0 / 471.2`，落後 `all37_first` `88.3` 分。
- **改善字典條件數並不足以追平直接搜 $z$**：37 sensors 搭配 `first_order`（`all37_first`，74 條規則、2,960 參數）將 $\mathbf{Z}_{\text{dict}}$ 條件數降至 `57.8`（變異數有效秩升至 `11.47`），4 環境平均升至 `566.3 / 561.3`，高出 T6.2 Cartesian custom `10.4` 分，但仍落後 T6.2 直接搜 $z$（`644.2 / 639.3`）`77.9` 分。

## 1. 規則字典 $\mathbf{Z}_{\text{dict}}$ 分析：方法與設定

在 `rule_mode: additive` 與 `normalize_firing: false` 下，每個 sensor $i$ 的第 $m$ 個 linguistic term 獨立作為一條規則，總規則數 $K = \sum_{i=1}^{D} M_i$，觸發權重為未經 softmax 正規化的 membership 值：

$$w_{i,m}(s) = \mu_{i,m}(\tilde{x}_i(s))^{1/T}$$

其中 $\tilde{x}$ 為 z-score 後的 sensors，預設溫度 $T = 1.0$。與 T6.3 相同，在固定 membership 參數與 `output_activation: identity` 下：

- `zero_order`：$r(s) = \sum_{i=1}^{D} \sum_{m=1}^{M_i} b_{i,m}\,w_{i,m}(s)$（即單層 Gaussian RBF-KAN），字典欄數為 $K$。
- `first_order`：$r(s) = \sum_{i=1}^{D} \sum_{m=1}^{M_i} w_{i,m}(s)\,(W_{i,m}\tilde{x}(s) + b_{i,m})$，字典欄數為 $K(D+1)$。

任務潛向量為 $z \propto \mathbf{Z}_{\text{dict}}\,\theta$，其中 $\mathbf{Z}_{\text{dict}} = \frac{1}{N} B^\top \Psi \in \mathbb{R}^{50 \times P}$。

**設定**（與 T6.3 字典秩分析完全對齊）

- FB 模型：Walker RND 660k（$d_z = 50$）。
- 狀態分布 $\rho$：從 Walker RND 離線資料集抽 100,000 筆 transitions，再取 10,000 個狀態（seed 42，discount 0.98）。
- Membership 參數：`gaussian`（`min_sigma: 0.001`、`init_sigma: 1.0`），以 `init_from_data` 的分位數初始化。
- 數值秩：$\sigma_i > 10^{-3}\,\sigma_1$ 的奇異值個數；任務殘差：$\|\mathbf{Z}_{\text{dict}}\theta^{\ast} - z_r\| / \|z_r\|$。

**Sensor 組合**

| 設定檔 | Sensors | 每個 sensor 的 terms | 規則數 $K$ | Consequent | 字典欄數 $P$ |
|---|---|---|---:|---|---:|
| `walker4_additive_zero` | 4：`torso_height`、`torso_upright`、`horizontal_velocity`、`angmomentum` | 3、2、3、2 | 10 | `zero_order` | 10 |
| `walker4_additive_first` | 同上 4 個 | 3、2、3、2 | 10 | `first_order` | 50 |
| `walker8_additive_zero` | 8：上述 4 個加左右 hip、knee 角速度 | 2 × 8 | 16 | `zero_order` | 16 |
| `walker8_additive_first` | 同上 8 個 | 2 × 8 | 16 | `first_order` | 144 |
| `walker_all37_additive_zero` | 37：Walker 全部 scalar sensors | 2 × 37 | 74 | `zero_order` | 74 |
| `walker_all37_additive_first` | 同上 37 個 | 2 × 37 | 74 | `first_order` | 2,812 |

## 2. 字典秩結果

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

**探索變異數分布**（依 $\theta$ 均向擾動在 $z$ 空間引出的共變異數 $\mathbf{Z}_{\text{dict}}\mathbf{Z}_{\text{dict}}^\top$，特徵值正比於 $\sigma_i^2$）

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

## 3. OOD Walker 適應實驗（`sep-CMA-ES`，4 環境 × 10 seeds）

在與 T6.1、T6.2 相同的四個 OOD 環境（`friction_mild`、`friction_high_mild`、`actuator_mild`、`actuator_high_mild`）與 10 個 finetune seeds（`1000..1090`）上測試三組加法規則 ANFIS：

- `all37_first`：`walker_all37_additive_first`（74 條規則，共 2,960 參數：2,812 consequent + 148 membership）。
- `8_first`：`walker8_additive_first`（16 條規則，共 176 參數：144 consequent + 32 membership）。
- `all37_zero`：`walker_all37_additive_zero`（74 條規則，共 222 參數：74 consequent + 148 membership）。

**超參數**（對齊 T6.2 `cmaes_anfis.yaml`）

- `algo: cmaes`、`diagonal: true`（sep-CMA-ES）、`param_scope: all`、`pop_size: 16`、$\sigma_0 = 0.5$、`min_sigma: 0.0001`、`max_sigma: 2.0`。
- 先以真實 reward warm-start 500 步（`lr: 0.01`）；每個任務訓練 800,000 環境步（完整 1000-step episode，共 50 代），每 40,000 步於 20 個共用評估環境（種子 `5..24`）評估一次。
- 彙整指標依 Table 6（Jeen et al. 2024，`analysis.table6_aggregate`）：`select IQM` 為所有評估點中 all-task IQM 最高者，`final IQM` 為第 800,000 步的 all-task IQM。

**4 環境平均**

資料來源：`data/summary.csv`（T6.2 對照列來自 `tasks/T6.2-faster-update-algo/data/inputs/`，皆為 4 環境 × 10 seeds 配對計算）。

| 條件 | select IQM | final IQM | stand | walk | run | flip |
|---|---:|---:|---:|---:|---:|---:|
| `all37_first`（T6.4） | **566.3** | **561.3** | **741.0** | 647.7 | **237.6** | **474.9** |
| `8_first`（T6.4） | 554.6 | 551.9 | 698.9 | 649.7 | 208.5 | 454.0 |
| `all37_zero`（T6.4） | 478.0 | 471.2 | 663.6 | 515.9 | 145.3 | 426.5 |
| Cartesian 4 sensors（T6.2） | 549.6 | 543.6 | 739.8 | 638.5 | 217.1 | 448.7 |
| Cartesian custom（T6.2） | 555.9 | 550.9 | 738.3 | **652.4** | 230.2 | 449.3 |
| 直接搜 $z$ full-cov（T6.2） | 644.2 | 639.3 | 867.3 | 742.7 | 284.6 | 535.9 |

各任務欄為第 800,000 步的分數（ANFIS 條件中粗體為最高）。

**各環境明細（select IQM / final IQM）**

| 環境 | `all37_first` | `8_first` | `all37_zero` | T6.2 Cartesian 4 sensors | T6.2 Cartesian custom | T6.2 直接搜 $z$ |
|---|---:|---:|---:|---:|---:|---:|
| `friction_mild` | 499.2 / 490.7 | 509.5 / 507.3 | 427.2 / 427.2 | 511.9 / 503.5 | 516.7 / 516.7 | 616.6 / 615.1 |
| `friction_high_mild` | **585.8 / 582.2** | 555.6 / 547.7 | 464.9 / 453.6 | 558.9 / 548.7 | 561.0 / 556.6 | 656.5 / 643.9 |
| `actuator_mild` | **501.8 / 501.8** | 488.9 / 488.1 | 412.1 / 401.7 | 489.7 / 489.7 | 491.0 / 475.4 | 549.3 / 543.7 |
| `actuator_high_mild` | **678.4 / 670.4** | 664.4 / 664.4 | 607.9 / 602.2 | 637.9 / 632.6 | 654.9 / 654.9 | 754.5 / 754.5 |

## 4. 解讀與限制

- **高階 Cartesian 規則交互作用不必要**：`8_first` 與 T6.2 Cartesian custom 使用相同的 8 個 actuator-group sensors（T6.2 在 friction 另用 7 sensors），拿掉高階乘積後參數量從 2,336 降到 176，4 環境平均仍幾乎完全一致（`554.6` vs `555.9`；`final` 為 `551.9` vs `550.9`）。
- **完全無交互作用的 1D 加法不足**：`all37_zero` 在全部 4 個環境都大幅落後 `all37_first`（4 環境平均差 `88.3` 分）與 `8_first`（差 `76.6` 分），反映 Walker 任務確實需要至少二階的感測器交互作用（$\mu_{i,m}(x_i)\,x_j$），且其 $\mathbf{Z}_{\text{dict}}$ 變異數有效秩僅 `1.36`。
- **字典條件數改善帶來有限增益，未消除與直接搜 $z$ 的差距**：`all37_first` 在 `friction_high_mild`、`actuator_mild`、`actuator_high_mild` 三個環境都創下 ANFIS 最高分，4 環境平均高出 Cartesian custom `10.4` 分；但仍低於直接搜 $z$ `77.9` 分。這與 T6.3 的發現一致：透過 `ANFIS`（從 $z_r$ warm-start 起跳，並限制在 smooth sensor-weighted reward 流形內）探索的結構性偏置，無法單靠改善 $\mathbf{Z}_{\text{dict}}$ 條件數就完全消除。
- **執行硬體差異**：`all37_first / friction_mild`（10 個）與 `all37_zero` 的部分 seeds（12 個）在 CPU 執行，其餘 98 個 jobs 在 CUDA 執行；各環境的 `baseline_table6.csv`（同一 seed 下 $z_r$ 評估）與先前 T6.3 觀察一致，單一任務可能受浮點與執行裝置微小誤差影響，但 10 seeds 平均下的主要差距（`all37_zero` 落後約 80–90 分、直接搜 $z$ 領先約 78 分）遠大於此變異。

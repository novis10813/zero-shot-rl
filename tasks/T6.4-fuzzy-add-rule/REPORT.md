# T6.4-fuzzy-add-rule：加法規則 ANFIS 字典秩分析

在 T6.3 中，Cartesian product 規則數隨 sensor 數量指數成長（37 個 sensors 需要 $2^{37}$ 條規則而無法建立），且 4 與 8 sensors 的 $\mathbf{Z}_{\text{dict}}$ 條件數偏高，使參數空間的探索集中在少數 $z$ 方向。本實驗改用加法規則（`rule_mode: additive`，規則數 $K = \sum_i M_i$）與未正規化觸發強度（`normalize_firing: false`），比較 4、8、37 個 sensors 在 `zero_order`（等價於單層 RBF-KAN）與 `first_order` consequent 下的 $\mathbf{Z}_{\text{dict}}$ 秩、條件數與任務殘差。

結論：

- `zero_order` 在 4 與 8 sensors 下無法表示 Walker 任務（四個任務平均殘差分別為 0.177 與 0.172，其中 `walk` 為 0.280 與 0.278）。擴展到 37 個 sensors 後殘差降為 0，但條件數為 3,388、以 $\sigma_i^2$ 計算的有效秩僅 1.36。
- 37 個 sensors 搭配 `first_order`（74 條規則、2,812 欄）顯著改善搜尋空間的數值條件：條件數從 T6.3 4-sensor Cartesian 的 1,982 降至 57.8（約 34 倍），有效秩（$\sigma_i$）從 14.33 升至 27.45，變異數有效秩（$\sigma_i^2$）從 6.39 升至 11.47。

## 1. 方法與設定

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

## 2. 結果

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

## 3. 解讀與限制

- **單純相加的 1D 基底（`zero_order`）不足以改善搜尋空間**：4 與 8 sensors 只有 10 與 16 條規則，缺少跨 sensor 乘積項，無法表示 `walk`、`run`、`flip`。擴張至 37 sensors 時雖足以在 $B(s)$ 欄空間內補齊 4 個任務，但能量高度集中於少數方向（變異數有效秩 1.36）。
- **加法規則讓 37 sensors 搭配 `first_order` 成為可能**：Cartesian 結構在 37 sensors 時規則數爆炸而無法建立；改用加法規則後僅需 74 條規則。與 T6.3 的 8-sensor Cartesian（2,304 欄）相比，`walker_all37_additive_first` 參數維度相近（2,812 欄），但條件數從 252 降至 57.8，最弱 25 個 $z$ 方向的變異數佔比提高約 6.4 倍（相對 4-sensor Cartesian 提高約 50 倍）。
- **限制**：本報告僅量測固定 membership 初始化下的 $\mathbf{Z}_{\text{dict}}$ 譜性質，尚未進行 OOD Walker 的實際 finetune；`walker_all37_additive_first` 的搜尋參數為 2,812 維以上（含 membership），實際表現仍需透過 `sep-CMA-ES` 與 `zero-shot` / 直接搜 $z$ 對照驗證。

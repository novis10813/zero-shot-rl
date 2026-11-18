# T5.1 結果：兩階段特徵預測器與動作轉換（$g_0, g_1, T_\theta$）

本實驗測試直接以兩個單步特徵預測器與參數化動作轉換網路實作 FB-AA，不修改 FB backbone（$B_0$、actor、$z_r$ 皆凍結），也不使用 online reward：

- **$g_0(s, a) \to B_0(s')$**：在原環境的 stock FB 軌跡上離線訓練（50,000 步，learning rate $3\times 10^{-4}$）。
- **$g_1(s, a) \to B_0(s')$**：在每個測試環境用獨立的 finetune seeds（加高斯探索雜訊 $\sigma=0.2$）收集轉移，從 $g_0$ 權重微調 2,000 步（learning rate $10^{-4}$）。
- **$T_\theta(s, a)$**：零初始化殘差網路 $\tanh(\mathrm{atanh}(\mathrm{clip}(a, \pm 0.999)) + \Delta_\theta(s, a))$，在微調軌跡上優化 2,000 步（learning rate $3\times 10^{-4}$，$\lambda=0.1$）：

$$\theta^* = \arg\min_\theta \mathbb{E}_{s,a}\left[\lVert g_1(s, T_\theta(s, a)) - g_0(s, a)\rVert^2 + \lambda \lVert T_\theta(s, a) - a\rVert^2\right]$$

以下依照 `analysis.table6_aggregate` 計算 **Table-6 全任務 IQM**（每個 seed 先在各任務內取 IQM，跨 seed 平均後再取四任務 IQM）。

## 主實驗：`660000.pt`（10 finetune seeds × 20 共用環境）

對齊 T4.1.3.1 的評估協定：單一凍結 FB checkpoint（`660000.pt`），在 10 個互不重疊的 finetune seeds（`1000, 1010, ..., 1090`，每個任務 10 個微調 episodes）微調 $g_1$ 與 $T_\theta$，並於四任務共用的 20 個評估環境（`5..24`，共 $4 \times 20 = 80$ episodes / finetune seed）評估。

以 clean 環境（`baseline none = 396.9`）為對齊目標，差值越接近 `0` 代表越貼近 clean Frozen FB 的表現：

| 環境 | `none` | `star` ($a^*$) | 新版 $T_\theta$ | `none` − 396.9 | $T_\theta$ − 396.9 | `mean \|a'-a\|` |
|---|---|---|---|---|---|---|
| `baseline` | 396.9 | 396.9 | **396.0** | 0.0 | −0.9 | 0.160 |
| `actuator_mild` | 178.5 | 171.4 | **251.8** | −218.4 | −145.1 | 0.215 |
| `actuator_strong` | 92.9 | 106.1 | **104.8** | −304.0 | −292.1 | 0.330 |
| `actuator_high_mild` | 452.5 | 352.7 | **449.6** | +55.6 | +52.7 | 0.257 |
| `actuator_high_strong` | 562.6 | 359.0 | **489.5** | +165.7 | +92.7 | 0.387 |
| `friction_mild` | 194.6 | — | **242.3** | −202.3 | −154.6 | 0.257 |
| `friction_strong` | 138.3 | — | **161.3** | −258.6 | −235.6 | 0.282 |
| `friction_high_mild` | 255.7 | — | **343.4** | −141.2 | −53.5 | 0.179 |
| `friction_high_strong` | 157.3 | — | **218.6** | −239.6 | −178.2 | 0.232 |

1. **所有 8 個偏移環境都比不修正（`none`）更接近 clean Frozen FB（396.9）**：
   - `friction_high_mild` 從 255.7 提升到 **343.4**（距離目標從 −141.2 縮小至 −53.5）。
   - `actuator_mild` 從 178.5 提升到 **251.8**（+73.3，且高於 `star` 的 171.4）。
   - `friction_high_strong`（157.3 → **218.6**）、`friction_mild`（194.6 → **242.3**）、`friction_strong`（138.3 → **161.3**）與 `actuator_strong`（92.9 → **104.8**）皆一致回升。
   - `actuator_high_strong` 從 562.6 拉回 **489.5**（距離目標從 +165.7 降至 +92.7）。
2. **無偏移環境幾乎零副作用**：`baseline` 為 396.0（與目標 396.9 僅差 −0.9）。

---

## 與 T4.1.3.2 舊版 FB-AA 對照（T05 3 FB seeds，5 episodes / task）

沿用 T4.1.3.2 協定（`zsrl-32/41/42.pt`，每個任務 5 個環境種子 `300000..300019`，目標為 `baseline none = 434.0`）：

| 環境 | `none` | `star` ($a^*$) | 舊版 FB-AA | 新版 $T_\theta$ | 舊版 − 434.0 | 新版 − 434.0 |
|---|---|---|---|---|---|---|
| `baseline` | 434.0 | 434.0 | 415.6 | **427.4** | −18.4 | −6.6 |
| `actuator_mild` | 346.0 | 347.2 | 321.1 | **324.5** | −112.8 | −109.4 |
| `actuator_strong` | 255.9 | 266.5 | 233.2 | **229.8** | −200.8 | −204.2 |
| `actuator_high_mild` | 488.7 | 438.8 | 458.4 | **465.7** | +24.5 | +31.7 |
| `actuator_high_strong` | 525.2 | 432.7 | 463.5 | **456.8** | +29.6 | +22.8 |
| `friction_mild` | 382.1 | — | 348.6 | **348.4** | −85.4 | −85.6 |
| `friction_strong` | 305.9 | — | 262.3 | **308.2** | −171.6 | −125.8 |
| `friction_high_mild` | 364.8 | — | 407.8 | **416.2** | −26.1 | −17.7 |
| `friction_high_strong` | 378.6 | — | 354.9 | **366.3** | −79.1 | −67.7 |

註：`star` ($a^* = \mathrm{clip}(a/m, -1, 1)$) 僅定義於 gear 偏移，故 `friction_*` 欄位留空。

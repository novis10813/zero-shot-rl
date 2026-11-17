# T5.1 結果：兩階段特徵預測器與動作轉換（$g_0, g_1, T_\theta$）

本實驗測試直接以兩個單步特徵預測器與參數化動作轉換網路實作 FB-AA，不修改 FB backbone（$B_0$、actor、$z_r$ 皆凍結），也不使用 online reward：

- **$g_0(s, a) \to B_0(s')$**：在原環境的 stock FB 軌跡上離線訓練（50,000 步，learning rate $3\times 10^{-4}$）。
- **$g_1(s, a) \to B_0(s')$**：在每個測試環境用獨立的 finetune seeds（每個任務 5 個 episode，加高斯探索雜訊 $\sigma=0.2$）收集轉移，從 $g_0$ 權重微調 2,000 步（learning rate $10^{-4}$）。
- **$T_\theta(s, a)$**：零初始化殘差網路 $\tanh(\mathrm{atanh}(\mathrm{clip}(a, \pm 0.999)) + \Delta_\theta(s, a))$，在微調軌跡上優化 2,000 步（learning rate $3\times 10^{-4}$，$\lambda=0.1$）：

$$\theta^* = \arg\min_\theta \mathbb{E}_{s,a}\left[\lVert g_1(s, T_\theta(s, a)) - g_0(s, a)\rVert^2 + \lambda \lVert T_\theta(s, a) - a\rVert^2\right]$$

評估在 3 個 FB 訓練 seed 上進行，每個環境與任務各跑 5 個獨立的 evaluation seeds（共 60 個 episodes）。分數依照 `analysis.table6_aggregate` 計算 **Table-6 全任務 IQM**（每個 seed 先在各任務內取 IQM，跨 seed 平均後再取四任務 IQM），並與 T4.1.3.2 舊版 FB-AA（`context_encoder` + per-step Adam）比較。

## Table-6 全任務 IQM 比較

以 clean 環境（`baseline none = 434.0`）作為對齊目標，最後兩欄越接近 `0` 代表越貼近 clean Frozen FB 的表現：

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

## 觀察

1. **無偏移環境（`baseline`）副作用減小**：新版 $T_\theta$ 與目標 434.0 只差 −6.6（舊版為 −18.4）。
2. **強摩擦偏移（`friction_strong`）與高摩擦環境優於舊版**：舊版在 `friction_strong` 距離目標 −171.6，新版縮小至 −125.8；`friction_high_mild`（−17.7 vs −26.1）與 `friction_high_strong`（−67.7 vs −79.1）也更接近目標 434.0。
3. **馬達變強（`actuator_high_mild/strong`）成功往目標拉回**：`none` 因馬達變強高於目標 +54.7 與 +91.2；新版拉回至 +31.7 與 +22.8（舊版為 +24.5 與 +29.6），`star` 為 +4.8 與 −1.3。
4. **馬達變弱與輕度低摩擦尚未改善**：`actuator_mild/strong` 受限於原 policy 動作飽和在 $\pm 1$；`friction_mild`（−85.6）仍低於 `none`（−51.9）。

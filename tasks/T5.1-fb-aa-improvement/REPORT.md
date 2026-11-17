# T5.1 結果：兩階段特徵預測器與動作轉換（$g_0, g_1, T_\theta$）

本實驗測試直接以兩個單步特徵預測器與參數化動作轉換網路實作 FB-AA，不修改 FB backbone（$B_0$、actor、$z_r$ 皆凍結），也不使用 online reward：

- **$g_0(s, a) \to B_0(s')$**：在原環境的 stock FB 軌跡上離線訓練（50,000 步，learning rate $3\times 10^{-4}$）。
- **$g_1(s, a) \to B_0(s')$**：在每個測試環境用獨立的 finetune seeds（每個任務 5 個 episode，加高斯探索雜訊 $\sigma=0.2$）收集轉移，從 $g_0$ 權重微調 2,000 步（learning rate $10^{-4}$）。
- **$T_\theta(s, a)$**：零初始化殘差網路 $\tanh(\mathrm{atanh}(\mathrm{clip}(a, \pm 0.999)) + \Delta_\theta(s, a))$，在微調軌跡上優化 2,000 步（learning rate $3\times 10^{-4}$，$\lambda=0.1$）：

$$\theta^* = \arg\min_\theta \mathbb{E}_{s,a}\left[\lVert g_1(s, T_\theta(s, a)) - g_0(s, a)\rVert^2 + \lambda \lVert T_\theta(s, a) - a\rVert^2\right]$$

評估在 3 個 FB 訓練 seed 上進行，每個環境與任務各跑 5 個獨立的 evaluation seeds（共 60 個 episodes）。分數依照 `analysis.table6_aggregate` 計算 **Table-6 全任務 IQM**（每個 seed 先在各任務內取 IQM，跨 seed 平均後再取四任務 IQM），並與 T4.1.3.2 舊版 FB-AA（`context_encoder` + per-step Adam）比較。

## Table-6 全任務 IQM 比較

| 環境 | `none` | `star` ($a^*$) | 舊版 FB-AA | 新版 $T_\theta$ | 舊版 − `none` | 新版 − `none` |
|---|---|---|---|---|---|---|
| `baseline` | 434.0 | 434.0 | 415.6 | **427.4** | −18.4 | −6.6 |
| `actuator_mild` | 346.0 | 347.2 | 321.1 | **324.5** | −24.9 | −21.5 |
| `actuator_strong` | 255.9 | 266.5 | 233.2 | **229.8** | −22.7 | −26.1 |
| `actuator_high_mild` | 488.7 | 438.8 | 458.4 | **465.7** | −30.3 | −23.0 |
| `actuator_high_strong` | 525.2 | 432.7 | 463.5 | **456.8** | −61.6 | −68.4 |
| `friction_mild` | 382.1 | 382.1 | 348.6 | **348.4** | −33.6 | −33.7 |
| `friction_strong` | 305.9 | 305.9 | 262.3 | **308.2** | −43.6 | +2.2 |
| `friction_high_mild` | 364.8 | 364.8 | 407.8 | **416.2** | +43.1 | +51.5 |
| `friction_high_strong` | 378.6 | 378.6 | 354.9 | **366.3** | −23.7 | −12.3 |

註：`actuator_high_mild` 與 `actuator_high_strong` 因馬達變強，`none` 分數高於原環境（434.0），因此 `star` 與目標一致的修正相對於 `none` 為負值，代表將動力學拉回原環境水準。

## 觀察

1. **無偏移環境（`baseline`）副作用減小**：新版 $T_\theta$ 在 Table-6 IQM 與 `none` 相差 −6.6（舊版為 −18.4）。
2. **強摩擦偏移（`friction_strong`）與高摩擦環境優於舊版**：舊版在 `friction_strong` 掉到 262.3（−43.6），新版維持在 308.2（+2.2）；`friction_high_mild` 與 `friction_high_strong` 也高於舊版（416.2 vs 407.8；366.3 vs 354.9）。
3. **馬達變強（`actuator_high_mild/strong`）與舊版趨勢一致**：兩版皆能將高於原環境的 return 往 `star` 方向拉回一部分（`actuator_high_strong` 新版拉回 −68.4，目標 `star` 為 −92.5）。
4. **馬達變弱與輕度低摩擦尚未改善**：`actuator_mild/strong` 受限於原 policy 動作飽和在 $\pm 1$；`friction_mild`（−33.7）仍低於 `none`。

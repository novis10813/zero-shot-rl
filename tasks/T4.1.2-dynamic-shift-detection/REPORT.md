# T4.1.2 結果：FB Bellman 殘差能否偵測動力學偏移

**結論：** Dynamics shifts induce negligible mean changes in FB Bellman residuals. Actuator shifts primarily reduce residual covariance, whereas friction shifts remain largely indistinguishable from baseline dynamics.

補充：actuator 調高時共變異數反而增加，所以共變異數的變化方向跟著 gear 的方向走（調低縮小、調高放大），殘差的大小比較像是反映動作的影響力，而不是動力學與訓練資料不一致的程度。

## 設定

殘差 $e=F(s,a,z)-B(s')-\gamma F(s',\pi_z(s'),z)$，$F$ 為兩個 forward head 的平均，$z$ 為 walk 的 $z_r$。Detector 以訓練資料估計 $\mu_0(s,a,z)=E[e\mid s,a,z]$（MLP 回歸）與一個共用的 $\Sigma_0$，分數為

$$D=(e-\mu_0)^\top(\Sigma_0+\epsilon I)^{-1}(e-\mu_0)$$

比較方式是配對的：在 baseline 環境用 $z_r$ 的 actor 跑 20 個 episode，每一步（999 步）取 $(s_t,a_t)$ 及其物理狀態，把同一個物理狀態與動作放進各偏移環境只走一步，得到 $s'_{\text{shift}}$。所以每個樣本的 $(s,a,z)$ 完全相同，只有 $s'$ 來自不同動力學。Baseline 環境的重播與原 rollout 逐位元相同（最大誤差 $7.5\times10^{-13}$）。

兩種 detector，各 3 個 FB 訓練 seed 一個：

- **資料版：** 以 ExORL RND 資料的 100k 轉移擬合。
- **軌跡版：** 以 baseline 環境中 $z_r$ 策略自己的 rollout 擬合（episode 100 到 119，與評估用的 episode 0 到 19 不重疊）。

「調低」四個設定為 T3.2 的 friction 與 actuator 的 mild、strong（設定檔見 `tasks/T3.2-ood-robustness/configs/`）。「調高」四個設定見 `configs/`：摩擦 1.5 與 3.0（baseline 為 0.7），gear 為 XML 原值的 130% 與 160%。

## $D$ 的配對比較

| 設定 | $s'$ 差異中位數 | 資料版 $D$ 配對差 | 資料版 AUROC | 軌跡版 $D$ 配對差 | 軌跡版 AUROC |
|---|---|---|---|---|---|
| Friction — Mild（調低） | 0.19 | -0.55 | 0.497 | +0.15 | 0.505 |
| Friction — Strong（調低） | 0.32 | -0.42 | 0.499 | +0.58 | 0.510 |
| Actuator — Mild（調低） | 3.40 | -2.20 | 0.460 | -1.73 | 0.477 |
| Actuator — Strong（調低） | 5.26 | -3.17 | 0.440 | -2.12 | 0.474 |
| Friction — Mild（調高） | 0.17 | +0.39 | 0.503 | +0.39 | 0.503 |
| Friction — Strong（調高） | 0.63 | -0.13 | 0.499 | -0.04 | 0.502 |
| Actuator — Mild（調高） | 4.19 | +3.55 | 0.555 | +3.73 | 0.551 |
| Actuator — Strong（調高） | 7.94 | +7.95 | 0.612 | +9.23 | 0.617 |

每格為 3 個訓練 seed 的平均；每個 seed 為 20 個 episode × 999 步的配對樣本。$s'$ 差異為 $\|s'_{\text{shift}}-s'_{\text{base}}\|_2$ 的中位數。配對差為 $D_{\text{shift}}-D_{\text{base}}$ 的平均。AUROC 為在全部樣本中 $P(D_{\text{shift}}>D_{\text{base}})$，1 為完全可分，0.5 為不可分，低於 0.5 為方向相反。$D_{\text{base}}$ 在資料版為 29.7，軌跡版為 31.3。資料版 detector 的 $\mu_0$ 每次擬合的結果差異不小（重跑一次時 $D_{\text{base}}$ 是 56.0），所以絕對值不穩定，本表只比較同一個 detector 內的配對差。

- **Friction（調低與調高）：** 單步 $s'$ 差異很小（0.17 到 0.63），$D$ 沒有可分辨的變化，AUROC 為 0.497 到 0.510。
- **Actuator 調低：** $s'$ 差異大（3.4 到 5.3），但 $D$ 反而降低，AUROC 為 0.440 到 0.477，方向相反。
- **Actuator 調高：** $D$ 上升，AUROC 為 0.551 到 0.617，強度越大越高，但 Mild 只比隨機好 0.05，Strong 約 0.61，仍然不是可靠的偵測器。

## 殘差的平均與共變異數

$r=e-\mu_0$。$C_0$ 為 baseline 環境下 $r$ 的共變異數，$P=(C_0+10^{-6}I)^{-1}$，$m$ 與 $C$ 為偏移環境相對 baseline 的平均差與共變異數：

| 量 | 意義 |
|---|---|
| $m^\top P m$ | 殘差平均的移動量，以 baseline 的變異為單位；0 為沒有移動 |
| $\mathrm{tr}(PC)/d$ | 偏移環境的變異量相對 baseline 的平均比例，$d=50$；1 為沒有改變，小於 1 為更集中 |
| $\log\det C-\log\det C_0$ | 殘差雲體積的對數比；0 為體積相同，負值為縮小 |

| 設定 | 資料版 $m^\top Pm$ | 資料版 $\mathrm{tr}(PC)/d$ | 資料版 $\log\det$ 差 | 軌跡版 $m^\top Pm$ | 軌跡版 $\mathrm{tr}(PC)/d$ | 軌跡版 $\log\det$ 差 |
|---|---|---|---|---|---|---|
| Friction — Mild（調低） | 0.011 | 0.991 | −0.55 | 0.015 | 1.001 | −0.03 |
| Friction — Strong（調低） | 0.017 | 0.994 | −0.44 | 0.023 | 1.011 | +0.41 |
| Actuator — Mild（調低） | 0.013 | 0.936 | −3.40 | 0.021 | 0.945 | −2.89 |
| Actuator — Strong（調低） | 0.032 | 0.906 | −5.11 | 0.051 | 0.928 | −3.82 |
| Friction — Mild（調高） | 0.001 | 1.005 | +0.23 | 0.002 | 1.010 | +0.45 |
| Friction — Strong（調高） | 0.015 | 0.989 | −0.66 | 0.024 | 0.996 | −0.27 |
| Actuator — Mild（調高） | 0.022 | 1.100 | +4.63 | 0.035 | 1.110 | +5.16 |
| Actuator — Strong（調高） | 0.090 | 1.219 | +9.48 | 0.145 | 1.265 | +11.52 |

每格為 3 個訓練 seed 的平均。

- **平均幾乎不動：** $m^\top Pm\le0.06$，即殘差平均的移動不到 baseline 標準差的四分之一。
- **共變異數只隨 actuator 改變，且方向跟著 gear：** gear 調低時 $\mathrm{tr}(PC)/d$ 為 0.906 到 0.945（殘差縮小），gear 調高時為 1.100 到 1.265（殘差放大）。friction 的 $\mathrm{tr}(PC)/d$ 為 0.989 到 1.011，與 baseline 相同。
- **對 $D$ 的影響：** $D$ 的期望約為 $m^\top P m+\mathrm{tr}(PC)$，平均項可忽略，所以 $D$ 的升降由共變異數決定：gear 調低時 $D$ 降低，調高時 $D$ 升高，friction 下 $D$ 不變。
- **偏移強度的排序一致：** actuator 在調低與調高兩個方向，Strong 的變化都大於 Mild。

## 限制

- 「$e$ 的幅度隨機器人動作大小變化，所以 actuator 變弱時殘差縮小」是從資料推測，沒有直接驗證（例如拿 $e$ 對速度回歸）。
- $\Sigma_0$ 是整個資料共用的一個，沒有依 $(s,a,z)$ 條件化。
- 只在 walk 的 $z_r$ 評估，每個 seed 只有 20 個 baseline episode。
- 單步重播只適用於靜態偏移，不支援有 `events` 的中途切換設定。
- 調高設定的強度沒有用 return 校準。
- 資料版 detector 的擬合不能逐位元重現（GPU 非決定性），重新擬合會得到不同的絕對 $D$，但 friction 無反應、actuator 調低時 $D$ 降低這兩點在兩次擬合中相同。

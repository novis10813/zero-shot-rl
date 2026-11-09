# T4.1.2.1 結果：動作條件動力學模型能否偵測動力學偏移

**結論：** 動力學模型的預測誤差，在聚合一個 episode 內多步後，可以偵測 actuator 偏移，調低與調高兩個方向都有訊號。Friction 偏移用這個模型找不到可用的閾值。單筆轉移的偵測很弱：以 baseline 第 99 百分位數為閾值，只有 actuator 調高 strong 的偵測率達 7.5%。

與 T4.1.2 的 FB Bellman 殘差相比，actuator 的 AUROC 較高（0.61 到 0.80，FB 殘差為 0.47 到 0.62），而且調低的方向正確。FB 殘差在調低時分數反而下降。

## 設定

模型為 3 層、寬度 512 的 ReLU MLP，輸入 $(s_t,a_t)$，輸出 $\widehat{\Delta s}_t$，誤差 $e^{Dyn}_t=(s_{t+1}-s_t)-\widehat{\Delta s}_t$。輸入與目標用 ExORL walker rnd 全資料集的逐維統計標準化。訓練只用 baseline 環境的互動轉移：三個 FB 訓練 seed 的 $z_r$ 策略各跑 20 個 episode（episode 100 到 119）混合，其中 48 個 episode 訓練，12 個 episode 做 validation（early stopping 與閾值）。測試資料為 T4.1.2 的配對單步資料：三個 seed 各 20 個 episode（episode 0 到 19）× 999 步，同一個 $(s,a)$ 放進 baseline 與八個偏移環境，只有 $s'$ 不同。切分的細節見 README。

兩種分數：

- $\lVert e\rVert_2$：誤差的 L2 範數，單位同觀測。
- $D_{dyn}=e^\top\Sigma_0^{-1}e$：以 validation 的 baseline 誤差共變異數 $\Sigma_0$ 正規化的 Mahalanobis 距離，與 T4.1.2 的 $D$ 對應。

閾值取 validation 中 baseline 分數的第 95 或 99 百分位數。「調低」四個設定為 T3.2 的 friction 與 actuator 的 mild、strong，「調高」四個設定為 T4.1.2 的 `configs/`。

## Baseline 的預測品質

在 60 個 test episode 的 baseline 轉移上。$R^2=1-\sum_t\lVert\Delta s_t-\widehat{\Delta s}_t\rVert^2/\sum_t\lVert\Delta s_t-\overline{\Delta s}\rVert^2$，24 維池化。

| 指標 | 預測 $\Delta s$ | 預測 $s_{t+1}$ |
|---|---|---|
| $R^2$ | 0.9385 | 0.9318 |
| 原始單位 MSE（每維平均） | 1.615 | 1.792 |
| 每筆轉移的總平方誤差 | 38.76 | 43.02 |
| $\lVert e\rVert_2$ 中位數 | 3.76 | 4.09 |
| $\lVert e\rVert_2$ 第 90 百分位數 | 9.31 | 9.92 |
| $\lVert e\rVert_2$ 第 95 百分位數 | 12.15 | 12.73 |
| $\lVert e\rVert_2$ 第 99 百分位數 | 19.97 | 20.22 |
| Validation $\lVert e\rVert_2$ 中位數 | 3.49 | 3.79 |

「預測 $s_{t+1}$」是把目標換成 $s_{t+1}$（loss 為標準化 $s_{t+1}$ 的 MSE，沒有 skip connection），再以 $e=s_{t+1}-\hat s_{t+1}$ 評分，所以分數與預測 $\Delta s$ 的模型是同一個量。各維度的 $R_j^2=1-\sum_t(\Delta s_{t,j}-\widehat{\Delta s}_{t,j})^2/\sum_t(\Delta s_{t,j}-\overline{\Delta s}_j)^2$：

| 維度 | 內容 | 預測 $\Delta s$ | 預測 $s_{t+1}$ |
|---|---|---|---|
| 0 到 13 | 姿態 | 0.959 到 0.983 | 0.937 到 0.968 |
| 14 | 身體高度 | 0.988 | 0.939 |
| 15 | 水平速度 | 0.912 | 0.891 |
| 16 | 垂直速度 | 0.892 | 0.856 |
| 17 | 軀幹角速度 | 0.949 | 0.939 |
| 18 到 23 | 關節速度 | 0.925 到 0.953 | 0.908 到 0.947 |

- 預測 $s_{t+1}$ 的 24 個維度全部低於預測 $\Delta s$。降幅最大的是身體高度（−0.048）與垂直速度（−0.036）。這些維度的 $\Delta s$ 相對於 $s$ 的尺度較小，網路要自己學出近似恆等映射。
- 最難預測的是垂直速度與水平速度，應與腳碰地的接觸事件有關（未驗證）。
- 24 個維度的完整數字見 `data/prediction_dims.csv`。

## 單筆轉移的偵測

下表為預測 $\Delta s$ 的模型，分數為 $\lVert e\rVert_2$。$s'$ 差異為 $\lVert s'_{\text{shift}}-s'_{\text{base}}\rVert_2$ 的中位數。AUROC 括號為以 episode 重抽的 95% 區間。TPR 與 FPR 是分數高於閾值的偏移樣本與 baseline 樣本的比例。

| 設定 | $s'$ 差異 | 中位數 | AUROC | TPR / FPR（p95，閾值 11.7） | TPR / FPR（p99，閾值 18.9） |
|---|---|---|---|---|---|
| Baseline | 0 | 3.76 | — | — / 5.7% | — / 1.2% |
| Friction — Mild（調低） | 0.19 | 3.84 | 0.510 (0.508–0.511) | 6.2% / 5.7% | 1.5% / 1.2% |
| Friction — Strong（調低） | 0.32 | 3.90 | 0.516 (0.515–0.518) | 6.6% / 5.7% | 1.6% / 1.2% |
| Actuator — Mild（調低） | 3.41 | 4.95 | 0.608 (0.604–0.612) | 6.5% / 5.7% | 1.2% / 1.2% |
| Actuator — Strong（調低） | 5.29 | 6.29 | 0.695 (0.690–0.699) | 11.4% / 5.7% | 1.4% / 1.2% |
| Friction — Mild（調高） | 0.17 | 3.81 | 0.506 (0.505–0.507) | 6.0% / 5.7% | 1.3% / 1.2% |
| Friction — Strong（調高） | 0.65 | 4.07 | 0.532 (0.531–0.534) | 6.5% / 5.7% | 1.4% / 1.2% |
| Actuator — Mild（調高） | 4.22 | 5.86 | 0.667 (0.664–0.670) | 11.6% / 5.7% | 2.0% / 1.2% |
| Actuator — Strong（調高） | 7.99 | 8.93 | 0.796 (0.793–0.799) | 33.9% / 5.7% | 7.5% / 1.2% |

- $D_{dyn}$ 的 AUROC 較低：actuator 為 0.582 到 0.795，friction 為 0.508 到 0.537，完整數字在 `data/detect_delta.csv`。
- 預測 $s_{t+1}$ 的模型，AUROC 在所有 actuator 設定低 0.01 到 0.03（0.586、0.667、0.654、0.782），friction 沒有差別（0.505 到 0.533），見 `data/detect_next_state.csv`。
- 以測試資料本身選的最佳閾值（TPR − FPR 最大，oracle）也不高：actuator 調高 strong 的 TPR − FPR 為 0.44，調高 mild 為 0.25，調低 strong 為 0.29，調低 mild 為 0.17，friction 低於 0.07（`thr_j`、`tpr_j`、`fpr_j`）。
- Baseline 的 test FPR 在 p95 為 5.7%，p99 為 1.2%，略高於名目值，因為 validation 只有 12 個 episode，且用於選模型。

### 與 FB Bellman 殘差比較

FB 殘差的 AUROC 來自 T4.1.2 REPORT（軌跡版 detector，3 個 seed 平均），動力學模型為本任務。

| 設定 | FB 殘差 $D$ | 動力學模型 $\lVert e\rVert_2$ | 動力學模型 $D_{dyn}$ |
|---|---|---|---|
| Actuator — Mild（調低） | 0.477 | 0.608 | 0.582 |
| Actuator — Strong（調低） | 0.474 | 0.695 | 0.669 |
| Actuator — Mild（調高） | 0.551 | 0.667 | 0.644 |
| Actuator — Strong（調高） | 0.617 | 0.796 | 0.795 |
| Friction（四個設定） | 0.502 到 0.510 | 0.506 到 0.532 | 0.508 到 0.537 |

## 聚合多步後的偵測

單筆轉移的 baseline 誤差有重尾，所以把一個 episode 切成 $K$ 步一段（不重疊），取每段 $\lVert e\rVert_2$ 的平均，再比較。閾值為 validation 的 baseline 窗平均的第 99 百分位數。表為預測 $\Delta s$ 的模型的 AUROC。

| 設定 | $K=1$ | $K=10$ | $K=50$ | $K=100$ | $K=333$ | $K=999$ |
|---|---|---|---|---|---|---|
| Friction — Mild（調低） | 0.510 | 0.521 | 0.533 | 0.535 | 0.545 | 0.573 |
| Friction — Strong（調低） | 0.516 | 0.534 | 0.554 | 0.559 | 0.575 | 0.617 |
| Actuator — Mild（調低） | 0.608 | 0.654 | 0.706 | 0.723 | 0.770 | 0.824 |
| Actuator — Strong（調低） | 0.695 | 0.785 | 0.863 | 0.882 | 0.944 | 0.992 |
| Friction — Mild（調高） | 0.506 | 0.511 | 0.518 | 0.521 | 0.529 | 0.547 |
| Friction — Strong（調高） | 0.532 | 0.557 | 0.580 | 0.588 | 0.608 | 0.663 |
| Actuator — Mild（調高） | 0.667 | 0.751 | 0.828 | 0.851 | 0.915 | 0.967 |
| Actuator — Strong（調高） | 0.796 | 0.911 | 0.961 | 0.973 | 1.000 | 1.000 |

整個 episode（$K=999$）的平均 $\lVert e\rVert_2$，在 60 個 test episode 上的最小、中位數與最大值：

| 設定 | 最小 | 中位數 | 最大 |
|---|---|---|---|
| Baseline | 3.83 | 4.62 | 6.43 |
| Friction — Mild（調低） | 3.92 | 4.76 | 6.58 |
| Friction — Strong（調低） | 3.99 | 4.85 | 6.68 |
| Actuator — Mild（調低） | 4.83 | 5.60 | 7.26 |
| Actuator — Strong（調低） | 6.05 | 6.80 | 8.46 |
| Friction — Mild（調高） | 3.88 | 4.75 | 6.42 |
| Friction — Strong（調高） | 4.10 | 4.95 | 6.77 |
| Actuator — Mild（調高） | 5.49 | 6.51 | 8.58 |
| Actuator — Strong（調高） | 8.32 | 9.58 | 11.92 |

- **Actuator 大幅偏移可偵測：** 調高 strong 的 episode 平均與 baseline 完全不重疊，調低 strong 與調高 mild 只有輕微重疊。調低 mild 重疊較多，AUROC 為 0.824。
- **Friction 不可偵測：** 整個 episode 的 AUROC 也只有 0.547 到 0.663。Friction 造成的 $s'$ 差異（0.17 到 0.65）比模型的 baseline 誤差（中位數 3.76）小一個數量級以上，平均不能消除這個差距。
- **閾值不穩定：** 以 validation 的第 99 百分位數為閾值，test baseline 窗的誤報率為 $K=1$ 時 1.2%、$K=10$ 時 0.9%、$K=50$ 時 1.1%、$K=100$ 時 3.3%、$K=333$ 時 7.8%、$K=999$ 時 11.7%。窗越長，validation 可用的窗口越少，百分位數估計越不穩。另外各 baseline episode 的平均誤差本身就有差異（3.83 到 6.43）。各 $K$ 的閾值與偵測率見 `data/windows.csv`。
- 同一 episode 的窗口共用偏移與策略，不是獨立樣本，所以窗口層級的 AUROC 不能當作獨立樣本的統計。

## 限制

- 測試是把 baseline rollout 的 $(s,a)$ 放進偏移環境走一步，不是讓策略在偏移環境裡實際執行。真實部署時偏移還會改變策略走到的狀態。
- 只有一個 baseline 策略（walk 的 $z_r$）、靜態偏移（friction 與 actuator gear）、一個訓練 seed。其他類型的偏移（質量、重力、感測器雜訊、隨時間變化的偏移）沒有測。
- 只有一個混合三個 seed 資料的模型。AUROC 區間只反映測試 episode 的抽樣，不含模型訓練的變異。預測 $\Delta s$ 與預測 $s_{t+1}$ 的差距（$R^2$ 差 0.007、AUROC 差 0.01 到 0.03）是否超過訓練變異，沒有驗證。
- 重新訓練不保證逐位元相同，`data/inputs/` 是某一次訓練的結果。

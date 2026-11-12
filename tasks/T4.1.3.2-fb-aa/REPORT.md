# T4.1.3.2 結果：單步轉移是否含有偵測 friction 偏移的資訊

**結論：** 單步轉移 $(s,a,s')$ 含有區分 friction 偏移的資訊，T4.1.2.1 偵測不到是分數設計的問題。直接訓練分類器區分 baseline 與偏移的 $s'$，friction strong 的 AUC 從 0.515 升到 0.751，actuator strong 從 0.697 升到 0.961。Friction 只在約 55% 的步驟改變 $s'$，只看這些步驟時分類器的 AUC 為 0.83 到 0.88，正確率 73% 到 78%，仍有約四分之一的 baseline 步驟被誤判，所以可靠的 friction 偵測需要多步累積。

## 設定

資料是 T4.1.2 的配對單步資料：3 個 FB 訓練 seed 的 walk $z_r$ policy 在 baseline 環境各跑 20 個 episode，每一步的物理狀態與動作放進 baseline 與各偏移環境只走一步。所以每對樣本的 $(s,a)$ 相同，只有 $s'$ 來自不同動力學。偏移設定同 T4.1.2：T3.2 的 friction 與 actuator 的 mild、strong（調低），以及 T4.1.2 `configs/` 的四個調高設定。

- **舊方法：** T4.1.2.1 的動力學模型誤差 $\lVert e\rVert_2$，$e=\Delta s-f(s,a)$，$f$ 只用 baseline 轉移訓練。
- **新方法：** 每個偏移設定各訓練一個分類器，輸入 $(s,a,s'-s)$，標籤為 baseline 或偏移。MLP 為 3 層、寬度 512 的 ReLU，Adam（學習率 $10^{-3}$）、batch 1024、60 epoch。60 個 episode 中隨機 45 個訓練、15 個測試（種子 0）。

兩者都在同樣 15 個測試 episode（14,985 對）上評估。分類器訓練時看過偏移資料，偵測器上線時沒有，所以分類器的結果是單步資訊量的參考，不是可部署的偵測器。

AUC 為隨機一筆偏移樣本的分數高於隨機一筆 baseline 樣本的機率，0.5 為無法區分。正確率以 logit > 0 判為偏移，為偏移被判對與 baseline 被判對的比例平均。

## 單步的分類效果

| 環境 | 舊方法 AUC | 新方法 AUC | 新方法正確率 |
|---|---|---|---|
| Friction Mild | 0.509 | 0.712 | 62.7% |
| Friction Strong | 0.515 | 0.751 | 65.4% |
| Actuator Mild | 0.610 | 0.914 | 82.1% |
| Actuator Strong | 0.697 | 0.961 | 88.8% |

調高的四個設定：

| 環境 | 舊方法 AUC | 新方法 AUC | 新方法正確率 |
|---|---|---|---|
| Friction Mild（調高） | 0.506 | 0.513 | 50.9% |
| Friction Strong（調高） | 0.532 | 0.734 | 64.2% |
| Actuator Mild（調高） | 0.669 | 0.932 | 84.5% |
| Actuator Strong（調高） | 0.798 | 0.976 | 91.7% |

- 除了 friction 調高 mild，分類器的 AUC 比舊方法高 0.18 到 0.31。
- Friction 調高 mild 的 $s'$ 差異是所有設定中最小的（中位數 0.17），分類器的訓練 AUC 到第 60 epoch 只有 0.518 且仍在上升，無法判斷是訓練不足還是單步真的無法區分。

## Friction 只在部分步驟起作用

Actuator 偏移在每一步都改變 $s'$。Friction 偏移在 55% 到 56% 的步驟改變 $s'$，其餘步驟的 $s'$ 與 baseline 完全相同（差異小於 $10^{-6}$）。這些步驟推測是腳不在地上的步驟，資料中沒有接觸資訊，未直接驗證。差異也集中在少數步驟：$\lVert s'_{\text{shift}}-s'_{\text{base}}\rVert^2$ 最大的 10% 步驟，在 friction 占總量的 75% 到 90%，在 actuator 只占 34% 到 35%。

只看 friction 有作用的步驟（約 8,200 對）：

| 環境 | 舊方法 AUC | 新方法 AUC | 新方法正確率 | 偏移被判對 | baseline 被誤判 |
|---|---|---|---|---|---|
| Friction Mild | 0.517 | 0.833 | 73.2% | 76.8% | 30.4% |
| Friction Strong | 0.528 | 0.881 | 78.1% | 80.6% | 24.4% |
| Friction Strong（調高） | 0.557 | 0.851 | 75.4% | 80.6% | 29.8% |

- 分類器在這些步驟的 AUC 比全部步驟高 0.12 到 0.13。不起作用的步驟兩邊輸入相同，只能猜。
- 舊方法限制在這些步驟也幾乎沒有提升（0.517 到 0.557）。偏移後的誤差為 $e+\delta$，$\delta=s'_{\text{shift}}-s'_{\text{base}}$：

$$\lVert e+\delta\rVert^2=\lVert e\rVert^2+2e^\top\delta+\lVert\delta\rVert^2$$

$e$ 與 $\delta$ 的方向無關時，分數只多了 $\lVert\delta\rVert^2$。在有作用的步驟上，$\lVert\delta\rVert/\lVert e\rVert$ 的中位數在 friction 為 0.16 到 0.61，actuator 為 0.78 到 1.83，所以 friction 造成的分數變化小於 baseline 本身的變異。MuJoCo 是確定性的，$e$ 完全來自模型的近似誤差。

## 限制

- 分類器訓練時看過偏移資料，只能說明單步的資訊量，不代表只用 baseline 資料訓練的偵測器能達到同樣的效果。
- 「有作用的步驟」以 $s'$ 是否改變判定，上線時無法得知，除非能從狀態推得接觸。
- 只有一個分類器訓練種子與一種 episode 切分，AUC 沒有區間。
- 評估仍是把 baseline rollout 的 $(s,a)$ 放進偏移環境走一步，不是讓 policy 在偏移環境中執行。

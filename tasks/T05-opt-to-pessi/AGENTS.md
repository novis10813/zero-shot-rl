## Task description

Use the implicit reward definitions already derived and validated in **Task 3**.

Keep the **FB representation and policy fixed**. Do not retrain (F), (B), or (\pi).

For each implicit reward from Task 3: $r + r_i^{\text{implicit}}(s,a)$

The main research question is:

> **Do different offline-RL biases correspond to different directions in the same FB latent space, and can these directions be used to select policies along a pessimistic-to-optimistic spectrum at test time?**
>

## Sub-tasks

- [x]  Compute (z) for the raw reward and each implicit reward from different algo of previous tasks.
- [x]  Evaluate the fixed FB policy with each (z).
- [x]  Compare policy performance and behavior across different reward definitions.
- [x]  Visualize and compare the latent vectors using cosine similarity, distance, and PCA (Standard FB only; one-step FB excluded).
- [x]  For methods with tunable bias strength, test several hyperparameter values and check whether (z) changes smoothly. (Existing saved Standard FB samples only; no claim of global continuity.)
- [x]  Check whether pessimistic, neutral, and optimistic methods approximately form an ordered direction in latent space. (A shared ordered axis is not established by the current geometry and AX diagnostics.)
- [ ]  If a clear direction exists, interpolate between pessimistic and optimistic (z)'s.
- [ ]  Evaluate whether interpolation produces a smooth change in policy behavior.
- [ ]  Determine whether a scalar (\lambda) can be used as a test-time knob to choose the desired level of conservatism.

## 現有實驗的問題可回答程度

以下是現有實驗的初步判讀，不是正式報告。Standard FB 與 one-step FB 應分開呈現，不混合統計。

| 問題 | 可回答程度 | 應怎麼回答 |
|---|---|---|
| **Q1：不同 implicit reward 是否產生不同 latent？** | 可回答 | 在同一模型 checkpoint、同一任務內，以僅使用原始任務 reward 算出的 latent 為基準，比較各 latent 與基準的夾角，以及移除基準方向分量後的偏移方向之餘弦相似度。另將推論資料切成互不重疊的子集，各自重算原始 reward 的 latent，以子集估計與完整資料估計的夾角作為有限資料造成的變動尺度。 |
| **Q2：不重訓 FB，換 latent 能否改變表現與行為？** | 可回答 | 用相同初始狀態配對比較累積任務 reward、水平速度、軀幹直立程度、軀幹低於高度門檻的時間比例、動作分量接近控制上下限的比例，以及標準化後的狀態—動作與最近訓練樣本之距離，再用既有 rollout 影片描述行為。 |
| **Q3：不同方法是否共享一條悲觀—樂觀方向？** | 可做有界判斷 | 在同一模型 checkpoint、同一任務內，移除各 latent 沿原始 reward latent 的分量，將剩餘偏移方向正規化後比較餘弦相似度與方向矩陣的奇異值。再移除「對所有推論樣本加上相同 reward」所產生的偏移方向分量，比較剩餘結構。主成分分析只能輔助視覺化，不能單獨證明共同軸。 |
| **Q4：bias strength 增加時，latent 是否平滑變化？** | 部分可回答 | 畫出 implicit reward 係數與相對原始 reward latent 夾角的關係，以及相鄰係數下的方向變化，標明搜尋未括住目標或求解未收斂的目標條件；中間失敗係數未保存，不能精確標示失敗係數區間。目前的係數是為了達到指定夾角而選出的離散取樣，不能宣稱全域連續。 |
| **Q5：沿共同方向移動，行為是否平滑、有序？** | 部分可回答 | Standard FB 已有沿候選悲觀—樂觀軸朝正負方向旋轉 latent 的角度掃描。該軸由不同 implicit reward 的 latent 偏移方向構造，並非已驗證的共同軸。逐訓練隨機種子、逐任務檢查角度與行為指標的曲線及單調性，且需區分沿構造軸旋轉與悲觀、樂觀端點的直接插值。 |
| **Q6：能否用單一 scalar 選擇保守程度？** | 尚不支持 | 已證明可以用 latent 旋轉角度選擇不同 policy，但「保守」必須先以可量測的行為或風險指標定義，且多個指標、任務與訓練隨機種子要有一致排序。 |
| **Q7：成功率／失敗率是多少？** | 不納入交付 | Walker 任務未定義二元成功條件，不額外設定成功門檻。改以原始任務 return、行為指標與不同 latent 的 rollout 影片比較；低軀幹高度時間比例不當作失敗率。 |

Q4–Q6 的既有資料診斷與重現命令已加入報告及 README。AX 旋轉不是端點插值，因此不勾選插值及其行為平滑性兩項。共同軸與可量測保守程度排序的前提尚未建立，也不將 scalar 控制保守程度標為完成。

## Delivery

- Comparison table of:
    - reward type;
    - bias strength;
    - return;
    - latent similarity;
    - main behavior characteristics.
- PCA / latent-space visualization.
- Plot of policy performance vs. optimism–pessimism level.
- Representative rollout videos.
- Short conclusion answering:

> **Can different offline-RL biases be represented as a common pessimistic-to-optimistic direction in FB latent space?**
> **Can we choose different policy behaviors at test time by moving along this direction without retraining FB?**

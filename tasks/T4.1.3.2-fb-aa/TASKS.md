# T4.1.3.2 FB-AA 任務紀錄（暫存）

工作筆記，任務結束時依 AGENTS.md 的檔案規則在 cleanup commit 刪除。

## 1. 回顧之前的 OOD detection module

**背景：** 之前的實驗中，actuator 的 distribution drift 可以被偵測，friction 則不行。State 內含關節角度等量，actuator 的影響可以直接從 state 預測，friction 在 state 中沒有對應的量。

相關的既有結果與程式：

- T4.1.2：FB Bellman 殘差的 Mahalanobis 距離（`zero_shot_rl/latent/shift_detector.py`）。Friction AUROC 0.497 到 0.510，actuator 調低方向相反。
- T4.1.2.1：動作條件 MLP 動力學模型（`zero_shot_rl/latent/dynamics_model.py`）。Actuator AUROC 0.61 到 0.80，friction 找不到可用閾值。
- T4.1.2.2：CVAE ensemble（`zero_shot_rl/latent/cvae_model.py`、`cvae_score.py`）。Actuator 最好 0.871，friction 0.51 到 0.54。
- 已知數據：friction 的單步 $s'$ 差異中位數為 0.17 到 0.63，而 baseline MLP 的 $\lVert e\rVert_2$ 中位數為 3.76。單步訊號小於模型本身的誤差。

**任務：**

1. 辨明之前判斷 distribution drift 的方式是否有問題（例如配對單步評估、只看單筆轉移、分數的正規化方式）。
2. 找出更好的 metric 來判斷 drift 是否發生。

**進度（zero-shot-rl-291）：** 單步分類器（每個偏移設定一個，輸入 $(s,a,s'-s)$）的 test AUROC：friction 0.71 到 0.75（調高 mild 0.51），只看 $\delta\neq0$ 的步驟 0.83 到 0.88；actuator 0.91 到 0.98。T4.1.2.1 的 $\lVert e\rVert_2$ 在同樣 15 個 test episode 上，friction 0.51 到 0.53，actuator 0.61 到 0.80。Friction 只在約 55% 的步驟改變 $s'$。

## 2. 基於之前的 module 建立新的 detector

**背景：** 其他論文用 $(s,a)$ 序列學習環境動力學，用來偵測或預測 distribution drift。

**任務：** 建立新的 detector，能偵測環境中所有的 distribution drift（包含 friction）。

## 3. 建立基本的 FB-AA 算法

**背景：** FB 估計 policy 未來對各 state 的 discounted visitation，並以 Forward–Backward 分解近似：

$$M_{P_0}^{\pi_z}(s,a,ds')=\sum_{t=0}^{\infty}\gamma^t P(s_t\in ds'\mid s_0=s,a_0=a,\pi_z)\approx F_0(s,a,z)^\top B_0(s')\rho(ds')$$

當 dynamics 從 $P_0(s'\mid s,a)$ 變成 $P_1(s'\mid s,a)$，$M_{P_0}^{\pi_z}\neq M_{P_1}^{\pi_z}$。即使 task reward 與 $z_r$ 不變，原 FB policy 也可能無法達到預期的控制效果。

**核心想法：** 不改 task latent $z_r$，而是修正 action，使 shifted 環境的 transition 接近原環境的 transition。

Action transformation：

$$a'=T_\theta(s,a),\qquad a=\pi(s,z_r)$$

目標為 $P_1(s'\mid s,T_\theta(s,a))\approx P_0(s'\mid s,a)$。

利用 pretrained FB 的 $B_0$ 定義兩個 one-step feature predictor：

$$g_0(s,a)=\mathbb E_{P_0}[B_0(s')\mid s,a],\qquad g_1(s,a)=\mathbb E_{P_1}[B_0(s')\mid s,a]$$

$g_0$ 以原環境資料訓練，$g_1$ 以 test-time trajectory 學習新的 dynamics。

$$\theta^*=\arg\min_\theta\mathbb E_{s,a}\left[\lVert g_1(s,T_\theta(s,a))-g_0(s,a)\rVert^2+\lambda\lVert T_\theta(s,a)-a\rVert^2\right]$$

不更新 FB backbone，也不需要 online reward，只用 transition feedback 找 action correction。

**Research hypothesis：** Can a frozen FB representation guide reward-free action adaptation under unseen dynamics by aligning predicted next-state features?

**下一階段：**

1. 檢驗 $B_0(s')$ 是否保留 actuator 與 friction shift 造成的 transition 差異。
2. 比較 single-step 與 history-conditioned dynamics predictor。
3. 確認這些訊號能否支持 action adaptation。

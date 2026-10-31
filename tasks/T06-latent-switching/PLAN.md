# T06 工作計畫：固定 FB 下以 latent 切換樂觀與悲觀動作

本檔是任務期間的工作筆記，任務結束時依 AGENTS.md 的檔案規則刪除或併入 README.md。

## 目標

用同一個固定的 standard FB，在每一步產生樂觀動作與悲觀動作兩個候選，再用一個信任訊號決定用哪一個。不重訓 FB，所有判斷都在 evaluation time 完成。

## 流程

```
[離線，一次性]
  FB 訓練完成（T04 的 FB checkpoint）  得到 F(s,a,z)、B(s)、π(s,z)
  T05 的推論                           得到 z_cql = z_r 加上 CQL 隱含懲罰的偏移

[每一步]
  a_opt  = π(s, z_r)      樂觀候選
  a_pess = π(s, z_cql)    悲觀候選
  信任訊號 u(s, a_opt)    FB 的估值在這裡是否可靠
  決策規則                 u 低用 a_opt，u 高用 a_pess

[可選外層，需少量線上互動]
  每個 episode 選一組設定（κ、候選 z），以 GP 與 UCB 挑選
```

## 四個層次

A. 候選來自 z 空間。樂觀端 $z_r$，悲觀端 T05 的 CQL 修正 $z_{cql}$（目標夾角 10 度）。$z_{cql}$ 只是 CQL 隱含懲罰的投影，是否更保守需驗證。

B. 信任訊號，階段 0 比較哪個最能預測過估計 $(\hat Q - G)$：

| 訊號 | 定義 | 成本 |
|---|---|---|
| head 差距 | $\lvert F_1^\top z_r - F_2^\top z_r\rvert$ | 零 |
| 隱含懲罰 | $w = F(s,a,z_r)^\top (z_r - z_{cql})$ | 零 |
| 常數 reward 一致性 | $F^\top z_{const}$ 偏離 $1/(1-\gamma)$ 的程度 | 零 |
| RND 式新穎度 | 在資料集 $(s,a)$ 上訓練的預測誤差 | 低，不動 FB |
| 更多 F head | 共享 torso，需重訓 FB | 中 |

最近鄰距離只當對照。

C. 決策規則，依 Neggatu 等（arXiv 2503.12222）Algorithm 1，BC 換成悲觀 latent：

$$Q_{opt} = Q_r(s,a_{opt}) - \kappa\,u(s,a_{opt}),\qquad Q_{pess} = Q_r(s,a_{pess})$$

$Q_{opt} \ge Q_{pess}$ 用 $a_{opt}$，否則用 $a_{pess}$。省略論文的 $\sigma_D$，因為它需要 return。

D. 外層選擇（後期），依 Konyushkova 等（arXiv 2106.10251）。候選為不同 $z_k$，OPE 為 $F(s,a,z_k)^\top z_r$，kernel 為資料集狀態上的動作距離。

## 階段

| 階段 | 內容 | 問題 | 需訓練 |
|---|---|---|---|
| 0 診斷 | 沿用 T04 probe 的作法，量各訊號與 $(\hat Q - G)$ 的關係，並檢查 $Q_{cql} \le G \le Q_r$ 的比例 | 有沒有訊號，區間是否括住真實值 | 否 |
| 1 切換 | 在凍結 FB 上執行規則 C，掃 κ | gate 是否贏過固定策略 | 否 |
| 2 補強 | 視階段 0 結果加入 RND 或更多 head | 更好的訊號能否改善 | 是，新 module |
| 3 外層 | 規則 D | 少量互動能否補上差距 | 否 |

階段 1 對照組：純 FB、永遠用 $z_{cql}$、不看訊號的固定機率混合、每任務最佳固定 κ。真的 CQL 與 TD3 policy（`agents/cql`、`agents/td3`）當上界，並檢查 $z_{cql}$ 是否近似 CQL 的行為。

## 輸入

- FB checkpoint（T05 用的 best checkpoint）：`artifacts/T04-mcfb-pessimism/zsrl-32/800000.pt`、`zsrl-41/680000.pt`、`zsrl-42/200000.pt`，對應訓練 seed 0、1、2，雜湊見 `tasks/T05-opt-to-pessi/configs/sources.json`。
- latent：`tasks/T05-opt-to-pessi/data/inputs/zs_fb_zero-shot-rl-{30,31,32}.npz`，讀取工具 `zero_shot_rl/latent/archive.py`。
- Probe 作法：`tasks/T04-mcfb-pessimism/pessimism_probe.py`（重設模擬器到資料狀態、執行動作、以 actor 跟隨 300 步的 MC return，$\gamma = 0.98$，reward 單位換算 $c_r$）。

## 主要風險

1. 信任訊號不能預測 FB 在哪裡過估計。階段 0 驗證，失敗則後續階段不成立。
2. 悲觀動作在這些地方並沒有比較好。T05 顯示 CQL 修正在 stand 與 run 會降低 return。

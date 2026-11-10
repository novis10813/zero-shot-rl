# T4.1.3.0 工作筆記：決策與待辦

這是給接手執行實驗的 session 看的工作筆記，任務結束時依 AGENTS.md 的 file rule 刪除。

## 方法

Frozen FB（F、B、actor、z 全部凍結），在固定 task latent $z$ 下線上訓練 $Q_\phi$，動作在 $a_0=\pi(s,z)$ 附近抽候選，用修正後的 Q 挑最高分。

- Critic 有兩種。Direct：$Q_{\text{adapt}}=Q_\phi$，是對照組。Residual：$Q_{\text{adapt}}=c\min_k F_k^\top z+Q_\phi$，最後一層零初始化，是主方法。
- Actor 有兩種。不訓練 actor（已實作，`zero_shot_rl/online/value_correction.py`）與微調 actor（尚未實作）。
- 候選半徑 `--radius`（$\epsilon$）限制候選與 $a_0$ 的 L2 距離，預設不限制。$\epsilon=0$ 等於 frozen FB。

## 已決定

- 環境用 Mild，不用 Strong：clean、`friction_mild`、`actuator_mild`（`tasks/T3.2-ood-robustness/configs/`）。
- 只報 walk。每個 job 用 `--tasks walk`。
- FB 用 T04 的 3 個 checkpoint（seed 0、1、2），訓練 500 個 episode，每 50 個 episode 測一次（`--eval_every 50`），每次 20 個測試 episode。
- 報告的 checkpoint 一律是第 500 個 episode，不挑最佳點。一定要與同一個 FB seed 的 frozen FB 配對比較。
- radius 的比較只用 Residual。

## 要做的三張表

1. **方法比較。** 列是 frozen FB、Direct、Residual（微調 actor 實作後再加兩列）。欄是 3 個環境。每格是 3 個 FB seed 的平均 return 與 95% bootstrap 區間（對 seed 重抽樣）。每個 seed 的值是 20 個測試 episode 的 IQM。另加一欄「相對同 seed frozen FB 的配對差」。`iqm` 從 `analysis.table6_aggregate` import，讀 `eval.csv` 的程式放在 task 目錄。
2. **radius 的影響（Residual）。** 欄是 $\epsilon\in\{0.1,0.2,0.4\}$ 與不限制，$\epsilon=0$ 就是 frozen FB。每格報第 500 個 episode 的 return，以及三個動作指標：`shift`、`shift/ε`、選到 $a_0$ 的步數比例。再報 `gain`（Q 宣稱的改善）與實際 return 差。半徑小於約 0.49 時多數候選被縮到球面，`shift` 幾乎等於 $\epsilon$，所以必須看 `shift/ε` 與選到 $a_0$ 的比例。
3. **收斂速度。** 第 50、100、200、500 個 episode 後的 return，加上各存檔點測試 return 的平均（曲線下面積）。不用「第一次超過 frozen FB」，因為測試 return 雜訊約 ±50。早期（50 以下）的曲線用 `train.csv` 的訓練 return。

## 實驗

| 組 | 內容 | job 數 |
|---|---|---|
| A（表 1、3） | Direct、Residual × 3 個環境 × 3 個 FB seed，不限制半徑 | 18 |
| B（表 2） | Residual × $\epsilon\in\{0.1,0.2,0.4\}$ × clean、`friction_mild`、`actuator_mild` × 3 個 FB seed | 27 |

每個 job 是 1 個 task（walk）、500 個 episode。上一批 pilot 的時間（100 個 episode 約 32 分鐘 Direct、45 分鐘 Residual，兩個 job 共用一張 GPU）推算，一個 job 約 3 到 4 小時。每個 job 約 8GB RAM、GPU 不到 1.5GB。idlab1 有 24 核、123GB RAM、2 張 GPU，同時最多跑 6 個 job。

組 B 包含 clean：沒有 drift 時，可以看學到的 Q 能把原本 FB 的 F 改善到什麼程度。

## 開跑前的程式待辦

1. 輸出加「選到 $a_0$ 的步數比例」欄位（`act` 的 `index == 0`），`eval.csv` 與 `train.csv` 都要。組 A 也要有，所以必須在送出任何組之前完成。
2. 新增 `configs/` 下的 sweep：組 A 與組 B，`on: idlab1`，`--tasks walk --episodes 500 --eval_every 50`。
3. 組 B 的 job 名稱要含 $\epsilon$，避免輸出目錄互相覆蓋。
4. 分析程式（表 1 到 3）放在 task 目錄，讀 `data/inputs/`。

## 現況

在 idlab 與 idlab1 上已經在跑、早於「改用 Mild」這個決定的 job：

| Job | 內容 | 程式版本 |
|---|---|---|
| `zsrl-ovc-4`、`zsrl-ovc-5` | idlab，Direct 與 Residual，walk，Actuator Strong，seed 0，500 個 episode | 舊輸出格式（`moved`） |
| `zsrl-ovc-12` 到 `17` | idlab1，clean、Friction Strong、Actuator Strong 各 Direct 與 Residual，4 個 task，seed 0，500 個 episode | 新輸出格式（`shift`、`gain`） |

`zsrl-ovc-12` 到 `17` 已由使用者決定停掉（`lab stop`，狀態 cancelled），其輸出不使用。`zsrl-ovc-4`、`zsrl-ovc-5` 仍在 idlab 上跑，結果不進三張表。

## 在 ssh novis 上跑實驗的注意事項

- **lab 的 job 前綴要與本機不同。** 這台機器的 lab 專案前綴是 `zsrl-ovc`，job 編號到 17。兩台控制機各有自己的資料庫，前綴相同會撞到伺服器上的 `runs/<id>` 目錄（之前 `zero-shot-rl-1` 就是這樣失敗的）。novis 的前綴請改成別的，例如 `zsrl-ovn`，修改 `~/.lab/config.yaml` 前要先給使用者看。
- idlab1 已有 3 個 FB checkpoint 與資料集（`~/zsrl-artifacts/T04-mcfb-pessimism/`、`~/zsrl-datasets/walker/rnd/dataset.npz`）。idlab 只有 seed 0 的 checkpoint。
- `lab run` / `lab sweep` 執行已 push 的 commit，送之前要先 push，且工作目錄不能有未追蹤檔案（`.pi/` 要放進本機的 git exclude）。
- 不要在本機（novis 以外的這台）跑 MuJoCo。
- 不要在視窗內 sleep 超過 10 秒，等待用 `spawn_background` 的 `lab wait`。
- 送之前向使用者顯示超參數，並確認記憶體不會塞爆 idlab1。

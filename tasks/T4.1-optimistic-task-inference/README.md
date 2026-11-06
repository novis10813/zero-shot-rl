# T4.1：OpTI-BFM 線上任務推論的重現

本任務在本 repo 的 FB 上重現 Rupf et al., "Optimistic Task Inference for Behavior Foundation
Models"（arXiv 2510.20264）。結果見 [REPORT.md](REPORT.md)，本文件說明程式、設定與重現指令。

作者公開了 JAX 程式（github.com/ThomasRupf/opti-bfm，commit 19a5e22），但他們用自己的 FB，
在完整的 ExORL 資料上訓練 2M 步，也沒有公開 checkpoint。本任務把方法移植到 PyTorch，直接
用 T04 的三個標準 FB checkpoint（Walker RND-100k，seed 0、1、2），只在 Walker 上評估。

## 程式

| 路徑 | 內容 |
|---|---|
| `zero_shot_rl/online/belief.py` | 帶 $\lambda$ 正則化的最小平方法，也就是 Bayesian linear regression 的後驗 $\mathcal{N}(\hat z, V^{-1})$，一次處理 n 個 run |
| `zero_shot_rl/online/selectors.py` | FB（離線 z）、Random、UCB（OpTI-BFM）、TS（OpTI-BFM-TS）四種選 $w$ 的規則 |
| `zero_shot_rl/online/fb.py` | 把 FB checkpoint 轉成論文的 $\phi$、$\psi$ 與 policy |
| `zero_shot_rl/eval/online_inference.py` | 連續多個 episode 的線上推論；也存下前 n 筆 transition 的最小平方 $z$，以及 `--predictions_dir` 的 belief 與預測 return |
| `zero_shot_rl/eval/latent_ranking.py` | 固定候選 latent 的預測 return 與實際 return，用來看 $\psi$ 的排序與校準 |
| `zero_shot_rl/eval/fixed_z_returns.py` | data efficiency：評估由前 n 筆 transition 推出的固定 $z$ |
| `tests/test_online_inference.py` | belief 與 selector 的元件測試 |
| `run_online.sh`、`run_fixed_z.sh`、`run_ranking.sh` | 單一 seed 的啟動腳本，供 `configs/*.yaml` 的 `lab sweep` 呼叫 |
| `analyze.py` | 從 `data/inputs/` 產生 `data/*.csv`、`figures/*.png` 與 REPORT.md 的數字 |

### 與論文的對應

FB 的特徵與 successor features 依論文附錄 D.1：

$$\phi(s) = \mathrm{Cov}_{\mathcal D}(B)^{-1} B(s),\qquad \psi(s, w) = F\big(s, \pi(s, w), w\big)$$

$\mathrm{Cov}_{\mathcal D}(B)$ 以 100k 訓練子樣本的 next observation 計算（論文用 50k 筆）。
$w$ 是 reward 的係數，policy 與 $F$ 都在 $\sqrt{d}\,w/\lVert w\rVert$ 上執行，也就是 FB 訓練時用的球面。
$\pi$ 為 deterministic 的 actor mean。$F$ 的兩個 head 都保留，UCB 取兩者中較小的分數，與作者程式相同。

每一步：

1. 選 $w$。UCB 從半徑 $2\beta$ 的信賴橢球 $\lVert w - \hat z\rVert_V \le 2\beta$ 均勻抽 128 個候選，
   取 $\min_{\text{head}} \psi(s,w)^\top \hat z + \beta \lVert \psi(s,w) \rVert_{V^{-1}}$ 最大者（論文 eq. 11、12 與註 6）。
   TS 從 $\mathcal N(\hat z, V^{-1})$ 抽一個。Random 在球面上均勻抽。FB（離線 z） 固定為離線推論的 $z_r$。
2. 執行動作，得到下一個狀態 $s'$ 和 reward $r$，再用 $(\phi(s'), r)$ 更新 $V$ 與 $\hat z$。

UCB 用 $\lambda = 1$、$\sigma = 1$。TS 用先驗 $\mathcal N(0, I)$，觀測除以 $\sigma$。
-EP 變體只在 episode 開始時選 $w$（`@1000`）。

與作者程式的差異：

- 取樣。作者的 `BLR.transform` 用 $R^{-\top}\xi$（$V = R^\top R$），它的共變異數是 $(RR^\top)^{-1}$ 而不是 $V^{-1}$。
  本實作照論文用 $L^{-\top}\xi$（$V = LL^\top$），共變異數為 $V^{-1}$。
  這會影響 TS 的樣本與 UCB 的候選點，但不影響 UCB 的評分。
- FB（離線 z）的 $z_r$ 沿用本 repo 的離線推論（`zero_shot_rl.latent.task_z`）：
  從同一個 100k 子樣本抽 10k 筆有 reward 的 transition，$z_r \propto \mathbb E[B(s) r(s)]$。
  論文則用 50k 筆、對 $\phi$ 做最小平方。兩者在資料分佈與訓練分佈相同時方向一致。
- -EP 變體在新 episode 的 reset 狀態選 $w$。作者程式是在上一個 episode 的最後一個狀態選。
- data efficiency 的 RND 來源是從完整 10M 的 RND 資料抽整條 episode。FB 只用其中 100k 筆訓練。

## 設定

- 以離線推論的 $z_r$ 執行 FB 的規則在程式中叫 `fb`（`--methods fb`）。已存的 `data/inputs/` 與 `exp/T4.1-*` tag 對應的 job 是用舊名 `oracle` 跑的，`analyze.py` 讀取時把它視為 `fb`。

- 每個 task 10 個 run，每個 run 連續 10 個 1000 步的 episode（-EP 為 50 個）。
- run i 用以 i 為 seed 的環境，每個 episode 前 reset，因此不同方法在 run i 的初始狀態相同。
- belief 在同一個 run 的 episode 之間保留，換 task 時重設。
- 超參數依論文附錄 F：$\beta \in \{1, 0.1, 0.001\}$，$\sigma \in \{0.1, 0.01, 0.001, 0.0001\}$（論文的三個值加上作者 README 的 0.01）。
  取 10 個 episode 累積 return（四個 task、三個 seed 平均）最高者：$\beta = 0.001$，$\sigma = 0.1$。
  -EP、dynamics 與 data efficiency 都用這兩個值。
- dynamics shift 沿用 T3.2 的四個設定（`tasks/T3.2-ood-robustness/configs/`）：
  friction 0.1、0.02，actuator gear 76%、62%。reward 函數不變，FB（離線 z）的 $z_r$ 仍從原始資料推出。
- data efficiency：n ∈ {100, 200, 500, 1000, 2000, 5000, 10000}。
  每個來源、每個 run 各推一個 $z$（eq. 4，無正則化），各跑 2 個 episode，每個 n 共 20 個 episode。
  RND 每次抽 10 條 episode，取前 n 筆，reward 由 next state 的 physics 計算。

## 重現

所有指令從 repo 根目錄執行。資料集見 [T01 README](../T01-fb-reproduction/README.md) 第 1 節。
三個 checkpoint 是 `artifacts/T04-mcfb-pessimism/{zsrl-32/800000.pt, zsrl-41/680000.pt, zsrl-42/200000.pt}`。
啟動腳本預設從 lab 主機的 `$HOME/zsrl-artifacts/T04-mcfb-pessimism/` 與
`$HOME/zsrl-datasets/walker/rnd/dataset.npz` 讀取，可用 `ARTIFACTS`、`DATASET` 覆寫。

```bash
T=tasks/T4.1-optimistic-task-inference
lab sweep $T/configs/main.yaml          # FB（離線 z）、Random 與 beta / sigma grid
lab sweep $T/configs/episodic_ts.yaml   # OpTI-BFM-TS-EP
lab sweep $T/configs/episodic_ucb.yaml  # OpTI-BFM-EP
lab sweep $T/configs/dynamics_ts.yaml   # 四種 shift 下的 FB（離線 z）、Random、OpTI-BFM-TS
lab sweep $T/configs/dynamics_ucb.yaml  # 四種 shift 下的 OpTI-BFM
lab sweep $T/configs/fixed_z.yaml       # data efficiency，需要已提交的 data/inputs/main/*_z/
lab sweep $T/configs/predictions.yaml   # OpTI-BFM 與 OpTI-BFM-TS 重跑並記錄 belief 與預測 return，五種環境
lab sweep $T/configs/ranking.yaml       # 32 個固定候選 latent 的預測與實際 return，需要已提交的 data/inputs/predictions/
```

每個 job 把結果寫在工作目錄的 `out/`，用 `lab fetch <job> out/` 取回後放進 `data/inputs/`：

| 實驗 | 檔案 |
|---|---|
| main | `data/inputs/main/{base,ucb,ts}_s<seed>.csv`；被選中的 $z$ snapshot 在 `*_z/` |
| -EP | `data/inputs/episodic/ep_{ucb,ts}_s<seed>.csv` |
| dynamics | `data/inputs/dynamics/<shift>.{base,ucb}_s<seed>.csv` |
| data efficiency | `data/inputs/fixed_z/fixed_z_s<seed>.csv` |

單一設定也可以直接執行，例如：

```bash
PYTHONPATH=. MUJOCO_GL=egl uv run python -m zero_shot_rl.eval.online_inference \
  --checkpoint artifacts/T04-mcfb-pessimism/zsrl-32/800000.pt \
  --dataset_path datasets/walker/rnd/dataset.npz \
  --methods fb ucb:0.001 ts:0.1 --out out/runs.csv --z_dir out/z
```

表格與圖：

```bash
PYTHONPATH=. uv run python tasks/T4.1-optimistic-task-inference/analyze.py
```

它會寫出 `data/hyperparameters.csv`、`main.csv`、`episodic.csv`、`data_efficiency.csv`、
`ranking.csv`、`dynamics.csv`、`dynamics_summary.csv`、`dynamics_raw_returns.csv`、`dynamics_inference.csv` 與 `figures/` 的五張圖，並印出 REPORT.md 的數字。

每個 job 的 commit 與 log 記錄在 `exp/T4.1-<job>` tag。

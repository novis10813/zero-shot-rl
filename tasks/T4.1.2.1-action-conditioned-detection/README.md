# T4.1.2.1：動作條件動力學模型的偏移偵測

結果見 [REPORT.md](REPORT.md)，任務說明見 [TASK.md](TASK.md)。以 baseline 轉移訓練 MLP 預測 $\Delta s=s_{t+1}-s_t$，用預測誤差 $e=\Delta s-f_\theta(s,a)$ 偵測動力學偏移。測試資料與 T4.1.2 完全相同：同一個 $(s,a)$ 在 baseline 與八個偏移環境各走一步。

| 路徑 | 內容 |
|---|---|
| `zero_shot_rl/latent/dynamics_model.py` | $f_\theta$、按 episode 切 validation、early stopping、$\Sigma_0$；命令列訓練。`--target delta` 預測 $\Delta s$，`--target next_state` 預測 $s_{t+1}$ |
| `zero_shot_rl/latent/dynamics_score.py` | 對 `paired_step` 的輸出算 $\lVert e\rVert_2$ 與 $D_{dyn}$，堆疊三個 seed 的檔案 |
| `tests/test_dynamics_model.py` | split、ExORL 統計、兩種目標、Mahalanobis、存取的元件測試 |
| `run_dynamics.sh` | 完整流程（見下） |
| `analyze/detect.py` | AUROC、閾值的偵測率、oracle 閾值 → `data/detect_<run>.csv` |
| `analyze/prediction.py` | baseline 的 $R^2$、MSE、誤差百分位數、各維度 $R_j^2$ → `data/prediction.csv`、`data/prediction_dims.csv` |
| `analyze/windows.py` | 以 $K$ 步窗平均的偵測 → `data/windows.csv`、`data/episode_mean.csv` |
| `tests/test_detect.py` | 上述分析函式的測試 |
| `data/inputs/dynamics_scored_<run>.npz` | 每個設定的 $\lVert e\rVert_2$、$D_{dyn}$、$s'$ 差異、各維度 MSE，以及 validation 的 baseline 分數。`delta` 為預測 $\Delta s$ 的模型，`next_state` 為預測 $s_{t+1}$ 的模型 |

## 資料切分

- 訓練池：T06 的三個 FB checkpoint 各自在 baseline 跑的 episode 100 到 119，共 60 個 episode（59,940 筆）。只用這些互動轉移訓練。
- 訓練 47,952 筆（48 個 episode），validation 11,988 筆（12 個 episode，每個 seed 4 個，亂數種子 0）。Validation 的 episode 編號存在模型檔的 `meta["val_episodes"]`。
- 測試：三個 seed 各自的 episode 0 到 19（共 60 個 episode，59,940 對），與 T4.1.2 的配對資料相同。
- 閾值取 validation 中 baseline 分數的第 95 與 99 百分位數，不使用偏移資料。
- 輸入 $[s,a]$ 與目標（$\Delta s$ 或 $s_{t+1}$）用 ExORL walker rnd 全資料集的逐維平均與標準差標準化，ExORL 資料只用來算這些統計。

## 執行

```bash
CKPT_DIR=<含 zsrl-32.pt zsrl-41.pt zsrl-42.pt 的目錄> DATASET=<walker/rnd/dataset.npz> \
  bash tasks/T4.1.2.1-action-conditioned-detection/run_dynamics.sh
cp out/dynamics_scored.npz tasks/T4.1.2.1-action-conditioned-detection/data/inputs/dynamics_scored_delta.npz
```

預測 $s_{t+1}$ 的模型加 `TARGET=next_state`，輸出複製為 `dynamics_scored_next_state.npz`。`HIDDEN`、`LAYERS` 預設 512、3。流程為蒐集三個 seed 的測試配對與訓練 rollout（CPU），再訓練（建議用 GPU，最多 1000 epoch，patience 20）並評分。模型檔 `out/dynamics.pt` 不入版控。重新訓練不保證逐位元相同，所以 `data/inputs/` 是某一次訓練的結果。

## 重現表格

```bash
export PYTHONPATH=.:tasks/T4.1.2.1-action-conditioned-detection
uv run python -m analyze.detect
uv run python -m analyze.prediction
uv run python -m analyze.windows
```

三個指令都支援 `--check`，逐 byte 核對既有輸出。AUROC 的區間為以 episode 重抽 300 次（種子固定）。

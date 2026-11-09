# T4.1.2.1：動作條件動力學模型的偏移偵測

結果見 [REPORT.md](REPORT.md)。任務說明見 [TASK.md](TASK.md)。以 baseline 轉移訓練 MLP 預測 $\Delta s=s_{t+1}-s_t$，用預測誤差 $e=\Delta s-f_\theta(s,a)$ 偵測動力學偏移，測試資料與 T4.1.2 完全相同（同一 $(s,a)$ 在 baseline 與八個偏移環境各走一步）。

| 路徑 | 內容 |
|---|---|
| `zero_shot_rl/latent/dynamics_model.py` | $f_\theta$、按 episode 切 validation、early stopping、$\Sigma_0$；命令列訓練 |
| `zero_shot_rl/latent/dynamics_score.py` | 對 `paired_step` 的輸出算 $\lVert e\rVert_2$ 與 $D_{dyn}$，堆疊三個 seed 的檔案 |
| `tests/test_dynamics_model.py` | split、Mahalanobis、訓練後模型對偏移資料的分數、存取的元件測試 |
| `run_dynamics.sh` | 完整流程（見下） |
| `analyze/detect.py` | 由 `data/inputs/dynamics_scored.npz` 產生 `data/detect.csv` |
| `tests/test_detect.py` | AUROC 與 Youden 閾值的測試 |
| `data/inputs/dynamics_scored.npz` | 每個設定的 $\lVert e\rVert_2$、$D_{dyn}$、$s'$ 差異、各維度 MSE，以及 validation 的 baseline 分數 |

## 資料切分

- 訓練池：T06 的三個 FB checkpoint 各自在 baseline 跑的 episode 100 到 119，共 60 個 episode（59,940 筆）。只用這些互動轉移，不用 ExORL 原始資料。
- 訓練 47,952 筆（48 個 episode），validation 11,988 筆（12 個 episode，每個 seed 4 個，亂數種子 0）。Validation 的 episode 編號存在模型檔的 `meta["val_episodes"]`。
- 測試：三個 seed 各自的 episode 0 到 19（共 60 個 episode，59,940 對），與 T4.1.2 的配對資料相同。
- 閾值取 validation 中 baseline 分數的第 95 與 99 百分位數，不使用任何偏移資料。

## 執行

```bash
CKPT_DIR=<含 zsrl-32.pt zsrl-41.pt zsrl-42.pt 的目錄> bash tasks/T4.1.2.1-action-conditioned-detection/run_dynamics.sh
cp out/dynamics_scored.npz tasks/T4.1.2.1-action-conditioned-detection/data/inputs/
```

蒐集資料用 CPU，約 10 分鐘。訓練在 CPU 上約 15 分鐘（GPU 較快）。模型檔 `out/dynamics.pt` 不入版控。重新訓練不保證逐位元相同，所以 `data/inputs/` 是某一次訓練的結果。

## 重現表格

```bash
PYTHONPATH=.:tasks/T4.1.2.1-action-conditioned-detection uv run python -m analyze.detect
```

加 `--check` 逐 byte 核對既有輸出。AUROC 的區間為以 episode 重抽 300 次（種子固定）。

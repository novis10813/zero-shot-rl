# T4.1.2.2：CVAE 轉移模型 $P(s_{t+1}\mid s_t,a_t)$

依 Prashant et al. 2025（arXiv 2503.05238）實作條件 VAE ensemble，資料、切分、標準化、測試配對與 T4.1.2.1 完全相同，所以分數可直接與 T4.1.2.1 比較。結果表格見 `data/`，尚未寫 REPORT.md。

| 路徑 | 內容 |
|---|---|
| `zero_shot_rl/latent/cvae_model.py` | CVAE（encoder $q(z\mid s',s,a)$、decoder $p(s'\mid z,s,a)$）、N 個成員的 ensemble、ELBO 訓練與 early stopping、ICP 閾值 `icp_threshold` 與判定 `is_ood`；命令列訓練 |
| `zero_shot_rl/latent/cvae_score.py` | 對 `paired_step` 的輸出算每個成員的重建 MSE $r_i$、成員平均 `mean_*`、成員最小值 `min_*` |
| `tests/test_cvae_model.py` | 偏移誤差較大、成員不同、兩種目標、ICP 規則、評分、存取的元件測試 |
| `run_cvae.sh` | 完整流程（見下） |
| `analyze/pack.py` | 把 `run_cvae.sh` 的輸出壓成 `data/inputs/`（每個設定 float32，只留成員的平均與最小值）與訓練診斷 `fit_info.json` |
| `analyze/detect.py` | AUROC 與信賴區間、ICP 偵測率、訓練診斷、與 T4.1.2.1 並列 → `data/detect.csv`、`icp.csv`、`fit.csv`、`compare.csv` |
| `tests/test_detect.py` | ICP 與訓練診斷彙整的測試 |
| `data/inputs/cvae_<run>.npz` | 每個設定（`beta<值>`、`learn_sigma`）的 $r_i$ 平均與最小值、各維度 MSE、baseline validation 誤差（校準集） |

## 與論文的差異

- 條件為 $(s,a)$，論文只用 $s_1$（其策略是確定性的）。
- 重建 $s'$，與論文相同（`--target delta` 改為重建標準化後的 $\Delta s$，供消融）。重建殘差 $e_i=s'-\hat s'_i$，分數為 $r_i=\operatorname{mean}(e_i^2)$（論文的重建 MSE）。
- 重建用 encoder 的平均 $\mu$ 當 $z$，分數為確定性。
- Calibration set 是 T4.1.2.1 的 12 個 baseline validation episode（11,988 筆）乘 N 個成員，閾值為其由小到大第 $\lfloor\delta MN\rfloor$ 個；沒有成員的誤差低於閾值才判為 OOD，即 $\min_i r_i\ge$ 閾值。
- 論文未給網路大小、$z$ 維度與 $\beta$，預設為 512×3、8、1，皆可用參數改。

## 執行

```bash
CKPT_DIR=<含 zsrl-32.pt zsrl-41.pt zsrl-42.pt 的目錄> DATASET=<walker/rnd/dataset.npz> \
  bash tasks/T4.1.2.2-conditioned-vae/run_cvae.sh
```

可用環境變數 `MEMBERS`（預設 5）、`LATENT`、`LEARN_SIGMA=1`（多訓練一組可學習 decoder 標準差、$\beta=1$ 的模型，`--learn_sigma`）、`BETAS`（預設 `"0.1 1 4"`，每個值訓練一組 ensemble）、`HIDDEN`、`LAYERS`、`TARGET`。若 `out/` 已有 T4.1.2.1 的 `paired_raw_s*.npz` 與 `train_rollouts_s*.npz`，會直接沿用。訓練最多 1000 epoch、patience 20，每個成員各自 early stopping，建議用 GPU。每個 $\beta$ 輸出 `out/cvae_beta<β>.pt`（不入版控）、`.log` 與 `out/cvae_scored_beta<β>.npz`。模型 meta 的 `fit_info` 記錄每個成員的最佳 epoch、validation 重建與 KL，以及每個 $z$ 維度的 KL。

本任務的 `data/inputs/` 來自兩次執行：`BETAS="0.1 1 4"` 與 `BETAS="0 0.0001 0.001 0.01 0.03" LEARN_SIGMA=1`。

## 重現表格

```bash
export PYTHONPATH=.:tasks/T4.1.2.2-conditioned-vae
uv run python -m analyze.pack <第一次執行的輸出目錄> <第二次執行的輸出目錄>   # 由 run_cvae.sh 的 out/ 產生 data/inputs/
uv run python -m analyze.detect
```

`analyze.detect` 支援 `--check`，逐 byte 核對既有輸出，耗時約 15 分鐘（信賴區間重抽 300 次，種子固定）。`compare.csv` 讀 T4.1.2.1 的 `data/detect_*.csv`。重新訓練不保證逐位元相同，所以 `data/inputs/` 是某一次訓練的結果。

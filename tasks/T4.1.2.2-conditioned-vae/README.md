# T4.1.2.2：CVAE 轉移模型 $P(s_{t+1}\mid s_t,a_t)$

依 Prashant et al. 2025（arXiv 2503.05238）實作條件 VAE ensemble，資料、切分、標準化、測試配對與 T4.1.2.1 完全相同，所以分數可直接與 T4.1.2.1 比較。結果尚未產生。

| 路徑 | 內容 |
|---|---|
| `zero_shot_rl/latent/cvae_model.py` | CVAE（encoder $q(z\mid s',s,a)$、decoder $p(s'\mid z,s,a)$）、N 個成員的 ensemble、ELBO 訓練與 early stopping、ICP 閾值 `icp_threshold` 與判定 `is_ood`；命令列訓練 |
| `zero_shot_rl/latent/cvae_score.py` | 對 `paired_step` 的輸出算每個成員的重建 MSE $r_i$、成員平均 `mean_*`、成員最小值 `min_*` |
| `tests/test_cvae_model.py` | 偏移誤差較大、成員不同、兩種目標、ICP 規則、評分、存取的元件測試 |
| `run_cvae.sh` | 完整流程（見下） |

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

可用環境變數 `MEMBERS`（預設 5）、`LATENT`、`BETAS`（預設 `"0.1 1 4"`，每個值訓練一組 ensemble）、`HIDDEN`、`LAYERS`、`TARGET`。若 `out/` 已有 T4.1.2.1 的 `paired_raw_s*.npz` 與 `train_rollouts_s*.npz`，會直接沿用。訓練最多 1000 epoch、patience 20，每個成員各自 early stopping，建議用 GPU。每個 $\beta$ 輸出 `out/cvae_beta<β>.pt`（不入版控）、`.log` 與 `out/cvae_scored_beta<β>.npz`。模型 meta 的 `fit_info` 記錄每個成員的最佳 epoch、validation 重建與 KL，以及每個 $z$ 維度的 KL。

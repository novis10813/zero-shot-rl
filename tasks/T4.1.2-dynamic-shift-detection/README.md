# T4.1.2：動力學偏移偵測的配對單步測試與重現

結果見 [REPORT.md](REPORT.md)。本任務檢驗 FB Bellman 殘差的 Mahalanobis 距離 $D$ 能否偵測動力學偏移：在 baseline 環境收集 $(s,a)$，把同一個物理狀態與動作放進偏移環境走一步，比較 $s'$ 的殘差。

| 路徑 | 內容 |
|---|---|
| `zero_shot_rl/latent/shift_detector.py` | 殘差 $e$、$\mu_0$ 的 MLP、$\Sigma_0$、$D$；命令列擬合 detector（`--dataset_path` 用 ExORL 資料，`--rollouts` 用 `paired_step` 的 baseline 軌跡） |
| `zero_shot_rl/eval/paired_step.py` | baseline rollout，並把每步的物理狀態與動作重播到各 yaml 的動力學，輸出 `next_<yaml>` |
| `zero_shot_rl/eval/paired_score.py` | 以 detector 對每個 `next_*` 算 $D$、$s'$ 差異，以及殘差 $r=e-\mu_0$ 的平均與共變異數 |
| `tests/test_shift_detector.py` | Mahalanobis、共變異數、存取的元件測試 |
| `configs/` | 調高 friction 與 actuator gear 的四個 yaml；調低的四個用 `tasks/T3.2-ood-robustness/configs/` |
| `run_paired.sh` | 一個訓練 seed 的完整流程（見下） |
| `analyze/paired.py`、`analyze/moments.py` | 由 `data/inputs/` 產生 `data/paired.csv`、`data/moments.csv`、`data/moments_summary.csv` |
| `data/inputs/scored_<detector>_s<seed>.npz` | 每個 detector（`dataset`、`rollout`）與訓練 seed 的 $D$、$s'$ 差異、殘差平均與共變異數 |

## 執行

`run_paired.sh` 對一個訓練 seed 依序：(1) 在 baseline 跑 episode 0 到 19 並重播到 baseline 與八個偏移設定；(2) 跑 episode 100 到 119 的 baseline 軌跡；(3) 擬合兩個 detector；(4) 各自評分。checkpoint 為 T06 的三個 FB（`tasks/T06-latent-switching/configs/sources.json`），latent 為其 CQL sweep 的 `z_r`。

```bash
SEED=0 CKPT_DIR=<含 zsrl-32.pt zsrl-41.pt zsrl-42.pt 的目錄> \
  DATASET=<walker/rnd/dataset.npz> bash tasks/T4.1.2-dynamic-shift-detection/run_paired.sh
```

輸出在 `out/scored_dataset_s<seed>.npz`、`out/scored_rollout_s<seed>.npz`，複製到 `data/inputs/`。擬合需要 GPU。

- 重播只適用於靜態偏移：`set_physics` 會 `reset()`，所以有 `events` 的設定不支援。
- baseline 的重播與原 rollout 逐位元相同（`analyze/paired.py` 會檢查 `D_baseline` 等於 `D_rollout`）。
- 資料版 detector 的擬合不能逐位元重現（GPU 非決定性），所以 `data/inputs/` 的數字是某一次擬合的結果。

## 重現表格

```bash
PYTHONPATH=.:tasks/T4.1.2-dynamic-shift-detection uv run python -m analyze.paired
PYTHONPATH=.:tasks/T4.1.2-dynamic-shift-detection uv run python -m analyze.moments
```

加 `--check` 逐 byte 核對既有輸出。REPORT 的表格是這兩個 csv 對 3 個 seed 的平均（`moments_summary.csv` 另有最小與最大值）。

# T6-anfis

以可自由組合的 PyTorch ANFIS（`zero_shot_rl/latent/anfis.py`）取代原始 reward 權重，透過 episodic REINFORCE（`zero_shot_rl/finetune/anfis_z.py`）更新固定 FB 的 task latent $z$。結果見 [REPORT.md](REPORT.md)。

| 路徑 | 內容 |
|---|---|
| `configs/anfis_z.yaml` | 可自由組合 ANFIS 元件（features、membership、t_norm、consequent、output_activation）與 REINFORCE 設定 |
| `run_anfis_z.sh` | 單一 FB checkpoint 四任務訓練與 Table-6 評估啟動腳本 |
| `data/inputs/` | `table6.csv`、`baseline_table6.csv`、`eval_episodes.csv`、`train.csv`、`reward_grid.npz` |
| `data/scores.csv` | 原始 $z_r$ 與 ANFIS-z 分數對照表 |
| `data/reward_probes.csv` | 8 個典型物理姿態的原始 $r(s)$ 與 ANFIS $\tilde{r}_\theta(s)$ 數值對照 |
| `figures/reward_slices.png` | 1D 物理特徵切面比較圖 |
| `figures/reward_heatmap.png` | 2D 主軸熱圖與差值圖 |
| `analyze/compare.py` | 從 `data/inputs/` 重現表格與圖表 |

## 執行實驗

```bash
CHECKPOINT=<path/to/660000.pt> DATASET=<path/to/walker/rnd/dataset.npz> OUT_DIR=<out/anfis_z> \
  bash tasks/T6-anfis/run_anfis_z.sh
```

彙整多 seed Table 6 分數：

```bash
PYTHONPATH=. uv run python -m analysis.table6_aggregate <seed0/table6.csv> <seed1/table6.csv> ...
```

## 重現表格與圖表

從已儲存的 ANFIS checkpoint 萃取 `reward_grid.npz`（僅重訓時需要）：

```bash
PYTHONPATH=.:tasks/T6-anfis uv run python -m analyze.compare --extract <out_dir>/models
```

由 `data/inputs/` 重新產生 CSV 與圖表（或加上 `--check` 逐 byte 核對）：

```bash
PYTHONPATH=.:tasks/T6-anfis uv run python -m analyze.compare
PYTHONPATH=.:tasks/T6-anfis uv run python -m analyze.compare --check
```

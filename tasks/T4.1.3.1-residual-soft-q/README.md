# T4.1.3.1：residual soft Q-learning 的 ω 掃描

結果在 `REPORT.md`。

| 路徑 | 內容 |
|---|---|
| `configs/train_fb.yaml` | vanilla FB，Walker RND 100k，seed 0 到 4，每次評估都存 checkpoint |
| `configs/frozen_fb.yaml` | frozen FB，測試 seed 5 到 24 |
| `configs/omega_{clean,mild,strong}.yaml` | ω 掃描，每個環境 12 個 job（3 個 ω × 4 個任務） |
| `run_omega_dicos.sh` | 同一個 ω 掃描的 Slurm array 版本（DiCOS A100）；報告的結果沒有用它 |
| `run.sh` | 以選出的 FB checkpoint 執行 `zero_shot_rl.eval.value_correction` |
| `analyze/select_seed.py` | 依 `analysis.table6_aggregate` 選步數，再選 4 任務 IQM 最高的 seed |
| `analyze/tables.py` | 從 `data/inputs/` 重建 `data/tables.md`，是 `REPORT.md` 每個 episode 表格的來源 |
| `data/fb_train/eval_fb_s<seed>.csv` | 5 個 FB 訓練 log 的評估分數 |
| `data/inputs/omega_<環境>_<任務>_w<ω>.csv` | 36 個 ω 掃描 job 的 `eval.csv` |

## 1. 訓練 FB 並選 checkpoint

```bash
lab sweep tasks/T4.1.3.1-residual-soft-q/configs/train_fb.yaml
python analysis/extract_eval.py <job output.log> \
  tasks/T4.1.3.1-residual-soft-q/data/fb_train/eval_fb_s<seed>.csv
PYTHONPATH=. uv run python -m analysis.table6_aggregate \
  tasks/T4.1.3.1-residual-soft-q/data/fb_train/eval_fb_s{0,1,2,3,4}.csv
PYTHONPATH=.:tasks/T4.1.3.1-residual-soft-q uv run python -m analyze.select_seed
```

選出的是 seed 1 的 step 660000。`run.sh` 在 `$ARTIFACTS/T4.1.3.1-residual-soft-q/zero-shot-rl-297/660000.pt`
找它（預設 `ARTIFACTS=$HOME/zsrl-artifacts`），資料集預設在 `$HOME/zsrl-datasets/walker/rnd/dataset.npz`。

## 2. ω 掃描

```bash
lab sweep tasks/T4.1.3.1-residual-soft-q/configs/omega_clean.yaml
lab sweep tasks/T4.1.3.1-residual-soft-q/configs/omega_mild.yaml
lab sweep tasks/T4.1.3.1-residual-soft-q/configs/omega_strong.yaml
```

在 idlab 上執行，一次一個檔，每個 job 1 個 CPU 執行緒。每個 job 寫出
`out/omega.<環境>_<任務>_w<ω>/eval.csv`，複製為 `data/inputs/omega_<環境>_<任務>_w<ω>.csv`。

## 3. 重建 REPORT.md 的表格

```bash
PYTHONPATH=.:tasks/T4.1.3.1-residual-soft-q uv run python -m analyze.tables
PYTHONPATH=.:tasks/T4.1.3.1-residual-soft-q uv run python -m analyze.tables --check
```

## 4. 10 個 finetune seed

```bash
lab sweep tasks/T4.1.3.1-residual-soft-q/configs/finetune_seeds_{clean,mild,strong}.yaml   # 各 40 個 job
PYTHONPATH=.:tasks/T4.1.3.1-residual-soft-q uv run python -m analyze.finetune_seeds
PYTHONPATH=.:tasks/T4.1.3.1-residual-soft-q uv run python -m analyze.finetune_seeds --check
```

`out/ft.<環境>_<任務>_s<seed>/eval.csv` 複製為 `data/inputs/finetune/ft.<環境>_<任務>_s<seed>.csv`，輸出是 `data/finetune_tables.md`。

## 5. 10 個 finetune seed 的跨環境遷移

```bash
lab sweep tasks/T4.1.3.1-residual-soft-q/configs/cross_eval_seeds_w0.1.yaml   # 6 個 job，每個跑 10 個 seed
PYTHONPATH=.:tasks/T4.1.3.1-residual-soft-q uv run python -m analyze.cross_seeds
PYTHONPATH=.:tasks/T4.1.3.1-residual-soft-q uv run python -m analyze.cross_seeds --check
```

`out/crossseeds.<訓練環境>_to_<測試環境>/s<seed>/eval.csv` 複製為 `data/inputs/crossseeds/<訓練環境>_to_<測試環境>_s<seed>.csv`，輸出是 `data/cross_seeds_tables.md`。checkpoint 來自第 4 節的 `ft.<環境>_<任務>_s<seed>`。

## 程式變更

- `agents/workspaces.py`：`OfflineRLWorkspace.train` 每次評估都存 `<step>.pt`，不再只留最佳與最後的 checkpoint。
- `zero_shot_rl/eval/value_correction.py`：新增 `--eval_seed_start`（預設 0），`--variants` 可不帶值，只測 frozen FB。
- `tasks/T04-mcfb-pessimism/extract_eval.py` 移到 `analysis/extract_eval.py`。

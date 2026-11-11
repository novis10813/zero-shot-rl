# T4.1.3.0：固定 latent 下對 FB 做線上價值修正

結果在 `REPORT.md`。本目錄只放設定、啟動腳本、分析與資料：

| 路徑 | 內容 |
|---|---|
| `zero_shot_rl/online/value_correction.py` | `QAdapt`：候選動作、Direct、Residual、`search` 變體與 critic 更新 |
| `zero_shot_rl/eval/value_correction.py` | 訓練與評估的 entry module，寫 `eval.csv`、`train.csv` |
| `zero_shot_rl/eval/value_correction_checkpoints.py` | 讀 `--save_every` 存下的 critic，另外做評估 |
| `run.sh` | 啟動 entry module，帶入 T04 的 FB checkpoint 與 Walker RND 資料集 |
| `configs/pilot_h64.yaml`、`pilot_h64_strong.yaml`、`pilot_h64_clean.yaml` | Residual 訓練曲線：Actuator Mild、Actuator Strong、Clean，各 2 個 job（半徑 0.2 與不限半徑） |
| `configs/search.yaml` | `search` 與 frozen FB，3 個環境 × 2 個半徑，共 6 個 job |
| `analyze/tables.py` | 從 `data/inputs/` 重建 `data/tables.md`，是 `REPORT.md` 表格的來源 |
| `data/inputs/` | 12 個 job 的 `eval.csv`：`curve_<環境>_<半徑>.csv` 與 `search_<環境>_<半徑>.csv` |

## 重建 REPORT.md 的表格

```bash
PYTHONPATH=.:tasks/T4.1.3.0-online-value-correction uv run python -m analyze.tables
PYTHONPATH=.:tasks/T4.1.3.0-online-value-correction uv run python -m analyze.tables --check
```

Clean 的 T04 frozen FB 值讀 `tasks/T04-mcfb-pessimism/data/eval_fb_s0.csv` 的第 800000 步，walk 欄。

## 重新產生 `data/inputs/` 的實驗

用 `lab sweep tasks/T4.1.3.0-online-value-correction/configs/<檔名>.yaml`，在 idlab 上執行，FB seed 0。`run.sh` 預設在 `$HOME/zsrl-artifacts/T04-mcfb-pessimism/` 找 FB checkpoint，在 `$HOME/zsrl-datasets/walker/rnd/dataset.npz` 找資料集。

**注意：** 後來 `main` 上的 `residual` 變體改成 residual soft Q-learning（Li et al. 2023，新參數 `--omega`、`--alpha`），不再是 `REPORT.md` 的 hard max 版本。`pilot_h64*.yaml` 的 6 個曲線 job 要從它們的 `exp/T4.1.3.0-zero-shot-rl-271` 到 `276` tag 的 commit 執行才會重現舊結果。`search.yaml` 不受影響，在 `main` 上就可重現。

每個 job 寫出 `out/<名稱>_s0/eval.csv`，複製到 `data/inputs/` 時：Mild 與 Strong 的曲線 job 是 `pilot_h64.r0.2`、`pilot_h64.unbounded`（Mild）與 `pilot_h64_strong.*`，Clean 是 `pilot_h64_clean.*`，Search 是 `search.<環境>_<半徑>`。

## 測試

```bash
PYTHONPATH=.:tasks/T4.1.3.0-online-value-correction MUJOCO_GL=egl uv run python -m unittest discover \
  -s tasks/T4.1.3.0-online-value-correction/tests -t tasks/T4.1.3.0-online-value-correction
PYTHONPATH=. uv run python -m unittest tests.test_value_correction
```

## 其他程式變更

- `value_correction` 的 `eval.csv` 與 `train.csv` 新增 `a0_frac`（選到 $a_0$ 的步數比例）。
- 新增 `--save_every`、`--train_only`（訓練與評估分開）、`--omega`、`--alpha`、變體 `search`（用 `--episodes 0`）。預設的 `--variants` 仍是 `direct residual`。
- `QAdapt` 的預設 `hidden` 為 64，entry module 的 `--hidden` 預設仍是 1024。報告的 job 都明確傳入 `--hidden 64`。
- 沒有收入 `REPORT.md` 的 residual soft Q-learning 試跑（`--omega`、`--alpha`）與 hidden 256 的試跑，留在 `task/T4.1.3.0-online-value-correction` 的 commit 歷史中。

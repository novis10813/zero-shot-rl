# T6.1-ood-fuzzy

在 Walker OOD 動力學偏移環境中比較四種取得 FB task latent $z$ 的方式，並驗證依動力學偏移類型客製化 ANFIS sensors 是否能進一步提升適應表現。

## 實驗設計

- **凍結 FB 模型**：Walker RND `660000.pt`（`/home/sam/zsrl-artifacts/T4.1.3.1-residual-soft-q/zero-shot-rl-297/660000.pt`）
- **評估環境**：Walker 四任務（`stand`, `walk`, `run`, `flip`），共用 20 個評估環境種子 `5..24`
- **Finetune 種子**：10 個獨立 finetune seeds（`1000, 1010, ..., 1090`），每個 seed 使用互不重疊的訓練環境種子（`--train_seed_offset {ft_seed}00`）
- **四個簡化 OOD 動力學對照（調低 / 調高）**：
  - `friction_mild`（`geom_friction: 0.1`，原值 `0.7`）
  - `friction_high_mild`（`geom_friction: 1.5`，原值 `0.7`）
  - `actuator_mild`（`actuator_gear: 76%`）
  - `actuator_high_mild`（`actuator_gear: 130%`）

### 四個比較條件

1. **Clean FB (`z_r`)**：凍結 FB 以原始 reward 推論的 $z_r$ 直接於 OOD 環境評估（記錄於 `baseline_table6.csv`）。
2. **Clean ANFIS-z 直接遷移**：於 clean Walker 訓練好的 4-sensor ANFIS-z，透過 `--eval_models_dir` zero-shot 評估於各 OOD 環境。
3. **4-sensor ANFIS-z OOD Finetune**：使用 `configs/anfis/walker.yaml`（4 個基礎 CoM / reward sensors，36 rules）在 OOD 環境做 REINFORCE 微調。
4. **客製化擴增 sensor ANFIS-z OOD Finetune**：
   - Friction（`configs/walker_friction.yaml`，7 sensors / 128 rules）：4 基礎 + `com_vz`, `qvel_right_ankle`, `qvel_left_ankle`（捕捉腳底打滑、黏滯與垂直彈跳）。
   - Actuator（`configs/walker_actuator.yaml`，8 sensors / 256 rules）：4 基礎 + `qvel_right_hip`, `qvel_left_hip`, `qvel_right_knee`, `qvel_left_knee`（捕捉力矩縮放後的關節角速度變化）。

## 超參數

詳見 `configs/reinforce.yaml`：
- `reference`: `dataset_transitions=100000`, `z_inference_steps=10000`, `discount=0.98`, `seed=42`
- `warmstart`: `enabled=true`, `steps=500`, `lr=0.01`
- `reinforce`: `iters=50`, `pop_size=16`, `antithetic=true`, `sigma=0.5`, `lr=0.001`, `grad_clip=1.0`, `train_episodes_per_z=1`
- `eval`: `eval_every=10`, `eval_episodes=20`, `eval_seed_start=5`

## 執行指令（`idlab`）

```bash
# 1. 訓練 clean ANFIS-z（10 finetune seeds）
lab sweep tasks/T6.1-ood-fuzzy/configs/sweep_clean.yaml

# 2. Clean ANFIS-z 直接遷移到 4 個 OOD 環境（以單一或多個 clean models/ 目錄執行）
CLEAN_MODELS_DIR=<out/clean_ft1000/models> OUT_ROOT=out/transfer_ft1000 \
  bash tasks/T6.1-ood-fuzzy/run_transfer_eval.sh

# 3. 4-sensor ANFIS-z 在 OOD 環境重新 finetune（10 ft_seeds × 4 dynamics = 40 jobs）
lab sweep tasks/T6.1-ood-fuzzy/configs/sweep_ood_4sensor.yaml

# 4. 客製化 sensor ANFIS-z 在 OOD 環境重新 finetune（各 10 ft_seeds × 2 dynamics = 20 jobs）
lab sweep tasks/T6.1-ood-fuzzy/configs/sweep_ood_friction.yaml
lab sweep tasks/T6.1-ood-fuzzy/configs/sweep_ood_actuator.yaml
```

實際執行時依序跑完每個環境的 10 個 `ft_seed`（每個 job `--vram 16G`，單 job 峰值約 8.4 GiB），參數與上列 sweep 相同，輸出的 `table6.csv`、`baseline_table6.csv`、`eval_episodes.csv`、`train.csv`、`config_resolved.yaml` 存入 `data/inputs/<condition>/<dynamics>/ft<seed>/`（`clean/baseline/ft<seed>/` 為原環境訓練，`frozen_fb/clean/` 為原環境的 Frozen FB $z_r$ 評估，由 `run_frozen_fb_ood.sh` 的 `--eval_zr_only` 產生）。

## 接續訓練到 200 步

第一段 finetune（step 1 到 50）完成後，每個 finetune 條件從各自儲存的 `<task>_final.pt` 接續到 step 200，參數與 `reinforce.yaml` 相同，只有 `iters` 改為 200（`configs/reinforce_resume.yaml`）。每個 job 的指令為第一段的指令加上：

```bash
--config tasks/T6.1-ood-fuzzy/configs/reinforce_resume.yaml \
--resume_models_dir <第一段 job 的 out/run/models> --resume_step 50
```

接續時不做 warm-start，Adam 狀態從零開始，並對 step 50 重新評估。結果存入 `data/inputs/<condition>_cont/<dynamics>/ft<seed>/`，`analyze/summarize.py` 與 `analyze/curves.py` 會把它接在第一段之後，並檢查 step 50 的重新評估與第一段一致。

## 彙整結果與重現

```bash
PYTHONPATH=.:tasks/T6.1-ood-fuzzy uv run python -m analyze.summarize
PYTHONPATH=.:tasks/T6.1-ood-fuzzy uv run python -m analyze.curves   # data/curves.csv 與 figures/curves.png
```

產生 `data/summary.csv` 與 `REPORT.md` 的表格（`data/summary.md`），finetune 方法依 Table 6 取各自最佳 step。原環境參考值：

```bash
PYTHONPATH=. uv run python -m analysis.table6_aggregate tasks/T6.1-ood-fuzzy/data/inputs/clean/baseline/ft*/table6.csv
PYTHONPATH=. uv run python -m analysis.table6_aggregate tasks/T6.1-ood-fuzzy/data/inputs/frozen_fb/clean/frozen_fb_clean/table6.csv
```

## 學到的 reward 權重圖（step 50）

以 4 sensors ANFIS、finetune seed 1000 的 step 50 模型（`<task>_final.pt`），沿用 `tasks/T6-anfis/analyze/compare.py` 的繪圖函式，畫出學到的權重 $\tilde{r}_\theta(s)$ 與原始 $r(s)$ 的比較，原環境與 4 個 OOD 環境各一組，圖在 `figures/step50/<環境>/`。客製 sensors 的模型（`figures/step50/<環境>_custom/`）除了 4 個 reward 特徵的切面與熱圖，還多畫每個額外 sensor 的 1D 切面（範圍為參考資料的平均 ± 3 個標準差）。掃 4 個 reward 特徵時，額外 sensors 固定在參考資料的平均值。

```bash
# 從 artifacts 中的模型重新萃取（需要模型）
PYTHONPATH=.:tasks/T6.1-ood-fuzzy uv run python -m analyze.reward_figs --extract clean=<models_dir> friction_mild=<models_dir> ...
PYTHONPATH=.:tasks/T6.1-ood-fuzzy uv run python -m analyze.reward_figs --extract_extended friction_mild_custom=<models_dir> ...
# 從 data/inputs/reward_grid/ 重新畫圖
PYTHONPATH=.:tasks/T6.1-ood-fuzzy uv run python -m analyze.reward_figs
```

接續訓練後的 step 200 模型加上 `--step 200`，資料在 `data/inputs/reward_grid_step200/`，圖在 `figures/step200/`。

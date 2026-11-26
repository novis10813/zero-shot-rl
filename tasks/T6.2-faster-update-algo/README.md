# T6.2-faster-update-algo

在 Walker 四個 OOD 動力學環境（`friction_mild`、`friction_high_mild`、`actuator_mild`、`actuator_high_mild`）中，於統一的 800,000 環境步數預算下比較 REINFORCE、LoLA、ReLA、CEM、CMA-ES 五類潛向量適應演算法，以及自由潛向量 $z$、4-sensor ANFIS、客製 sensor ANFIS 三種搜尋空間。

## 設定檔

位於 `tasks/T6.2-faster-update-algo/configs/`：
- `lola_z.yaml`、`rela_z.yaml`、`cem_z.yaml`、`cmaes_z.yaml`：直接優化自由潛向量 $z$（`param: z`）。
- `lola_anfis.yaml`、`cem_anfis.yaml`、`cmaes_anfis.yaml`：4-sensor ANFIS（`configs/anfis/walker.yaml`）。
- 動力學設定：`friction_mild`、`actuator_mild` 用 `tasks/T3.2-ood-robustness/configs/<env>.yaml`；`friction_high_mild`、`actuator_high_mild` 用 `tasks/T4.1.2-dynamic-shift-detection/configs/<env>.yaml`。
- 客製 sensor ANFIS：執行時透過 `--anfis_config` 覆寫為 `tasks/T6.1-ood-fuzzy/configs/walker_friction.yaml`（Friction 兩環境）或 `tasks/T6.1-ood-fuzzy/configs/walker_actuator.yaml`（Actuator 兩環境）。

## 執行範例

```bash
# GPU（LoLA / ReLA）
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. MUJOCO_GL=egl \
  uv run python -m zero_shot_rl.finetune.anfis_z \
  --config tasks/T6.2-faster-update-algo/configs/lola_z.yaml \
  --checkpoint <fb.pt> --dataset_path <dataset.npz> \
  --dynamics tasks/T3.2-ood-robustness/configs/friction_mild.yaml \
  --seed 1000 --train_seed_offset 100000 --out_dir out/lola_z_friction_mild_ft1000

# 純 CPU（CEM / CMA-ES）
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=. MUJOCO_GL=disabled \
  uv run python -m zero_shot_rl.finetune.anfis_z \
  --anfis_config configs/anfis/walker.yaml \
  --config tasks/T6.2-faster-update-algo/configs/cmaes_anfis.yaml \
  --device cpu --checkpoint <fb.pt> --dataset_path <dataset.npz> \
  --dynamics tasks/T3.2-ood-robustness/configs/friction_mild.yaml \
  --seed 1000 --train_seed_offset 100000 --out_dir out/cmaes_anfis_friction_mild_ft1000
```

## 彙整結果與重現

```bash
PYTHONPATH=.:tasks/T6.2-faster-update-algo uv run python -m analyze.summarize
```

產出 `data/summary.csv` 與 `data/summary.md`（與 `REPORT.md` 完整比較表一致；預設僅納入收齊 4 環境 × 10 seeds 的完整條件）。

## 學到的 reward 權重視覺化（`sep-CMA-ES` 800k，10 seeds 平均 $\pm 1\sigma$）

```bash
# 從 data/inputs/{reward_grid,ref_binned}/*.npz 重新繪製所有圖至 figures/cmaes_800k/
PYTHONPATH=.:tasks/T6.2-faster-update-algo uv run python -m analyze.reward_figs

# 從本地 artifacts 模型自動萃取 8 組 × 10 seeds 的固定姿態切面 grid
PYTHONPATH=.:tasks/T6.2-faster-update-algo uv run python -m analyze.reward_figs \
  --extract_from_artifacts artifacts/T6.2-faster-update-algo

# 從本地 artifacts 模型與離線資料集萃取 8 組 × 10 seeds 的真實資料集分箱平均
MUJOCO_GL=egl PYTHONPATH=.:tasks/T6.2-faster-update-algo uv run python -m analyze.reward_figs \
  --extract_ref_binned artifacts/T6.2-faster-update-algo \
  --dataset_path datasets/walker/rnd/dataset.npz

# 產出純前端互動式 ANFIS 檢視器（單檔 HTML，含 1D 切面、2D 熱圖與全感測器同步）
MUJOCO_GL=egl PYTHONPATH=.:tasks/T6.2-faster-update-algo uv run python -m analyze.interactive_explorer \
  --artifacts_dir artifacts/T6.2-faster-update-algo \
  --dataset_path datasets/walker/rnd/dataset.npz \
  --out tasks/T6.2-faster-update-algo/figures/cmaes_800k/anfis_explorer.html
```

## 測試

```bash
PYTHONPATH=.:tasks/T6.2-faster-update-algo MUJOCO_GL=egl \
  uv run python -m unittest discover -s tasks/T6.2-faster-update-algo/tests -t tasks/T6.2-faster-update-algo
```

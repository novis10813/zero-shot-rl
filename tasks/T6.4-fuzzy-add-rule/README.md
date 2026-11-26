# T6.4-fuzzy-add-rule

驗證加法規則 ANFIS（`rule_mode: additive`、`normalize_firing: false`）在 4、8、37 個 sensors 與 `zero_order` / `first_order` consequent 下的規則字典 $\mathbf{Z}_{\text{dict}}$ 譜性質，以及在 Walker 四個 OOD 動力學環境（`sep-CMA-ES`，10 seeds）上的實際適應表現。結果見 `REPORT.md`。

## 字典秩分析

```bash
CHECKPOINT=<FB 660000.pt> DATASET=datasets/walker/rnd/dataset.npz OUT_DIR=tasks/T6.4-fuzzy-add-rule/data \
  bash tasks/T6.4-fuzzy-add-rule/run_zdict_rank.sh --device cpu
```

產出 `data/zdict_rank.csv` 與 `data/zdict_spectra.npz`。`REPORT.md` 第 2 節探索變異數分布的數值可由以下指令重現：

```bash
uv run python -c "
import numpy as np
sp = np.load('tasks/T6.4-fuzzy-add-rule/data/zdict_spectra.npz')
for k in sp.files:
    s = sp[k]
    p = (s ** 2) / (s ** 2).sum()
    p = p[p > 0]
    print(k, round(float(np.exp(-(p * np.log(p)).sum())), 2), round(float(p[:10].sum()), 4), round(float(p[25:].sum()), 4))
"
```

## OOD Walker `sep-CMA-ES` 實驗（3 條件 × 4 環境 × 10 seeds）

基礎設定檔為 `configs/sepcmaes_anfis.yaml`，搭配三個 ANFIS 設定：
- `configs/walker_all37_additive_first.yaml`（`all37_first`）
- `configs/walker8_additive_first.yaml`（`8_first`）
- `configs/walker_all37_additive_zero.yaml`（`all37_zero`）

12 組遠端 sweep 設定位於 `configs/sweep_<cond>_<env>.yaml`，各包含 10 個 finetune seeds（`1000..1090`）：

```bash
lab sweep tasks/T6.4-fuzzy-add-rule/configs/sweep_all37_first_friction_mild.yaml
lab sweep tasks/T6.4-fuzzy-add-rule/configs/sweep_all37_first_friction_high_mild.yaml
lab sweep tasks/T6.4-fuzzy-add-rule/configs/sweep_all37_first_actuator_mild.yaml
lab sweep tasks/T6.4-fuzzy-add-rule/configs/sweep_all37_first_actuator_high_mild.yaml
lab sweep tasks/T6.4-fuzzy-add-rule/configs/sweep_8_first_friction_mild.yaml
lab sweep tasks/T6.4-fuzzy-add-rule/configs/sweep_8_first_friction_high_mild.yaml
lab sweep tasks/T6.4-fuzzy-add-rule/configs/sweep_8_first_actuator_mild.yaml
lab sweep tasks/T6.4-fuzzy-add-rule/configs/sweep_8_first_actuator_high_mild.yaml
lab sweep tasks/T6.4-fuzzy-add-rule/configs/sweep_all37_zero_friction_mild.yaml
lab sweep tasks/T6.4-fuzzy-add-rule/configs/sweep_all37_zero_friction_high_mild.yaml
lab sweep tasks/T6.4-fuzzy-add-rule/configs/sweep_all37_zero_actuator_mild.yaml
lab sweep tasks/T6.4-fuzzy-add-rule/configs/sweep_all37_zero_actuator_high_mild.yaml
```

每個 job 的 `table6.csv`、`baseline_table6.csv`、`train.csv`、`eval_episodes.csv`、`config_resolved.yaml` 存於 `data/inputs/<cond>/<env>/ft<seed>/`。

## 彙整與驗證

```bash
PYTHONPATH=.:tasks/T6.4-fuzzy-add-rule uv run python -m analyze.summarize
PYTHONPATH=.:tasks/T6.4-fuzzy-add-rule uv run python -m analyze.summarize --check
```

## 測試

```bash
PYTHONPATH=.:tasks/T6.4-fuzzy-add-rule MUJOCO_GL=egl \
  uv run python -m unittest discover -s tasks/T6.4-fuzzy-add-rule/tests -t tasks/T6.4-fuzzy-add-rule
```

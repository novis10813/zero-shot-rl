# T6.3-sensors

比較不同 sensor 組合的 ANFIS 搜尋空間。第一步分析 ANFIS 規則字典
$\mathbf{Z}_{\text{dict}} = B^\top \Psi / N$ 的奇異值，判斷增加 sensors 是否擴大可到達的潛向量集合。

## Sensor 組合

| 名稱 | 設定檔 | Sensors | 規則數 | 字典欄數 |
|---|---|---|---|---|
| 4 sensors | `configs/anfis/walker.yaml` | 4 | 36 | 180 |
| 8 sensors | `tasks/T6.1-ood-fuzzy/configs/walker_actuator.yaml` | 8 | 256 | 2304 |
| 全部 37 sensors | `tasks/T6.3-sensors/configs/walker_all37_linear.yaml` | 37 | 1 | 38 |

全部 37 sensors 每個只用一個 term，等於 sensors 的線性 reward，因為 Cartesian grid 放不下 $2^{37}$ 條規則。

## 字典秩分析

```bash
CHECKPOINT=<FB 660000.pt> DATASET=datasets/walker/rnd/dataset.npz OUT_DIR=tasks/T6.3-sensors/data \
  bash tasks/T6.3-sensors/run_zdict_rank.sh
```

FB checkpoint 與 T6.1、T6.2 相同（T4.1.3.1 的 660000 步 checkpoint）。reference set 參數與
`tasks/T6.2-faster-update-algo/configs/cmaes_anfis.yaml` 相同：100,000 筆 transitions、10,000 個狀態、seed 42。
membership 參數取 `init_from_data` 的初始化。

輸出 `data/zdict_rank.csv`（每組的數值秩、有效秩、條件數、四個任務 $z_r$ 的投影殘差）與
`data/zdict_spectra.npz`（完整奇異值）。

`REPORT.md` 的表格取自 `data/zdict_rank.csv`。文中提到的個別奇異值比例 $\sigma_i/\sigma_1$ 由下列指令列出：

```bash
uv run python -c "import numpy as np; d = np.load('tasks/T6.3-sensors/data/zdict_spectra.npz'); [print(k, d[k] / d[k][0]) for k in d.files]"
```

`REPORT.md` 第 4 節的探索變異數分布（以 $\sigma_i^2$ 計算的有效秩、前 10 個與最弱 25 個方向的佔比）：

```bash
uv run python -c "import numpy as np; d = np.load('tasks/T6.3-sensors/data/zdict_spectra.npz'); [print(k, round(float(np.exp(-(p * np.log(p)).sum())), 1), round(float(p[:10].sum()), 4), round(float(p[25:].sum()), 4)) for k in ('walker', 'walker_actuator') for p in [d[k] ** 2 / (d[k] ** 2).sum()]]"
```

## sep-CMA-ES 直接搜 $z$

`configs/sepcmaes_z.yaml` 與 T6.2 的 `cmaes_z.yaml` 只差 `diagonal: true`。4 個 OOD 環境 × 3 個 finetune seeds（1000、1010、1020），只用 CPU、每個 job 一個 thread：

```bash
lab sweep tasks/T6.3-sensors/configs/sweep_sepcmaes_z_mild.yaml
lab sweep tasks/T6.3-sensors/configs/sweep_sepcmaes_z_high_mild.yaml
```

每個 job 的 `out/sepcmaes_z_<env>_ft<seed>/` 中的 `table6.csv`、`baseline_table6.csv`、`train.csv`、`eval_episodes.csv`、`config_resolved.yaml` 存於 `data/inputs/sepcmaes_z/<env>/ft<seed>/`。

與 T6.2 同 seeds 的比較表（`data/sep_vs_full.csv`、`data/sep_vs_full.md`）：

```bash
PYTHONPATH=.:tasks/T6.3-sensors uv run python -m analyze.sep_vs_full
```

## 測試

```bash
PYTHONPATH=.:tasks/T6.3-sensors MUJOCO_GL=egl \
  uv run python -m unittest discover -s tasks/T6.3-sensors/tests -t tasks/T6.3-sensors
```

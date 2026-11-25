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

## 測試

```bash
PYTHONPATH=.:tasks/T6.3-sensors MUJOCO_GL=egl \
  uv run python -m unittest discover -s tasks/T6.3-sensors/tests -t tasks/T6.3-sensors
```

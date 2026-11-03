# T3.2：Walker 動力學偏移的設定與重現

本任務檢驗 Q-guided FB policy 選擇在動力學偏移下是否比在原始環境有更大的優勢。目前完成的部分是動力學偏移的設定（D1）與用標準 FB 校準偏移強度。任務目標見 [TASKS.md](TASKS.md)。

模擬器與評估程式位於 repo 根目錄的 `zero_shot_rl/`，本任務目前只保留設定檔：

| 路徑 | 內容 |
|---|---|
| `zero_shot_rl/env/dynamics.py` | `ShiftedWalkerEnv`：依 yaml 設定模擬器參數；`load_dynamics(path)` 讀取 yaml |
| `zero_shot_rl/env/walker.py` | `make_env(seed, dynamics=None)`；`dynamics` 為 `None` 時與原環境相同 |
| `zero_shot_rl/eval/dynamics_sweep.py` | 以標準 FB 在固定的 $z_r$ 上，於多個動力學設定下 rollout，寫成 CSV |
| `zero_shot_rl/eval/switch.py` | 新增 `--dynamics <yaml>`，在偏移環境中跑 opt、pess、random 與 Q-guided 條件 |
| `tests/test_dynamics.py` | 動力學設定的元件測試 |
| `configs/` | 五個動力學設定 |

## 動力學設定的格式

一份設定有 `base` 與 `events` 兩個鍵：

```yaml
base:                      # 建構時與每次 reset 時載入；沒列出的欄位使用 XML 原值
  geom_friction: {all: 0.7}
  actuator_gear: {right_hip: 100}
events:                    # 選填；在第 step 次 env.step 之前套用，只覆寫列出的欄位
  - step: 400
    set:
      geom_friction: {all: 0.2}
```

- `step` 是 episode 內的 control step，從 reset 之後的第 0 次 `env.step` 起算。一個 episode 為 1000 個 control step，每個 control step 為 0.025 秒。
- 事件依 `step` 排序，後面的事件疊加在前面的事件上。沒有 `events` 就是靜態偏移。
- `reset()` 會把所有欄位還原成 `base`，沒列出的欄位還原成 XML 原值，因此 episode 之間不會互相污染。
- 只接受絕對值，不支援倍率。

可設定的欄位與值的格式：

| 欄位 | 內容 |
|---|---|
| `geom_friction` | 純量只改滑動摩擦；3 個值依序是滑動、扭轉、滾動。mapping 的 key 是 geom 名稱或 `all` |
| `actuator_gear` | 純量、6 個值的列表，或以 actuator 名稱為 key 的 mapping。只改 gear 的第一欄 |
| `dof_damping` | 純量（包含 root 關節）、9 個值的列表，或以 joint 名稱為 key 的 mapping |
| `opt.gravity` | 3 個值的列表 |
| `body_mass` | 純量、列表，或以 body 名稱為 key 的 mapping。轉動慣量按質量比例縮放 |

名稱就是 `custom_dmc_tasks/walker.xml` 中的 `name`。未知的欄位、名稱或長度不符會在建構環境時丟出 `ValueError`，不會等到 episode 中途。

Task reward 由環境外部對 `env.physics` 計算，只讀狀態（torso 高度、速度等），不讀模型參數，所以動力學偏移不會改變 task 目標與 reward 函數。

## 本任務的設定

| 檔案 | 內容 |
|---|---|
| `configs/baseline.yaml` | 原始動力學 |
| `configs/friction_mild.yaml` | 所有 geom 的滑動摩擦由 0.7 降為 0.1 |
| `configs/friction_strong.yaml` | 滑動摩擦降為 0.02 |
| `configs/actuator_mild.yaml` | 六個 actuator 的 gear 為 XML 原值的 76%（hip 76、knee 38、ankle 15.2） |
| `configs/actuator_strong.yaml` | gear 為 XML 原值的 62%（hip 62、knee 31、ankle 12.4） |

摩擦只改滑動欄。此模型沒有設定 `condim`，使用 MuJoCo 預設值 3，扭轉與滾動摩擦在此不啟用。MuJoCo 的接觸摩擦取兩個 geom 的最大值，所以 floor 與身體 geom 要一起改，設定用 `all` 涵蓋。

### 選值方式

數值的選擇標準是讓標準 FB（用 $z_r$ 的 actor）在偏移下保留原本 return 的固定比例，mild 約 0.85，strong 約 0.65：

$$\text{比例} = \frac{\text{偏移下的平均 return}}{\text{baseline 的平均 return}}$$

選值分三輪掃描，每輪縮小範圍，最後的設定以未用於選值的 episode 驗證。Mild 與 strong 設定都落在目標附近，誤差約 ±0.04，差距在抽樣雜訊內。

> 校準的輸出尚未存入 `data/`，所以比例目前不能由本目錄重現。存入後，這裡補上重算的命令。

## 執行

在 repo 根目錄執行。以標準 FB 比較各設定：

```bash
PYTHONPATH=. MUJOCO_GL=egl uv run python -m zero_shot_rl.eval.dynamics_sweep \
  --checkpoint <fb.pt> --zs <zs.npz> \
  --dynamics tasks/T3.2-ood-robustness/configs/*.yaml \
  --episodes 20 --first_episode 0 --out <rollouts.csv>
```

- `--zs` 是 `zero_shot_rl.latent.infer` 的輸出，使用其中 `--task`（預設 `walk`）的 `z_r` 一列。
- Episode `e` 由以 `e` 為 seed 的環境 reset 起始，所有設定共用相同的起始狀態。`--first_episode` 指定第一個 episode 的 seed，用於選值與驗證使用不同的 episode。
- 輸出每個 episode 的 `return`、`speed` 與 `fall_frac`（torso 高度低於 0.6 的步數比例）。

在偏移環境中跑 opt、pess、random 與 Q-guided 條件，用 `zero_shot_rl.eval.switch`，加上 `--dynamics`：

```bash
PYTHONPATH=. MUJOCO_GL=egl uv run python -m zero_shot_rl.eval.switch \
  --checkpoint <fb.pt> --ensemble <ensemble.pt> --zs <zs_sweep.npz> \
  --dynamics tasks/T3.2-ood-robustness/configs/actuator_mild.yaml --out <switch.csv>
```

使用的 checkpoint 與 T06 相同（三個訓練 seed），路徑與雜湊見 `tasks/T06-latent-switching/configs/sources.json`。

## Optimistic 與 pessimistic 的比較

`run_opt_pess.sh` 是 sbatch 腳本，一個 array 任務對應一組（訓練 seed、動力學設定），共 15 個，每個任務跑 `EPISODES`（預設 50）個 episode，只跑 always-opt 與 always-pess。pess 是與 $z_r$ 夾角 7.5 度的 CQL latent。輸出為 `data/inputs/opt_pess_s<seed>_<設定>.csv`。腳本內的 checkpoint、ensemble 與 latent 路徑對應 T06，雜湊見 `tasks/T06-latent-switching/configs/sources.json`。

```bash
mkdir -p logs && sbatch tasks/T3.2-ood-robustness/run_opt_pess.sh
```

由 `data/inputs/` 產生 `data/opt_pess.csv`（[REPORT.md](REPORT.md) 的表格來源），`--check` 逐 byte 核對既有輸出：

```bash
PYTHONPATH=.:tasks/T3.2-ood-robustness uv run python -m analyze.opt_pess
PYTHONPATH=.:tasks/T3.2-ood-robustness uv run python -m analyze.opt_pess --check
```

## Random 與 Q-guided switching 的比較

`run_switch.sh` 的結構同上，每個任務跑四個條件：always-opt、always-pess、random（每步以 $p=0.5$ 取 pess 動作）與 Q-guided switching（`zero_shot_rl.eval.switch` 的 rule C，$\kappa=0$，即只比較 $Q(a_{opt})$ 與 $Q(a_{pess})$，$\kappa$ 與 random 的設定同 T06 的最終比較）。Q 值由 T06 對應 checkpoint 的 ensemble 估計。輸出為 `data/inputs/switch_s<seed>_<設定>.csv`，其中 opt 與 pess 的 return 與 `opt_pess_*` 逐位元相同。

```bash
mkdir -p logs && sbatch tasks/T3.2-ood-robustness/run_switch.sh
PYTHONPATH=.:tasks/T3.2-ood-robustness uv run python -m analyze.switch
PYTHONPATH=.:tasks/T3.2-ood-robustness uv run python -m analyze.switch --check
```

## Q 的排序與 ensemble 分散度

`zero_shot_rl.eval.qtrace` 沿著 always-opt 的軌跡逐步記錄兩個動作的 ensemble Q 與標準差 $u$，並每 25 步從該狀態分出兩支（第一步分別取 $a_{opt}$ 與 $a_{pess}$，之後都跟著 $\pi(\cdot,z_r)$ 走 100 步），得到實際折扣回報，用來檢查 Q 的排序是否與實際回報差一致。`run_qtrace.sh` 對每個（訓練 seed、動力學設定）跑 30 個 episode，輸出 `data/inputs/qtrace_s<seed>_<設定>.npz`。

```bash
mkdir -p logs && sbatch tasks/T3.2-ood-robustness/run_qtrace.sh
PYTHONPATH=.:tasks/T3.2-ood-robustness uv run python -m analyze.qtrace
PYTHONPATH=.:tasks/T3.2-ood-robustness uv run python -m analyze.qtrace --check
PYTHONPATH=.:tasks/T3.2-ood-robustness uv run python -m analyze.qtrace_plot
```

最後一個命令畫 `figures/qtrace_series.png`：每個訓練 seed 與偏移類型一張圖，實線是 always-opt 實際動作的 ensemble Q（30 個 episode 平均），虛線是 $Q\pm u$，比較 clean、mild 與 strong。

## 測試

```bash
PYTHONPATH=. MUJOCO_GL=egl uv run python -m unittest tests.test_dynamics
```

測試涵蓋：空設定與原環境逐位元相同；每個欄位都會改變軌跡；`reset()` 還原 `base`；事件在指定 step 才生效，之前的觀測與原環境相同；事件可疊加；改質量時慣量等比例縮放且不重設狀態；無效設定會報錯。

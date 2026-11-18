# T4.1.3.2：FB action adaptation（FB-AA）

結果見 [REPORT.md](REPORT.md)。目前完成的部分：

- 偵測動力學偏移的診斷：單步轉移是否含有 friction 偏移的資訊（每個偏移設定一個分類器，與 T4.1.2.1 的動力學模型誤差比較）、偏移差異隨步數的累積與多步預測、預訓練 FB 的 $B_0(s')$ 是否保留偏移差異。
- 序列 context encoder：在多組隨機動力學上訓練 encoder 與 $B_0(s')$ 預測器，在 8 個測試設定上評估偵測與預測。
- 動作修正：以 context 條件的 $B_0(s')$ 預測器優化動作（`zero_shot_rl.eval.adapt_step`）的單步檢查，gear 偏移下動作範圍造成的上限（`zero_shot_rl.eval.action_bound`），以及閉迴路 return（`zero_shot_rl.eval.adapt_episode`）。

| 路徑 | 內容 |
|---|---|
| `analyze/classifier.py` | 每個偏移設定一個 MLP 分類器，輸入 $(s,a,s'-s)$；按 episode 切測試集；輸出測試樣本的 logit、$s'$ 差異與學習曲線 |
| `run_classifier.sh` | 在遠端重新產生配對單步資料（`zero_shot_rl.eval.paired_step`，與 T4.1.2.2 相同）並訓練分類器 |
| `zero_shot_rl/eval/paired_horizon.py` | 從 baseline rollout 的同一物理狀態，在各動力學中跑開迴路與閉迴路 $K$ 步 |
| `run_horizon.sh` | 在遠端跑 `paired_horizon`：3 個 seed、episode 0 到 19、起點 0 到 850 每 50 步、100 步 |
| `analyze/pack.py` | 把 `run_horizon.sh` 的輸出（加上 T4.1.2.1 動力學模型的多步與單步誤差）與配對單步資料的 $B_0$ 距離壓成 `data/inputs/horizon.npz`、`b0.npz` |
| `analyze/single_step.py` | 有作用步驟的比例與集中度、舊方法與分類器的 AUC、正確率 → `data/single_step.csv` |
| `analyze/diagnose.py` | 差異累積、多步預測的 AUC、$B_0$ 的保留倍率與單步移動距離 → `data/horizon.csv`、`multistep.csv`、`b0.csv` |
| `configs/multi_dynamics.yaml` | 訓練用的 300 組隨機動力學（`zero_shot_rl.data.multi_dynamics`） |
| `run_multi_dynamics.sh` | 在遠端產生訓練用的閉迴路 episode：3 個 FB seed 平行，CPU |
| `run_context.sh` | 訓練 context encoder 與預測器（`zero_shot_rl.train.context_encoder`）：3 個 FB seed × Transformer、mean，一張 GPU 上平行 |
| `configs/test_dynamics.yaml` | baseline 與 8 個測試設定，各 240 個 episode |
| `run_context_eval.sh` | 產生測試 episode，再用 6 個模型的最佳 checkpoint 評分（`zero_shot_rl.eval.context_score`） |
| `analyze/pack_context.py` | 把訓練的驗證紀錄與評分結果壓成 `data/inputs/context.npz` |
| `analyze/context.py` | 驗證準確率、偵測 AUC（全部片段、按起點）、$g$ 的誤差 → `data/context_val.csv`、`context_auc.csv`、`context_start.csv`、`context_err.csv` |
| `zero_shot_rl/latent/action_adapt.py` | 對預測器優化動作 `adapt()`，以及 gear 偏移下動作範圍內的最佳動作 `gear_inverse()` |
| `zero_shot_rl/eval/adapt_step.py`、`run_adapt_step.sh` | 優化後的動作在各動力學中走一步，與 stock 結果比較：3 個 FB seed 平行，CPU |
| `zero_shot_rl/eval/action_bound.py`、`run_action_bound.sh` | stock 動作與 $a^*=\mathrm{clip}(a/m)$ 在各動力學中走一步：3 個 FB seed 平行，CPU，約 1 分鐘 |
| `analyze/action_bound.py` | 把兩個 job 的輸出壓成 `data/inputs/action_bound.npz`；力矩缺口與補回比例 → `data/action_bound.csv`；`figures/action_bound.png` |
| `zero_shot_rl/eval/adapt_episode.py`、`run_adapt_episode.sh` | 不修正、理想修正、FB-AA 在 stock 與 4 個 gear 偏移中跑完整 episode，記錄 return：3 個 FB seed 平行，CPU，約 5 分鐘。額外參數會傳給 entry（試跑：`--tasks walk --episodes 1 --calib-episodes 1`） |
| `analyze/adapt_episode.py` | 把輸出壓成 `data/inputs/adapt_episode.npz`；return 與配對差值 → `data/adapt_episode.csv`、`adapt_episode_task.csv`；`figures/adapt_episode.png` |
| `tests/` | 分析函式的測試 |
| `data/inputs/classifier.npz` | `run_classifier.sh` 輸出的 `out/classifier.npz` |

舊方法的分數讀 `tasks/T4.1.2.1-action-conditioned-detection/data/inputs/dynamics_scored_delta.npz`，測試轉移與分類器相同（`analyze.single_step` 會核對兩者的 $s'$ 差異一致）。

## 執行

```bash
CKPT_DIR=<含 zsrl-32.pt zsrl-41.pt zsrl-42.pt 的目錄> bash tasks/T4.1.3.2-fb-aa/run_classifier.sh
cp out/classifier.npz tasks/T4.1.3.2-fb-aa/data/inputs/classifier.npz
CKPT_DIR=<同上> bash tasks/T4.1.3.2-fb-aa/run_horizon.sh
PYTHONPATH=.:tasks/T4.1.3.2-fb-aa uv run python -m analyze.pack --horizon out --paired out \
  --model <T4.1.2.1 的 dynamics.pt（3x512，目標 Delta s）> \
  --checkpoints <zsrl-32.pt> <zsrl-41.pt> <zsrl-42.pt>
```

序列 context encoder：

```bash
CKPT_DIR=<同上> bash tasks/T4.1.3.2-fb-aa/run_multi_dynamics.sh
CKPT_DIR=<同上> DATA_DIR=<含 multi_dynamics_seed{0,1,2}.npz 的目錄> \
  STOCK_DIR=<含 T07 fb_trajectories_seed{0,1,2}.npz 的目錄> bash tasks/T4.1.3.2-fb-aa/run_context.sh
CKPT_DIR=<同上> MODEL_DIR=<含 context_s*_*_best.pt 的目錄> STOCK_DIR=<同上> \
  bash tasks/T4.1.3.2-fb-aa/run_context_eval.sh
PYTHONPATH=.:tasks/T4.1.3.2-fb-aa uv run python -m analyze.pack_context \
  --history <run_context.sh 的 out> --scores <run_context_eval.sh 的 out>
```

Baseline 軌跡取自 T07 的 `fb_trajectories_seed{0,1,2}.npz`（環境 seed 0 到 49），只讀取，不需要 T07 的程式。產生 episode 用 CPU，約 1 到 2 分鐘；6 個模型在一張 GPU 上平行訓練約 37 分鐘。訓練不保證逐位元相同，所以 `data/inputs/context.npz` 是某一次訓練的結果。

配對資料與多步重播在 CPU 上產生，三個 seed 平行；8 個分類器在一張 GPU 上依序訓練，各 60 epoch。重新訓練不保證逐位元相同，所以 `data/inputs/classifier.npz` 是某一次訓練的結果。`analyze.pack` 只做推論，在 CPU 上約一分鐘。

動作修正：

```bash
CKPT_DIR=<同上> MODEL_DIR=<含 context_s*_transformer_best.pt 的目錄> bash tasks/T4.1.3.2-fb-aa/run_adapt_step.sh
CKPT_DIR=<同上> bash tasks/T4.1.3.2-fb-aa/run_action_bound.sh
PYTHONPATH=.:tasks/T4.1.3.2-fb-aa uv run python -m analyze.action_bound pack \
  --bound <含 action_bound_s*.npz 的目錄> --adapt <含 adapt_step_s*.npz 的目錄>
CKPT_DIR=<同上> MODEL_DIR=<同上> bash tasks/T4.1.3.2-fb-aa/run_adapt_episode.sh
PYTHONPATH=.:tasks/T4.1.3.2-fb-aa uv run python -m analyze.adapt_episode pack --dir <含 adapt_episode_s*.npz 的目錄>
```

`adapt_step` 與 `action_bound` 兩個 job 的環境種子與 policy 相同，stock 軌跡是確定的，`analyze.action_bound pack` 會核對兩者的取樣步驟與不修正的 $B_0$ 距離一致。

## 重現表格

```bash
PYTHONPATH=.:tasks/T4.1.3.2-fb-aa uv run python -m analyze.single_step [--check]
PYTHONPATH=.:tasks/T4.1.3.2-fb-aa uv run python -m analyze.diagnose [--check]
PYTHONPATH=.:tasks/T4.1.3.2-fb-aa uv run python -m analyze.context [--check]
PYTHONPATH=.:tasks/T4.1.3.2-fb-aa uv run python -m analyze.action_bound [--check]
PYTHONPATH=.:tasks/T4.1.3.2-fb-aa uv run python -m analyze.action_bound figure
PYTHONPATH=.:tasks/T4.1.3.2-fb-aa uv run python -m analyze.adapt_episode [--check]
PYTHONPATH=.:tasks/T4.1.3.2-fb-aa uv run python -m analyze.adapt_episode figure
```

`--check` 逐 byte 核對 `data/` 的 CSV。對應 REPORT 的欄位：

- 單步：正確率為 `single_step.csv` 的 `clf_acc`，有作用步驟為 `*_active`，$\lVert\delta\rVert/\lVert e\rVert$ 為 `gap_ratio`。
- 差異累積：`horizon.csv` 的 `open_<config>`、`closed_<config>`，浮點誤差為 `floor`。
- 多步預測：`multistep.csv` 的 `auc_multi`、`auc_onestep`，baseline 的誤差為 `err_stock`。
- $B_0$：`b0.csv` 的 `ratio_s`、`ratio_b`、`keep_step`、`keep_scale`、`step_stock`、`step_config`、`auc_step`。
- Context encoder：驗證準確率為 `context_val.csv` 的 `acc_h*`；偵測 AUC 為 `context_auc.csv` 的 `mean`、`min`、`max`（`tasks` 為 `all` 或 `walk`）；按起點的 AUC 為 `context_start.csv` 的 `auc`；預測的改善比例為 `context_err.csv` 的 `gain`，誤差為 `ctx_s0`、`stock_s0`。
- 動作修正：`action_bound.csv` 的 `infeasible`（無法補償的動作維度）、`shortfall`（無法產生的力矩）、`rec_star`、`rec_window`、`rec_oracle`（補回比例），距離為 `b_none`、`b_star`、`b_window`、`b_oracle`。圖由 `figure` 子命令重畫，不核對 byte。
- 閉迴路 return：`adapt_episode.csv` 的 `mean`、`d_none`（與不修正的配對差值）、`d_none_se`、`d_stock`（與 stock 不修正的配對差值）、`da`（動作平均改變量）；各任務的平均為 `adapt_episode_task.csv`。圖由 `figure` 子命令重畫。

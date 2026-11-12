# T4.1.3.2：FB action adaptation（FB-AA）

結果見 [REPORT.md](REPORT.md)。目前完成的部分是偵測動力學偏移的診斷：單步轉移是否含有 friction 偏移的資訊（每個偏移設定一個分類器，與 T4.1.2.1 的動力學模型誤差比較）、偏移差異隨步數的累積與多步預測、預訓練 FB 的 $B_0(s')$ 是否保留偏移差異。

| 路徑 | 內容 |
|---|---|
| `analyze/classifier.py` | 每個偏移設定一個 MLP 分類器，輸入 $(s,a,s'-s)$；按 episode 切測試集；輸出測試樣本的 logit、$s'$ 差異與學習曲線 |
| `run_classifier.sh` | 在遠端重新產生配對單步資料（`zero_shot_rl.eval.paired_step`，與 T4.1.2.2 相同）並訓練分類器 |
| `zero_shot_rl/eval/paired_horizon.py` | 從 baseline rollout 的同一物理狀態，在各動力學中跑開迴路與閉迴路 $K$ 步 |
| `run_horizon.sh` | 在遠端跑 `paired_horizon`：3 個 seed、episode 0 到 19、起點 0 到 850 每 50 步、100 步 |
| `analyze/pack.py` | 把 `run_horizon.sh` 的輸出（加上 T4.1.2.1 動力學模型的多步與單步誤差）與配對單步資料的 $B_0$ 距離壓成 `data/inputs/horizon.npz`、`b0.npz` |
| `analyze/single_step.py` | 有作用步驟的比例與集中度、舊方法與分類器的 AUC、正確率 → `data/single_step.csv` |
| `analyze/diagnose.py` | 差異累積、多步預測的 AUC、$B_0$ 的保留倍率與單步移動距離 → `data/horizon.csv`、`multistep.csv`、`b0.csv` |
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

配對資料與多步重播在 CPU 上產生，三個 seed 平行；8 個分類器在一張 GPU 上依序訓練，各 60 epoch。重新訓練不保證逐位元相同，所以 `data/inputs/classifier.npz` 是某一次訓練的結果。`analyze.pack` 只做推論，在 CPU 上約一分鐘。

## 重現表格

```bash
PYTHONPATH=.:tasks/T4.1.3.2-fb-aa uv run python -m analyze.single_step [--check]
PYTHONPATH=.:tasks/T4.1.3.2-fb-aa uv run python -m analyze.diagnose [--check]
```

`--check` 逐 byte 核對 `data/` 的 CSV。對應 REPORT 的欄位：

- 單步：正確率為 `single_step.csv` 的 `clf_acc`，有作用步驟為 `*_active`，$\lVert\delta\rVert/\lVert e\rVert$ 為 `gap_ratio`。
- 差異累積：`horizon.csv` 的 `open_<config>`、`closed_<config>`，浮點誤差為 `floor`。
- 多步預測：`multistep.csv` 的 `auc_multi`、`auc_onestep`，baseline 的誤差為 `err_stock`。
- $B_0$：`b0.csv` 的 `ratio_s`、`ratio_b`、`keep_step`、`keep_scale`、`step_stock`、`step_config`、`auc_step`。

# T4.1.3.2：FB action adaptation（FB-AA）

結果見 [REPORT.md](REPORT.md)。目前完成的部分：檢驗單步轉移是否含有偵測 friction 偏移的資訊。對每個偏移設定訓練一個分類器區分 baseline 與偏移的 $s'$，與 T4.1.2.1 的動力學模型誤差比較。

| 路徑 | 內容 |
|---|---|
| `analyze/classifier.py` | 每個偏移設定一個 MLP 分類器，輸入 $(s,a,s'-s)$；按 episode 切測試集；輸出測試樣本的 logit、$s'$ 差異與學習曲線 |
| `run_classifier.sh` | 在遠端重新產生配對資料（`zero_shot_rl.eval.paired_step`，與 T4.1.2.2 相同）並訓練分類器 |
| `analyze/single_step.py` | 有作用步驟的比例與集中度、舊方法與分類器的 AUC、正確率 → `data/single_step.csv` |
| `tests/test_single_step.py` | 上述分析函式的測試 |
| `data/inputs/classifier.npz` | `run_classifier.sh` 輸出的 `out/classifier.npz` |

舊方法的分數讀 `tasks/T4.1.2.1-action-conditioned-detection/data/inputs/dynamics_scored_delta.npz`，測試轉移與分類器相同（`analyze.single_step` 會核對兩者的 $s'$ 差異一致）。

## 執行

```bash
CKPT_DIR=<含 zsrl-32.pt zsrl-41.pt zsrl-42.pt 的目錄> bash tasks/T4.1.3.2-fb-aa/run_classifier.sh
cp out/classifier.npz tasks/T4.1.3.2-fb-aa/data/inputs/classifier.npz
```

配對資料在 CPU 上產生，三個 seed 平行；8 個分類器在一張 GPU 上依序訓練，各 60 epoch。重新訓練不保證逐位元相同，所以 `data/inputs/classifier.npz` 是某一次訓練的結果。

## 重現表格

```bash
PYTHONPATH=.:tasks/T4.1.3.2-fb-aa uv run python -m analyze.single_step [--check]
```

`--check` 逐 byte 核對 `data/single_step.csv`。REPORT 的正確率為 `clf_acc`，有作用步驟的欄位為 `*_active`，$\lVert\delta\rVert/\lVert e\rVert$ 為 `gap_ratio`。

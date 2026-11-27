# T6.5-more-env

將 CEM、CMA-ES、LoLA 潛向量適應推廣至 `quadruped` 與 `point_mass_maze`。設計與預期結果見 `REPORT.md`。

## 資料集位置（`idlab1`）

- Quadruped RND-100k：`$HOME/zsrl-datasets/quadruped/rnd/dataset.npz`
- Point Mass Maze RND-100k：`$HOME/zsrl-datasets/point_mass_maze/rnd/dataset.npz`

## 第一階段：FB 預訓練

於 `idlab1` 以 `lab sweep` 提交 5 個 seed（`0..4`），每 20k learning steps 評估並儲存 checkpoint（共 1M learning steps）：

```bash
lab sweep tasks/T6.5-more-env/configs/train_fb_quadruped.yaml
lab sweep tasks/T6.5-more-env/configs/train_fb_point_mass_maze.yaml
```

## Checkpoint 挑選流程（比照 T4.1.3.1）

從各 seed 的訓練 log 萃取評估分數後，以 `analysis.table6_aggregate` 選出 5-seed 全任務 IQM 最高的步數，再於該步挑選四任務 IQM 最高的 seed 作為凍結 FB backbone：

```bash
python analysis/extract_eval.py <job output.log> \
  tasks/T6.5-more-env/data/fb_train/eval_fb_<domain>_s<seed>.csv

PYTHONPATH=. uv run python -m analysis.table6_aggregate \
  tasks/T6.5-more-env/data/fb_train/eval_fb_<domain>_s{0,1,2,3,4}.csv
```

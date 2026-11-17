# T5.2：Action Saturation，actuator 功率上界測試

問題：把 walker 全部 actuator 的 gear 由 XML 原值逐步調低，標準 FB 在固定 $z_r$ 上的 return 降到多少時開始崩跌。

## 設定

- Checkpoint：標準 FB（T4.1.3.1 的 `zero-shot-rl-297/660000.pt`），資料集 walker rnd。
- gear 縮放：100、95、90、85、80、75、70、65、60、50、40、30 % 的 XML 原值（hip 100、knee 50、ankle 20），六個關節同比例（`configs/gear_*.yaml`）。
- Tasks：walk、run、stand、flip。每個 task 與 gear 跑 20 個 episode，環境種子 0 到 19，所有 gear 共用；每 episode 1000 步，動作為 clipped mean action。
- 紀錄：return、平均水平速度、torso 高度低於 0.6 的步數比例。

## 重現

```
CKPT=<fb.pt> DATASET=<walker rnd dataset.npz> bash tasks/T5.2-action-saturation/run_sweep.sh   # 在 idlab 上
# 複製 out/sweep_*.csv 到 tasks/T5.2-action-saturation/data/inputs/
PYTHONPATH=. uv run python tasks/T5.2-action-saturation/analyze/curve.py
```

`zero_shot_rl/latent/infer.py` 新增 `--z_r_only`，只寫 $z_r$ 列；預設行為不變。

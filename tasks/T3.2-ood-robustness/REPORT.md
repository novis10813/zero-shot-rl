# T3.2 結果：optimistic、pessimistic、random 與 Q-guided 在動力學偏移下的 return

| 條件 | Optimistic | Pessimistic | Random | Q-guided | Q-guided − 較好的固定策略（標準誤差） |
|---|---|---|---|---|---|
| Clean | 552.1 ± 91.0 | 570.5 ± 117.3 | 566.9 ± 117.1 | 579.4 ± 125.8 | +9.0 (11.8) |
| Friction — Mild | 475.8 ± 138.6 | 481.6 ± 159.1 | 487.7 ± 153.9 | 500.1 ± 148.1 | +18.5 (13.8) |
| Friction — Strong | 371.1 ± 120.4 | 360.7 ± 142.1 | 365.9 ± 127.3 | 376.6 ± 135.3 | +5.6 (11.4) |
| Actuator — Mild | 435.5 ± 149.5 | 457.4 ± 145.3 | 435.4 ± 161.4 | 480.3 ± 136.8 | +22.9 (13.1) |
| Actuator — Strong | 317.4 ± 160.1 | 340.8 ± 152.7 | 325.9 ± 160.1 | 357.5 ± 154.9 | +16.7 (13.0) |

每格為 3 個訓練 seed × 50 個 episode（共 150 個配對 episode）的 return 平均 ± 標準差；Q-guided 的平均 return 在五個條件都最高，但與較好的固定策略的差值都在 1.4 個標準誤差內，只有 actuator 兩個條件勝過 random 達 2 個標準誤差以上（+44.9、+31.5，標準誤差 14.6、13.3），所以目前不能宣稱 Q-guided 優於固定策略。

## Q-guided 的選擇比例與 ensemble uncertainty

| 條件 | Q-guided 取 pess 的比例 | 相對 Clean 的差（標準誤差） | Uncertainty $u_{opt}$ | 相對 Clean 的差（標準誤差） |
|---|---|---|---|---|
| Clean | 0.463 ± 0.069 | — | 54.71 | — |
| Friction — Mild | 0.460 ± 0.100 | -0.004 (0.006) | 53.70 | -1.01 (0.39) |
| Friction — Strong | 0.448 ± 0.126 | -0.015 (0.010) | 52.69 | -2.03 (0.45) |
| Actuator — Mild | 0.445 ± 0.089 | -0.018 (0.008) | 54.88 | +0.17 (0.42) |
| Actuator — Strong | 0.471 ± 0.157 | +0.008 (0.013) | 53.49 | -1.23 (0.66) |

取 pess 的比例為 Q-guided（$\kappa=0$）在每個 episode 取 pess 動作的步數比例，平均 ± 標準差取自 3 個訓練 seed × 50 個 episode；$u_{opt}$ 為 always-opt 軌跡上 ensemble 對 Q 的標準差（單位為 $|\hat z_r|$），取自 3 個訓練 seed × 30 個 episode；兩欄的差值都是與 Clean 在相同起始狀態下的配對差，偏移下選擇比例與 uncertainty 都沒有上升（最大的比例變化為 −0.018，$u$ 在 friction 反而下降），所以 Q-guided 沒有因動力學偏移而改變選擇，ensemble 也沒有察覺到偏移。

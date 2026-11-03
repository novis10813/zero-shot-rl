# T3.2 結果：optimistic 與 pessimistic latent 在動力學偏移下的 return

| 條件 | Optimistic | Pessimistic | Pessimistic − Optimistic（標準誤差） |
|---|---|---|---|
| Clean | 552.1 ± 91.0 | 570.5 ± 117.3 | +18.3 (11.4) |
| Friction — Mild | 475.8 ± 138.6 | 481.6 ± 159.1 | +5.8 (12.6) |
| Friction — Strong | 371.1 ± 120.4 | 360.7 ± 142.1 | −10.4 (11.6) |
| Actuator — Mild | 435.5 ± 149.5 | 457.4 ± 145.3 | +21.9 (15.5) |
| Actuator — Strong | 317.4 ± 160.1 | 340.8 ± 152.7 | +23.4 (14.1) |

每格為 3 個訓練 seed × 50 個 episode（共 150 個配對 episode）的 return 平均 ± 標準差，差值為配對差的平均與標準誤差；所有差值都在 1.7 個標準誤差內，兩個 latent 在偏移下沒有一致的優劣。

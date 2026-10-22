Residual check: each family's direction away from z_r (10 deg, 5 deg if 10 is not reached) with its u_1 component removed. Mean over the 4 tasks per seed.

| Family | cos to u_1 (seed 0 / 1 / 2) | residual norm (seed 0 / 1 / 2) |
|---|---|---|
| CQL | -0.28 / -0.17 / -0.07 | 0.96 / 0.98 / 1.00 |
| ReBRAC | -0.99 / -0.99 / -0.99 | 0.13 / 0.11 / 0.13 |
| E2E | +1.00 / +1.00 / +1.00 | 0.00 / 0.00 / 0.00 |
| E2E-bi | +0.94 / +0.95 / +0.96 | 0.34 / 0.31 / 0.28 |
| C1 | +0.00 / +0.01 / +0.03 | 0.99 / 0.99 / 0.99 |

| Residual pair | cos (seed 0 / 1 / 2), mean over tasks | range over all seeds and tasks |
|---|---|---|
| C1 / CQL | -0.00 / -0.02 / -0.11 | -0.37 to +0.20 |
| C1 / E2E-bi | +0.01 / +0.05 / +0.05 | -0.28 to +0.28 |
| C1 / ReBRAC | -0.03 / +0.07 / +0.03 | -0.29 to +0.33 |
| CQL / E2E-bi | -0.37 / -0.58 / -0.44 | -0.59 to -0.35 |
| CQL / ReBRAC | -0.24 / -0.08 / -0.05 | -0.31 to +0.00 |
| E2E-bi / ReBRAC | +0.22 / +0.23 / +0.06 | -0.03 to +0.37 |

C1 rows are three random directions per task: their cosines with a family give the chance level (about +-0.14 for 48 free dimensions).

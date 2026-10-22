| Reward type | Strength: angle to z_r (coefficient, median) | Return IQM (change vs z_r) | Fall fraction | Latent: cos to z_r / level on axis / cos to axis | Main behaviour vs random rotation |
|---|---|---|---|---|---|
| task reward only | 0 | 440 (+0% +- 0) | 0.25 | 1.00 / +0 deg | reference |
| CQL(H), -alpha (mu/pi_beta - 1) | 10 deg (+0.491) | 416 (-6% +- 7) | 0.31 | 0.98 / -5 deg / -0.52 | falls more (+0.06) |
| CQL(H), -alpha (mu/pi_beta - 1) | 35 deg (+2) | 264 (-38% +- 18) | 0.50 | 0.82 / -18 deg / -0.53 | falls more (+0.13), more saturated actions (+0.07) |
| CQL(H), -alpha (mu/pi_beta - 1) | 70 deg (+9.28) | 213 (-51% +- 18) | 0.40 | 0.34 / -30 deg / -0.53 | falls less (-0.16), slower (-0.09), more saturated actions (+0.11) |
| ReBRAC, -beta ||pi(s') - a'||^2 | 10 deg (+0.00641) | 422 (-0% +- 8) | 0.27 | 0.98 / -9 deg / -0.90 | falls more (+0.02), faster (+0.07), more saturated actions (+0.02) |
| ReBRAC, -beta ||pi(s') - a'||^2 | 35 deg (+0.0178) | 392 (-3% +- 11) | 0.29 | 0.82 / -31 deg / -0.90 | falls less (-0.08), faster (+0.43), more saturated actions (+0.08) |
| ReBRAC, -beta ||pi(s') - a'||^2 | 70 deg (+0.0315) | 385 (-6% +- 17) | 0.29 | 0.34 / -58 deg / -0.90 | falls less (-0.27), faster (+0.60), more saturated actions (+0.14) |
| ReBRAC + E2E practical | 10 deg (+0.0629) | 399 (-9% +- 8) | 0.31 | 0.98 / +2 deg / +0.18 | falls more (+0.06), faster (+0.09) |
| ReBRAC + E2E practical | 35 deg (+0.243) * | 314 (-29% +- 10) | 0.49 | 0.82 / -1 deg / -0.02 | falls more (+0.11), faster (+0.45), more saturated actions (+0.03) |
| ReBRAC + E2E practical | 70 deg (+0.758) | 146 (-64% +- 10) | 0.72 | 0.34 / -11 deg / -0.21 | falls more (+0.16), closer to data (-0.06), faster (+0.58), more saturated actions (+0.10) |
| E2E practical, +c on data | 10 deg (+0.092) | 393 (-13% +- 10) | 0.30 | 0.98 / +9 deg / +0.91 | falls more (+0.05), fewer saturated actions (-0.02) |
| E2E practical, +c on data | 35 deg (+0.988) * | 161 (-62% +- 19) | 0.58 | 0.82 / +32 deg / +0.91 | falls more (+0.20), slower (-0.24), fewer saturated actions (-0.07) |
| E2E bilevel, +beta pi/pi_beta | 10 deg (+0.0846) | 382 (-14% +- 12) | 0.31 | 0.98 / +10 deg / +0.97 | falls more (+0.06) |
| common axis, pessimistic side | 35 deg | 359 (-11% +- 13) | 0.35 | 0.82 / -35 deg / -1.00 | falls less (-0.03), faster (+0.24), more saturated actions (+0.09) |
| common axis, pessimistic side | 10 deg | 425 (-1% +- 2) | 0.28 | 0.98 / -10 deg / -1.00 | falls more (+0.03), more saturated actions (+0.02) |
| common axis, optimistic side | 10 deg | 410 (-10% +- 8) | 0.27 | 0.98 / +10 deg / +1.00 | falls more (+0.02), fewer saturated actions (-0.02) |
| common axis, optimistic side | 35 deg | 257 (-42% +- 12) | 0.35 | 0.82 / +35 deg / +1.00 | falls less (-0.03), slower (-0.10), fewer saturated actions (-0.12) |
| control: random rotation | 10 deg | 428 (-3% +- 1) | 0.25 | 0.98 / +0 deg / +0.03 | - |
| control: random rotation | 35 deg | 284 (-35% +- 15) | 0.38 | 0.82 / +1 deg / +0.03 | - |
| control: random rotation | 70 deg | 131 (-73% +- 14) | 0.56 | 0.34 / +2 deg / +0.03 | - |

Direction of each family away from z_r (unit tangent at 10 deg, 5 deg if 10 is not reached): cosine, mean over seeds and tasks.

| | CQL | ReBRAC | ReBRAC+E2E | E2E | E2E-bi | C1 |
|---|---|---|---|---|---|---|
| CQL | +1.00 | +0.16 | -0.12 | -0.17 | -0.30 | -0.06 |
| ReBRAC | +0.16 | +1.00 | -0.06 | -0.99 | -0.94 | -0.00 |
| ReBRAC+E2E | -0.12 | -0.06 | +1.00 | +0.18 | +0.21 | +0.06 |
| E2E | -0.17 | -0.99 | +0.18 | +1.00 | +0.95 | +0.01 |
| E2E-bi | -0.30 | -0.94 | +0.21 | +0.95 | +1.00 | +0.02 |
| C1 | -0.06 | -0.00 | +0.06 | +0.01 | +0.02 | +1.00 |

Return: mean over 4 tasks of the IQM of 20 episodes, mean over 3 seeds; change vs z_r +- std over seeds. Fall fraction: share of steps with torso height < 0.6. Level: signed angle along the common axis (negative = pessimistic side). * = angle not reached on every task in every seed (fewer tasks averaged). Behaviour thresholds against C1 at the same angle: fall 0.02, kNN distance 0.05, speed 0.05, saturation 0.02.

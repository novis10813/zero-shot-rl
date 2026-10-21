| Reward type | Strength: angle to z_r (coefficient, median) | Return IQM (change vs z_r) | Fall fraction | Latent: cos to z_r / direction same as | Main behaviour vs random rotation |
|---|---|---|---|---|---|
| task reward only (linear z_r) | 0 | 213 (+0% +- 0) | 0.49 | 1.00 | reference |
| task reward only (paper softmax z) | 0 | 207 (-6% +- 8) | 0.52 | 0.85 | - |
| CQL(H), alpha > 0 | 10 deg (+0.317) | 204 (-4% +- 4) | 0.51 | 0.98 / own direction | like random rotation |
| CQL(H), alpha > 0 | 35 deg (+1.29) | 200 (-5% +- 8) | 0.52 | 0.82 / own direction | like random rotation |
| CQL(H), alpha > 0 | 70 deg (+6.25) | 173 (-17% +- 26) | 0.59 | 0.34 / own direction | falls more (+0.04) |
| CQL(H), alpha < 0 | 10 deg (-0.311) | 210 (-3% +- 4) | 0.48 | 0.98 / CQL alpha > 0, reversed (cos -1.00) | like random rotation |
| CQL(H), alpha < 0 | 35 deg (-1.18) | 193 (-10% +- 14) | 0.43 | 0.82 / CQL alpha > 0, reversed (cos -1.00) | falls less (-0.07), closer to data (-0.07) |
| CQL(H), alpha < 0 | 70 deg (-3.73) | 175 (-18% +- 28) | 0.49 | 0.34 / CQL alpha > 0, reversed (cos -1.00) | falls less (-0.07), closer to data (-0.11), more saturated actions (+0.08) |
| ReBRAC | 10 deg (+0.027) | 209 (-2% +- 2) | 0.49 | 0.98 / E2E c < 0 (cos 0.98) | like random rotation |
| ReBRAC | 35 deg (+0.0602) | 206 (-3% +- 1) | 0.50 | 0.82 / E2E c < 0 (cos 0.98) | faster (+0.05) |
| ReBRAC | 70 deg (+0.0917) | 185 (-14% +- 20) | 0.55 | 0.34 / E2E c < 0 (cos 0.98) | farther from data (+0.09), more saturated actions (+0.03) |
| ReBRAC, centred | 10 deg (+0.206) | 207 (-2% +- 3) | 0.50 | 0.98 / ReBRAC + E2E (cos 0.99) | like random rotation |
| ReBRAC, centred | 35 deg (+0.707) | 191 (-9% +- 13) | 0.52 | 0.82 / ReBRAC + E2E (cos 0.99) | slower (-0.06) |
| ReBRAC, centred | 70 deg (+2.96) | 171 (-19% +- 31) | 0.56 | 0.34 / ReBRAC + E2E (cos 0.99) | farther from data (+0.11), slower (-0.05), fewer saturated actions (-0.02) |
| E2E practical, c > 0 | 5 deg (+0.0563) | 211 (-1% +- 2) | 0.49 | 1.00 / own direction (= u_1) | like random rotation |
| E2E practical, c > 0 | 20 deg (+0.774) * | 176 (-7% +- 4) | 0.51 | 0.94 / own direction (= u_1) | falls more (+0.02) |
| E2E practical, c < 0 | 5 deg (-0.0401) | 209 (-1% +- 4) | 0.50 | 1.00 / ReBRAC (cos 0.98) | like random rotation |
| E2E practical, c < 0 | 20 deg (-0.12) | 208 (-3% +- 0) | 0.50 | 0.94 / ReBRAC (cos 0.98) | like random rotation |
| E2E practical, c < 0 | 50 deg (-0.207) | 203 (-5% +- 8) | 0.52 | 0.64 / ReBRAC (cos 0.98) | farther from data (+0.08) |
| ReBRAC + E2E practical | 10 deg (+0.205) | 208 (-1% +- 2) | 0.50 | 0.98 / ReBRAC centred (cos 0.99) | like random rotation |
| ReBRAC + E2E practical | 35 deg (+0.644) * | 202 (-5% +- 8) | 0.49 | 0.82 / ReBRAC centred (cos 0.99) | slower (-0.08) |
| control: random rotation | 10 deg | 208 (-2% +- 2) | 0.49 | 0.98 / none (random) | - |
| control: random rotation | 35 deg | 199 (-6% +- 7) | 0.50 | 0.82 / none (random) | - |
| control: random rotation | 70 deg | 177 (-16% +- 24) | 0.55 | 0.34 / none (random) | - |

Return: mean over 4 tasks of the IQM of 20 episodes, mean over 3 seeds; change vs z_r +- std over seeds. Fall fraction: share of steps with torso height < 0.6. * = angle not reached on every task in every seed (fewer tasks averaged). Behaviour thresholds against C1: fall 0.02, kNN distance 0.05, speed 0.05, saturation 0.02.

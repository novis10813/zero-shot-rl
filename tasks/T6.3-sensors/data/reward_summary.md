| task | z_r vs true | ANFIS 4 sensors vs true | direct z vs true | ANFIS 4 sensors vs z_r | direct z vs z_r |
|---|---:|---:|---:|---:|---:|
| stand | 0.78 | 0.70 | 0.35 | 0.87 | 0.45 |
| walk | 0.71 | 0.56 | 0.35 | 0.77 | 0.50 |
| run | 0.72 | 0.52 | 0.36 | 0.71 | 0.49 |
| flip | 0.75 | 0.61 | 0.39 | 0.80 | 0.51 |

| task | true | z_r | ANFIS 4 sensors (seed mean) | ANFIS 4 sensors (per seed) | direct z (seed mean) | direct z (per seed) |
|---|---:|---:|---:|---:|---:|---:|
| stand | 1.000 | 0.70 | 0.72 (0.71–0.73) | 0.69 | 0.38 (0.32–0.45) | 0.19 |
| walk | 0.999 | 0.67 | 0.71 (0.71–0.72) | 0.68 | 0.37 (0.36–0.40) | 0.21 |
| run | 0.992 | 0.66 | 0.72 (0.68–0.75) | 0.66 | 0.38 (0.32–0.42) | 0.22 |
| flip | 1.000 | 0.71 | 0.71 (0.69–0.72) | 0.67 | 0.43 (0.42–0.46) | 0.26 |

| target | env | b_v | R² |
|---|---|---:|---:|
| true | - | 0.84 | 1.00 |
| z_r | - | 0.74 | 0.67 |
| ANFIS 4 sensors | friction_mild | 0.38 | 0.72 |
| ANFIS 4 sensors | friction_high_mild | 0.27 | 0.71 |
| ANFIS 4 sensors | actuator_mild | 0.37 | 0.71 |
| ANFIS 4 sensors | actuator_high_mild | 0.47 | 0.72 |
| direct z | friction_mild | 0.71 | 0.40 |
| direct z | friction_high_mild | 0.82 | 0.36 |
| direct z | actuator_mild | 0.67 | 0.38 |
| direct z | actuator_high_mild | 0.80 | 0.36 |

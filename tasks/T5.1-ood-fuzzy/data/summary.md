| OOD 環境 | 方法 | stand | walk | run | flip | All-task IQM | 相對 $z_r$ |
|---|---|---:|---:|---:|---:|---:|---:|
| friction_mild | Frozen FB $z_r$ | 189.8 | 321.0 | 118.1 | 187.3 | 188.5 | +0.0 |
| friction_mild | Clean ANFIS-z 直接遷移 | 270.9 | 380.3 | 146.1 | 279.1 | 275.0 | +86.5 |
| friction_mild | OOD finetune，4 sensors | 285.2 | 373.0 | 153.0 | 280.0 | 282.6 | +94.0 |
| friction_mild | OOD finetune，客製 sensors | 242.7 | 405.6 | 156.4 | 296.8 | 269.8 | +81.2 |
| friction_high_mild | Frozen FB $z_r$ | 500.7 | 208.5 | 82.6 | 345.4 | 277.0 | +0.0 |
| friction_high_mild | Clean ANFIS-z 直接遷移 | 573.3 | 257.3 | 154.0 | 369.0 | 313.2 | +36.2 |
| friction_high_mild | OOD finetune，4 sensors | 569.2 | 300.2 | 151.5 | 371.7 | 335.9 | +59.0 |
| friction_high_mild | OOD finetune，客製 sensors | 575.3 | 314.0 | 138.4 | 371.4 | 342.7 | +65.7 |
| actuator_mild | Frozen FB $z_r$ | 442.6 | 122.7 | 99.0 | 239.0 | 180.9 | +0.0 |
| actuator_mild | Clean ANFIS-z 直接遷移 | 444.4 | 173.9 | 116.3 | 304.4 | 239.1 | +58.3 |
| actuator_mild | OOD finetune，4 sensors | 447.9 | 179.6 | 147.5 | 306.1 | 242.9 | +62.0 |
| actuator_mild | OOD finetune，客製 sensors | 439.8 | 181.2 | 154.7 | 319.2 | 250.2 | +69.3 |
| actuator_high_mild | Frozen FB $z_r$ | 681.9 | 617.2 | 251.0 | 350.3 | 483.8 | +0.0 |
| actuator_high_mild | Clean ANFIS-z 直接遷移 | 694.1 | 628.4 | 238.6 | 421.6 | 525.0 | +41.3 |
| actuator_high_mild | OOD finetune，4 sensors | 700.2 | 619.9 | 242.1 | 423.0 | 521.4 | +37.7 |
| actuator_high_mild | OOD finetune，客製 sensors | 711.3 | 637.0 | 252.2 | 435.1 | 536.1 | +52.3 |
| 四環境平均 | Frozen FB $z_r$ | 453.7 | 317.4 | 137.7 | 280.5 | 282.5 | +0.0 |
| 四環境平均 | Clean ANFIS-z 直接遷移 | 495.7 | 360.0 | 163.7 | 343.5 | 338.1 | +55.6 |
| 四環境平均 | OOD finetune，4 sensors | 500.6 | 368.2 | 173.5 | 345.2 | 345.7 | +63.2 |
| 四環境平均 | OOD finetune，客製 sensors | 492.3 | 384.5 | 175.4 | 355.6 | 349.7 | +67.2 |

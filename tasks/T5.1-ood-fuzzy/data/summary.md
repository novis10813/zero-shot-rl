| OOD 環境 | 方法 | 選定 step | stand | walk | run | flip | All-task IQM | 相對 $z_r$ | 第 50 步 | 最後一步 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| friction_mild | Frozen FB $z_r$ | - | 189.8 | 321.0 | 118.1 | 187.3 | 188.5 | +0.0 | 188.5 | 188.5 |
| friction_mild | Clean ANFIS-z 直接遷移 | - | 270.9 | 380.3 | 146.1 | 279.1 | 275.0 | +86.5 | 275.0 | 275.0 |
| friction_mild | OOD finetune，4 sensors | 50 | 285.2 | 373.0 | 153.0 | 280.0 | 282.6 | +94.0 | 282.6 | 282.6 |
| friction_mild | OOD finetune，客製 sensors | 50 | 242.7 | 405.6 | 156.4 | 296.8 | 269.8 | +81.2 | 269.8 | 269.8 |
| friction_high_mild | Frozen FB $z_r$ | - | 500.7 | 208.5 | 82.6 | 345.4 | 277.0 | +0.0 | 277.0 | 277.0 |
| friction_high_mild | Clean ANFIS-z 直接遷移 | - | 573.3 | 257.3 | 154.0 | 369.0 | 313.2 | +36.2 | 313.2 | 313.2 |
| friction_high_mild | OOD finetune，4 sensors | 40 | 573.6 | 332.8 | 141.5 | 363.4 | 348.1 | +71.2 | 335.9 | 335.9 |
| friction_high_mild | OOD finetune，客製 sensors | 30 | 553.8 | 322.7 | 136.6 | 364.2 | 343.5 | +66.5 | 342.7 | 342.7 |
| actuator_mild | Frozen FB $z_r$ | - | 442.6 | 122.7 | 99.0 | 239.0 | 180.9 | +0.0 | 180.9 | 180.9 |
| actuator_mild | Clean ANFIS-z 直接遷移 | - | 444.4 | 173.9 | 116.3 | 304.4 | 239.1 | +58.3 | 239.1 | 239.1 |
| actuator_mild | OOD finetune，4 sensors | 40 | 448.2 | 186.1 | 136.0 | 302.2 | 244.2 | +63.3 | 242.9 | 242.9 |
| actuator_mild | OOD finetune，客製 sensors | 50 | 439.8 | 181.2 | 154.7 | 319.2 | 250.2 | +69.3 | 250.2 | 250.2 |
| actuator_high_mild | Frozen FB $z_r$ | - | 681.9 | 617.2 | 251.0 | 350.3 | 483.8 | +0.0 | 483.8 | 483.8 |
| actuator_high_mild | Clean ANFIS-z 直接遷移 | - | 694.1 | 628.4 | 238.6 | 421.6 | 525.0 | +41.3 | 525.0 | 525.0 |
| actuator_high_mild | OOD finetune，4 sensors | 50 | 700.2 | 619.9 | 242.1 | 423.0 | 521.4 | +37.7 | 521.4 | 521.4 |
| actuator_high_mild | OOD finetune，客製 sensors | 50 | 711.3 | 637.0 | 252.2 | 435.1 | 536.1 | +52.3 | 536.1 | 536.1 |
| 四環境平均 | Frozen FB $z_r$ | - | 453.7 | 317.4 | 137.7 | 280.5 | 282.5 | +0.0 | 282.5 | 282.5 |
| 四環境平均 | Clean ANFIS-z 直接遷移 | - | 495.7 | 360.0 | 163.7 | 343.5 | 338.1 | +55.6 | 338.1 | 338.1 |
| 四環境平均 | OOD finetune，4 sensors | - | 501.8 | 378.0 | 168.2 | 342.1 | 349.1 | +66.6 | 345.7 | 345.7 |
| 四環境平均 | OOD finetune，客製 sensors | - | 486.9 | 386.6 | 175.0 | 353.8 | 349.9 | +67.3 | 349.7 | 349.7 |

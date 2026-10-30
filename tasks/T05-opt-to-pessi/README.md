# T05：重現 standard FB 的 latent 與配對 rollout 比較

Standard FB 推論與 rollout 不依賴 one-step FB 的模組。`implicit_z_fb.py` 使用
`latent_solver.py` 的固定點與夾角搜尋，以及 `agents/fb/inference_buffer.py` 的
`FBInferenceReplayBuffer`，保留資料中下一個動作的對齊與 episode 結尾排除規則。
`rollout_eval.py` 只需觀測與動作，因此使用一般 `FBReplayBuffer`。既有命令、取樣
順序、數值計算與輸出格式不變。

## 從既有 latent 重建所有 Q1 表格

在 repo 根目錄執行：

```bash
.venv/bin/python tasks/T05-opt-to-pessi/analyze_q1_fb.py
.venv/bin/python -m unittest discover -s tasks/T05-opt-to-pessi -p 'test_q1_fb.py'
.venv/bin/python tasks/T05-opt-to-pessi/analyze_q1_fb.py --check
.venv/bin/python tasks/T05-opt-to-pessi/analyze_q1_fb.py --report-tables
```

第一個命令重建六份 CSV。`--check` 重新計算並逐份核對 CSV，若缺少檔案或結果過期則失敗，不寫入檔案。Q1 的 `--report-tables` 輸出尚未填入 Q2 欄位的主表模板、方法兩兩比較表與子集結果文字，不寫入檔案。測試核對 [REPORT.md](REPORT.md) 的 Q1 欄位與數據，完整主表則由下方的 Q2 命令重建。

## 只在同一模型與任務內比較向量

輸入為 `data/zs_fb_zero-shot-rl-30.npz`、`data/zs_fb_zero-shot-rl-31.npz`、`data/zs_fb_zero-shot-rl-32.npz`，分別對應訓練隨機種子 0、1、2，來源記錄於 `exp/T05-zero-shot-rl-{30,31,32}` tags。每個模型包含站立、行走、跑步、翻轉四個 Walker 任務。

每份輸入以原始任務 reward 的 latent 為基準。計算先在各模型、各任務內完成，再彙整角度、距離與 cosine，不跨模型平均 latent 座標。結果為描述性平均與最小／最大值，不將四個任務當成額外訓練 seeds，也不提供顯著性檢定。

`implicit_z_fb.py` 產生的五種 reward 修正對應如下。這些名稱表示投影規則，不表示重現原始演算法訓練出的 policy。

| 輸入 family 欄位 | CSV 完整名稱 | reward 修正的意義 |
|---|---|---|
| `CQL` | CQL implicit-reward projection | 依價值偏好的動作分布與資料動作分布之差產生修正 |
| `ReBRAC` | ReBRAC action-deviation penalty | 懲罰 policy 動作與資料中下一個動作的平方距離 |
| `ReBRAC+E2E` | ReBRAC penalty plus norm-matched constant reward | 組合動作偏離懲罰與正的常數 reward，兩項在原始 reward latent 處的偏移向量長度配平 |
| `E2E` | Positive constant reward | 對所有推論樣本加上相同的正 reward |
| `E2E-bi` | Deterministic-policy next-state bonus (bilevel E2E-style) | 以密度比 reward 的換測度形式為出發點，但實作只取 policy 的確定性動作（`sample=False`），模擬下一狀態後平均 backward 特徵作為 latent 修正，不直接估計密度比 |

資料中的 `z_r` 是原始 reward 基準，`C2` 是將推論資料分成五個互不重疊子集後，各自重算的原始 reward latent。構造的候選 `AX` 旋轉軸與隨機旋轉不納入 Q1 方法比較；`AX` 的獨立行為診斷見下方 Q4–Q6 重現說明。

## 角度與偏移方向回答不同問題

以原始 reward latent 的單位向量為基準，先把各修正 reward latent 單位化，再移除沿基準的分量並正規化：

$$
u_r=\frac{z_r}{\|z_r\|},\qquad
u_i=\frac{z_i}{\|z_i\|},\qquad
t_i=\frac{u_i-(u_i^\top u_r)u_r}{\|u_i-(u_i^\top u_r)u_r\|}.
$$

`angle_to_raw_deg` 是完整 latent 與基準的夾角。`tangent_cosine` 是兩種修正的偏移方向內積，接近 1 表示同向，接近 −1 表示反向，接近 0 表示近乎垂直。`latent_cosine`、`latent_angle_deg` 則比較兩個完整 latent。

五種方法在所有模型與任務都具備目標 5 度和 10 度的結果，因此只在這兩個共同目標角度比較方法。其他角度保留在基準比較的明細，不將缺少某方法的角度混入平衡彙整。係數本來就是為達到指定角度而搜尋，偏離基準本身不是獨立的實驗發現。

子集控制比較每個子集的原始 reward latent 與完整推論資料估計的夾角。五個子集各約使用完整資料的五分之一，且與完整估計共用資料。這是有限樣本敏感度參考，不是完整資料估計的信賴區間，也沒有估計 implicit reward 偏移方向本身的誤差。

## 六份 CSV 保留明細與彙整

| 檔案，均位於 `data/` | 內容 |
|---|---|
| `q1_fb_baseline.csv` | 全部 279 個修正 reward latent：來源列、係數、目標／實際夾角、與原始 reward 的 cosine／距離、子集控制變動尺度 |
| `q1_fb_subsets.csv` | 60 個子集 latent 與完整資料基準的角度／cosine；子集序號僅是輸入檔內順序 |
| `q1_fb_pairs.csv` | 240 組同模型、同任務、同目標角度的方法比較：完整 latent 與偏移方向的相似度 |
| `q1_fb_baseline_summary.csv` | 共同目標角度下，各方法相對基準的平均與範圍，逐任務及四任務等權彙整 |
| `q1_fb_subset_summary.csv` | 子集估計角度的平均與範圍，逐任務及四任務等權彙整 |
| `q1_fb_pair_summary.csv` | 方法兩兩比較的平均與範圍，逐任務及四任務等權彙整 |

所有 summary 都記錄比較數、訓練 seed 數與任務數。每個方法 pair 的四任務彙整有 12 組比較，來自 3 個訓練 seeds，每個 seed 含 4 個任務。

## Standard FB 的 PCA、熱圖與常數方向控制可由既有 latent 重建

只讀上述三份 `zs_fb_*.npz`，不分析 one-step FB，不執行模型、不新增 rollout。主圖採五種修正都有的目標 10 度，所有 CSV 同時保存 5 度與 10 度。來源列可回溯到各 NPZ，既有 Q1 表格與原始資料不覆寫。

```bash
.venv/bin/python tasks/T05-opt-to-pessi/analyze_latent_fb.py
.venv/bin/python tasks/T05-opt-to-pessi/analyze_latent_fb.py --report-tables
.venv/bin/python tasks/T05-opt-to-pessi/analyze_latent_fb.py --check
.venv/bin/python -m unittest discover -s tasks/T05-opt-to-pessi -p 'test_latent_fb.py'
```

第一個命令重建七份 CSV 與八張 PNG。`--check` 重算並逐 byte 核對所有 CSV／PNG，不寫檔；PNG 的逐 byte 重現要求相同 NumPy、Matplotlib、字型與繪圖環境。`--report-tables` 印出報告中的兩表及完整 latent PCA 的變異保留範圍。測試包含已知單軸／正交向量、旋轉不變性、獨立重算殘差與 SVD、PCA 投影、來源覆蓋率、10 度熱圖矩陣與 CSV 一致性、表格與報告一致性，以及 `--check` 對缺檔／過期資料拒絕且不寫檔的行為。

常數 reward 的原始 latent 修正方向由 NPZ 的 `u_1`（推論資料的平均 backward 特徵）提供。先移除其沿原始 reward 單位向量的分量並正規化，得到常數控制，再從 Q1 的各單位偏移方向移除這個控制分量：

$$
c=\frac{u_1-(u_1^\top u_r)u_r}{\|u_1-(u_1^\top u_r)u_r\|},\qquad
q_i=t_i-(t_i^\top c)c,\qquad
\widetilde t_i=\frac{q_i}{\|q_i\|}.
$$

`retained_energy` 為殘差長度平方，表示移除常數方向後保留的單位偏移能量。殘差長度不超過 `1e-5` 時不正規化、不定義剩餘 cosine，不將數值捨入誤差當新方向。已存 E2E 與 `u_1` 控制的同向性會驗證，E2E 的殘差為零並從剩餘方向矩陣排除；其他方法若退化也會排除，實際納入數記在 `n_methods`。小殘差即使能正規化，也不表示效應很大，必須搭配保留能量解讀。

共同軸診斷將單位偏移方向逐列排入矩陣，**不扣方法平均**，以各奇異值平方占總和的比例衡量集中程度，允許方向正負相反但共線。常數控制後只對非零殘差重新單位化，再做相同診斷，所以前後矩陣納入方法數不同，都是各方向等權，不是原始 reward 修正幅度的能量比較。結果先在每個模型、每個任務內計算，再彙整純量。三種 PCA 則各自扣除點平均，不能把 PCA 的變異解釋率和未中心化 SVD 的能量占比混用。

| `data/` 輸出 | 內容 |
|---|---|
| `latent_fb_pairs.csv` | 240 組方法比較，含完整 latent cosine／距離、偏移 cosine 與移除常數後的 cosine；未定義值留空 |
| `latent_fb_directions.csv` | 120 個方向，含來源列、與常數控制的 cosine、殘差長度／能量及是否可定義方向 |
| `latent_fb_spectrum.csv` | 216 列未中心化奇異值及能量占比，分常數控制前後 |
| `latent_fb_pca.csv` | 480 個二維投影座標，分完整單位 latent、單位偏移與單位殘差；子集僅投影於完整 latent PCA |
| `latent_fb_pca_variance.csv` | 72 組 PCA 的前兩成分變異解釋率及擬合點數 |
| `latent_fb_axis_summary.csv` | 第一奇異方向能量占比的逐任務與四任務等權彙整 |
| `latent_fb_retention_summary.csv` | 各方法保留能量的逐任務與四任務等權彙整 |

八張圖均引用於 `REPORT.md` 的 Q3 一節，位於 `figures/latent_fb_*.png`：完整 latent cosine 與歐氏距離、偏移及殘差 cosine、三種 PCA、奇異值能量分布。各圖面板按模型與任務分開，PCA 不跨面板擬合、不平均座標，也不對齊座標的語意。完整 latent 距離使用已存的半徑，CSV 另存單位化距離。完整 latent PCA 只用原始 reward 與五種修正擬合，五個子集不影響 PCA 基底，投影不是信賴區間。未包含構造的 `AX` 軸與隨機旋轉控制，不藉既有軸的定義預設共同方向成立。

## Q2 從既有 rollout 重建配對差與報告表格

仍只分析 standard FB，讀取相同三個 latent 檔及 `data/rollouts_fb_zero-shot-rl-{30,31,32}.csv`。不重訓、不重新推論、不新增 rollout。命令如下：

```bash
.venv/bin/python tasks/T05-opt-to-pessi/analyze_q2_fb.py
.venv/bin/python tasks/T05-opt-to-pessi/analyze_q2_fb.py --report-tables
.venv/bin/python -m unittest discover -s tasks/T05-opt-to-pessi -p 'test_q*_fb.py'
.venv/bin/python tasks/T05-opt-to-pessi/analyze_q1_fb.py --check
.venv/bin/python tasks/T05-opt-to-pessi/analyze_q2_fb.py --check
```

`--report-tables` 輸出報告的完整五欄主表、原始 reward 行為基準表、Q2 配對行為表與代表案例的逐 seed 標準誤表。測試會確認報告中四表與重算結果逐字一致，也保留 Q1 已填欄位的核對。報告採目標 10 度以延續 Q1 的主表，CSV 同時保存全部方法都有的 5 度、10 度結果，不依 return 選取強度。

| 檔案，均位於 `data/` | 內容 |
|---|---|
| `q2_fb_pairs.csv` | 2400 筆同模型、同任務、同 episode 的配對數據，保留修正後值、基準值與兩者差值 |
| `q2_fb_seed_summary.csv` | 120 組模型／任務／方法／目標角度結果，各 20 episodes 的平均、配對差標準差／標準誤及差值為正的 episode 數 |
| `q2_fb_task_summary.csv` | 40 組任務／方法／目標角度結果，彙整三個訓練 seeds 的平均與最小／最大值 |
| `q2_fb_baseline.csv` | 12 組模型／任務的原始 reward latent 行為基準 |
| `q2_fb_baseline_summary.csv` | 四個任務各自的基準跨 seed 平均與範圍 |

`delta_` 一律是修正 latent 減原始 reward latent。`*_mean_mean` 是先對每個模型的 20 episodes 取平均，再對三個訓練 seeds 等權平均。`*_mean_min/max` 是三個 seed 平均的範圍，不是 episode 範圍或信賴區間。不混合任務，也不把 episodes 當額外的訓練 seeds。`delta_*_se` 是配對差樣本標準差除以 20 的平方根，僅衡量固定模型內初始狀態的變動，不包含模型訓練變異。報告列出跑步 CQL 與翻轉 ReBRAC 兩個例子，其餘 SE 可從 seed summary 查詢，不進行顯著性檢定或多重比較校正。三個選定模型的訓練步數不同，不能把模型間差異都歸因於隨機 seed。`delta_*_positive_episodes` 只是差值為正的次數，不代表成功次數，對低高度時間或飽和比例而言，正值也不表示較好。

配對由 `rollout_eval.py` 的 `make_env(e)` 與確定性 actor 確保相同初始化種子，沒有儲存初始物理狀態供獨立核對。CSV 與 NPZ 的列索引、任務、family、係數、角度及 20 episodes 覆蓋率會完整驗證。CSV 只有六位有效數字，不能藉數值微小非零就宣稱差異具統計或實務意義。輸出保留十位有效數字，報告採 Python 預設浮點格式化取位。`--check` 是本機相同 NumPy 環境下的逐字核對，其他 BLAS／NumPy 環境若出現末位差異，須先確認是數值精度而非輸入或演算法變更。

指標定義見 `rollout_eval.py`：`return` 是 1000 steps 的原始任務 reward 總和；`speed` 是平均水平速度；`upright` 是平均軀幹朝上分量；`fall_frac` 是軀幹高度低於 0.6 的時間比例；`sat_frac` 是動作分量絕對值大於 0.99 的比例；`abs_action` 是動作絕對值平均；`knn_dist` 是每維以資料集均值／標準差縮放後，狀態與動作拼接向量到最近訓練樣本的平均歐氏距離。實驗預設每 10 steps 計算最近鄰距離，所有 episodes 固定跑滿 1000 steps。`m7` 與 `q_corr` 不納入 Q2，後者本來就是每個 latent 跨 probes 的相關係數，不是每個 episode 的獨立測量。

## Q4–Q6 從已存係數與候選 AX 掃描重建曲線

`analyze_sweeps_fb.py` 只讀上述三份 standard FB NPZ、歷史 rollout CSV 及 `data/sweep_fb_missing.csv`。不載入模型、不重新推論、不新增 rollout，不使用 one-step FB 或後述補跑的 trace。

```bash
.venv/bin/python tasks/T05-opt-to-pessi/analyze_sweeps_fb.py
.venv/bin/python tasks/T05-opt-to-pessi/analyze_sweeps_fb.py --report-tables
.venv/bin/python tasks/T05-opt-to-pessi/analyze_sweeps_fb.py --check
.venv/bin/python -m unittest discover -s tasks/T05-opt-to-pessi -p 'test_sweeps_fb.py'
```

第一個命令產生下表十一份 CSV 與十張圖。`--report-tables` 輸出 Q4 覆蓋率文字、方法表及 Q5／Q6 單調性表。`--check` 逐 byte 驗證 CSV／PNG，缺檔或過期即失敗且不寫檔。PNG 重現需要相同 NumPy、Matplotlib、字型與繪圖環境。測試另以原始 CSV 獨立核對全部 episode 配對與標準誤，以 SciPy 核對含 ties 的 Spearman，並檢查幾何、AX 構造、輸入損壞、來源覆蓋率及報告一致性。

| `data/` 輸出 | 列數 | 內容 |
|---|---:|---|
| `sweep_fb_coverage.csv` | 360 | 所有預期方法／模型／任務／目標角度的保存或缺失狀態 |
| `sweep_fb_points.csv` | 339 | 已存係數與夾角，另為每條方法曲線加入零係數基準 |
| `sweep_fb_adjacent.csv` | 279 | 相鄰係數差、完整 latent 夾角、偏移方向轉角及跳過的目標數 |
| `sweep_fb_curves.csv` | 60 | 各模型／任務／方法的離散曲線診斷 |
| `sweep_fb_method_summary.csv` | 5 | 方法覆蓋率與全範圍、共有 5→10 度區間的轉角彙整 |
| `axis_fb_pairs.csv` | 2160 | 九個角度各 20 episodes 的指標、零度基準與配對差 |
| `axis_fb_seed_summary.csv` | 108 | 各模型／任務／角度的平均與標準誤、配對差平均與標準誤 |
| `axis_fb_task_summary.csv` | 36 | 三個模型平均的等權彙整與最小／最大值，不合併任務 |
| `axis_fb_adjacent.csv` | 672 | 各模型／任務／指標的八段相鄰角度配對差與標準誤 |
| `axis_fb_trends.csv` | 84 | 逐模型／任務／指標的單調性、Spearman、總變化量與最大相鄰變化 |
| `axis_fb_trend_summary.csv` | 7 | 各指標在 12 條模型／任務曲線的診斷計數 |

原係數搜尋固定使用正號修正，沒有負號分支；本節結論不延伸至負係數。係數曲線依實際正係數排序，不把目標角度當係數。先單位化並移除原始 reward 方向，再比較相鄰點的偏移方向轉角；零係數的偏移方向未定義，轉角留空。完整 latent 的相鄰夾角另存，不與偏移方向轉角混用。對數係數圖省略零係數；不同方法的係數單位不同，不能跨方法比較數值大小。角度取樣不等距且覆蓋不同，方法表另列共有的 5→10 度區間，不能據稀疏點或連線宣稱全域平滑。

缺失狀態表 `sweep_fb_missing.csv` 有 81 列，是從原推論 stdout 匯出的去識別輸入，保留 NPZ 檔名、訓練 seed、任務、方法、目標角度、狀態、已收斂網格最大角度與來源行號。來源對應前述 `exp/T05-zero-shot-rl-*` tags，不包含完整日誌文字、主機或 checkpoint 路徑。一般重現只需此表，不需要本機日誌。必要時可用以下命令重新匯出，並重新產生分析輸出及執行 `--check`：

```bash
.venv/bin/python tasks/T05-opt-to-pessi/analyze_sweeps_fb.py --export-missing
```

匯出預設讀取 `artifacts/T05-opt-to-pessi/zero-shot-rl-{30,31,32}/output.log`，可用 `--log-dir` 指定另一個含三個 job 子目錄的位置。每筆缺失會與 NPZ 的缺少條件逐一核對，重複、額外或漏列均拒絕。`unbracketed_grid` 對應日誌的 `unreachable`，僅指已收斂網格沒有括住目標；`bisection_not_converged` 對應 `no fixed point`，僅指二分搜尋中的固定點迭代未收斂；`target_mismatch` 對應 `discontinuous`，指恢復後角度偏差超過容許值，本次沒有此狀態。這些都不是不存在解的證明，中間失敗係數未保存，無法精確重建失敗係數區間。

AX 使用各模型／任務目標 10 度的單位偏移方向構造軸，並逐列核對已存向量：

$$
a=\operatorname{unit}\!\left(t_{\mathrm{E2E\text{-}bi}}-\frac{t_{\mathrm{CQL}}+t_{\mathrm{ReBRAC}}}{2}\right),\qquad
z(\theta)=\sqrt{50}\left(\cos\theta\,u_r+\sin\theta\,a\right).
$$

軸再移除沿原始 reward 的數值殘差並正規化。旋轉角度為 −50、−35、−20、−10、0、10、20、35、50 度，零度取 `z_r`，其餘取 `AX`；NPZ 的 `target_angle` 保存帶號旋轉角，`angle` 是無號的相對原始 reward 夾角。`AX` 的 `strength=0` 不是這條掃描的控制參數。此軸不是已驗證的共同軸，也不是兩個方法端點的直接插值。

指標定義沿用 Q2。每條曲線先平均固定模型的 20 episodes，再逐角度比較；單調性使用 `1e-8` 數值容差，不是實務差異門檻，不作顯著性檢定。Spearman 使用平均排名處理 ties，常數曲線不定義相關。總變化量是相鄰平均值差的絕對值總和，方向反轉次數略過數值 ties。跨模型只彙整純量，不平均 latent 座標或合併任務 return。

十張圖均於報告引用：`sweep_fb_coefficient_angle.png`、`sweep_fb_adjacent_tangent.png`、`sweep_fb_coverage.png`，以及七個 `axis_fb_<metric>.png`。覆蓋圖 S／U／N／D 分別對應保存及上述三種缺失狀態。AX 圖逐模型、逐任務畫 episode 平均與標準誤，非信賴區間，也不含模型訓練變異；CSV 另保留與基準及相鄰角度的初始化種子配對差標準誤。連線僅作離散點的閱讀輔助，不能據九點掃描保證行為連續，或將飽和比例與最近樣本距離當成保守程度。

## 任務指標補跑使用固定 latent，與歷史評估分開

為直接觀察每個任務的行為，`rollout_task_behavior_fb.py` 使用同三個 standard FB 模型與已存 latent，對原始 reward 及五種修正 reward 的目標 10 度條件補跑。每個模型有四個任務、每任務六種條件，共 24 個 latent；每條件沿用初始化種子 0–19、確定性 actor、batch size 20，各 1000 steps，控制間隔 0.025 秒。模型與 latent 均不更新，不訓練真正的 CQL／ReBRAC，也不新增影片。

每個 step 保存原始任務 reward、水平速度、軀幹高度、直立程度、繞世界 +Y 的角動量；軀幹角度另包含 reset 狀態，共 1001 個時間點。同一次執行同時計算新 return，不能把新指標接到歷史 CSV 或既有影片的軌跡。評估使用各任務原有 `rewards/walker/*.py`，並非以 implicit reward 評分。

指標定義如下：

- 站立：平均軀幹高度（m）與直立程度（軀幹朝上分量）。
- 行走：估計水平淨位移／平均水平速度（m／m/s），與每步速度距 1 m/s 的平均絕對誤差。環境 reward 對速度使用下限，不懲罰所有超速，因此誤差只是描述，不作新的任務評分或成功條件。
- 跑步：估計水平淨位移／平均水平速度（m／m/s），搭配直立程度以區分姿態差異。
- 翻轉：正向／反向完整軀幹旋轉圈數、包含不足一圈的有方向淨旋轉圈數，以及平均角動量。正向為繞世界 +Y 旋轉，和 flip reward 的正角動量同向，但軀幹旋轉與整個子樹角動量不是同一物理量。

`net_displacement_estimate` 以每個 step 結束時的水平速度乘以控制間隔 0.025 秒，再對 1000 steps 加總，是有方向的淨位移估計，不是總路程或直接量測的位置差。所有 episodes 時間固定為 25 秒，因此距離是平均速度的 25 倍，不構成額外獨立證據。三份行為 CSV 均保存距離、相對原始 FB 的配對差及原有層級的彙整。新增距離只重算既有 trace，不重新執行模型。

圈數由 `atan2(xmat[torso,xz], xmat[torso,zz])` 取得軀幹角度，再沿時間解開角度跨越正負 π 的跳變。以 reset 角度為錨點，偏移達正／負一整圈才計一次，並將錨點移動同方向一圈；不足一圈的反向運動會抵消進度，不累加每步角度絕對值。完整轉一圈再完整反轉，正反方向各計一次，淨旋轉為零。這是身體旋轉圈數，不要求騰空、落地或恢復直立，地面滾動也可能被計入，不能稱為「成功 flip 次數」。驗證會拒絕解開後相鄰角度差超過 90 度的軌跡，以檢查取樣是否足夠。

執行命令如下。補跑拒絕覆寫 trace 或來源紀錄，如只缺某個模型，可用 `--seed 0`、`--seed 1` 或 `--seed 2` 分別執行。

```bash
PYTHONPATH=. MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python tasks/T05-opt-to-pessi/rollout_task_behavior_fb.py --workers 4
.venv/bin/python tasks/T05-opt-to-pessi/analyze_task_behavior_fb.py
.venv/bin/python tasks/T05-opt-to-pessi/analyze_task_behavior_fb.py --check
.venv/bin/python tasks/T05-opt-to-pessi/analyze_task_behavior_fb.py --report-tables
PYTHONPATH=. MUJOCO_GL=egl .venv/bin/python -m unittest discover \
  -s tasks/T05-opt-to-pessi -p 'test_*fb.py'
```

`data/behavior_fb_seed{0,1,2}.npz` 保留上述逐步評估資料、已用 latent、來源列、初始化種子與控制間隔。`artifacts/T05-opt-to-pessi/task-behavior/` 保存本機日誌、每個模型的 JSON 來源紀錄與程式碼快照，包含 checkpoint／latent／trace／相關程式碼雜湊與軟體版本，不提交 checkpoint 或日誌。

分析從逐步數據產生 `data/behavior_fb_episodes.csv`（1440 筆）、`behavior_fb_seed_summary.csv`（72 筆）與 `behavior_fb_task_summary.csv`（24 筆）。`delta_` 一律是相同模型、任務、初始化種子的條件值減原始 reward 基準，再先平均 20 episodes、後等權平均三個模型。Seed summary 保留配對差的標準誤，task summary 的最小／最大值為三個模型的平均值範圍，不是信賴區間。所有統計逐任務呈現，不混合任務 return，也不把 60 episodes 當成 60 個訓練 seeds。`--check` 從逐步資料重算並核對三份 CSV，`--report-tables` 輸出任務行為表，兩者均不執行模型。

## 初始化種子 0 的影片只作行為展示

本機影片位於 `artifacts/T05-opt-to-pessi/videos/fb_zero-shot-rl-{30,31,32}/`，既有影片的來源與評估命令見相應 `exp/T05-zero-shot-rl-*` tags。補錄腳本只對原始 reward 與五種修正 reward 的已存 latent 補齊缺少的影片，保留所有既有檔案，不補構造軸、隨機旋轉與子集控制。已補齊原先缺少的 ReBRAC＋常數 reward 組合，共 64 支，各訓練 seed 為 24、16、24 支。原始 reward 與五種 reward 修正的全部已存 latent 現有 291 支影片，逐 seed 為 100、92、99 支，覆蓋所有四個任務與已存強度；這個數字不含既有構造軸控制影片。覆蓋率、500 幀格式及 64 支補錄影片的 manifest 雜湊均已核對。

影片沿用初始化種子 0、固定 latent、確定性 actor、單一 observation batch，執行 1000 steps，每兩步錄一幀，輸出 500 幀、30 fps、320×240。這不是重新評估表格，也不重訓或重新推論 latent。本機同設定重播與歷史 CSV 的 return 不完全一致，因此不保證影片精確重現 CSV 的 episode 0，不能把影片當作對應數值軌跡。其餘 19 個初始狀態未錄影。

補錄影片與原影片使用相同目錄與檔名規則，新增來源由 `artifacts/T05-opt-to-pessi/videos/supplement_fb.jsonl` 區分，記錄 checkpoint／latent／影片的 sha256、來源列、版本與設定。補錄日誌為同目錄的 `supplement_fb.log`。這些產物均保留於 git-ignored 的 `artifacts/`，不提交影片或日誌。命令如下：

```bash
PYTHONPATH=. MUJOCO_GL=egl .venv/bin/python tasks/T05-opt-to-pessi/record_missing_fb.py --dry-run
PYTHONPATH=. MUJOCO_GL=egl .venv/bin/python tasks/T05-opt-to-pessi/record_missing_fb.py
PYTHONPATH=. MUJOCO_GL=egl .venv/bin/python tasks/T05-opt-to-pessi/record_missing_fb.py --check
PYTHONPATH=. MUJOCO_GL=egl .venv/bin/python -m unittest discover -s tasks/T05-opt-to-pessi -p 'test_*fb.py'
```

`--dry-run` 只列缺漏，`--check` 只驗證覆蓋率、影片解碼幀數與補錄來源，不執行模型。錄影時會核對 checkpoint 雜湊，使用 NPZ 中已存 latent，原子發布影片且拒絕覆寫既有檔。若中斷發生在影片發布後、manifest 追加前，`--check` 會指出缺少來源，不將該檔默認為已驗證補錄。

報告抽查訓練 seed 0 的 `run_z_r_nan_112.mp4`、`run_CQL_10_114.mp4`、`run_E2E_10_134.mp4`、`flip_z_r_nan_168.mp4`、`flip_ReBRAC_10_176.mp4`。檔名末尾數字對應 NPZ 列索引。可對每支影片執行以下命令重建五張抽樣影格，把檔名換成以上列表中的其他項目即可。影格只用來確認不同姿態，不推斷翻轉成功次數或完整步態分類。

```bash
mkdir -p /tmp/t05-q2-frames
ffmpeg -hide_banner -loglevel error -y \
  -i artifacts/T05-opt-to-pessi/videos/fb_zero-shot-rl-30/run_z_r_nan_112.mp4 \
  -vf 'fps=1/3' -frames:v 5 /tmp/t05-q2-frames/run_z_r_%02d.png
```

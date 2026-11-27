# T6.5-more-env：將 CEM、CMA-ES、LoLA 潛向量適應推廣至 Quadruped 與 Point Mass Maze

## 背景

T6.2 已證明在 Walker 的四種 OOD 動力學偏移環境下，於 800,000 環境步數預算內，CEM、CMA-ES 與 LoLA 能有效適應凍結的 FB 潛空間（包含自由潛向量 $z \in \mathbb{R}^{50}$ 與 ANFIS 搜尋空間），並確立了演算法與搜尋空間的相對排序。然而目前結論僅建立於單一雙足環境（Walker），尚不清楚這套潛向量適應方法與其相對排序能否跨環境泛化至高自由度四足機器人（`quadruped`）與低維連續導航迷宮（`point_mass_maze`）。本任務旨在檢驗 CEM、CMA-ES 與 LoLA 在此兩類環境 OOD 動力學偏移下的適應表現與趨勢一致性。

## 方法

**第一階段（FB 預訓練與 checkpoint 挑選）**

比照 T4.1.3.1 的 Walker 流程，在 ExORL `quadruped/rnd`（任務：`stand`、`jump`、`roll`、`roll_fast`）與 `point_mass_maze/rnd`（任務：`reach_top_left`、`reach_top_right`、`reach_bottom_left`、`reach_bottom_right`）上，各以 `agents/fb/config.yaml` 預設超參數（1M learning steps、100k transitions、$d_z = 50$、`batch_size=512`、`lr=1e-4`、`discount=0.98`，每 20k 步評估並存 checkpoint）訓練 5 個 seed（`0..4`）。完成後使用 `analysis.table6_aggregate` 選出 5-seed 全任務 IQM 最高的步數，並取該步中四任務 IQM 最高的 seed 作為下游凍結 FB backbone。

**第二階段（多環境 OOD 動力學適應）**

將 OOD 動力學偏移環境泛化至 `quadruped` 與 `point_mass_maze`，在第一階段選出的固定 FB backbone 下比較 CEM、CMA-ES 與 LoLA（具體 OOD 偏移參數與搜尋空間細節將於開始第二階段前補入此節並標記為後續補充）。

**預期結果（設計時定案）**

1. `quadruped` 與 `point_mass_maze` 的預訓練 FB 在 nominal 任務上能達到具代表性的 zero-shot 分數（與 ExORL RND-100k 基準相當），可作為有效的凍結 backbone。
2. 在 `quadruped` 與 `point_mass_maze` 的 OOD 動力學偏移下，CEM、CMA-ES、LoLA 皆能顯著高於凍結 FB $z_r$，且在 50 維 $z$ 空間中 CMA-ES 與 LoLA 表現優於 CEM（延續 T6.2 在 Walker 上的趨勢）。
3. $z$ 與 ANFIS 搜尋空間的相對差距會隨環境感測器維度與動力學複雜度而異（例如低維導航 `point_mass_maze` 相較高自由度 `quadruped`）。

**關鍵設定摘要（第一階段）**

| 項目 | Quadruped | Point Mass Maze |
|---|---|---|
| 模型 | FB（`agents/fb/config.yaml` 預設，$d_z=50$） | FB（`agents/fb/config.yaml` 預設，$d_z=50$） |
| 領域與資料集 | `quadruped/rnd`（100k transitions） | `point_mass_maze/rnd`（100k transitions） |
| 評估任務 | `stand`、`jump`、`roll`、`roll_fast` | `reach_top_left`、`reach_top_right`、`reach_bottom_left`、`reach_bottom_right` |
| 預訓練 Seeds | `0..4`（共 5 seeds） | `0..4`（共 5 seeds） |
| 訓練步數與評估頻率 | 1,000,000 learning steps，每 20,000 步評估並存檔 | 1,000,000 learning steps，每 20,000 步評估並存檔 |

## 結論

（待實驗完成後填寫）

## 結果

（待實驗完成後填寫）

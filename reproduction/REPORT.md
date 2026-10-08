# 官方 FB 重現報告：ExORL Walker RND-100k

結論：用作者原始程式碼訓練 5 個 seed，四個任務的 95% 信賴區間都與論文 FB（Table 6）重疊。stand 和 run 接近論文，walk 和 flip 較高，但跨 seed 變異很大。

- 程式碼：[`novis10813/zero-shot-rl`，分支 `lab-repro`](https://github.com/novis10813/zero-shot-rl/tree/lab-repro)（fork 自 [`enjeeneer/zero-shot-rl`](https://github.com/enjeeneer/zero-shot-rl)）
- 論文：Jeen et al., *Zero-Shot Reinforcement Learning from Low Quality Data*, NeurIPS 2024, [arXiv:2309.15178](https://arxiv.org/abs/2309.15178)

## 交付項目

| # | 項目 | 狀態 | 位置 |
|---|---|---|---|
| 1 | 可執行的官方 FB 程式碼 | 完成 | `lab-repro` 分支。FB 演算法與評估未改動，改動見下方「對原始碼的修改」 |
| 2 | 可重現的訓練與評估指令 | 完成 | 本文「指令」段落，完整步驟見 [`README.md`](README.md) |
| 3 | Checkpoint | 自行訓練（作者未釋出任何權重） | `idlab1:/home/sam/zsrl-runs/zsrl-<job>/`，每個 seed 一份 |
| 4 | 訓練曲線 | 完成 | 圖 2 |
| 5 | 評估分數 | 完成 | 表 1、圖 1 |
| 6 | 與我們 FB 實作（fb-offline）的比較 | 完成 | 表 3 |
| 7 | 重現流程 README | 完成 | [`README.md`](README.md)（英文） |

## 實驗設定

| 項目 | 設定 |
|---|---|
| 資料集 | ExORL `walker` / `rnd`（10,000 episodes、10M transitions），以固定 `rng(42)` 抽樣 100k transitions |
| 任務 | `stand`、`walk`、`run`、`flip`，零樣本評估，reward 由 physics 重新標註 |
| 架構 | F：(s, a) 與 (s, z) preprocessor（2 層 1024 → 512），接 2 層 1024。B：3 層 256。Actor：2 層 1024。ReLU |
| 超參數 | z 維度 50、batch 512、lr 1e-4、γ 0.98、Polyak 0.01、z mix 0.5、orthonormality 1、policy smoothing σ 0.2 / clip 0.3，與論文 Table 4 一致 |
| 訓練 | 1M gradient steps，seed 42、0、1、2、3 |
| 評估 | 每 20k 步一次。z = √d · normalize(E[r · B(s)])，用 10k 筆 buffer 樣本推得。每個任務 10 個 rollout × 1000 步，取 IQM |
| 彙整 | 與論文 Table 6 相同：選 5 個 seed 全任務 IQM 最高的那一步，回報該步各任務在 5 個 seed 上的 IQM，95% stratified bootstrap 信賴區間（`rliable`，10,000 次重抽） |
| 硬體 | idlab1 RTX 5090。單獨跑一個 job 約 2.2 小時（125 it/s），兩個 job 共用一張 GPU 時各約 3 到 5 小時。VRAM 峰值約 8.4 GB |

## 結果

**表 1：各任務 IQM（括號為 95% 信賴區間）**

| | stand | walk | run | flip | 4 任務平均 |
|---|---|---|---|---|---|
| 本次重現，選定步數（900k） | 601 (560–643) | 362 (169–639) | 120 (84–178) | 275 (135–396) | 340 |
| 本次重現，最後一步（1M） | 480 (212–674) | 366 (273–516) | 112 (62–171) | 281 (114–389) | 310 |
| 論文 FB（Table 6） | 558 (498–637) | 184 (123–278) | 101 (90–135) | 163 (90–212) | 252 |

![圖 1：各任務 IQM 與論文 FB 比較](fb_vs_paper.png)

圖 1：各任務在 5 個 seed 上的 IQM。誤差線為 95% 信賴區間，論文數據取自 Table 6。

![圖 2：5 個 seed 的訓練曲線](fb_walker_rnd_5seeds.png)

圖 2：每 20k 步的評估分數（每個 rollout 組取 10 次的 IQM）。灰線為各 seed，藍線為 5 個 seed 的平均，虛線為選定步數（900k）。

- 分數在不同評估時點與不同 seed 之間波動很大，walk 在 900k 的區間是 169 到 639。
- 選定步數是用評估任務本身挑的，所以第一列會偏高。最後一步那列沒有這種選擇偏差。
- 每個 seed 若各自挑最好的一步，4 任務平均落在 397 到 488，明顯高於 5 個 seed 的彙整結果，所以單一 seed 的結果不能直接拿來和論文比。

## Checkpoint

每個 run 只保留自己表現最好的那一步，不是 5 個 seed 共同選出的 900k。檔案是整個 pickle 過的 `FB` agent，每份 188 MB，同一個資料夾內也有完整的 `output.log`。

| Seed | Job | Checkpoint | 該 run 的 4 任務平均 |
|---|---|---|---|
| 42 | zsrl-5 | `idlab1:/home/sam/zsrl-runs/zsrl-5/740000.pickle` | 403 |
| 0 | zsrl-6 | `idlab1:/home/sam/zsrl-runs/zsrl-6/800000.pickle` | 488 |
| 1 | zsrl-7 | `idlab1:/home/sam/zsrl-runs/zsrl-7/680000.pickle` | 446 |
| 2 | zsrl-12 | `idlab1:/home/sam/zsrl-runs/zsrl-12/200000.pickle` | 397 |
| 3 | zsrl-13 | `idlab1:/home/sam/zsrl-runs/zsrl-13/560000.pickle` | 407 |

## 指令

在 repo 根目錄、`lab-repro` 分支執行。`lab run` 會執行已 push 的 commit，環境由 `uv sync` 建立。

```bash
# 1. 下載並整理資料集（2.6 GB，只需一次）
lab run --server idlab1 --vram 0 -- 'D=$HOME/zsrl-datasets/walker/rnd; \
  uv run python -u exorl_reformatter.py walker_rnd && mkdir -p $D && \
  mv datasets/walker/rnd/dataset.npz $D/'

# 2. 訓練一個 seed（每 20k 步把各任務 IQM 寫入 log，並保留最佳 checkpoint）
lab run --server idlab1 --vram 10G -- 'PYTHONUNBUFFERED=1 uv run python main_exorl.py \
  fb walker rnd --eval_tasks stand walk run flip --wandb_logging False \
  --dataset_path $HOME/zsrl-datasets/walker/rnd/dataset.npz --seed 0'

# 3. 評估任一 checkpoint（流程與訓練中的評估相同）
uv run python eval_exorl.py <checkpoint.pickle> walker \
  --dataset_path $HOME/zsrl-datasets/walker/rnd/dataset.npz \
  --eval_tasks stand walk run flip --eval_rollouts 10

# 4. 彙整多個 seed，產生表 1、圖 1、圖 2
cd reproduction && uv run python aggregate_seeds.py zsrl-5_eval.csv zsrl-6_eval.csv \
  zsrl-7_eval.csv zsrl-12_eval.csv zsrl-13_eval.csv \
  -o fb_walker_rnd_5seeds.png --comparison_output fb_vs_paper.png
```

`zsrl-<job>_eval.csv` 是從各 job 的 log 中擷取 `Step <i> eval:` 那幾行得到的。

## 對原始碼的修改

| 修改 | 原因 |
|---|---|
| 新增 `pyproject.toml`、`uv.lock`：Python 3.9、torch 2.7.1（CUDA 12.8）、`setuptools<70` | 原本的 torch 2.1.0 不支援 Blackwell GPU（sm_120）。wandb 0.15.2 需要 `pkg_resources` |
| `main_exorl.py --dataset_path` | 讓資料集可以放在 job worktree 之外，給多個 job 共用 |
| `agents/workspaces.py`：訓練結束後保留最佳 checkpoint | 原始碼在沒上傳 wandb 時會把它刪掉 |
| `agents/workspaces.py`：每次評估都把各任務 IQM 寫入 log | 不用 wandb 時，原始碼不會記錄各任務分數 |
| 新增 `eval_exorl.py`、`reproduction/*.py` | 評估 checkpoint、彙整多個 seed 並畫圖 |

## 與 fb-offline 的比較

兩邊的分數不能直接比較，因為模擬器、資料和 reward 尺度都不同。DMC 的回報落在 0 到 1000，Gym walker2d 約 6000。

**表 3：實作差異**

| | 官方 FB | fb-offline |
|---|---|---|
| Benchmark | DMC `walker`，ExORL RND 探索資料，100k | Gym `walker2d`，Minari `medium-v0`，1M |
| 任務 | 4 個重新標註的 reward | 1 個，環境 reward |
| Actor loss | `−min(Q1, Q2)`，沒有 BC | TD3+BC，α = 1.0（沒有 BC 會發散） |
| 網路 | B 為 3 層 256，全部 ReLU | B 為 2 層 256，第一層 LayerNorm → Tanh |
| 觀測值正規化 | 無 | 用資料集的 mean / std |
| Batch | 512 | 1024 |
| 共同設定 | d = 50、lr 1e-4、γ 0.98、Polyak 0.01、z mix 0.5、orthonormality 1、σ 0.2 / clip 0.3、1M 步 | 相同 |
| 回報方式 | 評估最佳步數的 IQM，5 seeds | 1M 最後一步，10 episodes 的 mean ± std，3 seeds |
| 結果 | 4 任務平均 310（1M） | 5990 ± 48（raw z_r） |

- 官方 FB 沒有 BC 項，在 RND-100k 上回報沒有崩潰。fb-offline 的 vanilla FB 在 walker2d-medium 上會發散。差別可能來自資料覆蓋度（探索資料相對於單一策略資料），但這一點沒有驗證。
- 兩者的訓練過程都不穩定。fb-offline 在 80k 到 560k 之間會跌倒，官方 FB 的評估分數也大幅跳動（圖 2）。
- 官方 FB 的 loss 只會記錄到 wandb，這次沒有開，所以無法確認它的 loss 是否保持有界。

## 限制

- 只做了 Walker / RND-100k，其他 domain 和資料集沒有跑。
- 使用的 seed 和論文不同，且 torch 版本從 2.1.0 換成 2.7.1。
- 作者在論文發表後修改過 FB 的網路大小（commit `8083688`），目前的值與論文 Table 4 一致。
- 5 個 seed 共同選出的 900k checkpoint 沒有保存，保存的是每個 seed 各自的最佳步數。

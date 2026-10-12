# T02 新的 offline FB / zero-shot RL baseline 調查

結論：BREEZE 是我們設定下最有希望的較強 baseline。它是從本 repo 同一份 codebase（Jeen et al. 2024）fork 出來的 PyTorch 實作，論文也有 Walker RND-100k 的結果。One-Step FB 在理論上比較乾淨，但在 Walker 上沒有贏過 FB，而且是 JAX 實作。所有 offline 方法都沒有公開 ExORL / DMC 的預訓練權重，要用都得自己訓練。

資料來源與查核日期見 [`README.md`](README.md)。表中的分數都引自各論文，不是本 repo 跑出來的。

## 比較表

| Method | Year | Offline | FB-based | Main modification | Official code | Pretrained weights | Useful for us |
|---|---|---|---|---|---|---|---|
| FB | 2021（NeurIPS）/ 2023（ICLR） | 是 | 是 | 原始方法：$M^{\pi_z}(s,a,\cdot) \approx F(s,a,z)^\top B(\cdot)\rho$，policy 最大化 $F^\top z$ | `facebookresearch/controllable_agent`（已 archived）。本 repo 也有實作 | 無 | 現有 baseline（T01） |
| VC-FB | 2024（NeurIPS） | 是 | 是 | 在 FB loss 加 CQL 式保守懲罰，壓低 OOD action 的 $Q_z = F^\top z$ | `enjeeneer/zero-shot-rl`（本 repo 的 upstream） | 無 | 已在本 repo，可以直接跑 |
| MC-FB | 2024（NeurIPS） | 是 | 是 | 同 VC-FB，但懲罰的對象是 successor measure $F^\top B$ | 同上 | 無 | 同上 |
| BREEZE | 2025（NeurIPS） | 是 | 是 | F1、F2、F3 三項修改，見下方 | `Whiterrrrr/BREEZE`（PyTorch，MIT） | 無 | 高 |
| One-Step FB | 2026（arXiv 2602.11399） | 是 | 是（FB 變體） | 只學行為策略 $\pi_\beta$ 的 successor measure，見下方 | `chongyi-zheng/onestep-fb`（JAX，MIT） | 無 | 中 |
| FB-AWARE | 2025（RLC） | 是 | 是 | policy 改用 advantage-weighted regression，B 改成 auto-regressive features | 未找到公開程式碼 | 無 | 中：AW 是對 offline FB 的小修改，可以自己加 |
| TD-JEPA | 2026（ICLR） | 是 | 否（SF + latent prediction） | 用 TD 學 latent predictor 和 state / task encoder，再接 successor features | `facebookresearch/td_jepa`（PyTorch，CC BY-NC 4.0） | 無 | 高：可當較強的 FB 實作來源，見 C3 |
| Soft FB | 2026（ICLR） | 是 | 是 | maximum-entropy 版 FB，可以處理非加法型 utility | 未確認 | 未確認 | 低：目標是 general utilities，不是標準 reward |
| FB-CPR（Meta Motivo） | 2025（ICLR） | 否（online + 無標籤動作資料） | 是 | 用 discriminator 把 policy 拉向示範動作分布 | `facebookresearch/metamotivo` | 有，但只有 humanoid | 低：環境不同 |

## BREEZE 修改了 FB 的哪些部分

- **F1 F 的 TD target。** FB 原本用 $F(s',\pi_z(s'),z)^\top z$ 當 target，BREEZE 換成 IQL 的 expectile value $V(s',z)$，而這個 V 只用資料內的 $(s,a)$ 訓練（$\tau = 0.99$）。這樣 F 的學習就不會碰到 OOD action。
- **F2 Policy。** Actor 換成 task-conditioned diffusion policy，用權重 $\exp(\alpha(F^\top z - V))$ 做 weighted regression（$\alpha = 0.05$）。推論時抽 K 個候選動作，再挑 $F^\top z$ 最大的。
- **F3 網路架構。** F 把 $(s,a)$ 和 $(s,z)$ 兩個 embedding 當成長度 2 的序列，送進 self-attention。B 改成 8-head transformer。

B 的 orthonormality loss 和 z 的推斷方式都沒有改。

論文 Table 2（Walker RND-100k，IQM，5 seeds）：FB 264 ± 33，VC-FB 350 ± 29，MC-FB 287 ± 48，BREEZE 525 ± 13。

成本（論文 Table 9，A6000）：FB 跑 1M steps 約 4 小時。BREEZE 跑 200k steps 要 4.0 到 7.8 小時，每一步慢 5 到 10 倍。論文說 400k steps 就能追上其他方法跑 1M steps 的分數。

## One-Step FB 和標準 FB 的差別

- **差別。** FB 的 $F(s,a,z)$ 依賴 z，TD target 裡的 $a'$ 由 $\pi_z$ 產生，所以 F 和 policy 互相依賴，fixed point 一直在移動。One-Step FB 的 $F_\beta(s,a)$ 不依賴 z，$a'$ 直接取自資料，target 固定，問題就變成 supervised regression。Policy $\pi(a|s,z)$ 最大化 $F_\beta^\top z$，再加上 BC 正則，等於對行為策略的 Q 做一步 policy improvement。
- **支援 offline 嗎。** 支援。它本來就是為 offline 設計的，資料需要 $(s,a,s',a')$。實驗用的是 ExORL RND 5M transitions，z 用另外 100k 筆推斷。論文沒有做 100k 的小資料實驗。
- **是不是更乾淨或更穩定的 backbone。** 理論上是。在 didactic CMP 上，它收斂到的誤差比 FB 小 $10^5$ 倍。但在 ExORL Walker 上，它是 379，FB 是 400（論文 Table 1）。它的優勢在 cheetah（+107）和 quadruped（+399）。另外，它的上限受行為資料品質限制，論文 Appendix E.2 的 explore 資料集上效能會下降。

## 其他發現

- **C1** 這些 offline 方法都沒有 ExORL / DMC 的預訓練權重。唯一有權重的是 Meta Motivo，但只有 humanoid。
- **C2** 各論文回報的 FB Walker 分數差很多，原因是資料量、評估流程和實作都不同：Jeen 2024 的 Table 6（100k，四任務平均約 252）、BREEZE（100k，264）、本 repo T01（100k，340）、One-Step FB（5M，400）、TD-JEPA（全資料，789 到 812）。跨論文的數字不能直接比，候選方法要在 T01 的流程下重跑才能比較。
- **C3** TD-JEPA 的 repo 以 Meta Motivo 為基礎，內含 FB、HILP、ICVF、Laplacian、BYOL-γ、RLDP，每個方法都有 flow-BC 正則版本。它的 FB 在 Walker 全資料上有 789 到 812，是目前看到最強的 FB 實作。不過它是 CC BY-NC 授權，而且沒有 100k 的結果。
- **C4** 在 One-Step FB 的論文裡，Walker 上分數最高的是 ICVF（619），不是任何 FB 變體。

## 排除的方法

下列方法和 offline FB 在 ExORL 上的設定關係不大，所以沒有放進表格：Robust Successor Features（只做 grid）、Gaussian Distributional SF（IJCAI 2026，目標是 risk-sensitive D4RL）、FB-M（Jeen 2025，針對 partial observability）、Epistemically-guided FB exploration（online 探索）、BFM-Zero（humanoid online）、Switching Successor Measures（hierarchical goal-conditioned，沒有細看）。

## 限制

- 只看論文和 repo，沒有實際跑任何候選方法。
- Soft FB 的程式碼和權重沒有查證。
- 搜尋用的是一般搜尋引擎加上 arXiv / GitHub API，可能漏掉沒被收錄的工作。

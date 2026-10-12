# T02: survey of newer offline FB / zero-shot RL baselines

Literature and code survey. No training or evaluation was run, so there is nothing to
regenerate. Every number in [`REPORT.md`](REPORT.md) is quoted from the paper named next to
it. Sources were checked on 2026-10-12.

## Papers

| Method | Paper |
|---|---|
| FB | Touati & Ollivier, NeurIPS 2021, [arXiv:2103.07945](https://arxiv.org/abs/2103.07945). Touati et al., ICLR 2023, [arXiv:2209.14935](https://arxiv.org/abs/2209.14935) |
| VC-FB, MC-FB | Jeen et al., NeurIPS 2024, [arXiv:2309.15178](https://arxiv.org/abs/2309.15178) |
| BREEZE | Zheng, Teyssier et al., NeurIPS 2025, [arXiv:2510.15382](https://arxiv.org/abs/2510.15382) (v2) |
| One-Step FB | Zheng et al., [arXiv:2602.11399](https://arxiv.org/abs/2602.11399) (v2, 2026-07-08) |
| FB-AWARE | Cetin, Touati & Ollivier, RLC 2025, [arXiv:2412.04368](https://arxiv.org/abs/2412.04368) |
| TD-JEPA | Bagatella et al., ICLR 2026, [arXiv:2510.00739](https://arxiv.org/abs/2510.00739) |
| Soft FB | Bagatella et al., ICLR 2026, [arXiv:2602.06769](https://arxiv.org/abs/2602.06769) |
| FB-CPR (Meta Motivo) | Tirinzoni et al., ICLR 2025 |

## Code

| Repo | Framework | License | Weights |
|---|---|---|---|
| [facebookresearch/controllable_agent](https://github.com/facebookresearch/controllable_agent) | PyTorch | MIT | none (archived) |
| [enjeeneer/zero-shot-rl](https://github.com/enjeeneer/zero-shot-rl) | PyTorch | MIT | none |
| [Whiterrrrr/BREEZE](https://github.com/Whiterrrrr/BREEZE) | PyTorch, fork of `enjeeneer/zero-shot-rl` | MIT | none (no releases, no checkpoint files in the tree) |
| [chongyi-zheng/onestep-fb](https://github.com/chongyi-zheng/onestep-fb) | JAX / Flax | MIT | none (no releases, no checkpoint files in the tree) |
| [facebookresearch/td_jepa](https://github.com/facebookresearch/td_jepa) | PyTorch, based on Meta Motivo | CC BY-NC 4.0 (per README) | none |
| [facebookresearch/metamotivo](https://github.com/facebookresearch/metamotivo) | PyTorch | custom (not detected by GitHub) | humanoid FB-CPR models on Hugging Face |

"Weights: none" means that the README, the GitHub releases, and the file tree (checked with the
GitHub API) contain no model files.

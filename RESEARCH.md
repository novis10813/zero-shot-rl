# Research question

> **How pessimistic or optimistic should an offline RL policy be, and does this choice really
> need to be fixed during training?**

Existing offline RL methods encode different inductive biases through their critic objectives:

| Method | Bias |
|---|---|
| CQL | strong pessimism |
| standard TD / ReBRAC | intermediate pessimism |
| E2E decision shaping | less pessimistic, more value-seeking |

Instead of treating these as fundamentally different algorithms, we investigate whether their
critic regularizers can be interpreted as inducing different **implicit shaped rewards**.
The main question then becomes:

> **Can a single Forward-Backward model represent this family of implicit rewards and recover
> policies with different levels of conservatism at test time, without retraining?**

## Plan

- **Step 1:** interpret critic regularization as implicit reward shaping.
- **Step 2:** train an offline Forward-Backward model.
- **RQ1:** is conservatism a recoverable FB latent dimension?
- **RQ2:** does FB stabilize the optimism/pessimism choice?

## Tasks

| Task | Serves |
|---|---|
| T01 FB reproduction on ExORL Walker RND-100k | Step 2: the base FB model |
| T04 MC-FB pessimism vs VC-FB and DVC-FB | pessimism fixed at training time inside FB |
| T05 implicit rewards as FB latents | Step 1 and RQ1 |

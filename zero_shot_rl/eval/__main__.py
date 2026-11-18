"""Unified evaluation entry point: python -m zero_shot_rl.eval <command> [args...]."""

from zero_shot_rl.cli import dispatch

COMMANDS = {
    "checkpoint": ("zero_shot_rl.eval.checkpoint", "Evaluate a saved FB / CFB / RL checkpoint on ExORL tasks"),
    "dynamics": ("zero_shot_rl.eval.dynamics_sweep", "Sweep an FB checkpoint across shifted dynamics YAMLs"),
    "value_correction": ("zero_shot_rl.eval.value_correction_checkpoints", "Evaluate saved Q_adapt checkpoints"),
    "action_transform": ("zero_shot_rl.eval.action_transform", "Evaluate saved ActionTransformer checkpoints"),
    "context_score": ("zero_shot_rl.eval.context_score", "Score test episodes with a trained context encoder"),
    "dynamics_score": ("zero_shot_rl.eval.dynamics_score", "Score paired transitions with a dynamics model"),
    "cvae_score": ("zero_shot_rl.eval.cvae_score", "Score paired transitions with a CVAE ensemble"),
    "adapt_episode": ("zero_shot_rl.eval.adapt_episode", "Closed-loop evaluation of context-based action adaptation"),
    "adapt_step": ("zero_shot_rl.eval.adapt_step", "One-step evaluation of context-based action adaptation"),
    "action_bound": ("zero_shot_rl.eval.action_bound", "Evaluate one-step action-bound ceiling under gear shift"),
    "latents": ("zero_shot_rl.eval.latents", "Roll out stored FB latents"),
    "switch": ("zero_shot_rl.eval.switch", "Evaluate Rule C optimistic/pessimistic switching"),
    "qtrace": ("zero_shot_rl.eval.qtrace", "Evaluate ensemble Q ranking and spread"),
    "paired_step": ("zero_shot_rl.eval.paired_step", "Collect paired one-step transitions across dynamics"),
    "paired_horizon": ("zero_shot_rl.eval.paired_horizon", "Collect K-step paired trajectories across dynamics"),
    "paired_score": ("zero_shot_rl.eval.paired_score", "Score paired transitions with FB shift detector"),
    "episode_returns": ("zero_shot_rl.eval.episode_returns", "Evaluate episode returns across checkpoints"),
    "fixed_z_returns": ("zero_shot_rl.eval.fixed_z_returns", "Evaluate fixed task vectors from online runs"),
    "latent_ranking": ("zero_shot_rl.eval.latent_ranking", "Evaluate predicted vs actual return of candidate latents"),
    "record_episode": ("zero_shot_rl.eval.record_episode", "Record deterministic episode videos"),
    "trace": ("zero_shot_rl.eval.trace", "Replay stored FB latents and write behavior traces"),
    "video": ("zero_shot_rl.eval.video", "Render rollout videos of stored latents"),
}

ALIASES = {
    "dynamics_sweep": "dynamics",
    "vc": "value_correction",
    "aa": "action_transform",
}

if __name__ == "__main__":
    dispatch("zero_shot_rl.eval", COMMANDS, ALIASES)

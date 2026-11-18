"""Unified training entry point: python -m zero_shot_rl.train <command> [args...]."""

from zero_shot_rl.cli import dispatch

COMMANDS = {
    "exorl": ("zero_shot_rl.train.main_exorl", "Train FB / CFB / offline RL baselines on ExORL"),
    "d4rl": ("zero_shot_rl.train.main_d4rl", "Train FB / CFB / offline RL baselines on D4RL"),
    "ensemble": ("zero_shot_rl.train.ensemble", "Train forward-network ensemble for a fixed FB checkpoint"),
    "g0": ("zero_shot_rl.train.g0", "Train offline backward-feature predictor g_0(s, a)"),
    "context_encoder": ("zero_shot_rl.train.context_encoder", "Train dynamics context encoder & B_0 predictor"),
    "dynamics_model": ("zero_shot_rl.train.dynamics_model", "Train action-conditioned dynamics shift detector"),
    "cvae_model": ("zero_shot_rl.train.cvae_model", "Train CVAE ensemble dynamics shift detector"),
}

ALIASES = {
    "fb": "exorl",
}

if __name__ == "__main__":
    dispatch("zero_shot_rl.train", COMMANDS, ALIASES)

"""Unified finetuning entry point: python -m zero_shot_rl.finetune <command> [args...]."""

from zero_shot_rl.cli import dispatch

COMMANDS = {
    "value_correction": ("zero_shot_rl.finetune.value_correction", "Online Q_adapt / residual soft-Q finetuning"),
    "action_transform": ("zero_shot_rl.finetune.action_transform", "Test-time FB-AA (g_1 + T_theta) finetuning"),
    "online_inference": ("zero_shot_rl.finetune.online_inference", "OpTI-BFM online task-vector adaptation"),
    "anfis_z": ("zero_shot_rl.finetune.anfis_z", "Latent adaptation (REINFORCE / LoLA / ReLA) with ANFIS or direct z"),
}

ALIASES = {
    "vc": "value_correction",
    "aa": "action_transform",
    "opti": "online_inference",
    "anfis": "anfis_z",
}

if __name__ == "__main__":
    dispatch("zero_shot_rl.finetune", COMMANDS, ALIASES)

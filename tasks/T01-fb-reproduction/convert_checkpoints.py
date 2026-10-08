"""
One-off migration of the T01 checkpoints from pickled FB agents (.pickle) to the
state_dict format of AbstractAgent.save (.pt). The constructor arguments are those
main_exorl.py passes for `fb walker rnd` with default arguments. Each conversion is
checked: the state_dicts must be identical and both agents must act identically.
"""

from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
import yaml

from agents.base import load_agent
from agents.fb.agent import FB

parser = ArgumentParser()
parser.add_argument("pickles", nargs="+", type=str)
args = parser.parse_args()

device = torch.device("cpu")
with open("agents/fb/config.yaml", "rb") as f:
    config = yaml.safe_load(f)
config.update({"z_dimension": 50, "discount": 0.98})  # main_exorl.py defaults

for path in map(Path, args.pickles):
    old = torch.load(path, map_location=device, weights_only=False)
    old._device = device  # pylint: disable=protected-access  # pickled on a GPU host
    new = FB(
        observation_length=old.observation_length,
        action_length=old.action_length,
        preprocessor_hidden_dimension=config["preprocessor_hidden_dimension"],
        preprocessor_output_dimension=config["preprocessor_output_dimension"],
        preprocessor_hidden_layers=config["preprocessor_hidden_layers"],
        forward_hidden_dimension=config["forward_hidden_dimension"],
        forward_hidden_layers=config["forward_hidden_layers"],
        forward_number_of_features=config["forward_number_of_features"],
        backward_hidden_dimension=config["backward_hidden_dimension"],
        backward_hidden_layers=config["backward_hidden_layers"],
        actor_hidden_dimension=config["actor_hidden_dimension"],
        actor_hidden_layers=config["actor_hidden_layers"],
        preprocessor_activation=config["preprocessor_activation"],
        forward_activation=config["forward_activation"],
        backward_activation=config["backward_activation"],
        actor_activation=config["actor_activation"],
        z_dimension=config["z_dimension"],
        critic_learning_rate=config["critic_learning_rate"],
        actor_learning_rate=config["actor_learning_rate"],
        learning_rate_coefficient=config["learning_rate_coefficient"],
        orthonormalisation_coefficient=config["orthonormalisation_coefficient"],
        discount=config["discount"],
        batch_size=config["batch_size"],
        z_mix_ratio=config["z_mix_ratio"],
        gaussian_actor=config["gaussian_actor"],
        std_dev_clip=config["std_dev_clip"],
        std_dev_schedule=config["std_dev_schedule"],
        tau=config["tau"],
        device=device,
        name=config["name"],
    )
    new.load_state_dict(old.state_dict(), strict=True)
    new._name = path.stem  # pylint: disable=protected-access
    saved = load_agent(new.save(path.parent), device)

    old_state, saved_state = old.state_dict(), saved.state_dict()
    assert old_state.keys() == saved_state.keys()
    assert all(torch.equal(old_state[k], saved_state[k]) for k in old_state)

    rng = np.random.default_rng(0)
    old.eval()
    saved.eval()
    for _ in range(100):
        observation = rng.standard_normal(old.observation_length).astype(np.float32)
        z = rng.standard_normal(50).astype(np.float32)
        z = np.sqrt(50) * z / np.linalg.norm(z)
        old_action, _ = old.act(observation, task=z, step=None, sample=False)
        new_action, _ = saved.act(observation, task=z, step=None, sample=False)
        assert np.array_equal(old_action, new_action)

    print(f"{path} -> {path.with_suffix('.pt')}: {len(old_state)} tensors identical, actions identical")

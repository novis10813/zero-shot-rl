"""
module for retrieving reach_bottom_left reward function
from dm_control suite jaco
"""

from zero_shot_rl import dmc

env = dmc.make(name="jaco_reach_bottom_left")
reward_function = env._task.get_reward  # pylint: disable=protected-access

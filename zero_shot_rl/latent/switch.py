"""Rule C: per-state choice between the optimistic and pessimistic actions of a fixed FB.

Neggatu et al. (arXiv 2503.12222) Algorithm 1, with the behaviour-cloning action
replaced by the actor at a pessimistic latent. At a state s, with z_r and the CQL latent
z_cql of a CQL angle sweep NPZ (tasks/T06-latent-switching/cql_sweep.py),

    a_opt  = pi(s, z_r),  a_pess = pi(s, z_cql)
    Q(a)   = mean_k F_k(s, a, z_r)^T z_r,  u(a) = std_k F_k(s, a, z_r)^T z_r
    take a_pess if Q(a_opt) - kappa u(a_opt) < Q(a_pess), else a_opt

with F_k the members of a zero_shot_rl.train.ensemble ensemble trained for the same
checkpoint. Library only: zero_shot_rl.eval.switch runs the rule in the simulator and
zero_shot_rl.latent.deviation applies it to dataset states.
"""

import numpy as np

KAPPAS = [0.0, 0.001, 0.003, 0.01, 0.03, 0.1]


def choose_pess(q_opt, u_opt, q_pess, condition):
    """Boolean mask of the states that take a_pess under a condition: "opt", "pess" or
    a kappa."""
    if condition == "opt":
        return np.zeros(len(q_opt), dtype=bool)
    if condition == "pess":
        return np.ones(len(q_opt), dtype=bool)
    return q_opt - condition * u_opt < q_pess


def latents(zs, task, target_angle):
    """Unit z_r and z_cql of a CQL angle sweep NPZ."""
    in_task = zs["task"] == task
    (r,) = np.flatnonzero(in_task & (zs["family"] == "z_r"))
    (c,) = np.flatnonzero(in_task & (zs["family"] == "CQL") & (zs["target_angle"] == target_angle))
    return zs["z"][r], zs["z"][c]


def values(ensemble, obs, actions, z):
    """Mean and std over members of F_k(s, a, z)^T z."""
    q = ensemble(obs, actions, z) @ z[0]  # [K, N]
    return q.mean(0).numpy(), q.std(0).numpy()

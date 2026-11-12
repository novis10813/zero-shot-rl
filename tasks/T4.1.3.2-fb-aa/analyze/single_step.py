"""Single-step detection of dynamics shifts: the dynamics-model error of T4.1.2.1 against the classifier.

Reads data/inputs/classifier.npz (analyze.classifier, job zero-shot-rl-291) and the scores
tasks/T4.1.2.1-action-conditioned-detection/data/inputs/dynamics_scored_delta.npz (|e|_2 of the
model regressing Delta s), both on the same paired one-step data (60 episodes x 999 steps, the same
(s, a) replayed in the stock and in each shifted simulator). A transition is "active" when the shift
changes its next state, |s'_config - s'_stock|_2 > 1e-6. Writes data/single_step.csv, one row per
config:

    active          fraction of active transitions over all 60 episodes
    top10           share of sum |s'_config - s'_stock|^2 held by the 10% largest transitions (60 episodes)
    gap_ratio       median of |s'_config - s'_stock|_2 / |e_stock|_2 over the active transitions (60 episodes),
                    e_stock the error of the dynamics model on the stock transition
    old_auc         AUROC of |e|_2, stock vs config, on the 15 test episodes of the classifier
    old_auc_active  the same on the active test transitions
    clf_auc         AUROC of the classifier logit on the test episodes
    clf_auc_active  the same on the active test transitions
    clf_acc, clf_tpr, clf_fpr                      logit > 0 as shifted: (TPR + 1 - FPR) / 2, TPR, FPR
    clf_acc_active, clf_tpr_active, clf_fpr_active  the same on the active test transitions

Usage (repo root):
    PYTHONPATH=.:tasks/T4.1.3.2-fb-aa uv run python -m analyze.single_step [--check]
"""

import sys
from pathlib import Path

import numpy as np

from analysis.detection import auroc

TASK = Path(__file__).resolve().parents[1]
OLD = TASK.parent / "T4.1.2.1-action-conditioned-detection/data/inputs/dynamics_scored_delta.npz"
CONFIGS = [
    "friction_mild", "friction_strong", "actuator_mild", "actuator_strong",
    "friction_high_mild", "friction_high_strong", "actuator_high_mild", "actuator_high_strong",
]
ACTIVE = 1e-6
FIELDS = ["config", "active", "top10", "gap_ratio", "old_auc", "old_auc_active", "clf_auc", "clf_auc_active",
          "clf_acc", "clf_tpr", "clf_fpr", "clf_acc_active", "clf_tpr_active", "clf_fpr_active"]


def top_share(gap, fraction=0.1):
    """Share of sum gap^2 held by the largest `fraction` of the transitions."""
    sq = np.sort(gap.ravel() ** 2)[::-1]
    return sq[:int(len(sq) * fraction)].sum() / sq.sum()


def rates(base, shift):
    """Balanced accuracy, TPR and FPR of `logit > 0` as shifted."""
    tpr, fpr = (shift > 0).mean(), (base > 0).mean()
    return (tpr + 1 - fpr) / 2, tpr, fpr


def rows():
    clf, old = np.load(TASK / "data/inputs/classifier.npz"), np.load(OLD)
    test = clf["test_episodes"]
    out = []
    for name in CONFIGS:
        gap = clf["gap_" + name]
        if not np.allclose(gap, old["gap_" + name][test], atol=1e-9):
            raise ValueError(f"{name}: the test transitions of the classifier and of T4.1.2.1 differ")
        act = gap > ACTIVE
        lb, ls = clf["logit_base_" + name], clf["logit_shift_" + name]
        ob, os_ = old["norm_baseline"][test], old["norm_" + name][test]
        r = dict(config=name, active=(old["gap_" + name] > ACTIVE).mean(), top10=top_share(old["gap_" + name]),
                 gap_ratio=np.median((old["gap_" + name] / old["norm_baseline"])[old["gap_" + name] > ACTIVE]),
                 old_auc=auroc(ob.ravel(), os_.ravel()), old_auc_active=auroc(ob[act], os_[act]),
                 clf_auc=auroc(lb.ravel(), ls.ravel()), clf_auc_active=auroc(lb[act], ls[act]))
        r["clf_acc"], r["clf_tpr"], r["clf_fpr"] = rates(lb.ravel(), ls.ravel())
        r["clf_acc_active"], r["clf_tpr_active"], r["clf_fpr_active"] = rates(lb[act], ls[act])
        out.append(r)
    return out


def render(records):
    lines = [",".join(FIELDS)]
    for r in records:
        lines.append(",".join(r[k] if k == "config" else f"{r[k]:.6g}" for k in FIELDS))
    return "\n".join(lines) + "\n"


def main():
    text = render(rows())
    path = TASK / "data/single_step.csv"
    if "--check" in sys.argv:
        if path.read_text() != text:
            sys.exit(f"{path} differs from the regenerated table")
    else:
        path.write_text(text)
    print("ok" if "--check" in sys.argv else "written")


if __name__ == "__main__":
    main()

"""Threshold-free and threshold-based detection metrics shared by the detection tasks (T4.1.2.1, T4.1.2.2)."""

import numpy as np


def auroc(negative, positive):
    """P(positive > negative) with ties counted half."""
    neg = np.sort(negative)
    below = np.searchsorted(neg, positive, side="left")
    above = np.searchsorted(neg, positive, side="right")
    return ((below + above) / 2).mean() / len(neg)


def best_youden(negative, positive):
    """Threshold maximising TPR - FPR of `score > threshold`, with its TPR and FPR."""
    neg, pos = np.sort(negative), np.sort(positive)
    thr = np.unique(np.concatenate([neg, pos]))
    tpr = 1 - np.searchsorted(pos, thr, side="right") / len(pos)
    fpr = 1 - np.searchsorted(neg, thr, side="right") / len(neg)
    i = (tpr - fpr).argmax()
    return thr[i], tpr[i], fpr[i]

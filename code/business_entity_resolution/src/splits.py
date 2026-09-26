"""Disjoint partitions of the training Source-1 entities, shared by every stage.

A  first-stage LightGBM (also used to trim candidates for later stages)
V  validation (threshold / decision tuning, reported scores)
C  neural cross-encoder training (on Kaggle GPU)
D  final LightGBM, which uses the cross-encoder score as a feature
Every model is evaluated or used as a feature only on entities it was not trained on.
"""

import numpy as np

SIZES = {"A": 200_000, "V": 50_000, "C": 250_000, "D": 300_000}
ORDER = ("A", "V", "C", "D")


def partition(truth_ids, seed=0):
    ids = np.array(sorted(truth_ids))
    perm = np.random.default_rng(seed).permutation(len(ids))
    out, start = {}, 0
    for name in ORDER:
        out[name] = set(ids[perm[start:start + SIZES[name]]])
        start += SIZES[name]
    return out

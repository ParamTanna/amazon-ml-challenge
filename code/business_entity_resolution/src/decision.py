"""Stage 5: turn match probabilities into one prediction set per Source-1 entity.

The leaderboard metric is F0.5 averaged per Source-1 entity, so the best set for an
entity depends on the whole list of its candidates, not on each pair in isolation.
With candidate probabilities sorted p1 >= p2 >= ... and T true matches, predicting the
top j gives F0.5 = 1.25*TP / (j + 0.25*T).  We pick j maximising the (ratio-approximated)
expectation 1.25*sum(p1..pj) / (j + 0.25*(sum(p) + m0)), and compare it with predicting
nothing, which scores 1 only when the entity has no match at all:
    E[F | empty] = prod(1 - p_i) * q0.
m0 (expected true matches missing from the candidate list) and q0 (probability that no
match is missing) are estimated on the validation split; ``scale`` recalibrates p.
"""

import numpy as np
import pandas as pd


def expected_f05_decide(pairs, prob_col="p", m0=0.0, q0=1.0, scale=1.0, empty_bias=1.0):
    """Return {s1_id: set(cand_id)} maximising approximate expected F0.5 per entity."""
    df = pairs[["s1_id", "cand_id", prob_col]].copy()
    df["pp"] = np.clip(df[prob_col].to_numpy(np.float64) * scale, 1e-6, 1 - 1e-6)
    df = df.sort_values(["s1_id", "pp"], ascending=[True, False], kind="mergesort")
    key = df["s1_id"].astype(str).to_numpy() if not isinstance(
        df["s1_id"].dtype, pd.CategoricalDtype) else df["s1_id"].cat.codes.to_numpy()
    starts = np.r_[0, np.flatnonzero(key[1:] != key[:-1]) + 1]
    sizes = np.diff(np.r_[starts, len(key)])
    gid = np.repeat(np.arange(len(starts)), sizes)
    pos = np.arange(len(key)) - np.repeat(starts, sizes) + 1           # j = 1..n
    pp = df["pp"].to_numpy()
    csum = np.cumsum(pp)
    csum_group = csum - np.repeat(np.r_[0.0, csum[starts[1:] - 1]], sizes)
    total = np.add.reduceat(pp, starts)                                   # sum p per group
    e_j = 1.25 * csum_group / (pos + 0.25 * (total[gid] + m0))
    best_j_val = np.maximum.reduceat(e_j, starts)
    log_empty = np.add.reduceat(np.log1p(-pp), starts)
    e_empty = np.exp(log_empty) * q0 * empty_bias
    # best j per group = first position reaching the max
    is_best = e_j >= best_j_val[gid] - 1e-12
    first_best = np.full(len(starts), np.iinfo(np.int64).max)
    np.minimum.at(first_best, gid[is_best], pos[is_best])
    keep_group = best_j_val > e_empty
    sel = keep_group[gid] & (pos <= first_best[gid])
    out = df.loc[sel, ["s1_id", "cand_id"]]
    return out.groupby("s1_id", observed=True)["cand_id"].agg(lambda s: set(map(str, s))).to_dict()


def tune_expected(va, truth_va, prob_col="p", scorer=None):
    """Grid-search scale / empty_bias for the expected-F0.5 rule on validation."""
    from evaluate import f05_macro
    scorer = scorer or f05_macro
    # expected matches the candidate list misses, and chance that none are missed
    cand = va.groupby("s1_id", observed=True)["cand_id"].agg(lambda s: set(map(str, s)))
    missing = [len(t - cand.get(k, set())) for k, t in truth_va.items()]
    m0 = float(np.mean(missing))
    q0 = float(np.mean([m == 0 for m in missing]))
    best = (None, -1.0)
    for scale in (0.8, 0.9, 1.0, 1.1, 1.2):
        for eb in (1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 10.0):
            pred = expected_f05_decide(va, prob_col, m0=m0, q0=q0, scale=scale, empty_bias=eb)
            s = scorer(pred, truth_va)
            if s > best[1]:
                best = ({"m0": m0, "q0": q0, "scale": scale, "empty_bias": eb}, s)
    return best

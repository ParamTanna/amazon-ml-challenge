"""Local scorer: macro F0.5 per Source-1 entity, exactly as the challenge defines it."""

import pandas as pd


def parse_id_lists(series):
    """Map a Series of comma-separated ID strings to a list of sets."""
    return [set(s.split(",")) if isinstance(s, str) and s else set() for s in series]


def load_truth(path_or_df):
    df = pd.read_parquet(path_or_df) if isinstance(path_or_df, str) else path_or_df
    return dict(zip(df["source1_entity_id"], parse_id_lists(df["matched_entity_ids"])))


def f05_entity(pred, truth):
    """F0.5 for one Source-1 entity given predicted and true ID sets."""
    if not truth:
        return 1.0 if not pred else 0.0
    if not pred:
        return 0.0
    tp = len(pred & truth)
    if tp == 0:
        return 0.0
    p, r = tp / len(pred), tp / len(truth)
    return 1.25 * p * r / (0.25 * p + r)


def f05_macro(pred, truth):
    """Average F0.5 over every Source-1 entity in ``truth`` (missing predictions = empty)."""
    if not truth:
        return 0.0
    return sum(f05_entity(pred.get(k, set()), t) for k, t in truth.items()) / len(truth)


def breakdown(pred, truth, groups):
    """Macro F0.5 per group label (e.g. country or singleton / non-singleton)."""
    agg = {}
    for k, t in truth.items():
        g = groups.get(k, "?")
        s, n = agg.get(g, (0.0, 0))
        agg[g] = (s + f05_entity(pred.get(k, set()), t), n + 1)
    return {g: (s / n, n) for g, (s, n) in agg.items()}

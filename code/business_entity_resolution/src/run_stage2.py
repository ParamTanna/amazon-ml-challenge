"""Stage 5: final LightGBM on the trimmed candidate set, with neural, context, consensus
and competition features.

Inputs (all produced earlier, nothing is recomputed from text):
  runs/<stage1>/feats/{train,test}_*.parquet   first-stage features (partitions D, V; test)
  runs/<stage1>/stage1_{train,test}_*.parquet  first-stage probability p
  pairs_trim/{split}_*.parquet                 ranker probability r0 for every entity
  ce_scores.parquet (optional)                 cross-encoder probability (Kaggle GPU)
Candidates are trimmed to the first-stage top ``max_per_s1`` with p >= ``min_p``; this set
is exactly what the final model scores (-> candidate_pairs.tsv).

Extra features on top of the first-stage ones
  * p, ce and, per entity: rank, gap to the best candidate, sum, number of strong
    candidates, best score from the other source
  * probability-weighted consensus: score mass on the entity's other candidates that share
    this candidate's house number / name, versus the entity's own house number
  * competition: the best ranker score any *other* entity has for the same candidate, how
    many other entities want it, and the gap (every Source-2/3 record belongs to at most
    one entity)
Trained on partition D, tuned on V, streamed over the test split.
"""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd

COMP_FEATURES = ["c_other_r0_max", "c_other_r0_n50", "c_r0_gap"]


class Competition:
    """Best ranker score any *other* entity has for each candidate, per country.

    Stored as sorted integer keys (entity code * n_candidates + candidate code) so that
    lookups never materialise millions of string ids.
    """

    def __init__(self, split):
        import pyarrow.parquet as pq
        from model import split_countries
        from rank_candidates import trim_path
        from run_blocking import safe_country
        self.tables = {}
        for ctry in split_countries(split):
            t = pq.read_table(trim_path(split, ctry), columns=["s1_id", "cand_id", "r0"]).to_pandas()
            r0 = t["r0"].to_numpy(np.float32)
            s1c = t["s1_id"].cat.codes.to_numpy().astype(np.int64)
            ck = t["cand_id"].cat.codes.to_numpy().astype(np.int64)
            order = np.lexsort((-r0, ck))
            ck_s, r_s = ck[order], r0[order]
            first = np.r_[True, ck_s[1:] != ck_s[:-1]]
            start = np.maximum.accumulate(np.where(first, np.arange(len(ck_s)), 0))
            rank = np.arange(len(ck_s)) - start
            nxt = np.minimum(start + 1, len(ck_s) - 1)
            has2 = (nxt != start) & (ck_s[nxt] == ck_s)
            other = np.empty(len(r0), np.float32)
            other[order] = np.where(rank == 0, np.where(has2, r_s[nxt], -1.0), r_s[start])
            strong = (r0 > 0.5).astype(np.int32)
            n50 = (pd.Series(strong).groupby(ck).transform("sum").to_numpy() - strong)
            n_c = len(t["cand_id"].cat.categories)
            key = s1c * n_c + ck
            o = np.argsort(key)
            self.tables[safe_country(ctry)] = (
                t["s1_id"].cat.categories, t["cand_id"].cat.categories, n_c, key[o],
                other[o], n50[o].astype(np.float32), (r0 - other)[o].astype(np.float32))
            del t

    def add(self, df, ctry_safe):
        s1_cats, c_cats, n_c, keys, other, n50, gap = self.tables[ctry_safe]
        s1c = s1_cats.get_indexer(df["s1_id"])
        cc = c_cats.get_indexer(df["cand_id"])
        k = s1c.astype(np.int64) * n_c + cc
        pos = np.clip(np.searchsorted(keys, k), 0, len(keys) - 1)
        found = (keys[pos] == k) & (s1c >= 0) & (cc >= 0)
        df["c_other_r0_max"] = np.where(found, other[pos], -1).astype(np.float32)
        df["c_other_r0_n50"] = np.where(found, n50[pos], -1).astype(np.float32)
        df["c_r0_gap"] = np.where(found, gap[pos], -1).astype(np.float32)


def context_features(df, cols):
    """Per-entity context of score columns: rank, gap to max, sum, strong count, and the
    best score among the entity's candidates from the other source (S2 vs S3)."""
    names = []
    is_s2 = df["cand_id"].str.startswith("S2-").to_numpy()
    g_ids = df["s1_id"].to_numpy()
    for c in cols:
        v = df[c]
        g = v.groupby(g_ids)
        df[f"{c}_rank"] = g.rank(ascending=False, method="first").astype(np.float32).to_numpy()
        df[f"{c}_gap"] = (v - g.transform("max")).astype(np.float32).to_numpy()
        df[f"{c}_sum"] = g.transform("sum").astype(np.float32).to_numpy()
        df[f"{c}_n50"] = (v > 0.5).groupby(g_ids).transform("sum").astype(np.float32).to_numpy()
        best2 = v.where(is_s2).groupby(g_ids).transform("max")
        best3 = v.where(~is_s2).groupby(g_ids).transform("max")
        df[f"{c}_other_best"] = pd.Series(np.where(is_s2, best3, best2)).fillna(-1).to_numpy(
            np.float32)
        names += [f"{c}_rank", f"{c}_gap", f"{c}_sum", f"{c}_n50", f"{c}_other_best"]
    return names


def weighted_consensus(df, cols):
    """Probability-weighted agreement among an entity's candidates (see module doc)."""
    names = []
    g_ids = df["s1_id"].to_numpy()
    cn, sn, ck = (df[c].to_numpy() for c in ("k_c_num", "k_s1_num", "k_c_name"))
    has = cn >= 0
    eq_s1 = (cn == sn) & has
    for c in cols:
        v = df[c].to_numpy(np.float32)
        same = pd.Series(v).groupby([g_ids, cn]).transform("sum").to_numpy() - v
        df[f"{c}_same_num"] = np.where(has, same, -1).astype(np.float32)
        w = np.where(eq_s1, v, 0.0)
        df[f"{c}_s1_num"] = (pd.Series(w).groupby(g_ids).transform("sum").to_numpy()
                             - w).astype(np.float32)
        df[f"{c}_same_name"] = (pd.Series(v).groupby([g_ids, ck]).transform("sum").to_numpy()
                                - v).astype(np.float32)
        names += [f"{c}_same_num", f"{c}_s1_num", f"{c}_same_name"]
    return names


def build(feat_file, s1_booster, feats1, comp, ce, min_p, max_per_s1):
    """One saved first-stage feature file -> trimmed pairs with all final-stage features.

    Files hold complete entity groups, so trimming and per-entity context are exact.
    """
    df = pd.read_parquet(feat_file)
    df["p"] = s1_booster.predict(df[feats1].to_numpy(np.float32)).astype(np.float32)
    df = df[df["p"] >= min_p].sort_values(["s1_id", "p"], ascending=[True, False])
    df = df[df.groupby("s1_id").cumcount() < max_per_s1].reset_index(drop=True)
    if not len(df):
        return None, None
    ctry_safe = feat_file.stem.split("_", 1)[1].rsplit("_", 1)[0]
    comp.add(df, ctry_safe)
    score_cols = ["p"]
    if ce is not None:
        df = df.merge(ce, on=["s1_id", "cand_id"], how="left")
        df["ce"] = df["ce"].fillna(-1).astype(np.float32)
        score_cols.append("ce")
    extra = context_features(df, score_cols) + weighted_consensus(df, score_cols)
    return df, score_cols + COMP_FEATURES + extra


def main(stage1, run, ce_path, min_p, max_per_s1, seed):
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    import lightgbm as lgb
    from config import WORK_DIR
    from decision import expected_f05_decide, tune_expected
    from evaluate import breakdown
    from model import FEATURES, LGB_PARAMS, add_labels, load_truth
    from splits import partition

    s1_dir = WORK_DIR / "runs" / stage1
    run_dir = WORK_DIR / "runs" / run
    run_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    truth = load_truth()
    parts = partition(truth.keys(), seed)
    ce = pd.read_parquet(ce_path, columns=["s1_id", "cand_id", "ce"]) if ce_path else None

    s1_booster = lgb.Booster(model_file=str(s1_dir / "model.txt"))
    # ---------- partitions D (train) and V (validation) from the training split
    comp = Competition("train")
    frames = {"D": [], "V": []}
    extra = None
    for f in sorted((s1_dir / "feats").glob("train_*.parquet")):
        df, extra_f = build(f, s1_booster, FEATURES, comp, ce, min_p, max_per_s1)
        if df is None:
            continue
        extra = extra_f
        is_d = df["s1_id"].isin(parts["D"]).to_numpy()
        frames["D"].append(df[is_d])
        frames["V"].append(df[~is_d])
    del comp
    d = add_labels(pd.concat(frames["D"], ignore_index=True), truth)
    v = add_labels(pd.concat(frames["V"], ignore_index=True), truth)
    del frames
    feats = FEATURES + extra
    print(f"D {len(d):,} pairs  V {len(v):,}  features {len(feats)}  [{time.time() - t0:.0f}s]",
          flush=True)
    params = dict(LGB_PARAMS, learning_rate=0.05)
    dtr = lgb.Dataset(d[feats].to_numpy(np.float32), d["label"].to_numpy(), feature_name=feats)
    dva = lgb.Dataset(v[feats].to_numpy(np.float32), v["label"].to_numpy(), reference=dtr)
    booster = lgb.train(params, dtr, 4000, valid_sets=[dva],
                        callbacks=[lgb.early_stopping(150), lgb.log_evaluation(250)])
    booster.save_model(str(run_dir / "model.txt"))
    del dtr, dva, d
    truth_v = {k: truth[k] for k in parts["V"]}
    vv = v[["s1_id", "cand_id"]].copy()
    vv["p"] = booster.predict(v[feats].to_numpy(np.float32))
    exp_params, s_exp = tune_expected(vv, truth_v)
    pred_v = expected_f05_decide(vv, **exp_params)
    cand_v = v.groupby("s1_id")["cand_id"].agg(set).to_dict()
    n_true = sum(len(t) for t in truth_v.values())
    info = {"valid_f05": s_exp, "expected_params": exp_params, "decision": "expected",
            "min_p": min_p, "max_per_s1": max_per_s1, "features": feats, "ce": bool(ce_path),
            "by_singleton": breakdown(pred_v, truth_v, {k: ("singleton" if not truth[k]
                                                            else "matched") for k in truth_v}),
            "trimmed_pair_recall": sum(len(t & cand_v.get(k, set()))
                                       for k, t in truth_v.items()) / max(1, n_true),
            "cands_per_s1_V": len(v) / len(truth_v), "best_iter": booster.best_iteration}
    print(json.dumps({k: x for k, x in info.items() if k != "features"}, indent=1,
                     default=str), flush=True)
    imp = pd.Series(booster.feature_importance("gain"), index=feats).sort_values()
    print(imp.tail(25).to_string(), flush=True)
    vv.to_parquet(run_dir / "valid_preds.parquet", index=False)
    (run_dir / "info.json").write_text(json.dumps(info, indent=1, default=str))
    del v, vv

    # ---------- test split, streamed file by file
    comp = Competition("test")
    outs = []
    for f in sorted((s1_dir / "feats").glob("test_*.parquet")):
        df, _ = build(f, s1_booster, FEATURES, comp, ce, min_p, max_per_s1)
        if df is None:
            continue
        out = df[["s1_id", "cand_id"]].copy()
        out["p"] = booster.predict(df[feats].to_numpy(np.float32)).astype(np.float32)
        outs.append(out)
        print(f"  [test] {f.stem}: {len(out):,} pairs [{time.time() - t0:.0f}s]", flush=True)
    pd.concat(outs, ignore_index=True).to_parquet(run_dir / "final_test_all.parquet",
                                                  index=False)
    import make_submission
    make_submission.main(run, resolve=True)
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1", required=True, help="run folder with stage-1 outputs")
    ap.add_argument("--run", required=True)
    ap.add_argument("--ce", default=None, help="path to ce_scores.parquet")
    ap.add_argument("--min_p", type=float, default=0.005)
    ap.add_argument("--max_per_s1", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    main(a.stage1, a.run, a.ce, a.min_p, a.max_per_s1, a.seed)

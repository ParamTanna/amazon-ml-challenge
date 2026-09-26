"""Stage 2b: learned candidate ranker - trims the high-recall blocking union to top-K.

Blocking is tuned for recall (tens of candidates per Source-1 record).  This cheap model
re-ranks those candidates using only signals already produced by blocking - cosine
scores, forward / reverse ranks, how a score compares with the entity's other candidates
and with the other entities competing for the same candidate, name / address commonness
and lengths - and keeps the top-K per entity.  No string comparisons are computed, so it
scores ~100M pairs in minutes.  The trimmed set is what the matching model scores.

Trained on partition A, evaluated on V (recall kept vs K), applied to every split.

Streaming, memory-lean implementation: the id columns are read once as integer codes,
candidate-side statistics are computed from two columns, and the remaining columns are
streamed in row batches (pair files are sorted by Source-1 record; a record split across
two batches is carried over).  Labels are matched with integer keys - no Python strings.
Writes WORK_DIR/pairs_trim/{split}_{country}.parquet (+ r0 = ranker probability, rk_* =
statistics over the full candidate list).
"""

import argparse
import json
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from config import N_JOBS, WORK_DIR
from model import load_truth, split_countries
from run_blocking import pair_path, safe_country
from splits import partition

TRIM_DIR = WORK_DIR / "pairs_trim"
TRIM_DIR.mkdir(parents=True, exist_ok=True)
NUM_COLS = ["cand_src", "score", "fwd_rank", "rev_rank", "s1_nfreq", "c_nfreq", "c_afreq",
            "s1_nlen", "c_nlen", "s1_alen", "c_alen"]
RANK_FEATURES = ["score", "fwd_rank", "rev_rank", "cand_src", "g_s1_max", "g_gap_s1",
                 "g_rank_s1", "g_rank_src", "g_n_cands", "g_n_strong", "g_c_max", "g_gap_c",
                 "g_c_n_s1", "g_c_rank", "s1_nfreq", "c_nfreq", "c_afreq", "s1_nlen",
                 "c_nlen", "s1_alen", "c_alen", "score_ratio"]
CONTEXT_COLS = ["g_s1_max", "g_gap_s1", "g_rank_s1", "g_rank_src", "g_n_cands", "g_n_strong",
                "g_c_max", "g_gap_c", "g_c_n_s1", "g_c_rank", "score_ratio"]


def trim_path(split, country):
    return TRIM_DIR / f"{split}_{safe_country(country)}.parquet"


def _group_stats_sorted(key, val):
    """Per-group max, rank (1 = best) and size for arrays key/val, via one sort."""
    order = np.lexsort((-val, key))
    k_s = key[order]
    first = np.r_[True, k_s[1:] != k_s[:-1]]
    idx = np.arange(len(k_s))
    start = np.maximum.accumulate(np.where(first, idx, 0))
    rank_s = (idx - start + 1).astype(np.float32)
    max_s = val[order][start]
    sizes = np.diff(np.r_[np.flatnonzero(first), len(k_s)])
    size_s = np.repeat(sizes, sizes).astype(np.float32)
    out_max = np.empty(len(val), np.float32)
    out_rank = np.empty(len(val), np.float32)
    out_size = np.empty(len(val), np.float32)
    out_max[order], out_rank[order], out_size[order] = max_s, rank_s, size_s
    return out_max, out_rank, out_size


def prepare(df):
    """Entity-side statistics (the frame holds complete Source-1 groups)."""
    for c in ("fwd_rank", "rev_rank"):
        v = df[c].to_numpy()
        df[c] = np.where(v < 0, 99, v).astype(np.float32)
    s = df["score"].to_numpy(np.float32)
    key = df["s1c"].to_numpy()
    smax, srank, ssize = _group_stats_sorted(key, s)
    df["g_s1_max"] = smax
    df["g_gap_s1"] = s - smax
    df["g_rank_s1"] = srank
    df["g_n_cands"] = ssize
    src_key = key * 4 + df["cand_src"].to_numpy().astype(np.int64)
    df["g_rank_src"] = _group_stats_sorted(src_key, s)[1]
    df["g_n_strong"] = pd.Series((s > 0.5).astype(np.float32)).groupby(key).transform(
        "sum").to_numpy()
    df["score_ratio"] = s / np.maximum(smax, 1e-6)
    return df


class PairFile:
    """Integer-coded view of one pair file with candidate-side statistics precomputed."""

    def __init__(self, split, ctry):
        self.path = pair_path(split, ctry)
        ids = pq.read_table(self.path, columns=["s1_id", "cand_id"]).to_pandas()
        self.s1_codes = ids["s1_id"].cat.codes.to_numpy().astype(np.int64)
        self.cand_codes = ids["cand_id"].cat.codes.to_numpy().astype(np.int64)
        self.s1_cats = ids["s1_id"].cat.categories
        self.cand_cats = ids["cand_id"].cat.categories
        del ids
        score = pq.read_table(self.path, columns=["score"]).column("score").to_numpy()
        score = score.astype(np.float32)
        cmax, crank, csize = _group_stats_sorted(self.cand_codes, score)
        self.cstats = {"g_c_max": cmax, "g_gap_c": score - cmax, "g_c_n_s1": csize,
                       "g_c_rank": crank}
        self.n = len(self.s1_codes)

    def batches(self, rows_per_batch=2_000_000):
        """Yield DataFrames holding complete Source-1 groups, with all ranker features."""
        pf = pq.ParquetFile(self.path)
        offset, carry = 0, None
        for rb in pf.iter_batches(batch_size=rows_per_batch, columns=NUM_COLS):
            df = rb.to_pandas()
            lo, hi = offset, offset + len(df)
            offset = hi
            df["s1c"] = self.s1_codes[lo:hi]
            df["candc"] = self.cand_codes[lo:hi]
            for k, v in self.cstats.items():
                df[k] = v[lo:hi]
            if carry is not None:
                df = pd.concat([carry, df], ignore_index=True)
                carry = None
            if hi < self.n:     # hold back the last (possibly incomplete) record
                s1c = df["s1c"].to_numpy()
                cut = len(df) - int((s1c == s1c[-1]).sum())
                carry, df = df.iloc[cut:].reset_index(drop=True), df.iloc[:cut].copy()
            if len(df):
                yield prepare(df)
        if carry is not None and len(carry):
            yield prepare(carry)


def truth_keys(pf, truth, s1_ids):
    """Integer keys s1_code * n_cand + cand_code of the true pairs of the given entities."""
    s1_list = [s for s in s1_ids if truth.get(s)]
    s1_pos = pf.s1_cats.get_indexer(s1_list)
    rows_s, rows_c = [], []
    for s, sp_ in zip(s1_list, s1_pos):
        if sp_ < 0:
            continue
        for c in truth[s]:
            rows_s.append(sp_)
            rows_c.append(c)
    cpos = pf.cand_cats.get_indexer(rows_c)
    ok = cpos >= 0
    return np.asarray(rows_s, dtype=np.int64)[ok] * len(pf.cand_cats) + cpos[ok]


def top_k(df, k, score_col="r0"):
    order = np.lexsort((-df[score_col].to_numpy(), df["s1c"].to_numpy()))
    d = df.iloc[order]
    key = d["s1c"].to_numpy()
    first = np.r_[True, key[1:] != key[:-1]]
    start = np.maximum.accumulate(np.where(first, np.arange(len(key)), 0))
    return d[(np.arange(len(key)) - start) < k]


def main(k, seed=0, max_train_pairs=6_000_000):
    t0 = time.time()
    truth = load_truth()
    parts = partition(truth.keys(), seed)
    run_dir = WORK_DIR / "runs" / "ranker"
    run_dir.mkdir(parents=True, exist_ok=True)
    # ---- training / validation rows (partitions A and V), no string ids
    xs_tr, ys_tr, va_parts, n_true_v = [], [], [], 0
    for ci, ctry in enumerate(split_countries("train")):
        pf = PairFile("train", ctry)
        in_a = np.asarray(pf.s1_cats.isin(parts["A"]))
        in_v = np.asarray(pf.s1_cats.isin(parts["V"]))
        keys_true = truth_keys(pf, truth, list(pf.s1_cats[in_a | in_v]))
        n_true_v += sum(len(truth[s]) for s in pf.s1_cats[in_v])
        for df in pf.batches():
            s1c = df["s1c"].to_numpy()
            lab = np.isin(s1c * len(pf.cand_cats) + df["candc"].to_numpy(), keys_true)
            a, v = in_a[s1c], in_v[s1c]
            if a.any():
                xs_tr.append(df.loc[a, RANK_FEATURES].to_numpy(np.float32))
                ys_tr.append(lab[a])
            if v.any():
                sub = df.loc[v, RANK_FEATURES + ["s1c"]].copy()
                sub["s1c"] = sub["s1c"] + ci * 10**9      # unique across countries
                sub["label"] = lab[v]
                va_parts.append(sub)
        print(f"[ranker] {ctry}: collected training rows [{time.time() - t0:.0f}s]", flush=True)
        del pf
    x_tr, y_tr = np.vstack(xs_tr), np.concatenate(ys_tr)
    del xs_tr, ys_tr
    if len(y_tr) > max_train_pairs:
        idx = np.random.default_rng(seed).choice(len(y_tr), max_train_pairs, replace=False)
        x_tr, y_tr = x_tr[idx], y_tr[idx]
    va = pd.concat(va_parts, ignore_index=True)
    del va_parts
    print(f"[ranker] train {len(y_tr):,} pairs, valid {len(va):,}  [{time.time() - t0:.0f}s]",
          flush=True)
    params = dict(objective="binary", learning_rate=0.1, num_leaves=63, min_data_in_leaf=200,
                  feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1, verbose=-1,
                  num_threads=N_JOBS)
    dtr = lgb.Dataset(x_tr, y_tr, feature_name=RANK_FEATURES)
    dva = lgb.Dataset(va[RANK_FEATURES].to_numpy(np.float32), va["label"].to_numpy(),
                      reference=dtr)
    booster = lgb.train(params, dtr, 1000, valid_sets=[dva],
                        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)])
    booster.save_model(str(run_dir / "ranker.txt"))
    del x_tr, y_tr, dtr, dva
    va["r0"] = booster.predict(va[RANK_FEATURES].to_numpy(np.float32), num_threads=N_JOBS)
    n_s1_v = len(parts["V"])
    blocked = int(va["label"].sum())
    report = {"blocking_pair_recall": blocked / max(1, n_true_v),
              "cands_per_s1_blocking": len(va) / n_s1_v}
    for kk in (3, 5, 8, 10, 12, 15, 20, 25, 30):
        kept = top_k(va, kk)
        report[f"k{kk}"] = {"pair_recall": float(kept["label"].sum() / max(1, n_true_v)),
                            "kept_of_blocked": float(kept["label"].sum() / max(1, blocked)),
                            "cands_per_s1": len(kept) / n_s1_v}
    print(json.dumps(report, indent=1), flush=True)
    (run_dir / "ranker_report.json").write_text(json.dumps(report, indent=1))
    imp = pd.Series(booster.feature_importance("gain"), index=RANK_FEATURES).sort_values()
    print(imp.tail(10).to_string(), flush=True)
    del va
    # ---- trim every split / country, batch by batch
    for split in ("train", "test"):
        for ctry in split_countries(split):
            pf = PairFile(split, ctry)
            kept = []
            for df in pf.batches():
                df["r0"] = booster.predict(df[RANK_FEATURES].to_numpy(np.float32),
                                           num_threads=N_JOBS).astype(np.float32)
                kept.append(top_k(df, k))
            out = pd.concat(kept, ignore_index=True)
            del kept
            out = out.rename(columns={c: "rk_" + c for c in CONTEXT_COLS})
            s1c = out.pop("s1c").to_numpy().astype(np.int32)
            candc = out.pop("candc").to_numpy().astype(np.int32)
            out.insert(0, "s1_id", pd.Categorical.from_codes(s1c, categories=pf.s1_cats))
            out.insert(1, "cand_id", pd.Categorical.from_codes(candc, categories=pf.cand_cats))
            out["s1_id"] = out["s1_id"].cat.remove_unused_categories()
            out["cand_id"] = out["cand_id"].cat.remove_unused_categories()
            out["cand_src"] = out["cand_src"].astype(np.int8)
            out.to_parquet(trim_path(split, ctry), index=False)
            print(f"[ranker] {split} {ctry}: kept {len(out):,} of {pf.n:,} pairs "
                  f"[{time.time() - t0:.0f}s]", flush=True)
            del pf, out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=15)
    a = ap.parse_args()
    main(a.k)

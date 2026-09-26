"""Export data for the neural cross-encoder (trained and run on a Kaggle GPU).

--part records ce_records_train.parquet / ce_records_test.parquet  (entity_id, text)
--part train   ce_train_pairs.parquet  (s1_id, cand_id, label) from partition C
--part score   ce_score_pairs.parquet (s1_id, cand_id, split) - candidates kept by the
               first-stage model for partitions D and V (train split) and the test split

Record text = transliterated, accent-free, lower-case "name ; address".  The raw noise
(typos, junk symbols, abbreviations) is kept on purpose: the model learns it.
"""

import argparse
from multiprocessing import Pool

import numpy as np
import pandas as pd

from config import N_JOBS, WORK_DIR

CE_DIR = WORK_DIR / "ce_export"
CE_DIR.mkdir(parents=True, exist_ok=True)


def _texts(rows):
    from normalize import to_ascii
    return [(to_ascii(n, "name").strip() + " ; " + to_ascii(a, "addr").strip())
            for n, a in rows]


def record_texts(split, ids=None, pool=None):
    """entity_id + text for all records of a split (optionally only ``ids``)."""
    from preprocess import raw_table
    frames = []
    for src in ("source1", "source2", "source3"):
        df = raw_table(split, src)
        if ids is not None:
            df = df[df["entity_id"].isin(ids)]
        rows = list(zip(df["business_name"].astype(str), df["business_address"].astype(str)))
        chunks = [rows[i:i + 20000] for i in range(0, len(rows), 20000)]
        texts = [t for part in pool.imap(_texts, chunks) for t in part]
        frames.append(pd.DataFrame({"entity_id": df["entity_id"].astype(str).values,
                                    "text": texts}))
        print(f"  [ce] {split} {src}: {len(df):,} records", flush=True)
    return pd.concat(frames, ignore_index=True)


def ce_train_pairs(truth, c_ids, n_hard=8, n_rand=2, seed=0):
    """All positives + hard (top blocking score) and random negatives for partition C."""
    import pyarrow.parquet as pq
    from model import add_labels, split_countries
    from run_blocking import pair_path
    parts = []
    for ctry in split_countries("train"):
        p = pq.read_table(pair_path("train", ctry), columns=["s1_id", "cand_id", "score"]).to_pandas()
        p = p[p["s1_id"].isin(c_ids)].reset_index(drop=True)
        p["s1_id"] = p["s1_id"].astype(str)
        p["cand_id"] = p["cand_id"].astype(str)
        p = add_labels(p, truth)
        pos = p[p["label"]]
        neg = p[~p["label"]].sort_values(["s1_id", "score"], ascending=[True, False])
        neg["r"] = neg.groupby("s1_id").cumcount()
        hard = neg[neg["r"] < n_hard]
        rest = neg[neg["r"] >= n_hard]
        rand = rest.sample(n=min(len(rest), n_rand * rest["s1_id"].nunique()), random_state=seed)
        parts.append(pd.concat([pos, hard, rand])[["s1_id", "cand_id", "label"]])
        print(f"  [ce] {ctry}: pos {len(pos):,} hard {len(hard):,} rand {len(rand):,}", flush=True)
    out = pd.concat(parts, ignore_index=True)
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def main(part, run=None, min_p=0.005, max_per_s1=10):
    import os
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    from model import load_truth
    from splits import partition
    truth = load_truth()
    parts = partition(truth.keys())
    with Pool(N_JOBS) as pool:
        if part == "records":
            record_texts("train", None, pool).to_parquet(CE_DIR / "ce_records_train.parquet",
                                                         index=False, compression="zstd")
            record_texts("test", None, pool).to_parquet(CE_DIR / "ce_records_test.parquet",
                                                        index=False, compression="zstd")
        elif part == "train":
            pairs = ce_train_pairs(truth, parts["C"])
            pairs.to_parquet(CE_DIR / "ce_train_pairs.parquet", index=False)
            print(f"[ce] train pairs {len(pairs):,} (pos {pairs['label'].mean():.3f})", flush=True)
        else:
            # candidates the final stage scores: first-stage top ``max_per_s1`` with
            # p >= ``min_p`` for partitions D and V (train split) and for the test split.
            # (run_stage2.py applies exactly the same trimming.)
            run_dir = WORK_DIR / "runs" / run
            frames = []
            for path in sorted(run_dir.glob("stage1_*_*.parquet")):
                split = path.stem.split("_")[1]
                p = pd.read_parquet(path)
                p = p[p["p"] >= min_p].sort_values(["s1_id", "p"], ascending=[True, False])
                p = p[p.groupby("s1_id").cumcount() < max_per_s1]
                if split == "train":
                    p = p[p["s1_id"].isin(parts["D"] | parts["V"])]
                    p["split"] = np.where(p["s1_id"].isin(parts["D"]), "D", "V")
                else:
                    p["split"] = "test"
                frames.append(p[["s1_id", "cand_id", "split"]])
                print(f"  [ce] {path.stem}: {len(p):,} pairs", flush=True)
            out = pd.concat(frames, ignore_index=True)
            out.to_parquet(CE_DIR / "ce_score_pairs.parquet", index=False, compression="zstd")
            print(out["split"].value_counts().to_string(), flush=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["records", "train", "score"], required=True)
    ap.add_argument("--run")
    a = ap.parse_args()
    main(a.part, a.run)

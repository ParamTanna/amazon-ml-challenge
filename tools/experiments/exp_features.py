"""Quick check of pair features on a small train sample (memory-light: loads only needed rows)."""
import sys, time
sys.path.insert(0, r"D:\Amazon ML Challenge\code\business_entity_resolution\src")
import numpy as np, pandas as pd, pyarrow.dataset as ds, pyarrow as pa
from multiprocessing import Pool
from features import PAIR_FEATURES, compute_pair_features, records, pair_features
from model import load_pairs, load_truth, add_labels, REC_COLS
from preprocess import norm_path


def load_rows(split, source, ids):
    t = ds.dataset(norm_path(split, source)).to_table(
        columns=REC_COLS, filter=ds.field("entity_id").isin(pa.array(list(ids))))
    return t.to_pandas()


if __name__ == "__main__":
    p = load_pairs("train", "India")
    s1s = p["s1_id"].cat.categories[np.random.default_rng(0).choice(len(p["s1_id"].cat.categories), 3000, replace=False)]
    p = p[p["s1_id"].isin(set(s1s))].reset_index(drop=True)
    p["s1_id"] = p["s1_id"].astype(str); p["cand_id"] = p["cand_id"].astype(str)
    s1 = load_rows("train", "source1", set(p.s1_id))
    cand = pd.concat([load_rows("train", "source2", set(p.cand_id)), load_rows("train", "source3", set(p.cand_id))], ignore_index=True)
    li = pd.Index(s1.entity_id).get_indexer(p.s1_id); ri = pd.Index(cand.entity_id).get_indexer(p.cand_id)
    L, R = records(s1, li), records(cand, ri)
    t = time.time(); _ = [pair_features(a, b) for a, b in zip(L[:5000], R[:5000])]
    print("single-thread us/pair", (time.time() - t) / 5000 * 1e6, flush=True)
    with Pool(8) as pool:
        t = time.time(); X = compute_pair_features(L, R, pool, chunk=5000); print("pool(8)", len(L), "pairs in", time.time() - t, "s")
    df = pd.DataFrame(X, columns=PAIR_FEATURES)
    df["label"] = add_labels(p, load_truth())["label"].values
    print("pos rate", df.label.mean())
    print(df.groupby("label").mean().T.round(3).to_string())

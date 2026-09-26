"""Stage 4: pair features -> LightGBM match probability -> per-entity decision.

Training uses a random subset of Source-1 training records (all of their candidates);
a disjoint subset is held out to tune the decision threshold for macro F0.5.
"""

import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from config import N_JOBS
from evaluate import f05_macro, parse_id_lists
from features import (GROUP_FEATURES, PAIR_FEATURES, add_group_features,
                      compute_pair_features, records)
from run_blocking import pair_path

REC_COLS = ["entity_id", "country", "name_core", "name_tok", "addr_tok", "addr_nums",
            "is_domain", "has_alias", "name_indic", "addr_indic"]
BLOCK_FEATURES = ["score", "fwd_rank", "rev_rank", "cand_src", "r0"]
# candidate-ranker context computed over the full (untrimmed) blocking lists
RANKER_CONTEXT = ["rk_g_s1_max", "rk_g_gap_s1", "rk_g_rank_s1", "rk_g_n_cands", "rk_g_c_max",
                  "rk_g_gap_c", "rk_g_c_n_s1", "rk_g_c_rank", "rk_score_ratio"]
FREQ_FEATURES = ["s1_nfreq", "c_nfreq", "c_afreq"]
# within-entity consensus: true copies agree with the Source-1 record and with each other,
# decoys (another business) tend to share their own alternative number / name
CONSENSUS_FEATURES = ["cs_num_same_c", "cs_num_same_s1", "cs_num_c_is_s1", "cs_name_same_c",
                      "cs_name_same_s1", "cs_name_c_is_s1", "cs_num_alt_share"]
FEATURES = (PAIR_FEATURES + GROUP_FEATURES + BLOCK_FEATURES + RANKER_CONTEXT + FREQ_FEATURES
            + CONSENSUS_FEATURES)


class RecordStore:
    """Normalised records of one split and one country, addressable by entity_id.

    Strings stay Arrow-backed (compact); Python objects are only created per chunk.
    """

    def __init__(self, split, country):
        import pyarrow.dataset as ds
        from preprocess import norm_path
        flt = ds.field("country") == country
        mapper = {pa.string(): pd.StringDtype("pyarrow"),
                  pa.large_string(): pd.StringDtype("pyarrow")}.get

        def load(source):
            return ds.dataset(norm_path(split, source)).to_table(
                columns=REC_COLS, filter=flt).to_pandas(types_mapper=mapper)

        self.s1 = load("source1")
        self.cand = pd.concat([load("source2"), load("source3")], ignore_index=True)
        self.s1_index = pd.Index(self.s1["entity_id"].astype(str))
        self.cand_index = pd.Index(self.cand["entity_id"].astype(str))
        # how common a name / address is within the country: a shared name is weak evidence
        names = pd.concat([self.s1["name_core"], self.cand["name_core"]], ignore_index=True)
        nfreq = names.map(names.value_counts()).to_numpy(np.float32)
        self.s1_nfreq = np.log1p(nfreq[:len(self.s1)])
        self.c_nfreq = np.log1p(nfreq[len(self.s1):])
        afreq = self.cand["addr_tok"].map(self.cand["addr_tok"].value_counts())
        self.c_afreq = np.log1p(afreq.to_numpy(np.float32))
        # first house number and an order-free name key per record, as compact int64 keys
        def num0(df):
            first = df["addr_nums"].astype(str).str.split(" ", n=1).str[0].str[:9]
            return pd.to_numeric(first, errors="coerce").fillna(-1).to_numpy(np.int64)

        def name_key(df):
            keys = df["name_core"].astype(str).map(lambda x: " ".join(sorted(x.split())))
            return pd.util.hash_array(keys.to_numpy(dtype=object)).astype(np.int64)
        self.s1_num0, self.c_num0 = num0(self.s1), num0(self.cand)
        self.s1_nkey, self.c_nkey = name_key(self.s1), name_key(self.cand)


def split_countries(split):
    """Country labels present in a split's Source-1 file (an open set)."""
    import pyarrow.parquet as pq_
    from preprocess import norm_path
    col = pq_.read_table(norm_path(split, "source1"), columns=["country"]).column("country")
    return pd.Series(col.to_pandas()).value_counts().index.tolist()


def load_pairs(split, country, trimmed=True):
    """Candidate pairs of one country with ids as categoricals, plus group features.

    Uses the ranker-trimmed candidate set when it exists (see rank_candidates.py).
    """
    from rank_candidates import trim_path
    path = trim_path(split, country) if trimmed and trim_path(split, country).exists()         else pair_path(split, country)
    tbl = pq.read_table(path)
    for col in ("s1_id", "cand_id"):
        i = tbl.schema.get_field_index(col)
        if not pa.types.is_dictionary(tbl.schema.field(col).type):
            tbl = tbl.set_column(i, col, pc.dictionary_encode(tbl.column(col)))
    p = tbl.to_pandas()
    p["s1_key"] = p["s1_id"].cat.codes.astype(np.int32)
    p["cand_key"] = p["cand_id"].cat.codes.astype(np.int32)
    for col in ["r0"] + RANKER_CONTEXT:
        if col not in p.columns:
            p[col] = np.float32(-1)
    return add_group_features(p)


def _row_index(index, ids):
    """Row positions of categorical ``ids`` in ``index`` (looked up once per category)."""
    if isinstance(ids.dtype, pd.CategoricalDtype):
        pos = index.get_indexer(ids.cat.categories.astype(str))
        return pos[ids.cat.codes.values]
    return index.get_indexer(ids.astype(str))


def featurize(pairs, store, pool, chunk=400000):
    """Append the pairwise similarity features to ``pairs`` (in place, returns it).

    Record tuples are materialised chunk by chunk to keep memory bounded.
    """
    t0 = time.time()
    li = _row_index(store.s1_index, pairs["s1_id"])
    ri = _row_index(store.cand_index, pairs["cand_id"])
    x = np.empty((len(pairs), len(PAIR_FEATURES)), dtype=np.float32)
    for i in range(0, len(pairs), chunk):
        x[i:i + chunk] = compute_pair_features(records(store.s1, li[i:i + chunk]),
                                               records(store.cand, ri[i:i + chunk]), pool,
                                               chunk=10000)
    for j, name in enumerate(PAIR_FEATURES):
        pairs[name] = x[:, j]
    pairs["s1_nfreq"] = store.s1_nfreq[li]
    pairs["c_nfreq"] = store.c_nfreq[ri]
    pairs["c_afreq"] = store.c_afreq[ri]
    add_consensus_features(pairs, store.s1_num0[li], store.c_num0[ri],
                           store.s1_nkey[li], store.c_nkey[ri])
    print(f"    features for {len(pairs):,} pairs in {time.time() - t0:.0f}s", flush=True)
    return pairs


def add_consensus_features(pairs, s1_num, c_num, s1_key, c_key):
    """Counts over the same entity's candidates (needs all of an entity's pairs together).

    s1_num / c_num: first house number as int64 (-1 = none); s1_key / c_key: int64 hash of
    the order-free core name.
    """
    g = pairs["s1_id"].cat.codes.to_numpy() if isinstance(
        pairs["s1_id"].dtype, pd.CategoricalDtype) else pairs["s1_id"].astype(str).to_numpy()
    df = pd.DataFrame({"g": g, "cn": c_num, "sn": s1_num, "ck": c_key, "sk": s1_key})
    has_num = (df["cn"] >= 0).to_numpy()
    num_eq = (df["cn"].to_numpy() == df["sn"].to_numpy()) & has_num
    name_eq = df["ck"].to_numpy() == df["sk"].to_numpy()
    same_num = np.where(has_num, df.groupby(["g", "cn"])["cn"].transform("size").to_numpy(), 0)
    pairs["cs_num_same_c"] = (same_num - has_num).astype(np.float32)   # others sharing c's number
    pairs["cs_num_same_s1"] = pd.Series(num_eq).groupby(g).transform("sum").to_numpy(np.float32)
    pairs["cs_num_c_is_s1"] = np.where(has_num, num_eq, -1).astype(np.float32)
    pairs["cs_name_same_c"] = (df.groupby(["g", "ck"])["ck"].transform("size").to_numpy()
                               - 1).astype(np.float32)
    pairs["cs_name_same_s1"] = pd.Series(name_eq).groupby(g).transform("sum").to_numpy(np.float32)
    pairs["cs_name_c_is_s1"] = name_eq.astype(np.float32)
    # an alternative number shared by other candidates while differing from Source 1
    pairs["cs_num_alt_share"] = np.where(has_num & ~num_eq, same_num - 1, 0).astype(np.float32)
    # compact keys so later stages can weight the consensus by match probabilities
    pairs["k_c_num"] = np.asarray(c_num, dtype=np.int64)
    pairs["k_s1_num"] = np.asarray(s1_num, dtype=np.int64)
    pairs["k_c_name"] = np.asarray(c_key, dtype=np.int64)
    pairs["k_s1_name"] = np.asarray(s1_key, dtype=np.int64)


def add_labels(pairs, truth):
    pairs["label"] = np.fromiter((c in truth.get(s, ()) for s, c in
                                  zip(pairs["s1_id"].astype(str), pairs["cand_id"].astype(str))),
                                 dtype=bool, count=len(pairs))
    return pairs


def load_truth():
    from config import CACHE_DIR
    gt = pd.read_parquet(CACHE_DIR / "train_ground_truth.parquet")
    return dict(zip(gt["source1_entity_id"], parse_id_lists(gt["matched_entity_ids"])))


LGB_PARAMS = dict(objective="binary", learning_rate=0.1, num_leaves=127,
                  min_data_in_leaf=100, feature_fraction=0.8, bagging_fraction=0.8,
                  bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=N_JOBS)


def train_model(tr, va, rounds=1500):
    dtr = lgb.Dataset(tr[FEATURES].values.astype(np.float32), tr["label"].values,
                      feature_name=FEATURES, free_raw_data=True)
    dva = lgb.Dataset(va[FEATURES].values.astype(np.float32), va["label"].values,
                      reference=dtr)
    booster = lgb.train(LGB_PARAMS, dtr, rounds, valid_sets=[dva],
                        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)])
    return booster


def decide(pairs, prob_col="p", threshold=0.5):
    """Per Source-1 entity: every candidate whose probability clears the threshold."""
    sel = pairs[pairs[prob_col] >= threshold]
    return sel.groupby("s1_id")["cand_id"].agg(set).to_dict()


def tune_threshold(va, truth_va, grid=None):
    grid = grid if grid is not None else np.round(np.arange(0.2, 0.95, 0.025), 3)
    best = (None, -1.0)
    for t in grid:
        s = f05_macro(decide(va, threshold=t), truth_va)
        if s > best[1]:
            best = (t, s)
    return best

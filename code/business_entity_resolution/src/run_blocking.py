"""Stage 2 driver: generate candidate pairs for every Source-1 record of a split.

Writes WORK_DIR/pairs/{split}_{country}.parquet with columns
    s1_id, cand_id (categoricals), cand_src,
    score, fwd_rank, rev_rank          main-view cosine / forward rank / reverse rank
    <view>_score/_fwd/_rev             the same for every blocking view
    s1_nfreq, c_nfreq, c_afreq         log-frequency of the core name / candidate address
    s1_nlen, c_nlen, s1_alen, c_alen   token counts (cheap signals for the ranker)
Countries are processed independently (an open set: whatever labels appear).
"""

import argparse
import time

import numpy as np
import pandas as pd

from blocking import VIEWS, candidates_for_country, make_pool
from config import WORK_DIR

PAIR_DIR = WORK_DIR / "pairs"
PAIR_DIR.mkdir(parents=True, exist_ok=True)


def safe_country(c):
    return "".join(ch if ch.isalnum() else "_" for ch in c) or "unknown"


def pair_path(split, country):
    return PAIR_DIR / f"{split}_{safe_country(country)}.parquet"


def _ntok(series):
    s = series.astype(str)
    return (s.str.count(" ").to_numpy(np.int16) + (s.str.len().to_numpy() > 0)).astype(np.int16)


def load_country(split, source, country, cols):
    """One source table restricted to one country (Arrow-backed strings)."""
    import pyarrow as pa
    import pyarrow.dataset as ds
    from preprocess import norm_path
    mapper = {pa.string(): pd.StringDtype("pyarrow"),
              pa.large_string(): pd.StringDtype("pyarrow")}.get
    return ds.dataset(norm_path(split, source)).to_table(
        columns=cols, filter=ds.field("country") == country).to_pandas(types_mapper=mapper)


def main(split, views, countries=None):
    cols = ["entity_id", "country", "name_core", "name_tok", "addr_tok"]
    if not countries:
        from model import split_countries
        countries = split_countries(split)
    with make_pool() as pool:
        for ctry in countries:
            t0 = time.time()
            a, b, c = (load_country(split, s, ctry, cols) for s in ("source1", "source2", "source3"))
            print(f"[blocking] {split} {ctry}: S1={len(a):,} S2={len(b):,} S3={len(c):,}",
                  flush=True)
            if len(b) + len(c) == 0:
                pd.DataFrame(columns=["s1_id", "cand_id", "cand_src", "score", "fwd_rank",
                                      "rev_rank"]).to_parquet(pair_path(split, ctry))
                continue
            p = candidates_for_country(a, b, c, None, views=views, pool=pool)
            cand = pd.concat([b, c], ignore_index=True)
            codes = (p["cand_row"].values + np.where(p["cand_src"].values == 3, len(b), 0)
                     ).astype(np.int64)
            s1_rows = p["s1_row"].values.astype(np.int64)
            # cheap per-record signals for the candidate ranker
            names = pd.concat([a["name_core"], cand["name_core"]], ignore_index=True).astype(str)
            nfreq = np.log1p(names.map(names.value_counts()).to_numpy(np.float32))
            addrs = cand["addr_tok"].astype(str)
            afreq = np.log1p(addrs.map(addrs.value_counts()).to_numpy(np.float32))
            nlen = _ntok(names)
            alen_s1, alen_c = _ntok(a["addr_tok"]), _ntok(cand["addr_tok"])
            out = pd.DataFrame({
                "s1_id": pd.Categorical.from_codes(s1_rows.astype(np.int32),
                                                   categories=a["entity_id"].astype(str)),
                "cand_id": pd.Categorical.from_codes(
                    codes.astype(np.int32), categories=cand["entity_id"].astype(str)),
                "cand_src": p["cand_src"].values.astype(np.int8),
            })
            for col in p.columns:
                if col not in ("s1_row", "cand_row", "cand_src"):
                    out[col] = p[col].values
            out["s1_nfreq"] = nfreq[s1_rows]
            out["c_nfreq"] = nfreq[len(a) + codes]
            out["c_afreq"] = afreq[codes]
            out["s1_nlen"] = nlen[s1_rows]
            out["c_nlen"] = nlen[len(a) + codes]
            out["s1_alen"] = alen_s1[s1_rows]
            out["c_alen"] = alen_c[codes]
            out.to_parquet(pair_path(split, ctry), index=False)
            print(f"[blocking] {split} {ctry}: {len(out):,} pairs "
                  f"({len(out) / max(1, len(a)):.1f}/S1) in {time.time() - t0:.0f}s", flush=True)
            del a, b, c, p, out, cand


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--k", type=int, default=VIEWS["main"]["k_fwd"])
    ap.add_argument("--k_rev", type=int, default=VIEWS["main"]["k_rev"])
    ap.add_argument("--max_df", type=int, default=VIEWS["main"]["max_df"])
    ap.add_argument("--countries", nargs="*")
    a = ap.parse_args()
    views = {"main": {"max_df": a.max_df, "k_fwd": a.k, "k_rev": a.k_rev, "rev_raw": True}}
    main(a.split, views, a.countries)

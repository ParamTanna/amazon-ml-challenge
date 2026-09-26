"""Blocking recall on the full training split from saved pair files (per country)."""
import sys
sys.path.insert(0, r"D:\Amazon ML Challenge\code\business_entity_resolution\src")
import numpy as np, pandas as pd, pyarrow.parquet as pq
from evaluate import parse_id_lists

pair_dir = sys.argv[1] if len(sys.argv) > 1 else r"D:\Amazon ML Challenge\work\pairs"
gt = pd.read_parquet(r"D:\Amazon ML Challenge\work\cache\train_ground_truth.parquet")
g = gt.assign(m=gt.matched_entity_ids.str.split(",")).explode("m")
g = g[g.m.fillna("") != ""]
for ctry in ("US", "India"):
    p = pq.read_table(fr"{pair_dir}\train_{ctry}.parquet", columns=["s1_id", "cand_id", "fwd_rank", "rev_rank"]).to_pandas()
    s1c = p.s1_id.cat.categories; cc = p.cand_id.cat.categories
    gi = g[g.source1_entity_id.isin(set(s1c))]
    gs = pd.Index(s1c).get_indexer(gi.source1_entity_id); gcand = pd.Index(cc).get_indexer(gi.m)
    ok = gcand >= 0   # true candidates that exist in this country's pools
    n_true = len(gi)
    gkey = gs[ok].astype(np.int64) * len(cc) + gcand[ok]
    pkey = p.s1_id.cat.codes.values.astype(np.int64) * len(cc) + p.cand_id.cat.codes.values
    n_s1 = len(s1c)
    print(f"== {ctry}: S1 {n_s1:,}, true pairs {n_true:,} ({(~ok).sum():,} true cands missing from pools)")
    for k in (3, 5, 10):
        for rev in (False, True):
            sel = (p.fwd_rank.values >= 0) & (p.fwd_rank.values < k)
            if rev:
                sel |= p.rev_rank.values >= 0
            hit = np.isin(gkey, pkey[sel])
            # entity-level: all true matches of the S1 found
            s_hit = pd.Series(hit).groupby(gs[ok]).all()
            print(f"  k={k:2d} rev={rev!s:5}  pair_recall={hit.sum()/n_true:.4f}  "
                  f"entity_full={s_hit.mean():.4f}  cands/S1={sel.sum()/n_s1:.1f}", flush=True)

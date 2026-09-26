"""Experiment: recall of the multi-view blocking union on a train sample, per view / direction.

usage: python exp_blocking2.py N_QUERY COUNTRY '{"main": {...}, "ngram": {...}}'
"""
import sys, time, json
sys.path.insert(0, r"D:\Amazon ML Challenge\code\business_entity_resolution\src")
import numpy as np, pandas as pd


def main():
    from preprocess import load_norm
    from blocking import candidates_for_country, make_pool, VIEWS
    from evaluate import parse_id_lists
    n_query = int(sys.argv[1]); ctry = sys.argv[2]
    views = json.loads(sys.argv[3]) if len(sys.argv) > 3 else VIEWS
    cols = ["entity_id", "country", "name_core", "name_tok", "addr_tok"]
    s1, s2, s3 = (load_norm("train", s, cols) for s in ("source1", "source2", "source3"))
    a, b, c = (d[d.country == ctry].reset_index(drop=True) for d in (s1, s2, s3))
    del s1, s2, s3
    gt = pd.read_parquet(r"D:\Amazon ML Challenge\work\cache\train_ground_truth.parquet")
    truth = dict(zip(gt.source1_entity_id, parse_id_lists(gt.matched_entity_ids)))
    rng = np.random.default_rng(0)
    q_ids = set(rng.choice(a.entity_id.values, n_query, replace=False))
    qm = a.entity_id.isin(q_ids).values
    t = time.time()
    with make_pool() as pool:
        p = candidates_for_country(a, b, c, qm, views=views, pool=pool)
    print(f"blocking {time.time()-t:.0f}s", flush=True)
    ids = np.concatenate([b.entity_id.values, c.entity_id.values])
    p["s1_id"] = a.entity_id.values[p.s1_row]
    p["cand_id"] = ids[np.where(p.cand_src == 2, p.cand_row, p.cand_row + len(b))]
    tp = set((s, m) for s in q_ids for m in truth[s])
    p["label"] = [(s, m) in tp for s, m in zip(p.s1_id, p.cand_id)]
    p.to_parquet(r"D:\Amazon ML Challenge\work\exp_blocking2_pairs.parquet", index=False)
    n_true = len(tp)
    need = pd.Series({q: len(truth[q]) for q in q_ids})

    def report(name, sel):
        s = p[sel]
        got = s[s.label].groupby("s1_id").size()
        full = (got.reindex(need.index).fillna(0) == need).mean()
        print(f"{name:40s} pair_recall={s.label.sum()/n_true:.4f} entity_full={full:.4f} "
              f"cands/S1={len(s)/len(q_ids):.1f}", flush=True)

    V = list(views)
    for v in V:
        report(f"{v} fwd", p[f"{v}_fwd"] >= 0)
        report(f"{v} fwd+rev", (p[f"{v}_fwd"] >= 0) | (p[f"{v}_rev"] >= 0))
    report("ALL union", np.ones(len(p), bool))
    for v in V:
        for kf in (5, 10, 20, 30, 50, 100):
            for kr in (0, 1, 3, 5, 10, 20):
                if kf > p[f"{v}_fwd"].max() + 1 or kr > p[f"{v}_rev"].max() + 1:
                    continue
                sel = p[f"{v}_fwd"].between(0, kf - 1) | p[f"{v}_rev"].between(0, kr - 1)
                report(f"{v} fwd<{kf} rev<{kr}", sel)


if __name__ == "__main__":
    main()

"""Stage 6: build matching_results.tsv and candidate_pairs.tsv from saved test probabilities.

Reads WORK_DIR/runs/<run>/test_preds_*.parquet (s1_id, cand_id, p) and the decision
settings in info.json, processes one country at a time (memory-light) and applies:
  1. the per-entity decision rule tuned on validation (expected-F0.5 or threshold);
  2. optional one-to-one resolution: a Source-2/3 record belongs to at most one Source-1
     entity (true for 100% of training matches), so if several entities select the
     same record, only the one with the highest probability keeps it.
"""

import argparse
import json

import numpy as np
import pandas as pd

from config import OUT_DIR, WORK_DIR
from decision import expected_f05_decide
from preprocess import load_norm


def decide_frame(df, info):
    if info.get("decision") == "expected":
        return expected_f05_decide(df, **info["expected_params"])
    sel = df[df["p"] >= info["threshold"]]
    return sel.groupby("s1_id", observed=True)["cand_id"].agg(lambda s: set(map(str, s))).to_dict()


def one_to_one(matches, probs):
    """Keep each candidate only for the Source-1 entity with the highest probability."""
    rows = [(s, c) for s, cs in matches.items() for c in cs]
    if not rows:
        return matches, 0
    df = pd.DataFrame(rows, columns=["s1_id", "cand_id"]).merge(probs, on=["s1_id", "cand_id"])
    df = df.sort_values("p", ascending=False)
    keep = df[~df.duplicated("cand_id")]
    out = keep.groupby("s1_id")["cand_id"].agg(set).to_dict()
    return out, len(df) - len(keep)


def main(run, resolve):
    run_dir = WORK_DIR / "runs" / run
    info = json.loads((run_dir / "info.json").read_text())
    test_ids = load_norm("test", "source1", ["entity_id"])["entity_id"].astype(str).tolist()
    matches, cands, dropped = {}, {}, 0
    paths = (sorted(run_dir.glob("final_test_*.parquet"))
             or sorted(run_dir.glob("stage1_test_*.parquet"))
             or sorted(run_dir.glob("test_preds_*.parquet")))
    for path in paths:
        df = pd.read_parquet(path)
        df["s1_id"] = df["s1_id"].astype("category")
        m = decide_frame(df, info)
        if resolve:
            m, d = one_to_one(m, df[["s1_id", "cand_id", "p"]].assign(
                s1_id=lambda x: x["s1_id"].astype(str)))
            dropped += d
        matches.update(m)
        g = df.groupby("s1_id", observed=True)["cand_id"].agg(lambda s: ",".join(sorted(s)))
        cands.update(g.to_dict())
        print(f"[submission] {path.stem}: {len(m):,} entities with matches", flush=True)
        del df
    with open(OUT_DIR / "matching_results.tsv", "w", encoding="utf-8", newline="\n") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s in test_ids:
            f.write(s + "\t" + ",".join(sorted(matches.get(s, ()))) + "\n")
    with open(OUT_DIR / "candidate_pairs.tsv", "w", encoding="utf-8", newline="\n") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s in test_ids:
            f.write(s + "\t" + cands.get(s, "") + "\n")
    n = sum(len(v) for v in matches.values())
    stats = {"test_matches": n, "test_s1_with_match": len(matches), "test_s1": len(test_ids),
             "one_to_one_dropped": dropped,
             "test_cands_per_s1": sum(c.count(",") + 1 for c in cands.values() if c) / len(test_ids)}
    info["submission"] = stats
    (run_dir / "info.json").write_text(json.dumps(info, indent=1, default=str))
    print(json.dumps(stats, indent=1), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--no_resolve", action="store_true")
    a = ap.parse_args()
    main(a.run, not a.no_resolve)

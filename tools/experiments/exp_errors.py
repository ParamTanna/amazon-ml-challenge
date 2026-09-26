"""Error analysis on a run's validation predictions."""
import sys, json
sys.path.insert(0, r"D:\Amazon ML Challenge\code\business_entity_resolution\src")
import numpy as np, pandas as pd, pyarrow.dataset as ds, pyarrow as pa
from decision import expected_f05_decide
from model import load_truth
from preprocess import norm_path

run = sys.argv[1] if len(sys.argv) > 1 else "v0"
rd = rf"D:\Amazon ML Challenge\work\runs\{run}"
info = json.load(open(rd + r"\info.json"))
va = pd.read_parquet(rd + r"\valid_preds.parquet")
truth = load_truth()
pred = expected_f05_decide(va, **info["expected_params"])
ids = set(va.s1_id)
cols = ["entity_id", "name_raw", "addr_raw", "country"]
def load(src, idset):
    return ds.dataset(norm_path("train", src)).to_table(columns=cols, filter=ds.field("entity_id").isin(pa.array(list(idset)))).to_pandas().set_index("entity_id")
s1 = load("source1", ids)
va["pred"] = [c in pred.get(s, ()) for s, c in zip(va.s1_id, va.cand_id)]
fp = va[va.pred & ~va.label]; fn = va[~va.pred & va.label]
cand_ids = set(fp.cand_id) | set(fn.cand_id)
cand = pd.concat([load("source2", cand_ids), load("source3", cand_ids)])
print(f"FP pairs {len(fp)}  FN pairs (in candidates) {len(fn)}  pred pairs {va.pred.sum()}  true pairs {sum(len(truth[s]) for s in ids)}")
print("FP by country", s1.loc[fp.s1_id, "country"].value_counts().to_dict(), " FN by country", s1.loc[fn.s1_id, "country"].value_counts().to_dict())
print("FP where S1 is singleton:", sum(1 for s in fp.s1_id if not truth[s]))
rng = np.random.default_rng(0)
for name, df in (("FALSE POSITIVES (wrong merges)", fp), ("FALSE NEGATIVES (missed, in candidates)", fn)):
    print("\n=====", name)
    for i in rng.choice(len(df), min(25, len(df)), replace=False):
        r = df.iloc[i]; a = s1.loc[r.s1_id]; b = cand.loc[r.cand_id]
        others = [cand_id for cand_id in truth[r.s1_id]]
        print(f"p={r.p:.2f} S1: {a.name_raw} | {a.addr_raw}\n        C: {b.name_raw} | {b.addr_raw}   (S1 has {len(truth[r.s1_id])} true)")

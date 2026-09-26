"""End-to-end driver: train the matcher on the training split, predict the test split.

Prerequisites (run once):   python preprocess.py
                            python run_blocking.py --split train
                            python run_blocking.py --split test
Then:                       python run_pipeline.py --run v0

Outputs: OUT_DIR/matching_results.tsv, OUT_DIR/candidate_pairs.tsv and a run folder
WORK_DIR/runs/<run>/ with the model, decision settings, validation scores, cached
training features and test probabilities.

Heavy libraries are imported inside main() so the feature worker processes (spawned on
Windows) stay small.
"""

import argparse
import json
import os
import time
from multiprocessing import Pool


def write_id_lists(path, s1_ids, mapping, header):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\t".join(header) + "\n")
        for s in s1_ids:
            f.write(s + "\t" + ",".join(sorted(mapping.get(s, ()))) + "\n")


def build_labelled(id_sets, truth, pool, k_use, run_dir):
    """Featurised, labelled training pairs for each named set of Source-1 ids (cached).

    Each country's pair table is loaded once and split into the requested sets.
    """
    import pandas as pd
    from model import RecordStore, add_labels, featurize, load_pairs, split_countries
    caches = {name: run_dir / f"{name}_pairs.parquet" for name in id_sets}
    if all(c.exists() for c in caches.values()):
        return {name: pd.read_parquet(c) for name, c in caches.items()}
    parts = {name: [] for name in id_sets}
    for ctry in split_countries("train"):
        p_all = load_pairs("train", ctry)
        p_all = p_all[p_all["fwd_rank"] < k_use]
        store = RecordStore("train", ctry)
        for name, ids in id_sets.items():
            p = p_all[p_all["s1_id"].isin(ids)].reset_index(drop=True)
            print(f"  [{name}] {ctry}: {len(p):,} pairs", flush=True)
            if not len(p):
                continue
            p = add_labels(featurize(p, store, pool), truth)
            p["s1_id"] = p["s1_id"].astype(str)
            p["cand_id"] = p["cand_id"].astype(str)
            p["country"] = ctry
            parts[name].append(p.drop(columns=["s1_key", "cand_key"]))
        del store, p_all
    out = {}
    for name, lst in parts.items():
        out[name] = pd.concat(lst, ignore_index=True)
        out[name].to_parquet(caches[name], index=False)
    return out


def main(run, n_train, n_val, seed, chunk_s1, k_use):
    os.environ["OPENBLAS_NUM_THREADS"] = "1"   # inherited by the spawned workers
    import numpy as np
    import pandas as pd
    from config import N_JOBS, OUT_DIR, WORK_DIR
    from decision import expected_f05_decide, tune_expected
    from evaluate import breakdown, f05_macro
    from model import (FEATURES, RecordStore, decide, featurize, load_pairs, load_truth,
                       split_countries, train_model, tune_threshold)
    import lightgbm as lgb

    run_dir = WORK_DIR / "runs" / run
    run_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with Pool(N_JOBS) as pool:
        # ---------------- training ----------------
        from splits import partition
        info_path = run_dir / "info.json"
        reuse = (run_dir / "model.txt").exists() and info_path.exists() and             "expected_params" in json.loads(info_path.read_text())
        truth = load_truth()
        parts = partition(truth.keys(), seed)
        tr_ids, va_ids = parts["A"], parts["V"]
        if reuse:
            info = json.loads(info_path.read_text())
            booster = lgb.Booster(model_file=str(run_dir / "model.txt"))
            exp_params, thr = info["expected_params"], info["threshold"]
            use_expected = info["decision"] == "expected"
            print("[stage1] reusing trained model and tuned decision", flush=True)
        else:
            sets = build_labelled({"train": tr_ids, "valid": va_ids}, truth, pool, k_use,
                                  run_dir)
            tr, va = sets["train"], sets["valid"]
            del sets
            print(f"train pairs {len(tr):,} (pos {tr['label'].mean():.3f}), valid pairs "
                  f"{len(va):,}  [{time.time() - t0:.0f}s]", flush=True)
            model_path = run_dir / "model.txt"
            if model_path.exists():
                booster = lgb.Booster(model_file=str(model_path))
            else:
                booster = train_model(tr, va)
                booster.save_model(str(model_path))
            del tr
            va["p"] = booster.predict(va[FEATURES].values.astype(np.float32))
            truth_va = {k: truth[k] for k in va_ids}
            thr, s_thr = tune_threshold(va, truth_va)
            exp_params, s_exp = tune_expected(va, truth_va)
            use_expected = s_exp > s_thr
            pred_va = (expected_f05_decide(va, **exp_params) if use_expected
                       else decide(va, threshold=thr))
            country_of = dict(zip(va["s1_id"], va["country"]))
            cand_va = va.groupby("s1_id")["cand_id"].agg(set).to_dict()
            n_true = sum(len(t) for t in truth_va.values())
            n_cov = sum(len(t & cand_va.get(k, set())) for k, t in truth_va.items())
            info = {
                "valid_f05_threshold": s_thr, "threshold": float(thr),
                "valid_f05_expected": s_exp, "expected_params": exp_params,
                "decision": "expected" if use_expected else "threshold",
                "by_country": breakdown(pred_va, truth_va, country_of),
                "by_singleton": breakdown(pred_va, truth_va, {k: ("singleton" if not truth[k]
                                                                  else "matched") for k in va_ids}),
                "blocking_pair_recall": n_cov / max(1, n_true),
                "cands_per_s1": len(va) / len(va_ids), "k_use": k_use,
                "best_iter": booster.best_iteration,
            }
            print(json.dumps(info, indent=1, default=str), flush=True)
            (run_dir / "info.json").write_text(json.dumps(info, indent=1, default=str))
            va[["s1_id", "cand_id", "p", "label"]].to_parquet(run_dir / "valid_preds.parquet")
            imp = pd.Series(booster.feature_importance("gain"), index=FEATURES).sort_values()
            print(imp.tail(25).to_string(), flush=True)
            del va

        # ---------------- predict every Source-1 record of both splits ----------------
        # (all training records too: the final model's competition features need the
        #  first-stage probability of every entity that retrieved a candidate)
        # features of partitions D, V (train) and of the test split are saved so the final
        # stage can reuse them instead of recomputing string similarities
        feat_dir = run_dir / "feats"
        feat_dir.mkdir(exist_ok=True)
        save_ids = parts["D"] | parts["V"]
        save_cols = FEATURES + ["k_c_num", "k_s1_num", "k_c_name", "k_s1_name"]
        # test first (the submission only needs it); on the training split only the
        # partitions the final stage uses (D: training, V: validation) are scored
        for split in ("test", "train"):
            for ctry in split_countries(split):
                out_path = run_dir / f"stage1_{split}_{ctry}.parquet"
                if out_path.exists():
                    continue
                store = RecordStore(split, ctry)
                p_all = load_pairs(split, ctry)
                if k_use < 999:
                    p_all = p_all[p_all["fwd_rank"] < k_use].reset_index(drop=True)
                keep_cols = [c for c in p_all.columns if c in FEATURES or c in
                             ("s1_id", "cand_id", "s1_key", "cand_key")]
                p_all = p_all[keep_cols]
                if split == "train":
                    cats = p_all["s1_id"].cat.categories
                    in_dv = np.asarray(cats.isin(save_ids))[p_all["s1_id"].cat.codes.to_numpy()]
                    p_all = p_all[in_dv].reset_index(drop=True)
                s1_list = p_all["s1_id"].cat.codes.unique()
                parts = []
                for i in range(0, len(s1_list), chunk_s1):
                    part = p_all[p_all["s1_id"].cat.codes.isin(s1_list[i:i + chunk_s1])
                                 ].reset_index(drop=True)
                    featurize(part, store, pool)
                    part["p"] = booster.predict(part[FEATURES].values.astype(np.float32))
                    if split == "test":
                        sel = np.ones(len(part), dtype=bool)
                    else:
                        cats = part["s1_id"].cat.categories
                        sel = np.asarray(cats.isin(save_ids))[part["s1_id"].cat.codes.to_numpy()]
                    if sel.any():
                        fs = part.loc[sel, save_cols].astype(np.float32)
                        for kc in ("k_c_num", "k_s1_num", "k_c_name", "k_s1_name"):
                            fs[kc] = part.loc[sel, kc].to_numpy(np.int64)
                        fs.insert(0, "s1_id", part.loc[sel, "s1_id"].astype(str).to_numpy())
                        fs.insert(1, "cand_id", part.loc[sel, "cand_id"].astype(str).to_numpy())
                        fs.to_parquet(feat_dir / f"{split}_{ctry}_{i // chunk_s1:03d}.parquet",
                                      index=False)
                    parts.append(pd.DataFrame({"s1_id": part["s1_id"].astype(str),
                                               "cand_id": part["cand_id"].astype(str),
                                               "p": part["p"].astype(np.float32)}))
                    print(f"  [{split}] {ctry} chunk {i // chunk_s1}: {len(part):,} pairs "
                          f"[{time.time() - t0:.0f}s]", flush=True)
                cp = pd.concat(parts, ignore_index=True)
                cp.to_parquet(out_path, index=False)
                del store, p_all, cp, parts
    preds = None
    del preds
    info["runtime_s"] = time.time() - t0
    (run_dir / "info.json").write_text(json.dumps(info, indent=1, default=str))
    # outputs are written country by country (memory-light) by make_submission
    import make_submission
    make_submission.main(run, resolve=True)
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="v0")
    ap.add_argument("--n_train", type=int, default=200000)
    ap.add_argument("--n_val", type=int, default=50000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chunk_s1", type=int, default=100000)
    ap.add_argument("--k_use", type=int, default=999, help="max main-view forward rank kept (999 = all trimmed candidates)")
    a = ap.parse_args()
    main(a.run, a.n_train, a.n_val, a.seed, a.chunk_s1, a.k_use)

"""Learn native-script -> English word dictionaries from the training matches.

Indian records in Sources 2/3 often spell the business name (and the state) in a native
script (Hindi, Tamil, Telugu, Kannada, Bengali, ...).  The generator writes each English
word the same way every time, so aligning the words of matched training pairs yields an
almost exact lookup table:  मार्केटिंग -> marketing,  प्रा -> pvt,  ಕರ್ನಾಟಕ -> ka.

Names: when the Source-1 name and the native-script candidate name have the same number
of words, they are aligned position by position and each native word votes for the
English word it lines up with.
Addresses: each native word in a candidate address votes for the state code found in
the Source-1 address.
A mapping is kept when it has enough votes and a clear majority.  Words never seen in
training fall back to rule-based transliteration (normalize.translit_indic).

Output: WORK_DIR/cache/indic_dict.json  {"name": {...}, "addr": {...}}
"""

import json
import re
from collections import Counter, defaultdict

import pandas as pd

from config import CACHE_DIR
from normalize import _HAS_INDIC, _IN_STATES, _INDIC_RUN, normalize_address

DICT_PATH = CACHE_DIR / "indic_dict.json"
_TOKEN = re.compile(r"[ऀ-ൿ][ऀ-ൿ‌‍]*|[A-Za-z0-9]+")
_LATIN = re.compile(r"[A-Za-z0-9]+")
_STATE_CODES = set(_IN_STATES.values())


def _vote(counts, min_count, min_share):
    out = {}
    for w, c in counts.items():
        e, n = c.most_common(1)[0]
        tot = sum(c.values())
        if n >= min_count and n / tot >= min_share:
            out[w] = e
    return out


def learn(min_count=3, min_share=0.6):
    from preprocess import raw_table, read_tsv
    from config import DATA_DIR
    gt_path = CACHE_DIR / "train_ground_truth.parquet"
    if not gt_path.exists():
        read_tsv(DATA_DIR / "train" / "train_ground_truth.tsv").to_parquet(gt_path, index=False)
    gt = pd.read_parquet(gt_path)
    s1 = raw_table("train", "source1").set_index("entity_id")
    cand = pd.concat([raw_table("train", "source2"), raw_table("train", "source3")])
    cand = cand[cand["country"] == "India"]
    ind_name = cand["business_name"].str.contains(_HAS_INDIC.pattern, regex=True)
    ind_addr = cand["business_address"].str.contains(_HAS_INDIC.pattern, regex=True)
    cand = cand[ind_name | ind_addr].set_index("entity_id")
    pairs = gt.assign(m=gt["matched_entity_ids"].str.split(",")).explode("m")
    pairs = pairs[pairs["m"].isin(cand.index)]
    print(f"[indic] {len(pairs):,} matched pairs with native-script text", flush=True)

    name_votes, addr_votes = defaultdict(Counter), defaultdict(Counter)
    for s1_id, cid in zip(pairs["source1_entity_id"].values, pairs["m"].values):
        c_name, c_addr = cand.at[cid, "business_name"], cand.at[cid, "business_address"]
        if _HAS_INDIC.search(c_name):
            en = [t.lower() for t in _LATIN.findall(s1.at[s1_id, "business_name"])]
            nw = _TOKEN.findall(c_name)
            if len(en) == len(nw):
                for w, e in zip(nw, en):
                    if _HAS_INDIC.search(w):
                        name_votes[w][e] += 1
        if _HAS_INDIC.search(c_addr or ""):
            s1_toks, _ = normalize_address(s1.at[s1_id, "business_address"], "India")
            codes = [t for t in s1_toks if t in _STATE_CODES]
            if len(set(codes)) == 1:
                for w in _INDIC_RUN.findall(c_addr):
                    addr_votes[w][codes[0]] += 1
    d = {"name": _vote(name_votes, min_count, min_share),
         "addr": _vote(addr_votes, min_count, min_share)}
    print(f"[indic] learned {len(d['name']):,} name words (of {len(name_votes):,} seen), "
          f"{len(d['addr']):,} address words (of {len(addr_votes):,})", flush=True)
    DICT_PATH.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
    return d


if __name__ == "__main__":
    learn()

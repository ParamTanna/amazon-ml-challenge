"""Stage 3: pairwise similarity features for (Source-1 record, candidate) pairs.

All features are language-neutral string similarities, so they transfer to countries
that never appear in training (France in the test set).  Candidate addresses in this
data are usually a *subset* of the Source-1 address, so several features are
asymmetric containment scores (share of the candidate's tokens found in Source 1).
"""

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler, Levenshtein

from normalize import _LEGAL_VALUES

PAIR_FEATURES = [
    # name
    "n_ratio", "n_tset", "n_tsort", "n_partial", "n_full_ratio", "n_concat_jw",
    "n_concat_partial", "n_cont_c", "n_cont_s1", "n_jacc", "n_fuzzy_mean", "n_fuzzy_min",
    "n_len_s1", "n_len_c", "legal_conflict", "legal_shared", "c_is_domain", "c_has_alias",
    "c_name_indic",
    # address
    "a_c_empty", "a_cont_c", "a_cont_s1", "a_tset", "a_tsort", "a_partial", "a_fuzzy_mean",
    "a_len_s1", "a_len_c", "a_c_indic",
    # numbers
    "num_c_count", "num_cont_c", "num_c_extra", "num_first_in_c", "num_first_best_lev",
    "num_any_shared",
    # unmatched words / numeric closeness (decoys with shifted numbers or swapped words)
    "n_c_unmatched", "n_s1_unmatched", "a_c_unmatched", "num_min_absdiff", "num_prefix_ratio",
]

_JW = JaroWinkler.normalized_similarity


def _unmatched(cand_tokens, ref_tokens, thr=0.85):
    """How many candidate tokens have no reference token with Jaro-Winkler >= thr."""
    if not cand_tokens:
        return 0.0
    if not ref_tokens:
        return float(len(cand_tokens))
    return float(sum(process.extractOne(t, ref_tokens, scorer=_JW)[1] < thr
                     for t in cand_tokens))


def _prefix_ratio(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n / max(len(a), len(b))


def _fuzzy_containment(cand_tokens, ref_tokens):
    """Mean / min over candidate tokens of the best Jaro-Winkler match in the reference."""
    if not cand_tokens or not ref_tokens:
        return -1.0, -1.0
    best = [process.extractOne(t, ref_tokens, scorer=_JW)[1] for t in cand_tokens]
    return sum(best) / len(best), min(best)


def pair_features(a, b):
    """Feature vector for Source-1 record ``a`` and candidate ``b``.

    Each record is a tuple (name_core, name_tok, addr_tok, addr_nums, is_domain,
    has_alias, name_indic, addr_indic) of normalised fields (see preprocess.py).
    """
    a_core, a_tok, a_addr, a_nums = a[0], a[1], a[2], a[3]
    b_core, b_tok, b_addr, b_nums = b[0], b[1], b[2], b[3]
    ac, bc = a_core.split(), b_core.split()
    sa, sb = set(ac), set(bc)
    inter = len(sa & sb)
    ca, cb = "".join(ac), "".join(bc)
    f = [
        fuzz.ratio(a_core, b_core),
        fuzz.token_set_ratio(a_core, b_core),
        fuzz.token_sort_ratio(a_core, b_core),
        fuzz.partial_ratio(a_core, b_core),
        fuzz.ratio(a_tok, b_tok),
        _JW(ca, cb),
        fuzz.partial_ratio(ca, cb),
        inter / len(sb) if sb else -1.0,
        inter / len(sa) if sa else -1.0,
        inter / len(sa | sb) if (sa or sb) else -1.0,
    ]
    f.extend(_fuzzy_containment(bc, ac))
    la = set(a_tok.split()) & _LEGAL_VALUES
    lb = set(b_tok.split()) & _LEGAL_VALUES
    f += [len(ac), len(bc), float(bool(la and lb and not (la & lb))), len(la & lb),
          float(b[4]), float(b[5]), float(b[6])]

    aa, ba = a_addr.split(), b_addr.split()
    saa, sba = set(aa), set(ba)
    ai = len(saa & sba)
    if ba:
        f += [0.0, ai / len(sba), ai / len(saa) if saa else -1.0,
              fuzz.token_set_ratio(a_addr, b_addr), fuzz.token_sort_ratio(a_addr, b_addr),
              fuzz.partial_ratio(a_addr, b_addr),
              _fuzzy_containment([t for t in ba if not t.isdigit()],
                                 [t for t in aa if not t.isdigit()])[0]]
    else:
        f += [1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]
    f += [len(aa), len(ba), float(b[7])]

    na, nb = a_nums.split(), b_nums.split()
    sna, snb = set(na), set(nb)
    if nb:
        f += [len(nb), len(sna & snb) / len(snb), len(snb - sna)]
    else:
        f += [0.0, -1.0, 0.0]
    if na and nb:
        f += [float(na[0] in snb),
              max(Levenshtein.normalized_similarity(na[0], x) for x in nb),
              float(bool(sna & snb))]
    else:
        f += [-1.0, -1.0, -1.0]

    aw = [t for t in aa if not t.isdigit()]
    bw = [t for t in ba if not t.isdigit()]
    f += [_unmatched(bc, ac), _unmatched(ac, bc), _unmatched(bw, aw)]
    if na and nb:
        a0 = int(na[0][:9])
        f += [float(np.log1p(min(abs(a0 - int(x[:9])) for x in nb))),
              max(_prefix_ratio(na[0], x) for x in nb)]
    else:
        f += [-1.0, -1.0]
    return f


def _chunk(args):
    lefts, rights = args
    return np.asarray([pair_features(a, b) for a, b in zip(lefts, rights)], dtype=np.float32)


RECORD_COLS = ["name_core", "name_tok", "addr_tok", "addr_nums", "is_domain", "has_alias",
               "name_indic", "addr_indic"]


def records(df, rows):
    """Tuples of the normalised fields used by pair_features, for the given row indices."""
    sub = df.iloc[rows] if not isinstance(rows, slice) else df[rows]
    cols = [sub[c].astype(str).values if c in RECORD_COLS[:4] else sub[c].values
            for c in RECORD_COLS]
    return list(zip(*cols))


def compute_pair_features(left, right, pool, chunk=20000):
    """left/right: equal-length lists of record tuples. Returns float32 [n, F]."""
    jobs = [(left[i:i + chunk], right[i:i + chunk]) for i in range(0, len(left), chunk)]
    parts = list(pool.imap(_chunk, jobs))
    if not parts:
        return np.zeros((0, len(PAIR_FEATURES)), dtype=np.float32)
    return np.vstack(parts)


GROUP_FEATURES = ["g_s1_max", "g_gap_s1", "g_rank_s1", "g_rank_src", "g_n_cands",
                  "g_n_strong", "g_c_max", "g_gap_c", "g_c_n_s1", "g_c_rank"]


def add_group_features(pairs, score_col="score"):
    """Context features comparing a pair with the other candidates of the same entities.

    Per Source-1 record: best score, gap to best, rank overall and within source,
    number of candidates, number of strong candidates.  Per candidate: best score over
    all Source-1 records that retrieved it, gap to that, how many Source-1 records
    retrieved it, and this pair's rank among them (one-to-one assignment signal).
    """
    s = pairs[score_col].astype(np.float32)
    g1 = s.groupby(pairs["s1_key"])
    pairs["g_s1_max"] = g1.transform("max").astype(np.float32)
    pairs["g_gap_s1"] = (s - pairs["g_s1_max"]).astype(np.float32)
    pairs["g_rank_s1"] = g1.rank(ascending=False, method="first").astype(np.float32)
    pairs["g_rank_src"] = s.groupby([pairs["s1_key"], pairs["cand_src"]]).rank(
        ascending=False, method="first").astype(np.float32)
    pairs["g_n_cands"] = g1.transform("size").astype(np.float32)
    strong = (s > 0.5).astype(np.float32)
    pairs["g_n_strong"] = strong.groupby(pairs["s1_key"]).transform("sum").astype(np.float32)
    gc = s.groupby(pairs["cand_key"])
    pairs["g_c_max"] = gc.transform("max").astype(np.float32)
    pairs["g_gap_c"] = (s - pairs["g_c_max"]).astype(np.float32)
    pairs["g_c_n_s1"] = gc.transform("size").astype(np.float32)
    pairs["g_c_rank"] = gc.rank(ascending=False, method="first").astype(np.float32)
    return pairs

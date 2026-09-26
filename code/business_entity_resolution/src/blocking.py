"""Stage 2: candidate generation (blocking) - a union of complementary retrieval views.

Every record is embedded as a sparse, L2-normalised TF-IDF vector in each *view*:

  main   word-level name + address features
           n:<word>      cleaned core-name word          c:<concat>  core name, no spaces
           q:<prefix4>   typo-tolerant word prefix       a:<word>    address word / number
           b:<w1>_<w2>   consecutive address words       x:<num>_<w> number paired with any
                                                                     address word (survives
                                                                     truncated addresses)
  ngram  character 4-grams of the core name (spaces removed): robust to typos,
         transliteration variants, website-style names, word re-ordering and records with
         empty addresses; needs no language-specific rules, so it generalises to unseen
         countries.

Features seen in only one record or in more than ``max_df`` records are dropped (they
cannot create a pair / carry almost no signal and dominate compute), so rare features
drive the cosine.  In each view we keep, for each Source-1 record, its top-k Source-2 and
Source-3 records (forward search) and, for each Source-2/3 record, its top Source-1 records
(reverse search: every Source-2/3 record belongs to at most one entity, so this rescues
matches crowded out of a forward list by same-name decoys).  The candidate set is the
union over views and directions; every pair then gets its cosine in *every* view plus its
forward / reverse rank in each view, which the ranking model uses later.

Feature hashing runs in worker processes; PYTHONHASHSEED is fixed so all workers hash
identically.
"""

import os
import time
from array import array

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sparse_dot_topn import sp_matmul_topn

from config import N_JOBS
from normalize import _LEGAL_VALUES as _LEGAL

VIEWS = {
    "main": {"max_df": 2000, "k_fwd": 10, "k_rev": 3},
    "ngram": {"max_df": 5000, "k_fwd": 10, "k_rev": 3},
}


def main_features(name_core, addr_tok, name_tok=None):
    n = name_core.split() if name_core else []
    a = addr_tok.split() if addr_tok else []
    feats = ["n:" + t for t in n]
    if n:
        feats.append("c:" + "".join(n))
    if name_tok:
        # name incl. filler words, without legal forms: matches website-style names
        full = [t for t in name_tok.split() if t not in _LEGAL]
        if full and full != n:
            feats.append("c:" + "".join(full))
    feats += ["q:" + t[:4] for t in n if len(t) >= 5]
    feats += ["a:" + t for t in a]
    feats += ["b:" + a[i] + "_" + a[i + 1] for i in range(len(a) - 1)]
    nums = [t for t in a if t.isdigit()][:2]
    words = [t for t in a if not t.isdigit()]
    feats += ["x:" + x + "_" + w for x in nums for w in words]
    return feats


def _char_grams(name_core, n):
    s = "".join(name_core.split()) if name_core else ""
    if len(s) < 2:
        return []
    s = "^" + s + "$"
    return ["g:" + s[i:i + n] for i in range(max(1, len(s) - n + 1))]


def ngram_features(name_core, addr_tok=None, name_tok=None):
    return _char_grams(name_core, 4)


def ngram3_features(name_core, addr_tok=None, name_tok=None):
    return _char_grams(name_core, 3)


VIEW_FEATURES = {"main": main_features, "ngram": ngram_features, "ngram3": ngram3_features}
# backwards-compatible name used by older experiment scripts
block_features = main_features


def _hash_rows(args):
    """Worker: hashed feature ids for a chunk of (name_core, addr_tok, name_tok) rows."""
    view, rows = args
    fn = VIEW_FEATURES[view]
    lengths = np.empty(len(rows), dtype=np.int32)
    buf = array("q")
    for i, r in enumerate(rows):
        f = set(fn(*r))
        lengths[i] = len(f)
        buf.extend(hash(t) for t in f)
    return lengths, np.frombuffer(buf, dtype=np.int64).copy()


def make_pool():
    """Worker pool whose processes all use the same string-hash seed."""
    from multiprocessing import Pool
    os.environ["PYTHONHASHSEED"] = "0"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    return Pool(N_JOBS)


def hash_features(rows, view, pool=None, chunk=50000):
    """Per-record feature counts and the concatenated feature hashes for one view."""
    jobs = [(view, rows[i:i + chunk]) for i in range(0, len(rows), chunk)]
    res = pool.map(_hash_rows, jobs) if pool is not None else [_hash_rows(j) for j in jobs]
    lengths = np.concatenate([r[0] for r in res]).astype(np.int64)
    hashes = np.concatenate([r[1] for r in res])
    return lengths, hashes


def tfidf_from_hashes(lengths, hashes, max_df, return_norms=False):
    """L2-normalised binary TF-IDF CSR matrix built directly from feature hashes.

    Memory-lean: hash-table factorisation (no full sort), int32 indices, no COO copy.
    """
    n_docs = len(lengths)
    codes, uniq = pd.factorize(hashes, sort=False)
    del uniq
    counts = np.bincount(codes)
    keep = (counts >= 2) & (counts <= max_df)
    n_cols = int(keep.sum())
    col_of = np.full(len(counts), -1, dtype=np.int32)
    col_of[keep] = np.arange(n_cols, dtype=np.int32)
    idf = np.log(n_docs / counts[keep]).astype(np.float32)
    del counts
    cols = col_of[codes]
    del codes, col_of
    doc = np.repeat(np.arange(n_docs, dtype=np.int32), lengths)
    m = cols >= 0
    cols, doc = cols[m], doc[m]
    del m
    data = idf[cols]
    norms = np.sqrt(np.bincount(doc, weights=data.astype(np.float64) ** 2, minlength=n_docs))
    norms[norms == 0] = 1.0
    data /= norms[doc].astype(np.float32)
    indptr = np.zeros(n_docs + 1, dtype=np.int64)
    np.cumsum(np.bincount(doc, minlength=n_docs), out=indptr[1:])
    del doc
    mat = sp.csr_matrix((data, cols, indptr), shape=(n_docs, n_cols))
    mat.sum_duplicates()   # 64-bit hash collisions are astronomically rare, but stay canonical
    return (mat, norms.astype(np.float32)) if return_norms else mat


def tfidf_matrix(rows, view, max_df, pool=None, chunk=50000, return_norms=False):
    """L2-normalised binary TF-IDF matrix for one view (rows aligned with the input)."""
    lengths, hashes = hash_features(rows, view, pool, chunk)
    return tfidf_from_hashes(lengths, hashes, max_df, return_norms)


def topk(a, b, k, threshold=0.0):
    """Top-k columns of a @ b.T per row, as (row, col, score, rank) arrays."""
    c = sp_matmul_topn(a, b, top_n=k, threshold=threshold, sort=True, n_threads=N_JOBS)
    c = c.tocsr()
    rows = np.repeat(np.arange(c.shape[0]), np.diff(c.indptr))
    starts = np.repeat(c.indptr[:-1], np.diff(c.indptr))
    rank = np.arange(len(c.indices)) - starts
    return rows, c.indices.astype(np.int64), c.data.astype(np.float32), rank.astype(np.int16)


def pair_cosine(ma, mb, ia, ib, chunk=1_000_000):
    """Cosine of rows ia of ``ma`` with rows ib of ``mb`` (vectorised, chunked)."""
    out = np.empty(len(ia), dtype=np.float32)
    for i in range(0, len(ia), chunk):
        a = ma[ia[i:i + chunk]]
        b = mb[ib[i:i + chunk]]
        out[i:i + chunk] = np.asarray(a.multiply(b).sum(axis=1)).ravel()
    return out


_SHIFT = np.int64(1) << np.int64(34)   # bits 32-33 hold the source (2 or 3)


def _key(s1_row, src, cand_row):
    return s1_row.astype(np.int64) * _SHIFT + np.int64(src) * (np.int64(1) << 32) + cand_row


def candidates_for_country(s1, s2, s3, query_mask=None, views=None, pool=None, verbose=True,
                           **_ignored):
    """Candidate pairs for one country (union over views and search directions).

    s1, s2, s3   normalised frames of that country (all rows: they define the IDF)
    query_mask   boolean array over s1 rows to generate candidates for (default: all)
    Returns a DataFrame with s1_row, cand_src (2/3), cand_row and, per view v,
    v_score (cosine), v_fwd (forward rank or -1), v_rev (reverse rank or -1); plus
    score / fwd_rank / rev_rank = the main view's values.
    """
    views = views or VIEWS
    n1, n2 = len(s1), len(s2)
    cols = [pd.concat([s1[c], s2[c], s3[c]]).astype(str).tolist()
            for c in ("name_core", "addr_tok", "name_tok")]
    rows = list(zip(*cols))
    del cols
    q_rows = np.arange(n1) if query_mask is None else np.flatnonzero(query_mask)
    hashed = {v: hash_features(rows, v, pool) for v in views}   # then free the text
    del rows
    mats, found = {}, {}
    for v, cfg in views.items():
        t0 = time.time()
        lengths, hashes = hashed.pop(v)
        mat, norms = tfidf_from_hashes(lengths, hashes, cfg["max_df"], return_norms=True)
        del lengths, hashes
        m1, m2, m3 = mat[:n1], mat[n1:n1 + n2], mat[n1 + n2:]
        # reverse search ranks Source-1 records by how much of the candidate's evidence they
        # cover (un-normalised S1 vectors): the true owner has the most complete address, and
        # plain cosine would penalise it for that length.
        m1_rev = (sp.diags(norms[:n1]) @ m1).tocsr() if cfg.get("rev_raw", True) else m1
        mats[v] = (m1, m2, m3)
        keys = []
        mq = m1[q_rows]
        for src, mb in ((2, m2), (3, m3)):
            r, c, s, rk = topk(mq, mb, cfg["k_fwd"])
            keys.append((_key(q_rows[r], src, c), rk, "fwd"))
            if cfg["k_rev"]:
                r2, c2, s2_, rk2 = topk(mb, m1_rev, cfg["k_rev"])
                if query_mask is not None:
                    keep = query_mask[c2]
                    r2, c2, rk2 = r2[keep], c2[keep], rk2[keep]
                keys.append((_key(c2, src, r2), rk2, "rev"))
        found[v] = keys
        if verbose:
            print(f"    [{v}] tfidf {mat.shape} nnz={mat.nnz:,}, search done in "
                  f"{time.time() - t0:.0f}s", flush=True)
        del mat, mq, m1_rev
    # union of all candidate keys
    all_keys = np.unique(np.concatenate([k for v in found for k, _, _ in found[v]]))
    s1_row = (all_keys // _SHIFT).astype(np.int64)
    rem = all_keys % _SHIFT
    cand_src = (rem >> 32).astype(np.int8)
    cand_row = (rem & ((np.int64(1) << 32) - 1)).astype(np.int64)
    out = pd.DataFrame({"s1_row": s1_row, "cand_src": cand_src, "cand_row": cand_row})
    for v in views:
        m1, m2, m3 = mats[v]
        score = np.zeros(len(all_keys), dtype=np.float32)
        is2 = cand_src == 2
        score[is2] = pair_cosine(m1, m2, s1_row[is2], cand_row[is2])
        score[~is2] = pair_cosine(m1, m3, s1_row[~is2], cand_row[~is2])
        out[f"{v}_score"] = score
        for direction in ("fwd", "rev"):
            rank = np.full(len(all_keys), -1, dtype=np.int16)
            for k, rk, d in found[v]:
                if d != direction:
                    continue
                pos = np.searchsorted(all_keys, k)
                rank[pos] = np.where(rank[pos] < 0, rk, np.minimum(rank[pos], rk))
            out[f"{v}_{direction}"] = rank
    out["score"] = out["main_score"]
    out["fwd_rank"] = out["main_fwd"]
    out["rev_rank"] = out["main_rev"]
    if verbose:
        print(f"    union: {len(out):,} pairs ({len(out) / max(1, len(q_rows)):.1f} per S1)",
              flush=True)
    return out

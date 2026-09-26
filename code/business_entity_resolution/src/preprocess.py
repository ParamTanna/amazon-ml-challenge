"""Stage 1: read the raw TSVs, normalise every name and address, cache as parquet.

Output: WORK_DIR/cache/{split}_{source}_norm.parquet with columns
    entity_id, country, name_raw, addr_raw,
    name_tok, name_core   space-joined cleaned name tokens / core tokens
    is_domain, has_alias, name_indic
    addr_tok, addr_nums   space-joined cleaned address tokens / numeric tokens
    addr_indic
"""

import argparse
import time
from multiprocessing import Pool

import pandas as pd

from config import CACHE_DIR, DATA_DIR, N_JOBS, SOURCES, SPLITS
from normalize import has_indic, normalize_address, normalize_name

_CHUNK = 20000


def read_tsv(path):
    """Read a challenge TSV with every column as string and empty cells kept as ''."""
    return pd.read_csv(path, sep="\t", dtype="string[pyarrow]", engine="pyarrow",
                       keep_default_na=False)


def raw_table(split, source):
    """Load one raw source table, using a parquet copy after the first read."""
    pq = CACHE_DIR / f"{split}_{source}.parquet"
    if pq.exists():
        return pd.read_parquet(pq)
    df = read_tsv(DATA_DIR / split / f"{split}_{source}.tsv")
    df.to_parquet(pq, index=False)
    return df


def _normalise_rows(rows):
    """Worker: normalise a list of (name, address, country) tuples."""
    out = []
    for name, addr, country in rows:
        toks, core, flags = normalize_name(name)
        atoks, nums = normalize_address(addr, country)
        out.append((" ".join(toks), " ".join(core), flags["is_domain"], flags["has_alias"],
                    has_indic(name), " ".join(atoks), " ".join(nums), has_indic(addr)))
    return out


def normalise_table(df, pool):
    rows = list(zip(df["business_name"].astype(str), df["business_address"].astype(str),
                    df["country"].astype(str)))
    chunks = [rows[i:i + _CHUNK] for i in range(0, len(rows), _CHUNK)]
    res = [r for part in pool.imap(_normalise_rows, chunks) for r in part]
    cols = ["name_tok", "name_core", "is_domain", "has_alias", "name_indic",
            "addr_tok", "addr_nums", "addr_indic"]
    norm = pd.DataFrame(res, columns=cols)
    out = pd.DataFrame({
        "entity_id": df["entity_id"].astype(str).values,
        "country": df["country"].astype(str).values,
        "name_raw": df["business_name"].astype(str).values,
        "addr_raw": df["business_address"].astype(str).values,
    })
    return pd.concat([out, norm], axis=1)


def norm_path(split, source):
    return CACHE_DIR / f"{split}_{source}_norm.parquet"


def load_norm(split, source, columns=None):
    return pd.read_parquet(norm_path(split, source), columns=columns)


def main(splits=SPLITS, force=False):
    with Pool(N_JOBS) as pool:
        for split in splits:
            for source in SOURCES:
                path = norm_path(split, source)
                if path.exists() and not force:
                    print(f"[preprocess] {path.name} exists, skipping")
                    continue
                t0 = time.time()
                df = raw_table(split, source)
                out = normalise_table(df, pool)
                for c in ("name_tok", "name_core", "addr_tok", "addr_nums", "name_raw", "addr_raw"):
                    out[c] = out[c].astype("string[pyarrow]")
                out.to_parquet(path, index=False)
                print(f"[preprocess] {split}_{source}: {len(out):,} rows in {time.time() - t0:.0f}s",
                      flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=list(SPLITS))
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    main(a.splits, a.force)

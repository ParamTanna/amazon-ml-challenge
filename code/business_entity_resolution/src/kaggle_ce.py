"""Neural cross-encoder for entity matching - run on a Kaggle GPU notebook (T4 x2).

Model: microsoft/MiniLM-L12-H384-uncased (MIT licence, 33M parameters), fine-tuned as a
pair classifier: input "[CLS] s1 name ; s1 address [SEP] cand name ; cand address [SEP]",
output = probability that both records are the same business.

Usage inside a Kaggle notebook cell (dataset with the ce_* files attached as input):
    !python kaggle_ce.py train      # ~45-60 min: fine-tune on ce_train_pairs.parquet
    !python kaggle_ce.py score      # score ce_score_pairs.parquet -> ce_scores.parquet

Everything is written to /kaggle/working (enable Persistence -> "Files only" so the trained
model survives between sessions).  Both GPUs are used via DataParallel with fp16 autocast.
"""

import glob
import math
import os
import queue
import sys
import threading
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          get_linear_schedule_with_warmup)

MODEL = os.environ.get("CE_MODEL", "microsoft/MiniLM-L12-H384-uncased")
OUT = os.environ.get("CE_OUT", "/kaggle/working")
MODEL_DIR = os.path.join(OUT, "ce_model")
MAXLEN = 128
BS_TRAIN = 256
BS_SCORE = 1024
LR = 5e-5
EPOCHS = 1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"   # cpu only for local smoke tests
SMOKE = int(os.environ.get("CE_SMOKE", "0"))              # >0: use only this many pairs


def find(name):
    hits = glob.glob(f"/kaggle/input/**/{name}", recursive=True) + glob.glob(name)
    if not hits:
        raise FileNotFoundError(f"{name} not found under /kaggle/input")
    return hits[0]


def load_texts(path, needed):
    df = pd.read_parquet(path)
    df = df[df["entity_id"].isin(needed)]
    return dict(zip(df["entity_id"], df["text"]))


class AmpWrap(nn.Module):
    """Runs the wrapped model under fp16 autocast (works inside DataParallel replicas)."""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, **kw):
        with torch.autocast("cuda", dtype=torch.float16, enabled=DEVICE == "cuda"):
            return self.m(**kw).logits.squeeze(-1).float()


def prefetch(gen, depth=6):
    """Tokenise the next batches on a background thread while the GPUs work."""
    q = queue.Queue(depth)
    sentinel = object()

    def work():
        for item in gen:
            q.put(item)
        q.put(sentinel)

    threading.Thread(target=work, daemon=True).start()
    while True:
        item = q.get()
        if item is sentinel:
            return
        yield item


def batches(tok, a, b, order, bs, y=None):
    for i in range(0, len(order), bs):
        j = order[i:i + bs]
        enc = tok([a[k] for k in j], [b[k] for k in j], truncation=True, max_length=MAXLEN,
                  padding=True, return_tensors="pt")
        yield j, enc, (None if y is None else torch.tensor(y[j], dtype=torch.float32))


def train():
    pairs = pd.read_parquet(find("ce_train_pairs.parquet"))
    if SMOKE:
        pairs = pairs.head(SMOKE)
    texts = load_texts(find("ce_records_train.parquet"),
                       set(pairs["s1_id"]) | set(pairs["cand_id"]))
    a = pairs["s1_id"].map(texts).fillna("").tolist()
    b = pairs["cand_id"].map(texts).fillna("").tolist()
    y = pairs["label"].values.astype(np.float32)
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(y))
    n_hold = min(50000, len(y) // 50)
    hold, tr = idx[:n_hold], idx[n_hold:]
    print(f"train pairs {len(tr):,}  holdout {len(hold):,}  pos rate {y.mean():.3f}", flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=1)
    net = nn.DataParallel(AmpWrap(model)).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    steps = math.ceil(len(tr) / BS_TRAIN) * EPOCHS
    sched = get_linear_schedule_with_warmup(opt, int(0.05 * steps), steps)
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE == "cuda")
    lossf = nn.BCEWithLogitsLoss()
    t0, step, run_loss = time.time(), 0, 0.0
    net.train()
    for ep in range(EPOCHS):
        order = tr[rng.permutation(len(tr))]
        for _, enc, yb in prefetch(batches(tok, a, b, order, BS_TRAIN, y)):
            enc = {k: v.to(DEVICE, non_blocking=True) for k, v in enc.items()}
            loss = lossf(net(**enc), yb.to(DEVICE, non_blocking=True))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            run_loss = 0.98 * run_loss + 0.02 * loss.item() if step > 1 else loss.item()
            if step % 200 == 0:
                el = time.time() - t0
                print(f"step {step}/{steps} loss {run_loss:.4f} {step * BS_TRAIN / el:.0f} pairs/s "
                      f"eta {el / step * (steps - step) / 60:.1f} min", flush=True)
            if step % 3000 == 0:
                model.save_pretrained(MODEL_DIR)
                tok.save_pretrained(MODEL_DIR)
    model.save_pretrained(MODEL_DIR)
    tok.save_pretrained(MODEL_DIR)
    # holdout check (pairs never seen in training)
    net.eval()
    preds = np.zeros(len(hold), dtype=np.float32)
    pos_of = {h: i for i, h in enumerate(hold)}
    with torch.no_grad():
        for j, enc, _ in batches(tok, a, b, hold, BS_SCORE):
            enc = {k: v.to(DEVICE) for k, v in enc.items()}
            p = torch.sigmoid(net(**enc)).cpu().numpy()
            for k, v in zip(j, p):
                preds[pos_of[k]] = v
    yh = y[hold]
    ll = -np.mean(yh * np.log(np.clip(preds, 1e-6, 1)) + (1 - yh) * np.log(np.clip(1 - preds, 1e-6, 1)))
    acc = np.mean((preds > 0.5) == (yh > 0.5))
    print(f"holdout logloss {ll:.4f}  accuracy {acc:.4f}  train time {(time.time() - t0) / 60:.1f} min",
          flush=True)


def score():
    pairs = pd.read_parquet(find("ce_score_pairs.parquet"))
    if SMOKE:
        pairs = pairs.groupby("split").head(SMOKE // 3).reset_index(drop=True)
    is_test = (pairs["split"] == "test").values
    texts = load_texts(find("ce_records_train.parquet"),
                       set(pairs.loc[~is_test, "s1_id"]) | set(pairs.loc[~is_test, "cand_id"]))
    texts.update(load_texts(find("ce_records_test.parquet"),
                            set(pairs.loc[is_test, "s1_id"]) | set(pairs.loc[is_test, "cand_id"])))
    a = pairs["s1_id"].map(texts).fillna("").tolist()
    b = pairs["cand_id"].map(texts).fillna("").tolist()
    # sort by length so each batch pads very little
    order = np.argsort([len(x) + len(z) for x, z in zip(a, b)], kind="stable")
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR, num_labels=1)
    net = nn.DataParallel(AmpWrap(model)).to(DEVICE).eval()
    out = np.zeros(len(pairs), dtype=np.float32)
    t0, done = time.time(), 0
    with torch.no_grad():
        for j, enc, _ in prefetch(batches(tok, a, b, order, BS_SCORE)):
            enc = {k: v.to(DEVICE, non_blocking=True) for k, v in enc.items()}
            out[j] = torch.sigmoid(net(**enc)).cpu().numpy()
            done += len(j)
            if (done // BS_SCORE) % 500 == 0:
                el = time.time() - t0
                print(f"scored {done:,}/{len(order):,} {done / el:.0f} pairs/s "
                      f"eta {el / done * (len(order) - done) / 60:.1f} min", flush=True)
    pairs["ce"] = out
    pairs.to_parquet(os.path.join(OUT, "ce_scores.parquet"), index=False)
    print(f"done in {(time.time() - t0) / 60:.1f} min -> {OUT}/ce_scores.parquet", flush=True)


if __name__ == "__main__":
    print("GPUs:", torch.cuda.device_count(), flush=True)
    {"train": train, "score": score}[sys.argv[1]]()

# Amazon ML Challenge 2026: Business Entity Resolution
## Team handoff report (26 Sep 2026, ~10:00 IST)

> **Read first.** This file is the single source of truth for what has been built, what
> the numbers are, what to submit now, and the plan to push past **F0.5 = 0.985**.
> Code lives in `code/business_entity_resolution/src/` (its own `README.md` lists every
> command). Large data and intermediate files are **not** in git; see §6.

---

## 0. TL;DR

| | |
|---|---|
| **Current best (local validation)** | **macro F0.5 = 0.9776** (US 0.980, India 0.973), measured on 50,000 held-out training entities |
| First baseline (v0) | 0.955: **do not submit v0** |
| **Submit now** | `submissions/v2/matching_results.tsv` (stage-1 model v2), **validator PASS** (§5) |
| Leaderboard target | top-50 is at 0.982–0.985 → we aim for **≥ 0.986 local** |
| Biggest remaining loss | true matches the model rejects (≈ 0.0155 of the 0.0224 loss) |
| Next 3 moves | **(1)** final stage `run_stage2.py` (CPU, code ready), **(2)** neural cross-encoder on Kaggle GPU (code + data ready), **(3)** scale training data / blocking depth on a bigger machine |
| Deadline | **27 Sep 2026, 23:59 IST**; 5 leaderboard submissions per day |

---

## 1. Timeline and constraints

- Challenge window: 25 Sep 00:00 IST → **27 Sep 23:59 IST**. Max **5 submissions/day**.
  As of this report **no submission has been made on 26 Sep** (v0 was intentionally held back).
- Final package (zip with outputs, code, methodology doc) is required from every team; the
  top teams are audited. `candidate_pairs.tsv` counts too: **smaller candidate sets per
  entity rank higher** in the final review.
- Rules to respect: no external data / APIs / geocoding; final model must be **MIT or
  Apache-2.0 and ≤ 8B parameters** (we use LightGBM (MIT) and MiniLM-L12 (MIT, 33M)).
- Hardware used so far: laptop with 18 threads, **16 GB RAM (~9 GB usable)**. RAM, not
  CPU, was the main bottleneck: every stage is written to stream / chunk. **Never run
  two heavy stages at the same time on a 16 GB machine.**

---

## 2. What the data taught us (drives every design choice)

| Fact | Number | Consequence |
|---|---|---|
| Train / test Source-1 entities | 2.21M / 1.73M | Everything must scale; blocking is mandatory |
| Source-2 + Source-3 records | 10.3M train / 10.0M test | Test has **~23% more S2/S3 records per entity** than train (possible distribution shift) |
| Entities with no match (train) | **5.6%** | Predicting "no match" correctly is worth a full 1.0 per entity |
| Matched S2/S3 records that belong to exactly one entity | **100%** | One-to-one constraint: used in features and in the final decision |
| S2/S3 records that match nothing | 26% | Many are deliberate **look-alike decoys** (same name, house number ±1; same address, other name) |
| Source-1 names shared by other entities | ~40% | Name alone is weak evidence; the **address must carry the match** |
| Indian S2/S3 names in native scripts | 13–24% | Learned a native-script → English dictionary from training matches |
| France | 15% of test entities, **0 training labels** | Language-agnostic features only; no country one-hot |

Noise families (all handled in `normalize.py`): legal-form swaps (Pvt/Private, LLC/L.L.C.),
injected titles (Mr, Smt, Shri), alias phrases (aka/dba/formerly), website-style names,
look-alike characters (i↔l↔1, o↔0, s↔5), accents, filler words (Center, Services), word
drops and re-ordering; addresses truncated to "number, city, state", street-type
abbreviations, state codes / native-script state names, "NULL"/"N/A" padding.

---

## 3. What is built: pipeline, stage by stage (what + why + result)

```
raw TSVs
  │ 1. indic_dict.py      learn native-script → English words from training matches (1,318 words, 17 states)
  │ 2. preprocess.py      normalise every name / address (all 24M records)            ~6 min
  ▼
  │ 3. run_blocking.py    sparse TF-IDF retrieval, forward + coverage-scored reverse   ~35 + 30 min
  │                       → ~48 candidates / entity, 98.4% of true pairs kept
  │ 4. rank_candidates.py cheap LightGBM ranker, keeps top-15 / entity               ~47 min
  │                       → 98.2% of true pairs kept with 15 candidates
  ▼
  │ 5. run_pipeline.py    stage-1 LightGBM, 74 features, trained on partition A        ~15 min train
  │                       expected-F0.5 decision + one-to-one resolution → submission  ~35 min test
  │                       → local F0.5 0.9776
  ▼  (next)
  │ 6. kaggle_ce.py       MiniLM-L12 cross-encoder on GPU (train on partition C, score D/V/test)
  │ 7. run_stage2.py      final LightGBM on partition D: + stage-1 p, cross-encoder, context,
  │                       probability-weighted consensus, competition → submission
```

### 3.1 Normalisation (`normalize.py`, `indic_dict.py`)
- **What:** transliterate + strip accents + lowercase; canonical legal forms; drop titles /
  alias words / fillers into a "core name"; fold look-alike characters; expand/abbreviate
  street types to one canonical form; map state names (incl. native script) to codes;
  remove injected junk ("NULL", "H.No", "Door No").
- **Why:** undo each noise family so similarities compare like with like.
- **Learned dictionary:** native-script words aligned to English words in matched training
  pairs. The generator spells each English word identically every time, so the table is
  near-exact (e.g. प्राइवेट → private, சிவா லாஜிஸ்டிக்ஸ் → shiva logistics). Unknown words
  fall back to rule-based transliteration. Learned only from provided data (rule-compliant).

### 3.2 Blocking (`blocking.py`, `run_blocking.py`)
- **Representation:** each record → L2-normalised TF-IDF over word-level features: core
  words, joined core name (matches website names), 4-char prefixes (typos), address words,
  adjacent address-word pairs ("24_klondyke"), and number×word pairs (survive truncated
  addresses). Features seen once or in > 10,000 records are dropped.
- **Forward search:** each entity → top-20 Source-2 and top-20 Source-3 records.
- **Reverse search (key idea):** each Source-2/3 record → top-5 entities, ranked by
  *coverage of the candidate's evidence* (un-normalised entity vectors). The true owner has
  the most complete address; plain cosine penalises it for that.
- **Result (partition V):** 98.37% of true pairs recalled with 47.8 candidates / entity
  (v0 had 94.0% with 20). Tested and rejected: character-n-gram name views (+0.6% recall
  for +36 candidates). A rarity cap of 30,000 reaches 98.4% (India sample, depth 20/5) but
  is ~3× slower than 10,000 (97.9%), so it is left for a bigger machine (§7, step D).

### 3.3 Learned candidate ranker (`rank_candidates.py`)
- **What:** LightGBM on 22 cheap signals (blocking cosine, forward/reverse ranks, gaps to
  the entity's best candidate, competition between entities for the same candidate, name /
  address commonness, lengths). Keeps the top-15 per entity.
- **Why:** recall first, then shrink: small candidate sets are rewarded in the final review
  and make string features affordable.
- **Result (V):** top-15 keeps **99.82%** of blocked true pairs → **98.19%** overall recall;
  top-10 → 97.85%; top-8 → 97.38%. Most important signal: reverse-search rank.

### 3.4 Stage-1 matcher (`features.py`, `model.py`, `run_pipeline.py`)
- **74 features:** name (ratios, token set/sort, partial, Jaro-Winkler on joined names,
  containment both ways, typo-tolerant containment, unmatched-word counts, legal-form
  conflict, alias/website/native-script flags), address (containment, ratios, typo-tolerant
  containment, unmatched words, empty flag), numbers (shared, first-number match, edit
  similarity, **numeric distance**, shared digit prefix), group (gap to best, rank within
  source, competition between entities), ranker context, name/address commonness, and
  **within-entity consensus** (candidates that share an *alternative* house number or name
  form a decoy cluster).
- **Model:** LightGBM (563 trees), trained on partition A (200k entities, 3.0M pairs).
- **Decision (`decision.py`):** for each entity, choose the top-j candidates that maximise
  the *expected* F0.5 (or predict nothing when "no match" is more likely), then enforce
  one-to-one: a Source-2/3 record kept by several entities stays only with the most probable.
- **Result (V):** **0.9776** (US 0.9804, India 0.9734, entities without match 0.951).
  Top features: ranker probability, candidate competition, min typo-tolerant name match,
  reverse rank, numeric distance of house numbers, consensus count.

### 3.5 Written and unit-tested, not yet run
- `run_stage2.py`: final model on partition D reusing saved stage-1 features (no
  recomputation), adding stage-1 p, cross-encoder score (optional), per-entity context,
  probability-weighted consensus, and ranker-based competition. Streams test file by file.
- `kaggle_ce.py`: MiniLM-L12 cross-encoder (train + score) for a Kaggle T4×2 notebook.
- `export_ce.py`: builds the Kaggle input files.

### 3.6 Training-data partitions (`splits.py`, seed 0, all disjoint)
| Partition | Entities | Used for |
|---|---|---|
| A | 200,000 | ranker + stage-1 training |
| V | 50,000 | validation / decision tuning (all reported scores) |
| C | 250,000 | cross-encoder training (Kaggle) |
| D | 300,000 | final-stage training |
| (unused) | ~1.4M | available to scale training (§7, step C) |

---

## 4. Results so far

| Version | Blocking recall | Candidates / entity | Local F0.5 (V) | US | India | No-match entities |
|---|---|---|---|---|---|---|
| v0 | 94.0% | 20 | 0.9548 | 0.965 | 0.939 | 0.960 |
| **v2 (stage 1)** | 98.2% | **15** | **0.9776** | 0.980 | 0.973 | 0.951 |

**Error budget of v2 on V (50,000 entities, 173,038 true pairs):**

| | |
|---|---|
| Precision / recall (pairs) | 0.9939 / 0.9530 |
| True pairs rejected by the model (inside candidates) | 4,997 |
| True pairs never proposed by blocking | 3,137 (1.8%) |
| Wrong pairs predicted | 1,017 |
| F0.5 lost: entities that only miss matches | **0.0155** |
| F0.5 lost: entities with ≥1 wrong merge | 0.0042 |
| F0.5 lost: no-match entities given a match | 0.0027 |
| **Total loss** | 0.0224 (→ 0.9776) |

To reach **0.985** we must remove ≈ **0.0075** of loss. Most of it must come from
recovering rejected true matches (typos + truncated addresses + random trade names) while
not adding wrong merges.

---

## 5. What to submit now

1. File: **`submissions/v2/matching_results.tsv`** (+ `candidate_pairs.tsv` alongside).
   It is produced by `run_pipeline.py --run s1v2` → `make_submission.py`.
2. Validate first (from `student_resource/`):
   ```
   python utils/validate_submission.py --matching ../submissions/v2/matching_results.tsv --candidate ../submissions/v2/candidate_pairs.tsv --test-dir dataset/test
   ```
   Must print `PASS`.
3. Upload `matching_results.tsv` on the portal. **Record the leaderboard score** in the log
   below: the gap between leaderboard and local 0.9776 tells us whether the test set's
   extra S2/S3 records (see §2) hurt us.

**Submission log** (keep updating: the guidelines require version history)

| Version | Time (IST) | What changed | Local F0.5 | Leaderboard |
|---|---|---|---|---|
| v0 | 26 Sep 03:59 | baseline | 0.9548 | *not submitted* |
| v2 | 26 Sep 09:54 (file built, validator PASS incl. `--check-ids`) | dictionary, blocking v2, ranker, 74 features, expected-F0.5 | 0.9776 | *fill in after upload* |

---

## 6. Repository layout and how to run

```
<repo>/
  HANDOFF_REPORT.md                 ← this file
  code/business_entity_resolution/  ← pipeline (src/), README.md (commands), requirements.txt
  artifacts/                        ← small trained files: indic_dict.json, ranker.txt,
                                      stage1_model.txt, stage1_info.json (decision params)
  docs/KAGGLE_STEPS.md              ← GPU notebook steps
  tools/package_submission.py       ← builds <team>_submission.zip
  tools/experiments/                ← recall / error-analysis scripts
  submission_package/Documentation_template.md  ← methodology draft (fill team names + final numbers)
  student_resource/                 ← challenge files (dataset/ NOT in git)
  work/  output/  submissions/      ← NOT in git (35 GB of caches / pairs / runs)
```

Paths are configurable: `BER_DATA` (dataset dir), `BER_WORK` (work dir), `BER_OUT`
(output dir), `BER_JOBS` (worker processes).

### 6.1 Continuing on the same laptop (all intermediates exist in `work/`)
From `code/business_entity_resolution/src/`:
```
python run_stage2.py --stage1 s1v2 --run s2a                     # final stage without neural score (~30 min)
python ../../../tools/package_submission.py <TEAM>              # when a version is final
```

### 6.2 Starting on a fresh machine
Needs Python 3.13, `pip install -r requirements.txt`, the challenge dataset extracted to
`student_resource/dataset/`, **≥ 32 GB RAM recommended** (16 GB works only with nothing
else running), ideally ≥ 8 cores. Timings below are for an 18-thread laptop; multiply by
~3–4 on a 4-core Kaggle CPU session.
```
cd code/business_entity_resolution/src
python indic_dict.py                                                  #  2 min
python preprocess.py                                                  #  6 min
python run_blocking.py --split train --k 20 --k_rev 5 --max_df 10000  # 35 min
python run_blocking.py --split test  --k 20 --k_rev 5 --max_df 10000  # 30 min
python rank_candidates.py --k 15                                      # 47 min
# optional: reuse the trained stage-1 model instead of retraining (skips ~15 min)
mkdir -p ../../../work/runs/s1v2 && cp ../../../artifacts/stage1_model.txt ../../../work/runs/s1v2/model.txt && cp ../../../artifacts/stage1_info.json ../../../work/runs/s1v2/info.json
python run_pipeline.py --run s1v2        # scores test (+ partitions D,V), writes output/  ~60 min
python run_stage2.py --stage1 s1v2 --run s2a                          # 30 min
```

---

## 7. Plan to exceed F0.5 = 0.985

Estimated gains are **estimates** from the error budget and usual effect sizes. Every step is
measured on partition V before submitting, and the leaderboard score is logged.

| Step | Where | Effort | Est. gain (local) | Status |
|---|---|---|---|---|
| **A. Final stage (context + consensus + competition)** | CPU | 30 min run | +0.002 to +0.004 | code ready, unit-tested |
| **B. Neural cross-encoder score (MiniLM, MIT)** | **GPU (Kaggle)** | ~2.5 h wall, ~2 GPU-h | +0.003 to +0.006 | code + data ready |
| C. More training data (A: 200k → 800k; D: 300k → 600k) | CPU, ≥32 GB | 1–2 h | +0.001 to +0.002 | parameter change |
| D. Blocking recall 98.2% → ~99% | CPU, ≥32 GB, ≥8 cores | 2–3 h | +0.002 to +0.003 | parameter change |
| E. Ensembling (3 seeds / feature subsets) | CPU | 1 h | +0.0005 to +0.001 | small script |
| F. Error-driven features (see below) | CPU | 1–2 h | +0.001 | ideas listed |

Milestones: **A → ≥ 0.980**, **A+B → ≥ 0.984**, **+C/D/E → ≥ 0.986** local.

### Step A: final stage (CPU, run first)
```
python run_stage2.py --stage1 s1v2 --run s2a
```
Trains on partition D, tunes the decision on V, prints `valid_f05`, writes
`output/matching_results.tsv` + `candidate_pairs.tsv` (≤ 10 candidates / entity) and
`work/runs/s2a/info.json`. Submit if `valid_f05` > 0.9776, then copy the outputs to
`submissions/<version>/`.

### Step B: neural cross-encoder on a GPU (Kaggle is free: 30 GPU-h/week per person)
Why: it reads the raw "name ; address" text and learns the generator's noise patterns
(typos, random trade names, transliterations) that hand-made similarities miss; used as
a feature in the final stage.

1. **Training data** (already exported, 1 GB): `work/ce_export/ce_records_train.parquet`,
   `ce_records_test.parquet`, `ce_train_pairs.parquet` (3.34M pairs from partition C),
   plus `code/business_entity_resolution/src/kaggle_ce.py`.
   Upload as a private Kaggle dataset `ber-ce-data` (these files are not in git: get them
   from the laptop, or regenerate with `python export_ce.py --part records` and
   `python export_ce.py --part train`).
2. Kaggle notebook: **Accelerator GPU T4×2, Persistence "Files only", Internet on**.
   ```
   import glob, shutil; shutil.copy(glob.glob('/kaggle/input/**/kaggle_ce.py', recursive=True)[0], '.')
   !python kaggle_ce.py train        # ~45-60 min; prints holdout accuracy at the end
   ```
3. **Already done on the laptop:** `work/ce_export/ce_score_pairs.parquet` (72 MB,
   10.4M pairs: test 8.84M, D 1.33M, V 0.22M; the exact pairs the final stage scores).
   Upload it as a **new version** of the dataset (or include it in the first upload).
   To regenerate: `python export_ce.py --part score --run s1v2`.
4. In the same notebook: `!python kaggle_ce.py score` (~45 min for 10.4M pairs) → download
   `/kaggle/working/ce_scores.parquet` → put it in `work/ce_export/`.
5. `python run_stage2.py --stage1 s1v2 --run s2ce --ce ../../../work/ce_export/ce_scores.parquet`
6. Stop the Kaggle session after each step (the GPU quota burns while a session is open).
Smoke-tested end-to-end on CPU; see `docs/KAGGLE_STEPS.md`.

### Step C: more training data
Partitions are defined in `splits.py` (`SIZES`). Enlarge A (e.g. 800k) and D (e.g. 600k),
keeping V and C fixed. Re-run `rank_candidates.py` → `run_pipeline.py --run s1v3` (new run
folder) → `run_stage2.py --stage1 s1v3 ...`. Needs ~32 GB RAM to be comfortable.

### Step D: blocking recall
Recall is the largest lever left in the pipeline input (3,137 true pairs of V never proposed).
On a machine with ≥ 32 GB RAM and ≥ 8 cores: `run_blocking.py --k 30 --k_rev 10 --max_df 30000`
(India sample: 98.4% at depth 20/5 with max_df 30k vs 97.9% with 10k; deeper k adds more),
then re-run the ranker (keep K = 15, candidate size stays small) and the later stages.

### Step E: ensembling
Train the final stage with 3 different seeds / `feature_fraction` values and average the
probabilities before the decision rule (edit `run_stage2.py`: loop over params, average).

### Step F: error-driven ideas (from the validation error list)
- True matches whose name is replaced by a random trade name at the same address vs decoys
  at the same address: add the fraction of the entity's *other* strong candidates that share
  this candidate's address tokens; the cross-encoder should also help here.
- Empty-address copies of rare names: add a "name rarity × exact name match" feature.
- Per-country decision tuning (the decision rule is currently shared).
- Check the leaderboard/local gap: if the leaderboard is clearly below local, raise
  `empty_bias` (more "no match" answers) and re-tune on V.

---

## 8. Where to run what (GPU / fast compute)

| Task | Best place | Notes |
|---|---|---|
| Cross-encoder train + score | **Kaggle GPU T4×2** | free; 12 h session limit; keep "Persistence: Files only" |
| CPU pipeline (blocking, ranker, features) | a machine with **≥ 32 GB RAM, many cores** | Kaggle CPU session (30 GB, 4 cores) works but ~3–4× slower than the laptop |
| AWS | backup only | Free plan blocks GPU instances; Paid plan + quota request needed (see earlier notes) |

---

## 9. Final submission package checklist (due by 27 Sep 23:59 IST)
- [ ] Best `matching_results.tsv` submitted on the portal and logged in §5.
- [ ] `candidate_pairs.tsv` from the **same** run (it is the exact set the final model scored).
- [ ] `submission_package/Documentation_template.md`: fill team name, members, final numbers.
- [ ] `code/business_entity_resolution/README.md` matches the commands actually used.
- [ ] Build the zip: `python tools/package_submission.py <TEAM> submissions/<best_version>`.
- [ ] Validator `PASS` on the files inside the zip.

---

## 10. Pitfalls we hit (save yourself the time)
- **Memory:** on 16 GB, loading 60M+ pairs or converting millions of IDs to Python strings
  thrashes the machine. The current code streams / chunks and uses integer keys; keep it
  that way. One heavy job at a time.
- **Windows multiprocessing:** workers re-import the main script; keep heavy imports inside
  `main()` (already done) and never run pipeline code from `python -` / notebooks with a
  process pool.
- **numpy threads:** each worker must use `OPENBLAS_NUM_THREADS=1` (set automatically),
  otherwise 16 workers reserve ~1.2 GB each.
- **Hash determinism:** blocking workers use `PYTHONHASHSEED=0` (set in `make_pool()`).
- A job that dies mid-way can be resumed: `run_pipeline.py` reuses `model.txt` +
  `info.json` and skips already-written `stage1_*` files.

---

## Appendix A: module map (`code/business_entity_resolution/src/`)
| File | Role |
|---|---|
| `config.py` | paths (env-overridable), worker count |
| `normalize.py` | all text normalisation |
| `indic_dict.py` | learned native-script dictionaries |
| `preprocess.py` | normalised parquet caches |
| `blocking.py`, `run_blocking.py` | TF-IDF retrieval (forward + coverage-scored reverse) |
| `rank_candidates.py` | learned candidate ranker (streaming) |
| `features.py` | pair features + blocking-score group features |
| `model.py` | record store, consensus features, feature list, LightGBM params |
| `run_pipeline.py` | stage-1 train / tune / predict (+ saves features for the final stage) |
| `decision.py` | expected-F0.5 decision rule + tuning |
| `make_submission.py` | writes the two TSVs, one-to-one resolution |
| `export_ce.py`, `kaggle_ce.py` | cross-encoder data export / GPU training + scoring |
| `run_stage2.py` | final model (context, consensus, competition, cross-encoder) |
| `splits.py` | partitions A / V / C / D |
| `evaluate.py` | local macro F0.5 scorer |

## Appendix B: key parameters
| Parameter | Value | Where |
|---|---|---|
| Blocking rarity cap `max_df` | 10,000 | `run_blocking.py --max_df` |
| Forward / reverse depth | 20 / 5 | `--k` / `--k_rev` |
| Ranker keep K | 15 | `rank_candidates.py --k` |
| Stage-1 LightGBM | lr 0.1, 127 leaves, early stopping (563 trees) | `model.py` |
| Decision | expected-F0.5, scale 1.0, empty_bias 2.0 | `artifacts/stage1_info.json` |
| Final-stage trimming | top-10 with p ≥ 0.005 | `run_stage2.py --max_per_s1 / --min_p` |

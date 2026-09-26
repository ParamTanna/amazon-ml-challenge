# Business Entity Resolution: reproduction guide

Pipeline: **normalise → learned native-script dictionary → high-recall blocking (TF-IDF,
forward + coverage-scored reverse search) → learned candidate ranker (top-15) → stage-1
LightGBM (74 features) → [neural cross-encoder on GPU] → final LightGBM with context,
consensus and competition features → per-entity expected-F0.5 decision + one-to-one
resolution → outputs.**

See `../../HANDOFF_REPORT.md` for results, design rationale and the improvement plan.

All paths default to the repository layout below and can be overridden with environment
variables (`BER_DATA`, `BER_WORK`, `BER_OUT`, `BER_JOBS`).

```
<repo>/
  student_resource/dataset/{train,test}/*.tsv     # challenge data (BER_DATA)
  work/                                            # caches, pair tables, runs (BER_WORK)
  output/                                          # matching_results.tsv, candidate_pairs.tsv (BER_OUT)
  code/business_entity_resolution/src/             # this code
```

## Environment
Python 3.13, `pip install -r requirements.txt`. CPU stages ran on a 16 GB / 18-thread
laptop (32 GB recommended). The cross-encoder stage needs a GPU (Kaggle T4 x2 notebook).

## Run, in order (from `src/`)
| step | command | what it does | time* |
|---|---|---|---|
| 1 | `python indic_dict.py` | learns native-script → English word tables from training matches | 2 min |
| 2 | `python preprocess.py` | normalises every name / address (all splits) | 6 min |
| 3 | `python run_blocking.py --split train --k 20 --k_rev 5 --max_df 10000` | candidate pairs (~48 / entity) | 35 min |
| 4 | `python run_blocking.py --split test --k 20 --k_rev 5 --max_df 10000` | same for test | 30 min |
| 5 | `python rank_candidates.py --k 15` | learned ranker keeps the top-15 candidates per entity | 47 min |
| 6 | `python run_pipeline.py --run s1v2` | stage-1 LightGBM (train on A, tune on V), scores test + partitions D/V, writes outputs | 60–75 min |
| 7 | `python export_ce.py --part records` / `--part train` / `--part score --run s1v2` | data for the cross-encoder | 10 min |
| 8 | GPU: `python kaggle_ce.py train`, then `python kaggle_ce.py score` | fine-tunes MiniLM-L12 (MIT) cross-encoder, scores pairs | ~2.5 h |
| 9 | `python run_stage2.py --stage1 s1v2 --run s2ce --ce <path>/ce_scores.parquet` | final LightGBM (train on D, tune on V), writes outputs | 30 min |

*on the laptop described above. Step 9 also works without `--ce`.

To reuse the already-trained stage-1 model, copy `artifacts/stage1_model.txt` →
`work/runs/s1v2/model.txt` and `artifacts/stage1_info.json` → `work/runs/s1v2/info.json`
before step 6 (training is then skipped).

Steps 6 and 9 finish by calling `make_submission.py`, which writes
`output/matching_results.tsv` and `output/candidate_pairs.tsv` (the exact candidate set the
final model scored). Validate from `student_resource/` with
`python utils/validate_submission.py --matching ../output/matching_results.tsv --candidate ../output/candidate_pairs.tsv --test-dir dataset/test`.

## Module map
| file | role |
|---|---|
| `config.py` | paths and worker count |
| `normalize.py` | text normalisation (transliteration, legal forms, look-alike characters, street types, states) |
| `indic_dict.py` | learned native-script dictionaries |
| `preprocess.py` | normalised parquet caches |
| `blocking.py`, `run_blocking.py` | sparse TF-IDF retrieval (forward + coverage-scored reverse) |
| `rank_candidates.py` | learned candidate ranker / trimming (streaming) |
| `features.py` | pairwise similarity and blocking-score group features |
| `model.py` | record store, consensus features, feature list, LightGBM settings |
| `run_pipeline.py` | stage-1 train / tune / predict (+ saves features for the final stage) |
| `export_ce.py`, `kaggle_ce.py` | cross-encoder data export / GPU training and scoring |
| `run_stage2.py` | final LightGBM with cross-encoder, context, consensus and competition features |
| `decision.py` | per-entity expected-F0.5 decision rule |
| `make_submission.py` | outputs + one-to-one resolution |
| `evaluate.py` | local macro F0.5 scorer |
| `splits.py` | disjoint training partitions A / V / C / D |

No external data or services are used: every table (abbreviations, native-script words)
is either hand-written general knowledge or learned from the provided training data.

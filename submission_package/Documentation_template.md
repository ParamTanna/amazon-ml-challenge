# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** [Team name]
**Team Members:** [Members]
**Submission Date:** 27 September 2026

---

## 1. Executive Summary
We resolve entities with a **recall-first, learn-to-trim** pipeline: aggressive but
language-agnostic normalisation (including a native-script dictionary *learned from the
training matches*), high-recall sparse TF-IDF retrieval in both directions, a cheap learned
ranker that trims each entity's candidate list, a gradient-boosted matcher over ~60 string /
number / consensus features, a fine-tuned MiniLM cross-encoder (MIT, 33M parameters), and a
final model that reasons over the whole candidate group (one-to-one competition between
entities, agreement among an entity's candidates). Decisions are taken per entity by
maximising the expected F0.5, which directly optimises the leaderboard metric, including
"no match" for singletons.

---

## 2. Methodology

### 2.1 Problem Analysis (EDA)
- Train: 2.21M Source-1 entities, 5.03M Source-2 and 5.29M Source-3 records; test adds France
  (15% of test entities) with no training labels.
- 5.6% of training entities have no match; the rest have 1–11 (mostly 2–5).
- **Every matched Source-2/3 record belongs to exactly one entity** (100% of training
  matches) → a one-to-one constraint we exploit in features and decisions.
- 26% of Source-2/3 records match nothing: they are copies of *other* businesses, and many
  are deliberate look-alikes (same name with the house number off by a few, or the same
  address under another name).
- ~40% of Source-1 names are shared by other entities → the address must carry the match.
- Noise families: legal-form swaps (Pvt/Private, LLC/L.L.C.), injected titles (Mr, Smt,
  Shri), alias phrases (aka/dba/formerly), website-style names, look-alike characters
  (i↔l↔1, o↔0, s↔5), accents, filler words (Center, Services), word drops / re-ordering;
  addresses truncated to "number, city, state", street-type abbreviations, state codes,
  "NULL"/"N/A" padding, native-script names (Hindi, Tamil, Telugu, Kannada, Bengali,
  Gujarati, Punjabi, Odia) for 13–24% of Indian Source-2/3 records.

### 2.2 Solution Strategy
**Approach type:** blocking + learned ranker + two-stage gradient boosting + neural
cross-encoder (hybrid).
**Core innovations:** (1) native-script dictionary learned from the training matches
(1,318 words, exact); (2) coverage-scored reverse retrieval; (3) learned candidate trimming;
(4) group-level consensus and competition features; (5) per-entity expected-F0.5 decisions.

---

## 3. Candidate Generation (Blocking)
- **Representation:** each record → sparse, L2-normalised TF-IDF vector over word-level name
  features (core words, joined name, 4-char prefixes) and address features (words, adjacent
  word pairs, number×word pairs that survive truncation).
- **Retrieval:** exact sparse top-k (sparse_dot_topn), per country (open set of labels):
  forward = each Source-1 record → its top-20 Source-2 and top-20 Source-3 records;
  reverse = each Source-2/3 record → its top-5 Source-1 records, ranked by *coverage of the
  candidate's evidence* (un-normalised Source-1 vectors), because the true owner has the
  most complete address and plain cosine would penalise it.
- **Learned trimming:** a LightGBM ranker on blocking-only signals keeps the top-15 per entity.
- **Candidate pairs:** blocking proposes ~48 per entity (98.4% of true pairs, held-out
  entities); the ranker keeps **15 per entity** (98.2% of true pairs, i.e. 99.8% of what
  blocking found). Test: 1.73M entities x 15 = ~26M pairs scored by the matcher.
- **How true matches are not lost:** recall measured on held-out entities at every step
  (see §5); depth / rarity cap chosen from recall-vs-size curves.

## 4. Matching Model
**Features used:**
- Name: ratio, token-set / sort, partial ratio on core and full names, Jaro-Winkler on joined
  names (website forms), token containment both ways, typo-tolerant containment, unmatched
  word counts, legal-form conflict, alias / website / native-script flags, name commonness.
- Address: containment both ways, token-set / sort / partial ratios, typo-tolerant
  containment, unmatched words, empty-address flag, address commonness.
- Numbers: shared numbers, first-number match, edit similarity, numeric distance, shared
  digit prefix.
- Group / competition: gap to the entity's best candidate, rank within source, candidate's
  best score from any other entity, number of entities claiming it, within-entity
  consensus (candidates sharing an alternative number or name).
- Neural: MiniLM-L12 cross-encoder probability on "name ; address" text pairs.

**Model type:** LightGBM (binary) in two stages; MiniLM-L12-H384 cross-encoder (MIT).
**Threshold selection:** per-entity expected-F0.5 maximisation (calibrated on validation),
followed by one-to-one resolution between entities.

---

## 5. Results & Error Analysis
- **F0.5 (macro, validation, 50k held-out entities):** 0.9776 for the stage-1 model
  (US 0.980, India 0.973; entities without any match 0.951). Baseline v0: 0.955.
- Pair precision 0.994, pair recall 0.953. Remaining loss: entities that only miss matches
  0.0155, entities with a wrong merge 0.0042, "no match" entities given a match 0.0027.
- **Common false positives:** look-alike businesses at the same address, or the same
  name with a nearby house number.
- **Common false negatives:** empty-address copies of very common names; heavily
  truncated addresses.

## 6. Conclusion
Normalisation that reverses the generator's noise (including a learned native-script
dictionary), recall-first blocking with coverage-scored reverse search, a learned candidate
ranker and group-aware gradient boosting raised macro F0.5 from 0.955 to 0.978 while
shrinking the candidate set from 20 to 15 per entity. [Update with final-stage / cross-encoder
results before submission.]

---

## Appendix
### A. Code Artefacts
See `code/business_entity_resolution/README.md` (stage order, commands, run times).
### B. Additional Results
| Candidates kept per entity (ranker) | 5 | 8 | 10 | 12 | 15 | 20 |
|---|---|---|---|---|---|---|
| True pairs kept | 91.7% | 97.4% | 97.8% | 98.0% | 98.2% | 98.3% |

| Blocking rarity cap (India sample, depth 20/5) | 2,000 | 10,000 | 30,000 |
|---|---|---|---|
| True pairs recalled | 97.6% | 97.9% | 98.4% |

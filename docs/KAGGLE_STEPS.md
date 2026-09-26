# Neural matcher on Kaggle: steps for the runner

## 1. Upload the data (once)
1. kaggle.com -> Datasets -> New Dataset.
2. Drag in these files from `D:\Amazon ML Challenge\work\ce_export\`:
   `ce_records_train.parquet`, `ce_records_test.parquet`, `ce_train_pairs.parquet`, `kaggle_ce.py`
3. Title: `ber-ce-data`, visibility **Private** -> Create.

## 2. Notebook settings
- New Notebook -> Add Input -> your dataset `ber-ce-data`.
- Session options: **Accelerator = GPU T4 x2**, **Persistence = Files only**, **Internet = On**
  (needed once to download the MiniLM model from Hugging Face).

## 3. Train (about 45-60 minutes)
Run in one cell:
```
import glob, shutil; shutil.copy(glob.glob('/kaggle/input/**/kaggle_ce.py', recursive=True)[0], '.')
!python kaggle_ce.py train
```
It prints progress every 200 steps and a holdout accuracy at the end.
Send the last 5 lines of output to the team.

## 4. Score (later, when `ce_score_pairs.parquet` is ready)
1. Upload `ce_score_pairs.parquet` as a **new version** of `ber-ce-data` (Dataset -> New Version).
2. In the same notebook (the trained model stays in /kaggle/working/ce_model thanks to Persistence):
```
!python kaggle_ce.py score
```
3. Download `/kaggle/working/ce_scores.parquet` (right panel -> Output) and put it in
   `D:\Amazon ML Challenge\work\ce_export\`.

## 5. Save quota
Stop the session (power button) as soon as each step finishes.

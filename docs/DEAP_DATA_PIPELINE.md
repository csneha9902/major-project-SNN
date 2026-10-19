# DEAP Data Pipeline Documentation

**Project:** EEG-Based Adaptive Learning Prototype  
**Task:** Task 1 — Real DEAP EEG Data Pipeline  
**Status:** Active development (`development` branch)

---

## 1. Dataset Overview

The **DEAP dataset** (Dataset for Emotion Analysis using Physiological Signals) is a publicly
available multimodal dataset for studying human affective states.

> **Reference:**  
> S. Koelstra et al., "DEAP: A Database for Emotion Analysis Using Physiological Signals,"
> *IEEE Transactions on Affective Computing*, vol. 3, no. 1, pp. 18–31, 2012.

DEAP consists of EEG and peripheral physiological recordings from 32 participants who
watched 40 one-minute music video clips.  After each clip, participants gave self-assessment
ratings on four dimensions: **Valence, Arousal, Dominance, and Liking** (continuous scale 1–9).

---

## 2. Dataset Location Configuration

The dataset is **never** committed to this repository and **never** downloaded automatically.

Researchers must supply the dataset locally and configure its path via an environment variable:

```bash
export DEAP_DATA_DIR=/path/to/your/local/DEAP
```

For this project:

```bash
export DEAP_DATA_DIR=/Users/sneha/Datasets/DEAP
```

The `DEAP_DATA_DIR` variable is read by `deap_config.get_deap_data_dir()`.  
If it is not set, a clear `EnvironmentError` is raised immediately — the pipeline
never falls back to synthetic data.

See [`backend/.env.template`](../backend/.env.template) for the template entry.

> [!CAUTION]
> **Never commit `.dat` files or personal paths to Git.**  
> The `.gitignore` protects `*.dat`, `*.mat`, `*.edf`, and related extensions.

---

## 3. Available Dataset Subset

This experiment currently uses a **12-subject subset** of DEAP:

| File     | Subject ID |
|----------|-----------|
| `s01.dat` | s01 |
| `s02.dat` | s02 |
| `s03.dat` | s03 |
| `s05.dat` | s05 |
| `s07.dat` | s07 |
| `s08.dat` | s08 |
| `s09.dat` | s09 |
| `s11.dat` | s11 |
| `s15.dat` | s15 |
| `s16.dat` | s16 |
| `s22.dat` | s22 |
| `s25.dat` | s25 |

**Total available: 12 subjects** (out of 32 in the full DEAP release).

> [!IMPORTANT]
> Subjects `s04`, `s06`, `s10`, `s12`–`s14`, `s17`–`s21`, `s23`–`s24`,
> `s26`–`s32` are **not** in the current local dataset.  The loader
> auto-discovers which files exist — it never assumes all 32 are present.

---

## 4. File Format

DEAP provides preprocessed `.dat` files in Python `pickle` format (encoding: `latin1`).

Each file contains a dictionary with two keys:

### `data` — EEG + peripheral recordings

```
Shape: (40, 40, 8064)
       │    │    └── Time samples:  8064 = 63 seconds × 128 Hz
       │    └─────── Channels:      40 total
       └──────────── Trials:        40 music video clips
```

#### Channel layout

| Channel indices | Content |
|---|---|
| 0 – 31 (first 32) | **EEG channels** ← used by this research pipeline |
| 32 – 39 (last 8)  | Peripheral physiological (EOG, EMG, GSR, respiration, BVP, temperature) ← **excluded** |

**This pipeline uses only the 32 EEG channels.**  
The peripheral channels are intentionally excluded from the research model.  
The original 40-channel count is preserved in metadata (`n_total_channels = 40`).

EEG electrode positions follow the international 10-20 system with a 128 Hz sampling rate.

### `labels` — Self-assessment ratings

```
Shape: (40, 4)
       │    └── Ratings: 4 dimensions
       └──────── Trials: 40
```

| Column index | Rating dimension | Scale |
|---|---|---|
| 0 | **Valence**   | 1 (negative) → 9 (positive) |
| 1 | **Arousal**   | 1 (calm) → 9 (excited) ← **research target** |
| 2 | **Dominance** | 1 (submissive) → 9 (dominant) |
| 3 | **Liking**    | 1 (dislike) → 9 (like) |

All four rating dimensions are preserved in `record['ratings']` after loading.

---

## 5. Trial Structure

- Each subject watched **40 music video clips** (trials).
- Each trial is **63 seconds** of continuous EEG recorded at **128 Hz**.
- Total samples per trial per channel: `63 × 128 = 8064`.

### Trial ID convention

This pipeline uses **zero-based trial IDs** (0–39):

| Pipeline trial ID | DEAP documentation trial |
|---|---|
| 0 | Trial 1 |
| 1 | Trial 2 |
| … | … |
| 39 | Trial 40 |

Trial IDs are preserved in every record as `record['trial_ids'] = list(range(40))`.

---

## 6. Arousal Label Definition

For the initial classification task, only **arousal** is used as the research target.

### Binary label mapping

```
LOW_AROUSAL   : arousal rating ≤ threshold   (default threshold = 5.0)
HIGH_AROUSAL  : arousal rating >  threshold
```

**Default threshold: 5.0** (configurable via the `threshold` parameter).

| Arousal rating | Label |
|---|---|
| 1.0 – 5.0 | `LOW_AROUSAL` |
| 5.01 – 9.0 | `HIGH_AROUSAL` |

### What these labels DO NOT mean

> [!WARNING]
> DEAP does not provide ground-truth labels named:
> - **Stress** / **Focused** / **Neutral** / **Cognitive impairment**
>
> These binary labels reflect only the DEAP arousal self-report scale.
> They must not be relabelled without a separate validated methodology.

### Continuous ratings are preserved

The original floating-point arousal rating (and all other ratings) are stored in
`record['ratings']` alongside the binary label.  Binary label generation does not
discard any original data.

---

## 7. Subject-Wise Split Strategy

> [!IMPORTANT]
> **The same subject must never appear in both the training and test sets.**

### Method

`sklearn.model_selection.GroupShuffleSplit` with subject IDs as the grouping variable.

| Parameter | Default | Notes |
|---|---|---|
| `test_size` | `0.25` | ~3 test subjects from 12 |
| `random_seed` | `42` | Fixed for reproducibility |

With 12 subjects and `test_size=0.25`:
- **~9 subjects → training**
- **~3 subjects → test**

### Zero-overlap guarantee

After every split, the following assertion is evaluated:

```python
assert set(train_subjects).isdisjoint(set(test_subjects))
```

If this assertion ever fails, it is treated as a programming error, not a warning.

### Why subject-wise splitting matters

EEG signals contain strong **subject-specific signatures** (skull thickness, electrode
impedance, individual resting alpha frequency).  Random trial-level splitting
causes a model to learn these individual differences rather than generalising across
subjects — a form of **data leakage** that can inflate reported accuracy by 25–40%.

### No data leakage in preprocessing

> [!CAUTION]
> This pipeline performs **no normalisation, standardisation, or filtering
> using learned parameters** before the split.  
> Any preprocessing transforms (e.g. z-score per channel) must be fitted
> **only on training subjects** after the split, then applied to test subjects.

---

## 8. Relevant Source Files

| File | Purpose |
|---|---|
| [`deap_config.py`](../backend/snn_ai_optimizer/datasets/deap_config.py) | Environment variable, constants, expected dimensions |
| [`deap_loader.py`](../backend/snn_ai_optimizer/datasets/deap_loader.py) | Pickle loader, dimension validation, EEG channel selection |
| [`deap_labels.py`](../backend/snn_ai_optimizer/datasets/deap_labels.py) | Binary arousal label generation, flat array builder |
| [`deap_split.py`](../backend/snn_ai_optimizer/datasets/deap_split.py) | Subject-wise GroupShuffleSplit, overlap assertion |
| [`tests/test_deap_pipeline.py`](../backend/tests/test_deap_pipeline.py) | 13 unit tests |

---

## 9. Running the Pipeline

### Set environment variable

```bash
export DEAP_DATA_DIR=/Users/sneha/Datasets/DEAP
```

### Discover and load subjects

```python
from snn_ai_optimizer.datasets.deap_loader import DEAPLoader
from snn_ai_optimizer.datasets.deap_labels import records_to_flat_arrays
from snn_ai_optimizer.datasets.deap_split import subject_wise_split

loader = DEAPLoader()                       # reads DEAP_DATA_DIR from env
subjects = loader.discover_subjects()       # ['s01', 's02', ...]
print(f"Found {len(subjects)} subjects: {subjects}")

record = loader.load_subject('s01')
print(f"EEG shape: {record['eeg'].shape}")         # (40, 32, 8064)
print(f"Ratings shape: {record['ratings'].shape}") # (40, 4)
```

### Generate labels and split

```python
all_records = loader.load_all_subjects()
flat = records_to_flat_arrays(all_records, threshold=5.0)

split = subject_wise_split(flat, test_size=0.25, random_seed=42)
print("Train subjects:", split['train_subject_ids'])
print("Test subjects:", split['test_subject_ids'])
print("X_train shape:", split['X_train'].shape)
print("X_test shape:",  split['X_test'].shape)
```

### Run unit tests

```bash
cd backend
python -m pytest tests/test_deap_pipeline.py -v
```

---

## 10. Limitations

| Limitation | Detail |
|---|---|
| **12-subject subset** | Only 12 of 32 DEAP participants are available locally. Cross-subject generalisation results from 12 subjects have limited statistical power compared to the full 32-subject dataset. |
| **Subject-wise 80/20 split** | With 12 subjects and `test_size=0.25`, the test set has ~3 subjects. This is a small held-out set; results should be interpreted cautiously. |
| **No Leave-One-Subject-Out (LOSO)** | A full LOSO cross-validation (32 folds) would be more rigorous but requires all 32 subjects. This is a planned upgrade once the full dataset is available. |
| **Raw EEG only** | This pipeline loads raw EEG data without artifact rejection, ICA, or baseline correction. Preprocessing will be added in a subsequent task, fitted only on training subjects. |
| **Arousal only** | Only the arousal dimension is used as the classification target. Valence-based or multi-dimensional affect modelling is out of scope for Task 1. |
| **Single threshold** | The binary threshold of 5.0 is a common convention but splits a 1–9 scale at the midpoint. Other thresholds or ordinal encodings may be explored in later tasks. |

---

## 11. Git Protection Statement

> [!CAUTION]
> **The DEAP dataset is NOT committed to this Git repository.**
>
> - All `.dat`, `.mat`, `.edf`, `.fdt`, `.set` files are in `.gitignore`.
> - The `Datasets/`, `DEAP/`, and `data/` directories are also in `.gitignore`.
> - Researchers must set `DEAP_DATA_DIR` locally before running any pipeline code.
> - The dataset may be obtained by signing the DEAP data agreement at:  
>   https://www.eecs.qmul.ac.uk/mmv/datasets/deap/download.html

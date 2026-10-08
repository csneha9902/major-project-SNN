"""
Real classical ML baseline pipeline for DEAP EEG arousal classification.

What this module does
---------------------
1. Loads real DEAP EEG data via DEAPLoader (requires DEAP_DATA_DIR env var).
2. Generates binary arousal labels (LOW_AROUSAL / HIGH_AROUSAL).
3. Performs subject-wise train/test split (zero subject overlap).
4. Extracts Welch-PSD band-power features (32 channels × 4 bands = 128 features).
5. Fits StandardScaler on training data only (no test leakage).
6. Trains two models:
       - Support Vector Machine (RBF kernel, C grid-searched via 5-fold CV)
       - Random Forest (200 trees, balanced class weights)
7. Evaluates both models on the held-out test set.
8. Writes real metrics to results/baseline/metrics.json.
9. Writes full experiment provenance to results/baseline/experiment.json.

Research integrity (AGENTS.md compliance)
------------------------------------------
* All reported metrics come from the held-out test set — never from training data.
* No hardcoded accuracy, F1, or AUC constants.
* StandardScaler is fitted only on X_train — applied to X_test without refitting.
* Train/test subject IDs, random seed, preprocessing config, and model config
  are all recorded in experiment.json for reproducibility.
* If DEAP_DATA_DIR is not set, an EnvironmentError is raised immediately.
  The pipeline never silently substitutes synthetic data.
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    classification_report,
)
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC

from snn_ai_optimizer.datasets.deap_config import (
    DEFAULT_AROUSAL_THRESHOLD,
    LOW_AROUSAL,
    HIGH_AROUSAL,
)
from snn_ai_optimizer.datasets.deap_labels import (
    label_subject_record,
    records_to_flat_arrays,
)
from snn_ai_optimizer.datasets.deap_loader import DEAPLoader
from snn_ai_optimizer.datasets.deap_split import subject_wise_split
from snn_ai_optimizer.features.eeg_features import (
    extract_band_powers,
    feature_names,
    N_FEATURES,
    BAND_ORDER,
    BANDS,
    SAMPLING_RATE,
)

# ---------------------------------------------------------------------------
# Configuration — reproducibility seeds and model hyper-parameters
# ---------------------------------------------------------------------------

RANDOM_SEED: int = 42
TEST_SIZE: float = 0.25

# SVM grid-search candidates
SVM_C_GRID = [0.1, 1.0, 10.0]
SVM_KERNEL = "rbf"
SVM_CV_FOLDS = 5

# Random Forest
RF_N_ESTIMATORS = 200
RF_MAX_FEATURES = "sqrt"

# Welch PSD settings
WELCH_NPERSEG = 256  # 2 s at 128 Hz

# Output directory
RESULTS_DIR = Path("results/baseline")


# ---------------------------------------------------------------------------
# Helper: compute evaluation metrics from predictions
# ---------------------------------------------------------------------------

def _evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None,
    label_encoder: LabelEncoder,
) -> Dict[str, Any]:
    """
    Compute accuracy, macro F1, and AUC on a held-out test set.

    Parameters
    ----------
    y_true, y_pred : ndarray of int
        Integer-encoded true and predicted labels.
    y_prob : ndarray, shape (N, 2) or None
        Class probability estimates from ``predict_proba``.
        AUC is computed only when this is not None.
    label_encoder : LabelEncoder
        Used to recover human-readable class names for the report.

    Returns
    -------
    dict
        Keys: accuracy, f1_macro, auc, classification_report (text).
    """
    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    auc = None
    if y_prob is not None and len(np.unique(y_true)) == 2:
        # Binary case: use probability of the positive (HIGH_AROUSAL) class
        pos_class_idx = list(label_encoder.classes_).index(HIGH_AROUSAL)
        auc = float(roc_auc_score(y_true, y_prob[:, pos_class_idx]))

    report = classification_report(
        y_true,
        y_pred,
        target_names=label_encoder.classes_,
        zero_division=0,
    )

    return {
        "accuracy": acc,
        "f1_macro": f1,
        "auc": auc,
        "classification_report": report,
    }


# ---------------------------------------------------------------------------
# Main pipeline function
# ---------------------------------------------------------------------------

def baseline_run(
    test_size: float = TEST_SIZE,
    random_seed: int = RANDOM_SEED,
) -> Dict[str, Any]:
    """
    Train and evaluate real SVM and Random Forest baselines on DEAP EEG data.

    Parameters
    ----------
    test_size : float, optional
        Fraction of subjects held out for testing.  Default 0.25.
    random_seed : int, optional
        Random seed for reproducible splits and model training.  Default 42.

    Returns
    -------
    dict
        ``svm`` and ``random_forest`` sub-dicts, each with keys:
        ``accuracy``, ``f1_macro``, ``auc``, ``classification_report``.

    Raises
    ------
    EnvironmentError
        If ``DEAP_DATA_DIR`` environment variable is not set.
    FileNotFoundError
        If no DEAP subject files are found in the configured directory.
    """
    run_ts = datetime.now(timezone.utc).isoformat()

    # ------------------------------------------------------------------
    # 1. Load real DEAP data
    # ------------------------------------------------------------------
    print("[Baseline] Loading DEAP dataset ...")
    loader = DEAPLoader()                       # raises EnvironmentError if not set
    records = loader.load_all_subjects()
    print(f"[Baseline] Loaded {len(records)} subjects.")

    # ------------------------------------------------------------------
    # 2. Generate binary arousal labels and flatten across subjects
    # ------------------------------------------------------------------
    labelled = [label_subject_record(rec, threshold=DEFAULT_AROUSAL_THRESHOLD)
                for rec in records]
    flat = records_to_flat_arrays(labelled, threshold=DEFAULT_AROUSAL_THRESHOLD)

    X_raw = flat["X"]        # (N, 32, 8064)
    y_str = flat["y"]        # string labels: LOW_AROUSAL / HIGH_AROUSAL
    subject_ids = flat["subject_ids"]

    # Integer-encode labels for scikit-learn
    le = LabelEncoder()
    y = le.fit_transform(y_str)              # 0 → HIGH_AROUSAL, 1 → LOW_AROUSAL (sorted)

    print(f"[Baseline] Total trials: {len(y)} | "
          f"Classes: {dict(zip(le.classes_, np.bincount(y)))}")

    # Rebuild flat_data dict expected by subject_wise_split
    flat_for_split = {
        "X": X_raw,
        "y": y,
        "ratings": flat["ratings"],
        "subject_ids": subject_ids,
        "trial_ids": flat["trial_ids"],
    }

    # ------------------------------------------------------------------
    # 3. Subject-wise split (zero overlap guaranteed by deap_split.py)
    # ------------------------------------------------------------------
    print(f"[Baseline] Splitting subjects (test_size={test_size}, seed={random_seed}) ...")
    split = subject_wise_split(flat_for_split, test_size=test_size, random_seed=random_seed)

    X_train_raw = split["X_train"]   # (N_train, 32, 8064)
    X_test_raw  = split["X_test"]    # (N_test,  32, 8064)
    y_train     = split["y_train"]
    y_test      = split["y_test"]

    print(f"[Baseline] Train subjects: {split['train_subject_ids']} "
          f"({split['n_train_subjects']} subjects, {len(y_train)} trials)")
    print(f"[Baseline] Test  subjects: {split['test_subject_ids']} "
          f"({split['n_test_subjects']} subjects, {len(y_test)} trials)")

    # ------------------------------------------------------------------
    # 4. Extract Welch-PSD band-power features
    # ------------------------------------------------------------------
    print("[Baseline] Extracting band-power features (Welch PSD) ...")
    X_train_feats = extract_band_powers(X_train_raw, fs=SAMPLING_RATE, nperseg=WELCH_NPERSEG)
    X_test_feats  = extract_band_powers(X_test_raw,  fs=SAMPLING_RATE, nperseg=WELCH_NPERSEG)
    print(f"[Baseline] Feature shape — train: {X_train_feats.shape}, test: {X_test_feats.shape}")

    # ------------------------------------------------------------------
    # 5. Fit StandardScaler on TRAIN only — then transform both sets
    # ------------------------------------------------------------------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_feats)
    X_test_scaled  = scaler.transform(X_test_feats)     # no refitting on test

    # ------------------------------------------------------------------
    # 6a. Train SVM (RBF, C grid-searched via cross-validation)
    # ------------------------------------------------------------------
    print(f"[Baseline] Training SVM (C grid: {SVM_C_GRID}, {SVM_CV_FOLDS}-fold CV) ...")
    svm_base = SVC(kernel=SVM_KERNEL, class_weight="balanced",
                   probability=True, random_state=random_seed)
    svm_grid = GridSearchCV(
        svm_base,
        param_grid={"C": SVM_C_GRID},
        cv=SVM_CV_FOLDS,
        scoring="f1_macro",
        n_jobs=-1,
        refit=True,
    )
    svm_grid.fit(X_train_scaled, y_train)
    best_C = float(svm_grid.best_params_["C"])
    print(f"[Baseline] SVM best C={best_C}. Evaluating on test set ...")

    svm_pred = svm_grid.predict(X_test_scaled)
    svm_prob = svm_grid.predict_proba(X_test_scaled)
    svm_metrics = _evaluate(y_test, svm_pred, svm_prob, le)
    print(f"[Baseline] SVM   — acc={svm_metrics['accuracy']:.4f}  "
          f"F1={svm_metrics['f1_macro']:.4f}  "
          f"AUC={svm_metrics['auc']:.4f}")

    # ------------------------------------------------------------------
    # 6b. Train Random Forest
    # ------------------------------------------------------------------
    print(f"[Baseline] Training Random Forest "
          f"(n_estimators={RF_N_ESTIMATORS}, seed={random_seed}) ...")
    rf = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        max_features=RF_MAX_FEATURES,
        class_weight="balanced_subsample",
        random_state=random_seed,
        n_jobs=-1,
    )
    rf.fit(X_train_scaled, y_train)
    rf_pred = rf.predict(X_test_scaled)
    rf_prob = rf.predict_proba(X_test_scaled)
    rf_metrics = _evaluate(y_test, rf_pred, rf_prob, le)
    print(f"[Baseline] RF    — acc={rf_metrics['accuracy']:.4f}  "
          f"F1={rf_metrics['f1_macro']:.4f}  "
          f"AUC={rf_metrics['auc']:.4f}")

    # ------------------------------------------------------------------
    # 7. Save results
    # ------------------------------------------------------------------
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # metrics.json — human-readable summary
    metrics_summary = {
        "svm": {
            "accuracy": svm_metrics["accuracy"],
            "f1_macro": svm_metrics["f1_macro"],
            "auc":      svm_metrics["auc"],
        },
        "random_forest": {
            "accuracy": rf_metrics["accuracy"],
            "f1_macro": rf_metrics["f1_macro"],
            "auc":      rf_metrics["auc"],
        },
    }
    (RESULTS_DIR / "metrics.json").write_text(
        json.dumps(metrics_summary, indent=2), encoding="utf-8"
    )

    # experiment.json — full provenance record (AGENTS.md requirement)
    experiment = {
        "run_timestamp_utc": run_ts,
        "dataset": "DEAP",
        "n_subjects_total": split["n_total_subjects"],
        "label_definition": (
            f"arousal <= {DEFAULT_AROUSAL_THRESHOLD} → {LOW_AROUSAL}, "
            f"arousal > {DEFAULT_AROUSAL_THRESHOLD} → {HIGH_AROUSAL}"
        ),
        "preprocessing": {
            "method": "Welch PSD band-power",
            "bands": {name: list(freq) for name, freq in BANDS.items()},
            "band_order": list(BAND_ORDER),
            "n_channels": 32,
            "n_features": N_FEATURES,
            "sampling_rate_hz": SAMPLING_RATE,
            "welch_nperseg": WELCH_NPERSEG,
            "scaler": "StandardScaler (fitted on train only)",
        },
        "split": {
            "method": "subject_wise (GroupShuffleSplit)",
            "test_size": test_size,
            "random_seed": random_seed,
            "train_subject_ids": split["train_subject_ids"],
            "test_subject_ids": split["test_subject_ids"],
            "n_train_subjects": split["n_train_subjects"],
            "n_test_subjects": split["n_test_subjects"],
            "n_train_trials": int(len(y_train)),
            "n_test_trials": int(len(y_test)),
        },
        "models": {
            "svm": {
                "kernel": SVM_KERNEL,
                "C_grid_searched": SVM_C_GRID,
                "best_C": best_C,
                "cv_folds": SVM_CV_FOLDS,
                "cv_scoring": "f1_macro",
                "class_weight": "balanced",
                "probability": True,
            },
            "random_forest": {
                "n_estimators": RF_N_ESTIMATORS,
                "max_features": RF_MAX_FEATURES,
                "class_weight": "balanced_subsample",
                "random_state": random_seed,
            },
        },
        "evaluation": {
            "partition": "held-out test set only",
            "svm": svm_metrics,
            "random_forest": rf_metrics,
        },
    }
    (RESULTS_DIR / "experiment.json").write_text(
        json.dumps(experiment, indent=2), encoding="utf-8"
    )

    print(f"[Baseline] Results saved to {RESULTS_DIR.resolve()}")
    return {
        "svm": svm_metrics,
        "random_forest": rf_metrics,
    }


# Backwards-compat alias (some places may import train_baseline)
train_baseline = baseline_run


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        results = baseline_run()
        print("\n=== Baseline Complete ===")
        for model_name, m in results.items():
            print(f"  {model_name:>15s} | acc={m['accuracy']:.4f}  "
                  f"F1={m['f1_macro']:.4f}  AUC={m['auc']:.4f}")
    except EnvironmentError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
    except Exception:
        traceback.print_exc()
        sys.exit(1)

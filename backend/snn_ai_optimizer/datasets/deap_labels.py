"""
DEAP arousal label generation.

Converts the continuous DEAP arousal self-assessment rating (1–9 scale) into
a binary class label for the initial research classification task.

Label definition (per AGENTS.md §LABEL RULES)
----------------------------------------------
  LOW_AROUSAL  : arousal rating ≤ threshold   (default threshold = 5)
  HIGH_AROUSAL : arousal rating >  threshold

These labels reflect the DEAP self-report scale ONLY.  They do NOT correspond
to and must NOT be relabelled as:
  - Stress / Focused / Neutral / Cognitive impairment

The continuous ratings (valence, arousal, dominance, liking) are preserved in
the record.  This function adds binary labels without discarding the originals.

DEAP ratings column ordering (columns 0–3)
------------------------------------------
  0 : valence    (continuous 1–9)
  1 : arousal    (continuous 1–9)  ← used here
  2 : dominance  (continuous 1–9)
  3 : liking     (continuous 1–9)

Trial ID convention
-------------------
Trials are zero-based (0–39) throughout this pipeline.  Trial 0 in this codebase
corresponds to DEAP trial 1 in the original dataset documentation.
"""

from typing import Dict, List, Any, Optional

import numpy as np

from snn_ai_optimizer.datasets.deap_config import (
    DEFAULT_AROUSAL_THRESHOLD,
    LOW_AROUSAL,
    HIGH_AROUSAL,
    RATING_AROUSAL_IDX,
)


def generate_arousal_labels(
    ratings: np.ndarray,
    threshold: float = DEFAULT_AROUSAL_THRESHOLD,
) -> np.ndarray:
    """
    Convert continuous arousal ratings into binary class labels.

    Parameters
    ----------
    ratings : ndarray, shape (n_trials, 4)
        DEAP self-assessment ratings for one subject.
        Column ordering: [valence, arousal, dominance, liking].
    threshold : float, optional
        Binarisation threshold.  Default is ``5.0`` (per AGENTS.md).
        Trials with arousal ≤ threshold → LOW_AROUSAL.
        Trials with arousal >  threshold → HIGH_AROUSAL.

    Returns
    -------
    ndarray of str, shape (n_trials,)
        Binary labels: ``'LOW_AROUSAL'`` or ``'HIGH_AROUSAL'``.

    Raises
    ------
    ValueError
        If *ratings* does not have at least 2 columns (arousal at index 1).
    """
    ratings = np.asarray(ratings, dtype=np.float32)
    if ratings.ndim != 2 or ratings.shape[1] < 2:
        raise ValueError(
            f"ratings must be shape (n_trials, ≥2), got {ratings.shape}."
        )
    arousal_vals = ratings[:, RATING_AROUSAL_IDX]
    labels = np.where(arousal_vals <= threshold, LOW_AROUSAL, HIGH_AROUSAL)
    return labels


def label_subject_record(
    record: Dict[str, Any],
    threshold: float = DEFAULT_AROUSAL_THRESHOLD,
) -> Dict[str, Any]:
    """
    Annotate a subject record with binary arousal labels.

    The original continuous ratings are preserved in ``record['ratings']``.
    A new key ``'arousal_labels'`` is added containing the binary labels.
    The binarisation threshold used is recorded in ``'arousal_threshold'``.

    Parameters
    ----------
    record : dict
        Subject record as returned by :class:`DEAPLoader.load_subject`.
    threshold : float, optional
        Binarisation threshold.  Default is ``5.0``.

    Returns
    -------
    dict
        The input *record* augmented with:
            arousal_labels   : ndarray of str, shape (40,)
            arousal_threshold : float
    """
    record = dict(record)  # shallow copy — do not mutate caller's dict
    record["arousal_labels"] = generate_arousal_labels(
        record["ratings"], threshold=threshold
    )
    record["arousal_threshold"] = threshold
    return record


def label_distribution(labels: np.ndarray) -> Dict[str, int]:
    """
    Count occurrences of each binary label.

    Parameters
    ----------
    labels : ndarray of str
        Array of label strings (``'LOW_AROUSAL'`` / ``'HIGH_AROUSAL'``).

    Returns
    -------
    dict
        ``{'LOW_AROUSAL': n_low, 'HIGH_AROUSAL': n_high}``
    """
    unique, counts = np.unique(labels, return_counts=True)
    dist = {LOW_AROUSAL: 0, HIGH_AROUSAL: 0}
    for label, count in zip(unique, counts):
        dist[label] = int(count)
    return dist


def records_to_flat_arrays(
    records: List[Dict[str, Any]],
    threshold: float = DEFAULT_AROUSAL_THRESHOLD,
) -> Dict[str, Any]:
    """
    Flatten a list of per-subject records into contiguous arrays suitable for
    model training.

    For each record, binary arousal labels are generated if not already present.

    Parameters
    ----------
    records : list of dict
        Subject records from :class:`DEAPLoader`.
    threshold : float, optional
        Arousal binarisation threshold.

    Returns
    -------
    dict with keys:
        X         : ndarray, shape (n_total_trials, 32, 8064)
                    EEG data (32 channels) across all subjects and trials.
        y         : ndarray of str, shape (n_total_trials,)
                    Binary arousal labels.
        ratings   : ndarray, shape (n_total_trials, 4)
                    Original continuous ratings (preserved).
        subject_ids : list of str, length n_total_trials
                    Subject ID repeated once per trial.
        trial_ids   : list of int, length n_total_trials
                    Zero-based trial index within each subject.

    Notes
    -----
    No normalisation, standardisation, or learned-parameter transforms are
    applied here.  Preprocessing transforms must be fitted *only* on training
    data after the subject-wise split.
    """
    X_list, y_list, ratings_list, subj_list, trial_list = [], [], [], [], []

    for rec in records:
        labels = generate_arousal_labels(rec["ratings"], threshold=threshold)
        n_trials = rec["eeg"].shape[0]
        X_list.append(rec["eeg"])
        y_list.extend(labels.tolist())
        ratings_list.append(rec["ratings"])
        subj_list.extend([rec["subject_id"]] * n_trials)
        trial_list.extend(rec["trial_ids"])

    return {
        "X": np.concatenate(X_list, axis=0),          # (N, 32, 8064)
        "y": np.array(y_list),                         # (N,)
        "ratings": np.concatenate(ratings_list, axis=0),  # (N, 4)
        "subject_ids": subj_list,
        "trial_ids": trial_list,
    }

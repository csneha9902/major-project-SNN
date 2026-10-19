"""
Subject-wise train/test split for the DEAP dataset.

Guarantee
---------
The same subject NEVER appears in both the training and test sets.
A formal disjointness assertion is evaluated after every split.

Strategy
--------
``GroupShuffleSplit`` from scikit-learn is used with subject IDs as the
grouping variable.  This is a defensible, reproducible, single-fold
subject-wise hold-out split.

For a 12-subject subset the default test_size=0.25 produces approximately:
  - 9 training subjects
  - 3 test subjects

Configurable parameters
-----------------------
  test_size  : fraction of subjects to assign to the test set (default 0.25)
  random_seed : integer random seed for reproducibility (default 42)

No learned preprocessing transforms (mean, std, filter coefficients, etc.)
are fitted here.  All normalisation must be done AFTER the split, fitted
only on training subjects, to prevent data leakage.

Trial ID convention
-------------------
Zero-based throughout this pipeline (0–39 within each subject).
"""

from typing import Dict, List, Tuple, Any

import numpy as np
from sklearn.model_selection import GroupShuffleSplit

DEFAULT_TEST_SIZE: float = 0.25
DEFAULT_RANDOM_SEED: int = 42


def subject_wise_split(
    flat_data: Dict[str, Any],
    test_size: float = DEFAULT_TEST_SIZE,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> Dict[str, Any]:
    """
    Perform a subject-stratified train/test split with zero subject overlap.

    Parameters
    ----------
    flat_data : dict
        Output of :func:`deap_labels.records_to_flat_arrays`, containing:
            X           : ndarray, shape (N, 32, 8064)
            y           : ndarray, shape (N,)
            ratings     : ndarray, shape (N, 4)
            subject_ids : list of str, length N
            trial_ids   : list of int, length N
    test_size : float, optional
        Fraction of subjects to hold out for testing.  Default ``0.25``.
    random_seed : int, optional
        Random seed for reproducible splits.  Default ``42``.

    Returns
    -------
    dict with keys:
        train_subject_ids : list of str
            Unique subject IDs in the training partition.
        test_subject_ids  : list of str
            Unique subject IDs in the test partition.
        X_train           : ndarray
        X_test            : ndarray
        y_train           : ndarray
        y_test            : ndarray
        ratings_train     : ndarray
        ratings_test      : ndarray
        subject_ids_train : list of str
        subject_ids_test  : list of str
        trial_ids_train   : list of int
        trial_ids_test    : list of int
        test_size         : float   (echoed for experiment logging)
        random_seed       : int     (echoed for experiment logging)

    Raises
    ------
    ValueError
        If subject IDs overlap between train and test sets (should never occur;
        this is a programmatic safeguard).
    ValueError
        If *flat_data* has inconsistent array lengths.
    """
    X = np.asarray(flat_data["X"])
    y = np.asarray(flat_data["y"])
    ratings = np.asarray(flat_data["ratings"])
    subject_ids = list(flat_data["subject_ids"])
    trial_ids = list(flat_data["trial_ids"])

    n_samples = X.shape[0]
    if not (len(y) == len(ratings) == len(subject_ids) == len(trial_ids) == n_samples):
        raise ValueError(
            "flat_data arrays have inconsistent lengths. "
            f"X={n_samples}, y={len(y)}, subject_ids={len(subject_ids)}, "
            f"trial_ids={len(trial_ids)}, ratings={len(ratings)}."
        )

    groups = np.array(subject_ids)
    unique_subjects = sorted(set(subject_ids))
    n_subjects = len(unique_subjects)

    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_seed)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    # ---- Zero-overlap assertion (AGENTS.md rule 6) ----------------------
    train_subjects = sorted(set(groups[train_idx].tolist()))
    test_subjects = sorted(set(groups[test_idx].tolist()))

    overlap = set(train_subjects).intersection(set(test_subjects))
    assert set(train_subjects).isdisjoint(set(test_subjects)), (
        f"Subject leakage detected!  Subjects present in both train and test: "
        f"{overlap}.  This must never happen."
    )
    # ---------------------------------------------------------------------

    return {
        # Subject ID lists
        "train_subject_ids": train_subjects,
        "test_subject_ids": test_subjects,
        # Arrays
        "X_train": X[train_idx],
        "X_test": X[test_idx],
        "y_train": y[train_idx],
        "y_test": y[test_idx],
        "ratings_train": ratings[train_idx],
        "ratings_test": ratings[test_idx],
        "subject_ids_train": groups[train_idx].tolist(),
        "subject_ids_test": groups[test_idx].tolist(),
        "trial_ids_train": [trial_ids[i] for i in train_idx],
        "trial_ids_test": [trial_ids[i] for i in test_idx],
        # Logged metadata
        "n_total_subjects": n_subjects,
        "n_train_subjects": len(train_subjects),
        "n_test_subjects": len(test_subjects),
        "test_size": test_size,
        "random_seed": random_seed,
    }


def split_subject_ids_only(
    subject_ids: List[str],
    test_size: float = DEFAULT_TEST_SIZE,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> Tuple[List[str], List[str]]:
    """
    Split a list of subject IDs into train and test groups.

    Lighter-weight utility that operates on subject IDs alone, without
    requiring the full flattened data arrays (useful for planning the split
    before loading all data).

    Parameters
    ----------
    subject_ids : list of str
        All available subject IDs, e.g. ``['s01', 's02', ...]``.
    test_size : float, optional
        Fraction of subjects for testing.  Default ``0.25``.
    random_seed : int, optional
        Reproducibility seed.  Default ``42``.

    Returns
    -------
    train_ids : list of str
    test_ids  : list of str

    Raises
    ------
    ValueError
        If overlap is detected (programmatic safeguard).
    """
    rng = np.random.RandomState(random_seed)
    n = len(subject_ids)
    n_test = max(1, round(n * test_size))
    shuffled = sorted(subject_ids)
    rng.shuffle(shuffled)
    test_ids = sorted(shuffled[:n_test])
    train_ids = sorted(shuffled[n_test:])

    overlap = set(train_ids).intersection(set(test_ids))
    assert set(train_ids).isdisjoint(set(test_ids)), (
        f"Subject leakage in split_subject_ids_only: {overlap}"
    )
    return train_ids, test_ids

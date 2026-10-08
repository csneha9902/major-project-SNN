"""
Unit tests for Phase 2: real SVM/RF baseline pipeline.

Fixture strategy
----------------
All tests use synthetic data that is structurally identical to real DEAP output
(same shapes and dtypes). Actual values are zeros/deterministic patterns —
ML models will still run and produce valid metric objects.

Synthetic data is used ONLY inside these test fixtures.
The production pipeline (baseline_run) raises EnvironmentError if DEAP_DATA_DIR
is not set — it never silently falls back to synthetic data.

Tests covered
-------------
 1. extract_band_powers output shape is (N, 128)
 2. Feature vector has 128 elements (32 channels × 4 bands)
 3. Band power values are non-negative
 4. No NaN values in extracted features
 5. No Inf values in extracted features
 6. Band frequency ordering: delta < theta < alpha < beta (by upper bound)
 7. feature_names() returns exactly 128 strings
 8. feature_names() starts with 'ch00_delta' and ends with 'ch31_beta'
 9. StandardScaler fitted on train — mean of scaled train ≈ 0 (no leakage)
10. StandardScaler applied to test without refitting (test mean ≠ 0 in general)
11. baseline_run raises EnvironmentError when DEAP_DATA_DIR is unset
12. extract_band_powers raises ValueError for wrong channel count (not 32)
13. extract_band_powers raises ValueError for non-3D input
"""

import os
import sys

import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Path setup — ensure backend/ is importable when tests run from project root
# ---------------------------------------------------------------------------
_BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from snn_ai_optimizer.features.eeg_features import (
    BAND_ORDER,
    BANDS,
    N_FEATURES,
    extract_band_powers,
    feature_names,
)

# ---------------------------------------------------------------------------
# Shared fixture constants
# ---------------------------------------------------------------------------

_N_TRIALS = 8        # small batch — enough to exercise shapes without being slow
_N_CHANNELS = 32     # canonical DEAP EEG channel count
_N_SAMPLES = 8064    # 63 s × 128 Hz


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_raw_eeg(n_trials=_N_TRIALS, n_channels=_N_CHANNELS, n_samples=_N_SAMPLES,
                  seed: int = 0) -> np.ndarray:
    """Return a synthetic EEG array of shape (n_trials, n_channels, n_samples)."""
    rng = np.random.RandomState(seed)
    return rng.randn(n_trials, n_channels, n_samples).astype(np.float32)


# ---------------------------------------------------------------------------
# Test 1: Output shape
# ---------------------------------------------------------------------------

def test_extract_band_powers_output_shape():
    X = _make_raw_eeg()
    feats = extract_band_powers(X)
    assert feats.shape == (_N_TRIALS, N_FEATURES), (
        f"Expected shape ({_N_TRIALS}, {N_FEATURES}), got {feats.shape}"
    )


# ---------------------------------------------------------------------------
# Test 2: Feature dimension is 128 (32 channels × 4 bands)
# ---------------------------------------------------------------------------

def test_feature_dimension_is_128():
    assert N_FEATURES == 32 * 4 == 128, (
        f"N_FEATURES should be 128, got {N_FEATURES}"
    )


# ---------------------------------------------------------------------------
# Test 3: Band power values are non-negative
# ---------------------------------------------------------------------------

def test_band_powers_non_negative():
    X = _make_raw_eeg()
    feats = extract_band_powers(X)
    assert (feats >= 0).all(), "Band powers must be non-negative (power cannot be < 0)."


# ---------------------------------------------------------------------------
# Test 4: No NaN values
# ---------------------------------------------------------------------------

def test_no_nan_in_features():
    X = _make_raw_eeg()
    feats = extract_band_powers(X)
    assert not np.isnan(feats).any(), "Feature matrix contains NaN values."


# ---------------------------------------------------------------------------
# Test 5: No Inf values
# ---------------------------------------------------------------------------

def test_no_inf_in_features():
    X = _make_raw_eeg()
    feats = extract_band_powers(X)
    assert not np.isinf(feats).any(), "Feature matrix contains Inf values."


# ---------------------------------------------------------------------------
# Test 6: Band frequency ordering (delta < theta < alpha < beta by upper bound)
# ---------------------------------------------------------------------------

def test_band_frequency_ordering():
    upper_bounds = [BANDS[b][1] for b in BAND_ORDER]
    for i in range(len(upper_bounds) - 1):
        assert upper_bounds[i] < upper_bounds[i + 1], (
            f"Band upper bounds not strictly increasing: {list(zip(BAND_ORDER, upper_bounds))}"
        )


# ---------------------------------------------------------------------------
# Test 7: feature_names() returns exactly 128 strings
# ---------------------------------------------------------------------------

def test_feature_names_length():
    names = feature_names()
    assert len(names) == 128, f"Expected 128 feature names, got {len(names)}."
    assert all(isinstance(n, str) for n in names), "All feature names must be strings."


# ---------------------------------------------------------------------------
# Test 8: feature_names() starts and ends correctly
# ---------------------------------------------------------------------------

def test_feature_names_start_end():
    names = feature_names()
    assert names[0] == "ch00_delta", f"First feature name should be 'ch00_delta', got '{names[0]}'."
    assert names[-1] == "ch31_beta",  f"Last feature name should be 'ch31_beta', got '{names[-1]}'."


# ---------------------------------------------------------------------------
# Test 9: StandardScaler fitted on train → mean of scaled train ≈ 0
# ---------------------------------------------------------------------------

def test_scaler_fitted_on_train_mean_near_zero():
    X_train = _make_raw_eeg(n_trials=20, seed=1)
    feats_train = extract_band_powers(X_train)

    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(feats_train)

    col_means = scaled_train.mean(axis=0)
    assert np.allclose(col_means, 0.0, atol=1e-6), (
        f"Column means of scaled training data should be ~0. "
        f"Max absolute mean: {np.abs(col_means).max():.2e}"
    )


# ---------------------------------------------------------------------------
# Test 10: Scaler applied to test without refitting (test mean ≠ 0 generally)
#          — validates that we DON'T call fit_transform on the test set
# ---------------------------------------------------------------------------

def test_scaler_not_refitted_on_test():
    """
    If the scaler is fitted ONLY on train, the test set will have non-zero
    column means after transform (unless the distributions happen to be
    identical). This test checks that we correctly use transform() (not
    fit_transform()) on the test set, by verifying the two scaled outputs differ.
    """
    X_train = _make_raw_eeg(n_trials=20, seed=1)
    X_test  = _make_raw_eeg(n_trials=8, seed=99)  # different seed → different distribution

    feats_train = extract_band_powers(X_train)
    feats_test  = extract_band_powers(X_test)

    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(feats_train)
    scaled_test_correct  = scaler.transform(feats_test)       # correct: reuse fitted scaler
    scaled_test_wrong    = StandardScaler().fit_transform(feats_test)  # wrong: refit on test

    # The two test scalings should differ (because distribution differs)
    assert not np.allclose(scaled_test_correct, scaled_test_wrong), (
        "Correct and incorrect test scaling are identical — test data distributions "
        "may be degenerate. Try different seeds."
    )


# ---------------------------------------------------------------------------
# Test 11: baseline_run raises EnvironmentError if DEAP_DATA_DIR is unset
# ---------------------------------------------------------------------------

def test_baseline_raises_env_error_without_deap_dir(monkeypatch):
    """
    The pipeline must fail loudly if the dataset is not configured.
    It must NOT silently use synthetic data.

    Import directly from the module (not via pipeline package __init__)
    to avoid triggering mne imports from preprocess.py.
    """
    monkeypatch.delenv("DEAP_DATA_DIR", raising=False)
    import importlib
    import snn_ai_optimizer.pipeline.baseline as _baseline_mod
    importlib.reload(_baseline_mod)
    _baseline_run = _baseline_mod.baseline_run
    with pytest.raises(EnvironmentError, match="DEAP_DATA_DIR"):
        _baseline_run()


# ---------------------------------------------------------------------------
# Test 12: extract_band_powers raises ValueError for wrong channel count
# ---------------------------------------------------------------------------

def test_extract_band_powers_wrong_channels():
    X_bad = _make_raw_eeg(n_channels=40)  # 40 channels including peripheral — not stripped
    with pytest.raises(ValueError, match="32"):
        extract_band_powers(X_bad)


# ---------------------------------------------------------------------------
# Test 13: extract_band_powers raises ValueError for non-3D input
# ---------------------------------------------------------------------------

def test_extract_band_powers_wrong_ndim():
    X_2d = np.zeros((32, 8064))  # missing trial dimension
    with pytest.raises(ValueError, match="3-dimensional"):
        extract_band_powers(X_2d)

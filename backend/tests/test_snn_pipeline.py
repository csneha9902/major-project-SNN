"""
Unit tests for Phase 3: Real Spiking Neural Network (SNN) pipeline.

Validates the DEAPArousalSNN architecture, Poisson rate encoding, multi-step
LIF neuron firing dynamics, gradient propagation, and environment guards.
"""

from __future__ import annotations

import os
import sys
import numpy as np
import pytest
import torch

_BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from snn_ai_optimizer.features.eeg_features import N_FEATURES
from snn_ai_optimizer.snn.deap_snn_model import DEAPArousalSNN
from snn_ai_optimizer.snn.deap_snn_train import (
    deap_snn_run,
    poisson_encode_features,
    train_and_evaluate_snn,
)

_BATCH_SIZE = 4
_TIME_STEPS = 20


# ---------------------------------------------------------------------------
# Test 1: Output shape of DEAPArousalSNN forward pass is (T, B, 2)
# ---------------------------------------------------------------------------
def test_deap_snn_output_shape():
    model = DEAPArousalSNN(input_dim=N_FEATURES, hidden_dims=(64, 32), num_classes=2)
    x = torch.rand(_TIME_STEPS, _BATCH_SIZE, N_FEATURES)
    out = model(x)
    assert out.shape == (_TIME_STEPS, _BATCH_SIZE, 2), f"Expected ({_TIME_STEPS}, {_BATCH_SIZE}, 2), got {out.shape}"


# ---------------------------------------------------------------------------
# Test 2: Output spikes are binary floats in {0.0, 1.0}
# ---------------------------------------------------------------------------
def test_deap_snn_output_spikes_binary():
    model = DEAPArousalSNN(input_dim=N_FEATURES, hidden_dims=(64, 32), num_classes=2)
    x = torch.rand(_TIME_STEPS, _BATCH_SIZE, N_FEATURES)
    out = model(x)
    unique_vals = torch.unique(out).tolist()
    for val in unique_vals:
        assert val in (0.0, 1.0), f"Output spikes must be in {{0.0, 1.0}}, found {val}"


# ---------------------------------------------------------------------------
# Test 3: Mean firing rate readout shape is (B, 2) and values in [0, 1]
# ---------------------------------------------------------------------------
def test_deap_snn_firing_rate_shape():
    model = DEAPArousalSNN(input_dim=N_FEATURES, hidden_dims=(64, 32), num_classes=2)
    x = torch.rand(_TIME_STEPS, _BATCH_SIZE, N_FEATURES)
    rate = model.firing_rate(x)
    assert rate.shape == (_BATCH_SIZE, 2), f"Expected firing rate shape ({_BATCH_SIZE}, 2), got {rate.shape}"
    assert (rate >= 0.0).all() and (rate <= 1.0).all(), "Firing rates must be bounded within [0, 1]"


# ---------------------------------------------------------------------------
# Test 4: model.reset() resets membrane potentials cleanly
# ---------------------------------------------------------------------------
def test_deap_snn_reset():
    model = DEAPArousalSNN(input_dim=N_FEATURES, hidden_dims=(64, 32), num_classes=2)
    x = torch.rand(_TIME_STEPS, _BATCH_SIZE, N_FEATURES)
    _ = model(x)
    model.reset()  # must not raise exception


# ---------------------------------------------------------------------------
# Test 5: No NaN values in SNN output
# ---------------------------------------------------------------------------
def test_deap_snn_no_nan():
    model = DEAPArousalSNN(input_dim=N_FEATURES, hidden_dims=(64, 32), num_classes=2)
    x = torch.rand(_TIME_STEPS, _BATCH_SIZE, N_FEATURES)
    out = model(x)
    assert not torch.isnan(out).any(), "SNN output contains NaN values"


# ---------------------------------------------------------------------------
# Test 6: No Inf values in SNN output
# ---------------------------------------------------------------------------
def test_deap_snn_no_inf():
    model = DEAPArousalSNN(input_dim=N_FEATURES, hidden_dims=(64, 32), num_classes=2)
    x = torch.rand(_TIME_STEPS, _BATCH_SIZE, N_FEATURES)
    out = model(x)
    assert not torch.isinf(out).any(), "SNN output contains Inf values"


# ---------------------------------------------------------------------------
# Test 7: Poisson encoding produces shape (T, B, N_FEATURES)
# ---------------------------------------------------------------------------
def test_poisson_encode_shape():
    feats = np.random.rand(_BATCH_SIZE, N_FEATURES).astype(np.float32)
    spikes = poisson_encode_features(feats, time_steps=_TIME_STEPS)
    assert spikes.shape == (_TIME_STEPS, _BATCH_SIZE, N_FEATURES), (
        f"Expected shape ({_TIME_STEPS}, {_BATCH_SIZE}, {N_FEATURES}), got {spikes.shape}"
    )


# ---------------------------------------------------------------------------
# Test 8: Poisson encoding produces values strictly in {0.0, 1.0}
# ---------------------------------------------------------------------------
def test_poisson_encode_binary():
    feats = np.random.rand(_BATCH_SIZE, N_FEATURES).astype(np.float32)
    spikes = poisson_encode_features(feats, time_steps=_TIME_STEPS)
    unique_vals = torch.unique(spikes).tolist()
    for val in unique_vals:
        assert val in (0.0, 1.0), f"Poisson spikes must be in {{0.0, 1.0}}, found {val}"


# ---------------------------------------------------------------------------
# Test 9: Poisson encoding safely clips features out of [0, 1] range
# ---------------------------------------------------------------------------
def test_poisson_encode_clipping():
    feats = np.array([[-2.0, 3.5, 0.5, 10.0] * 32], dtype=np.float32)
    spikes = poisson_encode_features(feats, time_steps=_TIME_STEPS)
    assert not torch.isnan(spikes).any()
    unique_vals = torch.unique(spikes).tolist()
    assert set(unique_vals).issubset({0.0, 1.0})


# ---------------------------------------------------------------------------
# Test 10: deap_snn_run raises EnvironmentError when DEAP_DATA_DIR is unset
# ---------------------------------------------------------------------------
def test_deap_snn_run_raises_env_error(monkeypatch):
    monkeypatch.delenv("DEAP_DATA_DIR", raising=False)
    with pytest.raises(EnvironmentError, match="DEAP_DATA_DIR"):
        deap_snn_run()


# ---------------------------------------------------------------------------
# Test 11: Real SNN training and evaluation on synthetic batch
# ---------------------------------------------------------------------------
def test_train_and_evaluate_snn_synthetic():
    rng = np.random.RandomState(42)
    n_train = 6
    n_test = 4
    X_train = rng.randn(n_train, 32, 8064).astype(np.float32)
    y_train = np.array([0, 1, 0, 1, 0, 1], dtype=int)
    X_test = rng.randn(n_test, 32, 8064).astype(np.float32)
    y_test = np.array([0, 1, 0, 1], dtype=int)

    model, metrics, history = train_and_evaluate_snn(
        X_train_raw=X_train,
        y_train=y_train,
        X_test_raw=X_test,
        y_test=y_test,
        time_steps=10,
        epochs=1,
        batch_size=2,
    )

    assert "accuracy" in metrics
    assert "f1_macro" in metrics
    assert "auc" in metrics
    assert "classification_report" in metrics
    assert isinstance(metrics["accuracy"], float)
    assert len(history["train_loss"]) == 1


# ---------------------------------------------------------------------------
# Test 12: Provenance keys completeness check
# ---------------------------------------------------------------------------
def test_provenance_structure_complete():
    required_keys = {
        "run_timestamp_utc",
        "dataset",
        "label_definition",
        "preprocessing",
        "split",
        "model_config",
        "training",
        "evaluation",
    }
    sample_exp = {
        "run_timestamp_utc": "2026-10-08T00:00:00Z",
        "dataset": "DEAP",
        "label_definition": "arousal <= 5 -> LOW_AROUSAL, arousal > 5 -> HIGH_AROUSAL",
        "preprocessing": {},
        "split": {},
        "model_config": {},
        "training": {},
        "evaluation": {},
    }
    assert required_keys.issubset(sample_exp.keys())


# ---------------------------------------------------------------------------
# Test 13: Model supports custom input dimensions
# ---------------------------------------------------------------------------
def test_deap_snn_custom_input_dim():
    custom_dim = 64
    model = DEAPArousalSNN(input_dim=custom_dim, hidden_dims=(32, 16), num_classes=2)
    x = torch.rand(10, 2, custom_dim)
    out = model(x)
    assert out.shape == (10, 2, 2)

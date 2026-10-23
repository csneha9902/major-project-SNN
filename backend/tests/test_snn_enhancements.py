"""
Unit tests for SNN Architectural and Preprocessing Enhancements:
- Learnable Parametric LIF (PLIF)
- Dynamic trace extraction with dropout
- Focal loss implementation
- Checkpoint auto-detection in SNNInferenceEngine
- Pre-trial baseline subtraction
"""

import os
import sys
import numpy as np
import pytest
import torch

_BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from snn_ai_optimizer.features.eeg_features import (
    N_FEATURES,
    extract_windowed_band_powers,
    SAMPLING_RATE,
)
from snn_ai_optimizer.snn.deap_snn_model import DEAPArousalSNN
from snn_ai_optimizer.snn.deap_snn_train import FocalLoss
from snn_ai_optimizer.snn.inference import SNNInferenceEngine


def test_plif_architecture_and_parameters():
    model = DEAPArousalSNN(
        input_dim=N_FEATURES,
        hidden_dims=(64, 32),
        num_classes=2,
        dropout=0.1,
        use_plif=True,
    )
    x = torch.rand(10, 2, N_FEATURES)
    out = model(x)
    assert out.shape == (10, 2, 2)
    # Check that learnable membrane parameters exist
    param_names = [name for name, _ in model.named_parameters()]
    assert any("w" in name for name in param_names), "PLIF learnable parameter 'w' not found"


def test_forward_with_traces_with_dropout_and_plif():
    model = DEAPArousalSNN(
        input_dim=N_FEATURES,
        hidden_dims=(64, 32),
        num_classes=2,
        dropout=0.2,
        use_plif=True,
    )
    x = torch.rand(15, 1, N_FEATURES)
    traces = model.forward_with_traces(x, n_sampled_neurons=8)

    assert traces["time_steps"] == 15
    assert "layers" in traces
    assert "hidden1" in traces["layers"]
    assert "hidden2" in traces["layers"]
    assert "output" in traces["layers"]
    assert len(traces["layers"]["hidden1"]["spikes"]) == 15
    assert len(traces["layers"]["hidden1"]["v"]) == 15
    assert 0.0 <= traces["overall_sparsity"] <= 1.0


def test_focal_loss_computation():
    criterion = FocalLoss(alpha=torch.tensor([0.4, 0.6]), gamma=2.0)
    logits = torch.tensor([[2.0, -1.0], [-1.5, 3.0]], requires_grad=True)
    targets = torch.tensor([0, 1])
    loss = criterion(logits, targets)

    assert loss.dim() == 0  # scalar
    assert loss.item() > 0.0
    loss.backward()
    assert logits.grad is not None


def test_baseline_subtraction_features():
    # 2 trials, 32 channels, 63 seconds at 128 Hz = 8064 samples
    X_synthetic = np.random.randn(2, 32, 8064) * 50.0
    feats, n_win = extract_windowed_band_powers(
        X_synthetic,
        window_sec=4.0,
        step_sec=2.0,
        baseline_sec=3.0,
        fs=SAMPLING_RATE,
        subtract_baseline=True,
    )
    assert feats.shape == (2 * n_win, N_FEATURES)
    assert np.isfinite(feats).all(), "Baseline-subtracted features contain NaNs or Infs"


def test_inference_engine_checkpoint_detection(tmp_path):
    # Train a miniature PLIF model and save state_dict
    model = DEAPArousalSNN(
        input_dim=N_FEATURES,
        hidden_dims=(256, 128),
        num_classes=2,
        dropout=0.15,
        use_plif=True,
    )
    checkpoint_path = tmp_path / "test_plif_checkpoint.pth"
    torch.save(model.state_dict(), checkpoint_path)

    # Load via SNNInferenceEngine
    engine = SNNInferenceEngine(model_path=str(checkpoint_path))
    assert engine.is_model_loaded, "Engine failed to load PLIF checkpoint"
    assert engine._model.use_plif is True, "Engine did not detect PLIF parameter"

    # Test inference output
    dummy_feats = np.random.uniform(0.1, 0.9, size=(128,))
    pred = engine.predict(dummy_feats)
    assert "arousal_label" in pred
    assert "cognitive_state" in pred
    assert pred["arousal_label"] in ["LOW_AROUSAL", "HIGH_AROUSAL"]

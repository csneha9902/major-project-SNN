"""
Unit tests for Phase 4: Aligned SNN Inference Engine.

Verifies:
- 128-dimensional feature inference using DEAPArousalSNN.
- Elimination of zero-padded 3-element representations.
- Resolution of runtime NameError in cognitive.py.
- Accurate 10-20 spatial scalar projections.
- SNN metadata inclusion in real-time streaming frames.
- EDFProcessor 128-feature windowing.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

_BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from snn_ai_optimizer.cognitive import (
    compute_cognitive_state,
    compute_cognitive_state_full,
)
from snn_ai_optimizer.datasets.deap_config import HIGH_AROUSAL, LOW_AROUSAL
from snn_ai_optimizer.features.eeg_features import N_FEATURES
from snn_ai_optimizer.snn.deap_snn_model import DEAPArousalSNN
from snn_ai_optimizer.snn.inference import (
    SNNInferenceEngine,
    get_snn_inference_engine,
    project_scalars_to_128_features,
)
from snn_ai_optimizer.streaming import DataStreamer


# ---------------------------------------------------------------------------
# Test 1: SNNInferenceEngine initializes cleanly
# ---------------------------------------------------------------------------
def test_inference_engine_initialization():
    engine = SNNInferenceEngine(model_path="non_existent_weights.pth")
    assert engine is not None
    assert not engine.is_model_loaded


# ---------------------------------------------------------------------------
# Test 2: Inference engine handles missing model weights gracefully
# ---------------------------------------------------------------------------
def test_inference_engine_fallback_when_unweighted():
    engine = SNNInferenceEngine(model_path="non_existent_weights.pth")
    feats = np.random.rand(N_FEATURES).astype(np.float32)
    res = engine.predict(feats)
    assert res["arousal_label"] in (LOW_AROUSAL, HIGH_AROUSAL)
    assert res["cognitive_state"] in ("Focused", "Neutral", "Stressed")
    assert not res["using_snn"]


# ---------------------------------------------------------------------------
# Test 3: Inference engine with real DEAPArousalSNN weights
# ---------------------------------------------------------------------------
def test_inference_engine_with_active_model(tmp_path):
    model = DEAPArousalSNN(input_dim=N_FEATURES, hidden_dims=(32, 16), num_classes=2)
    weights_path = tmp_path / "test_snn.pth"
    torch.save(model.state_dict(), weights_path)

    engine = SNNInferenceEngine(model_path=weights_path, time_steps=10, hidden_dims=(32, 16))
    assert engine.is_model_loaded

    feats = np.random.rand(N_FEATURES).astype(np.float32)
    res = engine.predict(feats)
    assert res["using_snn"]
    assert res["arousal_label"] in (LOW_AROUSAL, HIGH_AROUSAL)
    assert 0.0 <= res["confidence"] <= 1.0


# ---------------------------------------------------------------------------
# Test 4: predict on 128 features returns all required contract keys
# ---------------------------------------------------------------------------
def test_predict_128_features_keys():
    engine = SNNInferenceEngine(model_path="non_existent_weights.pth")
    feats = np.random.rand(128).astype(np.float32)
    res = engine.predict(feats)
    required_keys = {"arousal_label", "cognitive_state", "confidence", "probabilities", "firing_rate", "using_snn"}
    assert required_keys.issubset(res.keys())


# ---------------------------------------------------------------------------
# Test 5: Output probabilities sum to 1.0 within numerical precision
# ---------------------------------------------------------------------------
def test_predict_probabilities_sum_to_one():
    engine = SNNInferenceEngine(model_path="non_existent_weights.pth")
    feats = np.random.rand(128).astype(np.float32)
    res = engine.predict(feats)
    probs = res["probabilities"]
    assert np.isclose(probs[LOW_AROUSAL] + probs[HIGH_AROUSAL], 1.0, atol=1e-3)


# ---------------------------------------------------------------------------
# Test 6: Batch prediction on multiple 128-feature vectors
# ---------------------------------------------------------------------------
def test_predict_batch_128_features():
    engine = SNNInferenceEngine(model_path="non_existent_weights.pth")
    batch = np.random.rand(5, 128).astype(np.float32)
    res_list = engine.predict_batch(batch)
    assert len(res_list) == 5
    for item in res_list:
        assert item["arousal_label"] in (LOW_AROUSAL, HIGH_AROUSAL)


# ---------------------------------------------------------------------------
# Test 7: Predict raises ValueError if feature dimension != 128
# ---------------------------------------------------------------------------
def test_predict_raises_on_invalid_feature_dim():
    engine = SNNInferenceEngine(model_path="non_existent_weights.pth")
    bad_feats = np.random.rand(32).astype(np.float32)
    with pytest.raises(ValueError, match="128"):
        engine.predict(bad_feats)


# ---------------------------------------------------------------------------
# Test 8: Scalar projection yields 128 positive features
# ---------------------------------------------------------------------------
def test_scalar_projection_shape_and_bounds():
    feats = project_scalars_to_128_features(alpha=0.8, beta=0.4, lf_hf=1.1, seed=42)
    assert feats.shape == (128,)
    assert (feats > 0.0).all(), "All band powers should be strictly positive"


# ---------------------------------------------------------------------------
# Test 9: compute_cognitive_state succeeds without NameError
# ---------------------------------------------------------------------------
def test_compute_cognitive_state_no_name_error():
    state = compute_cognitive_state(alpha=1.2, beta=0.3, lf_hf_ratio=0.8)
    assert state in ("Focused", "Neutral", "Stressed")


# ---------------------------------------------------------------------------
# Test 10: compute_cognitive_state_full returns full dictionary
# ---------------------------------------------------------------------------
def test_compute_cognitive_state_full():
    full = compute_cognitive_state_full(alpha=0.4, beta=1.2, lf_hf_ratio=2.0)
    assert "arousal_label" in full
    assert "cognitive_state" in full
    assert full["cognitive_state"] == "Stressed"


# ---------------------------------------------------------------------------
# Test 11: DataStreamer yields frames with snn_inference metadata
# ---------------------------------------------------------------------------
def test_datastreamer_yields_snn_metadata():
    streamer = DataStreamer()
    streamer.start_simulation()
    gen = streamer.stream(interval_sec=0)
    frame = next(gen)
    assert "snn_inference" in frame
    assert "arousal_label" in frame["snn_inference"]
    assert frame["snn_inference"]["arousal_label"] in (LOW_AROUSAL, HIGH_AROUSAL)


# ---------------------------------------------------------------------------
# Test 12: EDFProcessor extract_windowed_128_features produces valid windows
# ---------------------------------------------------------------------------
def test_edf_processor_windowed_features(tmp_path):
    from snn_ai_optimizer.upload.edf_processor import EDFProcessor

    proc = EDFProcessor(tmp_path / "dummy.edf")
    # Simulate loaded raw data: 32 channels, 512 samples at 128 Hz (4 seconds)
    class DummyRaw:
        def get_data(self):
            return np.random.randn(32, 512).astype(np.float32)

    proc.raw = DummyRaw()
    proc.metadata = {"sfreq": 128.0, "duration": 4.0, "n_channels": 32}

    window_feats = proc.extract_windowed_128_features(window_sec=2.0)
    assert window_feats.shape == (2, 128), f"Expected (2, 128), got {window_feats.shape}"


# ---------------------------------------------------------------------------
# Test 13: Singleton get_snn_inference_engine returns consistent instance
# ---------------------------------------------------------------------------
def test_singleton_engine():
    e1 = get_snn_inference_engine()
    e2 = get_snn_inference_engine()
    assert e1 is e2

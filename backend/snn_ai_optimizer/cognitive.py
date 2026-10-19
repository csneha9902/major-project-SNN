"""
Cognitive and Affective State Inference Module.

Provides bridges between EEG feature extraction and the neuromorphic SNN
inference engine. Aligns feature representations to 128 Welch PSD band-power
features while preserving backward compatibility for scalar stream inputs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Sequence

import numpy as np

from snn_ai_optimizer.snn.inference import (
    get_snn_inference_engine,
    project_scalars_to_128_features,
)

logger = logging.getLogger(__name__)


def compute_cognitive_state(
    alpha: float,
    beta: float,
    lf_hf_ratio: float | None = None,
    features_128: Sequence[float] | np.ndarray | None = None,
) -> str:
    """
    Compute affective/cognitive state using the aligned SNN inference engine.

    Parameters
    ----------
    alpha : float
        Mean alpha band power (8-13 Hz).
    beta : float
        Mean beta band power (13-30 Hz).
    lf_hf_ratio : float or None
        Low/high frequency ratio from heart rate variability (HRV).
    features_128 : array-like of shape (128,) or None
        Optional explicit 128-dimensional band-power feature vector (32 ch * 4 bands).
        If None, the scalar metrics are projected across the 10-20 electrode layout.

    Returns
    -------
    str
        "Focused", "Neutral", or "Stressed"
    """
    try:
        engine = get_snn_inference_engine()

        if features_128 is not None:
            feats = np.asarray(features_128, dtype=np.float32)
        else:
            feats = project_scalars_to_128_features(alpha=alpha, beta=beta, lf_hf=lf_hf_ratio)

        res = engine.predict(feats, alpha_hint=alpha, beta_hint=beta)
        return str(res["cognitive_state"])

    except Exception as exc:
        logger.warning(f"[compute_cognitive_state] SNN inference failed, falling back to heuristic: {exc}")

        # Deterministic biological heuristic fallback
        try:
            if lf_hf_ratio is not None and lf_hf_ratio > 1.5:
                return "Stressed"
        except Exception:
            pass

        try:
            if alpha is not None and beta is not None and (alpha - beta) > 0.1:
                return "Focused"
        except Exception:
            pass

        return "Neutral"


def compute_cognitive_state_full(
    alpha: float,
    beta: float,
    lf_hf_ratio: float | None = None,
    features_128: Sequence[float] | np.ndarray | None = None,
) -> Dict[str, Any]:
    """
    Compute full affective inference details including probabilities and arousal labels.

    Returns
    -------
    dict with keys:
        arousal_label : "LOW_AROUSAL" or "HIGH_AROUSAL"
        cognitive_state : "Focused", "Neutral", or "Stressed"
        confidence : float
        probabilities : dict
        firing_rate : list of 2 floats
        using_snn : bool
    """
    engine = get_snn_inference_engine()
    if features_128 is not None:
        feats = np.asarray(features_128, dtype=np.float32)
    else:
        feats = project_scalars_to_128_features(alpha=alpha, beta=beta, lf_hf=lf_hf_ratio)

    return engine.predict(feats, alpha_hint=alpha, beta_hint=beta)


@dataclass
class Recommendation:
    task: str
    difficulty: int

"""
FastAPI REST router for SNN model inspection, training, and aligned inference.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

from snn_ai_optimizer.snn.deap_snn_train import deap_snn_run
from snn_ai_optimizer.snn.inference import (
    DEFAULT_MODEL_PATH,
    get_snn_inference_engine,
    project_scalars_to_128_features,
)

router = APIRouter(prefix="/api/snn", tags=["snn"])


class PredictRequest(BaseModel):
    # Option A: Direct 128 Welch PSD band-power features
    features: Optional[List[float]] = None
    # Option B: Raw nested matrix or legacy format for backward compatibility
    data: Optional[List[List[float]]] = None
    # Option C: Scalar summaries
    alpha: Optional[float] = None
    beta: Optional[float] = None
    lf_hf: Optional[float] = None


class TrainResponse(BaseModel):
    message: str
    status: str


@router.post("/train", response_model=TrainResponse)
async def train_model(background_tasks: BackgroundTasks):
    """Triggers real DEAP SNN model training in the background."""
    background_tasks.add_task(deap_snn_run)
    return {
        "message": "Real DEAP SNN training initiated in background. Output will be saved to results/snn/.",
        "status": "started",
    }


@router.get("/status")
async def model_status():
    """Checks if the trained DEAPArousalSNN model is ready for inference."""
    engine = get_snn_inference_engine()
    model_exists = DEFAULT_MODEL_PATH.exists()
    return {
        "status": "ready" if model_exists else "not_trained",
        "model_path": str(DEFAULT_MODEL_PATH) if model_exists else None,
        "is_active_in_memory": engine.is_model_loaded,
        "input_features": 128,
        "architecture": "DEAPArousalSNN (SpikingJelly multi-step LIF)",
    }


@router.post("/predict")
async def predict_cognitive_state(request: PredictRequest):
    """
    Predict affective state (LOW_AROUSAL vs HIGH_AROUSAL) and mapped cognitive state.

    Accepts 128-dimensional Welch PSD features or scalar biometric summaries.
    """
    engine = get_snn_inference_engine()

    try:
        # 1. Direct 128 features
        if request.features is not None:
            if len(request.features) != 128:
                raise HTTPException(
                    status_code=400,
                    detail=f"Expected exactly 128 features (32 ch * 4 bands), received {len(request.features)}.",
                )
            feats = request.features

        # 2. Legacy data array
        elif request.data is not None:
            flattened = [item for sublist in request.data for item in (sublist if isinstance(sublist, list) else [sublist])]
            if len(flattened) == 128:
                feats = flattened
            elif len(flattened) == 3:
                feats = project_scalars_to_128_features(alpha=flattened[0], beta=flattened[1], lf_hf=flattened[2]).tolist()
            else:
                # Default projection
                feats = project_scalars_to_128_features(alpha=1.0, beta=1.0).tolist()

        # 3. Scalar summaries
        elif request.alpha is not None and request.beta is not None:
            feats = project_scalars_to_128_features(alpha=request.alpha, beta=request.beta, lf_hf=request.lf_hf).tolist()

        else:
            raise HTTPException(
                status_code=400,
                detail="Request must provide either 'features' (128 floats), 'data', or 'alpha' and 'beta' scalars.",
            )

        res = engine.predict(feats, alpha_hint=request.alpha, beta_hint=request.beta)

        return {
            "prediction": res["cognitive_state"],
            "arousal_label": res["arousal_label"],
            "confidence": res["confidence"],
            "probabilities": res["probabilities"],
            "firing_rate": res["firing_rate"],
            "using_snn": res["using_snn"],
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

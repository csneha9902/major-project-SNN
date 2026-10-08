"""
SNN execution pipeline orchestrator.

Integrates real SNN training on the DEAP dataset with neuromorphic task
recommendation initialization.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from snn_ai_optimizer.models.snn_recommender import SNNRecommender
from snn_ai_optimizer.snn.deap_snn_train import deap_snn_run
from snn_ai_optimizer.utils.logger import (
    create_run_folder,
    save_metrics,
    save_text_log,
)


def train_recommender(run_path: Path) -> Dict[str, Any]:
    """Initialize and persist neuromorphic task recommendation model."""
    save_text_log(run_path, "[SNN Recommender] Initializing numpy LIF model...")
    model = SNNRecommender(input_size=3, hidden_size=16, output_size=5, time_steps=20)
    out_dir = Path("results/snn")
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(out_dir / "recommender_model.npz"))
    save_text_log(run_path, "[SNN Recommender] Model saved to results/snn/recommender_model.npz")
    return {"status": "initialized", "model_path": "results/snn/recommender_model.npz"}


def snn_run(
    test_size: float = 0.25,
    random_seed: int = 42,
    epochs: int = 30,
) -> Dict[str, Any]:
    """
    Execute real SNN training on DEAP and log true evaluated metrics.

    Never hardcodes metrics. If DEAP_DATA_DIR is missing, raises EnvironmentError.
    """
    run_dir = create_run_folder("snn")
    save_text_log(run_dir, "[SNN Pipeline] Starting real SNN training...")

    # Train real DEAP SNN
    metrics = deap_snn_run(
        test_size=test_size,
        random_seed=random_seed,
        epochs=epochs,
    )
    save_text_log(run_dir, f"[SNN Pipeline] Training complete. Evaluated metrics: {metrics}")

    # Initialize recommender
    train_recommender(run_dir)

    # Save metrics to run folder
    save_metrics(run_dir, "snn", metrics)
    save_text_log(run_dir, "[SNN Pipeline] Done.")
    return metrics

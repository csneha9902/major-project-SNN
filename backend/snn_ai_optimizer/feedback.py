from __future__ import annotations
from pathlib import Path
import json

def _read_json(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None

def generate_feedback() -> dict:
    latest = _read_json(Path("results/latest_metrics.json")) or {}
    snn_m = _read_json(Path("results/snn/metrics.json")) or latest.get("snn") or {}
    base = _read_json(Path("results/baseline/metrics.json")) or latest.get("baseline") or {}
    eeg  = latest.get("preprocess_eeg") or {}

    tips = []
    actions = []

    # Model quality heuristics from genuine held-out evaluation
    acc = max(base.get("accuracy", 0), snn_m.get("accuracy", 0))
    auc = max(base.get("auc", 0) or 0.0, snn_m.get("auc", 0) or 0.0)

    if acc < 0.6:
        tips.append("Accuracy is low; consider collecting more samples or stronger features (e.g., EEG band-power, MRI ROIs).")
        actions.append("Enable feature engineering: delta/theta/alpha/beta powers; z-score per subject.")
    else:
        tips.append("Accuracy is reasonable—try tuning hyperparameters and adding cross-validation.")

    if not isinstance(auc, float) or auc != auc:  # NaN check
        tips.append("AUC is undefined (single-class split). Use stratified train/test and larger sample size.")
        actions.append("Ensure stratified split; increase N to avoid degenerate folds.")
    elif auc < 0.65:
        tips.append("AUC is modest—try tuning the SNN time steps or adding more LIF layers.")
        actions.append("Increase SNN simulation steps or adjust firing thresholds.")

    # Data readiness
    if eeg.get("ok") is False:
        tips.append("EEG preprocessing failed. Ensure DEAP_DATA_DIR points to real DEAP dataset files.")
        actions.append("Configure DEAP_DATA_DIR environment variable with local DEAP dataset path.")

    # Wellness / study pacing (simple placeholders)
    tips.append("Use 25–40 min focus blocks with 5–7 min breaks; hydrate and stretch between sessions.")
    tips.append("Schedule hardest topics at the time of day you typically have higher alertness.")

    return {"summary": {"accuracy": acc, "auc": auc},
            "tips": tips,
            "actions": actions}

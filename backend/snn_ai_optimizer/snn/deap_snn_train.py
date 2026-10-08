"""
Real SNN training and evaluation pipeline for DEAP EEG Arousal estimation.

Methodology
-----------
1. Loads real DEAP EEG subject data via DEAPLoader (requires DEAP_DATA_DIR).
2. Generates binary Arousal labels (LOW_AROUSAL <= 5, HIGH_AROUSAL > 5).
3. Executes subject-wise disjoint train/test split.
4. Extracts 128 Welch PSD band-power features per trial (32 channels * 4 bands).
5. Standardizes features using StandardScaler fitted only on training data.
6. Encodes features into temporal spike trains via Poisson rate coding over T steps.
7. Trains DEAPArousalSNN using SpikingJelly multi-step LIF neurons with ATan surrogate gradients.
8. Evaluates on the held-out subject test set for honest accuracy, macro F1, and AUROC.
9. Persists real metrics and experiment provenance record to results/snn/.

Research integrity (AGENTS.md compliance)
-----------------------------------------
* No synthetic data is used in research pipeline.
* No hardcoded metrics: all values derived directly from held-out test evaluation.
* Preprocessing parameters fitted strictly on training subjects to avoid data leakage.
"""

from __future__ import annotations

import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

try:
    from spikingjelly.activation_based import functional
    HAS_SPIKINGJELLY = True
except ImportError:
    HAS_SPIKINGJELLY = False

from snn_ai_optimizer.datasets.deap_config import (
    DEFAULT_AROUSAL_THRESHOLD,
    HIGH_AROUSAL,
    LOW_AROUSAL,
)
from snn_ai_optimizer.datasets.deap_labels import (
    label_subject_record,
    records_to_flat_arrays,
)
from snn_ai_optimizer.datasets.deap_loader import DEAPLoader
from snn_ai_optimizer.datasets.deap_split import subject_wise_split
from snn_ai_optimizer.features.eeg_features import (
    BAND_ORDER,
    BANDS,
    N_FEATURES,
    SAMPLING_RATE,
    extract_band_powers,
)
from snn_ai_optimizer.snn.deap_snn_model import DEAPArousalSNN

RESULTS_DIR = Path("results/snn")


def poisson_encode_features(
    features: np.ndarray,
    time_steps: int = 50,
    seed: int | None = None,
) -> torch.Tensor:
    """
    Encode continuous [0, 1] normalized features into Poisson spike trains.

    Parameters
    ----------
    features : ndarray of shape (B, N_FEATURES)
        Values normalized strictly within [0, 1].
    time_steps : int, default=50
        Number of simulation time-steps T.
    seed : int or None
        Optional random seed for deterministic spike generation.

    Returns
    -------
    torch.Tensor
        Binary spike tensor of shape (T, B, N_FEATURES) in {0.0, 1.0}.
    """
    if seed is not None:
        torch.manual_seed(seed)
    feats = np.clip(features, 0.0, 1.0)
    feat_tensor = torch.from_numpy(feats).float()  # (B, N_FEATURES)
    # Expand across temporal dimension: (T, B, N_FEATURES)
    expanded = feat_tensor.unsqueeze(0).expand(time_steps, -1, -1)
    spikes = torch.bernoulli(expanded)
    return spikes


def train_and_evaluate_snn(
    X_train_raw: np.ndarray,
    y_train: np.ndarray,
    X_test_raw: np.ndarray,
    y_test: np.ndarray,
    time_steps: int = 50,
    epochs: int = 30,
    batch_size: int = 16,
    lr: float = 1e-3,
    random_seed: int = 42,
    device: str | None = None,
) -> Tuple[DEAPArousalSNN, Dict[str, Any], Dict[str, Any]]:
    """
    Train and evaluate DEAPArousalSNN on raw EEG arrays with no data leakage.

    Parameters
    ----------
    X_train_raw : ndarray of shape (N_train, 32, 8064)
    y_train : ndarray of shape (N_train,)
    X_test_raw : ndarray of shape (N_test, 32, 8064)
    y_test : ndarray of shape (N_test,)

    Returns
    -------
    model : DEAPArousalSNN
    test_metrics : dict
    train_history : dict
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    # 1. Feature extraction
    X_train_feats = extract_band_powers(X_train_raw, fs=SAMPLING_RATE)
    X_test_feats = extract_band_powers(X_test_raw, fs=SAMPLING_RATE)

    # 2. Normalization fitted on training set only
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_feats)
    X_test_scaled = scaler.transform(X_test_feats)

    # MinMax scaling into [0, 1] for Poisson rate encoder
    minmax = MinMaxScaler(feature_range=(0.0, 1.0))
    X_train_norm = minmax.fit_transform(X_train_scaled)
    X_test_norm = np.clip(minmax.transform(X_test_scaled), 0.0, 1.0)

    # 3. Model setup
    model = DEAPArousalSNN(input_dim=N_FEATURES, hidden_dims=(256, 128), num_classes=2)
    model.to(dev)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # PyTorch Dataset for mini-batching
    train_dataset = TensorDataset(
        torch.from_numpy(X_train_norm).float(),
        torch.from_numpy(y_train).long(),
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    history = {"train_loss": [], "train_acc": []}

    # 4. Training loop
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        correct = 0
        total = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(dev), batch_y.to(dev)
            # Encode mini-batch to Poisson spikes: (T, B, N_FEATURES)
            spikes = torch.bernoulli(batch_x.unsqueeze(0).expand(time_steps, -1, -1))

            model.reset()
            optimizer.zero_grad()

            logits = model.firing_rate(spikes)  # (B, 2)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            model.reset()

            epoch_loss += loss.item() * len(batch_y)
            preds = logits.argmax(dim=-1)
            correct += (preds == batch_y).sum().item()
            total += len(batch_y)

        avg_loss = epoch_loss / total
        avg_acc = correct / total
        history["train_loss"].append(avg_loss)
        history["train_acc"].append(avg_acc)

    # 5. Held-out test evaluation
    model.eval()
    with torch.no_grad():
        test_x_tensor = torch.from_numpy(X_test_norm).float().to(dev)
        test_spikes = torch.bernoulli(test_x_tensor.unsqueeze(0).expand(time_steps, -1, -1))

        model.reset()
        test_logits = model.firing_rate(test_spikes)
        model.reset()

        test_probs = torch.softmax(test_logits, dim=-1).cpu().numpy()
        test_preds = test_probs.argmax(axis=1)

    acc = float(accuracy_score(y_test, test_preds))
    f1 = float(f1_score(y_test, test_preds, average="macro", zero_division=0))

    auc = None
    if len(np.unique(y_test)) == 2:
        auc = float(roc_auc_score(y_test, test_probs[:, 1]))

    report = classification_report(
        y_test,
        test_preds,
        target_names=[LOW_AROUSAL, HIGH_AROUSAL] if np.array_equal(sorted(np.unique(y_test)), [0, 1]) else None,
        zero_division=0,
    )

    metrics = {
        "accuracy": acc,
        "f1_macro": f1,
        "auc": auc,
        "classification_report": report,
    }

    return model, metrics, history


def deap_snn_run(
    test_size: float = 0.25,
    random_seed: int = 42,
    time_steps: int = 50,
    epochs: int = 30,
    batch_size: int = 16,
    lr: float = 1e-3,
) -> Dict[str, Any]:
    """
    Execute full DEAP SNN training and save reproducible provenance results.

    Raises
    ------
    EnvironmentError
        If DEAP_DATA_DIR environment variable is not configured.
    """
    run_ts = datetime.now(timezone.utc).isoformat()
    print("[SNN Pipeline] Initializing real DEAP dataset loader...")
    loader = DEAPLoader()
    records = loader.load_all_subjects()
    print(f"[SNN Pipeline] Successfully loaded {len(records)} subjects.")

    labelled = [
        label_subject_record(rec, threshold=DEFAULT_AROUSAL_THRESHOLD)
        for rec in records
    ]
    flat = records_to_flat_arrays(labelled, threshold=DEFAULT_AROUSAL_THRESHOLD)

    X_raw = flat["X"]
    y_str = flat["y"]
    subject_ids = flat["subject_ids"]

    # 0: LOW_AROUSAL, 1: HIGH_AROUSAL
    y = np.array([0 if label == LOW_AROUSAL else 1 for label in y_str], dtype=int)

    flat_for_split = {
        "X": X_raw,
        "y": y,
        "ratings": flat["ratings"],
        "subject_ids": subject_ids,
        "trial_ids": flat["trial_ids"],
    }

    print(f"[SNN Pipeline] Performing subject-wise split (test_size={test_size}, seed={random_seed})...")
    split = subject_wise_split(flat_for_split, test_size=test_size, random_seed=random_seed)

    X_train_raw = split["X_train"]
    y_train = split["y_train"]
    X_test_raw = split["X_test"]
    y_test = split["y_test"]

    print(f"[SNN Pipeline] Training SNN on {len(y_train)} trials ({split['n_train_subjects']} subjects)...")
    model, test_metrics, history = train_and_evaluate_snn(
        X_train_raw=X_train_raw,
        y_train=y_train,
        X_test_raw=X_test_raw,
        y_test=y_test,
        time_steps=time_steps,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        random_seed=random_seed,
    )

    print(f"[SNN Pipeline] Test evaluation results: Accuracy={test_metrics['accuracy']:.4f}, "
          f"F1={test_metrics['f1_macro']:.4f}, AUC={test_metrics['auc']}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Save model weights
    model_path = RESULTS_DIR / "deap_snn.pth"
    torch.save(model.state_dict(), model_path)

    # 2. Save metrics summary
    metrics_summary = {
        "accuracy": test_metrics["accuracy"],
        "f1_macro": test_metrics["f1_macro"],
        "auc": test_metrics["auc"],
    }
    (RESULTS_DIR / "metrics.json").write_text(json.dumps(metrics_summary, indent=2), encoding="utf-8")

    # 3. Save experiment provenance
    experiment = {
        "run_timestamp_utc": run_ts,
        "dataset": "DEAP",
        "n_subjects_total": split["n_total_subjects"],
        "label_definition": (
            f"arousal <= {DEFAULT_AROUSAL_THRESHOLD} -> {LOW_AROUSAL}, "
            f"arousal > {DEFAULT_AROUSAL_THRESHOLD} -> {HIGH_AROUSAL}"
        ),
        "preprocessing": {
            "feature_extraction": "Welch PSD band-power (4 bands x 32 channels)",
            "n_features": N_FEATURES,
            "bands": {name: list(freq) for name, freq in BANDS.items()},
            "band_order": list(BAND_ORDER),
            "sampling_rate_hz": SAMPLING_RATE,
            "scaler": "StandardScaler + MinMaxScaler -> [0, 1] (fitted on train only)",
            "encoding": f"Poisson rate coding, T={time_steps} time-steps",
        },
        "split": {
            "method": "subject_wise (GroupShuffleSplit)",
            "test_size": test_size,
            "random_seed": random_seed,
            "train_subject_ids": split["train_subject_ids"],
            "test_subject_ids": split["test_subject_ids"],
            "n_train_subjects": split["n_train_subjects"],
            "n_test_subjects": split["n_test_subjects"],
            "n_train_trials": int(len(y_train)),
            "n_test_trials": int(len(y_test)),
        },
        "model_config": {
            "class": "DEAPArousalSNN",
            "layers": [N_FEATURES, 256, 128, 2],
            "surrogate": "ATan",
            "step_mode": "m",
            "time_steps": time_steps,
        },
        "training": {
            "optimizer": "Adam",
            "lr": lr,
            "epochs": epochs,
            "batch_size": batch_size,
            "final_train_loss": history["train_loss"][-1] if history["train_loss"] else None,
            "final_train_acc": history["train_acc"][-1] if history["train_acc"] else None,
        },
        "evaluation": {
            "partition": "held-out test set only",
            "metrics": test_metrics,
        },
    }
    (RESULTS_DIR / "experiment.json").write_text(json.dumps(experiment, indent=2), encoding="utf-8")

    return test_metrics


if __name__ == "__main__":
    try:
        results = deap_snn_run()
        print("\n=== Real SNN Training Complete ===")
        print(f"Accuracy: {results['accuracy']:.4f} | F1: {results['f1_macro']:.4f} | AUC: {results['auc']}")
    except EnvironmentError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
    except Exception:
        traceback.print_exc()
        sys.exit(1)

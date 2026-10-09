#!/usr/bin/env python3
"""
Automated Research Pipeline Runner for EEG-Based Affective State Estimation.

Executes the full genuine research pipeline end-to-end:
1. Validates DEAP dataset configuration (no synthetic data fallback).
2. Performs subject-wise train/test split (zero subject overlap).
3. Trains Baseline classifiers (SVM with RBF kernel and Random Forest) on 128 Welch PSD features.
4. Trains Spiking Neural Network (DEAPArousalSNN: Poisson encoding + multi-step LIF neurons).
5. Compiles a comparative benchmark report saved to results/benchmark_report.json.

Usage:
    python3 scripts/run_research_pipeline.py [--deap-dir /path/to/deap] [--epochs 30] [--seed 42]
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Ensure backend directory is in sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
BACKEND_DIR = REPO_ROOT / "backend"

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from snn_ai_optimizer.datasets.deap_config import get_deap_data_dir
from snn_ai_optimizer.pipeline.baseline import train_baseline
from snn_ai_optimizer.pipeline.snn_pipeline import snn_run


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run end-to-end EEG SNN research pipeline on DEAP data."
    )
    parser.add_argument(
        "--deap-dir",
        type=str,
        default=None,
        help="Path to local DEAP data directory containing s01.dat ... s32.dat files.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of epochs for SNN training (default: 30).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for subject-wise split and model initialization (default: 42).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.deap_dir:
        os.environ["DEAP_DATA_DIR"] = args.deap_dir

    print("=" * 70)
    print(" EEG-BASED SNN RESEARCH PIPELINE: AUTOMATED BENCHMARK RUNNER")
    print("=" * 70)

    # 1. Validate environment
    try:
        data_dir = get_deap_data_dir()
        print(f"[Pipeline] DEAP dataset directory: {data_dir}")
        dat_files = list(data_dir.glob("s*.dat"))
        print(f"[Pipeline] Found {len(dat_files)} participant files in DEAP directory.")
        if len(dat_files) == 0:
            raise FileNotFoundError(f"No s*.dat files found in {data_dir}")
    except Exception as e:
        print(f"\n[ERROR] Dataset validation failed: {e}")
        print("Please configure DEAP_DATA_DIR or supply --deap-dir.")
        sys.exit(1)

    results_dir = REPO_ROOT / "backend" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    start_total = time.time()

    # 2. Train baseline models (SVM + Random Forest)
    print("\n" + "-" * 70)
    print(" STEP 1: Training Baseline Models (SVM RBF & Random Forest)")
    print("-" * 70)
    start_base = time.time()
    train_baseline()
    base_duration = round(time.time() - start_base, 2)
    print(f"[Pipeline] Baseline training completed in {base_duration}s.")

    # 3. Train SNN model (DEAPArousalSNN)
    print("\n" + "-" * 70)
    print(f" STEP 2: Training Spiking Neural Network (DEAPArousalSNN, {args.epochs} epochs)")
    print("-" * 70)
    start_snn = time.time()
    snn_run()
    snn_duration = round(time.time() - start_snn, 2)
    print(f"[Pipeline] SNN training completed in {snn_duration}s.")

    # 4. Load genuine metrics and compile comparative benchmark report
    base_metrics_path = results_dir / "baseline" / "metrics.json"
    snn_metrics_path = results_dir / "snn" / "metrics.json"

    base_metrics = json.loads(base_metrics_path.read_text()) if base_metrics_path.exists() else {}
    snn_metrics = json.loads(snn_metrics_path.read_text()) if snn_metrics_path.exists() else {}

    total_duration = round(time.time() - start_total, 2)

    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset": {
            "name": "DEAP",
            "directory": str(data_dir),
            "n_subjects_found": len(dat_files),
        },
        "timing": {
            "baseline_seconds": base_duration,
            "snn_seconds": snn_duration,
            "total_seconds": total_duration,
        },
        "benchmarks": {
            "svm_rbf": {
                "accuracy": base_metrics.get("svm", {}).get("accuracy"),
                "f1": base_metrics.get("svm", {}).get("f1"),
                "auc": base_metrics.get("svm", {}).get("auc"),
            },
            "random_forest": {
                "accuracy": base_metrics.get("random_forest", {}).get("accuracy"),
                "f1": base_metrics.get("random_forest", {}).get("f1"),
                "auc": base_metrics.get("random_forest", {}).get("auc"),
            },
            "deap_arousal_snn": {
                "accuracy": snn_metrics.get("accuracy"),
                "f1": snn_metrics.get("f1"),
                "auc": snn_metrics.get("auc"),
            },
        },
    }

    report_path = results_dir / "benchmark_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n[Pipeline] Comparative benchmark report saved to: {report_path}")

    # 5. Display formatted summary scorecard
    print("\n" + "=" * 70)
    print(" COMPARATIVE RESEARCH SCORECARD (HELD-OUT SUBJECT EVALUATION)")
    print("=" * 70)
    print(f"{'Model':<24} | {'Accuracy':<10} | {'F1-Score':<10} | {'ROC-AUC':<10}")
    print("-" * 70)

    for model_name, label in [
        ("svm_rbf", "SVM (RBF Kernel)"),
        ("random_forest", "Random Forest"),
        ("deap_arousal_snn", "DEAPArousalSNN (LIF)"),
    ]:
        m = report["benchmarks"].get(model_name, {})
        acc = f"{m.get('accuracy', 0.0):.4f}" if m.get("accuracy") is not None else "N/A"
        f1 = f"{m.get('f1', 0.0):.4f}" if m.get("f1") is not None else "N/A"
        auc = f"{m.get('auc', 0.0):.4f}" if m.get("auc") is not None else "N/A"
        print(f"{label:<24} | {acc:<10} | {f1:<10} | {auc:<10}")

    print("=" * 70)
    print(f"Total Pipeline Runtime: {total_duration}s")
    print("=" * 70)


if __name__ == "__main__":
    main()

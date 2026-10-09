# Research Reproducibility Guide

**Project:** EEG-Based Adaptive Learning Prototype  
**Objective:** Affective-state estimation from EEG via Spiking Neural Networks (SNN) and adaptive task difficulty optimization via Q-Learning.  
**Research Integrity Standards:** [AGENTS.md](file:///Users/sneha/Desktop/major-project-SNN/AGENTS.md)

---

## 1. System Requirements & Setup

### Environment
* **OS:** macOS / Linux / Windows
* **Python:** 3.10+ (tested on Python 3.12.4)
* **Core Libraries:** `torch>=2.0`, `spikingjelly>=0.0.0.0.14`, `scikit-learn`, `scipy`, `mne`, `fastapi`, `uvicorn`

### Local DEAP Dataset
The DEAP dataset (*Dataset for Emotion Analysis using Physiological Signals*, Koelstra et al., 2012) must be supplied locally.
Set the `DEAP_DATA_DIR` environment variable to the path containing `s01.dat` through `s32.dat`:

```bash
export DEAP_DATA_DIR="/Users/sneha/Datasets/DEAP"
```

> [!IMPORTANT]
> The DEAP dataset is **never** committed to Git, downloaded automatically, or substituted with synthetic data in the research training pipeline (AGENTS.md Rules 3, 5, 7).

---

## 2. One-Command Full Pipeline Reproduction

To execute the entire training pipeline from raw EEG `.dat` files through baseline models, SNN training, and benchmark report generation in a single command:

```bash
python3 scripts/run_research_pipeline.py --epochs 30 --seed 42
```

This automated runner performs:
1. **Subject-Wise Data Split**: Disjoint partition (GroupShuffleSplit, 0 subject overlap).
2. **Feature Extraction**: 128 Welch PSD band powers (4 bands $\times$ 32 channels) using `scipy.signal.welch`.
3. **Baseline Training**: Support Vector Machine (RBF kernel, 5-fold CV grid search) and Random Forest (200 estimators).
4. **SNN Training**: Multi-step Leaky Integrate-and-Fire (`DEAPArousalSNN`) with Poisson rate encoding and ATan surrogate gradient.
5. **Scorecard Compilation**: Held-out evaluation metrics saved to `results/benchmark_report.json`.

---

## 3. Step-by-Step Manual Reproduction

### Step A: Train Baseline Models (SVM & Random Forest)
```bash
cd backend
python3 -m snn_ai_optimizer.pipeline.baseline
```
* **Output:**
  * `results/baseline/metrics.json`: Held-out accuracy, macro F1, ROC-AUC, and full classification report.
  * `results/baseline/experiment.json`: Complete experiment provenance (split indices, random seeds, hyperparameters).

### Step B: Train Spiking Neural Network (DEAPArousalSNN)
```bash
cd backend
python3 -m snn_ai_optimizer.pipeline.snn_pipeline
```
* **Output:**
  * `results/snn/deap_snn.pth`: PyTorch model weights checkpoint.
  * `results/snn/metrics.json`: Held-out test evaluation metrics.
  * `results/snn/experiment.json`: Architecture specification, LIF parameters ($\tau=2.0, V_{th}=1.0$), and training hyper-parameters.

### Step C: Tabular Q-Learning Recommendation Evaluation
```bash
cd backend
python3 -c "from snn_ai_optimizer.optimizer import recommend_task, update_q_table; print(recommend_task('Focused'))"
```
* **Output:**
  * `results/q_table.json`: Tabular Q-values for (state, action) pairs with recorded $\epsilon$ exploration rate and update counter.

---

## 4. Research Integrity Audit

An automated integrity verification script checks that all non-negotiable rules are honored:

```bash
python3 scripts/verify_research_integrity.py
```

### Audited Invariants:
1. **Rule 6 (Subject Isolation)**: Asserts that $\text{TrainSubjects} \cap \text{TestSubjects} = \emptyset$.
2. **Rule 10 (Preprocessing Consistency)**: Asserts feature representation is exactly $128$ dimensions ($32 \times 4$ Welch PSD band powers) across training and inference.
3. **Rules 1 & 2 (No Fabricated Metrics)**: Asserts that neither baseline nor SNN metrics contain legacy hardcoded stubs (`0.85/0.90` or `0.92/0.95`).
4. **Core SNN Dynamics**: Asserts binary spike activations and multi-step LIF integration.
5. **Core RL Dynamics**: Asserts Bellman TD update $Q(s,a) \leftarrow Q(s,a) + \alpha[r + \gamma \max_{a'} Q(s',a') - Q(s,a)]$ and decaying $\epsilon$.

---

## 5. Automated Unit & Integration Test Suite

Run the complete test suite across all 8 phases:

```bash
cd backend
python3 -m pytest tests/ -v --tb=short
```

**Scorecard (86 tests total):**
* `test_deap_pipeline.py`: 13 tests (Phase 1 — DEAP loader, labels, subject split)
* `test_baseline_pipeline.py`: 13 tests (Phase 2 — Welch PSD, 128 features, SVM/RF)
* `test_snn_pipeline.py`: 13 tests (Phase 3 — SNN training, Poisson encoding, LIF)
* `test_snn_inference.py`: 13 tests (Phase 4 — 128-dim inference, EDF windowing)
* `test_q_learning.py`: 17 tests (Phase 5 — Bellman TD, $\epsilon$-decay, persistence)
* `test_api_integration.py`: 12 tests (Phase 6 — REST endpoints, live RL metadata)
* `test_reproducibility.py`: 5 tests (Phase 8 — Invariants, provenance, leakage check)

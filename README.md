# EEG-Based Adaptive Learning Prototype with SNN & Q-Learning

A research prototype for **affective-state estimation from EEG signals** using **Spiking Neural Networks (SNN)** and **adaptive task difficulty recommendation** using **Q-Learning**.

Built in strict compliance with the research integrity and provenance rules defined in [AGENTS.md](file:///Users/sneha/Desktop/major-project-SNN/AGENTS.md).

---

## Architecture Overview

```
                      ┌──────────────────────────────────────┐
                      │    DEAP Dataset (s01.dat..s32.dat)   │
                      └──────────────────┬───────────────────┘
                                         │
                         Subject-Wise Disjoint Partition
                                (0 Subject Overlap)
                                         │
                                         ▼
                      ┌──────────────────────────────────────┐
                      │  Welch PSD 128-Dim Feature Extractor  │
                      │  (32 Channels × 4 Frequency Bands)   │
                      └──────────┬───────────────────┬───────┘
                                 │                   │
                                 ▼                   ▼
                     ┌─────────────────────┐  ┌─────────────────────┐
                     │ Baseline Models     │  │ Spiking Neural Net  │
                     │ • SVM (RBF Kernel)  │  │ • DEAPArousalSNN    │
                     │ • Random Forest     │  │ • Poisson Encoding  │
                     │                     │  │ • Multi-Step LIF    │
                     └──────────┬──────────┘  └──────────┬──────────┘
                                │                        │
                                └───────────┬────────────┘
                                            ▼
                             Held-Out Subject Evaluation
                             (Accuracy, Macro F1, ROC-AUC)
                                            │
                                            ▼
                      ┌──────────────────────────────────────┐
                      │   Tabular Bellman Q-Learning Engine   │
                      │   Q(s,a) ← Q(s,a) + α[r + γ max Q - Q│
                      │   Adaptive Learning Task Recommender │
                      └──────────────────────────────────────┘
```

---

## Key Features

* **Real DEAP Data Pipeline**: Direct parsing of local `.dat` files with subject-wise isolation (`GroupShuffleSplit`, zero subject overlap).
* **Canonical 128-Dim Feature Representation**: Delta (1–4 Hz), Theta (4–8 Hz), Alpha (8–13 Hz), and Beta (13–30 Hz) Welch power spectral densities across 32 EEG channels.
* **Genuine SNN Classification**: 3-layer `DEAPArousalSNN` implemented with `SpikingJelly` multi-step Leaky Integrate-and-Fire (LIF) neurons and ArcTan surrogate gradients.
* **Tabular Q-Learning Recommendation**: Full Bellman optimality equation with $s \to s'$ state transitions, decaying $\epsilon$-greedy exploration, and persistent Q-table.
* **Zero Fabricated Metrics**: All metrics are computed strictly from held-out evaluation sets without synthetic substitution or hardcoded stubs.
* **DEMO vs. RESEARCH Mode Separation**: Prominent UI indicators distinguishing interactive synthetic demonstrations from authentic held-out DEAP research evaluations.

---

## Test Suite Scorecard

**86/86 Passing Unit & Integration Tests**

| Phase | Test File | Tests | Focus Area |
|---|---|:---:|---|
| Phase 1 | `test_deap_pipeline.py` | 13 | DEAP loader, arousal labeling, subject-wise split |
| Phase 2 | `test_baseline_pipeline.py` | 13 | Welch PSD 128 features, SVM RBF grid search, Random Forest |
| Phase 3 | `test_snn_pipeline.py` | 13 | Multi-step LIF dynamics, Poisson encoding, held-out SNN eval |
| Phase 4 | `test_snn_inference.py` | 13 | SNN singleton engine, 128-dim inference, EDF windowing |
| Phase 5 | `test_q_learning.py` | 17 | Bellman TD equation, $s \to s'$, $\epsilon$-decay, persistence |
| Phase 6 | `test_api_integration.py` | 12 | `/api/recommend`, `/feedback` TD stats, `/api/benchmark` |
| Phase 8 | `test_reproducibility.py` | 5 | Invariant checks, zero leakage, experiment provenance |
| **Total** | | **86** | **All Passing** ✅ |

---

## Quick Start

### 1. Environment Setup
```bash
pip install -r backend/requirements.txt
export DEAP_DATA_DIR="/path/to/local/deap"  # Contains s01.dat ... s32.dat
```

### 2. Run All Tests
```bash
cd backend
python3 -m pytest tests/ -v --tb=short
```

### 3. Verify Research Integrity
```bash
python3 scripts/verify_research_integrity.py
```

### 4. Run End-to-End Pipeline
```bash
python3 scripts/run_research_pipeline.py --epochs 30 --seed 42
```

### 5. Launch Application
```bash
# Terminal 1: Backend API
cd backend
uvicorn snn_ai_optimizer.app:app --reload --port 8000

# Terminal 2: React Frontend
cd frontend
npm install
npm run dev
```

---

## Documentation
* [Research Integrity Rules (AGENTS.md)](file:///Users/sneha/Desktop/major-project-SNN/AGENTS.md)
* [Research Reproducibility Guide](file:///Users/sneha/Desktop/major-project-SNN/docs/REPRODUCIBILITY.md)
* [DEAP Data Pipeline Documentation](file:///Users/sneha/Desktop/major-project-SNN/docs/DEAP_DATA_PIPELINE.md)

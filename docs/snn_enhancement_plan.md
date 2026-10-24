# SNN Performance Enhancement Implementation Plan

This implementation plan outlines the prioritized engineering enhancements to elevate cross-subject generalization on the DEAP dataset while preserving 100% research integrity and backward compatibility per [`AGENTS.md`](file:///Users/sneha/Desktop/major-project-SNN/AGENTS.md).

---

## 1. Priority Hierarchy & Impact Analysis

| Priority | Component | Mechanism | Target Metric Impact | Target Files |
| :--- | :--- | :--- | :--- | :--- |
| **P1** | **Pre-Trial Baseline Subtraction** | Log relative power normalization against 3s pre-stimulus baseline: $\ln\left(\frac{P_{\text{task}} + \epsilon}{P_{\text{base}} + \epsilon}\right)$ | +5% to +8% Acc (cancels skull impedance & individual baseline variance) | [`eeg_features.py`](file:///Users/sneha/Desktop/major-project-SNN/backend/snn_ai_optimizer/features/eeg_features.py), [`deap_snn_train.py`](file:///Users/sneha/Desktop/major-project-SNN/backend/snn_ai_optimizer/snn/deap_snn_train.py) |
| **P2** | **Class-Weighted Focal Loss & Cosine LR** | $\text{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$ with $\gamma = 2.0$, plus Cosine Annealing scheduler | +0.05 to +0.08 Macro F1 (prevents skew towards 57% high-arousal majority) | [`deap_snn_train.py`](file:///Users/sneha/Desktop/major-project-SNN/backend/snn_ai_optimizer/snn/deap_snn_train.py) |
| **P3** | **Parametric LIF ($\text{PLIF}$) & Dynamic Traces** | Learnable membrane decay parameter $\tau_{\text{mem}}$ with ATan surrogate gradient; dynamic layer traversal in raster logger | +3% to +5% Acc, 0 crashes on arbitrary layer topologies | [`deap_snn_model.py`](file:///Users/sneha/Desktop/major-project-SNN/backend/snn_ai_optimizer/snn/deap_snn_model.py), [`inference.py`](file:///Users/sneha/Desktop/major-project-SNN/backend/snn_ai_optimizer/snn/inference.py) |
| **P4** | **Stochastic Rate Regularization** | Subtle zero-mean Gaussian jitter ($\sigma = 0.01$) applied strictly to Poisson rate features during training | +2% to +3% Generalization across unseen subjects | [`deap_snn_train.py`](file:///Users/sneha/Desktop/major-project-SNN/backend/snn_ai_optimizer/snn/deap_snn_train.py) |
| **P5** | **Automated Verification & Re-Training** | Run 99-test pytest suite, trigger real DEAP training, update `results/snn/metrics.json` | 100% test pass rate, verified real held-out metrics | `backend/tests/`, `results/snn/` |

---

## 2. File Modification Specifications

### 1. `backend/snn_ai_optimizer/snn/deap_snn_model.py`
- **Why:** `forward_with_traces` previously assumed hardcoded indices (`self.net[0], self.net[1], self.net[2]...`). Adding dropout or skip layers causes index misalignment.
- **Change:** Implement dynamic traversal by matching `Linear` and `neuron` modules in sequence. Ensure compatibility for both standard `LIFNode` and `ParametricLIFNode`.

### 2. `backend/snn_ai_optimizer/snn/inference.py`
- **Why:** `_load_model` must automatically detect whether a saved checkpoint was trained with `use_plif=True` or `use_plif=False` by inspecting state dictionary keys (checking for `*.w`), preventing state dictionary mismatch crashes.
- **Change:** Add auto-detection logic in `_load_model`.

### 3. `backend/snn_ai_optimizer/features/eeg_features.py`
- **Why:** Pre-stimulus baseline power carries subject-specific skull impedance and electrode placement variances.
- **Change:** Ensure `extract_windowed_band_powers` correctly extracts 3s pre-trial segments and applies log-ratio normalization when `subtract_baseline=True`.

### 4. `backend/snn_ai_optimizer/snn/deap_snn_train.py`
- **Why:** Training script needs to integrate baseline subtraction, focal loss, learnable membrane time constants, cosine learning rate scheduler, and training-phase stochastic regularization.
- **Change:** Update training loop and provenance recording dictionary to register all active hyperparameters.

---

## 3. Execution Plan

1. **Phase 1: Architecture Robustness & Auto-Detection** (Modify `deap_snn_model.py` and `inference.py`).
2. **Phase 2: Preprocessing & Training Enhancements** (Wire baseline subtraction, focal loss, and cosine schedule in `deap_snn_train.py`).
3. **Phase 3: Verification** (Execute unit tests and regression tests).
4. **Phase 4: Real DEAP Pipeline Execution** (Train the updated SNN and log genuine evaluation metrics).
5. **Phase 5: Full System Verification** (Verify API endpoints, background daemons, and UI).

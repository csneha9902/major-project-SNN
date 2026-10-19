# PROJECT RULES

This is a research project.

## CORE OBJECTIVE

Build an EEG-based adaptive learning prototype using:

1. EEG data
2. Spiking Neural Network (SNN) for affective-state estimation
3. Q-learning for adaptive task recommendation

---

# NON-NEGOTIABLE RULES

1. Never fabricate metrics.

2. Never hard-code accuracy, F1, AUC, precision, recall, latency, or other research results.

3. Never use synthetic data in the research training pipeline.

4. Synthetic data may only be used in explicitly labelled DEMO or SIMULATION code.

5. Never silently fall back from real data to synthetic data.

6. Never mix training subjects and test subjects.

7. Never commit the DEAP dataset to Git.

8. Never claim that DEAP directly labels "stress", "focused", "neutral", or "cognitive impairment".

9. Dataset labels must come from documented DEAP ratings.

10. Keep preprocessing identical between training and inference.

11. Do not create duplicate or incompatible preprocessing pipelines.

12. Do not add unrelated medical or hospital modules.

13. Do not add MRI processing unless explicitly requested.

14. Do not modify the main branch directly.

15. Active development should occur on the development branch.

16. Before making substantial architectural changes, explain which files will change and why.

17. After changing code, run the relevant tests.

18. Never delete files unless explicitly instructed.

19. Do not rewrite working code unnecessarily.

20. Preserve backward compatibility unless the architecture is intentionally being replaced.

---

# RESEARCH INTEGRITY

Every reported metric must be calculated from an actual held-out evaluation set.

Every model checkpoint must correspond to a reproducible training configuration.

Every experiment must record:

- dataset
- label definition
- preprocessing
- train/test split
- random seed
- model configuration
- evaluation metrics

---

# DATASET RULES

The DEAP dataset must be supplied locally.

Do not download or commit the dataset automatically.

Use a configurable DEAP_DATA_DIR environment variable.

Research training must use real DEAP data.

If DEAP is unavailable, do not silently substitute synthetic data.

Synthetic fixtures may be used only for unit tests.

---

# LABEL RULES

Initial research task:

Low Arousal:
arousal <= 5

High Arousal:
arousal > 5

Do not describe these labels as:

- Stress
- Focused
- Neutral
- Cognitive impairment

unless a separate validated labeling methodology is implemented and documented.

---

# APPLICATION MODES

## RESEARCH MODE

Uses the real DEAP dataset and trained models.

## DEMO MODE

Synthetic streaming data may be used.

Demo output must be clearly labelled as DEMO.

Demo results must never be presented as research results.

---

# GIT RULES

The main branch is the stable branch.

The development branch is the active development branch.

Do not modify main directly.

Before committing:

1. Run relevant tests.
2. Check git status.
3. Review changed files.
4. Commit only intentional changes.

---

# CURRENT IMPLEMENTATION PRIORITY

The project should be rebuilt in this order:

1. Real DEAP data pipeline
2. Real SVM baseline
3. Real SNN training
4. Real SNN inference
5. Genuine Q-learning recommendation
6. Backend/frontend integration
7. Repository cleanup
8. Reproducibility and experiment automation

Do not skip directly to frontend or presentation improvements before the research pipeline is valid.
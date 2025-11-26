# Component X LSTM – Cascade Implementation Plan

This document tracks the phases for the new two-stage cascade pipeline. All work described here is separate from the original baseline. Reports will live in `docs/component-x-lstm-cascade-journal.md`, and the dedicated MLflow experiment name is `component-x-lstm-cascade`.

## Phase 0 – Prep & Data Artefacts
- Finalize cascade assumptions (Model A threshold = 0.3, oversampling strategy, validation metrics).
- Generate cascade-specific NPZs:
  - `data/processed/train_sequences_binary.npz` – binary labels {0, fault}, contains class-1–4 oversampling.
  - `data/processed/train_sequences_diag.npz` – multi-class {1,2,3,4} only, rebuilt windows centered near failures.
- Document CLI commands to rebuild these datasets in `docs/component-x-lstm-cascade.md`.

## Phase 1 – Training Script (`train_lstm_cascade.py`)
- Create a new entry point that shares existing dataset/feature modules.
- CLI arguments:
  - `--stage {a,b}` to select detector vs. diagnoser.
  - Stage-specific configs (oversampling flag for A, label smoothing for B, etc.).
- Stage A (detector):
  - Binary output, supports oversampling, focal loss option, recall-based promotion.
- Stage B (diagnoser):
  - Multiclass {1,2,3,4} training, standard cross-entropy (optional mild weights).
- MLflow:
  - Default experiment = `component-x-lstm-cascade`.
  - Artifact roots: `artifacts/cascade/detector` and `artifacts/cascade/diagnoser`.

## Phase 2 – Evaluation (`evaluate_cascade.py`)
- Implement a companion script that can:
  1. Evaluate Model A alone (ROC/PR curves, recall at threshold 0.3).
  2. Evaluate Model B alone on the balanced validation split.
  3. Run the cascade end-to-end on the official validation set and produce:
     - 5-class confusion matrix.
     - Combined accuracy/macro-F1.
     - Detector recall/precision stats for auditing.

## Phase 3 – Documentation & Journal
- Create `docs/component-x-lstm-cascade-journal.md` to log experiments (detector runs, diagnoser runs, combined evaluations).
- Update `docs/component-x-lstm-cascade.md` with:
  - Data build commands.
  - Training commands for Stage A/B (default epochs, LR, oversampling settings).
  - Threshold selection guidance.
- When new cascade runs complete, log them in the new journal with MLflow run IDs.

## Phase 4 – Iteration & Comparison
- After initial runs, compare cascade results vs. baseline (Run10 etc.).
- If cascade meets targets, plan next steps (inference integration, cost-based threshold tuning).

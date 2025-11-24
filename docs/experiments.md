# Component X LSTM – Experiment Tracking Guide

Phase 9 introduces MLflow for reproducible runs. This guide explains how to start a new experiment, how runs are named, and which artifacts/metrics must be reviewed before promoting a model.

## 1. Prerequisites

1. Install MLflow inside the training environment (already added to `requirements.txt`).
2. Decide where you want the tracking data to live:
   - **Local default:** leave `--mlflow-tracking-uri` unset; MLflow writes to `./mlruns`.
   - **Remote server:** start your preferred MLflow server and pass its URI via `--mlflow-tracking-uri`.

> ⚠️ Heavy training jobs still run on the GPU server/Colab instance. Launch the MLflow server wherever you have persistent storage, then execute the `train_lstm` command from that machine.

## 2. Naming convention

Runs are grouped under the `component-x-lstm` experiment. Each run can optionally receive a manual name via `--mlflow-run-name`, but the recommended convention is:

```
{data_version}-{window_size}-{model_tag}-{run_id}
```

Example: `val2025w48-l128-biLSTM-baseline-run4`.

## 3. Launching a run

```bash
python -m src.training.train_lstm \
  --train-npz data/train_sequences.npz \
  --val-npz data/val_sequences.npz \
  --feature-stats artifacts/feature_stats.json \
  --output-dir artifacts_colab_run5 \
  --batch-size 64 \
  --epochs 20 \
  --hidden-size 128 \
  --num-layers 2 \
  --bidirectional \
  --weighted-loss \
  --device cuda \
  --mlflow \
  --mlflow-run-name val2025w48-l128-biLSTM-baseline-run5
```

What gets logged:

- **Parameters:** every CLI flag (data paths, hyperparameters, weighting options, etc.).
- **Metrics per epoch:** training loss/accuracy, validation loss/accuracy/macro-F1, and learning rate.
- **Artifacts:** `metrics.csv`, `last.pt`, `best.pt`, and the `feature_stats.json` referenced by the run.

## 4. Promotion criteria

For a model to be considered “best” and promoted to `artifacts/best.pt`, it must:

1. Achieve a higher validation macro-F1 than the previously promoted run.
2. Have its evaluation bundle exported via `src/training/evaluate.py` and attached to the MLflow run (upload `artifacts/evaluations/<run>/` manually if needed).
3. Pass qualitative inspection of confusion/PR/ROC plots to ensure minority classes improved without catastrophic regressions.

## 5. Reviewing historical runs

1. Launch the MLflow UI (e.g., `mlflow ui --backend-store-uri ./mlruns`).
2. Filter by experiment `component-x-lstm`.
3. Compare runs using:
   - Macro-F1 (primary promotion metric).
   - Weighted-F1/accuracy (sanity checks).
   - Attached artifacts (metrics CSV, checkpoints, evaluation bundles).

Keeping MLflow in sync with notebook/journal notes ensures every run is reproducible and traceable, satisfying the Phase 9 objectives.

# SCANIA Component X – LSTM Classification Project Plan

Goal: build a production-ready LSTM-based classifier that maps vehicle time-series readouts to the 5-class “proximity-to-failure” labels defined in the Component X dataset.

Target label (per vehicle sample):

- Class 0: time-to-failure > 48 time steps
- Class 1: 48–24 steps before failure
- Class 2: 24–12 steps before failure
- Class 3: 12–6 steps before failure
- Class 4: 6–0 steps before failure

We will treat each training example as a variable-length multivariate time series (one vehicle, one window of its timeline) with a single 5-class label.

---

## Phase 0 – Project Setup

**Objectives**
- Create a clean, reproducible Python ML project structure.
- Use plain PyTorch as the base training stack (no Lightning), keeping the code explicit and easy to debug.
- Make it easy to add monitoring/serving later.

**Tasks**
- Set up Python environment (e.g., `uv` or `poetry` + Python 3.11).
- Add core dependencies: `pandas`, `numpy`, `pyarrow` or `polars`, `torch`, `torchvision`, `torchaudio`, `scikit-learn`, `matplotlib`/`seaborn`.
- (Optional) Add experiment tracking: `mlflow` or `wandb`.
- Scaffold folders:
  - `src/` – all code
  - `src/data/` – data loading & preprocessing
  - `src/features/` – feature transforms, windowing
  - `src/models/` – model architectures and heads
  - `src/training/` – training loop, evaluation, metrics
  - `src/config/` – YAML/JSON configs (paths, hyperparams)
  - `notebooks/` – exploratory work, quick checks
  - `artifacts/` – local models, plots, reports (gitignored)

Deliverable: minimal project skeleton, `pyproject.toml`/`requirements.txt`, and a `src` package ready for implementation.

---

## Phase 1 – Data Understanding & Schema

**Objectives**
- Translate the Scientific Data article + `data/dataset-overview` into a concrete schema.
- Validate that local files match expectations, starting from `data/sample_first_3_vehicles.csv`.

**Tasks**
- Define canonical schemas tied to the concrete CSV files:
  - Train operational readouts (`train_operational_readouts.csv`): `vehicle_id`, `time_step`, histogram bins, counters.
  - Validation operational readouts (`validation_operational_readouts.csv`) and test operational readouts (`test_operational_readouts.csv`): same schema as train, without TTE labels.
  - Train TTE labels (`train_tte.csv`): `vehicle_id`, `length_of_study_time_step`, `in_study_repair`.
  - Validation labels (`validation_labels.csv`) and test labels (`test_labels.csv`): `vehicle_id`, 5-class proximity label (0–4), plus any timing reference columns.
  - Specifications tables (`train_specifications.csv`, `validation_specifications.csv`, `test_specifications.csv`): `vehicle_id` + 8 categorical spec variables (Cat0…CatK).
- Create a small data dictionary in `docs/` describing each field (even if anonymized).
- Implement a simple `DatasetSummary` script/notebook that:
  - Confirms row counts, vehicle counts, and missingness patterns.
  - Examines distribution of sequence lengths per vehicle.
  - Checks class imbalance for the 5-class label.

Deliverable: `docs/data-schema.md` and a short notebook/script that prints basic stats, using `data/sample_first_3_vehicles.csv` as a smoke test.

---

## Phase 2 – Label Engineering for 5-Class Problem

**Objectives**
- Make the 5-class “proximity-to-failure” label consistent across train/val/test.
- Derive train labels from TTE information if not already provided.

**Tasks**
- From `data/dataset-overview`, the article, and the provided label files, codify the class mapping:
  - Map time-to-failure windows to classes {0,1,2,3,4} exactly as specified.
- Implement a label-building utility:
  - Inputs: per-vehicle TTE info from `train_tte.csv` + chosen reference time step, combined with `train_operational_readouts.csv`.
  - Outputs: one row per labeled example: `vehicle_id`, `reference_time_step`, `target_class`.
- Ensure consistency with publicly provided validation/test labels:
  - Cross-check that the generated training labels follow the same window logic as `validation_labels.csv` and `test_labels.csv` using sample records.
  - If only ranges are known, mimic article’s procedure: select a random last readout and assign class by its distance to failure.
- Handle censored vehicles (no failure):
  - Decide policy (most likely map them to class 0 if far from failure or exclude from classification training).
  - Document rationale in `docs/labeling-decisions.md`.

Deliverable: reusable `src/data/labels.py` module that can generate a labeled table for training the 5-class classifier.

---

## Phase 3 – Sequence Construction & Sampling Strategy

**Objectives**
- Turn each vehicle’s irregular time-series into supervised LSTM-ready sequences.
- Decide on sequence length, padding, and sampling strategy.

**Design decisions**
- Sequence definition:
  - Use a sliding or anchored window of length `L` time steps (e.g., 64–128).
  - For each labeled `reference_time_step`, build a window of the preceding `L` steps (or as many as available).
- Variable-length handling:
  - Pad shorter sequences at the start with zeros or feature-wise means.
  - Use an attention mask or sequence lengths to ignore padded positions.
- Sampling:
  - Given class imbalance, oversample minority-class windows or use class-weighting in loss.
  - Optionally downsample class 0 vehicles or restrict number of windows per vehicle.

**Tasks**
- Implement windowing utilities in `src/features/windowing.py`:
  - Functions: `build_sequences(df_op, df_labels, window_size, stride, min_length, max_windows_per_vehicle)`.
  - Return: tensors/arrays `(num_samples, seq_len, num_features)` + `(num_samples,)` labels + sequence lengths.
- Visual sanity checks:
  - Plot a few example sequences per class to validate that signals look sensible.

Deliverable: deterministic, unit-testable sequence builder that feeds into PyTorch datasets.

---

## Phase 4 – Feature Engineering & Normalization

**Objectives**
- Prepare numerical features for stable LSTM training.
- Apply consistent transformations across train/val/test and between sequences and any non-sequential metadata.

**Tasks**
- Separate feature groups:
  - Histogram bins (6 variables expanded into tens of columns).
  - Counter features (8 variables).
  - Optional: truck specification categorical features (to be appended as static features).
- Preprocessing:
  - Compute per-feature statistics on train only (mean, std, min/max).
  - Apply standardization (z-score) or log-transform where appropriate (e.g., counters).
  - Handle missingness (<1%): simple imputation (e.g., forward-fill per vehicle, then global mean).
- Implement a `FeatureTransformer`:
  - Encapsulate fit/transform logic in `src/features/transformer.py`.
  - Save fitted parameters to disk (e.g., `artifacts/feature_stats.json`) for reuse in inference.

Deliverable: end-to-end preprocessing pipeline that turns raw CSVs into ready-to-batch tensors.

---

## Phase 5 – Dataset & Dataloader Implementation

**Objectives**
- Wrap sequence and feature logic in clean PyTorch `Dataset`/`DataLoader` abstractions.

**Tasks**
- Implement `ComponentXSequenceDataset` in `src/data/datasets.py`:
  - Loads preprocessed data or generates windows on-the-fly.
  - Returns `(sequence_tensor, label, seq_length, optional_static_features)` per item.
- Implement DataLoaders in `src/data/dataloaders.py`:
  - Train loader with shuffling and class-aware sampling (e.g., `WeightedRandomSampler`).
  - Val/test loaders with deterministic ordering.
  - Custom collate function to handle variable-length sequences (if using packing).
- Add a config-driven interface:
  - E.g., `data: {window_size: 64, batch_size: 128, num_workers: 4, ...}` in a YAML.

Deliverable: reproducible train/val/test DataLoaders, wired into the training script.

---

## Phase 6 – LSTM Model Design

**Objectives**
- Define a robust sequence model that balances performance and overfitting risk.

**Baseline architecture**
- Input: `num_features` per time step.
- LSTM stack:
  - 1–2 bidirectional LSTM layers (e.g., hidden size 128–256).
  - Optional dropout between layers.
  - Use last hidden state or pooled representation (e.g., mean/max pooling across time).
- Classification head:
  - Fully connected layers (e.g., [hidden_dim -> 128 -> 64 -> 5]).
  - Non-linearities (ReLU/GELU), batch/layer normalization as needed.
  - Output logits for 5 classes.

**Tasks**
- Implement model class in `src/models/lstm_classifier.py`.
- Add configuration for model hyperparameters (hidden size, layers, dropout).
- (Optional) Experiment with variants:
  - GRU instead of LSTM.
  - Temporal convolutional network (TCN) baseline for comparison.

Deliverable: a modular PyTorch model that can be easily swapped or extended.

---

## Phase 7 - Training Loop, Loss, and Metrics

**Objectives**
- Train the LSTM model with appropriate loss and metrics, handling severe class imbalance.

**Tasks**
- Implement core training script in `src/training/train_lstm.py`:
  - Handles argument parsing or config loading.
  - Instantiates data, model, optimizer, scheduler.
  - Runs training/validation loops with robust checkpointing and resume support.
- Loss & metrics:
  - Use cross-entropy with class weights (`1 / class_freq`) or focal loss.
  - Track accuracy, macro F1, per-class recall, and confusion matrix.
  - Log metrics per epoch and optionally per class.
- Checkpointing:
  - Save best model by validation macro F1 or weighted F1.
  - Save both a `best.pt` (best validation metric) and `last.pt` (most recent state).
  - Store full training state in checkpoints (model weights, optimizer, scheduler, epoch, random seeds where possible).
  - Add a `--resume-from` (or config flag) that reloads a checkpoint and continues training after interruptions.

Deliverable: reproducible training run (single command) that outputs metrics and both resumable and best-model checkpoints.

---

## Phase 8 – Evaluation, Error Analysis, and Calibration

**Objectives**
- Understand where the model performs well/poorly, especially on minority classes.

**Tasks**
- On validation and test sets:
  - Compute detailed classification report by class.
  - Plot confusion matrix and per-class ROC/PR curves.
- Imbalance-focused checks:
  - Inspect recall for classes 1–4.
  - Evaluate macro and weighted F1.
- Calibration:
  - Assess probability calibration (e.g., reliability diagrams).
  - If needed, apply temperature scaling or isotonic regression using a held-out set.
- Save evaluation outputs:
  - Write a short report notebook/script into `notebooks/evaluation.ipynb` or `src/training/evaluate.py`.

**Implementation**
- `src/training/evaluate.py` now produces all required artifacts (classification report, confusion matrix visuals, ROC/PR curves, and a reliability diagram) for any checkpoint:

```bash
python -m src.training.evaluate \
  --npz data/val_sequences.npz \
  --checkpoint artifacts_colab_run4/best.pt \
  --feature-stats artifacts/feature_stats.json \
  --output-dir artifacts/evaluations/run4_val
```

The command above writes JSON/CSV metrics plus PNG plots so the journal and thesis can embed Phase 8 findings without manual notebook work.

Deliverable: documented evaluation results, plots, and insights to guide next iterations.

---

## Phase 9 - Experiment Tracking & Reproducibility

**Objectives**
- Make it easy to rerun experiments and compare configurations.

**Tasks**
- Choose an experiment tracker (MLflow or Weights & Biases).
- Integrate logging into training:
  - Hyperparameters, metrics, artifacts (plots, confusion matrices).
  - Model checkpoints and feature statistics.
- Define naming convention for experiments:
  - Include data version, window size, model architecture, and run ID.
- Write a short `docs/experiments.md` specifying:
  - How to start new experiments.
  - Which metrics determine "promotion" to best model.

Deliverable: fully reproducible experiment setup with logs and artifacts per run.

**Implementation**
- Integrated optional MLflow logging into `src/training/train_lstm.py` (flags: `--mlflow`, `--mlflow-tracking-uri`, `--mlflow-experiment`, `--mlflow-run-name`).
- Every epoch now streams metrics to MLflow and, when the run finishes, the script logs key artifacts (`metrics.csv`, `last.pt`, `best.pt`, and the referenced `feature_stats.json`).
- Authored `docs/experiments.md` covering run naming, launch instructions, and promotion criteria so collaborators can follow the same workflow.

---

## Phase 10 – Inference Pipeline & Serving-Ready Interface (Future Work)

**Objectives**
- Prepare for eventual deployment (batch or real-time), even if not fully implemented yet.

**Tasks**
- Implement `src/inference/predict.py`:
  - Given raw sequences for one or more vehicles, apply the same feature pipeline and return class probabilities.
- Define a simple serving contract:
  - Input JSON schema (vehicle_id, time_step, features).
  - Output schema (class probabilities, predicted class, optional explanations).
- (Optional future step) Wrap model in a FastAPI app or export to ONNX for optimized inference.

Deliverable: inference utilities and clear interfaces that can be integrated into a microservice later.

---

## Phase 11 – Documentation & Next Iterations

**Objectives**
- Keep the project understandable and maintainable.

**Tasks**
- Maintain `README.md` with:
  - Problem description.
  - How to set up the environment.
  - How to run training and evaluation.
- Keep `docs/` updated with:
  - Data schema and labeling decisions.
  - Key experimental findings.
  - Known limitations and ideas for future models (e.g., transformers, survival models).

Deliverable: minimal but clear documentation so future you (or collaborators) can quickly continue iterating on the Component X LSTM classifier.

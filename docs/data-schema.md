# Component X Dataset – Schema Overview

This document summarizes the expected schemas for the SCANIA Component X dataset files used in the LSTM classification project.

The dataset consists of three main types of tables:
- Operational readouts (time-series per vehicle)
- Repair / time-to-event (TTE) labels (train only)
- Proximity-to-failure labels (validation/test)
- Vehicle specifications (categorical metadata)

All tables are assumed to live in the local `data/` directory.

---

## 1. Operational Readouts (Time-Series)

Files:
- `train_operational_readouts.csv`
- `validation_operational_readouts.csv`
- `test_operational_readouts.csv`

Grain:
- One row per `(vehicle_id, time_step)` with multiple sensor/derived features.

Core columns:
- `vehicle_id` – integer or string identifier for each truck/engine.
- `time_step` – relative time index within a vehicle’s observation window.
- Histogram features (6 base variables with multiple bins each):
  - e.g., `167_0 ... 167_9`, `272_0 ... 272_9`, `291_0 ... 291_10`, `158_0 ... 158_9`, `459_0 ... 459_19`, `397_0 ... 397_35`
  - Each column represents the value/count in a particular bin.
- Counter features (8 variables, 1 column each):
  - e.g., `171_0`, `666_0`, `427_0`, `837_0`, `309_0`, `835_0`, `370_0`, `100_0`
  - Counters are roughly monotonically increasing usage/age signals.

Notes:
- The exact column names and counts should be confirmed by inspecting the CSV headers (e.g., in `data/sample_first_3_vehicles.csv`).
- Missingness in operational readouts is expected to be <1% per feature.

---

## 2. Train TTE Labels

File:
- `train_tte.csv`

Grain:
- One row per `vehicle_id`.

Columns:
- `vehicle_id` – matches operational readouts.
- `length_of_study_time_step` – final observation time for that vehicle (relative time index).
- `in_study_repair` – indicator of observed repair/failure (1 = event occurred at `length_of_study_time_step`, 0 = censored).

Usage:
- This table is used to derive training labels for the 5-class proximity-to-failure problem by combining it with `train_operational_readouts.csv` and defining where in the trajectory to place the labeled window.

---

## 3. Proximity Labels (Validation/Test)

Files:
- `validation_labels.csv`
- `test_labels.csv`

Grain:
- One row per labeled vehicle example in the validation/test splits.

Expected columns:
- `vehicle_id` – matches operational readouts and specifications.
- `class_label` (or similar) – integer in {0, 1, 2, 3, 4} representing proximity to failure:
  - 0: >48 time steps before failure
  - 1: 48–24 time steps before failure
  - 2: 24–12 time steps before failure
  - 3: 12–6 time steps before failure
  - 4: 6–0 time steps before failure
- Optional timing reference columns:
  - e.g., `reference_time_step`, `time_to_failure`, depending on the official schema.

Usage:
- These tables provide the ground-truth labels for validation and test sets and are the reference for how training labels must be derived from `train_tte.csv`.

---

## 4. Specifications (Categorical Metadata)

Files:
- `train_specifications.csv`
- `validation_specifications.csv`
- `test_specifications.csv`

Grain:
- One row per `vehicle_id`.

Columns:
- `vehicle_id` – key.
- 8 categorical specification variables:
  - Anonymized as something like `Cat0 ... Cat7` (exact names to be confirmed).

Usage:
- Optional static features:
  - Can be joined to per-vehicle time-series and used as additional inputs (e.g., appended as static embeddings to LSTM representations).
  - Distributions are expected to be similar across splits.

---

## 5. Sample Data File

File:
- `sample_first_3_vehicles.csv`

Notes:
- This is a convenience sample containing data for the first three vehicles from the operational readouts.
- Used for:
  - Sanity-checking parsing and schema assumptions.
  - Testing summary scripts without loading the full dataset.

---

## 6. Next Steps for Schema Validation

To fully validate the schema against local files:
- Run a lightweight summary script (see `src/data/dataset_summary.py`) to:
  - Print column names and basic stats for each CSV.
  - Check for missing columns or unexpected types.
  - Confirm vehicle counts and sequence length distributions.

Any discrepancies between this document and actual CSV headers should be reconciled and documented here as they are discovered.

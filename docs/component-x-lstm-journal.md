# Component X LSTM Project – Phase-by-Phase Journal

This journal captures key findings and decisions for each project phase so the full modeling timeline can be reconstructed later.

---

## Phase 0 – Project Setup

**Date:** (initial scaffolding)  
**Scope:** Repository layout, environment, dependencies, and high-level tooling choices.

**Environment & Tools**
- Python: `3.13.3` (current local interpreter).
- Virtual environment: `.venv/` created at repo root using `python -m venv .venv`.
- Data science stack installed into `.venv`:
  - `numpy`, `pandas`, `pyarrow`, `matplotlib`, `seaborn`, `scikit-learn`.
  - PyTorch is specified in `requirements.txt` but not yet installed (no official 3.13 wheels at this time; training will likely require a 3.12 environment or future PyTorch release for 3.13).

**Repository Structure**
- Root:
  - `.gitignore` configured to exclude Python bytecode, virtual envs, logs, ML artifacts (`mlruns/`, `wandb/`, `models/`, `checkpoints/`, `runs/`), and large data layers (`data/raw/`, `data/interim/`, `data/processed/`, `data/external/`).
  - `requirements.txt` listing core Python and ML dependencies.
  - `README.md` with a short description and a pointer to `docs/component-x-lstm-plan.md`.
- Code & notebooks:
  - `src/` – Python package root.
    - `src/data/` – data loading and summary utilities (contains `dataset_summary.py`).
    - `src/features/` – feature engineering and windowing utilities (planned).
    - `src/models/` – model architectures, including the future LSTM classifier.
    - `src/training/` – training and evaluation scripts, including checkpointing logic (planned).
    - `src/config/` – configuration management (planned).
  - `notebooks/` – placeholder for EDA and reporting notebooks (`.gitkeep` added to keep the folder in version control).
  - `artifacts/` – placeholder for local models, plots, and reports (`.gitkeep`, gitignored at folder level).

**Architectural Choices**
- Framework: plain **PyTorch** (no Lightning), to keep training loops explicit and easy to debug.
- Training robustness: plan explicitly calls for:
  - Saving both `best.pt` (best validation metric) and `last.pt` (most recent state).
  - Checkpoints containing model weights, optimizer, scheduler, epoch, and (where feasible) random seed state.
  - A `--resume-from` flag in `src/training/train_lstm.py` so training can be resumed seamlessly after interruption.
- Experiment tracking: MLflow or Weights & Biases are considered optional and will be integrated in a later phase.

**Outcome**
- Phase 0 deliverables from the plan are in place:
  - A minimal but production-oriented project skeleton.
  - A Python environment ready for data exploration and preprocessing.
  - Documentation and directory layout aligned with the multi-phase plan in `docs/component-x-lstm-plan.md`.

---

## Phase 1 – Data Understanding & Schema

**Date:** (initial analysis with local CSVs in `data/`)  
**Scope:** Confirm schemas, basic statistics, and imbalance characteristics using sampled reads only (no full-dataset loads).

### 1. Files Present in `data/`

At this stage, the following CSVs are available:
- `sample_first_3_vehicles.csv` – sample of operational readouts for first 3 vehicles.
- `train_operational_readouts.csv` – train split operational time-series.
- `train_tte.csv` – per-vehicle time-to-event labels (train).
- `train_specifications.csv` – per-vehicle categorical specs (train).
- `validation_labels.csv` – 5-class proximity labels for validation split.

Additional documentation files:
- `dataset-overview` / `dataset-overview.md` – human-written summary of the scientific article and public dataset description.
- `table-of-prediction-cost` / `table-of-prediction-cost.md` – cost matrix and evaluation considerations (not yet used in code).

### 2. Summary Script & Methodology

To avoid loading the full dataset into memory, a small CLI tool was implemented:
- File: `src/data/dataset_summary.py`
- Command pattern:
  - `.\.venv\Scripts\python.exe -m src.data.dataset_summary --file <path> --n-rows <cap>`
- Behavior:
  - Reads at most `n_rows` from the CSV (default 100,000).
  - Prints:
    - Row count loaded and number of columns.
    - Column names and dtypes.
    - Unique `vehicle_id` count and top-10 vehicles by row count (if present).
    - `time_step` descriptive statistics (if present).
    - Label distribution for a label column if detected (`class`, `class_label`, `proximity_label`, or `target_class`).
    - TTE stats and event/censoring distribution if `in_study_repair` and `length_of_study_time_step` are present.
    - Non-zero missing-value counts per column in the sampled rows.

This tool provides a reproducible way to regenerate Phase 1 findings without reading the entire dataset.

### 3. Sample Operational Data (`sample_first_3_vehicles.csv`)

Command:
- `python -m src.data.dataset_summary --file data/sample_first_3_vehicles.csv --n-rows 500`

Findings:
- Rows loaded: **276** (all rows in the file).
- Columns: **107**, including:
  - Identifiers: `vehicle_id`, `time_step`.
  - Counters: `171_0`, `666_0`, `427_0`, `837_0`, `309_0`, `835_0`, `370_0`, `100_0`.
  - Histogram bins:
    - `167_0`–`167_9` (10 bins),
    - `272_0`–`272_9` (10 bins),
    - `291_0`–`291_10` (11 bins),
    - `158_0`–`158_9` (10 bins),
    - `459_0`–`459_19` (20 bins),
    - `397_0`–`397_35` (36 bins).
- Dtypes:
  - `vehicle_id`: `int64`
  - `time_step` and all feature columns: `float64`
- Vehicle coverage:
  - Unique vehicles: **3**
  - Row counts per vehicle:
    - Vehicle `0`: 172 rows
    - Vehicle `3`: 71 rows
    - Vehicle `2`: 33 rows
  - This confirms **per-vehicle sequence length variability**, consistent with the article’s description.
- `time_step` statistics:
  - Count: 276
  - Min: 0.2
  - Max: 507.4
  - Mean: ≈ 214.8
  - Std: ≈ 141.0
  - Quartiles:
    - 25%: 96.75
    - 50% (median): 203.9
    - 75%: 301.1
  - Interpretation:
    - Vehicles are observed over fairly long windows (hundreds of time-steps).
    - Early and late observation times are both represented, confirming a wide observation horizon per vehicle.
- Missing values (only those with >0 missing in this sample):
  - Counters and histogram bases with notable missingness:
    - `427_0`: 113 missing
    - `370_0`: 113 missing
    - `100_0`: 113 missing
  - Histogram `167_*` (10 bins): each has **194** missing.
  - Histogram `291_*` (11 bins): each has **2** missing.
  - Histogram `459_*` (20 bins): each has **3** missing.
  - Histogram `397_*`: no missing values in this 3-vehicle sample.
  - Interpretation:
    - Even in a tiny sample, several features can be absent for specific vehicles or time-steps.
    - Models must handle sparse histograms gracefully (imputation/forward-fill + robust normalization).

### 4. Train Operational Readouts (`train_operational_readouts.csv`)

Command:
- `python -m src.data.dataset_summary --file data/train_operational_readouts.csv --n-rows 100000`

Findings (based on first 100,000 rows only):
- Rows loaded: **100,000** (capped for performance).
- Columns: **107**, same as the sample file; schema is consistent.
- Dtypes:
  - `vehicle_id`: `int64`
  - `time_step` and all features: `float64`
- Vehicles in the sample:
  - Unique `vehicle_id` in the first 100k rows: **1,491**.
  - Top-10 vehicles by row count (from this sample):
    - e.g., vehicle 293: 237 rows, 228: 236 rows, 329: 231 rows, 1888: 230 rows, etc.
  - Interpretation:
    - Per-vehicle sequence lengths vary substantially, with many vehicles having 150–250 observations.
    - Supports the need for variable-length sequence handling in the LSTM (padding + masking or packing).
- `time_step` statistics (sample):
  - Count: 100,000
  - Min: 0.0
  - Max: 507.4
  - Mean: ≈ 177.31
  - Std: ≈ 123.61
  - Quartiles:
    - 25%: 72.0
    - 50% (median): 153.6
    - 75%: 269.05
  - Interpretation:
    - Distribution of observation times is broadly similar to the article’s reported study horizons.
    - Many observations occur mid-horizon, with long tails reaching ~500 time-steps.
- Missing values (counts in first 100k rows; representative but not exhaustive):
  - Counters:
    - `666_0`: 2 missing
    - `427_0`: 368 missing
    - `837_0`: 2 missing
    - `309_0`: 2 missing
    - `370_0`: 367 missing
    - `100_0`: 367 missing
  - Histogram groups:
    - `167_*` (10 bins): each with 800 missing.
    - `272_*` (10 bins): each with 23 missing.
    - `291_*` (11 bins): each with 596 missing.
    - `158_*` (10 bins): each with 81 missing.
    - `459_*` (20 bins): each with 349 missing.
    - `397_*` (36 bins): each with 72 missing.
  - Interpretation:
    - Missingness is non-uniform across features.
    - Some histogram families are frequently absent for certain vehicles/time ranges (e.g., `167_*`, `291_*`).
    - Any preprocessing pipeline must:
      - Detect missingness per feature.
      - Apply a consistent imputation strategy (e.g., forward-fill per vehicle then global mean, or simple constant imputation) so the LSTM sees dense sequences.

### 5. Train TTE Labels (`train_tte.csv`)

Command:
- `python -m src.data.dataset_summary --file data/train_tte.csv`

Findings:
- Rows loaded: **23,550**.
- Columns (3):
  - `vehicle_id` (`int64`)
  - `length_of_study_time_step` (`float64`)
  - `in_study_repair` (`int64`)
- Vehicle coverage:
  - Unique `vehicle_id`: **23,550** (one row per vehicle), consistent with the article.
- TTE indicator distribution (`in_study_repair`):
  - Events (`1`): **2,272** (**9.648%**).
  - Censored (`0`): **21,278** (**90.352%**).
  - Interpretation:
    - Strong event/censoring imbalance.
    - Only about 1 in 10 vehicles experiences an observed failure/repair within the study window.
    - When converting this to 5-class proximity labels, near-failure classes will be even rarer.
- `length_of_study_time_step` statistics:
  - Count: 23,550
  - Min: 73.4
  - Max: 510.0
  - Mean: ≈ 240.35
  - Std: ≈ 88.78
  - Quartiles:
    - 25%: 163.8
    - 50%: 218.2
    - 75%: 312.0
  - Interpretation:
    - Study horizons span roughly 70–510 time-steps with a mean near 240.
    - These stats align with published figures (e.g., Figure 4 in the article), confirming data integrity.
- Missing values:
  - None detected in the sampled rows.

### 6. Validation Labels (`validation_labels.csv`)

Command:
- `python -m src.data.dataset_summary --file data/validation_labels.csv`

Findings:
- Rows loaded: **5,046**.
- Columns (2):
  - `vehicle_id` (`int64`)
  - `class_label` (`int64`)
- Vehicle coverage:
  - Unique `vehicle_id`: **5,046**.
  - Each vehicle appears exactly once (one labeled example per vehicle).
- Label distribution (`class_label`):
  - Class 0: **4,910** (**97.305%**)
  - Class 1: **16** (**0.317%**)
  - Class 2: **14** (**0.277%**)
  - Class 3: **30** (**0.595%**)
  - Class 4: **76** (**1.506%**)
- Interpretation:
  - The validation split is **extremely imbalanced** towards class 0 (>48 time-steps before failure).
  - Near-failure classes (especially 1 and 2) are very rare, on the order of tens of examples.
  - This matches the counts reported in the article and in `data/dataset-overview`, validating that local files follow the official construction.
  - For evaluation, macro-averaged metrics and per-class recall will be more informative than accuracy alone.
- Missing values:
  - None detected.

### 7. Train Specifications (`train_specifications.csv`)

Command:
- `python -m src.data.dataset_summary --file data/train_specifications.csv`

Findings:
- Rows loaded: **23,550**.
- Columns (9):
  - `vehicle_id` (`int64`)
  - `Spec_0`–`Spec_7` (all `object` / categorical).
- Vehicle coverage:
  - Unique `vehicle_id`: **23,550** (one row per vehicle).
- Missing values:
  - None detected in the sampled rows.
- Interpretation:
  - This table is clean and matches expectations of 8 categorical spec variables per vehicle.
  - Specs can be joined on `vehicle_id` to enrich the LSTM with static metadata (e.g., by using learned embeddings for each categorical spec).

### 8. Cross-File Consistency & Design Implications

Cross-checks:
- `train_tte.csv` and `train_specifications.csv` both have 23,550 unique `vehicle_id` values, consistent with the train operational data description in the article.
- `validation_labels.csv` has 5,046 unique vehicles, and the extreme imbalance matches published counts.
- Operational readouts include the full set of 14 anonymous features (6 histogram families, 8 counters) expanded into 107 columns, matching the dataset description.

Implications for later phases:
- **Sequence modeling:** variable-length per-vehicle sequences require padding and potentially packed sequences in PyTorch.
- **Imputation:** non-trivial, structured missingness across histogram families necessitates explicit imputation rules.
- **Imbalance handling:** both the TTE space and the 5-class proximity labels are heavily skewed, so future phases must:
  - Use class-weighted loss or focal loss.
  - Emphasize recall for minority classes in evaluation and possibly oversample rare classes in training.
- **Feature groups:** clear separation between counters, histograms, and specifications informs how we design preprocessing and model inputs (e.g., treat specs as static embeddings combined with dynamic sequence representations).

---

## Phase 2 – Label Engineering for 5-Class Problem

**Date:** (initial training label construction)  
**Scope:** Define the 5-class time-to-failure mapping, build training labels from TTE + operational data, and document labeling policies.

### 1. Proximity Class Mapping

Time-to-failure (TTF, in `time_step` units) is mapped to classes as:

- Class 4: `0 <= TTF <= 6`
- Class 3: `6 < TTF <= 12`
- Class 2: `12 < TTF <= 24`
- Class 1: `24 < TTF <= 48`
- Class 0: `TTF > 48` or TTF unknown (censored)

Implementation:

- Functions in `src/data/labels.py`:
  - `time_to_proximity_class(time_to_failure: float) -> int`
  - `time_to_proximity_class_array(time_to_failure: np.ndarray) -> np.ndarray`
- Negative TTF values (reference after failure) are treated as invalid and are excluded from label generation.
- TTF `NaN` is interpreted as “unknown” and mapped to class 0.

This mapping follows the windows described in `data/dataset-overview.md` and the underlying Scientific Data article.

### 2. Training Label Construction Inputs

Files used:

- `train_tte.csv`:
  - `vehicle_id`
  - `length_of_study_time_step`
  - `in_study_repair` (1 = event, 0 = censored)
- `train_operational_readouts.csv`:
  - Only `vehicle_id` and `time_step` columns are read for labeling.
- Validation labels:
  - `validation_labels.csv` with `vehicle_id`, `class_label` (used for distribution cross-checks, not modified).

### 3. Event Vehicles – Multiple Reference Points Per Class

Event vehicles:

- Determined by `in_study_repair = 1` in `train_tte.csv` (N = 2,272).
- For each event vehicle, we know:
  - `failure_time = length_of_study_time_step`
  - A timeline of `time_step` values in `train_operational_readouts.csv`.

Algorithm (in `build_training_proximity_labels`):

1. Collect time_steps for event vehicles only:
   - Read `train_operational_readouts.csv` in chunks (`chunksize=200_000`) with:
     - `usecols=["vehicle_id", "time_step"]`
   - Filter each chunk to event vehicle IDs (2,272 IDs).
   - Accumulate and sort time_steps per event vehicle:
     - `vehicle_times[vehicle_id] = sorted list of time_step values`
   - This avoids loading full 107-feature rows into memory and only stores sequences for event vehicles.

2. For each event vehicle:
   - Compute time-to-failure per time_step:
     - `TTF = failure_time - time_step`
   - Keep only `TTF > 0` (observations strictly before failure).
   - Map each TTF to a class using `time_to_proximity_class_array`.
   - For each non-zero class `c ∈ {1,2,3,4}`:
     - Identify time_steps mapped to `c`.
     - If any exist, randomly select one time_step (using a fixed random seed for reproducibility).
     - Record:
       - `vehicle_id`
       - `reference_time_step` (selected time_step)
       - `time_to_failure` (TTF at that time_step)
       - `class_label = c`

Outcome:

- Each event vehicle contributes up to 4 labeled examples (one per non-zero class that has at least one valid time_step).
- All references are actual observed time_steps from the operational data, ensuring compatibility with later sequence construction.

### 4. Censored Vehicles – Class 0 Policy

Censored vehicles:

- Determined by `in_study_repair = 0` in `train_tte.csv` (N = 21,278).
- True time-to-failure is unknown.

Policy:

- For each censored vehicle, create one labeled example:
  - `vehicle_id`
  - `reference_time_step = length_of_study_time_step`
  - `time_to_failure = NaN` (unknown)
  - `class_label = 0`

Rationale:

- Class 0 encompasses “far from failure” and censored cases by design.
- This mirrors the dominance of class 0 in validation/test while preserving a simple, interpretable policy.
- The `TrainLabelConfig.include_censored` flag in `src/data/labels.py` allows experiments that exclude censored vehicles if desired.

### 5. Generated Training Labels and Class Distribution

Command used:

- ```bash
  .\.venv\Scripts\python.exe -m src.data.labels \
    --tte data/train_tte.csv \
    --operational data/train_operational_readouts.csv \
    --output data/train_proximity_labels.csv \
    --chunksize 200000
  ```

Output file:

- `data/train_proximity_labels.csv` with columns:
  - `vehicle_id`
  - `reference_time_step`
  - `time_to_failure`
  - `class_label`

Class distribution (initial run):

- Total labels: **29,583**
- Class 0: 21,278 (**71.926%**)
- Class 1: 2,232 (**7.545%**)
- Class 2: 2,150 (**7.268%**)
- Class 3: 1,923 (**6.500%**)
- Class 4: 2,000 (**6.761%**)

Interpretation:

- Class 0 remains the majority class, reflecting the large number of censored vehicles and the “far from failure” regime, but:
  - Non-zero classes (1–4) are now all represented with a few thousand examples each.
  - The training distribution is more balanced than validation/test while retaining realistic skew.
- This should provide enough signal for the LSTM to learn separations among proximity classes, especially once sequence-level sampling (Phase 3) is applied.

### 6. Consistency with Validation Labels

Validation distribution (from `validation_labels.csv`):

- Class 0: 4,910 (**97.305%**)
- Class 1: 16 (**0.317%**)
- Class 2: 14 (**0.277%**)
- Class 3: 30 (**0.595%**)
- Class 4: 76 (**1.506%**)

Consistency notes:

- The class windows and semantics for TTF are identical between training labels and validation/test labels.
- The main difference is density:
  - Validation/test: one randomly chosen last readout per vehicle, resulting in extreme class 0 dominance.
  - Training: multiple reference points per event vehicle to cover all non-zero classes, plus one class 0 example for each censored vehicle.
- A model trained on `train_proximity_labels.csv` is therefore aligned with the official label semantics, but with richer supervision on rare proximity classes.

For additional detail and rationale, see `docs/labeling-decisions.md`.

---

## Phase 3 – Sequence Construction & Sampling Strategy

**Date:** (initial sequence window construction)  
**Scope:** Turn per-vehicle operational readouts + training labels into fixed-length sequences suitable for LSTM input.

### 1. Core Design Decisions

- **Window length (L):** 128 time steps.
  - Past-only: for each labeled example, the sequence covers up to the last 128 time_steps **ending at** `reference_time_step`.
  - Rationale: 128 steps provide a substantial history window for most vehicles (typical horizons ~200–300) without making sequences excessively long for an LSTM.
- **Window orientation:** past-inclusive
  - For each label row `(vehicle_id, reference_time_step, class_label)` we select all readouts with:
    - `time_step <= reference_time_step`
  - From that set, we use the **most recent** 128 steps; if fewer exist, we use all available.
- **Padding:** zero pre-padding
  - If a vehicle has `< 128` valid steps before the reference, we pad at the **start** of the sequence with zeros.
  - After normalization (Phase 4), zero corresponds to “typical” feature values, so padding is neutral.
- **Variable length handling:**
  - For each sequence, we store:
    - `seq_length` = number of real (non-padded) time-steps.
    - The padded sequence itself of shape `(128, num_features)`.
  - LSTMs will use `seq_length` to ignore padded positions via packing or masking.

### 2. Implementation – `src/features/windowing.py`

New module:

- `src/features/windowing.py`

Key configuration:

- `SequenceWindowConfig` dataclass:
  - `operational_path` (default `data/train_operational_readouts.csv`)
  - `labels_path` (default `data/train_proximity_labels.csv`)
  - `output_path` (optional, e.g. `data/train_sequences.npz`)
  - `window_size` (default `128`)
  - `pad_value` (default `0.0`)
  - `max_windows_per_vehicle` (optional cap for sampling)

Main function:

- `build_sequences_for_training(config: SequenceWindowConfig | None) -> (sequences, labels, seq_lengths, vehicle_ids, ref_times)`

Behavior:

1. Load training labels:
   - `labels_df = pd.read_csv(labels_path)`
   - Required columns: `vehicle_id`, `reference_time_step`, `class_label`.
   - Group labels by `vehicle_id` for efficient per-vehicle processing:
     - `labels_by_vehicle = dict(labels_df.groupby("vehicle_id"))`

2. Load operational readouts:
   - `op_df = pd.read_csv(operational_path)`
   - Required columns: `vehicle_id`, `time_step`.
   - Sort:
     - `op_df.sort_values(["vehicle_id", "time_step"])`
   - Feature columns:
     - `feature_cols = [c for c in op_df.columns if c not in ("vehicle_id", "time_step")]`
     - Typical count: 105 feature columns (all histograms + counters).

3. Per-vehicle sequence building:
   - Group operations by vehicle:
     - `op_grouped = op_df.groupby("vehicle_id")`
   - For each `vid` present in `labels_by_vehicle`:
     - If `vid` is absent from `op_grouped`, skip.
     - Extract and sort that vehicle’s operations:
       - `vehicle_ops = op_grouped.get_group(vid).sort_values("time_step")`
       - `times` = array of `time_step` values.
       - `feats` = array of feature values (`feature_cols`) as `float32`.
     - Optionally limit the number of windows per vehicle:
       - `max_windows_per_vehicle` (default: no cap).
     - For each label row for that vehicle:
       - `ref_t = reference_time_step`
       - `class_label` (0–4)
       - Select all rows where `time_step <= ref_t`:
         - `mask = times <= ref_t`
       - If no rows match → skip (no history).
       - Let `hist_feats = feats[mask]`, `seq_len = hist_feats.shape[0]`.
       - If `seq_len >= window_size`:
         - `window = hist_feats[-window_size:]` (last L steps).
         - `effective_len = window_size`
       - Else:
         - `pad_len = window_size - seq_len`
         - `pad_block = zeros((pad_len, num_features))`
         - `window = vstack([pad_block, hist_feats])`
         - `effective_len = seq_len`
       - Append:
         - `window` to `sequences`
         - `class_label` to `labels`
         - `effective_len` to `seq_lengths`
         - `vid` to `vehicle_ids`
         - `ref_t` to `ref_times`

4. Output:
   - Convert lists to arrays:
     - `sequences` → `float32` array of shape `(N, 128, F)`
     - `labels` → `int64` array of shape `(N,)`
     - `seq_lengths` → `int64` array of shape `(N,)`
     - `vehicle_ids` → `int64` array of shape `(N,)`
     - `ref_times` → `float32` array of shape `(N,)`
   - If `config.output_path` is set:
     - Save all arrays to a compressed `.npz`:
       - `np.savez_compressed(output_path, sequences=..., labels=..., seq_lengths=..., vehicle_ids=..., reference_time_step=...)`

CLI:

- Command pattern:
  ```bash
  .\.venv\Scripts\python.exe -m src.features.windowing \
    --operational data/train_operational_readouts.csv \
    --labels data/train_proximity_labels.csv \
    --output data/train_sequences.npz \
    --window-size 128 \
    --max-windows-per-vehicle 1  # optional, for sampling
  ```

The CLI prints:

- `shape=(N, 128, num_features)` for sequences.
- Class distribution of `class_label` in the generated sequences.

### 3. Sanity Check Run (Sample)

For a quick sanity check (not full dataset), a sample run was executed:

- Command:
  ```bash
  .\.venv\Scripts\python.exe -m src.features.windowing \
    --operational data/train_operational_readouts.csv \
    --labels data/train_proximity_labels.csv \
    --output data/train_sequences_sample.npz \
    --window-size 128 \
    --max-windows-per-vehicle 1
  ```

Sample run results:

- Built sequences:
  - Shape: `(23,540, 128, 105)` (105 feature columns).
- Label distribution in this sampled dataset:
  - Class 0: 21,278 (**90.391%**)
  - Class 1: 31 (**0.132%**)
  - Class 2: 70 (**0.297%**)
  - Class 3: 161 (**0.684%**)
  - Class 4: 2,000 (**8.496%**)

Interpretation:

- Even with at most one window per vehicle, class 0 remains dominant, but there are meaningful numbers of high-risk (class 4) sequences.
- Since `max_windows_per_vehicle=1` was applied in the sample, this is a lower bound on the number of available windows; dropping this cap would produce more sequences for event vehicles (especially in non-zero classes) at the cost of a larger dataset.

### 4. Implications for Later Phases

- **Phase 4 (Feature engineering):**
  - Normalization and feature grouping will operate on the feature dimension of these sequences (or on raw operational data before windowing, depending on the final design).
  - Zero-padding plus sequence lengths simplifies normalization and LSTM packing.
- **Phase 5 (Dataset & DataLoader):**
  - PyTorch `Dataset` can directly wrap:
    - `sequences`, `labels`, and `seq_lengths` (plus optional `vehicle_ids`, `ref_times` for analysis).
  - DataLoader collate functions will not need to pad further if sequences are pre-padded to length 128.
- **Imbalance handling:**
  - Class imbalance patterns seen in Phase 2/3 will inform:
    - Class-weighted loss, focal loss, or oversampling in the DataLoader.

---

## Phase 4 – Feature Engineering & Normalization

**Date:** (initial feature stats + transformer)  
**Scope:** Define how to normalize and impute the Component X operational variables using training data only, and encapsulate this logic in a reusable transformer for train/val/test.

### 1. Objectives and key decisions

- **Data used to fit statistics:**
  - Only `data/train_operational_readouts.csv` (training split).
  - Read in chunks to avoid loading the full CSV into memory.
- **Feature grouping:**
  - Counter features: `171_0, 666_0, 427_0, 837_0, 309_0, 835_0, 370_0, 100_0`.
  - Histogram features: all other signal columns (97 bins in total).
- **Normalization strategy:**
  - Counters → `log1p` + z-normalization:
    - Compute mean and standard deviation of `log1p(value)` for each counter.
    - At transform time, apply the same `log1p` and then `(x - mean) / std`.
  - Histograms → direct z-normalization:
    - Mean and standard deviation computed on raw values.
- **Missing value handling:**
  - Simple imputation with the **feature mean** (in the transformed space—log or linear as appropriate).
- **Padding after normalization:**
  - Use `seq_lengths` to set all padded positions at the start of each sequence to 0, so padding is exactly 0 in normalized space.

### 2. Statistics computation – `src/features/transformer.py`

New module:

- `src/features/transformer.py`

Main components:

- `FeatureStatsConfig`:
  - `operational_path` (default `data/train_operational_readouts.csv`).
  - `output_path` (default `artifacts/feature_stats.json`).
  - `chunksize` (default `200_000`).
- `compute_feature_stats(config)`:
  - Infers feature columns as all columns except `vehicle_id` and `time_step` (105 in total).
  - Splits into:
    - `counters`: intersection between columns and `COUNTER_FEATURES`.
    - `histograms`: the remaining features.
  - For each CSV chunk:
    - Casts feature columns to `float`.
    - If the column is a counter → applies `log1p` before accumulating sums.
    - For each column:
      - Accumulates `sum`, `sum_sq`, and `count`, ignoring NaNs.
  - At the end, for each feature computes:
    - `mean = sum / count`
    - `var = max(sum_sq / count - mean^2, eps)` with `eps = 1e-8`
    - `std = sqrt(var)`
    - `transform = "log1p-znorm"` for counters, `"znorm"` for histograms.
  - Writes a JSON with structure:
    - `feature_order`: ordered list of feature names.
    - `counters`, `histograms`.
    - `per_feature[col] = { "transform", "mean", "std" }`.

Command used:

```bash
.\.venv\Scripts\python.exe -m src.features.transformer ^
  --operational data/train_operational_readouts.csv ^
  --output artifacts/feature_stats.json ^
  --chunksize 200000
```

Reported output:

- Number of features: 105.
- Counters: 8.
- Histograms: 97.
- Stats file: `artifacts/feature_stats.json`.

### 3. Reusable transformer – `FeatureTransformer`

Main class:

- `FeatureTransformer` (in `src/features/transformer.py`):
  - `FeatureTransformer.from_json("artifacts/feature_stats.json")` loads:
    - `feature_order` (defining the order of the feature dimension in sequences).
    - `per_feature` with `transform`, `mean`, `std` per column.
  - `transform_sequences(sequences, seq_lengths, copy=True)` applies normalization to sequence tensors:
    - Expected input:
      - `sequences`: tensor `(N, L, F)` with `F == len(feature_order)`.
      - `seq_lengths`: vector `(N,)` with true (unpadded) sequence lengths.
    - For each feature `j`:
      - Reads specification `spec = per_feature[feat]`:
        - If `transform == "log1p-znorm"`:
          - Applies `log1p` to values (assuming non-negative inputs).
        - Imputes NaNs with `mean`.
        - Normalizes: `vals = (vals - mean) / std`.
      - Writes back column `j` in the transformed tensor.
    - After all features are normalized:
      - For each sequence `i`:
        - Computes `pad_len = L - seq_lengths[i]`.
        - If `pad_len > 0`, sets `arr[i, :pad_len, :] = 0`:
          - Ensures padding is exactly 0 across all features, consistent for train/val/test.

Example usage with prebuilt sequences:

```python
import numpy as np
from src.features.transformer import FeatureTransformer

data = np.load("data/train_sequences_sample.npz")
sequences = data["sequences"]
seq_lengths = data["seq_lengths"]

ft = FeatureTransformer.from_json("artifacts/feature_stats.json")
sequences_norm = ft.transform_sequences(sequences, seq_lengths)
```

### 4. Implications for later phases

- **Phase 5 (Dataset & DataLoader):**
  - The PyTorch Dataset will:
    - Load sequences and lengths from `.npz` (`train_sequences*.npz`).
    - Apply `FeatureTransformer` in `__getitem__` or operate on pre-normalized sequences.
  - Padding is already normalized to 0, which simplifies using packing or masks in the LSTM.
- **Phase 6 (LSTM model):**
  - The model receives inputs with mean ≈0 and variance ≈1 per feature, improving training stability.
  - Counter features have compressed scale (`log1p`), reducing the impact of heavy tails.
- **Generalization / reproducibility:**
  - The file `artifacts/feature_stats.json` explicitly documents the transform type and parameters per feature, enabling:
    - Experiment reproducibility.
    - Applying the same preprocessing to new partitions (validation, test, production).

---

## Phase 5 – Dataset & DataLoader Implementation

**Date:** (initial Dataset/DataLoader scaffolding)  
**Scope:** Wrap pre-built sequences and feature transforms in PyTorch Dataset/DataLoader abstractions suitable for training the LSTM.

### 1. Objectives and design choices

- Provide a reusable Dataset that:
  - Loads sequences/labels from `.npz` files produced in Phase 3.
  - Optionally applies the Phase 4 `FeatureTransformer` on-the-fly.
  - Returns tensors `(sequence, label, seq_length)` ready for an LSTM.
- Provide DataLoader helpers that:
  - Handle shuffling/batching.
  - Optionally apply class-weighted sampling to mitigate label imbalance.
- Keep the implementation simple:
  - `.npz` is loaded once; per-sample transform is applied in `__getitem__`.
  - No additional padding inside the DataLoader (sequences are already length 128).

### 2. ComponentXSequenceDataset – `src/data/datasets.py`

New module:

- `src/data/datasets.py`

Main class:

- `ComponentXSequenceDataset(torch.utils.data.Dataset)`:
  - Constructor arguments:
    - `npz_path`: path to `.npz` file with sequences.
    - `feature_stats_path` (optional): path to `feature_stats.json`.
    - `normalize`: whether to apply `FeatureTransformer` (default: True if stats path is given).
    - `device`: optional `torch.device` or string (`"cuda"`, `"cpu"`).
  - Expected `.npz` contents (from Phase 3 windowing):
    - `sequences`: `(N, L, F)` float32.
    - `labels`: `(N,)` int64.
    - `seq_lengths`: `(N,)` int64.
    - `vehicle_ids`: `(N,)` int64 (optional).
    - `reference_time_step`: `(N,)` float32 (optional).
  - Initialization:
    - Loads arrays from `np.load(npz_path)`.
    - Stores `self.sequences`, `self.labels`, `self.seq_lengths`, and optional metadata.
    - Creates a `FeatureTransformer` instance if `normalize=True` and `feature_stats_path` is provided.
  - `__len__`:
    - Returns the number of sequences `N`.
  - `_transform_single(seq_np, length)`:
    - If no transformer:
      - Returns the sequence unchanged.
    - With transformer:
      - Wraps the sequence as batch `(1, L, F)` and calls `transform_sequences`.
      - Unwraps back to `(L, F)`.
  - `__getitem__(idx)`:
    - Extracts:
      - `seq_np = sequences[idx]` (NumPy `(L, F)`).
      - `label = labels[idx]`.
      - `length = seq_lengths[idx]`.
    - Applies `_transform_single` (normalization + padding reset) if enabled.
    - Converts to tensors:
      - `seq`: `torch.float32` tensor `(L, F)`.
      - `y`: `torch.long` scalar label.
      - `seq_len`: `torch.long` scalar sequence length.
    - If `device` is set, moves tensors to that device (with `non_blocking=True`).
    - Returns `(seq, y, seq_len)`.

Notes:

- Imports from `torch` are wrapped in a try/except that raises a clear error if PyTorch is not installed.
- The dataset relies on the consistent feature order tracked in `feature_stats.json` to align with Phase 4 transforms.

### 3. DataLoader helpers – `src/data/dataloaders.py`

New module:

- `src/data/dataloaders.py`

Helper function:

- `create_sequence_dataloader(...) -> DataLoader`:
  - Arguments:
    - `npz_path`: path to `.npz` sequences file.
    - `feature_stats_path`: path to `feature_stats.json` (for normalization).
    - `batch_size`: batch size (default 64).
    - `shuffle`: whether to shuffle (ignored if `class_weighted=True`).
    - `class_weighted`: if True, use `WeightedRandomSampler` for class-imbalanced training.
    - `num_workers`: DataLoader workers (default 0).
    - `device`: optional device passed to the underlying Dataset.
  - Behavior:
    1. Instantiates `ComponentXSequenceDataset` with the given paths and device.
    2. If `class_weighted`:
       - Computes class frequencies from `dataset.labels` using `np.bincount`.
       - Derives per-class weights `1 / count` (0 for unseen classes).
       - Builds per-sample weights via indexing with labels.
       - Uses `WeightedRandomSampler` with replacement to construct batches.
       - Returns `DataLoader(..., sampler=sampler, shuffle=False, ...)`.
    3. If not `class_weighted`:
       - Returns a standard `DataLoader` with `shuffle` as specified.
  - `pin_memory` is set when a CUDA device string is provided.

Example usage:

```python
from src.data.dataloaders import create_sequence_dataloader

train_loader = create_sequence_dataloader(
    npz_path="data/train_sequences.npz",
    feature_stats_path="artifacts/feature_stats.json",
    batch_size=64,
    shuffle=True,
    class_weighted=True,   # or False for simple shuffling
    num_workers=0,
    device="cuda"          # or "cpu"
)
```

### 4. Implications for later phases

- **Phase 6 (LSTM model):**
  - The training loop can directly iterate over `train_loader`:
    - Each batch yields `(batch_seq, batch_labels, batch_seq_lengths)`.
  - It can use `batch_seq_lengths` to pack sequences or apply masks when aggregating over time.
- **Phase 7 (Training loop, loss, and metrics):**
  - Class-weighted sampling is available as a first lever against imbalance.
  - Alternatively, one can use class-weighted loss (via class frequencies from `dataset.labels`).
- **Evaluation:**
  - Equivalent loaders can be created for validation/test `.npz` files (once built), typically without class weighting and with `shuffle=False`.

---

Future phases (6 and beyond) will append their own sections here with:
- New findings (label construction, sequence statistics, feature transformations, model behavior).
- Explicit commands/scripts used.
- Key decisions and rationales.

---

## Phase 6 – LSTM Model Design

**Date:** (baseline LSTM implementation)  
**Scope:** Define and implement a baseline LSTM classifier that consumes the fixed-length, normalized sequences built in Phases 3–5.

### 1. Objectives and design choices

- Use a straightforward but robust architecture suitable as a baseline:
  - 1–2 LSTM layers with optional bidirectionality and dropout.
  - Simple pooling strategy over time (`"last"` by default, `"mean"` as an option).
  - MLP head mapping sequence representation → 5-class logits.
- Keep the interface compatible with existing Dataset/DataLoader:
  - Input shape: `(batch_size, seq_len=128, input_size=105)`.
  - Optional `seq_lengths` tensor to support mean pooling and future extensions.
- Rely on left-padding (zeros at the start) to simplify pooling:
  - For `"last"` pooling, the last time step is always a real observation, regardless of sequence length.

### 2. Model implementation – `src/models/lstm_classifier.py`

New module:

- `src/models/lstm_classifier.py`

Configuration:

- `LSTMClassifierConfig` dataclass:
  - `input_size`: number of features per time-step (e.g., 105).
  - `hidden_size`: LSTM hidden size (default 128).
  - `num_layers`: number of stacked LSTM layers (default 2).
  - `bidirectional`: whether the LSTM is bidirectional (default True).
  - `dropout`: dropout rate applied between LSTM layers and in the classifier head (default 0.1).
  - `num_classes`: number of output classes (default 5).
  - `pooling`: `"last"` or `"mean"` (default `"last"`).

Model class:

- `LSTMClassifier(nn.Module)`:
  - Constructor:
    - Builds an `nn.LSTM` with:
      - `input_size = config.input_size`
      - `hidden_size = config.hidden_size`
      - `num_layers = config.num_layers`
      - `batch_first = True`
      - `bidirectional = config.bidirectional`
      - `dropout = config.dropout` if `num_layers > 1` else `0.0`
    - Computes `lstm_output_size = hidden_size * (2 if bidirectional else 1)`.
    - Builds a classifier head:
      - `Linear(lstm_output_size, 128) → ReLU → Dropout`
      - `Linear(128, 64) → ReLU → Dropout`
      - `Linear(64, num_classes)`
  - `_pool_last(lstm_out)`:
    - Input: `lstm_out` of shape `(batch, seq_len, hidden*dir)`.
    - Returns: last time-step vector `lstm_out[:, -1, :]`.
    - Justification: sequences are left-padded, so the last index always corresponds to a real observation.
  - `_pool_mean(lstm_out, seq_lengths)`:
    - Inputs:
      - `lstm_out`: `(batch, seq_len, hidden*dir)`.
      - `seq_lengths`: `(batch,)` with true lengths.
    - For each sequence:
      - Computes start index of real data: `start = seq_len - length`.
      - Builds a mask where positions ≥ `start` are 1 (real), others 0 (padding).
      - Applies the mask to `lstm_out`, sums over time, and divides by `length`.
    - Returns: mean-pooled representation `(batch, hidden*dir)`.
  - `forward(x, seq_lengths=None)`:
    - Inputs:
      - `x`: `(batch, seq_len, input_size)` tensor.
      - `seq_lengths`: `(batch,)` tensor (required if `pooling="mean"`).
    - Operations:
      - Passes `x` through `self.lstm`:
        - `lstm_out, _ = self.lstm(x)` → `(batch, seq_len, hidden*dir)`.
      - Applies pooling:
        - `"last"`: uses `_pool_last`.
        - `"mean"`: uses `_pool_mean` and requires `seq_lengths`.
      - Feeds pooled representation into `self.classifier` to produce logits.
    - Output:
      - `logits` of shape `(batch, num_classes)`.

Notes:

- Imports from `torch` are wrapped so that importing this module without PyTorch installed raises a clear error.
- The model is agnostic to the exact feature semantics; it only assumes:
  - Input is already normalized per feature (via `FeatureTransformer`).
  - Sequences are left-padded with zeros and share a common `seq_len`.

### 3. Integration expectations

- **With Dataset/DataLoader (Phase 5):**
  - Training loop will typically receive batches:
    - `seq` `(batch, 128, 105)`, `labels` `(batch,)`, `seq_lengths` `(batch,)`.
  - Forward pass:
    - `logits = model(seq, seq_lengths)` (for `"mean"` pooling) or `model(seq)` for `"last"`.
  - Loss:
    - Cross-entropy with optional class weights derived from label frequencies.
- **With future training script (Phase 7):**
  - The model can be configured via `LSTMClassifierConfig`, making hyperparameters explicit and easy to log.
  - Pooling type (`"last"` vs `"mean"`) can be treated as a tunable hyperparameter.

The actual training/evaluation loop and metric logging are implemented in Phase 7.

---

## Phase 7 – Training Loop, Loss, and Metrics

**Date:** (baseline training script)  
**Scope:** Implement a reproducible training loop that wires together the LSTM model, sequence DataLoaders, loss, metrics, and checkpointing.

### 1. Objectives and design choices

- Provide a single entry point to train the LSTM classifier:
  - Takes `.npz` sequence files + `feature_stats.json` as inputs.
  - Configures the model, optimizer, scheduler, and loss from CLI arguments.
  - Supports resuming from a previous checkpoint.
- Implement baseline metrics:
  - Training/validation loss.
  - Accuracy and macro F1 on validation.
- Implement robust checkpointing:
  - `last.pt` – last epoch state (for resume).
  - `best.pt` – best validation macro F1.
  - Checkpoints include model, optimizer, scheduler states, and RNG states.

### 2. Training script – `src/training/train_lstm.py`

New module:

- `src/training/train_lstm.py`

Main components:

- **Argument parsing (`parse_args`)**:
  - Data:
    - `--train-npz`: path to training sequences (default `data/train_sequences.npz`).
    - `--val-npz`: path to validation sequences (optional).
    - `--feature-stats`: path to `feature_stats.json` (default `artifacts/feature_stats.json`).
  - Model:
    - `--hidden-size` (default 128), `--num-layers` (default 2).
    - `--bidirectional` (flag).
    - `--dropout` (default 0.1).
    - `--pooling` (`"last"` or `"mean"`, default `"last"`).
  - Optimization:
    - `--batch-size` (default 64).
    - `--epochs` (default 20).
    - `--lr` (default 1e-3).
    - `--weight-decay` (default 1e-4).
    - `--step-lr-gamma` (default 0.5).
    - `--step-lr-step-size` (default 10 epochs).
    - `--class-weighted` (flag) to enable class weighting.
  - Misc:
    - `--device` (e.g., `"cuda"`, `"cpu"`, or None for auto-select).
    - `--output-dir` (default `artifacts`).
    - `--resume-from` (path to checkpoint `.pt`).
    - `--log-interval` (batches between train logs).

- **Device selection**:
  - `select_device` chooses:
    - Preferred device if provided, otherwise `"cuda"` if available, else `"cpu"`.

- **Data loading (`build_datasets_and_loaders`)**:
  - Instantiates a `ComponentXSequenceDataset` for training:
    - Uses `args.train_npz` and `args.feature_stats` for normalization.
  - Builds:
    - `train_loader` via `create_sequence_dataloader`:
      - `class_weighted=True` uses `WeightedRandomSampler`.
    - `val_loader` if `--val-npz` is provided (no class weighting, `shuffle=False`).

- **Model and optimizer (`build_model_and_optim`)**:
  - Builds `LSTMClassifierConfig` with:
    - `input_size` from `train_dataset.num_features`.
    - Hyperparameters from CLI.
  - Instantiates `LSTMClassifier(config)` and moves it to the selected device.
  - Optimizer: `AdamW`.
  - Scheduler: `StepLR(optimizer, step_size, gamma)`.

- **Class weights (`compute_class_weights`)**:
  - Uses `np.bincount` over `train_dataset.labels`.
  - Sets weight for class `c` as `1 / count[c]` where count>0.
  - Falls back to uniform weights if something goes wrong.
  - If `--class-weighted` is set:
    - Uses these weights in `nn.CrossEntropyLoss(weight=class_weights)` and may also use class-weighted sampling (depending on loader settings).

- **Checkpointing (`save_checkpoint`, `load_checkpoint`)**:
  - `save_checkpoint(path, epoch, model, optimizer, scheduler, best_val_f1, args)`:
    - Saves:
      - `epoch`
      - `model_state`, `optimizer_state`, `scheduler_state`
      - `best_val_f1`
      - CLI `args` (as a dict)
      - `torch_rng_state`
      - `cuda_rng_state` if available.
  - `load_checkpoint(path, model, optimizer, scheduler)`:
    - Restores model/optimizer/scheduler states.
    - Restores RNG states (CPU and CUDA).
    - Returns `start_epoch = last_epoch + 1`, `best_val_f1`, and stored `args`.

### 3. Training and evaluation loops

- **Training (`train_one_epoch`)**:
  - Sets `model.train()`.
  - For each batch `(seq, labels, seq_lengths)`:
    - Moves tensors to device.
    - Forward pass: `logits = model(seq, seq_lengths)`.
    - Loss: `criterion(logits, labels)`.
    - Backprop: `loss.backward()`, `optimizer.step()`.
  - Tracks:
    - Running loss (sum over samples).
    - Predictions and targets (for accuracy).
  - Logs intermediate average loss every `log_interval` batches.
  - Returns:
    - `avg_loss` and training accuracy.

- **Evaluation (`evaluate`)**:
  - Sets `model.eval()`, wraps loop in `torch.no_grad()`.
  - For each batch:
    - Forward pass and loss computation.
    - Collects predictions and targets.
  - Computes:
    - Average loss.
    - Accuracy (`accuracy_score` from scikit-learn).
    - Macro F1 (`f1_score(..., average="macro")`).
  - Returns `(val_loss, val_acc, val_macro_f1)`.

- **Main training loop (`main`)**:
  - Parses args, selects device, creates output dir.
  - Builds datasets/loaders, model/optimizer/scheduler, and loss function.
  - Handles optional resume:
    - If `--resume-from` exists, loads checkpoint and continues from `epoch+1`.
  - For each epoch:
    - Runs `train_one_epoch`.
    - Runs `evaluate` if `val_loader` exists.
    - Steps LR scheduler.
    - Saves `last.pt` every epoch.
    - If validation macro F1 improves, updates `best_val_f1` and saves `best.pt`.

### 4. Usage example

Once PyTorch is installed and full train/validation sequences are available:

```bash
python -m src.training.train_lstm \
  --train-npz data/train_sequences.npz \
  --val-npz data/val_sequences.npz \
  --feature-stats artifacts/feature_stats.json \
  --batch-size 64 \
  --epochs 20 \
  --hidden-size 128 \
  --num-layers 2 \
  --bidirectional \
  --dropout 0.1 \
  --pooling last \
  --class-weighted \
  --device cuda \
  --output-dir artifacts \
  --log-interval 100
```

This will produce:

- `artifacts/last.pt` - latest checkpoint (for resume).
- `artifacts/best.pt` - checkpoint with best validation macro F1.

---

## Phase 8 – Evaluation, Error Analysis, and Calibration

**Date:** 2025-11-24  
**Scope:** Automate the Phase 8 diagnostics (confusion matrix, per-class PR/ROC curves, and calibration) so every run can be analyzed without ad-hoc notebooks.

### 1. Evaluation tooling

- Implemented `src/training/evaluate.py`, a CLI wrapper that loads any checkpoint + `.npz` split and generates the complete evaluation bundle.
- Typical invocation (used for Run4):
  ```bash
  python -m src.training.evaluate \
    --npz data/val_sequences.npz \
    --checkpoint artifacts_colab_run4/best.pt \
    --feature-stats artifacts/feature_stats.json \
    --output-dir artifacts/evaluations/run4_val
  ```
- Internals:
  - Restores the LSTM configuration from checkpoint metadata.
  - Streams batches through the model on CPU or GPU (auto device selection).
  - Logs headline metrics plus JSON/PNG artifacts for plots.
  - Falls back gracefully if plotting deps (matplotlib) are missing, making it safe for headless runs.

### 2. Outputs captured for Run4

Artifacts are stored in `artifacts/evaluations/run4_val/`:

| Artifact | Notes |
| --- | --- |
| `summary_metrics.json` | Accuracy 0.548, macro F1 0.155, weighted F1 0.692. |
| `classification_report.json` | Per-class precision/recall/F1/support using the descriptive class text. |
| `confusion_matrix.csv/.png` | Matrix re-labelled with numeric class IDs 0–4 to keep figures concise. |
| `roc_curves.json/.png` | ROC curve coordinates + AUC per class (keyed by class ID). |
| `pr_curves.json/.png` | Precision/Recall curve coordinates + area per class (class ID keys). |
| `reliability.json/.png` | Calibration histogram for top-1 probabilities. |

Regenerating the folder with different `--npz` / `--checkpoint` pairs gives comparable Phase‑8 packages for future experiments.

### 3. Findings

- Confusion matrix: while 2,734 class‑0 samples are correct, >2,100 majority-class sequences still leak into classes 2 and 4, illustrating how balanced loss keeps minority recall alive at the expense of false alarms on class 0.
- Minority support is tiny (e.g., only ~40 validation samples for classes 1–4 combined), so ROC/PR curves remain noisy but still show classes 2 and 4 benefitting the most from the weighting tweaks.
- Calibration: the reliability diagram shows that high-confidence predictions (0.8–0.9) achieve only ~0.6 empirical accuracy, so temperature scaling or conservative alert thresholds will be needed before deployment.
- With this tooling in place the Phase‑8 plan items are closed, and we can advance to Phase 9 (experiment tracking) knowing every run can emit a consistent diagnostics bundle.

## Phase 9 – Experiment Tracking & Reproducibility

**Date:** 2025-11-24  
**Scope:** Wire MLflow into the baseline training pipeline so every experiment records params/metrics/artifacts automatically, and document how to run/promote experiments.

### 1. MLflow wiring

- Added optional flags to `src/training/train_lstm.py`:
  - `--mlflow` toggles logging.
  - `--mlflow-tracking-uri`, `--mlflow-experiment`, `--mlflow-run-name` control the destination.
- When enabled:
  - The script starts an MLflow run before training and logs all CLI arguments as parameters.
  - After each epoch it logs metrics (`train_loss`, `train_acc`, `val_loss`, `val_acc`, `val_macro_f1`, and learning rate).
  - On completion it uploads `metrics.csv`, `last.pt`, `best.pt`, and the referenced `feature_stats.json`.
- Fails fast if MLflow isn’t installed but the flag is set, which keeps local CPU dev loops unaffected unless needed.

### 2. Experiment playbook (`docs/experiments.md`)

- Captures:
  - Prereqs (install MLflow, start tracking URI).
  - Naming convention for runs (`{data_version}-{window_size}-{model_tag}-{run_id}`).
  - Exact CLI invocation template for GPU training.
  - Promotion criteria (macro-F1 improvements + uploaded evaluation bundle).
- Serves as the canonical reference for future teammates.

### 3. Next actions

- For any heavy run, start MLflow on the GPU/Colab environment and launch training with `--mlflow`.
- After training finishes, generate the Phase 8 evaluation bundle and attach it to the MLflow run (either through `mlflow.log_artifact` or a manual upload) before promoting the checkpoint.

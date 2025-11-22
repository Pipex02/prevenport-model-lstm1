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

Future phases (2 and beyond) will append their own sections here with:
- New findings (label construction, sequence statistics, feature transformations, model behavior).
- Explicit commands/scripts used.
- Key decisions and rationales.

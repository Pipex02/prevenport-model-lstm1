# Component X – Labeling Decisions for 5-Class Proximity Problem

This document captures the design choices made when constructing 5-class proximity-to-failure labels for the training split, and how they relate to the official validation/test labels.

---

## 1. Class Definition (Time-to-Failure Windows)

The 5 classes are defined in terms of **time-to-failure** (TTF), measured in `time_step` units:

- **Class 4:** 0–6 steps before failure  
- **Class 3:** 6–12 steps before failure  
- **Class 2:** 12–24 steps before failure  
- **Class 1:** 24–48 steps before failure  
- **Class 0:** >48 steps before failure **or** no observed failure (censored)

Implementation details:

- We map a scalar `time_to_failure >= 0` to a class as:
  - `0 <= TTF <= 6`   → class 4  
  - `6 < TTF <= 12`   → class 3  
  - `12 < TTF <= 24`  → class 2  
  - `24 < TTF <= 48`  → class 1  
  - `TTF > 48` or `TTF` unknown (censored) → class 0
- Negative TTF (reference after failure) is treated as invalid and is excluded.
- The implementation lives in `src/data/labels.py` as:
  - `time_to_proximity_class(time_to_failure: float) -> int`
  - `time_to_proximity_class_array(time_to_failure: np.ndarray) -> np.ndarray`

This matches the description in `data/dataset-overview.md` and the Scientific Data article.

---

## 2. Inputs Used for Training Labels

Training labels are derived from:

- `train_tte.csv`
  - `vehicle_id`
  - `length_of_study_time_step` (final observation time)
  - `in_study_repair` (1 = failure/repair observed at that horizon, 0 = censored)
- `train_operational_readouts.csv`
  - `vehicle_id`
  - `time_step`
  - 107 columns in total, but for labeling we only use `vehicle_id` and `time_step` to avoid loading full feature matrices into memory.

We **do not** alter official validation/test labels:

- `validation_labels.csv` already contains:
  - `vehicle_id`
  - `class_label` ∈ {0,1,2,3,4}
- Training labels are constructed so their semantics (TTF windows) match the validation/test labels.

---

## 3. Reference Time Selection for Event Vehicles

From `train_tte.csv`:

- Event vehicles: `in_study_repair = 1` (N = 2,272)
- For each such vehicle, we know:
  - `failure_time = length_of_study_time_step`
  - A sequence of `time_step` values from `train_operational_readouts.csv`

To create **multiple labeled examples per event vehicle** and cover all non-zero classes:

1. We first collect time_steps for event vehicles only:
   - File: `train_operational_readouts.csv`
   - Columns read: `vehicle_id`, `time_step`
   - Reading strategy:
     - Chunked via `pandas.read_csv(..., usecols=["vehicle_id", "time_step"], chunksize=200_000)`
     - Filter to event vehicles only (2,272 IDs).
     - Accumulate `time_step` values per event vehicle, then sort them.
   - Outcome:
     - `vehicle_times[vehicle_id] = sorted list of time_step values`
     - Memory usage is limited to event vehicles (roughly a few hundred thousand time_steps).

2. For each event vehicle:
   - Compute `time_to_failure` for each time_step:
     - `TTF = failure_time - time_step`
   - Filter to `TTF > 0` (time-steps strictly before failure).
   - Map each `TTF` to a class using `time_to_proximity_class_array`.
   - For each non-zero class `c ∈ {1,2,3,4}`:
     - Find all time_steps whose TTF maps to `c`.
     - If there are any:
       - Randomly select one time_step for that class using a fixed random seed (for reproducibility).
       - Record a labeled example:
         - `(vehicle_id, reference_time_step, time_to_failure, class_label=c)`

This yields up to 4 labeled examples per event vehicle (one per non-zero class where observations exist).

Rationale:

- The official validation/test splits have only one labeled example per vehicle, but we want the training set to **cover all proximity classes** and support sequence-based learning.
- Using actual observed time_steps ensures that every `reference_time_step` corresponds to a real readout, which is critical for Phase 3 (sequence construction).
- Random selection with a fixed seed provides variation while remaining reproducible.

---

## 4. Policy for Censored Vehicles

From `train_tte.csv`:

- Censored vehicles: `in_study_repair = 0` (N = 21,278)
  - We do **not** know their true time-to-failure.

Policy choice:

- For each censored vehicle, create **one** training label:
  - `vehicle_id`
  - `reference_time_step = length_of_study_time_step` (end of observation)
  - `time_to_failure = NaN` (unknown)
  - `class_label = 0`

Rationale:

- The problem definition and article treat class 0 as “far from failure” or “no failure in the foreseeable future”.
- Censored vehicles are consistent with that semantics: by the end of the study window, no failure has been observed.
- Including censored vehicles as class 0 provides a large, realistic background class and matches the heavy class-0 dominance in validation/test.
- If needed, future experiments can:
  - Exclude censored vehicles via the `--exclude-censored` flag in `src/data/labels.py`.
  - Re-weight classes during training to mitigate imbalance.

---

## 5. Implementation and Generated Training Labels

Implementation file:

- `src/data/labels.py`

Key entry points:

- `time_to_proximity_class(time_to_failure: float) -> int`
- `time_to_proximity_class_array(time_to_failure: np.ndarray) -> np.ndarray`
- `build_training_proximity_labels(config: TrainLabelConfig | None = None) -> pd.DataFrame`
- CLI usage:
  ```bash
  .\.venv\Scripts\python.exe -m src.data.labels \
    --tte data/train_tte.csv \
    --operational data/train_operational_readouts.csv \
    --output data/train_proximity_labels.csv \
    --chunksize 200000
  ```

By default:

- `TrainLabelConfig.include_censored = True` → censored vehicles are labeled as class 0 at `length_of_study_time_step`.

Generated file:

- `data/train_proximity_labels.csv`
  - Columns:
    - `vehicle_id`
    - `reference_time_step`
    - `time_to_failure`
    - `class_label`

Class distribution (from the initial run):

- Total training labels: **29,583**
- Per-class counts:
  - Class 0: 21,278 (**71.926%**)
  - Class 1: 2,232 (**7.545%**)
  - Class 2: 2,150 (**7.268%**)
  - Class 3: 1,923 (**6.500%**)
  - Class 4: 2,000 (**6.761%**)

Notes:

- Each censored vehicle contributes exactly one labeled example in class 0.
- Each event vehicle contributes up to four labeled examples (one per non-zero class where observations exist). Some vehicles may lack observations in certain windows and thus contribute fewer.
- The resulting class distribution is much more balanced than validation/test, but still dominated by class 0, which is realistic for operational data.

---

## 6. Consistency with Validation/Test Labels

Validation labels:

- `data/validation_labels.csv` contains `vehicle_id` and `class_label`, with the following distribution:
  - Class 0: 4,910 (**97.305%**)
  - Class 1: 16 (**0.317%**)
  - Class 2: 14 (**0.277%**)
  - Class 3: 30 (**0.595%**)
  - Class 4: 76 (**1.506%**)

Consistency considerations:

- The class definitions in this document mirror the official TTF windows described in `data/dataset-overview.md` and the Scientific Data article.
- Our training labels adopt the same TTF-based semantics; the main difference is **how many reference points per vehicle** are generated:
  - Official validation/test: one random last readout per vehicle.
  - Training labels here: up to one reference point per non-zero class per event vehicle, plus one class-0 label per censored vehicle.
- This design is intentionally richer for training while preserving the same underlying class meaning, so a model trained on `train_proximity_labels.csv` should be well-aligned with the official validation/test label semantics.


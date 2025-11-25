"""
Label engineering utilities for the Component X 5-class proximity-to-failure problem.

This module:
- Defines the mapping from time-to-failure (TTF, in time_step units) to classes {0,1,2,3,4}.
- Provides a chunked pipeline to derive training labels from:
  - `train_tte.csv` (per-vehicle failure/censoring info)
  - `train_operational_readouts.csv` (per-vehicle time_step series)

Resulting training labels can be written to a CSV such as:
    data/processed/train_proximity_labels.csv
with columns:
    vehicle_id, reference_time_step, time_to_failure, class_label
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


PROXIMITY_CLASS_NAMES = {
    0: ">48 steps before failure or censored",
    1: "48–24 steps before failure",
    2: "24–12 steps before failure",
    3: "12–6 steps before failure",
    4: "6–0 steps before failure",
}


def time_to_proximity_class(time_to_failure: float) -> int:
    """Map a scalar time-to-failure (TTF) to a proximity class in {0,1,2,3,4}.

    The mapping follows the Component X description:
        - Class 4:  0–6   steps before failure (inclusive of 0, exclusive of 6+)
        - Class 3:  6–12  steps before failure
        - Class 2: 12–24  steps before failure
        - Class 1: 24–48  steps before failure
        - Class 0: >48    steps before failure (or no event / censored)

    Negative TTF values (reference after failure) are invalid and will raise.
    """
    if np.isnan(time_to_failure):
        # Treat unknown / censored TTF as class 0.
        return 0

    if time_to_failure < 0:
        raise ValueError(f"time_to_failure must be non-negative, got {time_to_failure}")

    if time_to_failure <= 6.0:
        return 4
    if time_to_failure <= 12.0:
        return 3
    if time_to_failure <= 24.0:
        return 2
    if time_to_failure <= 48.0:
        return 1
    return 0


def time_to_proximity_class_array(time_to_failure: np.ndarray) -> np.ndarray:
    """Vectorized version of time_to_proximity_class for NumPy arrays."""
    # Start with default class 0
    ttf = np.asarray(time_to_failure, dtype=float)
    labels = np.zeros_like(ttf, dtype=np.int64)

    # NaNs and negatives remain class 0 (we assume filtering for negatives done upstream)
    mask = ~np.isnan(ttf)
    t = ttf[mask]

    labels_local = np.zeros_like(t, dtype=np.int64)
    labels_local[(t >= 0) & (t <= 6.0)] = 4
    labels_local[(t > 6.0) & (t <= 12.0)] = 3
    labels_local[(t > 12.0) & (t <= 24.0)] = 2
    labels_local[(t > 24.0) & (t <= 48.0)] = 1
    # t > 48 stays 0

    labels[mask] = labels_local
    return labels


@dataclass
class TrainLabelConfig:
    """Configuration for building training proximity labels."""

    train_tte_path: Path = Path("data/raw/train_tte.csv")
    train_operational_path: Path = Path("data/raw/train_operational_readouts.csv")
    output_path: Path = Path("data/processed/train_proximity_labels.csv")
    chunksize: int = 200_000
    random_seed: int = 42
    include_censored: bool = True


def _collect_event_vehicle_times(
    operational_path: Path,
    event_vehicle_ids: Iterable[int],
    chunksize: int,
) -> Dict[int, List[float]]:
    """Collect time_step sequences for vehicles with observed events.

    Reads only `vehicle_id` and `time_step` columns in chunks to avoid loading
    the full dataset (all features) into memory.
    """
    event_ids = set(event_vehicle_ids)
    vehicle_times: Dict[int, List[float]] = {}

    for chunk in pd.read_csv(
        operational_path,
        usecols=["vehicle_id", "time_step"],
        chunksize=chunksize,
    ):
        chunk = chunk[chunk["vehicle_id"].isin(event_ids)]
        if chunk.empty:
            continue
        for vid, group in chunk.groupby("vehicle_id"):
            times = group["time_step"].astype(float).tolist()
            if vid in vehicle_times:
                vehicle_times[vid].extend(times)
            else:
                vehicle_times[vid] = times

    # Sort time_steps per vehicle
    for vid in vehicle_times:
        vehicle_times[vid].sort()

    return vehicle_times


def _select_reference_points_for_event_vehicle(
    vehicle_id: int,
    failure_time: float,
    time_steps: Sequence[float],
    rng: np.random.RandomState,
) -> List[Tuple[int, float, float, int]]:
    """Select up to one reference time_step per proximity class for an event vehicle.

    Returns a list of tuples:
        (vehicle_id, reference_time_step, time_to_failure, class_label)
    """
    if not time_steps:
        return []

    ts = np.asarray(time_steps, dtype=float)
    ttf = failure_time - ts

    # Keep only times strictly before failure (ttf > 0)
    mask = ttf > 0
    ts = ts[mask]
    ttf = ttf[mask]
    if ts.size == 0:
        return []

    labels = time_to_proximity_class_array(ttf)

    results: List[Tuple[int, float, float, int]] = []
    for class_label in (4, 3, 2, 1):  # focus on non-zero classes
        class_mask = labels == class_label
        if not np.any(class_mask):
            continue
        candidate_steps = ts[class_mask]
        candidate_ttf = ttf[class_mask]
        # Choose one candidate at random for diversity but deterministic per run.
        idx = rng.randint(0, candidate_steps.size)
        results.append(
            (
                vehicle_id,
                float(candidate_steps[idx]),
                float(candidate_ttf[idx]),
                int(class_label),
            )
        )

    return results


def build_training_proximity_labels(config: Optional[TrainLabelConfig] = None) -> pd.DataFrame:
    """Build training proximity labels from TTE and operational readouts.

    For each vehicle with an observed event (in_study_repair=1):
      - Collect all its time_steps from `train_operational_readouts.csv`.
      - Compute time_to_failure = failure_time - time_step.
      - For each non-zero proximity class (1–4), randomly select one time_step
        whose time_to_failure falls into that class.

    For censored vehicles (in_study_repair=0), this implementation:
      - Creates a single label with:
          reference_time_step = length_of_study_time_step
          class_label = 0
        (interpreting them as "far from failure" at the end of observation).

    Returns the resulting label DataFrame and also writes it to `config.output_path`.
    """
    if config is None:
        config = TrainLabelConfig()

    train_tte = pd.read_csv(config.train_tte_path)

    # Separate event and censored vehicles
    events = train_tte[train_tte["in_study_repair"] == 1].copy()
    censored = train_tte[train_tte["in_study_repair"] == 0].copy()

    event_ids = events["vehicle_id"].astype(int).tolist()
    vehicle_times = _collect_event_vehicle_times(
        operational_path=config.train_operational_path,
        event_vehicle_ids=event_ids,
        chunksize=config.chunksize,
    )

    rng = np.random.RandomState(config.random_seed)

    rows: List[Tuple[int, float, float, int]] = []

    # Event vehicles: collect up to one reference per non-zero class
    for _, row in events.iterrows():
        vid = int(row["vehicle_id"])
        failure_time = float(row["length_of_study_time_step"])
        times = vehicle_times.get(vid)
        if not times:
            continue
        rows.extend(
            _select_reference_points_for_event_vehicle(
                vehicle_id=vid,
                failure_time=failure_time,
                time_steps=times,
                rng=rng,
            )
        )

    # Censored vehicles: label as class 0 at end of observation (policy choice)
    if config.include_censored:
        for _, row in censored.iterrows():
            vid = int(row["vehicle_id"])
            ref_time = float(row["length_of_study_time_step"])
            # We do not know true TTF; mark as NaN to avoid misinterpretation,
            # but use class 0 to match "far from failure or censored".
            rows.append((vid, ref_time, np.nan, 0))

    labels_df = pd.DataFrame(
        rows,
        columns=["vehicle_id", "reference_time_step", "time_to_failure", "class_label"],
    )

    # Persist to disk
    labels_df.to_csv(config.output_path, index=False)
    return labels_df


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build 5-class training proximity labels for Component X.",
    )
    parser.add_argument(
        "--tte",
        type=str,
        default="data/raw/train_tte.csv",
        help="Path to train_tte.csv.",
    )
    parser.add_argument(
        "--operational",
        type=str,
        default="data/raw/train_operational_readouts.csv",
        help="Path to train_operational_readouts.csv.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/train_proximity_labels.csv",
        help="Output path for generated training labels CSV.",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=200_000,
        help="Chunk size for reading operational readouts.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when selecting reference time_steps for events.",
    )
    parser.add_argument(
        "--exclude-censored",
        action="store_true",
        help="If set, censored vehicles will be excluded (no class 0 rows).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = TrainLabelConfig(
        train_tte_path=Path(args.tte),
        train_operational_path=Path(args.operational),
        output_path=Path(args.output),
        chunksize=args.chunksize,
        random_seed=args.seed,
        include_censored=not args.exclude_censored,
    )
    labels_df = build_training_proximity_labels(config)
    class_counts = labels_df["class_label"].value_counts().sort_index()
    total = len(labels_df)
    print(f"Generated {total} training labels and wrote to {config.output_path}")
    print("Class distribution:")
    for cls, cnt in class_counts.items():
        pct = 100.0 * cnt / max(total, 1)
        print(f"  class={cls}: {cnt} ({pct:.3f}%)")


if __name__ == "__main__":
    main()


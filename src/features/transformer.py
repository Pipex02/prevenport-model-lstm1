"""
Feature engineering and normalization utilities for Component X.

Phase 4 goals:
- Compute per-feature statistics on train operational readouts only.
- Define a reusable transformer that:
  - Imputes missing values.
  - Applies log1p+z-score to counter features.
  - Applies z-score to histogram features.
- Provide a way to apply the same transforms to any sequence tensor
  (train/val/test) without data leakage.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


# Known counter feature names from dataset-overview
COUNTER_FEATURES = {
    "171_0",
    "666_0",
    "427_0",
    "837_0",
    "309_0",
    "835_0",
    "370_0",
    "100_0",
}


@dataclass
class FeatureStatsConfig:
    """Configuration for computing feature statistics on operational data."""

    operational_path: Path = Path("data/train_operational_readouts.csv")
    output_path: Path = Path("artifacts/feature_stats.json")
    chunksize: int = 200_000


def _infer_feature_groups(operational_path: Path) -> Dict[str, List[str]]:
    """Infer feature columns and group them into counters and histograms.

    We assume:
    - Operational file contains columns: vehicle_id, time_step, plus feature columns.
    - Counter features are among COUNTER_FEATURES.
    - All other feature columns (excluding vehicle_id/time_step) are histograms.
    """
    head = pd.read_csv(operational_path, nrows=1)
    all_cols = list(head.columns)
    feature_cols = [c for c in all_cols if c not in ("vehicle_id", "time_step")]

    counters = [c for c in feature_cols if c in COUNTER_FEATURES]
    histograms = [c for c in feature_cols if c not in counters]

    return {
        "feature_order": feature_cols,
        "counters": counters,
        "histograms": histograms,
    }


def compute_feature_stats(config: Optional[FeatureStatsConfig] = None) -> Dict[str, object]:
    """Compute per-feature statistics from train operational readouts.

    For each feature column:
    - If it's a counter: we compute stats on log1p(values).
    - If it's a histogram bin: we compute stats on raw values.
    - We track:
      - mean
      - std (with a minimum epsilon to avoid division by zero)

    Returns the stats dict and writes it to config.output_path as JSON.
    """
    if config is None:
        config = FeatureStatsConfig()

    groups = _infer_feature_groups(config.operational_path)
    feature_order: List[str] = groups["feature_order"]
    counters: List[str] = groups["counters"]
    histograms: List[str] = groups["histograms"]

    # Initialize running aggregates
    per_feature_sum: Dict[str, float] = {c: 0.0 for c in feature_order}
    per_feature_sum_sq: Dict[str, float] = {c: 0.0 for c in feature_order}
    per_feature_count: Dict[str, int] = {c: 0 for c in feature_order}

    for chunk in pd.read_csv(config.operational_path, chunksize=config.chunksize):
        # Ensure we're only dealing with feature columns (skip id/time)
        chunk_features = chunk[feature_order].astype(float)
        for col in feature_order:
            vals = chunk_features[col].to_numpy(dtype=float)

            if col in counters:
                # Counters are non-negative; apply log1p to stabilize scale.
                vals = np.log1p(vals)

            mask = ~np.isnan(vals)
            if not np.any(mask):
                continue

            v = vals[mask]
            per_feature_sum[col] += float(v.sum())
            per_feature_sum_sq[col] += float((v * v).sum())
            per_feature_count[col] += int(mask.sum())

    per_feature_stats: Dict[str, Dict[str, float]] = {}
    eps = 1e-8

    for col in feature_order:
        count = per_feature_count[col]
        if count == 0:
            # Fallback to mean=0, std=1 if we somehow never saw this feature.
            mean = 0.0
            std = 1.0
        else:
            s = per_feature_sum[col]
            s2 = per_feature_sum_sq[col]
            mean = s / count
            var = max(s2 / count - mean * mean, eps)
            std = float(np.sqrt(var))

        transform_type = "log1p-znorm" if col in counters else "znorm"
        per_feature_stats[col] = {
            "transform": transform_type,
            "mean": float(mean),
            "std": float(std),
        }

    stats = {
        "feature_order": feature_order,
        "counters": counters,
        "histograms": histograms,
        "per_feature": per_feature_stats,
    }

    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    with config.output_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    return stats


@dataclass
class FeatureTransformer:
    """Apply precomputed feature stats to sequences.

    Typical usage:
        ft = FeatureTransformer.from_json("artifacts/feature_stats.json")
        sequences_norm = ft.transform_sequences(sequences, seq_lengths)

    - `sequences` is expected to be an array of shape (N, L, F)
      with the same feature order used in stats computation.
    - `seq_lengths` is an array of shape (N,) with true lengths
      (number of non-padded steps) so we can reset padded positions to 0.
    """

    feature_order: List[str]
    per_feature: Dict[str, Dict[str, float]]

    @classmethod
    def from_json(cls, path: Path | str) -> "FeatureTransformer":
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        feature_order = data["feature_order"]
        per_feature = data["per_feature"]
        return cls(feature_order=feature_order, per_feature=per_feature)

    def transform_sequences(
        self,
        sequences: np.ndarray,
        seq_lengths: np.ndarray,
        copy: bool = True,
    ) -> np.ndarray:
        """Normalize and impute sequences in-place (or on a copy).

        Steps per feature:
        - If transform == "log1p-znorm":
          - apply log1p to all values (assumes non-negative inputs)
        - Impute NaNs with the feature mean (in transformed space).
        - Subtract mean, divide by std.

        After feature-wise normalization:
        - For each sequence i, set the padded prefix (positions 0..L-len_i-1)
          to 0 across all features so padding is always exactly 0.
        """
        if sequences.ndim != 3:
            raise ValueError(f"Expected sequences with shape (N, L, F), got {sequences.shape}")

        N, L, F = sequences.shape
        if F != len(self.feature_order):
            raise ValueError(
                f"Feature dimension {F} does not match feature_order length {len(self.feature_order)}"
            )

        if seq_lengths.shape[0] != N:
            raise ValueError("seq_lengths must have length N (number of sequences)")

        arr = sequences.astype(np.float32, copy=copy)

        for j, feat in enumerate(self.feature_order):
            spec = self.per_feature.get(feat)
            if spec is None:
                # If stats are missing, skip normalization for this feature.
                continue

            mean = float(spec.get("mean", 0.0))
            std = float(spec.get("std", 1.0)) or 1.0
            transform_type = spec.get("transform", "znorm")

            vals = arr[:, :, j]

            if transform_type == "log1p-znorm":
                # Assumes non-negative inputs; log1p(0) = 0
                vals = np.log1p(vals, where=~np.isnan(vals))

            # Impute NaNs with mean in transformed space
            nan_mask = np.isnan(vals)
            if np.any(nan_mask):
                vals[nan_mask] = mean

            vals -= mean
            vals /= std

            arr[:, :, j] = vals

        # Reset padding positions to 0 across all features
        seq_lengths = seq_lengths.astype(int)
        for i in range(N):
            valid_len = int(seq_lengths[i])
            if valid_len < 0 or valid_len > L:
                continue
            pad_len = L - valid_len
            if pad_len > 0:
                arr[i, :pad_len, :] = 0.0

        return arr


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute feature stats for Component X operational data.",
    )
    parser.add_argument(
        "--operational",
        type=str,
        default="data/train_operational_readouts.csv",
        help="Path to train_operational_readouts.csv.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="artifacts/feature_stats.json",
        help="Output JSON path for feature stats.",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=200_000,
        help="Chunk size for reading operational data.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = FeatureStatsConfig(
        operational_path=Path(args.operational),
        output_path=Path(args.output),
        chunksize=args.chunksize,
    )
    stats = compute_feature_stats(config)
    print(f"Computed feature stats for {len(stats['feature_order'])} features.")
    print(f"Counters: {len(stats['counters'])}, histograms: {len(stats['histograms'])}")
    print(f"Saved to {config.output_path}")


if __name__ == "__main__":
    main()


"""
Sequence window construction for Component X.

This module turns per-vehicle operational readouts and per-vehicle
proximity labels into fixed-length sequences suitable for LSTM training.

Design (Phase 3 defaults):
- Window length L = 128 time steps.
- Past-only windows: for each label, we use the L steps *before* and including
  the `reference_time_step`.
- Zero pre-padding for sequences shorter than L.
- We always return both the padded sequence and the true sequence length so
  the model can ignore padded positions.

Note:
- For simplicity and clarity, the current implementation loads the full
  `train_operational_readouts.csv` into memory once. This is acceptable for
  the Component X dataset (≈1.1M rows, 107 features) on a typical modern
  machine. If memory becomes an issue, this can be refactored to a streaming
  implementation that processes one vehicle at a time.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class SequenceWindowConfig:
    """Configuration for building fixed-length sequences."""

    operational_path: Path = Path("data/train_operational_readouts.csv")
    labels_path: Path = Path("data/train_proximity_labels.csv")
    output_path: Optional[Path] = None  # e.g., Path("data/train_sequences.npz")
    window_size: int = 128
    pad_value: float = 0.0
    max_windows_per_vehicle: Optional[int] = None


def build_sequences_for_training(
    config: Optional[SequenceWindowConfig] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build fixed-length sequences aligned with training proximity labels.

    Returns:
        sequences:  float32 array of shape (N, L, F)
        labels:     int64 array of shape (N,)
        seq_lengths:int64 array of shape (N,) – true lengths before padding
        vehicle_ids:int64 array of shape (N,)
        ref_times:  float32 array of shape (N,)
    """
    if config is None:
        config = SequenceWindowConfig()

    labels_df = pd.read_csv(config.labels_path)
    if labels_df.empty:
        raise ValueError(f"No labels found in {config.labels_path}")

    # Ensure expected columns exist
    required_label_cols = {"vehicle_id", "reference_time_step", "class_label"}
    missing = required_label_cols - set(labels_df.columns)
    if missing:
        raise ValueError(
            f"Missing columns in labels file {config.labels_path}: {missing}"
        )

    # Load operational readouts once and sort
    op_df = pd.read_csv(config.operational_path)
    if "vehicle_id" not in op_df.columns or "time_step" not in op_df.columns:
        raise ValueError(
            f"Operational file {config.operational_path} must contain "
            "`vehicle_id` and `time_step` columns."
        )

    op_df = op_df.sort_values(["vehicle_id", "time_step"]).reset_index(drop=True)

    # Identify feature columns (all except vehicle_id and time_step)
    feature_cols = [c for c in op_df.columns if c not in ("vehicle_id", "time_step")]
    num_features = len(feature_cols)

    # Group labels by vehicle for efficient per-vehicle processing
    labels_by_vehicle = dict(tuple(labels_df.groupby("vehicle_id")))

    sequences: List[np.ndarray] = []
    y: List[int] = []
    seq_lengths: List[int] = []
    vehicle_ids: List[int] = []
    ref_times: List[float] = []

    # Iterate over vehicles present in labels
    # (vehicles without labels are irrelevant for supervised training).
    op_grouped = op_df.groupby("vehicle_id")
    for vid, vehicle_labels in labels_by_vehicle.items():
        if vid not in op_grouped.groups:
            continue

        vehicle_ops = op_grouped.get_group(vid).sort_values("time_step")
        times = vehicle_ops["time_step"].to_numpy(dtype=float)
        feats = vehicle_ops[feature_cols].to_numpy(dtype=np.float32)

        # Optionally cap windows per vehicle
        max_windows = (
            config.max_windows_per_vehicle
            if config.max_windows_per_vehicle is not None
            else len(vehicle_labels)
        )

        used = 0
        for _, lab_row in vehicle_labels.iterrows():
            if used >= max_windows:
                break

            ref_t = float(lab_row["reference_time_step"])
            class_label = int(lab_row["class_label"])

            # Select all time_steps <= reference_time_step (past-only window)
            mask = times <= ref_t
            if not np.any(mask):
                # No history before reference_time_step; skip this label.
                continue

            hist_feats = feats[mask]
            seq_len = hist_feats.shape[0]

            if seq_len >= config.window_size:
                window = hist_feats[-config.window_size :]
                effective_len = config.window_size
            else:
                pad_len = config.window_size - seq_len
                pad_block = np.full(
                    (pad_len, num_features), config.pad_value, dtype=np.float32
                )
                window = np.vstack([pad_block, hist_feats])
                effective_len = seq_len

            sequences.append(window)
            y.append(class_label)
            seq_lengths.append(effective_len)
            vehicle_ids.append(int(vid))
            ref_times.append(ref_t)
            used += 1

    if not sequences:
        raise RuntimeError("No sequences were generated – check label and op files.")

    sequences_arr = np.stack(sequences, axis=0).astype(np.float32)
    labels_arr = np.asarray(y, dtype=np.int64)
    lengths_arr = np.asarray(seq_lengths, dtype=np.int64)
    vehicle_ids_arr = np.asarray(vehicle_ids, dtype=np.int64)
    ref_times_arr = np.asarray(ref_times, dtype=np.float32)

    if config.output_path is not None:
        config.output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            config.output_path,
            sequences=sequences_arr,
            labels=labels_arr,
            seq_lengths=lengths_arr,
            vehicle_ids=vehicle_ids_arr,
            reference_time_step=ref_times_arr,
        )

    return sequences_arr, labels_arr, lengths_arr, vehicle_ids_arr, ref_times_arr


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build fixed-length training sequences for Component X.",
    )
    parser.add_argument(
        "--operational",
        type=str,
        default="data/train_operational_readouts.csv",
        help="Path to train_operational_readouts.csv.",
    )
    parser.add_argument(
        "--labels",
        type=str,
        default="data/train_proximity_labels.csv",
        help="Path to train_proximity_labels.csv.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/train_sequences.npz",
        help="Output .npz path for sequences (set empty to skip saving).",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=128,
        help="Sequence window length L (default 128).",
    )
    parser.add_argument(
        "--max-windows-per-vehicle",
        type=int,
        default=None,
        help="Optional cap on number of windows per vehicle.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_path: Optional[Path]
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = None

    config = SequenceWindowConfig(
        operational_path=Path(args.operational),
        labels_path=Path(args.labels),
        output_path=output_path,
        window_size=args.window_size,
        pad_value=0.0,
        max_windows_per_vehicle=args.max_windows_per_vehicle,
    )

    sequences, labels, lengths, vids, refs = build_sequences_for_training(config)
    print(
        f"Built sequences: shape={sequences.shape}, "
        f"num_features={sequences.shape[-1]}"
    )
    print("Label distribution:")
    unique, counts = np.unique(labels, return_counts=True)
    total = labels.shape[0]
    for cls, cnt in zip(unique, counts):
        pct = 100.0 * cnt / max(total, 1)
        print(f"  class={cls}: {cnt} ({pct:.3f}%)")


if __name__ == "__main__":
    main()


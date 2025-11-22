"""
Quick, lightweight summaries for Component X CSV files.

Usage (from repo root):
    python -m src.data.dataset_summary --file data/sample_first_3_vehicles.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def summarize_csv(path: Path, n_rows: int = 100_000) -> None:
    """Print a small summary of a CSV file.

    The `n_rows` parameter caps the number of rows loaded for very large files,
    to avoid running out of memory during quick inspections. For the full
    SCANIA Component X dataset you should keep this reasonably small.
    """
    print(f"\n=== Summary for: {path} ===")
    if not path.exists():
        print("File not found.")
        return

    # Load up to n_rows rows for a quick summary.
    df = pd.read_csv(path, nrows=n_rows)

    print(f"Rows loaded (capped at {n_rows}): {len(df):,}")
    print(f"Columns ({len(df.columns)}):")
    for col in df.columns:
        print(f"  - {col} (dtype={df[col].dtype})")

    # Vehicle-level info if present
    if "vehicle_id" in df.columns:
        num_vehicles = df["vehicle_id"].nunique()
        print(f"\nUnique vehicles: {num_vehicles:,}")
        counts = df["vehicle_id"].value_counts().head(10)
        print("Top 10 vehicles by row count:")
        for vid, cnt in counts.items():
            print(f"  vehicle_id={vid}: {cnt} rows")

    # Time_step distribution if present
    if "time_step" in df.columns:
        print("\nTime_step stats:")
        print(df["time_step"].describe())

    # 5-class proximity label distribution if present
    label_col = None
    for candidate in ("class", "class_label", "proximity_label", "target_class"):
        if candidate in df.columns:
            label_col = candidate
            break
    if label_col is not None:
        print(f"\nLabel distribution for column '{label_col}':")
        value_counts = df[label_col].value_counts(normalize=False).sort_index()
        total = len(df)
        for label, cnt in value_counts.items():
            pct = 100.0 * cnt / max(total, 1)
            print(f"  class={label}: {cnt} ({pct:.3f}%)")

    # TTE repair/censoring distribution if present
    if "in_study_repair" in df.columns:
        print("\nTTE in_study_repair distribution:")
        counts = df["in_study_repair"].value_counts()
        events = int(counts.get(1, 0))
        censored = int(counts.get(0, 0))
        total = events + censored
        if total > 0:
            ev_pct = 100.0 * events / total
            cen_pct = 100.0 * censored / total
        else:
            ev_pct = cen_pct = 0.0
        print(f"  events (1):   {events} ({ev_pct:.3f}%)")
        print(f"  censored (0): {censored} ({cen_pct:.3f}%)")

    if "length_of_study_time_step" in df.columns:
        print("\nlength_of_study_time_step stats:")
        print(df["length_of_study_time_step"].describe())

    # Basic missingness
    print("\nMissing values (non-zero only):")
    missing = df.isna().sum()
    missing = missing[missing > 0]
    if missing.empty:
        print("  No missing values in the sampled rows.")
    else:
        for col, cnt in missing.items():
            print(f"  - {col}: {cnt} missing")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Component X CSV files.")
    parser.add_argument(
        "--file",
        type=str,
        required=True,
        help="Path to the CSV file to summarize (e.g., data/sample_first_3_vehicles.csv).",
    )
    parser.add_argument(
        "--n-rows",
        type=int,
        default=100_000,
        help="Maximum number of rows to load for the summary.",
    )
    args = parser.parse_args()

    summarize_csv(Path(args.file), n_rows=args.n_rows)


if __name__ == "__main__":
    main()

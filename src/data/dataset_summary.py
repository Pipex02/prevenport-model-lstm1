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
    to avoid running out of memory during quick inspections.
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


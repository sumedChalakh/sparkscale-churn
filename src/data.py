"""
generate_data.py
================
SparkScale Churn - Week 1 | Data Generation Script
Zaalima Development - Project 4

Purpose:
    The raw IBM Telco dataset has only ~7,000 rows.
    Spark is overkill at that scale. This script synthetically
    scales the dataset to ~2 Million rows so that distributed
    processing is actually justified — simulating a real
    telecom company's monthly user log volume.

How it works:
    1. Loads the original WA_Fn-UseC_-Telco-Customer-Churn.csv
    2. Adds realistic statistical noise to numeric columns
       (so rows aren't just duplicates — they simulate real variance)
    3. Regenerates unique CustomerIDs
    4. Saves the output as Parquet (columnar format, Spark-optimized)

Usage:
    python generate_data.py --input data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv
                            --output data/scaled/
                            --target_rows 2000000

Requirements:
    pip install pandas numpy pyarrow tqdm
"""

import argparse
import os
import uuid
import time

import numpy as np
import pandas as pd
from tqdm import tqdm


# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

# Numeric columns that will receive gaussian noise during scaling
# These mimic real-world variance in telecom usage patterns
NUMERIC_NOISY_COLS = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]

# Noise intensity: std dev as a fraction of the column's original std dev
# 0.05 = 5% noise — realistic drift without distorting distributions
NOISE_FACTOR = 0.05

# Parquet row-group size (tune for Spark read performance)
PARQUET_ROW_GROUP_SIZE = 100_000

# Chunk size for iterative scaling (controls RAM usage)
CHUNK_SIZE = 500_000


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def load_raw_data(filepath: str) -> pd.DataFrame:
    """
    Load and perform minimal cleaning on the raw Telco CSV.
    TotalCharges arrives as a string with spaces — fix that here.
    """
    print(f"\n[1/5] Loading raw data from: {filepath}")
    df = pd.read_csv(filepath)

    original_rows = len(df)
    print(f"      Raw row count      : {original_rows:,}")
    print(f"      Columns found      : {list(df.columns)}")

    # TotalCharges has blank strings for new customers (tenure=0)
    # Replace with 0.0 so the column is fully numeric
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)

    # Standardise the Churn column to binary int (0/1)
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    null_counts = df.isnull().sum()
    if null_counts.any():
        print(f"\n  ⚠  Nulls detected after cleaning:\n{null_counts[null_counts > 0]}")
    else:
        print("      Schema check       : ✅ No nulls after cleaning")

    return df


def compute_column_stats(df: pd.DataFrame) -> dict:
    """
    Pre-compute std deviations for numeric columns.
    Used to scale noise proportionally per column.
    """
    stats = {}
    for col in NUMERIC_NOISY_COLS:
        if col in df.columns:
            stats[col] = df[col].std()
    return stats


def generate_chunk(
    df_source: pd.DataFrame,
    col_stats: dict,
    chunk_size: int,
    chunk_index: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Generate one chunk of synthetic rows by:
      - Sampling (with replacement) from the source DataFrame
      - Adding proportional Gaussian noise to numeric columns
      - Regenerating unique CustomerIDs
      - Clipping values to realistic bounds (no negative charges)

    Args:
        df_source   : The cleaned original dataframe
        col_stats   : Dict of {col_name: std_dev}
        chunk_size  : How many rows to generate
        chunk_index : Used to seed CustomerID uniqueness
        rng         : Numpy random generator (seeded for reproducibility)

    Returns:
        pd.DataFrame of synthetic rows
    """
    # Sample rows with replacement from original data
    chunk = df_source.sample(n=chunk_size, replace=True, random_state=chunk_index).copy()
    chunk = chunk.reset_index(drop=True)

    # Add Gaussian noise to numeric columns
    for col in NUMERIC_NOISY_COLS:
        if col not in chunk.columns:
            continue

        noise_std = col_stats[col] * NOISE_FACTOR
        noise = rng.normal(loc=0.0, scale=noise_std, size=chunk_size)
        chunk[col] = chunk[col] + noise

        # Clip: tenure >= 0, charges >= 0
        chunk[col] = chunk[col].clip(lower=0.0)

    # Round tenure to integer (it's months)
    if "tenure" in chunk.columns:
        chunk["tenure"] = chunk["tenure"].round().astype(int)

    # Regenerate completely unique CustomerIDs
    # Format: CUST-{chunk_index}-{row_index} for debuggability
    chunk["customerID"] = [
        f"CUST-{chunk_index:04d}-{i:06d}" for i in range(chunk_size)
    ]

    return chunk


def scale_dataset(
    df: pd.DataFrame,
    target_rows: int,
    col_stats: dict,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Iteratively build the full scaled dataset in chunks.
    Chunked approach keeps RAM usage bounded regardless of target size.
    """
    print(f"\n[3/5] Scaling to {target_rows:,} rows (chunk size: {CHUNK_SIZE:,})")

    rng = np.random.default_rng(seed)
    chunks = []
    rows_generated = 0
    chunk_index = 0

    with tqdm(total=target_rows, unit="rows", unit_scale=True) as pbar:
        while rows_generated < target_rows:
            current_chunk_size = min(CHUNK_SIZE, target_rows - rows_generated)
            chunk = generate_chunk(df, col_stats, current_chunk_size, chunk_index, rng)
            chunks.append(chunk)
            rows_generated += current_chunk_size
            chunk_index += 1
            pbar.update(current_chunk_size)

    print(f"      Chunks generated   : {chunk_index}")
    df_scaled = pd.concat(chunks, ignore_index=True)
    print(f"      Final row count    : {len(df_scaled):,}")
    return df_scaled


def save_as_parquet(df: pd.DataFrame, output_dir: str) -> str:
    """
    Save the scaled DataFrame as a Parquet file.
    Parquet is the production-standard format for Spark ingestion:
      - Columnar storage → faster reads for feature-specific queries
      - Built-in compression → smaller disk footprint
      - Schema embedded → Spark infers types automatically
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "telco_churn_scaled.parquet")

    print(f"\n[4/5] Saving Parquet to: {output_path}")
    df.to_parquet(
        output_path,
        index=False,
        engine="pyarrow",
        row_group_size=PARQUET_ROW_GROUP_SIZE,
        compression="snappy",   # Snappy: fast decompression, good for Spark
    )

    size_mb = os.path.getsize(output_path) / (1024 ** 2)
    print(f"      File size          : {size_mb:.1f} MB")
    print(f"      Compression        : snappy")
    return output_path


def validate_output(parquet_path: str, target_rows: int) -> None:
    """
    Quick sanity check: reload the Parquet and verify row count + schema.
    This mimics what your PySpark ETL script will do in Week 1 Step 2.
    """
    print(f"\n[5/5] Validating output...")
    df_check = pd.read_parquet(parquet_path)

    print(f"      Row count          : {len(df_check):,} / {target_rows:,} target")
    print(f"      Columns            : {list(df_check.columns)}")
    print(f"      Dtypes:")
    for col, dtype in df_check.dtypes.items():
        print(f"        {col:<25} {dtype}")

    churn_dist = df_check["Churn"].value_counts(normalize=True) * 100
    print(f"\n      Churn distribution :")
    print(f"        No  Churn (0)     : {churn_dist.get(0, 0):.1f}%")
    print(f"        Yes Churn (1)     : {churn_dist.get(1, 0):.1f}%")

    assert len(df_check) == target_rows, \
        f"Row count mismatch: got {len(df_check)}, expected {target_rows}"

    print("\n  ✅  Validation passed. File is ready for Spark ingestion.")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Scale the Telco Churn dataset for Spark simulation."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv",
        help="Path to the raw Telco CSV file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/scaled/",
        help="Output directory for the Parquet file",
    )
    parser.add_argument(
        "--target_rows",
        type=int,
        default=2_000_000,
        help="Target number of rows in scaled dataset (default: 2,000,000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 55)
    print("  SparkScale Churn — Data Generation Script")
    print("  Zaalima Development | Project 4 | Week 1")
    print("=" * 55)
    print(f"  Target rows  : {args.target_rows:,}")
    print(f"  Random seed  : {args.seed}")
    print(f"  Output dir   : {args.output}")

    start_time = time.time()

    # Step 1: Load & clean
    df_raw = load_raw_data(args.input)

    # Step 2: Compute noise parameters
    print(f"\n[2/5] Computing column statistics for noise generation...")
    col_stats = compute_column_stats(df_raw)
    for col, std in col_stats.items():
        print(f"      {col:<20} std={std:.4f}  →  noise_std={std * NOISE_FACTOR:.4f}")

    # Step 3: Scale
    df_scaled = scale_dataset(df_raw, args.target_rows, col_stats, seed=args.seed)

    # Step 4: Save
    parquet_path = save_as_parquet(df_scaled, args.output)

    # Step 5: Validate
    validate_output(parquet_path, args.target_rows)

    elapsed = time.time() - start_time
    print(f"\n  ⏱  Total time: {elapsed:.1f}s")
    print("\n  Next step: Run etl.py to ingest this Parquet into PySpark.")
    print("=" * 55)


if __name__ == "__main__":
    main()



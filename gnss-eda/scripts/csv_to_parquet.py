#!/usr/bin/env python3
"""Convert CSV to Parquet using DuckDB streaming (no RAM spike)."""

import duckdb
from pathlib import Path

SRC = Path("/home/tempest/Documents/ISRO_Project/data/combined_csvs")
DST = Path("/home/tempest/Documents/ISRO_Project/data/gnss-eda/data/parquet")
DST.mkdir(parents=True, exist_ok=True)

# Skip > 800MB files (RAM constraint)
# SKIP = {"Raw.csv", "UncalGyro.csv", "GameRotationVector.csv", "FOP.csv", "OrientationDeg.csv", "UncalAccel.csv", "UncalMag.csv"}

con = duckdb.connect(":memory:")

for csv_path in sorted(SRC.glob("*.csv")):
    # if csv_path.name in SKIP:
    #     print(f"SKIP {csv_path.name} (> 800MB)")
    #     continue

    parquet_path = DST / csv_path.with_suffix(".parquet").name

    # DuckDB streams CSV → Parquet without loading full table
    con.execute(f"""
        COPY (SELECT * FROM read_csv_auto('{csv_path}', ignore_errors=true, header=true))
        TO '{parquet_path}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)

    print(f"✓ {csv_path.name} → {parquet_path.name}")

con.close()
print(f"\nDone. Output: {DST}")

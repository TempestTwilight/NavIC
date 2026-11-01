#!/usr/bin/env python3
"""EDA on GNSS data using DuckDB (lazy loading, 2GB RAM limit)."""

import duckdb
from pathlib import Path

PARQUET = Path("/home/tempest/Documents/ISRO_Project/data/gnss-eda/data/parquet")
DB = Path("/home/tempest/Documents/ISRO_Project/data/gnss-eda/database/gnss.duckdb")
DB.parent.mkdir(parents=True, exist_ok=True)

con = duckdb.connect(str(DB))

# Register parquet files as views (no load)
for p in PARQUET.glob("*.parquet"):
    table = p.stem.lower()
    con.execute(f"CREATE OR REPLACE VIEW {table} AS SELECT * FROM read_parquet('{p}')")
    print(f"Registered: {table}")

# Basic stats (lazy eval)
print("\n=== Row Counts ===")
for t in ["status", "accel", "agc", "fix", "nav"]:
    count = con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
    print(f"{t:12s}: {count:>10,}")

print("\n=== Status Schema ===")
print(con.execute("DESCRIBE status").df())

print("\n=== C/N0 Distribution ===")
print(
    con.execute("""
    SELECT 
        MIN(Cn0DbHz) as min_cn0,
        AVG(Cn0DbHz) as avg_cn0,
        MAX(Cn0DbHz) as max_cn0,
        STDDEV(Cn0DbHz) as std_cn0
    FROM status WHERE Cn0DbHz IS NOT NULL
""").df()
)

print("\n=== Constellation Counts ===")
print(
    con.execute("""
    SELECT ConstellationType, COUNT(*) as n
    FROM status
    GROUP BY ConstellationType
    ORDER BY n DESC
""").df()
)

print("\n=== UsedInFix Distribution ===")
print(
    con.execute("""
    SELECT UsedInFix, COUNT(*) as n, AVG(Cn0DbHz) as avg_cn0
    FROM status
    GROUP BY UsedInFix
""").df()
)

con.close()
print(f"\nDB saved: {DB}")

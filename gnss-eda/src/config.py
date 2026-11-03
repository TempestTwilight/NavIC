from pathlib import Path
import duckdb

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data" / "Raw_Logs"
PARQUET_DIR = PROJECT_ROOT / "gnss-eda" / "data" / "parquet"
SAMPLE_DIR = PROJECT_ROOT / "gnss-eda" / "data" / "samples"

DATABASE_DIR = PROJECT_ROOT / "gnss-eda" / "database"
REPORT_DIR = PROJECT_ROOT / "gnss-eda" / "reports"

DB_PATH = DATABASE_DIR / "gnss_eda.duckdb"

# Memory configuration
DUCKDB_MEMORY_LIMIT = "2GB"
DUCKDB_THREADS = 2
DUCKDB_TEMP_LIMIT = "20GB"

# Change this if your disk layout requires it.
DUCKDB_TEMP_DIR = Path.home() / "gnss-eda-temp"

# Sampling configuration
SAMPLE_SIZE = 100_000

print(
    PROJECT_ROOT,
    "\n",
    PARQUET_DIR,
    "\n",
    SAMPLE_DIR,
    "\n",
    DATABASE_DIR,
    "\n",
    REPORT_DIR,
)

con = duckdb.connect(str(PROJECT_ROOT / "eda.duckdb"))

con.execute("SET memory_limit = '3GB'")
con.execute("SET threads = 2")
con.execute("SET preserve_insertion_order = false")

con.execute(f"SET temp_directory = '{DUCKDB_TEMP_DIR}'")

con.execute("SET max_temp_directory_size = '20GB'")

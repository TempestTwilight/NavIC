import duckdb
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
TEMP_DIR = BASE_DIR / "duckdb_temp"
OUTPUT_DIR = BASE_DIR / "outputs"

TEMP_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

con = duckdb.connect(str(BASE_DIR / "eda.duckdb"))

con.execute("SET memory_limit = '3GB'")
con.execute("SET threads = 2")
con.execute("SET preserve_insertion_order = false")

con.execute(f"SET temp_directory = '{TEMP_DIR}'")

con.execute("SET max_temp_directory_size = '20GB'")

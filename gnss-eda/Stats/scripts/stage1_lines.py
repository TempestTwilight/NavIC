"""Stage 1: stream-convert each raw .txt GNSS Logger file into a per-file
parquet 'lines cache': (session, label, line). Byte-preserving except that
'#' comment lines and blank lines are dropped. Raw cache, not processed data.
"""

import duckdb
import os
import glob
import psutil
import time

RAW = "/home/tempest/Documents/ISRO_Project/data/Raw_Logs"
OUT = "/home/tempest/Documents/ISRO_Project/docs/EDA"
LINES = os.path.join(OUT, "scratch", "lines")
os.makedirs(LINES, exist_ok=True)

con = duckdb.connect(database=":memory:")
con.execute(
    "SET memory_limit='1GB'; SET threads=2; SET preserve_insertion_order=false;"
)
con.execute(f"SET temp_directory='{OUT}/scratch';")

files = sorted(glob.glob(os.path.join(RAW, "*.txt")))
t0 = time.time()
for i, f in enumerate(files):
    name = os.path.basename(f)[:-4]
    dest = os.path.join(LINES, name + ".parquet")
    if os.path.exists(dest):
        continue
    rss0 = psutil.Process().memory_info().rss / 1e6
    con.execute(f"""
        COPY (
            SELECT '{name}' AS session,
                   split_part(line, ',', 1) AS label,
                   line
            FROM read_csv('{f}', auto_detect=false, header=false, comment='#',
                          delim=E'\\x01', columns={{'line':'VARCHAR'}})
            WHERE line <> ''
        ) TO '{dest}' (COMPRESSION zstd)
    """)
    rss1 = psutil.Process().memory_info().rss / 1e6
    print(f"[{i + 1}/{len(files)}] {name} rss {rss0:.0f}->{rss1:.0f} MB", flush=True)
print("done in", round(time.time() - t0, 1), "s")

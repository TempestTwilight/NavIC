"""Phase B: per-session summaries for every table with a time column.
For each table: one output CSV row per session: n_rows, n_epochs (distinct
timestamps), tmin, tmax, duration, median/p90/p99/max gap (full per-session
lag computation — no sampling), n_negative_gaps, n_gaps>60s.
For Raw/Status additionally: rows-per-epoch stats (median, p90, max, n_epochs).
Run per table; DuckDB window partitions by session; memory kept < 1 GB via
temp spill if needed.
"""
import duckdb, os, sys, time, psutil

OUT = "/home/tempest/Documents/ISRO_Project/docs/EDA"
TYPED = os.path.join(OUT, "scratch", "typed")
PROF = os.path.join(OUT, "scratch", "profiles")
os.makedirs(PROF, exist_ok=True)
con = duckdb.connect(database=":memory:")
con.execute("SET memory_limit='1GB'; SET threads=2; SET preserve_insertion_order=false;")
con.execute(f"SET temp_directory='{PROF}';")

TABLES = {"Raw": "utcTimeMillis", "Status": "UnixTimeMillis", "Fix": "UnixTimeMillis",
          "Accel": "utcTimeMillis", "UncalAccel": "utcTimeMillis",
          "UncalGyro": "utcTimeMillis", "UncalMag": "utcTimeMillis",
          "OrientationDeg": "utcTimeMillis", "GameRotationVector": "utcTimeMillis",
          "FOP": "utcTimeMillis", "Agc": "utcTimeMillis"}

def gaps(table, tcol):
    return f"""
    COPY (
      WITH t AS (
        SELECT session, {tcol} AS ts FROM '{TYPED}/{table}.parquet'
        WHERE {tcol} IS NOT NULL
      ),
      d AS (
        SELECT session, ts - lag(ts) OVER (PARTITION BY session ORDER BY ts) AS gap FROM t
      ),
      dedup AS (
        SELECT session, ts, ts - lag(ts) OVER (PARTITION BY session ORDER BY ts) AS gap
        FROM (SELECT DISTINCT session, ts FROM t)
      )
      SELECT session,
             count(*) AS n_rows,
             (SELECT count(*) FROM (SELECT DISTINCT session, ts FROM t) x WHERE x.session = d.session) AS dummy,
             count(gap) AS n_gaps,
             count(gap) FILTER (gap < 0) AS n_negative_gaps,
             count(gap) FILTER (gap = 0) AS n_zero_gaps,
             approx_quantile(gap, 0.5) AS median_gap_ms,
             approx_quantile(gap, 0.9) AS p90_gap_ms,
             approx_quantile(gap, 0.99) AS p99_gap_ms,
             max(gap) AS max_gap_ms,
             count(gap) FILTER (gap > 60000) AS n_gap_gt60s -- ms (timestamps are ms)
      FROM d GROUP BY session ORDER BY session
    ) TO '{PROF}/{table}_gaps.csv' (HEADER)
    """
    # NOTE: 'dummy' removed below — simpler: two queries.

def run(table, tcol):
    t0 = time.time(); rss0 = psutil.Process().memory_info().rss/1e6
    # per-session gap stats on the DISTINCT-epoch series (correct pairing)
    q = f"""
      COPY (
        WITH t AS (SELECT session, {tcol} AS ts FROM '{TYPED}/{table}.parquet' WHERE {tcol} IS NOT NULL),
             u AS (SELECT DISTINCT session, ts FROM t),
             d AS (SELECT session, ts - lag(ts) OVER (PARTITION BY session ORDER BY ts) AS gap FROM u),
             dd AS (SELECT session, ts - lag(ts) OVER (PARTITION BY session ORDER BY ts) AS gap FROM t),
             per_sess AS (
               SELECT session, count(*) AS n_gaps,
                 count(gap) FILTER (gap < 0) AS n_negative_gaps,
                 approx_quantile(gap, 0.5) AS median_gap_ms,
                 approx_quantile(gap, 0.9) AS p90_gap_ms,
                 approx_quantile(gap, 0.99) AS p99_gap_ms,
                 max(gap) AS max_gap_ms,
                 count(gap) FILTER (gap > 60000) AS n_gap_gt60s -- ms (timestamps are ms)
               FROM d GROUP BY session)
        SELECT session, t1.n_rows, ep.n_epochs, t1.tmin, t1.tmax, t1.dur_s, p.*
        FROM (SELECT session, count(*) AS n_rows, min(ts) AS tmin, max(ts) AS tmax,
                     (max(ts)-min(ts))/1e3 AS dur_s FROM t GROUP BY session) t1
        JOIN (SELECT session, count(*) AS n_epochs FROM u GROUP BY session) ep USING (session)
        LEFT JOIN per_sess p USING (session)
        ORDER BY session
      ) TO '{PROF}/{table}_sessions.csv' (HEADER)
    """
    con.execute(q)
    # rows per epoch (only meaningful for Raw/Status/Agc — multi-row epochs)
    if table in ("Raw", "Status"):
        q2 = f"""
        COPY (
          SELECT session, count(DISTINCT ts) AS n_epochs,
                 approx_quantile(rpe, 0.5) AS median_rows_per_epoch,
                 approx_quantile(rpe, 0.9) AS p90_rows_per_epoch,
                 max(rpe) AS max_rows_per_epoch, min(rpe) AS min_rows_per_epoch
          FROM (
            SELECT session, ts, count(*) AS rpe FROM (
              SELECT session, {tcol} AS ts FROM '{TYPED}/{table}.parquet' WHERE {tcol} IS NOT NULL)
            GROUP BY session, ts
          ) GROUP BY session ORDER BY session
        ) TO '{PROF}/{table}_rowsperepoch.csv' (HEADER)"""
        con.execute(q2)
    rss1 = psutil.Process().memory_info().rss/1e6
    print(f"[{table}] {time.time()-t0:.1f}s rss {rss0:.0f}->{rss1:.0f} MB", flush=True)

for table, tcol in TABLES.items():
    run(table, tcol)
print("phaseB done")

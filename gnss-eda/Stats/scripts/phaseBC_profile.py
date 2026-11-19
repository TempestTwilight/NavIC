"""Phase B/C profiling: for each typed cache, compute per-column profiles and
per-table grain/time stats. Results saved as JSON per table in scratch/profiles/.
Every query is aggregate-only (memory-safe). Numeric profiling uses
approx_count_distinct and approx_quantile per the resource rules.
"""

import glob
import json
import os
import re
import time
import duckdb
import psutil

OUT = "/home/tempest/Documents/ISRO_Project/docs/EDA"
TYPED = os.path.join(OUT, "scratch", "typed")
PROF = os.path.join(OUT, "scratch", "profiles")
os.makedirs(PROF, exist_ok=True)

TIMECOL = {
    "Raw": "utcTimeMillis",
    "Status": "UnixTimeMillis",
    "Fix": "UnixTimeMillis",
    "Agc": "utcTimeMillis",
}
for t in [
    "Accel",
    "UncalAccel",
    "UncalGyro",
    "UncalMag",
    "OrientationDeg",
    "FOP",
    "GameRotationVector",
]:
    TIMECOL[t] = "utcTimeMillis"
TIMECOL["Nav"] = None


def cols_of(path):
    con = duckdb.connect()
    r = con.execute(f"DESCRIBE SELECT * FROM read_parquet('{path}')").fetchall()
    con.close()
    return [(c[0], c[1]) for c in r]


def profile_numeric(con, tbl, col):
    q = f"""
    SELECT count(*) AS n,
      count({col}) AS n_nonnull,
      count(*) - count({col}) AS n_null,
      approx_count_distinct({col}) AS n_distinct,
      min({col}) AS mn, max({col}) AS mx,
      avg({col}) AS mean, stddev_pop({col}) AS std,
      approx_quantile({col}, 0.01) AS p1, approx_quantile({col}, 0.05) AS p5,
      approx_quantile({col}, 0.25) AS p25, approx_quantile({col}, 0.50) AS p50,
      approx_quantile({col}, 0.75) AS p75, approx_quantile({col}, 0.95) AS p95,
      approx_quantile({col}, 0.99) AS p99,
      count(*) FILTER ({col} = 0) AS n_zero,
      count(*) FILTER ({col} < 0) AS n_neg,
      count(*) FILTER (isnan({col})) AS n_nan,
      count(*) FILTER (isinf({col})) AS n_inf
    FROM {tbl}"""
    r = con.execute(q).fetchone()
    keys = [
        "n",
        "n_nonnull",
        "n_null",
        "n_distinct",
        "mn",
        "mx",
        "mean",
        "std",
        "p1",
        "p5",
        "p25",
        "p50",
        "p75",
        "p95",
        "p99",
        "n_zero",
        "n_neg",
        "n_nan",
        "n_inf",
    ]
    return dict(zip(keys, r))


def profile_string(con, tbl, col):
    top = con.execute(f"""
      SELECT {col}, count(*) AS n FROM {tbl}
      WHERE {col} IS NOT NULL GROUP BY 1 ORDER BY n DESC LIMIT 10""").fetchall()
    st = con.execute(f"""
      SELECT count(*) - count({col}) AS n_null,
             approx_count_distinct({col}) AS n_distinct,
             min(length({col})) AS min_len, max(length({col})) AS max_len,
             avg(length({col})) AS avg_len
      FROM {tbl}""").fetchone()
    return {
        "n_null": st[0],
        "n_distinct": st[1],
        "min_len": st[2],
        "max_len": st[3],
        "avg_len": st[4],
        "top10": top,
    }


def table_meta(con, tbl, timecol):
    if timecol is None:
        return con.execute(
            f"SELECT count(*), count(DISTINCT session) FROM {tbl}"
        ).fetchone(), None
    r = con.execute(f"""
      SELECT count(*), count(DISTINCT session),
             count({timecol}), min({timecol}), max({timecol})
      FROM {tbl}""").fetchone()
    return r[:2], {"n_time": r[2], "tmin": r[3], "tmax": r[4]}


def gap_stats(con, tbl, timecol):
    # per-session median gap via per-session aggregate; avoids global sort
    q = f"""
    WITH g AS (
      SELECT session, {timecol} - lag({timecol}) OVER (
               PARTITION BY session ORDER BY {timecol}) AS gap
      FROM (SELECT session, {timecol} FROM {tbl} WHERE {timecol} IS NOT NULL
            USING SAMPLE 10 PERCENT) sample_s
    )
    SELECT count(gap) FILTER (gap < 0) AS n_negative_gap,
           count(gap) FILTER (gap = 0) AS n_zero_gap,
           approx_quantile(gap, 0.5)/1e6 AS median_gap_ms,
           approx_quantile(gap, 0.9)/1e6 AS p90_gap_ms,
           approx_quantile(gap, 0.99)/1e6 AS p99_gap_ms,
           max(gap)/1e6 AS max_gap_ms
    FROM g"""
    return con.execute(q).fetchone()


def main(tables=None):
    con = duckdb.connect(database=":memory:")
    con.execute(
        "SET memory_limit='1GB'; SET threads=2; SET preserve_insertion_order=false;"
    )
    con.execute(f"SET temp_directory='{PROF}';")
    files = sorted(glob.glob(os.path.join(TYPED, "*.parquet")))
    for path in files:
        tbl = os.path.basename(path)[:-8]
        if tables and tbl not in tables:
            continue
        t0 = time.time()
        rss0 = psutil.Process().memory_info().rss / 1e6
        cols = [
            c
            for c in cols_of(path)
            if c[0] not in ("session", "line", "_extra_field_1")
        ]
# Drop unused variable
        # register as view name tbl
        con.execute(
            f"CREATE OR REPLACE VIEW {tbl} AS SELECT * FROM read_parquet('{path}')"
        )
        meta, tm = table_meta(con, tbl, TIMECOL.get(tbl))
        gaps = gap_stats(con, tbl, TIMECOL.get(tbl)) if TIMECOL.get(tbl) else None
        prof = {}
        for cname, ctype in cols:
            if re.match(
                r"(BIGINT|HUGEINT|INTEGER|UBIGINT|DOUBLE|FLOAT|DECIMAL)", ctype
            ):
                prof[cname] = {"dtype": ctype, **profile_numeric(con, tbl, cname)}
            else:
                prof[cname] = {"dtype": ctype, **profile_string(con, tbl, cname)}
        out = {
            "table": tbl,
            "n_rows": meta[0],
            "n_sessions": meta[1],
            "time": tm,
            "gaps_sampled10pct": gaps,
            "columns": prof,
        }
        with open(os.path.join(PROF, tbl + ".json"), "w") as fh:
            json.dump(out, fh, indent=1, default=str)
        rss1 = psutil.Process().memory_info().rss / 1e6
        print(
            f"[profiled {tbl}] {meta[0]} rows, {len(cols)} cols, "
            f"{time.time() - t0:.1f}s, rss {rss0:.0f}->{rss1:.0f} MB",
            flush=True,
        )


if __name__ == "__main__":
    import sys

    main(sys.argv[1:] or None)

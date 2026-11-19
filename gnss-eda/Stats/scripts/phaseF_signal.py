# Phase F: C/N0 temporal behaviour analyses. Aggregate-only outputs.
# Outputs: scratch/profiles/phaseF_results.json + printed digest.
import duckdb
import os
import json
import time
import psutil

OUT = "/home/tempest/Documents/ISRO_Project/docs/EDA"
T = os.path.join(OUT, "scratch", "typed")
PROF = os.path.join(OUT, "scratch", "profiles")
res = {}

con = duckdb.connect(":memory:")
con.execute(
    "SET memory_limit='1GB'; SET threads=2; SET preserve_insertion_order=false;"
)
con.execute(f"SET temp_directory='{PROF}';")
con.execute(
    "CREATE VIEW m AS SELECT * FROM read_csv_auto('%s/scratch/sessions_meta.csv', header=true)"
    % OUT
)


def rss():
    return psutil.Process().memory_info().rss / 1e6


t0 = time.time()
r0 = rss()

# ---- F1: C/N0 lag correlation at h = 1, 5, 15 epochs (1 Hz epochs), per signal
# track (session, ConstellationType, Svid, CarrierFrequencyHz); keep pairs whose
# actual time difference matches h s (±150 ms).
con.execute(f"""CREATE VIEW pairs AS
SELECT session, utcTimeMillis AS t, ConstellationType AS ct, Svid, CarrierFrequencyHz AS freq,
       Cn0DbHz AS cn0 FROM read_parquet('{T}/Raw.parquet') WHERE Cn0DbHz IS NOT NULL""")
for h in (1, 5, 15):
    q = f"""
    WITH l AS (
      SELECT ct, svid, freq, session, t, cn0,
             lag(t, {h}) OVER w AS t_lag, lag(cn0, {h}) OVER w AS cn0_lag
      FROM pairs WINDOW w AS (PARTITION BY session, ct, svid, freq ORDER BY t)
    )
    SELECT count(*) AS n_pairs,
           corr(cn0, cn0_lag) AS corr,
           approx_quantile(cn0 - cn0_lag, 0.5) AS med_diff,
           approx_quantile(abs(cn0 - cn0_lag), 0.5) AS med_abs_diff,
           approx_quantile(abs(cn0 - cn0_lag), 0.9) AS p90_abs_diff
    FROM l WHERE abs(t - t_lag - {h} * 1000) <= 150 AND cn0_lag IS NOT NULL
    """
    res[f"lag{h}"] = con.execute(q).fetchone()
    print(f"[F1 lag{h}] {res[f'lag{h}']} rss={rss():.0f} MB", flush=True)

# ---- F2: survivorship — fraction of (epoch, signal) still tracked at +h s
# (same identity key), computed per session and summed.
surv = {}
for h in (1, 5, 15):
    matched = 0
    total = 0
    for (s,) in con.execute("SELECT DISTINCT session FROM pairs").fetchall():
        a = f"SELECT t, ct, svid, freq FROM pairs WHERE session='{s}'"
        b = f"SELECT t - {h * 1000} AS t, ct, svid, freq FROM pairs WHERE session='{s}'"
        row_n = con.execute(
            f"SELECT count(*) FROM ({a}) x SEMI JOIN ({b}) y USING(t, ct, svid, freq)"
        ).fetchone()
        n = row_n[0] if row_n is not None else 0
        row_tot = con.execute(f"SELECT count(*) FROM ({a})").fetchone()
        tot = row_tot[0] if row_tot is not None else 0
        matched += n
        total += tot
    surv[f"h{h}"] = (matched, total)
    print(
        f"[F2 surv+{h}s] {matched:,}/{total:,} = {100 * matched / total:.1f}% rss={rss():.0f} MB",
        flush=True,
    )
res["survivorship"] = {
    k: {"matched": v[0], "total": v[1], "pct": 100 * v[0] / v[1]}
    for k, v in surv.items()
}

# ---- F3: elevation bins vs C/N0 (Status rows carry both fields)
q = f"""
SELECT floor((ElevationDegrees + 5) / 5.0) AS bin,
       bin*5 - 10 + 2.5 AS bin_center,
       count(*) AS n,
       round(approx_quantile(Cn0DbHz, 0.5), 1) AS med_cn0,
       round(approx_quantile(Cn0DbHz, 0.25), 1) AS p25,
       round(approx_quantile(Cn0DbHz, 0.75), 1) AS p75,
       round(100.0*count(*) FILTER (WHERE Cn0DbHz = 0)/count(*), 1) AS pct_zero
FROM read_parquet('{T}/Status.parquet')
WHERE ElevationDegrees IS NOT NULL AND Cn0DbHz IS NOT NULL
GROUP BY bin ORDER BY bin
"""
rows = con.execute(q).fetchall()
res["elevation_bins"] = [
    {
        "bin_center_deg": r[1],
        "n": r[2],
        "med_cn0": r[3],
        "p25": r[4],
        "p75": r[5],
        "pct_zero": r[6],
    }
    for r in rows
]
print(f"[F3 elevation] {len(rows)} bins rss={rss():.0f} MB", flush=True)

# ---- F4: within-epoch C/N0 spread (std across signals per epoch), by device
con.execute(f"""CREATE VIEW rawdev AS
SELECT r.*, m.model FROM read_parquet('{T}/Raw.parquet') r JOIN m USING(session)""")
q = """
WITH ep AS (
  SELECT session, model, utcTimeMillis AS t, stddev_samp(Cn0DbHz) AS sd, count(*) AS n
  FROM rawdev GROUP BY ALL HAVING count(*) > 1)
SELECT model, count(*) AS epochs,
       round(approx_quantile(sd, 0.5), 2) AS med_sd,
       round(approx_quantile(sd, 0.9), 2) AS p90_sd
FROM ep GROUP BY model ORDER BY model
"""
res["epoch_spread"] = [
    dict(zip(["model", "epochs", "med_sd", "p90_sd"], r))
    for r in con.execute(q).fetchall()
]
print(f"[F4 epoch spread] {res['epoch_spread']} rss={rss():.0f} MB", flush=True)

# ---- F5: AGC — Raw.AgcDb availability and relation to epoch mean C/N0
q = """
SELECT model, count(*) AS rows, count(*) FILTER (WHERE AgcDb IS NOT NULL) AS agc_nonnull,
       count(DISTINCT AgcDb) AS distinct_agc
FROM rawdev GROUP BY model ORDER BY model
"""
res["raw_agc_by_device"] = [
    dict(zip(["model", "rows", "agc_nonnull", "distinct_agc"], r))
    for r in con.execute(q).fetchall()
]
print(f"[F5 raw agc] {res['raw_agc_by_device']}", flush=True)
# per-epoch AgcDb vs mean Cn0 on the devices that report AgcDb in Raw
q = """
WITH ep AS (SELECT session, utcTimeMillis AS t, avg(Cn0DbHz) AS mean_cn0, avg(AgcDb) AS agc
            FROM rawdev WHERE AgcDb IS NOT NULL GROUP BY ALL)
SELECT corr(agc, mean_cn0) AS corr_agc_cn0, count(*) AS epochs,
       round(min(agc), 1) AS min_agc, round(max(agc), 1) AS max_agc
FROM ep
"""
res["raw_agc_vs_cn0"] = con.execute(q).fetchone()
print(f"[F5 corr] {res['raw_agc_vs_cn0']}", flush=True)

# ---- F6: C/N0=0 rows by constellation (Raw) for target-cleaning rules
q = """
SELECT ConstellationType AS ct, count(*) AS rows,
       count(*) FILTER (WHERE Cn0DbHz = 0) AS zero_rows,
       round(100.0 * count(*) FILTER (WHERE Cn0DbHz = 0) / count(*), 1) AS pct_zero
FROM rawdev GROUP BY ct ORDER BY ct
"""
res["cn0_zero_by_constellation"] = [
    dict(zip(["ct", "rows", "zero_rows", "pct_zero"], r))
    for r in con.execute(q).fetchall()
]
print(f"[F6 cn0=0] {res['cn0_zero_by_constellation']}", flush=True)

# ---- F7: per-session C/N0 mean trend vs session minute (drift within session)
q = """
WITH s AS (SELECT session, min(utcTimeMillis) AS t0 FROM rawdev GROUP BY session)
SELECT round((r.utcTimeMillis - s.t0)/60000.0, 0) AS minute,
       round(avg(r.Cn0DbHz) FILTER (WHERE r.Cn0DbHz > 0), 2) AS mean_cn0,
       count(*) AS n
FROM rawdev r JOIN s USING(session)
WHERE r.Cn0DbHz > 0
GROUP BY minute ORDER BY minute LIMIT 50
"""
res["cn0_by_session_minute"] = con.execute(q).fetchall()
print(
    f"[F7 minute trend] {len(res['cn0_by_session_minute'])} minutes rss={rss():.0f} MB",
    flush=True,
)

json.dump(res, open(os.path.join(PROF, "phaseF_results.json"), "w"), indent=1)
print(f"phaseF done in {time.time() - t0:.0f}s, peak rss {rss():.0f} MB")

# Phase G: orientation conventions, quaternion vs Euler, gravity check.
# Aggregate-only outputs → scratch/profiles/phaseG_results.json
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


def rss():
    return psutil.Process().memory_info().rss / 1e6


t0 = time.time()

# ---- G1: quaternion → Euler candidate mappings vs OrientationDeg
# Join on the exact (session, elapsedRealtimeNanos) stamps (99.997% match, STEP-021).
# 2% sample to bound memory. FOP/GRV quaternions are unit-norm checked separately.
q = f"""
WITH j AS (
  SELECT o.yawDeg, o.rollDeg, o.pitchDeg, g.x, g.y, g.z, g.w
  FROM read_parquet('{T}/OrientationDeg.parquet') o
  JOIN read_parquet('{T}/GameRotationVector.parquet') g
    USING(session, elapsedRealtimeNanos)
  WHERE random() < 0.02
),
e AS (
  SELECT yawDeg, rollDeg, pitchDeg, x, y, z, w,
    -- candidate azimuth (about +Z, Android getOrientation values[0])
    degrees(atan2(2*(x*y + w*z), w*w + x*x - y*y - z*z)) AS az_cand,
    -- candidate pitch (asin term, two sign variants)
    degrees(asin(least(1.0, greatest(-1.0, 2*(w*y - x*z))))) AS pitch_cand,
    degrees(asin(least(1.0, greatest(-1.0, 2*(x*z - w*y))))) AS pitch_cand_neg,
    -- candidate roll (atan2 term, two sign variants)
    degrees(atan2(2*(w*x + y*z), w*w - x*x - y*y + z*z)) AS roll_cand,
    degrees(atan2(2*(w*x + y*z), -(w*w - x*x - y*y + z*z))) AS roll_cand_neg
  FROM j
),
d AS (
  SELECT *,
    least(abs(((yawDeg - az_cand) % 360 + 360) % 360), abs(((az_cand - yawDeg) % 360 + 360) % 360)) AS yaw_diff,
    abs(pitchDeg - pitch_cand) AS pd1, abs(pitchDeg - pitch_cand_neg) AS pd2,
    abs(rollDeg - roll_cand) AS rd1, abs(rollDeg - roll_cand_neg) AS rd2,
    abs(rollDeg - pitch_cand) AS rp1, abs(pitchDeg - roll_cand) AS pr1
  FROM e
)
SELECT
  count(*) AS n,
  round(approx_quantile(yaw_diff, 0.5), 2) AS yaw_med_absdiff,
  round(avg(CASE WHEN yaw_diff < 5 THEN 1.0 ELSE 0.0 END), 3) AS yaw_frac_lt5deg,
  round(approx_quantile(pd1, 0.5), 2) AS pitch_med_absdiff,
  round(approx_quantile(pd2, 0.5), 2) AS pitchneg_med_absdiff,
  round(approx_quantile(rd1, 0.5), 2) AS roll_med_absdiff,
  round(approx_quantile(rd2, 0.5), 2) AS rollneg_med_absdiff,
  round(approx_quantile(rp1, 0.5), 2) AS roll_as_pitch_med_absdiff,
  round(approx_quantile(pr1, 0.5), 2) AS pitch_as_roll_med_absdiff
FROM d
"""
r = con.execute(q).fetchone()
res["euler_vs_quat"] = dict(
    zip(
        [
            "n",
            "yaw_med_absdiff",
            "yaw_frac_lt5deg",
            "pitch_med_absdiff",
            "pitchneg_med_absdiff",
            "roll_med_absdiff",
            "rollneg_med_absdiff",
            "roll_as_pitch_med_absdiff",
            "pitch_as_roll_med_absdiff",
        ],
        r,
    )
)
print("[G1 euler]", res["euler_vs_quat"], f"rss={rss():.0f} MB", flush=True)

# ---- G2: FOP heading vs OrientationDeg yaw (exact-stamp match test + sampled compare)
q = f"""
SELECT count(*) AS fop_rows,
  count(*) FILTER (WHERE o.elapsedRealtimeNanos IS NOT NULL) AS stamp_match
FROM read_parquet('{T}/FOP.parquet') f
LEFT JOIN (SELECT DISTINCT session, elapsedRealtimeNanos FROM read_parquet('{T}/OrientationDeg.parquet')) o
  USING(session, elapsedRealtimeNanos)
"""
r = con.execute(q).fetchone()
res["fop_stamp_match"] = dict(zip(["fop_rows", "stamp_match"], r))
print("[G2 stamps]", res["fop_stamp_match"], flush=True)
q = f"""
WITH j AS (
  SELECT f.headingDegrees AS h, o.yawDeg
  FROM read_parquet('{T}/FOP.parquet') f
  JOIN read_parquet('{T}/OrientationDeg.parquet') o USING(session, elapsedRealtimeNanos)
  WHERE random() < 0.02
),
d AS (
  SELECT *,
    least(abs(((yawDeg - h) % 360 + 360) % 360), abs(((h - yawDeg) % 360 + 360) % 360)) AS circ_diff
  FROM j
)
SELECT count(*) AS n,
  round(approx_quantile(circ_diff, 0.5), 2) AS med_absdiff_deg,
  round(avg(CASE WHEN circ_diff < 5 THEN 1.0 ELSE 0.0 END), 3) AS frac_lt5deg,
  round(avg(CASE WHEN circ_diff < 10 THEN 1.0 ELSE 0.0 END), 3) AS frac_lt10deg
FROM d
"""
r = con.execute(q).fetchone()
res["fop_vs_ori_yaw"] = dict(
    zip(["n", "med_absdiff_deg", "frac_lt5deg", "frac_lt10deg"], r)
)
print("[G2 fop-yaw]", res["fop_vs_ori_yaw"], flush=True)

# ---- G3: quaternion norms
for tbl in ("GameRotationVector", "FOP"):
    q = f"""
    WITH s AS (SELECT x*x + y*y + z*z + w*w AS n2 FROM read_parquet('{T}/{tbl}.parquet') WHERE random() < 0.01)
    SELECT round(approx_quantile(n2, 0.5), 6), round(min(n2), 6), round(max(n2), 6) FROM s
    """
    r = con.execute(q).fetchone()
    res[f"norm_{tbl}"] = dict(zip(["med", "min", "max"], r))
    print(f"[G3 norm {tbl}]", res[f"norm_{tbl}"], flush=True)

# ---- G4: accelerometer gravity check (calibrated Accel rows)
q = f"""
SELECT round(approx_quantile(sqrt(AccelXMps2*AccelXMps2 + AccelYMps2*AccelYMps2 + AccelZMps2*AccelZMps2), 0.5), 3) AS med_norm,
       round(approx_quantile(sqrt(AccelXMps2*AccelXMps2 + AccelYMps2*AccelYMps2 + AccelZMps2*AccelZMps2), 0.05), 3) AS p05_norm,
       round(approx_quantile(sqrt(AccelXMps2*AccelXMps2 + AccelYMps2*AccelYMps2 + AccelZMps2*AccelZMps2), 0.95), 3) AS p95_norm,
       round(approx_quantile(AccelZMps2, 0.5), 2) AS med_z,
       round(approx_quantile(pitchDeg, 0.5), 1) AS med_pitch
FROM read_parquet('{T}/Accel.parquet') a
JOIN read_parquet('{T}/OrientationDeg.parquet') o USING(session, elapsedRealtimeNanos)
"""
r = con.execute(q).fetchone()
res["accel_gravity"] = dict(
    zip(["med_norm", "p05_norm", "p95_norm", "med_z", "med_pitch"], r)
)
print("[G4 accel]", res["accel_gravity"], flush=True)

# ---- G5: yaw integer-like check + orientation grid occupancy (yaw octants × face)
q = f"""
SELECT count(*) AS n,
  round(avg(CASE WHEN yawDeg = floor(yawDeg) THEN 1.0 ELSE 0.0 END), 5) AS frac_integer,
  count(DISTINCT yawDeg) AS distinct_yaw
FROM read_parquet('{T}/OrientationDeg.parquet')
"""
r = con.execute(q).fetchone()
res["yaw_integerlike"] = dict(zip(["n", "frac_integer", "distinct_yaw"], r))
print("[G5 yaw]", res["yaw_integerlike"], flush=True)

json.dump(res, open(os.path.join(PROF, "phaseG_results.json"), "w"), indent=1)
print(f"phaseG done in {time.time() - t0:.0f}s, peak rss {rss():.0f} MB")

"""Phase G extra queries for 06_orientation_and_sessions.md"""

import duckdb
import json
import os

T = "/home/tempest/Documents/ISRO_Project/docs/EDA/scratch/typed"
PROF = "/home/tempest/Documents/ISRO_Project/docs/EDA/scratch/profiles"
con = duckdb.connect(":memory:")
con.execute(
    "SET memory_limit='1GB'; SET threads=2; SET preserve_insertion_order=false;"
)
con.execute(f"SET temp_directory='{PROF}';")
res = {}

# =====================================================================
# GQ1: Row counts per table for 14 named sessions
# =====================================================================
q = f"""
WITH ori AS (SELECT session, count(*) AS ori_rows FROM read_parquet('{T}/OrientationDeg.parquet')
        WHERE session NOT LIKE 'gnss_log_%' GROUP BY session),
grv AS (SELECT session, count(*) AS grv_rows FROM read_parquet('{T}/GameRotationVector.parquet')
        WHERE session NOT LIKE 'gnss_log_%' GROUP BY session),
fop AS (SELECT session, count(*) AS fop_rows FROM read_parquet('{T}/FOP.parquet')
        WHERE session NOT LIKE 'gnss_log_%' GROUP BY session),
acc AS (SELECT session, count(*) AS accel_rows FROM read_parquet('{T}/Accel.parquet')
        WHERE session NOT LIKE 'gnss_log_%' GROUP BY session)
SELECT ori.session, ori.ori_rows, grv.grv_rows, coalesce(fop.fop_rows,0) AS fop_rows, coalesce(acc.accel_rows,0) AS accel_rows
FROM ori
JOIN grv USING(session)
LEFT JOIN fop USING(session)
LEFT JOIN acc USING(session)
ORDER BY ori.session
"""
rows = con.execute(q).fetchall()
res["named_session_rowcounts"] = [list(r) for r in rows]
print("[GQ1 named session rowcounts]", len(rows), "sessions")
for r in rows:
    print(" ", r)

# =====================================================================
# GQ2: Per named-session angle stats from OrientationDeg
# =====================================================================
q = f"""
SELECT session,
  round(avg(yawDeg),1) AS yaw_mean,
  round(stddev(yawDeg),1) AS yaw_std,
  min(yawDeg) AS yaw_min, max(yawDeg) AS yaw_max,
  round(avg(pitchDeg),1) AS pitch_mean,
  round(stddev(pitchDeg),1) AS pitch_std,
  min(pitchDeg) AS pitch_min, max(pitchDeg) AS pitch_max,
  round(avg(rollDeg),1) AS roll_mean,
  round(stddev(rollDeg),1) AS roll_std,
  min(rollDeg) AS roll_min, max(rollDeg) AS roll_max
FROM read_parquet('{T}/OrientationDeg.parquet')
WHERE session NOT LIKE 'gnss_log_%'
GROUP BY session ORDER BY session
"""
rows = con.execute(q).fetchall()
res["named_ori_stats"] = [list(r) for r in rows]
print("[GQ2 named ori stats]")
for r in rows:
    print(" ", r)

# =====================================================================
# GQ3: FOP does NOT have pitchDeg or rollDeg, only x, y, z, w, headingDegrees, headingErrorDegrees, conservativeHeadingErrorDegrees.
# Let's inspect FOP sessions and row counts for named sessions:
# =====================================================================
# In GQ1, we saw that ALL 14 named sessions have fop_rows = 0 and accel_rows = 0!
# FOP is only present in 75 sessions (gnss_log_* sessions)!
# Let's check which sessions actually have FOP:
q = f"""
SELECT session,
  round(avg(headingDegrees),1) AS hdg_mean,
  round(stddev(headingDegrees),1) AS hdg_std,
  min(headingDegrees) AS hdg_min, max(headingDegrees) AS hdg_max
FROM read_parquet('{T}/FOP.parquet')
GROUP BY session ORDER BY session LIMIT 10
"""
rows = con.execute(q).fetchall()
res["fop_sample_stats"] = [list(r) for r in rows]
print("[GQ3 FOP sample stats for gnss_log sessions]")
for r in rows:
    print(" ", r)

# =====================================================================
# GQ4: OrientationDeg global angle ranges
# =====================================================================
q = f"""
SELECT
  min(yawDeg) AS yaw_min, max(yawDeg) AS yaw_max, count(DISTINCT yawDeg) AS yaw_dist,
  min(pitchDeg) AS pitch_min, max(pitchDeg) AS pitch_max,
  min(rollDeg) AS roll_min, max(rollDeg) AS roll_max
FROM read_parquet('{T}/OrientationDeg.parquet')
"""
r = con.execute(q).fetchone()
res["ori_global_ranges"] = list(r)
print("[GQ4 OrientationDeg global ranges]", r)

# =====================================================================
# GQ5: FOP global angle ranges (headingDegrees, and euler pitch/roll from quaternion)
# =====================================================================
q = f"""
SELECT
  min(headingDegrees) AS hdg_min, max(headingDegrees) AS hdg_max, count(DISTINCT headingDegrees) AS hdg_dist,
  min(headingErrorDegrees) AS err_min, max(headingErrorDegrees) AS err_max
FROM read_parquet('{T}/FOP.parquet')
"""
r = con.execute(q).fetchone()
res["fop_global_ranges"] = list(r)
print("[GQ5 FOP global ranges]", r)

# =====================================================================
# GQ6: GameRotationVector global component ranges
# =====================================================================
q = f"""
SELECT
  round(min(x),4) AS x_min, round(max(x),4) AS x_max,
  round(min(y),4) AS y_min, round(max(y),4) AS y_max,
  round(min(z),4) AS z_min, round(max(z),4) AS z_max,
  round(min(w),4) AS w_min, round(max(w),4) AS w_max,
  round(avg(w),4) AS w_mean,
  round(sum(CASE WHEN w < 0 THEN 1 ELSE 0 END) * 1.0 / count(*), 5) AS frac_w_negative
FROM read_parquet('{T}/GameRotationVector.parquet')
"""
r = con.execute(q).fetchone()
res["grv_ranges"] = list(r)
print("[GQ6 GRV ranges]", r)

# =====================================================================
# GQ7: FOP w negative fraction (from G2 stamps: FOP w sometimes negative)
# =====================================================================
q = f"""
SELECT
  round(sum(CASE WHEN w < 0 THEN 1 ELSE 0 END) * 1.0 / count(*), 5) AS frac_w_negative,
  count(*) AS total
FROM read_parquet('{T}/FOP.parquet')
"""
r = con.execute(q).fetchone()
res["fop_w_negative"] = list(r)
print("[GQ7 FOP w negative]", r)

# =====================================================================
# GQ8: Per-session Accel gravity norm stats (all sessions that have Accel)
# =====================================================================
q = f"""
SELECT session,
  round(avg(sqrt(AccelXMps2*AccelXMps2 + AccelYMps2*AccelYMps2 + AccelZMps2*AccelZMps2)), 3) AS mean_norm,
  round(stddev(sqrt(AccelXMps2*AccelXMps2 + AccelYMps2*AccelYMps2 + AccelZMps2*AccelZMps2)), 4) AS std_norm,
  round(min(sqrt(AccelXMps2*AccelXMps2 + AccelYMps2*AccelYMps2 + AccelZMps2*AccelZMps2)), 3) AS min_norm,
  round(max(sqrt(AccelXMps2*AccelXMps2 + AccelYMps2*AccelYMps2 + AccelZMps2*AccelZMps2)), 3) AS max_norm,
  count(*) AS n_rows
FROM read_parquet('{T}/Accel.parquet')
GROUP BY session ORDER BY session
"""
rows = con.execute(q).fetchall()
res["accel_per_session"] = [list(r) for r in rows]
print("[GQ8 Accel per session]", len(rows), "sessions")
for r in rows:
    print(" ", r)

# =====================================================================
# GQ9: Within-session pitch/roll std per session (OrientationDeg, named only)
# =====================================================================
q = f"""
SELECT session,
  round(stddev(pitchDeg),2) AS pitch_std,
  round(avg(pitchDeg),1) AS pitch_mean,
  round(stddev(rollDeg),2) AS roll_std,
  round(avg(rollDeg),1) AS roll_mean
FROM read_parquet('{T}/OrientationDeg.parquet')
WHERE session NOT LIKE 'gnss_log_%'
GROUP BY session ORDER BY session
"""
rows = con.execute(q).fetchall()
res["named_pitch_roll_std"] = [list(r) for r in rows]
print("[GQ9 named pitch/roll std]")
for r in rows:
    print(" ", r)

# =====================================================================
# GQ10: GRV Euler yaw: compute degrees(atan2(2*(x*y+w*z), w*w+x*x-y*y-z*z))
# Compare to OrientationDeg yaw, for one session (0_down_facing), first 2000 rows
# to show drift
# =====================================================================
q = f"""
WITH j AS (
  SELECT o.elapsedRealtimeNanos AS t,
    o.yawDeg AS ori_yaw,
    degrees(atan2(2*(g.x*g.y + g.w*g.z), g.w*g.w + g.x*g.x - g.y*g.y - g.z*g.z)) AS grv_az
  FROM read_parquet('{T}/OrientationDeg.parquet') o
  JOIN read_parquet('{T}/GameRotationVector.parquet') g
    USING(session, elapsedRealtimeNanos)
  WHERE o.session = '0_down_facing'
  ORDER BY t LIMIT 500
)
SELECT
  round(avg(least(abs(((ori_yaw - grv_az) % 360 + 360) % 360),
                  abs(((grv_az - ori_yaw) % 360 + 360) % 360))), 2) AS mean_circ_diff,
  round(stddev(least(abs(((ori_yaw - grv_az) % 360 + 360) % 360),
                     abs(((grv_az - ori_yaw) % 360 + 360) % 360))), 2) AS std_circ_diff,
  count(*) AS n
FROM j
"""
r = con.execute(q).fetchone()
res["grv_vs_ori_yaw_sample"] = list(r)
print("[GQ10 GRV vs OrientationDeg yaw drift]", r)

# =====================================================================
# GQ11: Across all sessions: range of mean pitchDeg and rollDeg (OrientationDeg)
# =====================================================================
q = f"""
SELECT
  round(min(mean_pitch),1) AS min_mean_pitch,
  round(max(mean_pitch),1) AS max_mean_pitch,
  round(min(mean_roll),1) AS min_mean_roll,
  round(max(mean_roll),1) AS max_mean_roll
FROM (
  SELECT session,
    avg(pitchDeg) AS mean_pitch,
    avg(rollDeg) AS mean_roll
  FROM read_parquet('{T}/OrientationDeg.parquet')
  GROUP BY session
) sub
"""
r = con.execute(q).fetchone()
res["all_sessions_pitch_roll_range"] = list(r)
print("[GQ11 all sessions pitch/roll range]", r)

# =====================================================================
# GQ12: OrientationDeg yaw vs FOP heading comparison for a single session that HAS both
# Let's find sessions that have both OrientationDeg and FOP:
# =====================================================================
# Session 0_down_facing has NO FOP. We pick gnss_log_2026_10_22_08_30_05 (or similar).
q = f"""
WITH j AS (
  SELECT o.yawDeg, f.headingDegrees AS hdg
  FROM read_parquet('{T}/OrientationDeg.parquet') o
  JOIN read_parquet('{T}/FOP.parquet') f USING(session, elapsedRealtimeNanos)
  WHERE o.session = 'gnss_log_2026_10_22_08_30_05'
),
d AS (
  SELECT yawDeg, hdg,
    least(abs(((yawDeg - hdg) % 360 + 360) % 360),
          abs(((hdg - yawDeg) % 360 + 360) % 360)) AS circ_diff
  FROM j
)
SELECT count(*) AS n,
  round(avg(circ_diff), 2) AS mean_diff,
  round(stddev(circ_diff), 2) AS std_diff,
  round(approx_quantile(circ_diff, 0.5), 2) AS med_diff,
  round(min(circ_diff), 2) AS min_diff,
  round(max(circ_diff), 2) AS max_diff
FROM d
"""
r = con.execute(q).fetchone()
res["fop_vs_ori_yaw_single_session"] = list(r)
print("[GQ12 FOP vs OrientationDeg yaw for gnss_log_2026_10_22_08_30_05]", r)

# =====================================================================
# GQ13: FOP quaternion-derived pitch/roll vs OrientationDeg pitch/roll
# =====================================================================
q = f"""
WITH j AS (
  SELECT o.pitchDeg AS o_pitch, o.rollDeg AS o_roll,
         degrees(asin(least(1.0, greatest(-1.0, 2*(f.w*f.y - f.x*f.z))))) AS f_pitch,
         degrees(atan2(2*(f.w*f.x + f.y*f.z), f.w*f.w - f.x*f.x - f.y*f.y + f.z*f.z)) AS f_roll
  FROM read_parquet('{T}/OrientationDeg.parquet') o
  JOIN read_parquet('{T}/FOP.parquet') f USING(session, elapsedRealtimeNanos)
  WHERE random() < 0.05
)
SELECT count(*) AS n,
  round(avg(abs(o_pitch - f_pitch)), 2) AS mean_abs_pitch_diff,
  round(avg(abs(o_roll - f_roll)), 2) AS mean_abs_roll_diff,
  round(approx_quantile(abs(o_pitch - f_pitch), 0.5), 2) AS med_pitch_diff,
  round(approx_quantile(abs(o_roll - f_roll), 0.5), 2) AS med_roll_diff
FROM j
"""
r = con.execute(q).fetchone()
res["fop_vs_ori_pitchroll"] = list(r)
print("[GQ13 FOP vs OrientationDeg pitch/roll]", r)

# =====================================================================
# GQ14: Calendar day grouping for gnss_log sessions
# =====================================================================
# Session names encode dates: gnss_log_2026_10_19_...
# Extract calendar day from session name prefix
print("[GQ14 calendar day grouping] from sessions_meta.csv analysis")
import csv

meta_path = "/home/tempest/Documents/ISRO_Project/docs/EDA/scratch/sessions_meta.csv"
with open(meta_path) as f:
    reader = csv.DictReader(f)
    meta = list(reader)

from collections import defaultdict

day_sessions = defaultdict(list)
named_sessions = []
gnss_sessions = []
for row in meta:
    s = row["session"]
    if s.startswith("gnss_log_"):
        parts = s.split("_")
        day = f"{parts[2]}-{parts[3]}-{parts[4]}"
        day_sessions[day].append(s)
        gnss_sessions.append(s)
    else:
        named_sessions.append(s)

print(f"  Named sessions (orientation): {len(named_sessions)}")
print(f"  gnss_log sessions: {len(gnss_sessions)}")
print("  Calendar day breakdown:")
for day in sorted(day_sessions):
    print(f"    {day}: {len(day_sessions[day])} sessions")

res["calendar_day_breakdown"] = {d: v for d, v in day_sessions.items()}
res["named_sessions"] = named_sessions
res["gnss_sessions_count"] = len(gnss_sessions)

# =====================================================================
# GQ15: Named session tmin/tmax (from OrientationDeg_sessions.csv) to get timestamps
# =====================================================================
# Already in the CSVs loaded above. Let me parse for doc
print("\n[GQ15 named session time ranges]")
ori_csv = "/home/tempest/Documents/ISRO_Project/docs/EDA/scratch/profiles/OrientationDeg_sessions.csv"
with open(ori_csv) as f:
    reader = csv.DictReader(f)
    ori_sessions = {r["session"]: r for r in reader}

for sess in sorted(named_sessions):
    if sess in ori_sessions:
        row = ori_sessions[sess]
        tmin_ms = int(row["tmin"])
        tmax_ms = int(row["tmax"])
        dur_s = float(row["dur_s"])
        n_rows = row["n_rows"]
        import datetime

        tmin_dt = datetime.datetime.fromtimestamp(
            tmin_ms / 1000, tz=datetime.timezone.utc
        )
        tmax_dt = datetime.datetime.fromtimestamp(
            tmax_ms / 1000, tz=datetime.timezone.utc
        )
        print(
            f"  {sess}: {tmin_dt.strftime('%Y-%m-%d %H:%M')} → {tmax_dt.strftime('%H:%M')} UTC ({dur_s / 60:.1f} min), {n_rows} rows"
        )

# Save results
json.dump(res, open(os.path.join(PROF, "phaseG_extra_results.json"), "w"), indent=2)
print("\nDone. Results saved to phaseG_extra_results.json")

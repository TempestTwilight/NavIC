-- 008_cn0_zero_rows.sql — what are the C/N0 = 0 rows?
-- (a) state distribution of Cn0=0 rows: dominated by State=32995
--     (GLO_TOD_KNOWN|GLO_TOD_DECODED|GLO_STRING_SYNC|BIT_SYNC|CODE_LOCK) = GLONASS fully tracked
SELECT State, count(*) FROM read_parquet('docs/EDA/scratch/typed/Raw.parquet')
WHERE Cn0DbHz = 0 GROUP BY 1 ORDER BY 2 DESC LIMIT 10;

-- (b) tracking quality: median ReceivedSvTimeUncertaintyNanos 74 ns for Cn0=0
--     vs 30 ns for Cn0>0 → receiver IS tracking, C/N0 is simply not reported.
SELECT approx_quantile(ReceivedSvTimeUncertaintyNanos, 0.5) FROM
  read_parquet('docs/EDA/scratch/typed/Raw.parquet') WHERE Cn0DbHz = 0;
SELECT approx_quantile(ReceivedSvTimeUncertaintyNanos, 0.5) FROM
  read_parquet('docs/EDA/scratch/typed/Raw.parquet') WHERE Cn0DbHz > 0;

-- (c) by constellation: 35.5% of ALL GLONASS Raw rows have Cn0=0
SELECT ConstellationType, count(*), count(*)*100.0/
  (SELECT count(*) FROM read_parquet('docs/EDA/scratch/typed/Raw.parquet') x
   WHERE x.ConstellationType = t.ConstellationType) AS pct
FROM read_parquet('docs/EDA/scratch/typed/Raw.parquet') t
WHERE Cn0DbHz = 0 GROUP BY 1 ORDER BY 2 DESC;

-- (d) device dependence: Cn0=0 GLONASS exists ONLY on MTK devices in Raw
--     (OnePlus CPH2401 38 sessions, OPPO CPH2325 6 sessions);
--     Samsung SM-M146B (Exynos) reports GLONASS C/N0 normally.
--     In Status, Cn0=0 GLONASS also appears on OnePlus CPH2487 (Qualcomm).

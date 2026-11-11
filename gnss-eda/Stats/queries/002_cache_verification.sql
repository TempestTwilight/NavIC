-- 002_cache_verification.sql — label counts from stage-1 cache must equal the
-- independent awk survey (STEP-004). Run on stage-1 parquet glob.
SELECT label, count(*) AS rows, count(DISTINCT session) AS files
FROM read_parquet('docs/EDA/scratch/lines/*.parquet')
GROUP BY 1 ORDER BY rows DESC;
-- Result (rows/files): UncalGyro 16661992/89, OrientationDeg 11550170/89,
-- GameRotationVector 11550067/89, UncalAccel 8365329/60, FOP 7337813/75,
-- Status 6956811/95, UncalMag 6723621/89, Accel 6369268/29, Raw 5482734/95,
-- Agc 1555526/29, Nav 783399/67, Fix 189986/95 — all identical to awk survey.

-- Q-002 RESULT [2026-11-11]: 87,512 distinct (session, utcTimeMillis) epochs in Raw.
-- Duplicate test 1: GROUP BY (session, utcTimeMillis, ConstellationType, Svid,
--   CarrierFrequencyHz) HAVING count(*)>1  → 0 groups.
-- Duplicate test 2 (same + CodeType)                  → 0 groups.
-- CONCLUSION [OBSERVED]: signal identity is unique per epoch WITHOUT CodeType;
--   CodeType is metadata, not a disambiguator, in this corpus.

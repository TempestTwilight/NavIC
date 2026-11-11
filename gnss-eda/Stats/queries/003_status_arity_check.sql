-- 003_status_arity_check.sql — check declared vs actual field counts per label
-- on the lines cache (guard against hidden extra/missing commas).
SELECT label,
       min(nf) AS min_fields, max(nf) AS max_fields,
       count(*) AS rows,
       count(*) FILTER (nf = 14) AS rows_14_fields,
       count(*) FILTER (nf <> 14) AS rows_other_arity
FROM (
    SELECT label, array_length(str_split(line, ','), 1) AS nf
    FROM read_parquet('docs/EDA/scratch/lines/*.parquet')
    WHERE label IN ('Status')
) GROUP BY 1;
-- Repeat pattern used per label with its declared arity:
-- Raw=54, Status=14, Fix=18, Accel=7, UncalAccel=10, UncalGyro=10,
-- UncalMag=10, OrientationDeg=7, FOP=10, GameRotationVector=7, Agc=14,
-- Nav=variable (expected > 6).

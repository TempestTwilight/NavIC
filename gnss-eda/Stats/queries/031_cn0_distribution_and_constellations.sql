-- 031_cn0_distribution_and_constellations.sql
-- Computes overall C/N0 distribution, by ConstellationType, by carrier frequency band,
-- C/N0 vs BasebandCn0DbHz difference, and by CodeType from Raw.parquet.
-- Memory safe: SET memory_limit='1GB'; SET threads=2;

SET memory_limit='1GB';
SET threads=2;
SET preserve_insertion_order=false;

-- 1. Overall C/N0 distribution (excluding Cn0DbHz = 0)
SELECT
  COUNT(*) AS n,
  MIN(Cn0DbHz) AS min_cn0,
  PERCENTILE_CONT(0.01) WITHIN GROUP (ORDER BY Cn0DbHz) AS p1,
  PERCENTILE_CONT(0.05) WITHIN GROUP (ORDER BY Cn0DbHz) AS p5,
  PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY Cn0DbHz) AS p25,
  PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY Cn0DbHz) AS p50,
  PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY Cn0DbHz) AS p75,
  PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY Cn0DbHz) AS p95,
  PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY Cn0DbHz) AS p99,
  MAX(Cn0DbHz) AS max_cn0,
  AVG(Cn0DbHz) AS mean_cn0,
  STDDEV(Cn0DbHz) AS std_cn0
FROM read_parquet('/home/tempest/Documents/ISRO_Project/docs/EDA/scratch/typed/Raw.parquet')
WHERE Cn0DbHz > 0;

-- 2. C/N0 distribution by ConstellationType (1=GPS, 2=SBAS, 3=GLONASS, 4=QZSS, 5=BeiDou, 6=Galileo, 7=NavIC)
SELECT
  ConstellationType,
  COUNT(*) AS n,
  MIN(Cn0DbHz) AS min_cn0,
  PERCENTILE_CONT(0.05) WITHIN GROUP (ORDER BY Cn0DbHz) AS p5,
  PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY Cn0DbHz) AS p25,
  PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY Cn0DbHz) AS p50,
  PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY Cn0DbHz) AS p75,
  PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY Cn0DbHz) AS p95,
  MAX(Cn0DbHz) AS max_cn0,
  AVG(Cn0DbHz) AS mean_cn0,
  STDDEV(Cn0DbHz) AS std_cn0
FROM read_parquet('/home/tempest/Documents/ISRO_Project/docs/EDA/scratch/typed/Raw.parquet')
WHERE Cn0DbHz > 0
GROUP BY ConstellationType
ORDER BY ConstellationType;

-- 3. C/N0 distribution by carrier frequency band
SELECT
  CASE
    WHEN CarrierFrequencyHz BETWEEN 1559000000 AND 1591000000 THEN 'L1/E1/B1 (~1575 MHz)'
    WHEN CarrierFrequencyHz BETWEEN 1164000000 AND 1190000000 THEN 'L5/E5a/B2a (~1176 MHz)'
    WHEN CarrierFrequencyHz BETWEEN 1593000000 AND 1610000000 THEN 'GLONASS L1 (~1600 MHz)'
    WHEN CarrierFrequencyHz BETWEEN 1240000000 AND 1256000000 THEN 'GLONASS L2 (~1248 MHz)'
    WHEN CarrierFrequencyHz BETWEEN 1269000000 AND 1280000000 THEN 'E5b/B2b/L2 (~1278 MHz)'
    WHEN CarrierFrequencyHz BETWEEN 1191000000 AND 1207000000 THEN 'E5/AltBOC (~1191 MHz)'
    WHEN CarrierFrequencyHz IS NULL THEN 'Unknown/NULL'
    ELSE 'Other (' || CAST(ROUND(CarrierFrequencyHz/1e6,0) AS VARCHAR) || ' MHz)'
  END AS band,
  COUNT(*) AS n,
  MIN(Cn0DbHz) AS min_cn0,
  PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY Cn0DbHz) AS p25,
  PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY Cn0DbHz) AS p50,
  PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY Cn0DbHz) AS p75,
  PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY Cn0DbHz) AS p95,
  MAX(Cn0DbHz) AS max_cn0,
  AVG(Cn0DbHz) AS mean_cn0,
  STDDEV(Cn0DbHz) AS std_cn0
FROM read_parquet('/home/tempest/Documents/ISRO_Project/docs/EDA/scratch/typed/Raw.parquet')
WHERE Cn0DbHz > 0
GROUP BY band
ORDER BY n DESC;

-- 4. Difference between Cn0DbHz and BasebandCn0DbHz by constellation
SELECT
  ConstellationType,
  COUNT(*) AS n,
  AVG(Cn0DbHz - BasebandCn0DbHz) AS mean_diff,
  STDDEV(Cn0DbHz - BasebandCn0DbHz) AS std_diff,
  MIN(Cn0DbHz - BasebandCn0DbHz) AS min_diff,
  MAX(Cn0DbHz - BasebandCn0DbHz) AS max_diff,
  PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY Cn0DbHz - BasebandCn0DbHz) AS p50_diff,
  SUM(CASE WHEN ABS(Cn0DbHz - BasebandCn0DbHz) < 0.01 THEN 1 ELSE 0 END) AS identical_count,
  SUM(CASE WHEN ABS(Cn0DbHz - BasebandCn0DbHz) >= 0.01 THEN 1 ELSE 0 END) AS differ_count
FROM read_parquet('/home/tempest/Documents/ISRO_Project/docs/EDA/scratch/typed/Raw.parquet')
WHERE Cn0DbHz > 0 AND BasebandCn0DbHz IS NOT NULL
GROUP BY ConstellationType
ORDER BY ConstellationType;

-- 5. Mean C/N0 per CodeType per ConstellationType
SELECT
  ConstellationType,
  CodeType,
  COUNT(*) AS n,
  AVG(Cn0DbHz) AS mean_cn0,
  PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY Cn0DbHz) AS p50_cn0,
  STDDEV(Cn0DbHz) AS std_cn0
FROM read_parquet('/home/tempest/Documents/ISRO_Project/docs/EDA/scratch/typed/Raw.parquet')
WHERE Cn0DbHz > 0
GROUP BY ConstellationType, CodeType
ORDER BY ConstellationType, CodeType;

-- 015_outliers_and_impossible_values.sql
-- Phase D: Outliers and physically impossible values across tables
-- Goal: Verify domain boundaries for C/N0, elevation/azimuth, ADR, Fix coordinates, and Accel gravity norm.
SET memory_limit='1GB'; SET threads=1; SET preserve_insertion_order=false;

-- 1. C/N0 = 0 counts and Q-001 cross check
SELECT 'C/N0_zero_summary' as check_name,
  (SELECT COUNT(*) FROM '/home/tempest/Documents/ISRO_Project/docs/EDA/scratch/typed/Raw.parquet' WHERE Cn0DbHz = 0) as raw_cn0_zero,
  (SELECT COUNT(*) FROM '/home/tempest/Documents/ISRO_Project/docs/EDA/scratch/typed/Raw.parquet' WHERE Cn0DbHz = 0 AND ConstellationType = 3) as raw_glonass_cn0_zero,
  (SELECT COUNT(*) FROM '/home/tempest/Documents/ISRO_Project/docs/EDA/scratch/typed/Status.parquet' WHERE Cn0DbHz = 0) as status_cn0_zero,
  (SELECT COUNT(*) FROM '/home/tempest/Documents/ISRO_Project/docs/EDA/scratch/typed/Status.parquet' WHERE Cn0DbHz = 0 AND ConstellationType = 3) as status_glonass_cn0_zero;

-- 2. Status elevation and azimuth boundaries
SELECT 'Status_geometry_bounds' as check_name,
  COUNT_IF(ElevationDegrees < 0) as neg_elevation_rows,
  COUNT_IF(ElevationDegrees > 90) as gt90_elevation_rows,
  MIN(ElevationDegrees) as min_elevation,
  MAX(ElevationDegrees) as max_elevation,
  COUNT_IF(AzimuthDegrees < 0 OR AzimuthDegrees > 360) as bad_azimuth_rows,
  MIN(AzimuthDegrees) as min_azimuth,
  MAX(AzimuthDegrees) as max_azimuth
FROM '/home/tempest/Documents/ISRO_Project/docs/EDA/scratch/typed/Status.parquet';

-- 3. Raw C/N0 extreme values (> 60 dB-Hz)
SELECT 'Raw_high_cn0' as check_name,
  COUNT_IF(Cn0DbHz > 60) as cn0_gt_60,
  MAX(Cn0DbHz) as max_cn0,
  COUNT_IF(BasebandCn0DbHz > 60) as bb_cn0_gt_60,
  MAX(BasebandCn0DbHz) as max_bb_cn0
FROM '/home/tempest/Documents/ISRO_Project/docs/EDA/scratch/typed/Raw.parquet';

-- 4. Accumulated Delta Range (ADR) extremes
SELECT 'ADR_extremes' as check_name,
  MIN(AccumulatedDeltaRangeMeters) as min_adr,
  MAX(AccumulatedDeltaRangeMeters) as max_adr,
  COUNT_IF(abs(AccumulatedDeltaRangeMeters) > 1e9) as adr_gt_1e9
FROM '/home/tempest/Documents/ISRO_Project/docs/EDA/scratch/typed/Raw.parquet';

-- 5. Fix coordinates and speed validity
SELECT 'Fix_validity' as check_name,
  COUNT_IF(LatitudeDegrees < -90 OR LatitudeDegrees > 90) as invalid_lat,
  MIN(LatitudeDegrees) as min_lat, MAX(LatitudeDegrees) as max_lat,
  COUNT_IF(LongitudeDegrees < -180 OR LongitudeDegrees > 180) as invalid_lon,
  MIN(LongitudeDegrees) as min_lon, MAX(LongitudeDegrees) as max_lon,
  COUNT_IF(SpeedMps < 0) as negative_speed,
  MIN(SpeedMps) as min_speed, MAX(SpeedMps) as max_speed
FROM '/home/tempest/Documents/ISRO_Project/docs/EDA/scratch/typed/Fix.parquet';

-- 6. Accelerometer gravity norm
WITH accel_norm AS (
  SELECT sqrt(AccelXMps2*AccelXMps2 + AccelYMps2*AccelYMps2 + AccelZMps2*AccelZMps2) as norm
  FROM '/home/tempest/Documents/ISRO_Project/docs/EDA/scratch/typed/Accel.parquet'
)
SELECT 'Accel_gravity_norm' as check_name,
  COUNT(*) as total_samples,
  MIN(norm) as min_norm,
  quantile_cont(norm, 0.01) as p1_norm,
  quantile_cont(norm, 0.50) as p50_norm,
  quantile_cont(norm, 0.99) as p99_norm,
  MAX(norm) as max_norm,
  AVG(norm) as mean_norm,
  COUNT_IF(norm < 7.0) as norm_lt_7,
  COUNT_IF(norm > 12.0) as norm_gt_12
FROM accel_norm;

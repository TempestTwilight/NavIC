-- 013_duplicate_checks_raw_status.sql
-- Phase D: Duplicate signal checks on Raw and Status
-- Goal: Verify if (epoch, constellation, svid, carrier, [code]) uniquely identifies signals.
SET memory_limit='1GB'; SET threads=1; SET preserve_insertion_order=false;

-- 1. Raw table: duplicate check with CodeType
WITH raw_dup_with_code AS (
  SELECT session, utcTimeMillis, ConstellationType, Svid, CarrierFrequencyHz, CodeType, COUNT(*) as cnt
  FROM '/home/tempest/Documents/ISRO_Project/docs/EDA/scratch/typed/Raw.parquet'
  GROUP BY session, utcTimeMillis, ConstellationType, Svid, CarrierFrequencyHz, CodeType
  HAVING COUNT(*) > 1
)
SELECT 'Raw_with_CodeType' as check_type, COUNT(*) as duplicate_groups, COALESCE(SUM(cnt), 0) as total_duplicate_rows
FROM raw_dup_with_code;

-- 2. Raw table: duplicate check WITHOUT CodeType (same carrier, multiple codes)
WITH raw_dup_no_code AS (
  SELECT session, utcTimeMillis, ConstellationType, Svid, CarrierFrequencyHz, COUNT(*) as cnt, COUNT(DISTINCT CodeType) as num_codes
  FROM '/home/tempest/Documents/ISRO_Project/docs/EDA/scratch/typed/Raw.parquet'
  GROUP BY session, utcTimeMillis, ConstellationType, Svid, CarrierFrequencyHz
  HAVING COUNT(*) > 1
)
SELECT 'Raw_without_CodeType' as check_type, COUNT(*) as multi_signal_on_carrier, COALESCE(SUM(cnt), 0) as total_rows
FROM raw_dup_no_code;

-- 3. Status table: duplicate check on (epoch, constellation, svid, carrier)
WITH status_dups AS (
  SELECT session, UnixTimeMillis, ConstellationType, Svid, CarrierFrequencyHz, COUNT(*) as cnt
  FROM '/home/tempest/Documents/ISRO_Project/docs/EDA/scratch/typed/Status.parquet'
  WHERE UnixTimeMillis IS NOT NULL
  GROUP BY session, UnixTimeMillis, ConstellationType, Svid, CarrierFrequencyHz
  HAVING COUNT(*) > 1
)
SELECT 'Status_duplicates' as check_type, COUNT(*) as dup_signal_keys, COALESCE(SUM(cnt), 0) as total_rows, COUNT(DISTINCT (session || '_' || CAST(UnixTimeMillis AS VARCHAR))) as affected_epochs
FROM status_dups;

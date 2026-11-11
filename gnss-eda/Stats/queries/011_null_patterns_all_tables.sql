-- 011_null_patterns_all_tables.sql
-- Goal: Per-column null count and % for all 12 tables
SET memory_limit='1GB'; SET threads=2; SET preserve_insertion_order=false;
-- Raw nulls by column (aggregate over full dataset)
SELECT 'Raw' AS tbl,
  COUNT(*) AS total_rows,
  COUNT_IF(utcTimeMillis IS NULL) AS null_utcTimeMillis,
  COUNT_IF(TimeNanos IS NULL) AS null_TimeNanos,
  COUNT_IF(LeapSecond IS NULL) AS null_LeapSecond,
  COUNT_IF(Cn0DbHz IS NULL) AS null_Cn0DbHz,
  COUNT_IF(BasebandCn0DbHz IS NULL) AS null_BasebandCn0DbHz,
  COUNT_IF(CarrierFrequencyHz IS NULL) AS null_CarrierFrequencyHz,
  COUNT_IF(CarrierCycles IS NULL) AS null_CarrierCycles,
  COUNT_IF(CarrierPhase IS NULL) AS null_CarrierPhase,
  COUNT_IF(CarrierPhaseUncertainty IS NULL) AS null_CarrierPhaseUncertainty,
  COUNT_IF(MultipathIndicator IS NULL) AS null_MultipathIndicator,
  COUNT_IF(SnrInDb IS NULL) AS null_SnrInDb,
  COUNT_IF(AgcDb IS NULL) AS null_AgcDb,
  COUNT_IF(FullInterSignalBiasNanos IS NULL) AS null_FullISBNanos,
  COUNT_IF(SatelliteInterSignalBiasNanos IS NULL) AS null_SatISBNanos,
  COUNT_IF(CodeType IS NULL) AS null_CodeType,
  COUNT_IF(ChipsetElapsedRealtimeNanos IS NULL) AS null_ChipsetElapsed,
  COUNT_IF(IsFullTracking IS NULL) AS null_IsFullTracking,
  COUNT_IF(SvPositionEcefXMeters IS NULL) AS null_SvPosX,
  COUNT_IF(KlobucharAlpha0 IS NULL) AS null_Klobuchar0
FROM '/home/tempest/Documents/ISRO_Project/docs/EDA/scratch/typed/Raw.parquet';

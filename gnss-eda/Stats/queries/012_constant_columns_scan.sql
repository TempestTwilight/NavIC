-- 012_constant_columns_scan.sql
-- Phase D: Constant and near-constant columns across Raw, Status, Fix, Sensor tables
-- Goal: Identify columns with distinct count <= 2 or stddev ≈ 0 across the entire corpus.
SET memory_limit='1GB'; SET threads=1; SET preserve_insertion_order=false;

-- Raw near-constant scan
SELECT 'Raw' as table_name,
  count(distinct LeapSecond) as dist_LeapSecond,
  count(distinct CarrierCycles) as dist_CarrierCycles,
  count(distinct CarrierPhase) as dist_CarrierPhase,
  count(distinct CarrierPhaseUncertainty) as dist_CarrierPhaseUncertainty,
  count(distinct MultipathIndicator) as dist_MultipathIndicator,
  count(distinct IsFullTracking) as dist_IsFullTracking,
  count(distinct TimeOffsetNanos) as dist_TimeOffsetNanos
FROM '/home/tempest/Documents/ISRO_Project/docs/EDA/scratch/typed/Raw.parquet';

-- Status flags scan
SELECT 'Status' as table_name,
  count(distinct UsedInFix) as dist_UsedInFix,
  count(distinct HasAlmanacData) as dist_HasAlmanacData,
  count(distinct HasEphemerisData) as dist_HasEphemerisData
FROM '/home/tempest/Documents/ISRO_Project/docs/EDA/scratch/typed/Status.parquet';

-- Fix near-constant scan
SELECT 'Fix' as table_name,
  count(distinct MockLocation) as dist_MockLocation,
  count(distinct NumberOfUsedSignals) as dist_NumberOfUsedSignals,
  count(distinct VerticalSpeedAccuracyMps) as dist_VerticalSpeedAccuracyMps,
  count(distinct SolutionType) as dist_SolutionType
FROM '/home/tempest/Documents/ISRO_Project/docs/EDA/scratch/typed/Fix.parquet';

-- 005_adr_state_bitdecode.sql — AccumulatedDeltaRangeState bitmask frequencies
-- Bit values VERIFIED against AOSP GnssMeasurement.java (fetched 2026-11-11):
-- ADR_STATE_UNKNOWN=0, VALID=1<<0, RESET=1<<1, CYCLE_SLIP=1<<2,
-- HALF_CYCLE_RESOLVED=1<<3, HALF_CYCLE_REPORTED=1<<4.
SELECT AccumulatedDeltaRangeState AS adr_state, count(*) AS rows
FROM read_parquet('docs/EDA/scratch/typed/Raw.parquet')
GROUP BY 1 ORDER BY rows DESC LIMIT 30;

SELECT count(*) AS total,
       count(*) FILTER (bit_and(AccumulatedDeltaRangeState, 1) <> 0) AS b0_valid,
       count(*) FILTER (bit_and(AccumulatedDeltaRangeState, 2) <> 0) AS b1_reset,
       count(*) FILTER (bit_and(AccumulatedDeltaRangeState, 4) <> 0) AS b2_cycle_slip,
       count(*) FILTER (bit_and(AccumulatedDeltaRangeState, 8) <> 0) AS b3_half_cycle_resolved,
       count(*) FILTER (bit_and(AccumulatedDeltaRangeState,16) <> 0) AS b4_half_cycle_reported,
       count(*) FILTER (AccumulatedDeltaRangeState > 31) AS b5plus_unexpected
FROM read_parquet('docs/EDA/scratch/typed/Raw.parquet');

-- RESULTS [2026-11-11, 5,482,734 rows]:
-- Per-bit: HALF_CYCLE_REPORTED (1<<4) set on 100.00% of rows (every row!);
--   VALID (1<<0) 39.42%; HALF_CYCLE_RESOLVED (1<<3) 32.94%; RESET 0.78%;
--   CYCLE_SLIP 0.38%; no value > 31; ADR state 0 never occurs (min = 16).
-- Interpretation: the vendor always flags 'half cycle reported'; ADR is usable
--   (VALID) on only ~39% of rows, matching AccumulatedDeltaRangeMeters = 0 on
--   ~55% of rows (invalid ADR reported as 0).

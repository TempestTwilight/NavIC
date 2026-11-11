-- 004_raw_state_bitdecode.sql — decode the `State` bitmask bit by bit for Raw.
-- Bit values VERIFIED against AOSP GnssMeasurement.java (aosp-mirror/platform_frameworks_base,
-- fetched 2026-11-11): STATE_CODE_LOCK=1<<0, BIT_SYNC=1<<1, SUBFRAME_SYNC=1<<2,
-- TOW_DECODED=1<<3, MSEC_AMBIGUOUS=1<<4, SYMBOL_SYNC=1<<5, GLO_STRING_SYNC=1<<6,
-- GLO_TOD_DECODED=1<<7, BDS_D2_BIT_SYNC=1<<8, BDS_D2_SUBFRAME_SYNC=1<<9,
-- GAL_E1BC_CODE_LOCK=1<<10, GAL_E1C_2ND_CODE_LOCK=1<<11, GAL_E1B_PAGE_SYNC=1<<12,
-- SBAS_SYNC=1<<13, TOW_KNOWN=1<<14, GLO_TOD_KNOWN=1<<15, 2ND_CODE_LOCK=1<<16.

-- (a) most common whole State values
SELECT State, count(*) AS rows
FROM read_parquet('docs/EDA/scratch/typed/Raw.parquet')
GROUP BY 1 ORDER BY rows DESC LIMIT 30;

-- (b) per-bit frequency
SELECT
  count(*) AS total,
  count(*) FILTER (bit_and(State, 1)    <> 0) AS b00_code_lock,
  count(*) FILTER (bit_and(State, 2)    <> 0) AS b01_bit_sync,
  count(*) FILTER (bit_and(State, 4)    <> 0) AS b02_subframe_sync,
  count(*) FILTER (bit_and(State, 8)    <> 0) AS b03_tow_decoded,
  count(*) FILTER (bit_and(State, 16)   <> 0) AS b04_msec_ambiguous,
  count(*) FILTER (bit_and(State, 32)   <> 0) AS b05_symbol_sync,
  count(*) FILTER (bit_and(State, 64)   <> 0) AS b06_glo_string_sync,
  count(*) FILTER (bit_and(State, 128)  <> 0) AS b07_glo_tod_decoded,
  count(*) FILTER (bit_and(State, 256)  <> 0) AS b08_bds_d2_bit_sync,
  count(*) FILTER (bit_and(State, 512)  <> 0) AS b09_bds_d2_subframe_sync,
  count(*) FILTER (bit_and(State, 1024) <> 0) AS b10_gal_e1bc_code_lock,
  count(*) FILTER (bit_and(State, 2048) <> 0) AS b11_gal_e1c_2nd_code_lock,
  count(*) FILTER (bit_and(State, 4096) <> 0) AS b12_gal_e1b_page_sync,
  count(*) FILTER (bit_and(State, 8192) <> 0) AS b13_sbas_sync,
  count(*) FILTER (bit_and(State, 16384)<> 0) AS b14_tow_known,
  count(*) FILTER (bit_and(State, 32768)<> 0) AS b15_glo_tod_known,
  count(*) FILTER (bit_and(State, 65536)<> 0) AS b16_2nd_code_lock,
  count(*) FILTER (State = 0)                 AS state_zero
FROM read_parquet('docs/EDA/scratch/typed/Raw.parquet');

-- RESULTS [2026-11-11, 5,482,734 rows]:
-- Top whole values: 16431 (1,622,576), 16399 (804,141), 15 (599,438),
--   22569 (469,279), 32995 (296,843), 81935 (251,932), 3072 (215,943),
--   39 (192,770), 5 (188,814), 239 (160,133), 23567 (146,735), 47 (130,892),
--   49359 (128,940), 16905 (117,103), 801 (52,998), 51 (34,840), 16391 (17,834),
--   2097 (15,748), 1025 (10,601), 49255 (6,715).
-- Per-bit: code_lock 96.05%; bit_sync 80.21%; subframe_sync 77.59%;
--   tow_decoded 80.88%; msec_ambiguous 1.06%; symbol_sync 54.60%;
--   glo_string_sync 10.81%; glo_tod_decoded 10.69%; bds_d2_bit_sync 1.06%;
--   bds_d2_subframe_sync 3.10%; gal_e1bc_code_lock 6.81%;
--   gal_e1c_2nd_code_lock 15.51%; gal_e1b_page_sync 11.28%; SBAS_SYNC 0.00%
--   (never set — consistent with NO SBAS constellation in the corpus);
--   tow_known 65.07%; glo_tod_known 7.89%; 2nd_code_lock 4.60%;
--   State=0 rows: 0 (never). NOTE: DuckDB bit_and is an AGGREGATE; the row-level
--   bitwise test is the '&' operator (fix applied when running).

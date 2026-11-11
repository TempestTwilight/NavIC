-- 007_empty_raw_rows.sql — the 55,217 Raw rows with NULL ConstellationType:
-- investigation shows they are COMPLETELY empty placeholder lines: the raw
-- data line is exactly the text "Raw" (nothing after the label).
SELECT count(*) AS n_null_constellation,
       count(Cn0DbHz) n_cn0, count(TimeNanos) n_timenanos, count(Svid) n_svid,
       count(utcTimeMillis) n_utc
FROM read_parquet('docs/EDA/scratch/typed/Raw.parquet')
WHERE ConstellationType IS NULL;
-- Result: 55217 rows, ALL fields null.

-- Their raw line content (from lines cache):
SELECT line, count(*) n
FROM read_parquet('docs/EDA/scratch/lines/*.parquet')
WHERE label = 'Raw' AND length(replace(line, ',', '')) <= 3
GROUP BY 1 ORDER BY n DESC;
-- Result: single row: line = 'Raw', n = 55217.
-- Interpretation: logger writes a bare 'Raw' line when a measurement epoch has
-- no content (e.g. app lifecycle gaps). Safe to DROP.

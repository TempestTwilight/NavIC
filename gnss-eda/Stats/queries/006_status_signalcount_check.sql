-- 006_status_signalcount_check.sql — how plausible are per-epoch row counts?
-- Grain: one Status row per signal per epoch. Typical phones see 20-70 signals.
SELECT SignalCount, count(*) AS rows, count(DISTINCT session) AS sessions
FROM read_parquet('docs/EDA/scratch/typed/Status.parquet')
GROUP BY 1 ORDER BY 1 LIMIT 50;

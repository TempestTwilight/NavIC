
-- Compute session-level median residual C/N0 for the 14 orientation sessions vs gnss_log sessions
SELECT 
    r.session,
    CASE 
        WHEN r.session LIKE '%facing%' OR r.session LIKE '%horizontal%' OR r.session LIKE '%vertical%' THEN 'Orientation Rig Session'
        ELSE 'Natural Campaign Session'
    END as session_group,
    ROUND(MEDIAN(r.Cn0DbHz), 2) as median_cn0,
    ROUND(QUANTILE_CONT(r.Cn0DbHz, 0.75) - QUANTILE_CONT(r.Cn0DbHz, 0.25), 2) as iqr_cn0,
    COUNT(*) as n_signals
FROM read_parquet('/home/tempest/Documents/ISRO_Project/gnss-eda/Stats/scratch/typed/Raw.parquet') r
JOIN (
    SELECT session FROM read_csv_auto('/home/tempest/Documents/ISRO_Project/gnss-eda/Stats/scratch/sessions_meta.csv') WHERE model = 'CPH2401'
) s ON r.session = s.session
WHERE r.Cn0DbHz > 0
GROUP BY r.session, session_group
ORDER BY session_group, median_cn0 DESC;

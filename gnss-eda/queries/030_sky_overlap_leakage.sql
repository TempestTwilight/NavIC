-- Compute satellite constellation overlap between pairs of sessions by time gap
WITH session_time_sats AS (
    SELECT 
        session,
        MIN(UnixTimeMillis) as min_time,
        LIST(DISTINCT Svid) as sat_list
    FROM read_parquet('/home/tempest/Documents/ISRO_Project/gnss-eda/Stats/scratch/typed/Status.parquet')
    WHERE session LIKE 'gnss_log_%' AND ConstellationType = 1 -- GPS
    GROUP BY session
),
session_pairs AS (
    SELECT 
        s1.session as sess1,
        s2.session as sess2,
        ABS(s2.min_time - s1.min_time) / (1000.0 * 3600.0) as time_gap_hours,
        len(list_intersect(s1.sat_list, s2.sat_list)) * 1.0 / len(list_distinct(list_concat(s1.sat_list, s2.sat_list))) as jaccard_overlap
    FROM session_time_sats s1
    JOIN session_time_sats s2 ON s1.session < s2.session
    WHERE ABS(s2.min_time - s1.min_time) / (1000.0 * 3600.0) <= 96.0
)
SELECT 
    CASE 
        WHEN time_gap_hours < 2.0 THEN '< 2 Hours (Intra-Day)'
        WHEN time_gap_hours < 12.0 THEN '2 - 12 Hours'
        WHEN time_gap_hours < 26.0 THEN '12 - 26 Hours (Adjacent Day)'
        ELSE '26 - 96 Hours (Multi-Day)'
    END as time_gap_bin,
    COUNT(*) as n_pairs,
    ROUND(AVG(jaccard_overlap) * 100.0, 1) as mean_overlap_pct,
    ROUND(MEDIAN(jaccard_overlap) * 100.0, 1) as median_overlap_pct,
    ROUND(QUANTILE_CONT(jaccard_overlap, 0.25) * 100.0, 1) as p25_overlap,
    ROUND(QUANTILE_CONT(jaccard_overlap, 0.75) * 100.0, 1) as p75_overlap
FROM session_pairs
GROUP BY time_gap_bin
ORDER BY MIN(time_gap_hours);

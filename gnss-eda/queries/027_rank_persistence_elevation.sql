-- Compute rank persistence by elevation band of the current top-1 satellite
WITH base_signals AS (
    SELECT 
        r.session,
        r.ConstellationType,
        r.Svid,
        r.CarrierFrequencyHz,
        r.CodeType,
        CAST(ROUND(r.utcTimeMillis / 1000.0) AS BIGINT) as sec_epoch,
        r.Cn0DbHz as c_t,
        st.ElevationDegrees as el
    FROM read_parquet('/home/tempest/Documents/ISRO_Project/gnss-eda/Stats/scratch/typed/Raw.parquet') r
    JOIN (
        SELECT DISTINCT CAST(ROUND(UnixTimeMillis / 1000.0) AS BIGINT) as sec_epoch,
               ConstellationType, Svid, CarrierFrequencyHz, ElevationDegrees
        FROM read_parquet('/home/tempest/Documents/ISRO_Project/gnss-eda/Stats/scratch/typed/Status.parquet')
        WHERE session = 'gnss_log_2026_10_22_08_30_05'
    ) st ON CAST(ROUND(r.utcTimeMillis / 1000.0) AS BIGINT) = st.sec_epoch
       AND r.ConstellationType = st.ConstellationType
       AND r.CarrierFrequencyHz = st.CarrierFrequencyHz
    WHERE r.session = 'gnss_log_2026_10_22_08_30_05'
      AND r.Cn0DbHz > 0
),
lagged AS (
    SELECT 
        t1.sec_epoch,
        t1.ConstellationType,
        t1.Svid,
        t1.c_t,
        t1.el,
        COALESCE(t2.c_t, 12.0) as c_future
    FROM base_signals t1
    LEFT JOIN base_signals t2
      ON t1.ConstellationType = t2.ConstellationType
     AND t1.Svid = t2.Svid
     AND t1.CarrierFrequencyHz = t2.CarrierFrequencyHz
     AND t1.CodeType = t2.CodeType
     AND t2.sec_epoch = t1.sec_epoch + 15
),
ranked_epochs AS (
    SELECT 
        sec_epoch,
        ConstellationType,
        Svid,
        el,
        c_t,
        c_future,
        ROW_NUMBER() OVER (PARTITION BY sec_epoch ORDER BY c_t DESC) as current_rank,
        ROW_NUMBER() OVER (PARTITION BY sec_epoch ORDER BY c_future DESC) as future_rank
    FROM lagged
)
SELECT 
    CASE 
        WHEN el < 30 THEN 'Low (<30 deg)'
        WHEN el < 60 THEN 'Mid (30-60 deg)'
        ELSE 'High (>=60 deg)'
    END as top_satellite_elevation,
    COUNT(*) as n_epochs,
    SUM(CASE WHEN future_rank = 1 THEN 1 ELSE 0 END) as retained_top1_count,
    ROUND(AVG(CASE WHEN future_rank = 1 THEN 1.0 ELSE 0.0 END) * 100.0, 1) as rank_persistence_top1_pct,
    ROUND(AVG(CASE WHEN future_rank <= 3 THEN 1.0 ELSE 0.0 END) * 100.0, 1) as retained_top3_pct
FROM ranked_epochs
WHERE current_rank = 1
GROUP BY top_satellite_elevation
ORDER BY top_satellite_elevation;

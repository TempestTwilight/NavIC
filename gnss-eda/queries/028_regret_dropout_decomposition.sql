-- Decompose selection regret into dropout loss vs continuous track drift
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
        CASE WHEN t2.c_t IS NOT NULL THEN 0 ELSE 1 END as is_dropped,
        COALESCE(t2.c_t, 12.0) as c_future
    FROM base_signals t1
    LEFT JOIN base_signals t2
      ON t1.ConstellationType = t2.ConstellationType
     AND t1.Svid = t2.Svid
     AND t1.CarrierFrequencyHz = t2.CarrierFrequencyHz
     AND t1.CodeType = t2.CodeType
     AND t2.sec_epoch = t1.sec_epoch + 15
),
ranked AS (
    SELECT 
        sec_epoch,
        c_future,
        is_dropped,
        MAX(c_future) OVER (PARTITION BY sec_epoch) as oracle_best_future,
        ROW_NUMBER() OVER (PARTITION BY sec_epoch ORDER BY c_t DESC) as rank_pers
    FROM lagged
)
SELECT 
    CASE WHEN is_dropped = 1 THEN 'Signal Dropped (Lost Track)' ELSE 'Continuously Tracked' END as track_status,
    COUNT(*) as n_epochs,
    ROUND(AVG(oracle_best_future - c_future), 2) as mean_regret_db,
    ROUND(SUM(oracle_best_future - c_future), 1) as total_regret_loss_db
FROM ranked
WHERE rank_pers = 1
GROUP BY is_dropped;

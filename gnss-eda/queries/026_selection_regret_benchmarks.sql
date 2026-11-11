-- Compute selection regret across candidate heuristic selectors
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
        t1.CarrierFrequencyHz,
        t1.CodeType,
        t1.c_t,
        t1.el,
        COALESCE(t2.c_t, 12.0) as c_future -- Penalize lost tracks with 12 dB sensitivity floor
    FROM base_signals t1
    LEFT JOIN base_signals t2
      ON t1.ConstellationType = t2.ConstellationType
     AND t1.Svid = t2.Svid
     AND t1.CarrierFrequencyHz = t2.CarrierFrequencyHz
     AND t1.CodeType = t2.CodeType
     AND t2.sec_epoch = t1.sec_epoch + 15
),
epoch_ranks AS (
    SELECT 
        sec_epoch,
        ConstellationType,
        Svid,
        CarrierFrequencyHz,
        CodeType,
        c_t,
        el,
        c_future,
        MAX(c_future) OVER (PARTITION BY sec_epoch) as oracle_best_future,
        ROW_NUMBER() OVER (PARTITION BY sec_epoch ORDER BY c_t DESC) as rank_persistence,
        ROW_NUMBER() OVER (PARTITION BY sec_epoch ORDER BY el DESC) as rank_elevation,
        ROW_NUMBER() OVER (PARTITION BY sec_epoch ORDER BY (c_t + 0.1 * el) DESC) as rank_pers_plus_el,
        ROW_NUMBER() OVER (PARTITION BY sec_epoch ORDER BY HASH(ConstellationType, Svid, sec_epoch)) as rank_random
    FROM lagged
)
SELECT 
    'Persistence (Max Current C/N0)' as policy,
    COUNT(*) as n_evals,
    ROUND(AVG(oracle_best_future - c_future), 2) as mean_regret_db,
    ROUND(MEDIAN(oracle_best_future - c_future), 2) as median_regret_db,
    ROUND(AVG(CASE WHEN (oracle_best_future - c_future) <= 0.5 THEN 1.0 ELSE 0.0 END) * 100.0, 1) as hit_rate_05db_pct,
    ROUND(AVG(CASE WHEN (oracle_best_future - c_future) == 0.0 THEN 1.0 ELSE 0.0 END) * 100.0, 1) as top1_accuracy_pct
FROM epoch_ranks WHERE rank_persistence = 1
UNION ALL
SELECT 
    'Max Elevation' as policy,
    COUNT(*) as n_evals,
    ROUND(AVG(oracle_best_future - c_future), 2) as mean_regret_db,
    ROUND(MEDIAN(oracle_best_future - c_future), 2) as median_regret_db,
    ROUND(AVG(CASE WHEN (oracle_best_future - c_future) <= 0.5 THEN 1.0 ELSE 0.0 END) * 100.0, 1) as hit_rate_05db_pct,
    ROUND(AVG(CASE WHEN (oracle_best_future - c_future) == 0.0 THEN 1.0 ELSE 0.0 END) * 100.0, 1) as top1_accuracy_pct
FROM epoch_ranks WHERE rank_elevation = 1
UNION ALL
SELECT 
    'Persistence + Elevation Hybrid' as policy,
    COUNT(*) as n_evals,
    ROUND(AVG(oracle_best_future - c_future), 2) as mean_regret_db,
    ROUND(MEDIAN(oracle_best_future - c_future), 2) as median_regret_db,
    ROUND(AVG(CASE WHEN (oracle_best_future - c_future) <= 0.5 THEN 1.0 ELSE 0.0 END) * 100.0, 1) as hit_rate_05db_pct,
    ROUND(AVG(CASE WHEN (oracle_best_future - c_future) == 0.0 THEN 1.0 ELSE 0.0 END) * 100.0, 1) as top1_accuracy_pct
FROM epoch_ranks WHERE rank_pers_plus_el = 1
UNION ALL
SELECT 
    'Random Pick' as policy,
    COUNT(*) as n_evals,
    ROUND(AVG(oracle_best_future - c_future), 2) as mean_regret_db,
    ROUND(MEDIAN(oracle_best_future - c_future), 2) as median_regret_db,
    ROUND(AVG(CASE WHEN (oracle_best_future - c_future) <= 0.5 THEN 1.0 ELSE 0.0 END) * 100.0, 1) as hit_rate_05db_pct,
    ROUND(AVG(CASE WHEN (oracle_best_future - c_future) == 0.0 THEN 1.0 ELSE 0.0 END) * 100.0, 1) as top1_accuracy_pct
FROM epoch_ranks WHERE rank_random = 1;

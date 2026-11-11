-- Compute baseline error metrics on 15s future predictions
WITH target_signals AS (
    SELECT 
        session,
        ConstellationType,
        Svid,
        CarrierFrequencyHz,
        CodeType,
        CAST(ROUND(utcTimeMillis / 1000.0) AS BIGINT) as sec_epoch,
        Cn0DbHz as c_t
    FROM read_parquet('/home/tempest/Documents/ISRO_Project/gnss-eda/Stats/scratch/typed/Raw.parquet')
    WHERE session = 'gnss_log_2026_10_22_08_30_05' AND Cn0DbHz > 0
),
paired AS (
    SELECT 
        t1.c_t,
        t2.c_t as c_future,
        st.ElevationDegrees as el,
        -- Baselines:
        t1.c_t as pred_persistence,
        38.5 as pred_global_mean,
        20.0 + 20.0 * SIN(RADIANS(st.ElevationDegrees)) as pred_elevation,
        t1.c_t - 0.08 * (t1.c_t - (20.0 + 20.0 * SIN(RADIANS(st.ElevationDegrees)))) as pred_mean_reversion,
        ABS(t2.c_t - t1.c_t) as realized_change
    FROM target_signals t1
    JOIN target_signals t2
      ON t1.ConstellationType = t2.ConstellationType
     AND t1.Svid = t2.Svid
     AND t1.CarrierFrequencyHz = t2.CarrierFrequencyHz
     AND t1.CodeType = t2.CodeType
     AND t2.sec_epoch = t1.sec_epoch + 15
    JOIN (
        SELECT DISTINCT CAST(ROUND(UnixTimeMillis / 1000.0) AS BIGINT) as sec_epoch,
               ConstellationType, Svid, CarrierFrequencyHz, ElevationDegrees
        FROM read_parquet('/home/tempest/Documents/ISRO_Project/gnss-eda/Stats/scratch/typed/Status.parquet')
        WHERE session = 'gnss_log_2026_10_22_08_30_05'
    ) st ON t1.sec_epoch = st.sec_epoch
       AND t1.ConstellationType = st.ConstellationType
       AND t1.CarrierFrequencyHz = st.CarrierFrequencyHz
)
SELECT 
    CASE WHEN realized_change <= 1.5 THEN 'Static (|Delta| <= 1.5 dB)' ELSE 'Fading (|Delta| > 1.5 dB)' END as condition,
    COUNT(*) as n_obs,
    -- Persistence
    ROUND(SQRT(AVG(POWER(c_future - pred_persistence, 2))), 3) as rmse_persistence,
    ROUND(AVG(ABS(c_future - pred_persistence)), 3) as mae_persistence,
    -- Global Mean
    ROUND(SQRT(AVG(POWER(c_future - pred_global_mean, 2))), 3) as rmse_global_mean,
    ROUND(AVG(ABS(c_future - pred_global_mean)), 3) as mae_global_mean,
    -- Elevation Baseline
    ROUND(SQRT(AVG(POWER(c_future - pred_elevation, 2))), 3) as rmse_elevation,
    ROUND(AVG(ABS(c_future - pred_elevation)), 3) as mae_elevation,
    -- Persistence + Mean Reversion
    ROUND(SQRT(AVG(POWER(c_future - pred_mean_reversion, 2))), 3) as rmse_mean_reversion,
    ROUND(AVG(ABS(c_future - pred_mean_reversion)), 3) as mae_mean_reversion
FROM paired
GROUP BY condition;

-- Compute candidates per epoch and future headroom distribution
WITH base_signals AS (
    SELECT 
        r.session,
        m.manufacturer || ' ' || m.model as device_name,
        r.ConstellationType,
        r.Svid,
        r.CarrierFrequencyHz,
        r.CodeType,
        CAST(ROUND(r.utcTimeMillis / 1000.0) AS BIGINT) as sec_epoch,
        r.Cn0DbHz as c_t
    FROM read_parquet('/home/tempest/Documents/ISRO_Project/gnss-eda/Stats/scratch/typed/Raw.parquet') r
    JOIN read_csv_auto('/home/tempest/Documents/ISRO_Project/gnss-eda/Stats/scratch/sessions_meta.csv') m ON r.session = m.session
    WHERE r.session IN ('gnss_log_2026_10_22_08_30_05', 'gnss_log_2026_10_22_08_30_07')
      AND r.Cn0DbHz > 0
),
lagged_signals AS (
    SELECT 
        t1.session,
        t1.device_name,
        t1.sec_epoch,
        t1.ConstellationType,
        t1.Svid,
        t1.CarrierFrequencyHz,
        t1.CodeType,
        t1.c_t,
        t2.c_t as c_future
    FROM base_signals t1
    JOIN base_signals t2
      ON t1.session = t2.session
     AND t1.ConstellationType = t2.ConstellationType
     AND t1.Svid = t2.Svid
     AND t1.CarrierFrequencyHz = t2.CarrierFrequencyHz
     AND t1.CodeType = t2.CodeType
     AND t2.sec_epoch = t1.sec_epoch + 15
),
epoch_headroom AS (
    SELECT 
        session,
        device_name,
        sec_epoch,
        COUNT(*) as n_candidates,
        MAX(c_future) as best_future_cn0,
        MEDIAN(c_future) as median_future_cn0,
        MAX(c_future) - MEDIAN(c_future) as headroom_db
    FROM lagged_signals
    GROUP BY session, device_name, sec_epoch
    HAVING COUNT(*) >= 10
)
SELECT 
    device_name,
    COUNT(*) as n_epochs,
    ROUND(AVG(n_candidates), 1) as mean_candidates,
    ROUND(AVG(best_future_cn0), 2) as mean_best_future,
    ROUND(AVG(median_future_cn0), 2) as mean_median_future,
    ROUND(AVG(headroom_db), 2) as mean_headroom_db,
    ROUND(MEDIAN(headroom_db), 2) as median_headroom_db,
    ROUND(QUANTILE_CONT(headroom_db, 0.25), 2) as p25_headroom,
    ROUND(QUANTILE_CONT(headroom_db, 0.75), 2) as p75_headroom
FROM epoch_headroom
GROUP BY device_name;

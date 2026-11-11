-- Compute autocorrelation of C/N0 for lags 1 to 30s
WITH target_signals AS (
    SELECT 
        session,
        ConstellationType,
        Svid,
        CarrierFrequencyHz,
        CodeType,
        CAST(ROUND(utcTimeMillis / 1000.0) AS BIGINT) as sec_epoch,
        Cn0DbHz
    FROM read_parquet('/home/tempest/Documents/ISRO_Project/gnss-eda/Stats/scratch/typed/Raw.parquet')
    WHERE session = 'gnss_log_2026_10_22_08_30_05' AND Cn0DbHz > 0
),
lag_calc AS (
    SELECT 
        t1.Cn0DbHz as c0,
        t2.Cn0DbHz as c_lag,
        (t2.sec_epoch - t1.sec_epoch) as lag_sec
    FROM target_signals t1
    JOIN target_signals t2
      ON t1.ConstellationType = t2.ConstellationType
     AND t1.Svid = t2.Svid
     AND t1.CarrierFrequencyHz = t2.CarrierFrequencyHz
     AND t1.CodeType = t2.CodeType
     AND t2.sec_epoch BETWEEN t1.sec_epoch + 1 AND t1.sec_epoch + 30
)
SELECT 
    lag_sec,
    COUNT(*) as n_pairs,
    ROUND(CORR(c0, c_lag), 4) as cn0_autocorr
FROM lag_calc
GROUP BY lag_sec
ORDER BY lag_sec;

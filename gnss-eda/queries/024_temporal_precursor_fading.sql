-- Compute past 5s variance and future 15s delta
WITH ordered_signals AS (
    SELECT 
        ConstellationType,
        Svid,
        CarrierFrequencyHz,
        CodeType,
        CAST(ROUND(utcTimeMillis / 1000.0) AS BIGINT) as sec_epoch,
        Cn0DbHz
    FROM read_parquet('/home/tempest/Documents/ISRO_Project/gnss-eda/Stats/scratch/typed/Raw.parquet')
    WHERE session = 'gnss_log_2026_10_22_08_30_05' AND Cn0DbHz > 0
),
rolling_precursors AS (
    SELECT 
        sec_epoch,
        ConstellationType,
        Svid,
        CarrierFrequencyHz,
        CodeType,
        Cn0DbHz,
        -- Past 5s std dev
        STDDEV(Cn0DbHz) OVER (
            PARTITION BY ConstellationType, Svid, CarrierFrequencyHz, CodeType
            ORDER BY sec_epoch
            ROWS BETWEEN 5 PRECEDING AND CURRENT ROW
        ) as past_5s_std,
        -- Future 15s lead
        LEAD(Cn0DbHz, 15) OVER (
            PARTITION BY ConstellationType, Svid, CarrierFrequencyHz, CodeType
            ORDER BY sec_epoch
        ) as future_15s_cn0
    FROM ordered_signals
)
SELECT 
    CASE WHEN ABS(future_15s_cn0 - Cn0DbHz) > 1.5 THEN 1 ELSE 0 END as is_fading_outcome,
    FLOOR(past_5s_std / 0.1) * 0.1 as past_std_bin,
    COUNT(*) as n_obs
FROM rolling_precursors
WHERE past_5s_std IS NOT NULL AND future_15s_cn0 IS NOT NULL
GROUP BY is_fading_outcome, past_std_bin
ORDER BY past_std_bin, is_fading_outcome;

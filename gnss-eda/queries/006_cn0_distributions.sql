
-- Compute binned distributions of Cn0DbHz across device, constellation, and code type
WITH clean_raw AS (
    SELECT 
        r.session,
        s.device_name,
        r.ConstellationType,
        r.CodeType,
        CASE 
            WHEN r.CarrierFrequencyHz BETWEEN 1.15e9 AND 1.25e9 THEN 'L5/E5a/B2a (~1.18 GHz)'
            WHEN r.CarrierFrequencyHz BETWEEN 1.55e9 AND 1.62e9 THEN 'L1/E1/B1 (~1.58 GHz)'
            ELSE 'Other'
        END as freq_band,
        r.Cn0DbHz
    FROM read_parquet('/home/tempest/Documents/ISRO_Project/gnss-eda/Stats/scratch/typed/Raw.parquet') r
    JOIN (
        SELECT session, manufacturer || ' ' || model as device_name
        FROM read_csv_auto('/home/tempest/Documents/ISRO_Project/gnss-eda/Stats/scratch/sessions_meta.csv')
    ) s ON r.session = s.session
    WHERE r.Cn0DbHz > 0
)
SELECT 
    device_name,
    ConstellationType,
    freq_band,
    CodeType,
    COUNT(*) as n_rows,
    ROUND(AVG(Cn0DbHz), 2) as mean_cn0,
    ROUND(STDDEV(Cn0DbHz), 2) as std_cn0,
    ROUND(QUANTILE_CONT(Cn0DbHz, 0.05), 2) as p05_cn0,
    ROUND(QUANTILE_CONT(Cn0DbHz, 0.25), 2) as p25_cn0,
    ROUND(MEDIAN(Cn0DbHz), 2) as median_cn0,
    ROUND(QUANTILE_CONT(Cn0DbHz, 0.75), 2) as p75_cn0,
    ROUND(QUANTILE_CONT(Cn0DbHz, 0.95), 2) as p95_cn0
FROM clean_raw
GROUP BY device_name, ConstellationType, freq_band, CodeType
ORDER BY device_name, ConstellationType, freq_band, CodeType;

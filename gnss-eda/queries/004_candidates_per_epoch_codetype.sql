
-- Compute histogram of signal count per epoch with and without CodeType
WITH epoch_counts AS (
    SELECT 
        session,
        utcTimeMillis,
        COUNT(*) as count_with_codetype,
        COUNT(DISTINCT ConstellationType || '_' || Svid || '_' || CarrierFrequencyHz) as count_sat_freq
    FROM read_parquet('/home/tempest/Documents/ISRO_Project/gnss-eda/Stats/scratch/typed/Raw.parquet')
    WHERE Cn0DbHz > 0
    GROUP BY session, utcTimeMillis
)
SELECT 
    count_with_codetype,
    count_sat_freq,
    COUNT(*) as num_epochs
FROM epoch_counts
GROUP BY count_with_codetype, count_sat_freq
ORDER BY count_with_codetype;

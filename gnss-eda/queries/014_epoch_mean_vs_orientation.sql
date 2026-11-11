
-- Compare variance of epoch mean across orientations vs variance of satellite deviations
WITH epoch_stats AS (
    SELECT 
        session,
        CAST(ROUND(utcTimeMillis / 1000.0) AS BIGINT) as sec_epoch,
        AVG(Cn0DbHz) as epoch_mean_cn0,
        STDDEV(Cn0DbHz) as epoch_sat_std
    FROM read_parquet('/home/tempest/Documents/ISRO_Project/gnss-eda/Stats/scratch/typed/Raw.parquet')
    WHERE session IN ('0_up_facing', '0_down_facing', '90_front_vertical', '90_back_vertical')
      AND Cn0DbHz > 0
    GROUP BY session, sec_epoch
    HAVING COUNT(*) >= 10
)
SELECT 
    session,
    ROUND(AVG(epoch_mean_cn0), 2) as mean_of_epoch_means,
    ROUND(STDDEV(epoch_mean_cn0), 2) as std_between_epochs,
    ROUND(AVG(epoch_sat_std), 2) as avg_within_epoch_std
FROM epoch_stats
GROUP BY session
ORDER BY mean_of_epoch_means DESC;

-- Extract time series of selected satellites and Accel magnitude
WITH sess_accel AS (
    SELECT 
        CAST(ROUND(utcTimeMillis / 1000.0) AS BIGINT) as sec_epoch,
        SQRT(POWER(AccelXMps2, 2) + POWER(AccelYMps2, 2) + POWER(AccelZMps2, 2)) as acc_norm
    FROM read_parquet('/home/tempest/Documents/ISRO_Project/gnss-eda/Stats/scratch/typed/Accel.parquet')
    WHERE session = 'gnss_log_2026_10_22_08_30_05'
),
acc_stats AS (
    SELECT 
        sec_epoch,
        ROUND(STDDEV(acc_norm), 3) as acc_std
    FROM sess_accel
    GROUP BY sec_epoch
),
sat_tracks AS (
    SELECT 
        CAST(ROUND(r.utcTimeMillis / 1000.0) AS BIGINT) as sec_epoch,
        r.Svid,
        r.Cn0DbHz,
        st.ElevationDegrees as el
    FROM read_parquet('/home/tempest/Documents/ISRO_Project/gnss-eda/Stats/scratch/typed/Raw.parquet') r
    JOIN (
        SELECT DISTINCT CAST(ROUND(UnixTimeMillis / 1000.0) AS BIGINT) as sec_epoch,
               ConstellationType, Svid, ElevationDegrees
        FROM read_parquet('/home/tempest/Documents/ISRO_Project/gnss-eda/Stats/scratch/typed/Status.parquet')
        WHERE session = 'gnss_log_2026_10_22_08_30_05' AND ConstellationType = 1
    ) st ON CAST(ROUND(r.utcTimeMillis / 1000.0) AS BIGINT) = st.sec_epoch
       AND r.ConstellationType = st.ConstellationType
       AND r.Svid = st.Svid
    WHERE r.session = 'gnss_log_2026_10_22_08_30_05'
      AND r.ConstellationType = 1 -- GPS
      AND r.Svid IN (10, 14, 27)
      AND r.Cn0DbHz > 0
)
SELECT 
    s.sec_epoch - (SELECT MIN(sec_epoch) FROM sat_tracks) as rel_sec,
    s.Svid,
    s.Cn0DbHz,
    s.el,
    COALESCE(a.acc_std, 0.0) as acc_std
FROM sat_tracks s
LEFT JOIN acc_stats a ON s.sec_epoch = a.sec_epoch
WHERE s.sec_epoch - (SELECT MIN(sec_epoch) FROM sat_tracks) BETWEEN 100 AND 300
ORDER BY s.Svid, rel_sec;

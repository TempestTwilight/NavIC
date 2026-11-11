
-- Compute sequential sum of squares (ANOVA / R^2 decomposition) for Cn0DbHz
WITH target_sessions AS (
    SELECT session, manufacturer || ' ' || model as device_name
    FROM (
        SELECT session, manufacturer, model,
               ROW_NUMBER() OVER (PARTITION BY model ORDER BY session) as rn
        FROM read_csv_auto('/home/tempest/Documents/ISRO_Project/gnss-eda/Stats/scratch/sessions_meta.csv')
    ) WHERE rn <= 4
),
base_data AS (
    SELECT 
        r.session,
        s.device_name,
        r.ConstellationType,
        r.CarrierFrequencyHz,
        r.CodeType,
        CAST(ROUND(r.utcTimeMillis / 1000.0) AS BIGINT) as sec_epoch,
        st.ElevationDegrees,
        FLOOR(st.ElevationDegrees / 5.0) * 5.0 as elev_bin,
        r.Cn0DbHz
    FROM read_parquet('/home/tempest/Documents/ISRO_Project/gnss-eda/Stats/scratch/typed/Raw.parquet') r
    JOIN target_sessions s ON r.session = s.session
    JOIN (
        SELECT DISTINCT session, CAST(ROUND(UnixTimeMillis / 1000.0) AS BIGINT) as sec_epoch,
               ConstellationType, Svid, CarrierFrequencyHz, ElevationDegrees
        FROM read_parquet('/home/tempest/Documents/ISRO_Project/gnss-eda/Stats/scratch/typed/Status.parquet')
    ) st ON r.session = st.session
       AND CAST(ROUND(r.utcTimeMillis / 1000.0) AS BIGINT) = st.sec_epoch
       AND r.ConstellationType = st.ConstellationType
       AND r.CarrierFrequencyHz = st.CarrierFrequencyHz
    WHERE r.Cn0DbHz > 0
),
stats AS (
    SELECT 
        COUNT(*) as total_n,
        AVG(Cn0DbHz) as grand_mean,
        VAR_SAMP(Cn0DbHz) as total_variance
    FROM base_data
),
dev_stats AS (
    SELECT 
        SUM(dev_n * POWER(dev_mean - grand_mean, 2)) / ((total_n - 1) * total_variance) as r2_device
    FROM (
        SELECT device_name, COUNT(*) as dev_n, AVG(Cn0DbHz) as dev_mean
        FROM base_data
        GROUP BY device_name
    ), stats
    GROUP BY total_n, total_variance
),
elev_stats AS (
    SELECT 
        SUM(el_n * POWER(el_mean - grand_mean, 2)) / ((total_n - 1) * total_variance) as r2_elev
    FROM (
        SELECT elev_bin, COUNT(*) as el_n, AVG(Cn0DbHz) as el_mean
        FROM base_data
        GROUP BY elev_bin
    ), stats
    GROUP BY total_n, total_variance
),
const_stats AS (
    SELECT 
        SUM(c_n * POWER(c_mean - grand_mean, 2)) / ((total_n - 1) * total_variance) as r2_const
    FROM (
        SELECT ConstellationType, COUNT(*) as c_n, AVG(Cn0DbHz) as c_mean
        FROM base_data
        GROUP BY ConstellationType
    ), stats
    GROUP BY total_n, total_variance
),
epoch_stats AS (
    -- Common-mode epoch variance within session
    SELECT 
        AVG(epoch_var) / total_variance as within_epoch_var_share
    FROM (
        SELECT session, sec_epoch, VAR_SAMP(Cn0DbHz) as epoch_var
        FROM base_data
        GROUP BY session, sec_epoch
        HAVING COUNT(*) >= 5
    ), stats
    GROUP BY total_variance
)
SELECT 
    ROUND(d.r2_device * 100.0, 2) as pct_device,
    ROUND(e.r2_elev * 100.0, 2) as pct_elevation,
    ROUND(c.r2_const * 100.0, 2) as pct_constellation,
    ROUND((1.0 - ep.within_epoch_var_share) * 100.0, 2) as pct_epoch_common_mode,
    ROUND(ep.within_epoch_var_share * 100.0, 2) as pct_satellite_residual
FROM dev_stats d, elev_stats e, const_stats c, epoch_stats ep;

-- =============================================================================
-- Feature Engineering Experiments
-- =============================================================================
-- 
-- This file defines candidate features to test for predictive value.
-- All features must be computable via SQL for memory safety.
-- =============================================================================

-- =============================================================================
-- Category 1: Temporal Features
-- =============================================================================

-- Already implemented:
-- - current_cn0
-- - delta_recent_5s
-- - sigma_5s (rolling 5-second std)

-- New candidates:

-- 1. Momentum indicators
SELECT 
    Cn0DbHz - LAG(Cn0DbHz, 10) OVER (PARTITION BY signal_key ORDER BY sec_epoch) AS delta_10s,
    
    -- Exponential moving average (approximation via weighted window)
    SUM(Cn0DbHz * CASE WHEN row_num <= 5 THEN 0.3 
                       WHEN row_num <= 10 THEN 0.2 
                       ELSE 0.1 END) OVER (
        PARTITION BY signal_key
        ROWS BETWEEN 10 PRECEDING AND CURRENT ROW
    ) / SUM(CASE WHEN row_num <= 5 THEN 0.3 
                 WHEN row_num <= 10 THEN 0.2 
                 ELSE 0.1 END) OVER (
        PARTITION BY signal_key
        ROWS BETWEEN 10 PRECEDING AND CURRENT ROW
    ) AS ema_10s,
    
    -- Second derivative (acceleration of change)
    (Cn0DbHz - LAG(Cn0DbHz, 1) OVER w) - 
    (LAG(Cn0DbHz, 1) OVER w - LAG(Cn0DbHz, 2) OVER w) AS cn0_acceleration
FROM features
WINDOW w AS (PARTITION BY signal_key ORDER BY sec_epoch);

-- =============================================================================
-- Category 2: Physics Features
-- =============================================================================

-- Already implemented:
-- - sin_elevation
-- - ElevationDegrees

-- New candidates:

-- 2. Airmass (but HIGHLY correlated with sin_elevation, may cause collinearity)
-- Kasten-Young formula: X = 1 / (sin_elev + 0.50572 * (elev_deg + 6.07995)^-1.6364)
-- WARNING: ρ = -0.99 with sin_elevation, exclude or use instead of sin_elevation

-- 3. Time to culmination (satellite rising vs setting)
-- Derive from elevation trend
SELECT 
    CASE 
        WHEN elevation_delta_5s > 0.5 THEN 'rising'
        WHEN elevation_delta_5s < -0.5 THEN 'setting'
        ELSE 'transiting'
    END AS satellite_phase,
    
    -- Azimuth change rate (satellite motion across sky)
    AzimuthDegrees - LAG(AzimuthDegrees, 5) OVER (
        PARTITION BY signal_key ORDER BY sec_epoch
    ) AS azimuth_delta_5s
FROM features;

-- 4. Atmospheric attenuation proxy
-- Higher elevation = less atmosphere
-- log scale: -k * 1/sin(elevation)
SELECT 
    -0.5 / SIN(RADIANS(ElevationDegrees)) AS atmospheric_loss_proxy
FROM features
WHERE ElevationDegrees > 5;

-- =============================================================================
-- Category 3: Device-Specific Features
-- =============================================================================

-- 5. Device-relative C/N0 (normalize per device)
SELECT 
    Cn0DbHz - device_median_cn0 AS cn0_vs_device_median,
    Cn0DbHz - device_elevation_median_cn0 AS cn0_vs_expected
FROM features
JOIN (
    SELECT 
        device_name,
        MEDIAN(Cn0DbHz) AS device_median_cn0
    FROM features
    GROUP BY device_name
) device_stats ON features.device_name = device_stats.device_name
JOIN (
    SELECT 
        device_name,
        CASE 
            WHEN ElevationDegrees < 20 THEN 'low'
            WHEN ElevationDegrees < 45 THEN 'mid'
            ELSE 'high'
        END AS elevation_band,
        MEDIAN(Cn0DbHz) AS device_elevation_median_cn0
    FROM features
    GROUP BY device_name, elevation_band
) device_elev_stats ON features.device_name = device_elev_stats.device_name;

-- =============================================================================
-- Category 4: Context Features
-- =============================================================================

-- 6. Epoch-level statistics (common-mode signal)
SELECT 
    Cn0DbHz - epoch_mean_cn0 AS cn0_vs_epoch_mean,
    epoch_std_cn0 AS epoch_variability,
    epoch_n_signals AS constellation_size,
    
    -- Rank within epoch (1 = best signal)
    RANK() OVER (PARTITION BY session, sec_epoch ORDER BY Cn0DbHz DESC) AS current_rank,
    
    -- Gap to best signal in epoch
    MAX(Cn0DbHz) OVER (PARTITION BY session, sec_epoch) - Cn0DbHz AS gap_to_best
FROM features;

-- 7. Constellation features
SELECT 
    ConstellationType,
    COUNT(*) OVER (PARTITION BY session, sec_epoch, ConstellationType) AS n_constellation_signals,
    AVG(Cn0DbHz) OVER (PARTITION BY session, sec_epoch, ConstellationType) AS constellation_mean_cn0
FROM features;

-- =============================================================================
-- Category 5: Orientation Features (Device Body Frame)
-- =============================================================================

-- Requires join with OrientationDeg.parquet (only in June sessions)

-- 8. Body-frame directivity
-- Already implemented:
-- - s_bz (screen-facing axis component)
-- - off_axis_angle

-- New candidates:
SELECT 
    -- Lateral directivity (holding phone sideways)
    s_by AS lateral_component,
    
    -- Combined directivity index
    SQRT(s_bx^2 + s_by^2 + s_bz^2) AS body_vector_norm,
    
    -- Pitch angle (phone tilt)
    pitchDeg,
    
    -- Interaction: elevation × orientation
    sin_elevation * s_bz AS elev_orientation_product
FROM features_with_orientation;

-- =============================================================================
-- Category 6: Motion Features
-- =============================================================================

-- Requires UncalAccel.parquet

-- 9. Accelerometer-based motion indicator
SELECT 
    STDDEV(accel_norm) OVER (
        PARTITION BY session
        ORDER BY sec_epoch
        ROWS BETWEEN 5 PRECEDING AND CURRENT ROW
    ) AS motion_indicator,
    
    -- Phone movement → orientation change → signal fluctuation
    MAX(accel_norm) - MIN(accel_norm) OVER (
        PARTITION BY session
        ORDER BY sec_epoch
        ROWS BETWEEN 5 PRECEDING AND CURRENT ROW
    ) AS motion_range
FROM features_with_accel;

-- =============================================================================
-- Feature Selection Guidance
-- =============================================================================

-- High-value (verified): current_cn0, sin_elevation, device_id, sigma_5s
-- Medium-value (test): delta_recent, motion_indicator, epoch_mean_cn0
-- Low-value (exclude): airmass (collinear), Euler yaw/pitch (unstable)
-- Unknown (need experiment): orientation features, second derivative

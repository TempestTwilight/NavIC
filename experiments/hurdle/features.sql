-- =============================================================================
-- Experiment B: Two-Stage Hurdle Model Feature Extraction
-- =============================================================================
-- 
-- Stage 1: Binary classifier for fading detection (|Δ C/N0| > 1.5 dB)
-- Stage 2: Specialized regressor for fading magnitude
--
-- This query extends baseline features with additional precursors
-- =============================================================================

-- Inherit all baseline features, plus:

-- =============================================================================
-- Additional temporal features for fading prediction
-- =============================================================================

-- Past 10-second statistics
SELECT 
    *,
    -- Extended jitter window
    STDDEV_SAMP(Cn0DbHz) OVER (
        PARTITION BY ConstellationType, Svid, CarrierFrequencyHz, CodeType
        ROWS BETWEEN 10 PRECEDING AND CURRENT ROW
    ) AS sigma_10s,
    
    -- Range (max - min) in past 5s
    MAX(Cn0DbHz) OVER (
        PARTITION BY ConstellationType, Svid, CarrierFrequencyHz, CodeType
        ROWS BETWEEN 5 PRECEDING AND CURRENT ROW
    ) - MIN(Cn0DbHz) OVER (
        PARTITION BY ConstellationType, Svid, CarrierFrequencyHz, CodeType
        ROWS BETWEEN 5 PRECEDING AND CURRENT ROW
    ) AS range_5s,
    
    -- Rate of change
    (Cn0DbHz - cn0_lag_1) AS delta_1s,
    (Cn0DbHz - cn0_lag_2) / 2.0 AS delta_2s_avg,
    
    -- Trend indicator (positive = improving signal)
    CASE 
        WHEN Cn0DbHz > cn0_lag_1 AND cn0_lag_1 > cn0_lag_2 THEN 1  -- Rising
        WHEN Cn0DbHz < cn0_lag_1 AND cn0_lag_1 < cn0_lag_2 THEN -1  -- Falling
        ELSE 0  -- Stable
    END AS trend_2s,
    
    -- Elevation change (satellite motion)
    ElevationDegrees - LAG(ElevationDegrees, 5) OVER (
        PARTITION BY ConstellationType, Svid, CarrierFrequencyHz, CodeType
        ORDER BY sec_epoch
    ) AS elevation_delta_5s
    
FROM baseline_features;

-- =============================================================================
-- Class weighting for imbalanced training
-- =============================================================================

-- Fading events are ~16% of epochs
-- Use class_weight='balanced' or compute sample weights:
-- weight_fading = n_total / (2 * n_fading)
-- weight_static = n_total / (2 * n_static)

-- =============================================================================
-- Evaluation metrics for two-stage model
-- =============================================================================

-- Stage 1: AUC, precision@k, recall, F1
-- Stage 2: RMSE on fading epochs only, MAE
-- Combined: Expected regret = P(fade) * MSE_fade + (1-P(fade)) * 0

-- =============================================================================
-- Threshold optimization
-- =============================================================================

-- Select threshold for Stage 1 classifier that minimizes expected regret:
-- If we predict fading but signal is static: we use fading regressor (noise)
-- If we predict static but signal fades: we predict 0, large error
-- Optimize threshold to minimize combined cost

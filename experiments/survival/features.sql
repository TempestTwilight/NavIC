-- =============================================================================
-- Experiment E: Survival Model Feature Extraction
-- =============================================================================

-- Binary target: signal remains above sensitivity floor at +15s
SELECT 
    *,
    CASE 
        WHEN cn0_future_15s >= 15.0 THEN 1  -- Survived
        ELSE 0  -- Dropped out or fell below threshold
    END AS survived_15s,
    
    -- Distance from floor (margin)
    Cn0DbHz - 12.0 AS margin_above_floor,
    
    -- Elevation-band dropout rate (from Phase 2 INS-14)
    CASE 
        WHEN ElevationDegrees >= 45 THEN 0.002  -- 0.2% dropout
        WHEN ElevationDegrees >= 20 THEN 0.012  -- ~1.2% dropout
        ELSE 0.076  -- 7.6% dropout at low elevation
    END AS empirical_dropout_rate
FROM baseline_features;

-- =============================================================================
-- Features specific to survival prediction
-- =============================================================================

-- Current signal strength relative to device baseline
-- Signals near floor are more likely to drop

-- Recent stability (low jitter = stable tracking)
-- sigma_5s is already computed

-- Elevation trend (rising satellites are more stable)
-- elevation_delta_5s (already computed in hurdle model)

-- =============================================================================
-- Train survival model
-- =============================================================================

-- Classifier: LogisticRegression(class_weight='balanced')
-- or GradientBoostingClassifier for non-linear effects
-- 
-- CV: Day-grouped (NOT random)
-- 
-- Evaluation: Calibration curve, Brier score, AUC

-- =============================================================================
-- Compute expected value for selection
-- =============================================================================

-- For each candidate satellite at epoch:
-- E[C/N0] = P(survive | features) * predicted_C/N0 + (1 - P(survive)) * 12.0
--
-- Select satellite with max E[C/N0]
--
-- Compare realized C/N0 vs selection to compute regret

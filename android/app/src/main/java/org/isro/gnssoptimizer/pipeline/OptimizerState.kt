package org.isro.gnssoptimizer.pipeline

import org.isro.gnssoptimizer.model.RidgeModel

enum class PipelineStatus {
    PERMISSION_REQUIRED,
    INITIALIZING,           // Buffering 30s observation window
    INSUFFICIENT_DATA,      // No satellites or C/N0 values
    MONITORING,             // Signal stable, normal monitoring
    DEGRADATION_PREDICTED,  // Temporal model predicts drop
    RECOMMENDATION_READY    // Turn recommendation calculated
}

enum class SignalQuality {
    EXCELLENT,  // >= 35 dB-Hz
    GOOD,       // 30 - 35 dB-Hz
    FAIR,       // 25 - 30 dB-Hz
    POOR        // < 25 dB-Hz
}

data class CandidateResult(
    val angleDeg: Double,
    val meanCn0: Double
)

data class OptimizerUiState(
    val status: PipelineStatus = PipelineStatus.INITIALIZING,
    val statusMessage: String = "Initializing GNSS observation window...",
    val bufferProgressSeconds: Int = 0, // 0 to 30

    // Current orientation
    val azimuthDeg: Double = 0.0,
    val pitchDeg: Double = 0.0,
    val rollDeg: Double = 0.0,

    // Current signal
    val currentMeanCn0: Double = 0.0,
    val signalQuality: SignalQuality = SignalQuality.FAIR,

    // Satellite counts
    val satellitesVisible: Int = 0,
    val satellitesWithMeasurements: Int = 0,
    val satellitesUsedInFix: Int = 0,

    // Stage 1: Temporal model prediction
    val predictedCn0Temporal: Double = 0.0,
    val predictedDeltaTemporal: Double = 0.0,
    val isDegradationPredicted: Boolean = false,

    // Stage 2: Orientation model recommendation
    val bestYawDeg: Double = 0.0,
    val deltaYawDeg: Double = 0.0,
    val turnDirectionText: String = "Maintain Heading",
    val turnDegreesText: String = "0°",
    val recommendationInstruction: String = "Signal is optimal. No rotation needed.",
    val predictedImprovementDb: Double = 0.0,
    val candidates: List<CandidateResult> = emptyList(),

    // Stage 3: Top feature explanations
    val temporalTopFeatures: List<RidgeModel.FeatureContribution> = emptyList(),
    val orientationTopFeatures: List<RidgeModel.FeatureContribution> = emptyList(),

    // Last update timestamp
    val lastUpdateEpochMs: Long = 0L
)

package org.isro.gnssoptimizer.model

import kotlin.math.abs

/**
 * Native Ridge regression inference engine.
 * Computes standard scaling: z_i = (x_i - mean_i) / scale_i
 * and linear dot product: y = sum(z_i * coef_i) + intercept.
 *
 * Feature contribution: c_i = z_i * coef_i (in dB-Hz relative to intercept).
 */
class RidgeModel(
    val featureNames: Array<String>,
    val scalerMean: DoubleArray,
    val scalerScale: DoubleArray,
    val coefficients: DoubleArray,
    val intercept: Double
) {
    init {
        require(featureNames.size == scalerMean.size) { "Feature count mismatch: names vs scalerMean" }
        require(scalerMean.size == scalerScale.size) { "Scaler mean/scale dimension mismatch" }
        require(scalerMean.size == coefficients.size) { "Scaler vs coefficient dimension mismatch" }
    }

    val numFeatures: Int get() = featureNames.size

    /**
     * Compute prediction for raw feature vector x.
     */
    fun predict(x: DoubleArray): Double {
        require(x.size == numFeatures) { "Input vector length ${x.size} != expected $numFeatures" }
        var sum = intercept
        for (i in 0 until numFeatures) {
            val scaled = (x[i] - scalerMean[i]) / scalerScale[i]
            sum += scaled * coefficients[i]
        }
        return sum
    }

    /**
     * Data class holding a feature's calculated additive contribution to the prediction.
     */
    data class FeatureContribution(
        val name: String,
        val contribution: Double,
        val absContribution: Double = abs(contribution),
        val isPositive: Boolean = contribution >= 0.0
    )

    /**
     * Compute top N most influential feature contributions for raw feature vector x.
     */
    fun topContributions(x: DoubleArray, topN: Int = 3): List<FeatureContribution> {
        require(x.size == numFeatures) { "Input vector length ${x.size} != expected $numFeatures" }
        val contributions = ArrayList<FeatureContribution>(numFeatures)

        for (i in 0 until numFeatures) {
            val scaled = (x[i] - scalerMean[i]) / scalerScale[i]
            val c = scaled * coefficients[i]
            contributions.add(FeatureContribution(featureNames[i], c))
        }

        contributions.sortByDescending { it.absContribution }
        return contributions.take(topN)
    }

    companion object {
        fun createTemporalModel(): RidgeModel = RidgeModel(
            featureNames = ModelArtifacts.TEMPORAL_FEATURES,
            scalerMean = ModelArtifacts.TEMPORAL_SCALER_MEAN,
            scalerScale = ModelArtifacts.TEMPORAL_SCALER_SCALE,
            coefficients = ModelArtifacts.TEMPORAL_RIDGE_COEF,
            intercept = ModelArtifacts.TEMPORAL_RIDGE_INTERCEPT
        )

        fun createOrientationModel(): RidgeModel = RidgeModel(
            featureNames = ModelArtifacts.ORIENTATION_FEATURES,
            scalerMean = ModelArtifacts.ORIENTATION_SCALER_MEAN,
            scalerScale = ModelArtifacts.ORIENTATION_SCALER_SCALE,
            coefficients = ModelArtifacts.ORIENTATION_RIDGE_COEF,
            intercept = ModelArtifacts.ORIENTATION_RIDGE_INTERCEPT
        )
    }
}

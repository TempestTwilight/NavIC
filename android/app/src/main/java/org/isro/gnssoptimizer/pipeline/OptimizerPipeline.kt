package org.isro.gnssoptimizer.pipeline

import org.isro.gnssoptimizer.gnss.GnssRepository
import org.isro.gnssoptimizer.gnss.SatelliteObservation
import org.isro.gnssoptimizer.model.FeatureEngineering
import org.isro.gnssoptimizer.model.RidgeModel
import org.isro.gnssoptimizer.sensor.OrientationData
import kotlin.math.*

class OptimizerPipeline(
    private val gnssRepository: GnssRepository
) {
    private val temporalModel = RidgeModel.createTemporalModel()
    private val orientationModel = RidgeModel.createOrientationModel()

    // 8 candidate yaw angles evaluated during optimization (matching training and full compass circle)
    private val candidateAngles = doubleArrayOf(0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0)

    private var firstEpochTimestampMs: Long = 0L
    private var lastOptimizationTimestampMs: Long = 0L
    private val cooldownMs = 5000L // 5s hysteresis cooldown

    // Preallocated buffers for zero-allocation inference in hot loops
    private val temporalFeatureBuffer = DoubleArray(temporalModel.numFeatures)
    private val orientationFeatureBuffer = DoubleArray(orientationModel.numFeatures)

    fun reset() {
        firstEpochTimestampMs = 0L
        lastOptimizationTimestampMs = 0L
    }

    fun processEpoch(
        currentObs: List<SatelliteObservation>,
        orientation: OrientationData
    ): OptimizerUiState {
        val nowMs = System.currentTimeMillis()
        if (firstEpochTimestampMs == 0L) {
            firstEpochTimestampMs = nowMs
        }

        val elapsedBufferSeconds = ((nowMs - firstEpochTimestampMs) / 1000L).toInt().coerceIn(0, 30)

        val visibleCount = gnssRepository.totalSatellitesInView
        val measuredCount = gnssRepository.totalSatellitesWithMeasurements
        val fixCount = gnssRepository.totalSatellitesUsedInFix

        if (currentObs.isEmpty()) {
            return OptimizerUiState(
                status = PipelineStatus.INSUFFICIENT_DATA,
                statusMessage = "Waiting for GNSS satellite measurements...",
                bufferProgressSeconds = elapsedBufferSeconds,
                azimuthDeg = orientation.azimuthDeg,
                pitchDeg = orientation.pitchDeg,
                rollDeg = orientation.rollDeg,
                satellitesVisible = visibleCount,
                satellitesWithMeasurements = measuredCount,
                satellitesUsedInFix = fixCount,
                lastUpdateEpochMs = nowMs
            )
        }

        // Calculate current baseline mean C/N0
        var sumCn0 = 0.0
        for (obs in currentObs) {
            sumCn0 += obs.cn0DbHz
        }
        val currentMeanCn0 = sumCn0 / currentObs.size

        val quality = when {
            currentMeanCn0 >= 35.0 -> SignalQuality.EXCELLENT
            currentMeanCn0 >= 30.0 -> SignalQuality.GOOD
            currentMeanCn0 >= 25.0 -> SignalQuality.FAIR
            else -> SignalQuality.POOR
        }

        // Initial 30s observation buffer gate
        if (elapsedBufferSeconds < 30) {
            return OptimizerUiState(
                status = PipelineStatus.INITIALIZING,
                statusMessage = "Buffering observation window (${elapsedBufferSeconds}s / 30s)...",
                bufferProgressSeconds = elapsedBufferSeconds,
                azimuthDeg = orientation.azimuthDeg,
                pitchDeg = orientation.pitchDeg,
                rollDeg = orientation.rollDeg,
                currentMeanCn0 = currentMeanCn0,
                signalQuality = quality,
                satellitesVisible = visibleCount,
                satellitesWithMeasurements = measuredCount,
                satellitesUsedInFix = fixCount,
                lastUpdateEpochMs = nowMs
            )
        }

        // ==========================================
        // STAGE 1: TEMPORAL RIDGE MODEL INFERENCE
        // ==========================================
        var predTemporalSum = 0.0
        var validTemporalPredictions = 0
        var topTemporalContributors: List<RidgeModel.FeatureContribution> = emptyList()

        for (obs in currentObs) {
            val track = gnssRepository.tracks[obs.signalKey] ?: continue
            if (buildTemporalFeatures(obs, track, orientation, temporalFeatureBuffer)) {
                val pred = temporalModel.predict(temporalFeatureBuffer)
                predTemporalSum += pred
                validTemporalPredictions++

                if (topTemporalContributors.isEmpty()) {
                    topTemporalContributors = temporalModel.topContributions(temporalFeatureBuffer, 3)
                }
            }
        }

        val predictedCn0 = if (validTemporalPredictions > 0) {
            predTemporalSum / validTemporalPredictions
        } else {
            currentMeanCn0
        }

        val deltaCn0 = predictedCn0 - currentMeanCn0
        // Degradation condition: predicted drop >= 1.5 dB-Hz OR absolute weak signal < 28 dB-Hz
        val isDegradation = (deltaCn0 <= -1.5) || (currentMeanCn0 < 28.0)

        // ==========================================
        // STAGE 2: CONDITIONAL ORIENTATION MODEL
        // ==========================================
        var bestYaw = orientation.azimuthDeg
        var deltaYaw = 0.0
        var improvementDb = 0.0
        var candidateResults = emptyList<CandidateResult>()
        var topOrientContributors: List<RidgeModel.FeatureContribution> = emptyList()

        var turnDirectionText = "Maintain Heading"
        var turnDegreesText = "0°"
        var instructionText = "Signal is optimal. No rotation needed."
        var currentStatus = PipelineStatus.MONITORING
        var statusMessage = "Monitoring: GNSS signal stable."

        if (isDegradation) {
            currentStatus = PipelineStatus.DEGRADATION_PREDICTED
            statusMessage = "Degradation predicted (Δ ${String.format("%.1f", deltaCn0)} dB-Hz). Optimizing orientation..."

            // Evaluate all candidate angles
            val evaluated = ArrayList<CandidateResult>(candidateAngles.size)
            var maxCandidateCn0 = -1e9
            var bestAngle = orientation.azimuthDeg

            for (angle in candidateAngles) {
                var candidateSum = 0.0
                var candidateCount = 0

                for (obs in currentObs) {
                    buildOrientationFeatures(obs, angle, orientationFeatureBuffer)
                    val p = orientationModel.predict(orientationFeatureBuffer)
                    candidateSum += p
                    candidateCount++
                }

                val meanCand = if (candidateCount > 0) candidateSum / candidateCount else currentMeanCn0
                evaluated.add(CandidateResult(angle, meanCand))

                if (meanCand > maxCandidateCn0) {
                    maxCandidateCn0 = meanCand
                    bestAngle = angle
                }
            }

            candidateResults = evaluated
            bestYaw = bestAngle

            // Find candidate closest to current yaw for baseline comparison
            val currentCandidate = evaluated.minByOrNull {
                abs(FeatureEngineering.circularDeltaDeg(it.angleDeg, orientation.azimuthDeg))
            }
            val currentCandCn0 = currentCandidate?.meanCn0 ?: currentMeanCn0
            improvementDb = (maxCandidateCn0 - currentCandCn0).coerceAtLeast(0.0)

            deltaYaw = FeatureEngineering.circularDeltaDeg(bestYaw, orientation.azimuthDeg)

            // Compute top orientation features for the best candidate
            if (currentObs.isNotEmpty()) {
                buildOrientationFeatures(currentObs.first(), bestYaw, orientationFeatureBuffer)
                topOrientContributors = orientationModel.topContributions(orientationFeatureBuffer, 3)
            }

            if (abs(deltaYaw) > 10.0 && improvementDb >= 0.5) {
                currentStatus = PipelineStatus.RECOMMENDATION_READY
                val dir = if (deltaYaw > 0) "Right" else "Left"
                val deg = abs(deltaYaw).roundToInt()
                turnDirectionText = "Turn $dir $deg°"
                turnDegreesText = "${if (deltaYaw > 0) "+" else "-"}${deg}°"
                instructionText = "Rotate phone $dir by $deg° to face bearing ${bestYaw.roundToInt()}° (gain: +${String.format("%.1f", improvementDb)} dB-Hz)"
                statusMessage = "Optimization complete: $instructionText"
            } else {
                instructionText = "Current orientation is already near optimal."
                statusMessage = "Orientation optimal for current sky view."
            }
        }

        return OptimizerUiState(
            status = currentStatus,
            statusMessage = statusMessage,
            bufferProgressSeconds = 30,
            azimuthDeg = orientation.azimuthDeg,
            pitchDeg = orientation.pitchDeg,
            rollDeg = orientation.rollDeg,
            currentMeanCn0 = currentMeanCn0,
            signalQuality = quality,
            satellitesVisible = visibleCount,
            satellitesWithMeasurements = measuredCount,
            satellitesUsedInFix = fixCount,
            predictedCn0Temporal = predictedCn0,
            predictedDeltaTemporal = deltaCn0,
            isDegradationPredicted = isDegradation,
            bestYawDeg = bestYaw,
            deltaYawDeg = deltaYaw,
            turnDirectionText = turnDirectionText,
            turnDegreesText = turnDegreesText,
            recommendationInstruction = instructionText,
            predictedImprovementDb = improvementDb,
            candidates = candidateResults,
            temporalTopFeatures = topTemporalContributors,
            orientationTopFeatures = topOrientContributors,
            lastUpdateEpochMs = nowMs
        )
    }

    /**
     * Build 46 temporal features exactly matching 02_future_model.py.
     */
    private fun buildTemporalFeatures(
        obs: SatelliteObservation,
        track: org.isro.gnssoptimizer.gnss.SatelliteTrack,
        orient: OrientationData,
        out: DoubleArray
    ): Boolean {
        val now = obs.timestampMs

        // Find lag observations (nearest within 400ms tolerance)
        val lag1s = track.getLaggedObservation(now, 1000L) ?: obs
        val lag2s = track.getLaggedObservation(now, 2000L) ?: obs
        val lag5s = track.getLaggedObservation(now, 5000L) ?: obs
        val lag10s = track.getLaggedObservation(now, 10000L) ?: obs

        val el = Math.toRadians(obs.elevationDegrees)
        val az = Math.toRadians(obs.azimuthDegrees)
        val yaw = Math.toRadians(orient.azimuthDeg)
        val pitch = Math.toRadians(orient.pitchDeg)
        val roll = Math.toRadians(orient.rollDeg)

        val relAz = az - yaw
        val elSin = sin(el)
        val elCos = cos(el)
        val relAzSin = sin(relAz)
        val relAzCos = cos(relAz)

        val airmass = FeatureEngineering.airmassSimple(obs.elevationDegrees)
        val airmassKy = FeatureEngineering.airmassKY(obs.elevationDegrees)
        val isAboveHorizon = if (obs.elevationDegrees > 0) 1.0 else 0.0

        val freqGhz = obs.signalKey.carrierFrequencyHz / 1e9
        val freqLossDb = FeatureEngineering.freqLossDb(obs.signalKey.carrierFrequencyHz.toDouble())

        val pitchSin = sin(pitch)
        val pitchCos = cos(pitch)
        val rollSin = sin(roll)
        val rollCos = cos(roll)

        val ctype = obs.signalKey.constellationType
        val isGps = if (ctype == 1) 1.0 else 0.0
        val isSbas = if (ctype == 2) 1.0 else 0.0
        val isGlonass = if (ctype == 3) 1.0 else 0.0
        val isQzss = if (ctype == 4) 1.0 else 0.0
        val isBeidou = if (ctype == 5) 1.0 else 0.0
        val isGalileo = if (ctype == 6) 1.0 else 0.0
        val isNavic = if (ctype == 7) 1.0 else 0.0

        val relAzXEl = relAzCos * elSin
        val navicXAirmass = isNavic * airmassKy
        val gpsXAirmass = isGps * airmassKy
        val galileoXAirmass = isGalileo * airmassKy
        val freqLossXElev = freqLossDb * elSin

        val cn0Delta1s = obs.cn0DbHz - lag1s.cn0DbHz
        val cn0Delta2s = obs.cn0DbHz - lag2s.cn0DbHz
        val cn0Delta5s = obs.cn0DbHz - lag5s.cn0DbHz

        val cn0Slope1s = cn0Delta1s / 1.0
        val cn0Slope5s = cn0Delta5s / 5.0

        val elDelta1s = obs.elevationDegrees - lag1s.elevationDegrees
        val elDelta5s = obs.elevationDegrees - lag5s.elevationDegrees
        val azDelta1s = FeatureEngineering.circularDeltaDeg(obs.azimuthDegrees, lag1s.azimuthDegrees)

        // Populate array in exact order of ModelArtifacts.TEMPORAL_FEATURES (36 features)
        out[0] = elSin
        out[1] = relAzSin
        out[2] = relAzCos
        out[3] = airmassKy
        out[4] = isAboveHorizon
        out[5] = freqLossDb
        out[6] = pitchSin
        out[7] = pitchCos
        out[8] = rollSin
        out[9] = rollCos
        out[10] = isGps
        out[11] = isSbas
        out[12] = isGlonass
        out[13] = isGalileo
        out[14] = isBeidou
        out[15] = isQzss
        out[16] = isNavic
        out[17] = relAzXEl
        out[18] = freqLossXElev
        out[19] = obs.cn0DbHz
        out[20] = obs.basebandCn0DbHz
        out[21] = lag1s.cn0DbHz
        out[22] = lag2s.cn0DbHz
        out[23] = lag5s.cn0DbHz
        out[24] = lag10s.cn0DbHz
        out[25] = cn0Delta1s
        out[26] = cn0Delta2s
        out[27] = cn0Delta5s
        out[28] = cn0Slope1s
        out[29] = cn0Slope5s
        out[30] = lag1s.elevationDegrees
        out[31] = lag5s.elevationDegrees
        out[32] = lag1s.azimuthDegrees
        out[33] = elDelta1s
        out[34] = elDelta5s
        out[35] = azDelta1s

        return true
    }

    /**
     * Build 24 orientation features exactly matching 02_orientation_model.py and 03_orientation_inference.py.
     */
    private fun buildOrientationFeatures(
        obs: SatelliteObservation,
        targetYawDeg: Double,
        out: DoubleArray
    ) {
        val (cosInc, incDeg) = FeatureEngineering.computeIncidenceAngle(
            azimuthDeg = obs.azimuthDegrees,
            elevationDeg = obs.elevationDegrees,
            yawDeg = targetYawDeg,
            pitchDeg = 0.0, // Assumed flat phone per training and inference spec
            rollDeg = 0.0
        )

        val elRad = Math.toRadians(obs.elevationDegrees)
        val relAz = (obs.azimuthDegrees - targetYawDeg)
        val relAzRad = Math.toRadians(relAz)

        val elSin = sin(elRad)
        val elCos = cos(elRad)
        val isAboveHorizon = if (obs.elevationDegrees > 0) 1.0 else 0.0
        val relAzSin = sin(relAzRad)
        val relAzCos = cos(relAzRad)

        val airmassSimple = FeatureEngineering.airmassSimple(obs.elevationDegrees)
        val airmassKy = FeatureEngineering.airmassKY(obs.elevationDegrees)

        val freqGhz = obs.signalKey.carrierFrequencyHz / 1e9
        val freqLossDb = FeatureEngineering.freqLossDb(obs.signalKey.carrierFrequencyHz.toDouble())

        val ctype = obs.signalKey.constellationType
        val isGps = if (ctype == 1) 1.0 else 0.0
        val isSbas = if (ctype == 2) 1.0 else 0.0
        val isGlonass = if (ctype == 3) 1.0 else 0.0
        val isQzss = if (ctype == 4) 1.0 else 0.0
        val isBeidou = if (ctype == 5) 1.0 else 0.0
        val isGalileo = if (ctype == 6) 1.0 else 0.0
        val isNavic = if (ctype == 7) 1.0 else 0.0

        val relAzXEl = relAzCos * elSin
        val cosIncXAirmass = cosInc * airmassKy
        val navicXAirmass = isNavic * airmassKy
        val gpsXAirmass = isGps * airmassKy
        val galileoXAirmass = isGalileo * airmassKy
        val freqLossXElev = freqLossDb * elSin

        // Order matching ModelArtifacts.ORIENTATION_FEATURES (17 features)
        out[0] = cosInc
        out[1] = elSin
        out[2] = isAboveHorizon
        out[3] = relAzSin
        out[4] = relAzCos
        out[5] = airmassKy
        out[6] = freqLossDb
        out[7] = isGps
        out[8] = isSbas
        out[9] = isGlonass
        out[10] = isQzss
        out[11] = isBeidou
        out[12] = isGalileo
        out[13] = isNavic
        out[14] = relAzXEl
        out[15] = cosIncXAirmass
        out[16] = freqLossXElev
    }
}

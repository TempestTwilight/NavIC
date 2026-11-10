package org.isro.gnssoptimizer.gnss

/**
 * Key identifying a unique physical carrier signal track.
 */
data class SignalKey(
    val constellationType: Int,
    val svid: Int,
    val carrierFrequencyHz: Long
)

/**
 * Snapshot of a satellite signal at a point in time.
 */
data class SatelliteObservation(
    val timestampMs: Long,
    val signalKey: SignalKey,
    val cn0DbHz: Double,
    val basebandCn0DbHz: Double,
    val azimuthDegrees: Double,
    val elevationDegrees: Double,
    val usedInFix: Boolean,
    val hasAlmanac: Boolean,
    val hasEphemeris: Boolean,
    val signalCount: Int
)

/**
 * Track buffer for a single satellite over a rolling observation window.
 */
class SatelliteTrack(val signalKey: SignalKey) {
    private val history = ArrayList<SatelliteObservation>(60)

    fun add(obs: SatelliteObservation) {
        history.add(obs)
        // Prune older than 35 seconds to keep memory bounded
        val cutoff = obs.timestampMs - 35_000L
        while (history.isNotEmpty() && history.first().timestampMs < cutoff) {
            history.removeAt(0)
        }
    }

    val latest: SatelliteObservation? get() = history.lastOrNull()
    val observationCount: Int get() = history.size
    val durationMs: Long get() = if (history.size >= 2) history.last().timestampMs - history.first().timestampMs else 0L

    /**
     * Find the closest observation observed approximately lagMs earlier, within toleranceMs.
     */
    fun getLaggedObservation(targetTimeMs: Long, lagMs: Long, toleranceMs: Long = 300L): SatelliteObservation? {
        val queryTime = targetTimeMs - lagMs
        var closest: SatelliteObservation? = null
        var minDiff = Long.MAX_VALUE

        for (obs in history) {
            val diff = kotlin.math.abs(obs.timestampMs - queryTime)
            if (diff <= toleranceMs && diff < minDiff) {
                minDiff = diff
                closest = obs
            }
        }
        return closest
    }
}

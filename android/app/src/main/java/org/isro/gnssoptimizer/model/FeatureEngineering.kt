package org.isro.gnssoptimizer.model

import kotlin.math.*

object FeatureEngineering {

    /**
     * Smallest signed angular difference (a - b) in [-180, 180] degrees.
     * Matches Python: ((a - b + 180) % 360) - 180
     */
    fun circularDeltaDeg(a: Double, b: Double): Double {
        val diff = (a - b + 180.0)
        val modulo = ((diff % 360.0) + 360.0) % 360.0
        return modulo - 180.0
    }

    /**
     * Standard simple secant-like airmass: 1 / sin(clamp(el, 1°, 89°))
     */
    fun airmassSimple(elevationDeg: Double): Double {
        val elClipped = elevationDeg.coerceIn(1.0, 89.0)
        val elRad = Math.toRadians(elClipped)
        return 1.0 / sin(elRad)
    }

    /**
     * Kasten-Young (1989) tropospheric airmass model:
     * 1 / (sin(el) + 0.15 * (clamp(el, 0°, 90°) + 3.885)^(-1.25))
     */
    fun airmassKY(elevationDeg: Double): Double {
        val elRad = Math.toRadians(elevationDeg)
        val elClipped = elevationDeg.coerceIn(0.0, 90.0)
        val term = (elClipped + 3.885).pow(-1.25)
        val denom = sin(elRad) + 0.15 * term
        return 1.0 / denom.coerceAtLeast(1e-6)
    }

    /**
     * Carrier frequency relative path loss in dB compared to GPS L1 (1575.42 MHz):
     * 20 * log10(clamp(freq, 1e8, 1e11) / 1575420000.0)
     */
    fun freqLossDb(carrierFreqHz: Double): Double {
        val freqClipped = carrierFreqHz.coerceIn(1e8, 1e11)
        return 20.0 * log10(freqClipped / 1575420000.0)
    }

    /**
     * Line-of-sight unit vector and Antenna normal dot product (cos_inc).
     * ENU coordinates.
     */
    fun computeIncidenceAngle(
        azimuthDeg: Double,
        elevationDeg: Double,
        yawDeg: Double,
        pitchDeg: Double,
        rollDeg: Double
    ): Pair<Double, Double> {
        val az = Math.toRadians(azimuthDeg)
        val el = Math.toRadians(elevationDeg)
        val yaw = Math.toRadians(yawDeg)
        val pitch = Math.toRadians(pitchDeg)
        val roll = Math.toRadians(rollDeg)

        // Satellite LOS vector in ENU
        val sEast = cos(el) * sin(az)
        val sNorth = cos(el) * cos(az)
        val sUp = sin(el)

        // Phone antenna normal vector in ENU (from body [0, 0, 1] rotated by roll, pitch, yaw)
        val nEast = sin(roll) * sin(yaw) + cos(roll) * sin(pitch) * cos(yaw)
        val nNorth = -sin(roll) * cos(yaw) + cos(roll) * sin(pitch) * sin(yaw)
        val nUp = cos(roll) * cos(pitch)

        val cosInc = (sEast * nEast + sNorth * nNorth + sUp * nUp).coerceIn(-1.0, 1.0)
        val incDeg = Math.toDegrees(acos(cosInc))
        return Pair(cosInc, incDeg)
    }
}

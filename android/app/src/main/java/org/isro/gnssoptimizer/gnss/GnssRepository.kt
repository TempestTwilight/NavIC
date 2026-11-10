package org.isro.gnssoptimizer.gnss

import android.annotation.SuppressLint
import android.content.Context
import android.location.*
import android.os.Build
import android.os.Handler
import android.os.Looper
import android.util.Log
import java.util.concurrent.ConcurrentHashMap

class GnssRepository(private val context: Context) {

    private val locationManager = context.getSystemService(Context.LOCATION_SERVICE) as LocationManager
    private val mainHandler = Handler(Looper.getMainLooper())

    // Satellite ephemeris and geometry cache from GnssStatus
    data class StatusInfo(
        val azimuthDeg: Double,
        val elevationDeg: Double,
        val usedInFix: Boolean,
        val hasAlmanac: Boolean,
        val hasEphemeris: Boolean
    )

    private val statusCache = ConcurrentHashMap<Pair<Int, Int>, StatusInfo>()
    val tracks = ConcurrentHashMap<SignalKey, SatelliteTrack>()

    @Volatile var totalSatellitesInView: Int = 0
    @Volatile var totalSatellitesUsedInFix: Int = 0
    @Volatile var totalSatellitesWithMeasurements: Int = 0
    @Volatile var isGnssActive: Boolean = false

    var onEpochReceived: ((List<SatelliteObservation>) -> Unit)? = null
    var onStatusChanged: ((visibleCount: Int, inFixCount: Int) -> Unit)? = null

    private val gnssStatusCallback = object : GnssStatus.Callback() {
        override fun onSatelliteStatusChanged(status: GnssStatus) {
            val count = status.satelliteCount
            var inFix = 0
            for (i in 0 until count) {
                val constellation = status.getConstellationType(i)
                val svid = status.getSvid(i)
                val az = status.getAzimuthDegrees(i).toDouble()
                val el = status.getElevationDegrees(i).toDouble()
                val used = status.usedInFix(i)
                if (used) inFix++

                statusCache[Pair(constellation, svid)] = StatusInfo(
                    azimuthDeg = az,
                    elevationDeg = el,
                    usedInFix = used,
                    hasAlmanac = status.hasAlmanacData(i),
                    hasEphemeris = status.hasEphemerisData(i)
                )
            }
            totalSatellitesInView = count
            totalSatellitesUsedInFix = inFix
            onStatusChanged?.invoke(count, inFix)
        }
    }

    private val measurementsCallback = object : GnssMeasurementsEvent.Callback() {
        override fun onGnssMeasurementsReceived(event: GnssMeasurementsEvent) {
            val nowMs = System.currentTimeMillis()
            val epochObs = ArrayList<SatelliteObservation>()
            val rawMeasurements = event.measurements

            var validMeasCount = 0
            for (meas in rawMeasurements) {
                val cn0 = meas.cn0DbHz
                if (cn0 <= 0.0 || cn0.isNaN() || cn0.isInfinite()) continue

                val constellation = meas.constellationType
                val svid = meas.svid
                val freqHz = if (meas.hasCarrierFrequencyHz()) {
                    meas.carrierFrequencyHz.toLong()
                } else {
                    1575420000L // Default L1 frequency fallback
                }

                val status = statusCache[Pair(constellation, svid)] ?: StatusInfo(
                    azimuthDeg = 0.0,
                    elevationDeg = 45.0, // default if ephemeris not yet parsed
                    usedInFix = false,
                    hasAlmanac = false,
                    hasEphemeris = false
                )

                val basebandCn0 = if (meas.hasBasebandCn0DbHz()) {
                    meas.basebandCn0DbHz
                } else {
                    cn0 // fallback to antenna C/N0
                }

                val key = SignalKey(constellation, svid, freqHz)
                val obs = SatelliteObservation(
                    timestampMs = nowMs,
                    signalKey = key,
                    cn0DbHz = cn0,
                    basebandCn0DbHz = basebandCn0,
                    azimuthDegrees = status.azimuthDeg,
                    elevationDegrees = status.elevationDeg,
                    usedInFix = status.usedInFix,
                    hasAlmanac = status.hasAlmanac,
                    hasEphemeris = status.hasEphemeris,
                    signalCount = rawMeasurements.size
                )

                val track = tracks.getOrPut(key) { SatelliteTrack(key) }
                track.add(obs)
                epochObs.add(obs)
                validMeasCount++
            }

            totalSatellitesWithMeasurements = validMeasCount
            if (epochObs.isNotEmpty()) {
                onEpochReceived?.invoke(epochObs)
            }
        }
    }

    private val locationListener = LocationListener { /* Keep location provider active */ }

    @SuppressLint("MissingPermission")
    fun start() {
        if (isGnssActive) return
        try {
            locationManager.registerGnssStatusCallback(gnssStatusCallback, mainHandler)
            locationManager.registerGnssMeasurementsCallback(measurementsCallback, mainHandler)

            // Request location updates to actively spin up the GNSS receiver
            if (locationManager.isProviderEnabled(LocationManager.GPS_PROVIDER)) {
                locationManager.requestLocationUpdates(
                    LocationManager.GPS_PROVIDER,
                    1000L,
                    0f,
                    locationListener,
                    Looper.getMainLooper()
                )
            }
            isGnssActive = true
            Log.i("GnssRepository", "GNSS callbacks registered successfully.")
        } catch (e: Exception) {
            Log.e("GnssRepository", "Failed to start GNSS collection", e)
        }
    }

    fun stop() {
        if (!isGnssActive) return
        try {
            locationManager.unregisterGnssStatusCallback(gnssStatusCallback)
            locationManager.unregisterGnssMeasurementsCallback(measurementsCallback)
            locationManager.removeUpdates(locationListener)
            isGnssActive = false
            Log.i("GnssRepository", "GNSS callbacks stopped.")
        } catch (e: Exception) {
            Log.e("GnssRepository", "Failed to stop GNSS collection", e)
        }
    }
}

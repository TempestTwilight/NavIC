package org.isro.gnssoptimizer.sensor

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.util.Log

data class OrientationData(
    val azimuthDeg: Double,
    val pitchDeg: Double,
    val rollDeg: Double,
    val accuracy: Int,
    val timestampNs: Long
)

class OrientationTracker(context: Context) : SensorEventListener {

    private val sensorManager = context.getSystemService(Context.SENSOR_SERVICE) as SensorManager
    private val rotationSensor: Sensor? = sensorManager.getDefaultSensor(Sensor.TYPE_ROTATION_VECTOR)

    private val rotationMatrix = FloatArray(9)
    private val orientationAngles = FloatArray(3)

    @Volatile var currentOrientation: OrientationData = OrientationData(0.0, 0.0, 0.0, 3, 0L)
        private set

    @Volatile var isTracking: Boolean = false
        private set

    var onOrientationChanged: ((OrientationData) -> Unit)? = null

    fun start() {
        if (isTracking) return
        if (rotationSensor != null) {
            sensorManager.registerListener(this, rotationSensor, SensorManager.SENSOR_DELAY_UI)
            isTracking = true
            Log.i("OrientationTracker", "Sensor listener registered.")
        } else {
            Log.w("OrientationTracker", "TYPE_ROTATION_VECTOR sensor not available.")
        }
    }

    fun stop() {
        if (!isTracking) return
        sensorManager.unregisterListener(this)
        isTracking = false
        Log.i("OrientationTracker", "Sensor listener unregistered.")
    }

    override fun onSensorChanged(event: SensorEvent) {
        if (event.sensor.type == Sensor.TYPE_ROTATION_VECTOR) {
            SensorManager.getRotationMatrixFromVector(rotationMatrix, event.values)
            SensorManager.getOrientation(rotationMatrix, orientationAngles)

            // orientationAngles[0] = azimuth in radians (-pi to pi)
            // orientationAngles[1] = pitch in radians (-pi/2 to pi/2)
            // orientationAngles[2] = roll in radians (-pi to pi)
            val azDegRaw = Math.toDegrees(orientationAngles[0].toDouble())
            val pitchDeg = Math.toDegrees(orientationAngles[1].toDouble())
            val rollDeg = Math.toDegrees(orientationAngles[2].toDouble())

            // Normalize azimuth to [0, 360) matching OrientationDeg.csv
            val azimuthDeg = ((azDegRaw % 360.0) + 360.0) % 360.0

            currentOrientation = OrientationData(
                azimuthDeg = azimuthDeg,
                pitchDeg = pitchDeg,
                rollDeg = rollDeg,
                accuracy = event.accuracy,
                timestampNs = event.timestamp
            )
            onOrientationChanged?.invoke(currentOrientation)
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {
        // Updated on sensor changed
    }
}

package org.isro.gnssoptimizer.ui

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import org.isro.gnssoptimizer.gnss.GnssRepository
import org.isro.gnssoptimizer.pipeline.OptimizerPipeline
import org.isro.gnssoptimizer.pipeline.OptimizerUiState
import org.isro.gnssoptimizer.pipeline.PipelineStatus
import org.isro.gnssoptimizer.sensor.OrientationTracker

class OptimizerViewModel(application: Application) : AndroidViewModel(application) {

    private val gnssRepository = GnssRepository(application)
    private val orientationTracker = OrientationTracker(application)
    private val pipeline = OptimizerPipeline(gnssRepository)

    private val _uiState = MutableStateFlow(OptimizerUiState())
    val uiState: StateFlow<OptimizerUiState> = _uiState.asStateFlow()

    init {
        orientationTracker.onOrientationChanged = { orient ->
            _uiState.value = _uiState.value.copy(
                azimuthDeg = orient.azimuthDeg,
                pitchDeg = orient.pitchDeg,
                rollDeg = orient.rollDeg
            )
        }

        gnssRepository.onStatusChanged = { visible, inFix ->
            _uiState.value = _uiState.value.copy(
                satellitesVisible = visible,
                satellitesUsedInFix = inFix,
                statusMessage = if (_uiState.value.status == PipelineStatus.INITIALIZING) {
                    "Tracking $visible satellites ($inFix in fix). Buffering..."
                } else _uiState.value.statusMessage
            )
        }

        gnssRepository.onEpochReceived = { epochObs ->
            viewModelScope.launch(Dispatchers.Default) {
                val orient = orientationTracker.currentOrientation
                val newState = pipeline.processEpoch(epochObs, orient)
                _uiState.value = newState
            }
        }
    }

    fun startTracking() {
        orientationTracker.start()
        gnssRepository.start()
    }

    fun stopTracking() {
        gnssRepository.stop()
        orientationTracker.stop()
    }

    fun onPermissionDenied() {
        _uiState.value = _uiState.value.copy(
            status = PipelineStatus.PERMISSION_REQUIRED,
            statusMessage = "Fine Location permission is required to read GNSS satellite signals."
        )
    }

    override fun onCleared() {
        super.onCleared()
        stopTracking()
    }
}

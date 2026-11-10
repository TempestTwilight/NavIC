package org.isro.gnssoptimizer.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import org.isro.gnssoptimizer.model.RidgeModel
import org.isro.gnssoptimizer.pipeline.OptimizerUiState
import org.isro.gnssoptimizer.pipeline.PipelineStatus
import org.isro.gnssoptimizer.pipeline.SignalQuality
import kotlin.math.roundToInt

@Composable
fun OptimizerDashboard(
    state: OptimizerUiState,
    modifier: Modifier = Modifier
) {
    Scaffold(
        modifier = modifier.fillMaxSize(),
        containerColor = BgDark
    ) { innerPadding ->
        LazyColumn(
            modifier = Modifier
                .fillMaxSize()
                .padding(innerPadding)
                .padding(horizontal = 16.dp, vertical = 12.dp),
            verticalArrangement = Arrangement.spacedBy(14.dp)
        ) {
            // ================= HEADER =================
            item {
                HeaderSection(state)
            }

            // ================= STATUS BANNER =================
            item {
                StatusBanner(state)
            }

            // ================= SECTION 1: CURRENT CONDITIONS =================
            item {
                SectionTitle("CURRENT CONDITIONS")
            }
            item {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(10.dp)
                ) {
                    OrientationCard(
                        modifier = Modifier.weight(1f),
                        azimuth = state.azimuthDeg,
                        pitch = state.pitchDeg,
                        roll = state.rollDeg
                    )
                    SignalStrengthCard(
                        modifier = Modifier.weight(1f),
                        cn0 = state.currentMeanCn0,
                        quality = state.signalQuality
                    )
                }
            }
            item {
                SatelliteStatusCard(
                    visible = state.satellitesVisible,
                    measured = state.satellitesWithMeasurements,
                    inFix = state.satellitesUsedInFix
                )
            }

            // ================= SECTION 2: MODEL PREDICTIONS =================
            item {
                SectionTitle("MODEL PREDICTIONS")
            }
            item {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(10.dp)
                ) {
                    TemporalPredictionCard(
                        modifier = Modifier.weight(1f),
                        state = state
                    )
                    OrientationModelCard(
                        modifier = Modifier.weight(1f),
                        state = state
                    )
                }
            }

            // ================= SECTION 3: RECOMMENDATIONS =================
            item {
                SectionTitle("RECOMMENDATIONS")
            }
            item {
                RecommendationCard(state = state)
            }
            item {
                ExpectedSignalChangeCard(state = state)
            }

            // ================= SECTION 4: MODEL EXPLANATIONS =================
            item {
                SectionTitle("FEATURE CONTRIBUTIONS (TOP 3)")
            }
            item {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(10.dp)
                ) {
                    FeatureContributionCard(
                        modifier = Modifier.weight(1f),
                        title = "Temporal Model",
                        subtitle = "Predicting t+1s",
                        features = state.temporalTopFeatures
                    )
                    FeatureContributionCard(
                        modifier = Modifier.weight(1f),
                        title = "Orientation Model",
                        subtitle = "Physics Interaction",
                        features = state.orientationTopFeatures
                    )
                }
            }

            item {
                Spacer(modifier = Modifier.height(24.dp))
            }
        }
    }
}

@Composable
fun HeaderSection(state: OptimizerUiState) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically
    ) {
        Column {
            Text(
                text = "ISRO GNSS OPTIMIZER",
                color = CyanAccent,
                fontSize = 18.sp,
                fontWeight = FontWeight.Bold,
                letterSpacing = 1.sp
            )
            Text(
                text = "On-Device Ridge Inference Engine",
                color = TextSecondary,
                fontSize = 12.sp
            )
        }
        val (badgeBg, badgeText, badgeColor) = when (state.status) {
            PipelineStatus.PERMISSION_REQUIRED -> Triple(RedAlert.copy(alpha = 0.2f), "NO PERM", RedAlert)
            PipelineStatus.INITIALIZING -> Triple(YellowWarning.copy(alpha = 0.2f), "BUFFERING", YellowWarning)
            PipelineStatus.INSUFFICIENT_DATA -> Triple(YellowWarning.copy(alpha = 0.2f), "NO GNSS", YellowWarning)
            PipelineStatus.MONITORING -> Triple(GreenSignal.copy(alpha = 0.2f), "TRACKING", GreenSignal)
            PipelineStatus.DEGRADATION_PREDICTED -> Triple(RedAlert.copy(alpha = 0.2f), "ALERT", RedAlert)
            PipelineStatus.RECOMMENDATION_READY -> Triple(CyanAccent.copy(alpha = 0.2f), "OPTIMIZED", CyanAccent)
        }
        Surface(
            color = badgeBg,
            shape = RoundedCornerShape(6.dp),
            border = androidx.compose.foundation.BorderStroke(1.dp, badgeColor.copy(alpha = 0.4f))
        ) {
            Text(
                text = badgeText,
                color = badgeColor,
                fontSize = 11.sp,
                fontWeight = FontWeight.Bold,
                modifier = Modifier.padding(horizontal = 8.dp, vertical = 4.dp)
            )
        }
    }
}

@Composable
fun StatusBanner(state: OptimizerUiState) {
    Surface(
        color = CardBg,
        shape = RoundedCornerShape(8.dp),
        border = androidx.compose.foundation.BorderStroke(1.dp, CardBorder),
        modifier = Modifier.fillMaxWidth()
    ) {
        Row(
            modifier = Modifier.padding(12.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Column(modifier = Modifier.weight(1f)) {
                Text(
                    text = state.statusMessage,
                    color = TextPrimary,
                    fontSize = 13.sp,
                    fontWeight = FontWeight.Medium
                )
                if (state.status == PipelineStatus.INITIALIZING) {
                    LinearProgressIndicator(
                        progress = { state.bufferProgressSeconds / 30f },
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(top = 8.dp)
                            .height(4.dp),
                        color = CyanAccent,
                        trackColor = MutedBg
                    )
                }
            }
        }
    }
}

@Composable
fun SectionTitle(title: String) {
    Text(
        text = title,
        color = TextSecondary,
        fontSize = 11.sp,
        fontWeight = FontWeight.Bold,
        letterSpacing = 1.2.sp,
        modifier = Modifier.padding(top = 6.dp)
    )
}

// ================= CARD 1: CURRENT ORIENTATION =================
@Composable
fun OrientationCard(
    modifier: Modifier = Modifier,
    azimuth: Double,
    pitch: Double,
    roll: Double
) {
    DashboardCard(modifier = modifier, title = "Current Orientation") {
        Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
            OrientationMetric("YAW", "${azimuth.roundToInt()}°")
            OrientationMetric("PITCH", "${pitch.roundToInt()}°")
            OrientationMetric("ROLL", "${roll.roundToInt()}°")
        }
    }
}

@Composable
fun OrientationMetric(label: String, value: String) {
    Column(horizontalAlignment = Alignment.CenterHorizontally) {
        Text(text = label, color = TextSecondary, fontSize = 10.sp, fontWeight = FontWeight.SemiBold)
        Text(
            text = value,
            color = TextPrimary,
            fontSize = 16.sp,
            fontWeight = FontWeight.Bold,
            fontFamily = FontFamily.Monospace
        )
    }
}

// ================= CARD 2: SIGNAL STRENGTH =================
@Composable
fun SignalStrengthCard(
    modifier: Modifier = Modifier,
    cn0: Double,
    quality: SignalQuality
) {
    val (color, label) = when (quality) {
        SignalQuality.EXCELLENT -> Pair(GreenSignal, "Excellent")
        SignalQuality.GOOD -> Pair(GreenSignal, "Good")
        SignalQuality.FAIR -> Pair(YellowWarning, "Fair")
        SignalQuality.POOR -> Pair(RedAlert, "Poor")
    }

    DashboardCard(modifier = modifier, title = "Signal Strength") {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.Bottom
        ) {
            Column {
                Text(
                    text = if (cn0 > 0) String.format("%.1f", cn0) else "--",
                    color = color,
                    fontSize = 24.sp,
                    fontWeight = FontWeight.Bold,
                    fontFamily = FontFamily.Monospace
                )
                Text(text = "dB-Hz (Mean)", color = TextSecondary, fontSize = 10.sp)
            }
            Surface(
                color = color.copy(alpha = 0.15f),
                shape = RoundedCornerShape(4.dp)
            ) {
                Text(
                    text = label,
                    color = color,
                    fontSize = 10.sp,
                    fontWeight = FontWeight.Bold,
                    modifier = Modifier.padding(horizontal = 6.dp, vertical = 2.dp)
                )
            }
        }
    }
}

// ================= CARD 3: SATELLITE STATUS =================
@Composable
fun SatelliteStatusCard(visible: Int, measured: Int, inFix: Int) {
    DashboardCard(modifier = Modifier.fillMaxWidth(), title = "Satellite Status") {
        Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceAround) {
            SatelliteMetric("VISIBLE", visible.toString())
            SatelliteMetric("MEASURED", measured.toString(), CyanAccent)
            SatelliteMetric("IN FIX", inFix.toString(), GreenSignal)
        }
    }
}

@Composable
fun SatelliteMetric(label: String, count: String, highlightColor: Color = TextPrimary) {
    Column(horizontalAlignment = Alignment.CenterHorizontally) {
        Text(text = label, color = TextSecondary, fontSize = 10.sp, fontWeight = FontWeight.SemiBold)
        Text(
            text = count,
            color = highlightColor,
            fontSize = 18.sp,
            fontWeight = FontWeight.Bold,
            fontFamily = FontFamily.Monospace
        )
    }
}

// ================= CARD 4: TEMPORAL PREDICTION =================
@Composable
fun TemporalPredictionCard(modifier: Modifier = Modifier, state: OptimizerUiState) {
    val delta = state.predictedDeltaTemporal
    val deltaColor = if (delta >= 0) GreenSignal else if (delta > -1.5) YellowWarning else RedAlert
    val deltaSign = if (delta >= 0) "+" else ""

    DashboardCard(modifier = modifier, title = "Temporal (t+1s)") {
        Text(
            text = if (state.bufferProgressSeconds >= 30) String.format("%.1f dB-Hz", state.predictedCn0Temporal) else "Buffering...",
            color = TextPrimary,
            fontSize = 18.sp,
            fontWeight = FontWeight.Bold,
            fontFamily = FontFamily.Monospace
        )
        Spacer(modifier = Modifier.height(4.dp))
        Text(
            text = if (state.bufferProgressSeconds >= 30) "Δ $deltaSign${String.format("%.1f", delta)} dB-Hz" else "Horizon: 1000ms",
            color = deltaColor,
            fontSize = 12.sp,
            fontWeight = FontWeight.SemiBold
        )
    }
}

// ================= CARD 5: ORIENTATION MODEL =================
@Composable
fun OrientationModelCard(modifier: Modifier = Modifier, state: OptimizerUiState) {
    DashboardCard(modifier = modifier, title = "Target Heading") {
        Text(
            text = if (state.isDegradationPredicted) "${state.bestYawDeg.roundToInt()}°" else "Optimal",
            color = if (state.isDegradationPredicted) CyanAccent else GreenSignal,
            fontSize = 18.sp,
            fontWeight = FontWeight.Bold,
            fontFamily = FontFamily.Monospace
        )
        Spacer(modifier = Modifier.height(4.dp))
        Text(
            text = "Current: ${state.azimuthDeg.roundToInt()}°",
            color = TextSecondary,
            fontSize = 12.sp
        )
    }
}

// ================= CARD 6: RECOMMENDATION =================
@Composable
fun RecommendationCard(state: OptimizerUiState) {
    DashboardCard(modifier = Modifier.fillMaxWidth(), title = "Orientation Recommendation") {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Column(modifier = Modifier.weight(1f)) {
                Text(
                    text = state.turnDirectionText,
                    color = if (state.isDegradationPredicted) CyanAccent else GreenSignal,
                    fontSize = 20.sp,
                    fontWeight = FontWeight.Bold
                )
                Spacer(modifier = Modifier.height(4.dp))
                Text(
                    text = state.recommendationInstruction,
                    color = TextSecondary,
                    fontSize = 12.sp
                )
            }
            Surface(
                color = MutedBg,
                shape = RoundedCornerShape(8.dp),
                border = androidx.compose.foundation.BorderStroke(1.dp, CardBorder)
            ) {
                Text(
                    text = state.turnDegreesText,
                    color = if (state.deltaYawDeg != 0.0) CyanAccent else TextSecondary,
                    fontSize = 18.sp,
                    fontWeight = FontWeight.Bold,
                    fontFamily = FontFamily.Monospace,
                    modifier = Modifier.padding(horizontal = 12.dp, vertical = 8.dp)
                )
            }
        }
    }
}

// ================= CARD 7: EXPECTED SIGNAL CHANGE =================
@Composable
fun ExpectedSignalChangeCard(state: OptimizerUiState) {
    val gain = state.predictedImprovementDb
    val gainColor = if (gain > 0) GreenSignal else TextSecondary

    DashboardCard(modifier = Modifier.fillMaxWidth(), title = "Expected Signal Change") {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Column {
                Text(text = "CURRENT MEASURED", color = TextSecondary, fontSize = 10.sp)
                Text(
                    text = "${String.format("%.1f", state.currentMeanCn0)} dB-Hz",
                    color = TextPrimary,
                    fontSize = 15.sp,
                    fontWeight = FontWeight.Bold,
                    fontFamily = FontFamily.Monospace
                )
            }
            Text(text = "➔", color = CyanAccent, fontSize = 18.sp)
            Column {
                Text(text = "PROJECTED", color = TextSecondary, fontSize = 10.sp)
                Text(
                    text = "${String.format("%.1f", state.currentMeanCn0 + gain)} dB-Hz",
                    color = CyanAccent,
                    fontSize = 15.sp,
                    fontWeight = FontWeight.Bold,
                    fontFamily = FontFamily.Monospace
                )
            }
            Surface(
                color = gainColor.copy(alpha = 0.15f),
                shape = RoundedCornerShape(6.dp)
            ) {
                Text(
                    text = "+${String.format("%.1f", gain)} dB-Hz",
                    color = gainColor,
                    fontSize = 12.sp,
                    fontWeight = FontWeight.Bold,
                    modifier = Modifier.padding(horizontal = 8.dp, vertical = 4.dp)
                )
            }
        }
    }
}

// ================= CARD 8 & 9: FEATURE CONTRIBUTIONS =================
@Composable
fun FeatureContributionCard(
    modifier: Modifier = Modifier,
    title: String,
    subtitle: String,
    features: List<RidgeModel.FeatureContribution>
) {
    DashboardCard(modifier = modifier, title = title) {
        Text(text = subtitle, color = TextSecondary, fontSize = 10.sp)
        Spacer(modifier = Modifier.height(6.dp))

        if (features.isEmpty()) {
            Text(text = "Awaiting data...", color = TextSecondary, fontSize = 11.sp)
        } else {
            Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
                for (f in features) {
                    val sign = if (f.isPositive) "+" else "-"
                    val color = if (f.isPositive) GreenSignal else RedAlert
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween
                    ) {
                        Text(
                            text = f.name.take(12),
                            color = TextPrimary,
                            fontSize = 11.sp,
                            fontWeight = FontWeight.Medium
                        )
                        Text(
                            text = "$sign${String.format("%.2f", f.absContribution)}",
                            color = color,
                            fontSize = 11.sp,
                            fontWeight = FontWeight.Bold,
                            fontFamily = FontFamily.Monospace
                        )
                    }
                }
            }
        }
    }
}

// ================= REUSABLE CARD CONTAINER =================
@Composable
fun DashboardCard(
    modifier: Modifier = Modifier,
    title: String,
    content: @Composable ColumnScope.() -> Unit
) {
    Surface(
        modifier = modifier,
        color = CardBg,
        shape = RoundedCornerShape(8.dp),
        border = androidx.compose.foundation.BorderStroke(1.dp, CardBorder)
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(12.dp)
        ) {
            Text(
                text = title.uppercase(),
                color = TextSecondary,
                fontSize = 10.sp,
                fontWeight = FontWeight.Bold,
                letterSpacing = 0.8.sp
            )
            Spacer(modifier = Modifier.height(8.dp))
            content()
        }
    }
}

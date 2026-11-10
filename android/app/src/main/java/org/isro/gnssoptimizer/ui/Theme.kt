package org.isro.gnssoptimizer.ui

import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color

val BgDark = Color(0xFF0B0F19)
val CardBg = Color(0xFF161B26)
val CardBorder = Color(0xFF283042)
val CyanAccent = Color(0xFF38BDF8)
val GreenSignal = Color(0xFF22C55E)
val YellowWarning = Color(0xFFFACC15)
val RedAlert = Color(0xFFF87171)
val TextPrimary = Color(0xFFF8FAFC)
val TextSecondary = Color(0xFF94A3B8)
val MutedBg = Color(0xFF1E2638)

private val DarkColorScheme = darkColorScheme(
    primary = CyanAccent,
    background = BgDark,
    surface = CardBg,
    onPrimary = Color.Black,
    onBackground = TextPrimary,
    onSurface = TextPrimary
)

@Composable
fun GnssOptimizerTheme(content: @Composable () -> Unit) {
    MaterialTheme(
        colorScheme = DarkColorScheme,
        content = content
    )
}

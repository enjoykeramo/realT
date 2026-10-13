package realt.core.ml

import realt.core.indicator.Indicators
import realt.core.indicator.closes
import realt.core.model.Bar
import kotlin.math.max

/**
 * 开盘前可知的个股日线特征(只用目标交易日之前的日线),以日 ATR 为单位。
 */
object DailyFeatures {
    val names = listOf("yRetAtr", "yClosePos", "yRangeAtr", "y2RetAtr", "toMa5Atr", "toMa20Atr", "yVolRatio", "upDays5")

    fun compute(before: List<Bar>): Map<String, Double> {
        if (before.size < 25) return emptyMap()
        val n = before.size - 1
        val c = before.closes()
        val atr = max(Indicators.atr(before, 14)[n], c[n] * 0.005)
        val y = before[n]
        val ma5 = Indicators.sma(c, 5)[n]
        val ma20 = Indicators.sma(c, 20)[n]
        val vma20 = Indicators.sma(DoubleArray(before.size) { before[it].volume }, 20)[n]
        val range = max(y.high - y.low, 1e-9)
        return mapOf(
            "yRetAtr" to (c[n] - c[n - 1]) / atr,
            "yClosePos" to (y.close - y.low) / range,
            "yRangeAtr" to range / atr,
            "y2RetAtr" to (c[n - 1] - c[n - 2]) / atr,
            "toMa5Atr" to (c[n] - ma5) / atr,
            "toMa20Atr" to (c[n] - ma20) / atr,
            "yVolRatio" to if (vma20 > 0) y.volume / vma20 else 1.0,
            "upDays5" to (n - 4..n).count { c[it] > c[it - 1] }.toDouble(),
        )
    }
}

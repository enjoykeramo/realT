package realt.core.regime

import realt.core.indicator.Indicators
import realt.core.indicator.closes
import realt.core.model.Bar

/** L1:大盘环境五档 */
enum class MarketLevel(val label: String, val bias: Double) {
    BULL("主升", 1.0),
    STRONG("偏强震荡", 0.5),
    NEUTRAL("震荡", 0.0),
    WEAK("偏弱震荡", -0.5),
    BEAR("主跌", -1.0),
}

/** 某日全市场宽度统计(由全部 A 股日线计算) */
data class Breadth(
    val date: Int,
    val up: Int,
    val down: Int,
    val limitUp: Int,
    val limitDown: Int,
    val total: Int,
) {
    val upRatio: Double get() = if (total > 0) up.toDouble() / total else 0.5
}

data class MarketRegime(
    val level: MarketLevel,
    /** -1..1 综合分 */
    val score: Double,
    val detail: String,
) {
    companion object {
        val UNKNOWN = MarketRegime(MarketLevel.NEUTRAL, 0.0, "无大盘数据")
    }
}

/**
 * 大盘环境判定。只使用目标交易日"之前"的日线,保证盘前即可得出结论。
 *
 * 分项(各自归一到 -1..1 后加权):
 *  - 趋势:收盘相对 MA20/MA60、MA20 斜率
 *  - 动量:20 日涨幅
 *  - 位置:60 日区间位置
 *  - 宽度:近 5 日上涨家数占比、涨停-跌停差(有宽度数据时)
 */
class MarketRegimeDetector {

    fun detect(index: List<Bar>, breadth: List<Breadth> = emptyList()): MarketRegime {
        if (index.size < 60) return MarketRegime.UNKNOWN
        val c = index.closes()
        val n = c.size - 1
        val ma20 = Indicators.sma(c, 20)
        val ma60 = Indicators.sma(c, 60)
        val last = c[n]

        val trend = (sign(last - ma20[n]) * 0.4 + sign(last - ma60[n]) * 0.3 +
            sign(ma20[n] - ma20[n - 5]) * 0.3)
        val ret20 = last / c[n - 20] - 1
        val momentum = (ret20 / 0.08).coerceIn(-1.0, 1.0)
        var hi = Double.MIN_VALUE; var lo = Double.MAX_VALUE
        for (i in n - 59..n) { hi = maxOf(hi, index[i].high); lo = minOf(lo, index[i].low) }
        val pos = if (hi > lo) (last - lo) / (hi - lo) * 2 - 1 else 0.0

        var breadthScore = 0.0
        var hasBreadth = false
        if (breadth.size >= 5) {
            hasBreadth = true
            val recent = breadth.takeLast(5)
            val upr = recent.map { it.upRatio }.average()
            val lim = recent.sumOf { it.limitUp - it.limitDown }.toDouble() / recent.size
            breadthScore = ((upr - 0.5) / 0.15).coerceIn(-1.0, 1.0) * 0.6 +
                (lim / 60.0).coerceIn(-1.0, 1.0) * 0.4
        }

        val score = if (hasBreadth)
            trend * 0.35 + momentum * 0.25 + pos * 0.15 + breadthScore * 0.25
        else
            trend * 0.45 + momentum * 0.35 + pos * 0.20

        val level = when {
            score >= 0.55 -> MarketLevel.BULL
            score >= 0.2 -> MarketLevel.STRONG
            score > -0.2 -> MarketLevel.NEUTRAL
            score > -0.55 -> MarketLevel.WEAK
            else -> MarketLevel.BEAR
        }
        val detail = "趋势%.2f 动量%.2f(20日%+.1f%%) 位置%.2f%s".format(
            trend, momentum, ret20 * 100, pos,
            if (hasBreadth) " 宽度%.2f".format(breadthScore) else ""
        )
        return MarketRegime(level, score, detail)
    }

    private fun sign(x: Double) = when {
        x > 0 -> 1.0
        x < 0 -> -1.0
        else -> 0.0
    }
}

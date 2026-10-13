package realt.core.stage

import realt.core.indicator.Indicators
import realt.core.indicator.closes
import realt.core.indicator.volumes
import realt.core.model.Bar
import kotlin.math.exp
import kotlin.math.max

/** L2:主力运作阶段 */
enum class Stage(val label: String, val bias: Double) {
    ACCUMULATION("建仓", 0.1),
    MARKUP("拉升", 0.7),
    SHAKEOUT("洗盘", 0.0),
    MAIN_RISE("主升", 0.9),
    DISTRIBUTION("派发", -0.8),
    DECLINE("下跌", -0.7),
}

data class StageEstimate(
    val probs: Map<Stage, Double>,
    val features: Map<String, Double>,
) {
    val top: Stage get() = probs.maxByOrNull { it.value }!!.key

    /** 阶段带来的方向偏置 -1..1(按概率加权) */
    val bias: Double get() = probs.entries.sumOf { it.key.bias * it.value }

    fun describe(): String =
        probs.entries.sortedByDescending { it.value }.take(3)
            .joinToString(" ") { "%s%.0f%%".format(it.key.label, it.value * 100) }

    companion object {
        val UNKNOWN = StageEstimate(Stage.entries.associateWith { 1.0 / Stage.entries.size }, emptyMap())
    }
}

/**
 * 基于日线量价的主力阶段识别(规则打分 + softmax 得到概率)。
 * 只使用目标交易日之前的日线。
 *
 * 主要特征:
 *  - pos120:120 日区间位置
 *  - ret20 / ret60:20、60 日涨幅
 *  - dd20:距 20 日最高点回撤
 *  - vr:5 日均量 / 60 日均量
 *  - bw:布林带宽在 120 日中的百分位(低 = 收敛)
 *  - maUp:均线多头排列程度(0..1)
 *  - obvDiv:OBV 斜率与价格斜率的背离(+ 表示量领先价,吸筹迹象)
 *  - upperShadow:近 10 日平均上影线比例(派发迹象)
 */
class StageDetector(private val temperature: Double = 0.6) {

    fun detect(daily: List<Bar>): StageEstimate {
        if (daily.size < 130) return StageEstimate.UNKNOWN
        val c = daily.closes()
        val v = daily.volumes()
        val n = c.size - 1
        val last = c[n]

        var hi120 = Double.MIN_VALUE; var lo120 = Double.MAX_VALUE
        for (i in n - 119..n) { hi120 = max(hi120, daily[i].high); lo120 = minOf(lo120, daily[i].low) }
        val pos120 = if (hi120 > lo120) (last - lo120) / (hi120 - lo120) else 0.5
        val ret20 = last / c[n - 20] - 1
        val ret60 = last / c[n - 60] - 1
        var hi20 = Double.MIN_VALUE
        for (i in n - 19..n) hi20 = max(hi20, daily[i].high)
        val dd20 = last / hi20 - 1
        val vma5 = Indicators.sma(v, 5)[n]
        val vma60 = Indicators.sma(v, 60)[n]
        val vr = if (vma60 > 0) vma5 / vma60 else 1.0
        val bw = Indicators.boll(c).width
        val bwPct = Indicators.percentile(bw, n - 119, n, bw[n])
        val ma5 = Indicators.sma(c, 5)[n]
        val ma10 = Indicators.sma(c, 10)[n]
        val ma20 = Indicators.sma(c, 20)[n]
        val ma60 = Indicators.sma(c, 60)[n]
        val maUp = listOf(last > ma5, ma5 > ma10, ma10 > ma20, ma20 > ma60).count { it } / 4.0
        val obv = Indicators.obv(daily)
        val obvSlope = Indicators.slope(obv, n - 29, n) / (vma60.coerceAtLeast(1.0))
        val pxSlope = Indicators.slope(c, n - 29, n) / last * 30
        val obvDiv = (obvSlope * 0.5 - pxSlope * 5).coerceIn(-1.0, 1.0)
        var us = 0.0
        for (i in n - 9..n) {
            val b = daily[i]
            val r = b.high - b.low
            if (r > 0) us += (b.high - maxOf(b.open, b.close)) / r
        }
        val upperShadow = us / 10

        val s = HashMap<Stage, Double>()
        s[Stage.ACCUMULATION] =
            tri(pos120, 0.0, 0.2, 0.45) * 1.2 + tri(ret20, -0.08, 0.0, 0.08) +
                (1 - bwPct) * 0.8 + obvDiv.coerceAtLeast(0.0) * 0.8 + tri(vr, 0.5, 0.9, 1.4) * 0.4
        s[Stage.MARKUP] =
            tri(ret20, 0.03, 0.12, 0.30) * 1.2 + tri(vr, 1.0, 1.6, 3.5) + maUp * 0.8 +
                tri(pos120, 0.4, 0.75, 1.0) * 0.6 + tri(ret60, -0.1, 0.05, 0.25) * 0.4
        s[Stage.SHAKEOUT] =
            tri(ret60, 0.08, 0.25, 0.8) * 1.0 + tri(dd20, -0.18, -0.08, -0.02) * 1.2 +
                tri(vr, 0.3, 0.7, 1.0) * 0.9 + (if (last > ma60) 0.5 else 0.0)
        s[Stage.MAIN_RISE] =
            tri(ret20, 0.10, 0.30, 0.8) * 1.2 + maUp * 1.0 + tri(pos120, 0.75, 1.0, 1.01) * 0.9 +
                tri(dd20, -0.06, 0.0, 0.01) * 0.5
        s[Stage.DISTRIBUTION] =
            tri(pos120, 0.65, 0.9, 1.01) * 0.9 + tri(ret20, -0.06, 0.0, 0.05) * 0.6 +
                tri(vr, 1.0, 1.5, 3.0) * 0.6 + (-obvDiv).coerceAtLeast(0.0) * 0.6 +
                tri(upperShadow, 0.15, 0.35, 0.8) * 0.7 + tri(ret60, 0.15, 0.4, 1.5) * 0.4
        s[Stage.DECLINE] =
            tri(ret20, -0.35, -0.12, -0.02) * 1.2 + (1 - maUp) * 0.8 +
                tri(pos120, -0.01, 0.15, 0.5) * 0.5 + (if (last < ma60) 0.5 else 0.0)

        val mx = s.values.max()
        val ex = s.mapValues { exp((it.value - mx) / temperature) }
        val z = ex.values.sum()
        val probs = Stage.entries.associateWith { ex[it]!! / z }
        val feats = mapOf(
            "pos120" to pos120, "ret20" to ret20, "ret60" to ret60, "dd20" to dd20,
            "vr" to vr, "bwPct" to bwPct, "maUp" to maUp, "obvDiv" to obvDiv, "upperShadow" to upperShadow,
        )
        return StageEstimate(probs, feats)
    }

    /** 三角隶属函数:a..b 上升,b..c 下降,峰值 1 */
    private fun tri(x: Double, a: Double, b: Double, c: Double): Double = when {
        x <= a || x >= c -> 0.0
        x <= b -> (x - a) / (b - a)
        else -> (c - x) / (c - b)
    }
}

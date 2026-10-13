package realt.core.daytype

import realt.core.indicator.Indicators
import realt.core.model.Bar
import realt.core.model.Session
import realt.core.regime.MarketRegime
import realt.core.stage.Stage
import realt.core.stage.StageEstimate
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/** L3:分时形态 */
enum class DayType(val label: String) {
    TREND_UP("单边上涨"),
    TREND_DOWN("单边下跌"),
    FADE("冲高回落"),
    RECOVER("探底回升"),
    RANGE("宽幅震荡"),
    FLAT("窄幅横盘"),
}

typealias DayDist = Map<DayType, Double>

fun DayDist.top(): DayType = maxByOrNull { it.value }!!.key
fun DayDist.p(t: DayType): Double = this[t] ?: 0.0
fun DayDist.describe(): String =
    entries.sortedByDescending { it.value }.take(3)
        .joinToString(" ") { "%s%.0f%%".format(it.key.label, it.value * 100) }

fun normalize(m: Map<DayType, Double>): DayDist {
    val z = m.values.sum().takeIf { it > 0 } ?: 1.0
    return DayType.entries.associateWith { (m[it] ?: 0.0) / z }
}

/**
 * 事后给一整天的分时打标签(用于训练先验、评估预判准确率)。
 * [atrPct] 为日线 ATR / 昨收,用来把"波动大小"标准化到个股自身水平。
 */
object DayTypeLabeler {
    fun label(day: List<Bar>, prevClose: Double, atrPct: Double): DayType {
        val o = day.first().open
        val c = day.last().close
        var h = Double.MIN_VALUE; var l = Double.MAX_VALUE
        var ih = 0; var il = 0
        for ((i, b) in day.withIndex()) {
            if (b.high > h) { h = b.high; ih = i }
            if (b.low < l) { l = b.low; il = i }
        }
        val r = (h - l) / prevClose
        if (r < max(0.012, 0.5 * atrPct)) return DayType.FLAT
        val span = h - l
        val net = (c - o) / span
        val closePos = (c - l) / span
        val fh = ih.toDouble() / day.size
        val fl = il.toDouble() / day.size
        return when {
            net >= 0.55 && closePos >= 0.7 -> DayType.TREND_UP
            net <= -0.55 && closePos <= 0.3 -> DayType.TREND_DOWN
            fh < 0.6 && closePos <= 0.45 && (h - o) / span >= 0.3 -> DayType.FADE
            fl < 0.6 && closePos >= 0.55 && (o - l) / span >= 0.3 -> DayType.RECOVER
            else -> DayType.RANGE
        }
    }
}

/** 开盘前已知的上下文 */
data class DayContext(
    val prevClose: Double,
    /** 日线 ATR(14) / 昨收 */
    val atrPct: Double,
    val regime: MarketRegime,
    val stage: StageEstimate,
    /** 涨跌停幅度(主板 10%,创业板/科创板 20%,北交所 30%) */
    val limitPct: Double = 0.10,
    /** 开盘前可知的日线特征,见 [realt.core.ml.DailyFeatures] */
    val daily: Map<String, Double> = emptyMap(),
)

/**
 * 分时形态预判:先验(大盘 + 主力阶段 + 开盘缺口)× 盘中似然,逐 bar 贝叶斯式更新。
 *
 * 先验可以由 [priorOverride] 注入(L6 根据历史统计学到的经验分布),
 * 否则使用内置的经验基础比例并按方向偏置倾斜。
 */
class DayTypePredictor(
    private val baseRates: DayDist = mapOf(
        DayType.TREND_UP to 0.14, DayType.TREND_DOWN to 0.14,
        DayType.FADE to 0.18, DayType.RECOVER to 0.18,
        DayType.RANGE to 0.26, DayType.FLAT to 0.10,
    ),
    private val priorOverride: ((DayContext) -> DayDist?)? = null,
) {

    /** 开盘前方向偏置 -1..1 */
    fun bias(ctx: DayContext): Double =
        (ctx.regime.score * 0.4 + ctx.stage.bias * 0.6).coerceIn(-1.0, 1.0)

    fun prior(ctx: DayContext, openPrice: Double): DayDist {
        priorOverride?.invoke(ctx)?.let { return it }
        val b = bias(ctx)
        val gap = (openPrice / ctx.prevClose - 1) / max(ctx.atrPct, 0.005) // 以 ATR 计的缺口
        val m = HashMap(baseRates)
        m[DayType.TREND_UP] = m.p(DayType.TREND_UP) * exp(1.2 * b + 0.3 * gap.coerceIn(-2.0, 2.0))
        m[DayType.TREND_DOWN] = m.p(DayType.TREND_DOWN) * exp(-1.2 * b - 0.3 * gap.coerceIn(-2.0, 2.0))
        // 大幅高开更易冲高回落,大幅低开更易探底回升
        m[DayType.FADE] = m.p(DayType.FADE) * exp(-0.5 * b + 0.5 * gap.coerceIn(0.0, 2.0))
        m[DayType.RECOVER] = m.p(DayType.RECOVER) * exp(0.5 * b + 0.5 * (-gap).coerceIn(0.0, 2.0))
        // 波动收敛(布林带宽低分位)更容易横盘
        val bw = ctx.stage.features["bwPct"] ?: 0.5
        m[DayType.FLAT] = m.p(DayType.FLAT) * exp(1.0 * (0.5 - bw)) *
            (if (ctx.stage.top == Stage.ACCUMULATION) 1.3 else 1.0)
        return normalize(m)
    }

    /**
     * 根据今日已走出的分钟线更新后验。
     * [today] 为今日截至当前的 1 分钟线(至少 1 根)。
     */
    fun posterior(prior: DayDist, ctx: DayContext, today: List<Bar>): DayDist {
        if (today.isEmpty()) return prior
        val atrAbs = max(ctx.atrPct, 0.005) * ctx.prevClose
        val o = today.first().open
        val c = today.last().close
        var h = Double.MIN_VALUE; var l = Double.MAX_VALUE
        var ih = 0; var il = 0
        for ((i, b) in today.withIndex()) {
            if (b.high > h) { h = b.high; ih = i }
            if (b.low < l) { l = b.low; il = i }
        }
        val vwap = Indicators.vwap(today)
        var above = 0
        var cross = 0
        for (i in today.indices) {
            if (today[i].close > vwap[i]) above++
            if (i > 0 && (today[i].close - vwap[i]) * (today[i - 1].close - vwap[i - 1]) < 0) cross++
        }
        val n = today.size
        val f = min(1.0, Session.elapsed(today.last().minute).toDouble() / Session.MINUTES_PER_DAY)
        val fracAbove = above.toDouble() / n
        // 以"到此刻为止随机游走应有的幅度" sqrt(f) 归一化,开盘早期与午后的特征可比
        val scale = max(sqrt(f), 0.25)
        val nz = (c - o) / atrAbs / scale          // 净涨跌
        val rz = (h - l) / atrAbs / scale          // 振幅
        val pos = if (h > l) (c - l) / (h - l) - 0.5 else 0.0
        val fromHigh = (h - c) / atrAbs / scale
        val fromLow = (c - l) / atrAbs / scale
        val riseFirst = (h - o) / atrAbs / scale
        val dropFirst = (o - l) / atrAbs / scale
        val crossRate = cross.toDouble() / max(1, n) * 30  // 每 30 分钟穿越均价线次数
        val fa = fracAbove - 0.5

        val s = HashMap<DayType, Double>()
        s[DayType.TREND_UP] = 1.5 * nz.coerceIn(-2.0, 2.0) + 2.0 * fa + 1.0 * pos - 0.2 * crossRate - 0.8 * fromHigh.coerceAtMost(1.5)
        s[DayType.TREND_DOWN] = -1.5 * nz.coerceIn(-2.0, 2.0) - 2.0 * fa - 1.0 * pos - 0.2 * crossRate - 0.8 * fromLow.coerceAtMost(1.5)
        // 冲高回落:先涨出空间,再从高点回落,且高点出现在前段
        s[DayType.FADE] = 1.6 * min(riseFirst, 1.5) * min(fromHigh, 1.5) - 0.8 * max(nz, 0.0) +
            (if (ih < n * 0.7) 0.3 else -0.3)
        s[DayType.RECOVER] = 1.6 * min(dropFirst, 1.5) * min(fromLow, 1.5) - 0.8 * max(-nz, 0.0) +
            (if (il < n * 0.7) 0.3 else -0.3)
        // 与标注器同口径:全天振幅 < max(1.2%, 0.5×ATR) 才算窄幅横盘;已超过则基本不可能
        val flatLimit = max(0.012, 0.5 * ctx.atrPct) * ctx.prevClose
        val flatRoom = ((flatLimit - (h - l)) / flatLimit).coerceIn(-1.0, 1.0) // >0 表示振幅仍在横盘范围内
        // 宽幅震荡:来回穿越均价线、净位移小,且振幅已超出横盘范围
        s[DayType.RANGE] = 0.8 - 0.8 * abs(nz).coerceAtMost(2.0) + 0.25 * crossRate.coerceAtMost(4.0) +
            1.2 * (rz - 0.6).coerceIn(-1.0, 0.4) - 2.0 * flatRoom.coerceAtLeast(0.0) * (0.3 + 0.7 * f)
        s[DayType.FLAT] = if (flatRoom <= 0) -3.0
        else 2.5 * flatRoom * (0.3 + 0.7 * f) + 0.25 * crossRate.coerceAtMost(4.0) - 0.8 * abs(nz).coerceAtMost(2.0)

        val w = 0.6 + 1.2 * sqrt(f)   // 越到后面越相信盘面
        val m = HashMap<DayType, Double>()
        for (t in DayType.entries) m[t] = prior.p(t) * exp(w * (s[t] ?: 0.0))
        return normalize(m)
    }
}

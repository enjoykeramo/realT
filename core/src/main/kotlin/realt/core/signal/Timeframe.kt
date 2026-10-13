package realt.core.signal

import realt.core.model.Bar
import realt.core.model.Session
import kotlin.math.sqrt

/** 可增长的 double 数组(避免装箱) */
class DoubleList(cap: Int = 512) {
    var data = DoubleArray(cap); private set
    var size = 0; private set
    fun add(v: Double) {
        if (size == data.size) data = data.copyOf(size * 2)
        data[size++] = v
    }
    operator fun get(i: Int) = data[i]
    fun last() = data[size - 1]
}

/**
 * 某一周期(1/5/15 分钟)K 线及其指标的增量状态。
 *
 * 每来一根 1 分钟线调用 [update]:同一周期桶内合并为"正在形成的 K 线",跨桶时把上一根定稿并推进指标状态。
 * "当前值"(含正在形成的 K 线)按需从已定稿状态推一步得到,复杂度 O(1)(布林为 O(20))。
 * 与 [realt.core.indicator.Indicators] 的批量算法逐值一致(EMA 首值初始化、通达信 SMA 平滑)。
 */
class Timeframe(val period: Int) {
    /** 已定稿 K 线 */
    val bars = ArrayList<Bar>(256)
    val closes = DoubleList()
    val dif = DoubleList()
    val dea = DoubleList()
    val hist = DoubleList()
    private var ema12 = 0.0
    private var ema26 = 0.0
    private var deaS = 0.0
    private var rsiUp = 0.0
    private var rsiAll = 0.0

    /** 正在形成的 K 线 */
    var current: Bar? = null; private set
    private var curKey = -1L

    fun update(b: Bar) {
        val key = if (period <= 1) b.date * 10000L + b.minute
        else b.date * 10000L + (Session.elapsed(b.minute) - 1) / period
        val c = current
        if (c != null && key == curKey) {
            current = Bar(c.date, b.minute, c.open, maxOf(c.high, b.high), minOf(c.low, b.low), b.close,
                c.volume + b.volume, c.amount + b.amount)
            return
        }
        if (c != null) commit(c)
        current = b
        curKey = key
    }

    private fun commit(b: Bar) {
        val x = b.close
        if (closes.size == 0) {
            ema12 = x; ema26 = x; deaS = 0.0; rsiUp = 0.0; rsiAll = 0.0
        } else {
            ema12 += 2.0 / 13 * (x - ema12)
            ema26 += 2.0 / 27 * (x - ema26)
            val d = ema12 - ema26
            deaS += 2.0 / 10 * (d - deaS)
            val ch = x - closes.last()
            rsiUp = (maxOf(ch, 0.0) + 5 * rsiUp) / 6
            rsiAll = (kotlin.math.abs(ch) + 5 * rsiAll) / 6
        }
        bars += b
        closes.add(x)
        val d = ema12 - ema26
        dif.add(d); dea.add(deaS); hist.add(2 * (d - deaS))
    }

    /** 含正在形成 K 线的总根数 */
    val size: Int get() = closes.size + if (current != null) 1 else 0

    fun closeNow(): Double = current?.close ?: closes.last()

    /** 当前(含未定稿 K 线)的 MACD:dif, dea, hist */
    fun macdNow(): Triple<Double, Double, Double> {
        val c = current ?: return Triple(dif.last(), dea.last(), hist.last())
        if (closes.size == 0) return Triple(0.0, 0.0, 0.0)
        val e12 = ema12 + 2.0 / 13 * (c.close - ema12)
        val e26 = ema26 + 2.0 / 27 * (c.close - ema26)
        val d = e12 - e26
        val de = deaS + 2.0 / 10 * (d - deaS)
        return Triple(d, de, 2 * (d - de))
    }

    /** 上一根已定稿 K 线的 MACD 柱;不足时为 0 */
    fun histBack(k: Int): Double = if (hist.size >= k) hist[hist.size - k] else 0.0

    fun rsiNow(): Double {
        val c = current ?: return if (rsiAll > 1e-12) 100 * rsiUp / rsiAll else 50.0
        if (closes.size == 0) return 50.0
        val ch = c.close - closes.last()
        val up = (maxOf(ch, 0.0) + 5 * rsiUp) / 6
        val all = (kotlin.math.abs(ch) + 5 * rsiAll) / 6
        return if (all > 1e-12) 100 * up / all else 50.0
    }

    /**
     * 布林(20, 2):返回 (%b, 带宽绝对值);[back] = 0 表示当前,1 表示上一根……数据不足返回 NaN。
     */
    fun boll(back: Int = 0, n: Int = 20): Pair<Double, Double> {
        val total = size
        val endExclusive = total - back
        if (endExclusive < n) return Double.NaN to Double.NaN
        var s = 0.0
        for (i in endExclusive - n until endExclusive) s += closeAt(i)
        val mid = s / n
        var v = 0.0
        for (i in endExclusive - n until endExclusive) { val d = closeAt(i) - mid; v += d * d }
        val sd = sqrt(v / n)
        val band = 4 * sd
        val x = closeAt(endExclusive - 1)
        return (if (band > 1e-12) (x - (mid - 2 * sd)) / band else 0.5) to band
    }

    /** 第 i 根(含正在形成的)收盘价 */
    fun closeAt(i: Int): Double = if (i < closes.size) closes[i] else current!!.close

    /** 含正在形成 K 线的 bar 列表(给缠论用) */
    fun barsNow(): List<Bar> = current?.let { bars + it } ?: bars

    /** 含正在形成 K 线的 MACD 柱数组 */
    fun histNow(): DoubleArray {
        val n = hist.size
        val out = DoubleArray(n + if (current != null) 1 else 0)
        System.arraycopy(hist.data, 0, out, 0, n)
        if (current != null) out[n] = macdNow().third
        return out
    }
}

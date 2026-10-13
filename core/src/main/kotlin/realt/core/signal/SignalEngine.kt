package realt.core.signal

import realt.core.chan.ChanAnalyzer
import realt.core.chan.ChanPoint
import realt.core.indicator.Indicators
import realt.core.indicator.closes
import realt.core.model.Bar
import realt.core.model.aggregate

/** 信号家族,用于统计不同来源信号的胜率 */
enum class SignalFamily(val label: String) { CHAN("缠论"), BOLL("布林"), VWAP("均价"), NONE("-") }

/**
 * L4 某一时刻的短线信号快照。分数均为 0..1。
 *  - structure:结构信号(缠论买卖点 / 布林带拒绝)
 *  - location:位置(偏离均价线程度,以日 ATR 为单位)
 *  - confirm:5 分钟级别确认(区间套)
 */
data class SignalSnapshot(
    val sellStructure: Double,
    val sellLocation: Double,
    val sellConfirm: Double,
    val sellFamily: SignalFamily,
    val buyStructure: Double,
    val buyLocation: Double,
    val buyConfirm: Double,
    val buyFamily: SignalFamily,
    val vwap: Double,
    val price: Double,
    /** (价格 - 均价) / 日ATR */
    val vwapDev: Double,
    val percentB1: Double,
    val percentB5: Double,
    /** 近 30 分钟位移(日 ATR 单位),正为上涨 */
    val trend5: Double,
    /** 1 分钟 RSI(6) */
    val rsi1: Double,
    val chan: ChanPoint?,
    val note: String,
) {
    val sellScore: Double
        get() = if (sellStructure <= 0 || sellLocation <= 0) 0.0
        else 0.4 * sellStructure + 0.35 * sellLocation + 0.25 * sellConfirm

    val buyScore: Double
        get() = if (buyStructure <= 0 || buyLocation <= 0) 0.0
        else 0.4 * buyStructure + 0.35 * buyLocation + 0.25 * buyConfirm
}

/**
 * L4:基于 1 分钟线出信号、5 分钟线确认。
 * 输入为截至当前的 1 分钟线序列(可包含前几日作为结构上下文),只看过去。
 */
class SignalEngine(
    private val chan: ChanAnalyzer = ChanAnalyzer(),
    /** 位置分数从 vwapDev = start 开始计分,到 start+span 满分(单位:日 ATR) */
    private val locStart: Double = 0.08,
    private val locSpan: Double = 0.30,
) {

    fun evaluate(bars: List<Bar>, atrAbs: Double): SignalSnapshot {
        val n = bars.size - 1
        val today = bars.dropWhile { it.date != bars[n].date }
        val vwap = Indicators.vwap(today).last()
        val price = bars[n].close
        val dev = (price - vwap) / atrAbs

        val c1 = bars.closes()
        val boll1 = Indicators.boll(c1, 20, 2.0)
        val pb = boll1.percentB
        val pbNow = pb[n]
        val pbPrevMax = if (n >= 3) maxOf(pb[n - 1], pb[n - 2], pb[n - 3]) else Double.NaN
        val pbPrevMin = if (n >= 3) minOf(pb[n - 1], pb[n - 2], pb[n - 3]) else Double.NaN

        val five = bars.aggregate(5)
        val c5 = five.closes()
        val pb5 = Indicators.boll(c5, 20, 2.0).percentB.lastOrNull() ?: Double.NaN
        val pb5Arr = Indicators.boll(c5, 20, 2.0).percentB
        val h5 = Indicators.macd(c5).hist
        val k = h5.size - 1
        // 5 分钟动能连续两根减弱/增强
        val h5Falling = k >= 2 && h5[k] < h5[k - 1] && h5[k - 1] < h5[k - 2]
        val h5Rising = k >= 2 && h5[k] > h5[k - 1] && h5[k - 1] > h5[k - 2]
        // 5 分钟布林:近 3 根曾到上轨附近、现已回落(或下轨附近收回)
        val pb5RecentMax = if (k >= 3) maxOf(pb5Arr[k - 1], pb5Arr[k - 2], pb5Arr[k - 3]) else Double.NaN
        val pb5RecentMin = if (k >= 3) minOf(pb5Arr[k - 1], pb5Arr[k - 2], pb5Arr[k - 3]) else Double.NaN
        // 近 30 分钟(6 根 5 分钟线)趋势:以日 ATR 为单位的位移
        val trend5 = if (k >= 6) (c5[k] - c5[k - 6]) / atrAbs else 0.0

        val cp = chan.analyze(bars).freshPoint
        val notes = ArrayList<String>()

        // ---- 结构 ----
        var sellS = 0.0; var sellF = SignalFamily.NONE
        var buyS = 0.0; var buyF = SignalFamily.NONE
        if (cp != null) {
            if (cp.type.buy) { buyS = cp.strength; buyF = SignalFamily.CHAN }
            else { sellS = cp.strength; sellF = SignalFamily.CHAN }
            notes += "${cp.type.label}(${cp.note})"
        }
        // 布林拒绝:前 3 根曾刺破上轨、本根收回轨内
        if (!pbNow.isNaN() && !pbPrevMax.isNaN() && pbPrevMax > 1.0 && pbNow < 0.95 && 0.6 > sellS) {
            sellS = 0.6; sellF = SignalFamily.BOLL; notes += "上轨回落 %%b=%.2f".format(pbNow)
        }
        if (!pbNow.isNaN() && !pbPrevMin.isNaN() && pbPrevMin < 0.0 && pbNow > 0.05 && 0.6 > buyS) {
            buyS = 0.6; buyF = SignalFamily.BOLL; notes += "下轨收回 %%b=%.2f".format(pbNow)
        }

        // RSI 衰竭:1 分钟 RSI(6) 从极值区拐头
        val rsi = Indicators.rsi(c1, 6)
        if (n >= 2 && !rsi[n].isNaN()) {
            if (maxOf(rsi[n - 1], rsi[n - 2]) >= 80 && rsi[n] < rsi[n - 1] && 0.5 > sellS) {
                sellS = 0.5; sellF = SignalFamily.VWAP; notes += "RSI 高位拐头 %.0f".format(rsi[n])
            }
            if (minOf(rsi[n - 1], rsi[n - 2]) <= 20 && rsi[n] > rsi[n - 1] && 0.5 > buyS) {
                buyS = 0.5; buyF = SignalFamily.VWAP; notes += "RSI 低位拐头 %.0f".format(rsi[n])
            }
        }

        // ---- 位置(以均价线偏离度计)----
        val sellLoc = ((dev - locStart) / locSpan).coerceIn(0.0, 1.0)
        val buyLoc = ((-dev - locStart) / locSpan).coerceIn(0.0, 1.0)

        // ---- 5 分钟确认(区间套:要求高一级别出现"转折",而不仅是位置高低)----
        val sellConf = (if (h5Falling) 0.5 else 0.0) +
            (if (!pb5RecentMax.isNaN() && pb5RecentMax > 0.9 && pb5 < pb5RecentMax - 0.1) 0.5 else 0.0)
        val buyConf = (if (h5Rising) 0.5 else 0.0) +
            (if (!pb5RecentMin.isNaN() && pb5RecentMin < 0.1 && pb5 > pb5RecentMin + 0.1) 0.5 else 0.0)

        return SignalSnapshot(
            sellS, sellLoc, sellConf, sellF,
            buyS, buyLoc, buyConf, buyF,
            vwap, price, dev, pbNow, pb5, trend5, if (rsi[n].isNaN()) 50.0 else rsi[n], cp, notes.joinToString(";"),
        )
    }
}

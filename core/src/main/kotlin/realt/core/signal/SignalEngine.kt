package realt.core.signal

import realt.core.chan.ChanAnalyzer
import realt.core.chan.ChanPoint
import realt.core.model.Bar

/** 信号家族,用于统计不同来源信号的胜率 */
enum class SignalFamily(val label: String) { CHAN("缠论"), BOLL("布林"), VWAP("均价"), NONE("-") }

/** 各指标模块的连续状态在 [SignalSnapshot.ind] 中的下标 */
object Ind {
    // MACD(价格类数值均以日 ATR 归一)
    const val M1_DIF = 0; const val M1_DEA = 1; const val M1_HIST = 2; const val M1_SLOPE = 3; const val M1_CROSS = 4
    const val M5_DIF = 5; const val M5_HIST = 6; const val M5_SLOPE = 7; const val M15_HIST = 8; const val M15_SLOPE = 9
    // 布林
    const val PB1 = 10; const val PB5 = 11; const val PB15 = 12; const val BW1 = 13; const val BW5 = 14
    const val PB1_MAX5 = 15; const val PB1_MIN5 = 16
    // RSI
    const val RSI1 = 17; const val RSI5 = 18
    // 缠论 1 分钟
    const val C1_UP_RATIO = 19; const val C1_UP_NEW = 20; const val C1_DN_RATIO = 21; const val C1_DN_NEW = 22
    const val C1_LEG = 23; const val C1_SINCE = 24; const val C1_LAST_TOP = 25; const val C1_CPOS = 26; const val C1_HAS_C = 27
    // 缠论 5 分钟
    const val C5_UP_RATIO = 28; const val C5_DN_RATIO = 29; const val C5_BUY = 30; const val C5_SELL = 31; const val C5_LEG = 32
    const val C5_LAST_TOP = 33
    const val SIZE = 34
}

/**
 * L4 某一时刻的短线信号快照。分数均为 0..1。
 *  - structure:结构信号(缠论买卖点 / 布林带拒绝 / RSI 拐头)
 *  - location:位置(偏离均价线程度,以日 ATR 为单位)
 *  - confirm:5 分钟级别确认(区间套)
 *  - ind:各模块的连续状态(见 [Ind]),供训练模型使用
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
    val ind: DoubleArray = DoubleArray(Ind.SIZE),
) {
    val sellScore: Double
        get() = if (sellStructure <= 0 || sellLocation <= 0) 0.0
        else 0.4 * sellStructure + 0.35 * sellLocation + 0.25 * sellConfirm

    val buyScore: Double
        get() = if (buyStructure <= 0 || buyLocation <= 0) 0.0
        else 0.4 * buyStructure + 0.35 * buyLocation + 0.25 * buyConfirm
}

/**
 * L4:1 分钟出信号、5/15 分钟确认,全部指标增量计算(每分钟 O(n) 仅来自缠论结构识别)。
 * 有状态:每只股票每个交易日新建一个实例;先用 [warm] 喂入历史分钟线,再逐分钟调用 [update]。
 */
class SignalEngine(
    private val chan: ChanAnalyzer = ChanAnalyzer(),
    /** 位置分数从 vwapDev = start 开始计分,到 start+span 满分(单位:日 ATR) */
    private val locStart: Double = 0.08,
    private val locSpan: Double = 0.30,
) {
    private val tf1 = Timeframe(1)
    private val tf5 = Timeframe(5)
    private val tf15 = Timeframe(15)
    private val all = ArrayList<Bar>(512)
    private val rsi1Hist = DoubleList()
    private val pb1Hist = DoubleList()
    private var day = Int.MIN_VALUE
    private var pv = 0.0
    private var vol = 0.0
    private var lastCrossIdx = -100
    private var lastCrossSign = 0

    /** 喂入历史 K 线(只推进指标状态) */
    fun warm(b: Bar) {
        push(b)
    }

    private fun push(b: Bar) {
        if (b.date != day) { day = b.date; pv = 0.0; vol = 0.0 }
        pv += (b.high + b.low + b.close) / 3 * b.volume
        vol += b.volume
        // 记录上一根(已定稿)1 分钟的 RSI / %b,供"拐头/收回"判断
        if (tf1.current != null) {
            rsi1Hist.add(tf1.rsiNow())
            pb1Hist.add(tf1.boll().first)
        }
        tf1.update(b); tf5.update(b); tf15.update(b)
        all += b
    }

    fun update(b: Bar, atrAbs: Double): SignalSnapshot {
        val prevDif = if (tf1.size > 0) tf1.macdNow().let { it.first - it.second } else 0.0
        push(b)
        val n = all.size - 1
        val price = b.close
        val vwap = if (vol > 0) pv / vol else price
        val dev = (price - vwap) / atrAbs

        // ---- 1 分钟 ----
        val (pbNow, bw1) = tf1.boll()
        val ph = pb1Hist.size
        fun pbBack(k: Int) = if (ph >= k) pb1Hist[ph - k] else Double.NaN
        val pbPrevMax = if (ph >= 3) maxOf(pbBack(1), pbBack(2), pbBack(3)) else Double.NaN
        val pbPrevMin = if (ph >= 3) minOf(pbBack(1), pbBack(2), pbBack(3)) else Double.NaN
        val (d1, de1, h1) = tf1.macdNow()
        val cross = (d1 - de1).let { cur -> if (prevDif <= 0 && cur > 0) 1 else if (prevDif >= 0 && cur < 0) -1 else 0 }
        if (cross != 0) { lastCrossIdx = n; lastCrossSign = cross }
        val rsiNow = tf1.rsiNow()

        // ---- 5 / 15 分钟 ----
        val (pb5, bw5) = tf5.boll()
        val (d5, _, h5) = tf5.macdNow()
        val h5b1 = tf5.histBack(1); val h5b2 = tf5.histBack(2)
        val k5 = tf5.size - 1
        val h5Falling = k5 >= 2 && h5 < h5b1 && h5b1 < h5b2
        val h5Rising = k5 >= 2 && h5 > h5b1 && h5b1 > h5b2
        val pb5RecentMax = if (k5 >= 3) maxOf(tf5.boll(1).first, tf5.boll(2).first, tf5.boll(3).first) else Double.NaN
        val pb5RecentMin = if (k5 >= 3) minOf(tf5.boll(1).first, tf5.boll(2).first, tf5.boll(3).first) else Double.NaN
        val trend5 = if (k5 >= 6) (tf5.closeNow() - tf5.closeAt(k5 - 6)) / atrAbs else 0.0
        val (pb15, _) = tf15.boll()
        val h15 = tf15.macdNow().third

        // ---- 缠论:1 分钟(出信号)+ 5 分钟(区间套)----
        val hist1 = tf1.histNow()
        val r1 = chan.analyze(all, hist1)
        val cp = r1.freshPoint
        val legs1 = chan.legs(r1, all, hist1, atrAbs)
        val bars5 = tf5.barsNow()
        val hist5 = tf5.histNow()
        val r5 = chan.analyze(bars5, hist5)
        val legs5 = chan.legs(r5, bars5, hist5, atrAbs)
        val cp5 = r5.freshPoint

        val notes = ArrayList<String>()
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
        val rh = rsi1Hist.size
        if (rh >= 2) {
            val r1b = rsi1Hist[rh - 1]; val r2b = rsi1Hist[rh - 2]
            if (maxOf(r1b, r2b) >= 80 && rsiNow < r1b && 0.5 > sellS) {
                sellS = 0.5; sellF = SignalFamily.VWAP; notes += "RSI 高位拐头 %.0f".format(rsiNow)
            }
            if (minOf(r1b, r2b) <= 20 && rsiNow > r1b && 0.5 > buyS) {
                buyS = 0.5; buyF = SignalFamily.VWAP; notes += "RSI 低位拐头 %.0f".format(rsiNow)
            }
        }

        val sellLoc = ((dev - locStart) / locSpan).coerceIn(0.0, 1.0)
        val buyLoc = ((-dev - locStart) / locSpan).coerceIn(0.0, 1.0)
        val sellConf = (if (h5Falling) 0.5 else 0.0) +
            (if (!pb5RecentMax.isNaN() && pb5RecentMax > 0.9 && pb5 < pb5RecentMax - 0.1) 0.5 else 0.0)
        val buyConf = (if (h5Rising) 0.5 else 0.0) +
            (if (!pb5RecentMin.isNaN() && pb5RecentMin < 0.1 && pb5 > pb5RecentMin + 0.1) 0.5 else 0.0)

        fun nz(x: Double, d: Double = 0.5) = if (x.isNaN()) d else x
        var pbMax5 = nz(pbNow); var pbMin5 = nz(pbNow)
        for (k in 1..4) { val v = pbBack(k); if (!v.isNaN()) { pbMax5 = maxOf(pbMax5, v); pbMin5 = minOf(pbMin5, v) } }
        val ind = DoubleArray(Ind.SIZE)
        ind[Ind.M1_DIF] = d1 / atrAbs; ind[Ind.M1_DEA] = de1 / atrAbs; ind[Ind.M1_HIST] = h1 / atrAbs
        ind[Ind.M1_SLOPE] = (h1 - tf1.histBack(1)) / atrAbs
        // 最近 5 分钟内的金叉(+)/死叉(-),越近绝对值越大
        ind[Ind.M1_CROSS] = (n - lastCrossIdx).let { ago -> if (ago in 0..5) lastCrossSign * (6 - ago) / 6.0 else 0.0 }
        ind[Ind.M5_DIF] = d5 / atrAbs; ind[Ind.M5_HIST] = h5 / atrAbs; ind[Ind.M5_SLOPE] = (h5 - h5b1) / atrAbs
        ind[Ind.M15_HIST] = h15 / atrAbs; ind[Ind.M15_SLOPE] = (h15 - tf15.histBack(1)) / atrAbs
        ind[Ind.PB1] = nz(pbNow); ind[Ind.PB5] = nz(pb5); ind[Ind.PB15] = nz(pb15)
        ind[Ind.BW1] = nz(bw1, 0.0) / atrAbs; ind[Ind.BW5] = nz(bw5, 0.0) / atrAbs
        ind[Ind.PB1_MAX5] = pbMax5; ind[Ind.PB1_MIN5] = pbMin5
        ind[Ind.RSI1] = rsiNow; ind[Ind.RSI5] = tf5.rsiNow()
        ind[Ind.C1_UP_RATIO] = legs1.upRatio; ind[Ind.C1_UP_NEW] = legs1.upNew
        ind[Ind.C1_DN_RATIO] = legs1.dnRatio; ind[Ind.C1_DN_NEW] = legs1.dnNew
        ind[Ind.C1_LEG] = legs1.legAmp; ind[Ind.C1_SINCE] = legs1.barsSince; ind[Ind.C1_LAST_TOP] = legs1.lastTop
        ind[Ind.C1_CPOS] = legs1.centerPos; ind[Ind.C1_HAS_C] = legs1.hasCenter
        ind[Ind.C5_UP_RATIO] = legs5.upRatio; ind[Ind.C5_DN_RATIO] = legs5.dnRatio
        ind[Ind.C5_BUY] = if (cp5 != null && cp5.type.buy) cp5.strength else 0.0
        ind[Ind.C5_SELL] = if (cp5 != null && !cp5.type.buy) cp5.strength else 0.0
        ind[Ind.C5_LEG] = legs5.legAmp; ind[Ind.C5_LAST_TOP] = legs5.lastTop

        return SignalSnapshot(
            sellS, sellLoc, sellConf, sellF,
            buyS, buyLoc, buyConf, buyF,
            vwap, price, dev, pbNow, pb5, trend5, rsiNow, cp, notes.joinToString(";"), ind,
        )
    }
}

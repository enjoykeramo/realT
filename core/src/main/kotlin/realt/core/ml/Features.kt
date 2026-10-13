package realt.core.ml

import realt.core.daytype.DayContext
import realt.core.daytype.DayType
import realt.core.daytype.p
import realt.core.engine.Observation
import realt.core.engine.TDirection
import realt.core.signal.SignalFamily
import realt.core.stage.Stage
import kotlin.math.max

/**
 * 候选做T点的特征。大多数特征按方向"同向化":数值越大越有利于该方向
 * (先卖后买时价格越高越有利,先买后卖时价格越低越有利),这样一个模型可同时服务两个方向。
 */
object Features {

    val names: List<String> = listOf(
        "dirSell",
        // L4 信号
        "structure", "location", "confirm", "score", "famChan", "famBoll", "famRsi", "chanClass",
        "vwapDevS", "pb1S", "pb5S", "trend5S", "rsiS",
        // L3 形态后验
        "pUp", "pDown", "pFade", "pRecover", "pRange", "pFlat", "withTrend",
        // 分时状态
        "elapsed", "netFromOpenS", "rangeAtr", "posInRangeS", "toExtremeS", "toOtherExtremeS",
        "dayChgS", "gapS", "vwapSlopeS", "volRatio5", "fracAboveS", "extremeAtS",
        // L1 / L2
        "regimeS", "stageBiasS", "pAccum", "pMarkup", "pShake", "pMainRise", "pDistrib", "pDecline",
        "pos120", "ret20", "ret60", "dd20", "vr", "bwPct", "maUp",
        // 个股波动
        "atrPct",
        // 量价微观结构
        "flow15S", "rv30",
        // 指数分时(同一分钟)与相对强弱
        "hasIndex", "idxRetS", "idxVsVwapS", "idxSlope30S", "relStrengthS",
        // 涨跌停距离(ATR):逆向一侧(先卖时为涨停)/ 同向一侧
        "toLimitAgainst", "toLimitWith",
        // 昨日及日线位置
        "yRetS", "yClosePosS", "yRangeAtr", "y2RetS", "toMa5S", "toMa20S", "yVolRatio", "upDays5S",
    )

    val size: Int get() = names.size

    fun extract(ctx: DayContext, obs: Observation, dir: TDirection): DoubleArray {
        val sell = dir == TDirection.SELL_FIRST
        val s = if (sell) 1.0 else -1.0
        val sig = obs.sig
        val d = obs.dist
        val day = obs.day
        val atr = max(ctx.atrPct, 0.005) * ctx.prevClose
        val fam = if (sell) sig.sellFamily else sig.buyFamily
        val chan = sig.chan?.takeIf { it.type.buy != sell }
        val upish = d.p(DayType.TREND_UP) + d.p(DayType.RECOVER)
        val downish = d.p(DayType.TREND_DOWN) + d.p(DayType.FADE)
        val range = max(day.high - day.low, 1e-9)
        val pos = (day.close - day.low) / range
        val f = ctx.stage.features
        val st = ctx.stage.probs
        fun nz(x: Double) = if (x.isNaN()) 0.5 else x
        fun dl(k: String, v: Double) = ctx.daily[k] ?: v

        return doubleArrayOf(
            if (sell) 1.0 else 0.0,
            if (sell) sig.sellStructure else sig.buyStructure,
            if (sell) sig.sellLocation else sig.buyLocation,
            if (sell) sig.sellConfirm else sig.buyConfirm,
            if (sell) sig.sellScore else sig.buyScore,
            if (fam == SignalFamily.CHAN) 1.0 else 0.0,
            if (fam == SignalFamily.BOLL) 1.0 else 0.0,
            if (fam == SignalFamily.VWAP) 1.0 else 0.0,
            chan?.type?.let { (it.ordinal % 3 + 1).toDouble() } ?: 0.0,
            s * sig.vwapDev,
            if (sell) nz(sig.percentB1) else 1 - nz(sig.percentB1),
            if (sell) nz(sig.percentB5) else 1 - nz(sig.percentB5),
            s * sig.trend5,
            if (sell) sig.rsi1 else 100 - sig.rsi1,
            d.p(DayType.TREND_UP), d.p(DayType.TREND_DOWN), d.p(DayType.FADE),
            d.p(DayType.RECOVER), d.p(DayType.RANGE), d.p(DayType.FLAT),
            if (sell) downish - upish else upish - downish,
            day.elapsed / 240.0,
            s * (day.close - day.open) / atr,
            range / atr,
            if (sell) pos else 1 - pos,
            (if (sell) day.high - day.close else day.close - day.low) / atr,
            (if (sell) day.close - day.low else day.high - day.close) / atr,
            s * (day.close / ctx.prevClose - 1) / ctx.atrPct.coerceAtLeast(0.005),
            s * (day.open / ctx.prevClose - 1) / ctx.atrPct.coerceAtLeast(0.005),
            s * day.vwapSlope30,
            day.volRatio5,
            if (sell) day.fracAbove else 1 - day.fracAbove,
            if (sell) day.highAt else day.lowAt,
            -s * ctx.regime.score,
            -s * ctx.stage.bias,
            st[Stage.ACCUMULATION] ?: 0.0, st[Stage.MARKUP] ?: 0.0, st[Stage.SHAKEOUT] ?: 0.0,
            st[Stage.MAIN_RISE] ?: 0.0, st[Stage.DISTRIBUTION] ?: 0.0, st[Stage.DECLINE] ?: 0.0,
            f["pos120"] ?: 0.5, f["ret20"] ?: 0.0, f["ret60"] ?: 0.0, f["dd20"] ?: 0.0,
            f["vr"] ?: 1.0, f["bwPct"] ?: 0.5, f["maUp"] ?: 0.5,
            ctx.atrPct,
            s * day.flow15, day.rv30,
            if (day.hasIndex) 1.0 else 0.0, s * day.idxRet, s * day.idxVsVwap, s * day.idxSlope30,
            if (day.hasIndex) s * ((day.close / ctx.prevClose - 1) * 100 - day.idxRet) else 0.0,
            (if (sell) ctx.prevClose * (1 + ctx.limitPct) - day.close else day.close - ctx.prevClose * (1 - ctx.limitPct)) / atr,
            (if (sell) day.close - ctx.prevClose * (1 - ctx.limitPct) else ctx.prevClose * (1 + ctx.limitPct) - day.close) / atr,
            s * dl("yRetAtr", 0.0), if (sell) dl("yClosePos", 0.5) else 1 - dl("yClosePos", 0.5),
            dl("yRangeAtr", 1.0), s * dl("y2RetAtr", 0.0), s * dl("toMa5Atr", 0.0), s * dl("toMa20Atr", 0.0),
            dl("yVolRatio", 1.0), if (sell) dl("upDays5", 2.5) else 5 - dl("upDays5", 2.5),
        )
    }

    /**
     * 候选点定义:该方向出现任一"转折结构"(缠论买卖点 / 布林刺破收回 / RSI 极值拐头)。
     * 训练与实盘使用同一定义,保证模型看到的样本分布一致。
     */
    fun candidates(obs: Observation): List<TDirection> {
        val out = ArrayList<TDirection>(2)
        if (obs.sig.sellStructure > 0) out += TDirection.SELL_FIRST
        if (obs.sig.buyStructure > 0) out += TDirection.BUY_FIRST
        return out
    }
}

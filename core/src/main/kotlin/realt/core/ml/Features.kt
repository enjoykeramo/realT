package realt.core.ml

import realt.core.daytype.DayContext
import realt.core.daytype.DayType
import realt.core.daytype.p
import realt.core.engine.Observation
import realt.core.engine.TDirection
import realt.core.signal.Ind
import realt.core.signal.SignalFamily
import realt.core.stage.Stage
import kotlin.math.max

/** 特征分组 = 算法架构的层 / 模块。训练时逐组做消融,评估各层贡献并选择最优组合 */
enum class FeatureGroup(val label: String) {
    SIGNAL("L4信号综合"),
    CHAN("缠论"),
    MACD("MACD"),
    BOLL("布林%b"),
    VWAP_RSI("均价线/RSI"),
    DAYTYPE("L3分时形态"),
    INTRADAY("分时状态"),
    INDEX("指数分时"),
    MARKET("L1大盘环境"),
    STAGE("L2主力阶段/日线"),
    LIMIT("涨跌停距离"),
}

/**
 * 候选做T点的特征。大多数特征按方向"同向化"(后缀 S):数值越大越有利于该方向
 * (先卖后买时价格越高越有利,先买后卖时价格越低越有利),一个模型同时服务两个方向。
 */
object Features {

    class Spec(val name: String, val group: FeatureGroup)

    val specs: List<Spec> = buildList {
        fun g(group: FeatureGroup, vararg n: String) = n.forEach { add(Spec(it, group)) }
        g(FeatureGroup.SIGNAL, "dirSell", "structure", "location", "confirm", "score", "famChan", "famBoll", "famRsi")
        g(FeatureGroup.CHAN, "chanClass", "c1LegRatio", "c1LegNew", "c1OppRatio", "c1OppNew", "c1LegS", "c1Since",
            "c1PivotWith", "c1CenterPosS", "c1HasCenter", "c5LegRatio", "c5OppRatio", "c5PointWith", "c5PointAgainst",
            "c5LegS", "c5PivotWith")
        g(FeatureGroup.MACD, "m1DifS", "m1DeaS", "m1HistS", "m1SlopeS", "m1CrossS", "m5DifS", "m5HistS", "m5SlopeS",
            "m15HistS", "m15SlopeS")
        g(FeatureGroup.BOLL, "pb1S", "pb5S", "pb15S", "bw1", "bw5", "pbExtremeS")
        g(FeatureGroup.VWAP_RSI, "vwapDevS", "rsi1S", "rsi5S", "trend5S")
        g(FeatureGroup.DAYTYPE, "pUp", "pDown", "pFade", "pRecover", "pRange", "pFlat", "withTrend")
        g(FeatureGroup.INTRADAY, "elapsed", "netFromOpenS", "rangeAtr", "posInRangeS", "toExtremeS", "toOtherExtremeS",
            "dayChgS", "gapS", "vwapSlopeS", "volRatio5", "fracAboveS", "extremeAtS", "flow15S", "rv30", "atrPct")
        g(FeatureGroup.INDEX, "hasIndex", "idxRetS", "idxVsVwapS", "idxSlope30S", "relStrengthS")
        g(FeatureGroup.MARKET, "regimeS")
        g(FeatureGroup.STAGE, "stageBiasS", "pAccum", "pMarkup", "pShake", "pMainRise", "pDistrib", "pDecline",
            "pos120", "ret20", "ret60", "dd20", "vr", "bwPct", "maUp",
            "yRetS", "yClosePosS", "yRangeAtr", "y2RetS", "toMa5S", "toMa20S", "yVolRatio", "upDays5S")
        g(FeatureGroup.LIMIT, "toLimitAgainst", "toLimitWith")
    }

    val names: List<String> = specs.map { it.name }
    val size: Int get() = specs.size

    /** 某组特征的列下标 */
    fun columnsOf(group: FeatureGroup): IntArray = specs.indices.filter { specs[it].group == group }.toIntArray()

    fun extract(ctx: DayContext, obs: Observation, dir: TDirection): DoubleArray {
        val sell = dir == TDirection.SELL_FIRST
        val s = if (sell) 1.0 else -1.0
        val sig = obs.sig
        val ind = sig.ind
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
        fun with(x: Double) = if (sell) x else 1 - x
        val out = DoubleArray(specs.size)
        var i = 0
        fun put(x: Double) { out[i++] = if (x.isNaN() || x.isInfinite()) 0.0 else x }

        // L4 综合
        put(if (sell) 1.0 else 0.0)
        put(if (sell) sig.sellStructure else sig.buyStructure)
        put(if (sell) sig.sellLocation else sig.buyLocation)
        put(if (sell) sig.sellConfirm else sig.buyConfirm)
        put(if (sell) sig.sellScore else sig.buyScore)
        put(if (fam == SignalFamily.CHAN) 1.0 else 0.0)
        put(if (fam == SignalFamily.BOLL) 1.0 else 0.0)
        put(if (fam == SignalFamily.VWAP) 1.0 else 0.0)
        // 缠论:"本方向段"= 先卖时的上涨段 / 先买时的下跌段(即将被反转的那一段)
        put(chan?.type?.let { (it.ordinal % 3 + 1).toDouble() } ?: 0.0)
        put(if (sell) ind[Ind.C1_UP_RATIO] else ind[Ind.C1_DN_RATIO])
        put(if (sell) ind[Ind.C1_UP_NEW] else ind[Ind.C1_DN_NEW])
        put(if (sell) ind[Ind.C1_DN_RATIO] else ind[Ind.C1_UP_RATIO])
        put(if (sell) ind[Ind.C1_DN_NEW] else ind[Ind.C1_UP_NEW])
        put(s * ind[Ind.C1_LEG])
        put(ind[Ind.C1_SINCE])
        put(with(ind[Ind.C1_LAST_TOP]))
        put(if (ind[Ind.C1_HAS_C] > 0) (if (sell) ind[Ind.C1_CPOS] else 1 - ind[Ind.C1_CPOS]) else 0.0)
        put(ind[Ind.C1_HAS_C])
        put(if (sell) ind[Ind.C5_UP_RATIO] else ind[Ind.C5_DN_RATIO])
        put(if (sell) ind[Ind.C5_DN_RATIO] else ind[Ind.C5_UP_RATIO])
        put(if (sell) ind[Ind.C5_SELL] else ind[Ind.C5_BUY])
        put(if (sell) ind[Ind.C5_BUY] else ind[Ind.C5_SELL])
        put(s * ind[Ind.C5_LEG])
        put(with(ind[Ind.C5_LAST_TOP]))
        // MACD
        put(s * ind[Ind.M1_DIF]); put(s * ind[Ind.M1_DEA]); put(s * ind[Ind.M1_HIST]); put(s * ind[Ind.M1_SLOPE])
        put(s * ind[Ind.M1_CROSS])
        put(s * ind[Ind.M5_DIF]); put(s * ind[Ind.M5_HIST]); put(s * ind[Ind.M5_SLOPE])
        put(s * ind[Ind.M15_HIST]); put(s * ind[Ind.M15_SLOPE])
        // 布林
        put(with(nz(ind[Ind.PB1]))); put(with(nz(ind[Ind.PB5]))); put(with(nz(ind[Ind.PB15])))
        put(ind[Ind.BW1]); put(ind[Ind.BW5])
        put(if (sell) ind[Ind.PB1_MAX5] else 1 - ind[Ind.PB1_MIN5])
        // 均价线 / RSI
        put(s * sig.vwapDev)
        put(if (sell) sig.rsi1 else 100 - sig.rsi1)
        put(if (sell) ind[Ind.RSI5] else 100 - ind[Ind.RSI5])
        put(s * sig.trend5)
        // L3
        put(d.p(DayType.TREND_UP)); put(d.p(DayType.TREND_DOWN)); put(d.p(DayType.FADE))
        put(d.p(DayType.RECOVER)); put(d.p(DayType.RANGE)); put(d.p(DayType.FLAT))
        put(if (sell) downish - upish else upish - downish)
        // 分时状态
        put(day.elapsed / 240.0)
        put(s * (day.close - day.open) / atr)
        put(range / atr)
        put(with(pos))
        put((if (sell) day.high - day.close else day.close - day.low) / atr)
        put((if (sell) day.close - day.low else day.high - day.close) / atr)
        put(s * (day.close / ctx.prevClose - 1) / ctx.atrPct.coerceAtLeast(0.005))
        put(s * (day.open / ctx.prevClose - 1) / ctx.atrPct.coerceAtLeast(0.005))
        put(s * day.vwapSlope30)
        put(day.volRatio5)
        put(with(day.fracAbove))
        put(if (sell) day.highAt else day.lowAt)
        put(s * day.flow15)
        put(day.rv30)
        put(ctx.atrPct)
        // 指数分时
        put(if (day.hasIndex) 1.0 else 0.0)
        put(s * day.idxRet); put(s * day.idxVsVwap); put(s * day.idxSlope30)
        put(if (day.hasIndex) s * ((day.close / ctx.prevClose - 1) * 100 - day.idxRet) else 0.0)
        // L1
        put(-s * ctx.regime.score)
        // L2 / 日线
        put(-s * ctx.stage.bias)
        put(st[Stage.ACCUMULATION] ?: 0.0); put(st[Stage.MARKUP] ?: 0.0); put(st[Stage.SHAKEOUT] ?: 0.0)
        put(st[Stage.MAIN_RISE] ?: 0.0); put(st[Stage.DISTRIBUTION] ?: 0.0); put(st[Stage.DECLINE] ?: 0.0)
        put(f["pos120"] ?: 0.5); put(f["ret20"] ?: 0.0); put(f["ret60"] ?: 0.0); put(f["dd20"] ?: 0.0)
        put(f["vr"] ?: 1.0); put(f["bwPct"] ?: 0.5); put(f["maUp"] ?: 0.5)
        put(s * dl("yRetAtr", 0.0)); put(with(dl("yClosePos", 0.5)))
        put(dl("yRangeAtr", 1.0)); put(s * dl("y2RetAtr", 0.0)); put(s * dl("toMa5Atr", 0.0)); put(s * dl("toMa20Atr", 0.0))
        put(dl("yVolRatio", 1.0)); put(if (sell) dl("upDays5", 2.5) else 5 - dl("upDays5", 2.5))
        // 涨跌停距离:逆向一侧(先卖时为涨停)/ 同向一侧
        put((if (sell) ctx.prevClose * (1 + ctx.limitPct) - day.close else day.close - ctx.prevClose * (1 - ctx.limitPct)) / atr)
        put((if (sell) day.close - ctx.prevClose * (1 - ctx.limitPct) else ctx.prevClose * (1 + ctx.limitPct) - day.close) / atr)

        check(i == out.size) { "特征数与定义不一致: $i != ${out.size}" }
        return out
    }

    /**
     * 候选点定义:该方向出现任一"转折结构"(缠论 1/5 分钟买卖点 / 布林刺破收回 / RSI 极值拐头)。
     * 训练与实盘使用同一定义,保证模型看到的样本分布一致。
     */
    fun candidates(obs: Observation): List<TDirection> {
        val out = ArrayList<TDirection>(2)
        val ind = obs.sig.ind
        if (obs.sig.sellStructure > 0 || ind[Ind.C5_SELL] > 0) out += TDirection.SELL_FIRST
        if (obs.sig.buyStructure > 0 || ind[Ind.C5_BUY] > 0) out += TDirection.BUY_FIRST
        return out
    }
}

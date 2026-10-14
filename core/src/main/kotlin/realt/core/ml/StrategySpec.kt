package realt.core.ml

import realt.core.daytype.DayContext
import realt.core.daytype.DayType
import realt.core.daytype.p
import realt.core.engine.Observation
import realt.core.engine.TDirection
import realt.core.regime.MarketLevel
import realt.core.signal.Ind
import realt.core.signal.SignalFamily

/**
 * 一条可执行的做T策略(迭代寻优的搜索对象)。训练与实盘用同一份定义判断"这个候选点做不做"。
 *
 *  - dirs:0 双向 / 1 只先卖后买 / 2 只先买后卖
 *  - families:信号来源位掩码(1 缠论1分钟、2 布林、4 RSI、8 缠论5分钟)
 *  - timeStart..timeEnd:允许开仓的时间窗(开盘后经过的分钟数)
 *  - regimeMask:允许的大盘环境(按 [MarketLevel] 序号的位掩码)
 *  - atrMin..atrMax:个股日 ATR 比例范围
 *  - vwapDevMin:同向化均价线偏离(ATR)下限,即"位置要多有利"
 *  - withTrendMin:同向化形态顺势程度下限
 *  - useModel / modelThr:是否再用胜率模型过滤及其阈值(校准胜率)
 *  - maxTrades:每日最多做T次数
 */
data class StrategySpec(
    val dirs: Int = 0,
    val families: Int = 15,
    val timeStart: Int = 5,
    val timeEnd: Int = 220,
    val regimeMask: Int = 31,
    val atrMin: Double = 0.0,
    val atrMax: Double = 1.0,
    val vwapDevMin: Double = -9.0,
    val withTrendMin: Double = -9.0,
    val useModel: Boolean = false,
    val modelThr: Double = 0.5,
    val maxTrades: Int = 2,
) {
    fun allows(familyCode: Int, dir: TDirection, elapsed: Int, regimeOrd: Int, atrPct: Double, vwapDevS: Double, withTrend: Double): Boolean =
        firstFail(familyCode, dir, elapsed, regimeOrd, atrPct, vwapDevS, withTrend) == 0

    /**
     * 第一个不满足的条件:0 全部满足;1 方向;2 信号来源;3 时间;4 大盘;5 ATR;6 均价偏离;7 顺势程度。
     * 训练报告用它统计"出手漏斗"(候选点在哪一步被过滤掉)。
     */
    fun firstFail(familyCode: Int, dir: TDirection, elapsed: Int, regimeOrd: Int, atrPct: Double, vwapDevS: Double, withTrend: Double): Int {
        if (dirs == 1 && dir != TDirection.SELL_FIRST) return 1
        if (dirs == 2 && dir != TDirection.BUY_FIRST) return 1
        if (familyCode !in 1..4 || (families and (1 shl (familyCode - 1))) == 0) return 2
        if (elapsed < timeStart || elapsed > timeEnd) return 3
        if ((regimeMask and (1 shl regimeOrd)) == 0) return 4
        if (atrPct < atrMin || atrPct > atrMax) return 5
        if (vwapDevS < vwapDevMin) return 6
        if (withTrend < withTrendMin) return 7
        return 0
    }

    /** 人可读描述;[modelText] 不为空时替换默认的"模型胜率≥阈值"文字 */
    fun describe(modelText: String? = null): String = buildList {
        add(when (dirs) { 1 -> "只先卖后买"; 2 -> "只先买后卖"; else -> "双向" })
        add("信号:" + listOf("缠论1分钟", "布林", "RSI", "缠论5分钟").filterIndexed { i, _ -> (families and (1 shl i)) != 0 }.joinToString("+"))
        add("时间:%s~%s".format(clock(timeStart), clock(timeEnd)))
        if (regimeMask != 31) add("大盘:" + MarketLevel.entries.filter { (regimeMask and (1 shl it.ordinal)) != 0 }.joinToString("/") { it.label })
        if (atrMin > 0 || atrMax < 1) add("ATR %.1f%%~%s".format(atrMin * 100, if (atrMax >= 1) "不限" else "%.1f%%".format(atrMax * 100)))
        if (vwapDevMin > -9) add("均价偏离≥%.2fATR".format(vwapDevMin))
        if (withTrendMin > -9) add("顺势≥%.1f".format(withTrendMin))
        add(if (!useModel) "不用模型" else modelText ?: "模型胜率≥%.3f".format(modelThr))
        add("每日≤${maxTrades}次")
    }.joinToString(",")

    fun serialize(): String =
        "$dirs,$families,$timeStart,$timeEnd,$regimeMask,$atrMin,$atrMax,$vwapDevMin,$withTrendMin,$useModel,$modelThr,$maxTrades"

    companion object {
        fun parse(s: String): StrategySpec {
            val p = s.split(',')
            return StrategySpec(
                p[0].toInt(), p[1].toInt(), p[2].toInt(), p[3].toInt(), p[4].toInt(),
                p[5].toDouble(), p[6].toDouble(), p[7].toDouble(), p[8].toDouble(),
                p[9].toBoolean(), p[10].toDouble(), p[11].toInt(),
            )
        }

        /** 开盘后经过的分钟数 → 时钟 */
        fun clock(elapsed: Int): String {
            val m = if (elapsed <= 120) 9 * 60 + 30 + elapsed else 13 * 60 + (elapsed - 120)
            return "%02d:%02d".format(m / 60, m % 60)
        }

        /** 候选点的信号来源:1 缠论1分钟 2 布林 3 RSI 4 仅缠论5分钟;0 无 */
        fun familyOf(o: Observation, dir: TDirection): Int {
            val sell = dir == TDirection.SELL_FIRST
            return when (if (sell) o.sig.sellFamily else o.sig.buyFamily) {
                SignalFamily.CHAN -> 1
                SignalFamily.BOLL -> 2
                SignalFamily.VWAP -> 3
                SignalFamily.NONE -> if ((if (sell) o.sig.ind[Ind.C5_SELL] else o.sig.ind[Ind.C5_BUY]) > 0) 4 else 0
            }
        }

        fun vwapDevS(o: Observation, dir: TDirection) = (if (dir == TDirection.SELL_FIRST) 1 else -1) * o.sig.vwapDev

        fun withTrend(o: Observation, dir: TDirection): Double {
            val d = o.dist
            val up = d.p(DayType.TREND_UP) + d.p(DayType.RECOVER)
            val down = d.p(DayType.TREND_DOWN) + d.p(DayType.FADE)
            return if (dir == TDirection.SELL_FIRST) down - up else up - down
        }

        /** 同一时刻两个方向都可做时的优先级:不用模型时比较 L4 综合分 */
        fun score(o: Observation, dir: TDirection) =
            if (dir == TDirection.SELL_FIRST) o.sig.sellScore else o.sig.buyScore

        fun allows(s: StrategySpec, ctx: DayContext, o: Observation, dir: TDirection): Boolean =
            s.allows(familyOf(o, dir), dir, o.day.elapsed, ctx.regime.level.ordinal, ctx.atrPct, vwapDevS(o, dir), withTrend(o, dir))
    }
}

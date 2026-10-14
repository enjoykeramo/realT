package realt.core.engine

import realt.core.daytype.DayContext
import realt.core.daytype.DayDist
import realt.core.daytype.DayType
import realt.core.daytype.DayTypePredictor
import realt.core.daytype.describe
import realt.core.daytype.p
import realt.core.daytype.top
import realt.core.model.Bar
import realt.core.model.Session
import realt.core.signal.SignalEngine
import realt.core.signal.SignalFamily
import realt.core.ml.Features
import realt.core.ml.TradeModel
import realt.core.ml.StrategySpec
import realt.core.signal.SignalSnapshot
import kotlin.math.max

/** 做T方向 */
enum class TDirection(val label: String) {
    /** 先卖底仓,低位买回 */
    SELL_FIRST("先卖后买"),
    /** 先低吸,高位卖出等量底仓 */
    BUY_FIRST("先买后卖"),
}

enum class ActionKind(val label: String) {
    OPEN_SELL("高抛"), OPEN_BUY("低吸"),
    CLOSE_BUY("回补"), CLOSE_SELL("卖出"),
}

enum class CloseReason(val label: String) {
    TARGET("止盈"), VWAP("回归均价"), SIGNAL("反向信号"), BREAKEVEN("保本离场"), LATE_LOCK("尾盘锁利"),
    STOP("止损认错"), REGIME_FLIP("形态逆转"), TIME("尾盘收尾"),
}

/** 引擎给出的操作建议(实时场景下即为推送内容) */
data class TAction(
    val kind: ActionKind,
    val date: Int,
    val minute: Int,
    val price: Double,
    val shares: Int,
    val reason: String,
    val closeReason: CloseReason? = null,
    /** 限价;null 为市价(下一根开盘成交) */
    val limitPrice: Double? = null,
    /** 已在本根 K 线内以限价成交(止盈挂单被触及) */
    val fillNow: Boolean = false,
    /** 限价开仓单的有效期(开盘后经过的分钟数,含);-1 表示不限 */
    val validUntil: Int = -1,
)

/**
 * 一次做T开仓时的上下文"坐标",L6 按它统计历史胜率(迷宫的"路口")。
 */
data class ContextKey(
    val regime: String,
    val stage: String,
    val dayType: DayType,
    val direction: TDirection,
    val family: SignalFamily,
) {
    override fun toString() = "$regime|$stage|${dayType.name}|${direction.name}|${family.name}"
}

data class TConfig(
    /** T 仓占底仓比例 */
    val tRatio: Double = 0.3,
    val maxTradesPerDay: Int = 2,
    /** 开盘后多少分钟内不开新仓 */
    val noOpenAfterOpenMin: Int = 5,
    /** 收盘前多少分钟不再开新仓 */
    val noOpenBeforeCloseMin: Int = 60,
    /** 收盘前多少分钟强制收尾(恢复底仓) */
    val forceCloseBeforeCloseMin: Int = 6,
    /** 开仓信号阈值 */
    val entryThreshold: Double = 0.55,
    /** 平仓信号阈值 */
    val exitThreshold: Double = 0.45,
    /** 止盈距离(日 ATR 倍数) */
    val takeProfitAtr: Double = 0.45,
    /** 止损距离(日 ATR 倍数) */
    val stopLossAtr: Double = 0.6,
    /** 近 30 分钟位移超过该值(日 ATR)视为单边动量,不逆势开仓 */
    val momentumBlock: Double = 0.15,
    /** 最小期望价差 = max(minEdgePct, edgeCostMultiple × 来回成本) */
    val minEdgePct: Double = 0.004,
    val edgeCostMultiple: Double = 3.0,
    /** 偏多(单边上涨+探底回升)或偏空(单边下跌+冲高回落)概率超过该值时,禁止逆势方向 */
    val trendBlock: Double = 0.55,
    /** 窄幅横盘概率超过该值时不做 */
    val flatBlock: Double = 0.45,
    /** 持有超过该分钟数仍无进展,且盘面偏向不利单边时离场 */
    val maxHoldMin: Int = 90,
    /** 收盘前多少分钟进入"尾盘锁利":只要扣费后有盈利就立即完成T,不再等止盈;0 关闭 */
    val lateLockMin: Int = 30,
    /** 是否启用形态逆转/时间止损离场(默认关闭,用 sweep 在真实数据上检验) */
    val regimeExit: Boolean = false,
    /** 平仓后冷却分钟数 */
    val cooldownMin: Int = 10,
    /** 挂单开仓:先卖时挂在现价上方、先买时挂在现价下方的距离(日 ATR 倍数);0 = 市价 */
    val entryOffsetAtr: Double = 0.0,
    /** 挂单开仓的有效分钟数 */
    val entryWaitMin: Int = 5,
    /** 止盈以限价单挂出:K 线触及即按止盈价成交(否则按收盘判断、下一根开盘成交) */
    val tpLimit: Boolean = true,
    val allowSellFirst: Boolean = true,
    val allowBuyFirst: Boolean = true,
)

/** 一次完整的做T(开 + 平) */
data class TTrade(
    val date: Int,
    val direction: TDirection,
    val shares: Int,
    val openMinute: Int,
    val openPrice: Double,
    val closeMinute: Int,
    val closePrice: Double,
    val closeReason: CloseReason,
    val context: ContextKey,
    val openNote: String,
) {
    /** 未扣费价差收益 */
    val grossPnl: Double
        get() = when (direction) {
            TDirection.SELL_FIRST -> (openPrice - closePrice) * shares
            TDirection.BUY_FIRST -> (closePrice - openPrice) * shares
        }

    fun netPnl(costs: Costs): Double {
        val sellPx = if (direction == TDirection.SELL_FIRST) openPrice else closePrice
        val buyPx = if (direction == TDirection.SELL_FIRST) closePrice else openPrice
        return grossPnl - costs.sellFee(sellPx * shares) - costs.buyFee(buyPx * shares)
    }

    fun returnPct(costs: Costs): Double = netPnl(costs) / (openPrice * shares)
}

/**
 * 今日截至当前的分时状态(增量维护,O(1) 更新),作为模型特征的一部分。
 * 价格类距离均以日 ATR 为单位。
 */
data class DayState(
    val elapsed: Int,
    val open: Double,
    val high: Double,
    val low: Double,
    val close: Double,
    /** 收盘在均价线之上的分钟占比 */
    val fracAbove: Double,
    /** 均价线近 30 分钟的位移(ATR) */
    val vwapSlope30: Double,
    /** 最近 5 分钟成交量 / 今日每分钟平均量 */
    val volRatio5: Double,
    /** 日内最高、最低点出现的时刻(占已过时间比例) */
    val highAt: Double,
    val lowAt: Double,
    /** 近 15 分钟 (上涨分钟成交量 - 下跌分钟成交量) / 总量,-1..1 */
    val flow15: Double = 0.0,
    /** 近 30 分钟 1 分钟收益的标准差 × √240 / 日 ATR(短期波动是否放大) */
    val rv30: Double = 0.0,
    /** 指数(同一分钟):相对昨收涨跌 %、相对其均价线 %、近 30 分钟涨跌 %;无指数分钟数据时为 0 */
    val idxRet: Double = 0.0,
    val idxVsVwap: Double = 0.0,
    val idxSlope30: Double = 0.0,
    /** 是否有指数分钟数据 */
    val hasIndex: Boolean = false,
)

/** 某一分钟的"观察":分时形态后验 + 短线信号 + 分时状态。与 [TConfig] 无关,可被多组参数共享。 */
class Observation(val dist: DayDist, val sig: SignalSnapshot, val day: DayState)

/**
 * 观察者:维护当日 1 分钟线序列,计算 L3 形态后验与 L4 信号。
 * [history] 为今日之前的若干日 1 分钟线(给缠论/布林提供结构上下文)。
 */
class Observer(
    private val ctx: DayContext,
    history: List<Bar>,
    private val predictor: DayTypePredictor = DayTypePredictor(),
    private val signals: SignalEngine = SignalEngine(),
    /** 当日指数 1 分钟线(可为空);只会读取不晚于当前分钟的部分 */
    private val indexBars: List<Bar> = emptyList(),
    /** 指数昨收 */
    private val indexPrevClose: Double = 0.0,
) {
    init {
        for (b in history) signals.warm(b)
    }
    private var idxPtr = 0
    private var idxPv = 0.0
    private var idxV = 0.0
    private val idxCloses = ArrayList<Double>()
    private val bars = ArrayList<Bar>(history)
    private val todayStart = history.size
    private var prior: DayDist? = null
    val atrAbs = max(ctx.atrPct, 0.005) * ctx.prevClose
    private val vwaps = ArrayList<Double>()
    private val vols = ArrayList<Double>()
    private var hi = Double.NEGATIVE_INFINITY
    private var lo = Double.POSITIVE_INFINITY
    private var hiIdx = 0
    private var loIdx = 0
    private var above = 0
    private var volSum = 0.0

    fun observe(bar: Bar): Observation {
        bars += bar
        val today = bars.subList(todayStart, bars.size)
        if (prior == null) prior = predictor.prior(ctx, today.first().open)
        val dist = predictor.posterior(prior!!, ctx, today)
        val sig = signals.update(bar, atrAbs)

        val i = today.size - 1
        if (bar.high > hi) { hi = bar.high; hiIdx = i }
        if (bar.low < lo) { lo = bar.low; loIdx = i }
        if (bar.close > sig.vwap) above++
        vwaps += sig.vwap
        vols += bar.volume
        volSum += bar.volume
        val n = today.size
        val v5 = (maxOf(0, n - 5) until n).sumOf { vols[it] } / minOf(5, n)
        // 资金流向近似:近 15 分钟上涨/下跌分钟的成交量差
        var up = 0.0; var dn = 0.0
        for (k in maxOf(0, n - 15) until n) {
            val b = today[k]
            if (b.close > b.open) up += b.volume else if (b.close < b.open) dn += b.volume
        }
        var ss = 0.0; var cnt = 0
        for (k in maxOf(1, n - 30) until n) {
            val r = today[k].close / today[k - 1].close - 1
            ss += r * r; cnt++
        }
        val rv = if (cnt > 1) kotlin.math.sqrt(ss / cnt * 240) / ctx.atrPct.coerceAtLeast(0.005) else 0.0
        // 指数:推进到当前分钟(不看未来)
        while (idxPtr < indexBars.size && indexBars[idxPtr].minute <= bar.minute) {
            val ib = indexBars[idxPtr]
            idxPv += (ib.high + ib.low + ib.close) / 3 * ib.volume
            idxV += ib.volume
            idxCloses += ib.close
            idxPtr++
        }
        val hasIdx = idxCloses.isNotEmpty() && indexPrevClose > 0
        val ic = if (hasIdx) idxCloses.last() else 0.0
        val day = DayState(
            elapsed = Session.elapsed(bar.minute),
            open = today.first().open, high = hi, low = lo, close = bar.close,
            fracAbove = above.toDouble() / n,
            vwapSlope30 = (sig.vwap - vwaps[maxOf(0, n - 31)]) / atrAbs,
            volRatio5 = if (volSum > 0) v5 / (volSum / n) else 1.0,
            highAt = hiIdx.toDouble() / n, lowAt = loIdx.toDouble() / n,
            flow15 = if (up + dn > 0) (up - dn) / (up + dn) else 0.0,
            rv30 = rv,
            idxRet = if (hasIdx) (ic / indexPrevClose - 1) * 100 else 0.0,
            idxVsVwap = if (hasIdx && idxV > 0) (ic / (idxPv / idxV) - 1) * 100 else 0.0,
            idxSlope30 = if (hasIdx) (ic / idxCloses[maxOf(0, idxCloses.size - 31)] - 1) * 100 else 0.0,
            hasIndex = hasIdx,
        )
        return Observation(dist, sig, day)
    }
}

/**
 * L5:做T决策与风控状态机(单只股票、单个交易日)。
 *
 * 状态:FLAT → (OPEN_SELL | OPEN_BUY) → FLAT …
 * 实时:每根 1 分钟 bar 调用 [onBar];引擎返回建议,调用方成交后调用 [onFill] 告知真实成交价。
 * 回测:由回放器在下一根 bar 开盘价成交;参数扫描时多个引擎共享同一 [Observation],调用 [decide]。
 *
 * [gate] 为 L6 的"迷宫记忆":返回 false 表示该上下文历史上是死路,不出手。
 */
class TEngine(
    private val config: TConfig,
    private val costs: Costs,
    private val ctx: DayContext,
    baseShares: Int,
    private val observer: Observer? = null,
    private val gate: (ContextKey) -> Boolean = { true },
    /** 训练产出的参数包;非空时由模型胜率决定是否出手(替代规则打分与 L6 门控) */
    private val model: TradeModel? = null,
) : IntradayStrategy {
    constructor(
        config: TConfig, costs: Costs, ctx: DayContext, baseShares: Int, history: List<Bar>,
        predictor: DayTypePredictor = DayTypePredictor(), signals: SignalEngine = SignalEngine(),
        gate: (ContextKey) -> Boolean = { true },
        model: TradeModel? = null,
        indexBars: List<Bar> = emptyList(),
        indexPrevClose: Double = 0.0,
    ) : this(config, costs, ctx, baseShares, Observer(ctx, history, predictor, signals, indexBars, indexPrevClose), gate, model)

    /** 当前持仓(方向, 开仓价);无持仓为 null */
    val position: Pair<TDirection, Double>? get() = openDir?.let { it to openPrice }

    var dist: DayDist = emptyMap(); private set
    var lastSignal: SignalSnapshot? = null; private set

    private val tShares = (baseShares * config.tRatio / 100).toInt() * 100
    private val atrAbs = max(ctx.atrPct, 0.005) * ctx.prevClose
    /** 本笔T仓规模下的实际来回成本比例(含最低 5 元佣金) */
    private val costRate = costs.roundTripRateAt(tShares * ctx.prevClose)
    private val minEdge = max(config.minEdgePct, config.edgeCostMultiple * costRate)

    private var pending: TAction? = null
    private var openDir: TDirection? = null
    private var openPrice = 0.0
    private var openMinute = 0
    private var openKey: ContextKey? = null
    private var openNote = ""
    private var maxGain = 0.0
    private var openByLimit = false
    private var lastCloseElapsed = -1000
    override val trades = ArrayList<TTrade>()
    override val log = ArrayList<String>()

    val isOpen: Boolean get() = openDir != null

    /** 当天是否具备做T条件(波动足以覆盖成本) */
    val tradable: Boolean get() = tShares > 0 && ctx.atrPct * config.takeProfitAtr >= minEdge * 0.8

    override fun onBar(bar: Bar): TAction? =
        decide(bar, (observer ?: error("未提供 Observer,请使用 decide()")).observe(bar))

    fun decide(bar: Bar, obs: Observation): TAction? {
        dist = obs.dist
        lastSignal = obs.sig
        val sig = obs.sig
        val elapsed = Session.elapsed(bar.minute)
        val remaining = Session.MINUTES_PER_DAY - elapsed
        val px = bar.close

        // 持仓中:无论何种情况都要管理离场
        val dir = openDir
        if (dir != null) {
            return manageOpen(dir, bar, sig, remaining)
        }
        // 规则模式要求当日波动足以覆盖成本;模型模式由模型(含 atrPct 特征)自行判断
        if (tShares <= 0 || (model == null && !tradable)) return null

        // ---- 开新仓 ----
        if (trades.size >= config.maxTradesPerDay) return null
        if (elapsed < config.noOpenAfterOpenMin || remaining <= config.noOpenBeforeCloseMin) return null
        if (elapsed - lastCloseElapsed < config.cooldownMin) return null
        if (model != null) return decideByModel(bar, obs, model)
        if (dist.p(DayType.FLAT) > config.flatBlock) return null

        // 冲高回落的后半程等同下跌、探底回升的后半程等同上涨,一并计入方向判断
        val upish = upish()
        val downish = downish()
        val tilt = upish - downish

        // 顺势方向门槛降低,逆势方向门槛提高
        val sellTh = config.entryThreshold + 0.25 * tilt
        val buyTh = config.entryThreshold - 0.25 * tilt
        // 动量过滤:30 分钟内单边位移过大时不逆势(除非 5 分钟级别已确认转折)
        val sellMomOk = sig.trend5 < config.momentumBlock || sig.sellConfirm >= 1.0
        val buyMomOk = sig.trend5 > -config.momentumBlock || sig.buyConfirm >= 1.0
        val sellOk = config.allowSellFirst && upish < config.trendBlock && sellMomOk && sig.sellScore >= sellTh
        val buyOk = config.allowBuyFirst && downish < config.trendBlock && buyMomOk && sig.buyScore >= buyTh

        // 预期空间:到均价线/对侧的距离至少覆盖 minEdge
        val roomSell = (px - minOf(sig.vwap, px - config.takeProfitAtr * atrAbs)) / px
        val roomBuy = (maxOf(sig.vwap, px + config.takeProfitAtr * atrAbs) - px) / px

        val choice: TDirection = when {
            sellOk && buyOk -> if (sig.sellScore - sellTh >= sig.buyScore - buyTh) TDirection.SELL_FIRST else TDirection.BUY_FIRST
            sellOk -> TDirection.SELL_FIRST
            buyOk -> TDirection.BUY_FIRST
            else -> null
        } ?: return null
        if (choice == TDirection.SELL_FIRST && roomSell < minEdge) return null
        if (choice == TDirection.BUY_FIRST && roomBuy < minEdge) return null

        val family = if (choice == TDirection.SELL_FIRST) sig.sellFamily else sig.buyFamily
        val key = ContextKey(ctx.regime.level.name, ctx.stage.top.name, dist.top(), choice, family)
        if (!gate(key)) {
            log += "${bar.hhmm} 跳过 $key(历史统计为死路)"
            return null
        }
        val score = if (choice == TDirection.SELL_FIRST) sig.sellScore else sig.buyScore
        val reason = "%s 分%.2f | %s | 形态:%s".format(choice.label, score, sig.note.ifEmpty { "均价偏离%.2fATR".format(sig.vwapDev) }, dist.describe())
        val action = TAction(
            if (choice == TDirection.SELL_FIRST) ActionKind.OPEN_SELL else ActionKind.OPEN_BUY,
            bar.date, bar.minute, px, tShares, reason,
        )
        pending = action
        openKey = key
        openNote = reason
        return action
    }

    /** 模型模式:对每个候选方向计算胜率概率,取最高且 ≥ 阈值者 */
    private fun decideByModel(bar: Bar, obs: Observation, m: TradeModel): TAction? {
        val st = m.strategy
        var best: TDirection? = null
        var bestP = Double.NEGATIVE_INFINITY
        var bestWin = Double.NaN
        for (d in Features.candidates(obs)) {
            if (d == TDirection.SELL_FIRST && !config.allowSellFirst) continue
            if (d == TDirection.BUY_FIRST && !config.allowBuyFirst) continue
            if (st != null && !StrategySpec.allows(st, ctx, obs, d)) continue
            val useModel = st == null || st.useModel
            val win = if (useModel) m.winProb(Features.extract(ctx, obs, d)) else Double.NaN
            if (useModel && win < (st?.modelThr ?: m.threshold)) continue
            // 优先级:用模型时取胜率高者,否则取 L4 综合分高者(与训练时一致)
            val pr = if (useModel) win else StrategySpec.score(obs, d)
            if (pr > bestP) { bestP = pr; best = d; bestWin = win }
        }
        val choice = best ?: return null
        val sig = obs.sig
        val note = "%s %s| %s | 形态:%s".format(
            choice.label, if (bestWin.isNaN()) "" else "模型胜率%.0f%% ".format(bestWin * 100),
            sig.note.ifEmpty { "均价偏离%.2fATR".format(sig.vwapDev) }, dist.describe())
        return open(choice, bar, note)
    }

    /** 不经任何过滤直接按 [dir] 开仓(训练时给每个候选点打标签用) */
    fun forceOpen(dir: TDirection, bar: Bar, obs: Observation, note: String = "候选"): TAction {
        dist = obs.dist
        lastSignal = obs.sig
        return open(dir, bar, note)
    }

    private fun open(dir: TDirection, bar: Bar, note: String): TAction {
        val fam = if (dir == TDirection.SELL_FIRST) lastSignal?.sellFamily else lastSignal?.buyFamily
        val sell = dir == TDirection.SELL_FIRST
        val limit = if (config.entryOffsetAtr > 0) {
            val off = config.entryOffsetAtr * atrAbs
            if (sell) Fills.ceilTick(bar.close + off) else Fills.floorTick(bar.close - off)
        } else null
        val action = TAction(
            if (sell) ActionKind.OPEN_SELL else ActionKind.OPEN_BUY,
            bar.date, bar.minute, limit ?: bar.close, tShares,
            if (limit != null) note + " | 挂单 " + "%.2f".format(limit) else note,
            limitPrice = limit,
            validUntil = if (limit != null) Session.elapsed(bar.minute) + config.entryWaitMin else -1,
        )
        pending = action
        openKey = ContextKey(ctx.regime.level.name, ctx.stage.top.name, dist.top(), dir, fam ?: SignalFamily.NONE)
        openNote = note
        return action
    }

    private fun upish() = dist.p(DayType.TREND_UP) + dist.p(DayType.RECOVER)
    private fun downish() = dist.p(DayType.TREND_DOWN) + dist.p(DayType.FADE)

    private fun manageOpen(dir: TDirection, bar: Bar, sig: SignalSnapshot, remaining: Int): TAction? {
        val px = bar.close
        val sell = dir == TDirection.SELL_FIRST
        // 有利方向的浮动(价格比例)
        val gain = if (sell) (openPrice - px) / openPrice else (px - openPrice) / openPrice
        val adverse = if (sell) (bar.high - openPrice) else (openPrice - bar.low)
        val best = if (sell) (openPrice - bar.low) / openPrice else (bar.high - openPrice) / openPrice
        if (best > maxGain) maxGain = best
        val held = Session.elapsed(bar.minute) - Session.elapsed(openMinute)
        val oppTrend = if (sell) upish() else downish()

        // 止盈挂单:本根 K 线触及止盈价即按止盈价成交(挂单开仓成交的那一根不判断,避免同一根内先后顺序不明)
        if (config.tpLimit && remaining > config.forceCloseBeforeCloseMin && adverse < config.stopLossAtr * atrAbs &&
            !(openByLimit && bar.minute == openMinute)
        ) {
            val tp = if (sell) Fills.floorTick(openPrice - config.takeProfitAtr * atrAbs)
            else Fills.ceilTick(openPrice + config.takeProfitAtr * atrAbs)
            val touched = if (sell) bar.low <= tp - Fills.TICK else bar.high >= tp + Fills.TICK
            if (touched) {
                val a = TAction(
                    if (sell) ActionKind.CLOSE_BUY else ActionKind.CLOSE_SELL,
                    bar.date, bar.minute, tp, tShares, "止盈挂单成交 %.2f".format(tp), CloseReason.TARGET,
                    limitPrice = tp, fillNow = true,
                )
                pending = a
                return a
            }
        }
        val reason: CloseReason? = when {
            remaining <= config.forceCloseBeforeCloseMin -> CloseReason.TIME
            adverse >= config.stopLossAtr * atrAbs -> CloseReason.STOP
            !config.tpLimit && gain * openPrice >= config.takeProfitAtr * atrAbs -> CloseReason.TARGET
            // 均值回归目标:价格回到均价线且已覆盖成本
            gain >= minEdge && (if (sell) px <= sig.vwap else px >= sig.vwap) -> CloseReason.VWAP
            gain >= minEdge && (if (sell) sig.buyScore else sig.sellScore) >= config.exitThreshold -> CloseReason.SIGNAL
            // 保本:曾经浮盈超过 2 倍 minEdge,回吐到只剩 minEdge/2 时离场,不让盈利单变亏损单
            maxGain >= 2 * minEdge && gain <= minEdge * 0.5 -> CloseReason.BREAKEVEN
            // 当日T必须当日完成:临近收盘,只要已覆盖成本就落袋,避免被迫按尾盘价收尾
            remaining <= config.lateLockMin && gain >= costRate * 1.2 -> CloseReason.LATE_LOCK
            // 时间止损:持有过久且已亏损,盘面明显偏向不利的单边
            config.regimeExit && held >= config.maxHoldMin && gain < -minEdge * 0.5 && oppTrend > 0.6 -> CloseReason.REGIME_FLIP
            // 形态被确认为与持仓相反的单边:立即认错
            // 形态已被确认为与持仓方向相反的单边,且已出现实质亏损:认错
            config.regimeExit && gain < -minEdge && oppTrend > 0.7 -> CloseReason.REGIME_FLIP
            else -> null
        }
        reason ?: return null
        val action = TAction(
            if (sell) ActionKind.CLOSE_BUY else ActionKind.CLOSE_SELL,
            bar.date, bar.minute, px, tShares,
            "%s %s 浮动%+.2f%%".format(reason.label, if (sell) "买回" else "卖出", gain * 100),
            reason,
        )
        pending = action
        return action
    }

    /** 通知引擎上一条建议已按 [price] 成交 */
    override fun onFill(action: TAction, price: Double, minute: Int) {
        if (pending != action) return
        pending = null
        when (action.kind) {
            ActionKind.OPEN_SELL, ActionKind.OPEN_BUY -> {
                openDir = if (action.kind == ActionKind.OPEN_SELL) TDirection.SELL_FIRST else TDirection.BUY_FIRST
                openPrice = price
                openMinute = minute
                openByLimit = action.limitPrice != null
                maxGain = 0.0
                log += "${hhmm(minute)} ${action.kind.label} ${action.shares}股 @%.2f  %s".format(price, action.reason)
            }
            ActionKind.CLOSE_BUY, ActionKind.CLOSE_SELL -> {
                val t = TTrade(
                    action.date, openDir!!, action.shares, openMinute, openPrice,
                    minute, price, action.closeReason!!, openKey!!, openNote,
                )
                trades += t
                log += "${hhmm(minute)} ${action.kind.label} ${action.shares}股 @%.2f  %s 净%+.0f元".format(price, action.reason, t.netPnl(costs))
                openDir = null
                lastCloseElapsed = Session.elapsed(minute)
            }
        }
    }

    override fun forceClose(bar: Bar): TAction? {
        val dir = openDir ?: return null
        val a = TAction(
            if (dir == TDirection.SELL_FIRST) ActionKind.CLOSE_BUY else ActionKind.CLOSE_SELL,
            bar.date, bar.minute, bar.close, tShares, "数据结束强制收尾", CloseReason.TIME,
        )
        pending = a
        return a
    }

    override fun onCancel(action: TAction) {
        if (pending == action) pending = null
    }

    /** 建议未成交(例如实时场景用户放弃),撤销 */
    fun cancelPending() {
        pending = null
    }

    private fun hhmm(m: Int) = "%02d:%02d".format(m / 60, m % 60)
}

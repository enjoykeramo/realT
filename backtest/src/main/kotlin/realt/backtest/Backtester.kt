package realt.backtest

import realt.core.daytype.DayContext
import realt.core.daytype.DayDist
import realt.core.daytype.DayType
import realt.core.daytype.DayTypeLabeler
import realt.core.daytype.DayTypePredictor
import realt.core.daytype.top
import realt.core.engine.BaselineEngine
import realt.core.engine.Costs
import realt.core.engine.IntradayStrategy
import realt.core.engine.TAction
import realt.core.engine.TConfig
import realt.core.engine.TEngine
import realt.core.engine.Observer
import realt.core.ml.TradeModel
import realt.core.ml.DailyFeatures
import realt.core.engine.TTrade
import realt.core.engine.ActionKind
import realt.core.indicator.Indicators
import realt.core.model.Bar
import realt.core.model.Session
import realt.core.model.Symbol
import realt.core.model.byDay
import realt.core.regime.MarketLevel
import realt.core.stage.Stage
import realt.core.stage.StageDetector

data class BacktestConfig(
    val startDate: Int = 0,
    val endDate: Int = 99999999,
    /** 每只股票的底仓市值(元),按当日昨收折算股数 */
    val baseValue: Double = 100_000.0,
    val tConfig: TConfig = TConfig(),
    val costs: Costs = Costs(),
    /** 作为结构上下文的前几日 1 分钟线 */
    val historyDays: Int = 1,
    /** 一天至少需要多少根 1 分钟线才参与回测 */
    val minBarsPerDay: Int = 200,
    /** 训练产出的参数包;非空时由模型决定出手 */
    val model: TradeModel? = null,
)

/** 某只股票某个交易日回测所需的全部输入(只含当日开盘前可知的信息 + 当日分钟线) */
class DaySetup(
    val symbol: Symbol,
    val date: Int,
    val dayBars: List<Bar>,
    val ctx: DayContext,
    val shares: Int,
    val history: List<Bar>,
    /** 参考指数当日 1 分钟线与昨收(可为空) */
    val indexBars: List<Bar> = emptyList(),
    val indexPrevClose: Double = 0.0,
    /** 下一个交易日的 1 分钟线(训练时评估"隔日回补"用;除权或无数据为 null) */
    val nextBars: List<Bar>? = null,
) {
    fun observer(predictor: DayTypePredictor = DayTypePredictor()) =
        Observer(ctx, history, predictor, indexBars = indexBars, indexPrevClose = indexPrevClose)
}

/** 某只股票某一天的回测记录 */
data class DayRecord(
    val symbol: Symbol,
    val date: Int,
    val regime: MarketLevel,
    val stage: Stage,
    val actual: DayType,
    val prior: DayType,
    /** 开盘 30 / 60 / 120 分钟时的预判 */
    val pred30: DayType,
    val pred60: DayType,
    val pred120: DayType,
    val tradable: Boolean,
    val trades: List<TTrade>,
    val baselineTrades: List<TTrade>,
    val log: List<String>,
    /** 参数扫描时每组参数的做T结果(下标与 variants 对应,0 为主参数) */
    val variantTrades: List<List<TTrade>> = listOf(trades),
)

/**
 * 回放器:逐日、逐分钟把历史 1 分钟线喂给策略,建议在下一根 bar 开盘价(含滑点)成交。
 * 这里跑的是"无记忆门控"的 realT 与原型对照组;L6 记忆门控在 [MemoryReplay] 中按时间顺序叠加。
 */
class Backtester(
    private val data: MarketData,
    private val cfg: BacktestConfig,
    private val variants: List<TConfig> = listOf(cfg.tConfig),
) {

    private val stageDetector = StageDetector()

    /** 逐日生成回测输入;数据不足、除权日等会被跳过 */
    fun setups(symbol: Symbol): Sequence<DaySetup> = sequence {
        val daily = data.reader.daily(symbol)
        val minutes = data.reader.minute1(symbol)
        if (daily.size < 140 || minutes.isEmpty()) return@sequence
        val days = minutes.byDay()
        val dates = days.keys.toList()
        for ((di, date) in dates.withIndex()) {
            if (date < cfg.startDate || date > cfg.endDate) continue
            val dayBars = days[date]!!
            if (dayBars.size < cfg.minBarsPerDay) continue
            val before = daily.filter { it.date < date }.takeLast(260)
            if (before.size < 140) continue
            val prevClose = before.last().close
            // 除权日:分钟线开盘与昨收差异超出涨跌停幅度,说明价格基准不一致,跳过
            if (kotlin.math.abs(dayBars.first().open / prevClose - 1) > symbol.limitPct + 0.01) continue
            val atrPct = Indicators.atr(before, 14).last() / prevClose
            val ctx = DayContext(
                prevClose, atrPct, data.regimeBefore(date), stageDetector.detect(before),
                symbol.limitPct, DailyFeatures.compute(before),
            )
            val shares = ((cfg.baseValue / prevClose) / 100).toInt() * 100
            val history = (maxOf(0, di - cfg.historyDays) until di).flatMap { days[dates[it]]!! }
            val (idxBars, idxPrev) = data.indexDay(symbol, date)
            val next = dates.getOrNull(di + 1)?.let { days[it] }
                ?.takeIf { it.size >= cfg.minBarsPerDay && kotlin.math.abs(it.first().open / dayBars.last().close - 1) <= symbol.limitPct + 0.01 }
            yield(DaySetup(symbol, date, dayBars, ctx, shares, history, idxBars, idxPrev, next))
        }
    }

    fun run(symbol: Symbol): List<DayRecord> {
        val out = ArrayList<DayRecord>()
        for (su in setups(symbol)) {
            val date = su.date
            val dayBars = su.dayBars
            val ctx = su.ctx
            val shares = su.shares
            val prevClose = ctx.prevClose
            val atrPct = ctx.atrPct
            val regime = ctx.regime
            val stage = ctx.stage
            val history = su.history

            val predictor = DayTypePredictor()
            val observer = su.observer(predictor)
            val engines = variants.map { TEngine(it, cfg.costs, ctx, shares, null, model = cfg.model) }
            val pendings = arrayOfNulls<TAction>(engines.size)
            val preds = HashMap<Int, DayDist>()
            for (bar in dayBars) {
                val obs = observer.observe(bar)
                for ((k, e) in engines.withIndex()) {
                    pendings[k]?.let { fill(e, it, bar.open, bar.minute) }
                    pendings[k] = e.decide(bar, obs)
                }
                val el = Session.elapsed(bar.minute)
                if (el == 30 || el == 60 || el == 120) preds[el] = obs.dist
            }
            val lastBar = dayBars.last()
            for ((k, e) in engines.withIndex()) {
                pendings[k]?.let { fill(e, it, lastBar.close, lastBar.minute) }
                e.forceClose(lastBar)?.let { fill(e, it, lastBar.close, lastBar.minute) }
            }
            val engine = engines[0]
            val baseline = BaselineEngine(prevClose, shares, cfg.tConfig.tRatio, cfg.costs)
            replay(baseline, dayBars) {}

            val prior = predictor.prior(ctx, dayBars.first().open)
            out += DayRecord(
                symbol, date, regime.level, stage.top,
                DayTypeLabeler.label(dayBars, prevClose, atrPct),
                prior.top(),
                (preds[30] ?: prior).top(), (preds[60] ?: prior).top(), (preds[120] ?: prior).top(),
                engine.tradable, engine.trades.toList(), baseline.trades.toList(), engine.log.toList(),
                engines.map { it.trades.toList() },
            )
        }
        return out
    }

    /** 下一根开盘成交;最后一根的建议按收盘价成交 */
    private fun replay(s: IntradayStrategy, dayBars: List<Bar>, onBar: (Bar) -> Unit) {
        var pending: TAction? = null
        for (bar in dayBars) {
            pending?.let { fill(s, it, bar.open, bar.minute) }
            pending = s.onBar(bar)
            onBar(bar)
        }
        val last = dayBars.last()
        pending?.let { fill(s, it, last.close, last.minute) }
        s.forceClose(last)?.let { fill(s, it, last.close, last.minute) }
    }

    private fun fill(s: IntradayStrategy, a: TAction, px: Double, minute: Int) {
        val slip = cfg.costs.slippage
        val buy = a.kind == ActionKind.OPEN_BUY || a.kind == ActionKind.CLOSE_BUY
        s.onFill(a, if (buy) px * (1 + slip) else px * (1 - slip), minute)
    }
}

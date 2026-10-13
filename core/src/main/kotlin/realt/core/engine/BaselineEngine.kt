package realt.core.engine

import realt.core.daytype.DayType
import realt.core.indicator.Indicators
import realt.core.model.Bar
import realt.core.model.Session
import realt.core.signal.SignalFamily
import kotlin.math.abs

/**
 * 对照组:复刻 stockAPP 原型《做T规则》页面的规则,用于与新算法比较。
 *
 *  卖T(满足任意 2 项):价格高于均价线 ≥1.5%;分时 RSI(6) ≥80;冲高量比 ≤0.8
 *  买T(满足任意 2 项):回落均价线 ±0.3% 且缩量;RSI(6) ≤25;触及日内前低/昨收 ±0.2%
 *  每日最多 2 次;开盘 5 分钟、尾盘 10 分钟不提示;先卖后买。
 *  原型没有止损;为了让账目可比,收盘前统一按市价买回恢复底仓。
 */
class BaselineEngine(
    private val prevClose: Double,
    baseShares: Int,
    tRatio: Double = 0.3,
    private val costs: Costs = Costs(),
) : IntradayStrategy {
    private val tShares = (baseShares * tRatio / 100).toInt() * 100
    private val today = ArrayList<Bar>()
    private var pending: TAction? = null
    private var openPrice = 0.0
    private var openMinute = 0
    private var isOpen = false
    override val trades = ArrayList<TTrade>()
    override val log = ArrayList<String>()

    private val key = ContextKey("-", "-", DayType.RANGE, TDirection.SELL_FIRST, SignalFamily.VWAP)

    override fun onBar(bar: Bar): TAction? {
        today += bar
        if (tShares <= 0) return null
        val elapsed = Session.elapsed(bar.minute)
        val remaining = Session.MINUTES_PER_DAY - elapsed
        val px = bar.close
        val vwap = Indicators.vwap(today).last()
        val rsi = Indicators.rsi(DoubleArray(today.size) { today[it].close }, 6).last()
        val n = today.size
        val volRatio = if (n >= 10) {
            today.subList(n - 5, n).sumOf { it.volume } / today.subList(n - 10, n - 5).sumOf { it.volume }.coerceAtLeast(1.0)
        } else 1.0

        if (isOpen) {
            if (remaining <= 6) return emit(ActionKind.CLOSE_BUY, bar, "尾盘恢复底仓", CloseReason.TIME)
            val dayLow = today.minOf { it.low }
            var hits = 0
            if (abs(px / vwap - 1) <= 0.003 && volRatio < 1.0) hits++
            if (rsi <= 25) hits++
            if (abs(px / dayLow - 1) <= 0.002 || abs(px / prevClose - 1) <= 0.002) hits++
            return if (hits >= 2) emit(ActionKind.CLOSE_BUY, bar, "原型买T信号", CloseReason.SIGNAL) else null
        }
        if (trades.size >= 2 || elapsed < 5 || remaining <= 10) return null
        var hits = 0
        if (px / vwap - 1 >= 0.015) hits++
        if (rsi >= 80) hits++
        if (volRatio <= 0.8) hits++
        return if (hits >= 2) emit(ActionKind.OPEN_SELL, bar, "原型卖T信号", null) else null
    }

    override fun forceClose(bar: Bar): TAction? =
        if (isOpen) emit(ActionKind.CLOSE_BUY, bar, "数据结束强制收尾", CloseReason.TIME) else null

    private fun emit(kind: ActionKind, bar: Bar, reason: String, cr: CloseReason?): TAction {
        val a = TAction(kind, bar.date, bar.minute, bar.close, tShares, reason, cr)
        pending = a
        return a
    }

    override fun onFill(action: TAction, price: Double, minute: Int) {
        if (pending != action) return
        pending = null
        if (action.kind == ActionKind.OPEN_SELL) {
            isOpen = true; openPrice = price; openMinute = minute
        } else {
            val t = TTrade(action.date, TDirection.SELL_FIRST, action.shares, openMinute, openPrice, minute, price,
                action.closeReason!!, key, "原型规则")
            trades += t
            log += "原型 %d 卖%.2f 买%.2f 净%+.0f".format(action.date, openPrice, price, t.netPnl(costs))
            isOpen = false
        }
    }
}

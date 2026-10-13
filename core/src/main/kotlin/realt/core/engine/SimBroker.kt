package realt.core.engine

import realt.core.model.Bar
import realt.core.model.Session
import kotlin.math.ceil
import kotlin.math.floor

/** 成交规则(回测与训练共用) */
object Fills {
    const val TICK = 0.01

    fun floorTick(x: Double) = floor(x / TICK + 1e-9) * TICK
    fun ceilTick(x: Double) = ceil(x / TICK - 1e-9) * TICK

    private fun isBuy(a: TAction) = a.kind == ActionKind.OPEN_BUY || a.kind == ActionKind.CLOSE_BUY

    /**
     * 订单在 [bar] 上能否成交,返回成交价;不能成交返回 null。
     *  - 市价单:按开盘价 ± 滑点成交
     *  - 限价单:开盘即优于限价则按开盘价;盘中价格须穿过限价至少 1 个最小价位才算成交(考虑排队),按限价成交
     */
    fun tryFill(a: TAction, bar: Bar, slippage: Double): Double? {
        val buy = isBuy(a)
        val lp = a.limitPrice ?: return if (buy) bar.open * (1 + slippage) else bar.open * (1 - slippage)
        return if (buy) {
            when {
                bar.open <= lp -> bar.open
                bar.low <= lp - TICK -> lp
                else -> null
            }
        } else {
            when {
                bar.open >= lp -> bar.open
                bar.high >= lp + TICK -> lp
                else -> null
            }
        }
    }

    fun marketAt(a: TAction, px: Double, slippage: Double) = if (isBuy(a)) px * (1 + slippage) else px * (1 - slippage)
}

/**
 * 模拟撮合:一个策略一个实例,逐根 K 线调用 [onBar],收盘调用 [endOfDay]。
 * 挂单在下一根起开始撮合;到期未成交则撤单;止盈挂单([TAction.fillNow])在本根内按限价成交。
 */
class SimBroker(private val s: IntradayStrategy, private val slippage: Double) {
    var working: TAction? = null; private set

    /** 先撮合已有订单,再调用 [decide] 取得新指令 */
    fun onBar(bar: Bar, decide: () -> TAction?) {
        working?.let { a ->
            val px = Fills.tryFill(a, bar, slippage)
            if (px != null) {
                working = null
                s.onFill(a, px, bar.minute)
            } else if (a.validUntil >= 0 && Session.elapsed(bar.minute) > a.validUntil) {
                working = null
                s.onCancel(a)
            }
        }
        val act = decide() ?: return
        if (act.fillNow) {
            s.onFill(act, act.limitPrice ?: bar.close, bar.minute)
        } else {
            working?.let { if (it !== act) s.onCancel(it) }
            working = act
        }
    }

    /** 收盘:未成交市价单按收盘价成交,挂单撤销,仍有持仓则按收盘价恢复底仓 */
    fun endOfDay(last: Bar) {
        working?.let { a ->
            working = null
            if (a.limitPrice == null) s.onFill(a, Fills.marketAt(a, last.close, slippage), last.minute) else s.onCancel(a)
        }
        s.forceClose(last)?.let { s.onFill(it, Fills.marketAt(it, last.close, slippage), last.minute) }
    }
}

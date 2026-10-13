package realt.core.engine

import realt.core.model.Bar

/** 单只股票、单个交易日的做T策略,回放器与实时盯盘共用此接口 */
interface IntradayStrategy {
    val trades: List<TTrade>
    val log: List<String>
    /** 喂入一根新的 1 分钟线,返回建议(可空) */
    fun onBar(bar: Bar): TAction?
    /** 建议成交回报 */
    fun onFill(action: TAction, price: Double, minute: Int)
    /** 数据提前结束(停牌、缺数据)时强制收尾,恢复底仓;无持仓返回 null */
    fun forceClose(bar: Bar): TAction?
}

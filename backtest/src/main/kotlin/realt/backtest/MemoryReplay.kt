package realt.backtest

import realt.core.engine.Costs
import realt.core.engine.TTrade
import realt.core.learn.ContextMemory
import realt.core.model.Symbol

data class GatedTrade(val symbol: Symbol, val trade: TTrade, val taken: Boolean)

/**
 * L6 迷宫记忆的时间顺序回放。
 *
 * 所有候选做T(无门控时引擎会做的)按日期排序;处理某一天时,记忆里只包含此前交易日的结果,
 * 由记忆决定是否真的出手。当天结束后把当天全部候选(含未出手的"影子交易")的结果写入记忆——
 * 实盘中这些信号同样可以被跟踪记录,因此不构成未来函数。
 */
class MemoryReplay(private val costs: Costs, private val memory: ContextMemory = ContextMemory()) {

    fun run(records: List<DayRecord>): Pair<List<GatedTrade>, ContextMemory> {
        val out = ArrayList<GatedTrade>()
        val byDate = records.groupBy { it.date }.toSortedMap()
        for ((_, recs) in byDate) {
            val today = recs.flatMap { r -> r.trades.map { r.symbol to it } }
                .sortedBy { it.second.openMinute }
            for ((sym, t) in today) out += GatedTrade(sym, t, memory.allow(t.context))
            for ((_, t) in today) {
                val pnl = t.netPnl(costs)
                memory.record(t.context, pnl > 0, pnl)
            }
        }
        return out to memory
    }
}

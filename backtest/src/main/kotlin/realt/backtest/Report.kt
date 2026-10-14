package realt.backtest

import realt.core.daytype.DayType
import realt.core.engine.Costs
import realt.core.engine.TTrade
import realt.core.learn.ContextMemory
import realt.core.ml.TradeModel
import realt.core.model.Symbol

/** 一组做T的统计 */
data class TStats(
    val n: Int,
    val wins: Int,
    val netPnl: Double,
    val avgRetBps: Double,
    val maxLoss: Double,
    val profitFactor: Double,
    val daysTraded: Int,
) {
    val winRate: Double get() = if (n == 0) 0.0 else wins.toDouble() / n

    /** 达标:胜率 ≥ 目标 且 扣费后期望为正 */
    fun meets(target: Double) = n > 0 && winRate >= target && netPnl > 0

    companion object {
        fun of(trades: List<Pair<Symbol, TTrade>>, costs: Costs): TStats {
            if (trades.isEmpty()) return TStats(0, 0, 0.0, 0.0, 0.0, 0.0, 0)
            val pnl = trades.map { it.second.netPnl(costs) }
            val gain = pnl.filter { it > 0 }.sum()
            val loss = -pnl.filter { it < 0 }.sum()
            return TStats(
                trades.size,
                pnl.count { it > 0 },
                pnl.sum(),
                trades.map { it.second.returnPct(costs) }.average() * 10000,
                pnl.minOrNull()?.coerceAtMost(0.0) ?: 0.0,
                if (loss > 0) gain / loss else Double.POSITIVE_INFINITY,
                trades.map { it.first to it.second.date }.distinct().size,
            )
        }
    }
}

class Report(
    private val records: List<DayRecord>,
    private val gated: List<GatedTrade>,
    private val memory: ContextMemory,
    private val costs: Costs,
    private val target: Double,
    private val model: TradeModel? = null,
) {
    private val totalDays = records.size
    private val raw = records.flatMap { r -> r.trades.map { r.symbol to it } }
    private val base = records.flatMap { r -> r.baselineTrades.map { r.symbol to it } }
    private val taken = gated.filter { it.taken }.map { it.symbol to it.trade }
    private val actualOf = records.associate { (it.symbol to it.date) to it.actual }

    fun markdown(): String = buildString {
        val symbols = records.map { it.symbol }.distinct()
        val dates = records.map { it.date }.distinct().sorted()
        appendLine("# realT 回测报告")
        appendLine()
        appendLine("- 标的数:${symbols.size}  交易日:${dates.firstOrNull() ?: "-"} ~ ${dates.lastOrNull() ?: "-"}  个股·日样本:$totalDays")
        appendLine("- 目标:出手胜率 ≥ ${pct(target)} 且扣费后期望为正")
        appendLine("- 成交假设:信号出现后下一根 1 分钟线开盘价成交,单边滑点 ${Trainer.feeText(costs.slippage)}")
        appendLine("- 费率:佣金 ${Trainer.feeText(costs.commissionRate)}(双向,单笔最低 ${costs.minCommission} 元),印花税 ${Trainer.feeText(costs.stampDutyRate)}(仅卖出),过户费 ${Trainer.feeText(costs.transferRate)}(双向);一次完整做T(一买一卖)合计约 ${pct(costs.roundTripRate - 2 * costs.slippage, 3)} + 滑点 ${pct(2 * costs.slippage, 2)}")
        appendLine()
        appendLine("## 1. 总体对比")
        appendLine()
        appendLine("| 策略 | 做T次数 | 胜率 | 平均每次(bp) | 净盈亏(元) | 最大单笔亏损 | 盈亏比 | 出手日占比 | 达标 |")
        appendLine("|---|---|---|---|---|---|---|---|---|")
        row("原型规则(对照)", TStats.of(base, costs))
        if (model == null) {
            row("realT 无记忆", TStats.of(raw, costs))
            row("realT + L6 记忆", TStats.of(taken, costs))
            appendLine()
            appendLine("> 「realT + L6 记忆」:按时间顺序在线学习,只在历史胜率达标的上下文出手。")
            appendLine("> 未加载参数包。建议先 `realt train` 训练,再 `realt backtest --model` 回测。")
        } else {
            val end = model.trainEnd
            row("realT 参数包 · 全部", TStats.of(raw, costs))
            row("realT 参数包 · 样本内 ≤ $end", TStats.of(raw.filter { it.second.date <= end }, costs), records.count { it.date <= end })
            row("**realT 参数包 · 样本外 > $end**", TStats.of(raw.filter { it.second.date > end }, costs), records.count { it.date > end })
            appendLine()
            appendLine("> 参数包训练数据截止 $end。**只有样本外一行代表真实能力**,样本内结果会偏乐观。")
            appendLine("> 出手阈值 ${model.threshold},止盈 ${model.config.takeProfitAtr} ATR,止损 ${model.config.stopLossAtr} ATR。")
        }
        appendLine()

        if (dates.size >= 20) {
            val mid = dates[dates.size / 2]
            appendLine("## 2. 稳定性(前后半段)")
            appendLine()
            appendLine("| 区间 | 做T次数 | 胜率 | 平均每次(bp) | 净盈亏(元) | 最大单笔亏损 | 盈亏比 | 出手日占比 | 达标 |")
            appendLine("|---|---|---|---|---|---|---|---|---|")
            row("前半段 < $mid", TStats.of(taken.filter { it.second.date < mid }, costs), records.count { it.date < mid })
            row("后半段 ≥ $mid", TStats.of(taken.filter { it.second.date >= mid }, costs), records.count { it.date >= mid })
            appendLine()
        }

        appendLine("## 3. 按真实分时形态(事后标注)拆分 · realT + 记忆")
        appendLine()
        appendLine("| 形态 | 样本日 | 做T次数 | 胜率 | 平均(bp) | 净盈亏 | 原型胜率 | 原型净盈亏 |")
        appendLine("|---|---|---|---|---|---|---|---|")
        for (t in DayType.entries) {
            val days = records.count { it.actual == t }
            val s = TStats.of(taken.filter { actualOf[it.first to it.second.date] == t }, costs)
            val b = TStats.of(base.filter { actualOf[it.first to it.second.date] == t }, costs)
            appendLine("| ${t.label} | $days | ${s.n} | ${pct(s.winRate)} | ${f1(s.avgRetBps)} | ${f0(s.netPnl)} | ${pct(b.winRate)} | ${f0(b.netPnl)} |")
        }
        appendLine()

        appendLine("## 4. 分时形态预判准确率(L3)")
        appendLine()
        appendLine("| 时点 | Top1 准确率 |")
        appendLine("|---|---|")
        if (records.isNotEmpty()) {
            appendLine("| 开盘前(先验) | ${pct(records.count { it.prior == it.actual }.toDouble() / records.size)} |")
            appendLine("| 10:00 | ${pct(records.count { it.pred30 == it.actual }.toDouble() / records.size)} |")
            appendLine("| 10:30 | ${pct(records.count { it.pred60 == it.actual }.toDouble() / records.size)} |")
            appendLine("| 11:30 | ${pct(records.count { it.pred120 == it.actual }.toDouble() / records.size)} |")
        }
        appendLine()
        appendLine("10:30 预判混淆矩阵(行=实际,列=预判):")
        appendLine()
        appendLine("| 实际\\预判 | " + DayType.entries.joinToString(" | ") { it.label } + " |")
        appendLine("|---|" + DayType.entries.joinToString("") { "---|" })
        for (a in DayType.entries) {
            appendLine("| ${a.label} | " + DayType.entries.joinToString(" | ") { p -> records.count { it.actual == a && it.pred60 == p }.toString() } + " |")
        }
        appendLine()

        appendLine("## 5. 按方向 / 平仓原因 / 信号来源 / 大盘 / 主力阶段 · realT + 记忆")
        appendLine()
        appendLine("| 维度 | 取值 | 次数 | 胜率 | 平均(bp) | 净盈亏 |")
        appendLine("|---|---|---|---|---|---|")
        group("方向", taken) { it.second.direction.label }
        group("平仓原因", taken) { it.second.closeReason.label }
        group("信号来源", taken) { it.second.context.family.label }
        group("大盘", taken) { realt.core.regime.MarketLevel.valueOf(it.second.context.regime).label }
        group("主力阶段", taken) { realt.core.stage.Stage.valueOf(it.second.context.stage).label }
        group("入场时预判形态", taken) { it.second.context.dayType.label }
        appendLine()

        appendLine("## 6. 多目标验证(逐只股票)")
        appendLine()
        appendLine("| 股票 | 样本日 | 做T次数 | 胜率 | 平均(bp) | 净盈亏 | 达标 | 原型次数 | 原型胜率 | 原型净盈亏 |")
        appendLine("|---|---|---|---|---|---|---|---|---|---|")
        var ok = 0
        for (sym in symbols) {
            val s = TStats.of(taken.filter { it.first == sym }, costs)
            val b = TStats.of(base.filter { it.first == sym }, costs)
            if (s.meets(target)) ok++
            appendLine("| $sym | ${records.count { it.symbol == sym }} | ${s.n} | ${pct(s.winRate)} | ${f1(s.avgRetBps)} | ${f0(s.netPnl)} | ${if (s.meets(target)) "√" else if (s.n == 0) "-" else "×"} | ${b.n} | ${pct(b.winRate)} | ${f0(b.netPnl)} |")
        }
        appendLine()
        appendLine("达标股票:$ok / ${symbols.size}")
        appendLine()
        appendLine("## 7. L6 记忆中样本最多的路口")
        appendLine()
        appendLine("| 路口(大盘\\|阶段\\|形态\\|方向\\|信号) | 样本 | 胜率 | 净盈亏 | 允许出手 |")
        appendLine("|---|---|---|---|---|")
        memory.snapshot().entries.filter { !it.key.startsWith("*") && !it.key.contains("|*|") }
            .sortedByDescending { it.value.n }.take(25).forEach { (k, s) ->
                appendLine("| ${k.replace("|", "\\|")} | ${s.n} | ${pct(s.winRate)} | ${f0(s.pnl)} | ${if (s.posteriorMean >= target - 0.15 && s.pnl >= 0) "是" else "否"} |")
            }
    }

    private fun StringBuilder.row(name: String, s: TStats, days: Int = totalDays) {
        appendLine("| $name | ${s.n} | ${pct(s.winRate)} | ${f1(s.avgRetBps)} | ${f0(s.netPnl)} | ${f0(s.maxLoss)} | ${"%.2f".format(s.profitFactor)} | ${pct(if (days > 0) s.daysTraded.toDouble() / days else 0.0)} | ${if (s.meets(target)) "√" else "×"} |")
    }

    private fun StringBuilder.group(dim: String, ts: List<Pair<Symbol, TTrade>>, key: (Pair<Symbol, TTrade>) -> String) {
        for ((k, g) in ts.groupBy(key).toSortedMap()) {
            val s = TStats.of(g, costs)
            appendLine("| $dim | $k | ${s.n} | ${pct(s.winRate)} | ${f1(s.avgRetBps)} | ${f0(s.netPnl)} |")
        }
    }

    fun tradesCsv(): String = buildString {
        appendLine("strategy,symbol,date,direction,shares,openTime,openPrice,closeTime,closePrice,closeReason,netPnl,retBp,actualDayType,context,taken,note")
        fun line(strategy: String, sym: Symbol, t: TTrade, taken: Boolean) {
            appendLine(listOf(
                strategy, sym.id, t.date, t.direction.name, t.shares, hhmm(t.openMinute), "%.3f".format(t.openPrice),
                hhmm(t.closeMinute), "%.3f".format(t.closePrice), t.closeReason.name, "%.2f".format(t.netPnl(costs)),
                "%.1f".format(t.returnPct(costs) * 10000), actualOf[sym to t.date]?.name ?: "", t.context.toString(), taken,
                "\"" + t.openNote.replace("\"", "'") + "\"",
            ).joinToString(","))
        }
        for (g in gated) line("realT", g.symbol, g.trade, g.taken)
        for ((s, t) in base) line("baseline", s, t, true)
    }

    private fun hhmm(m: Int) = "%02d:%02d".format(m / 60, m % 60)
    private fun pct(x: Double, digits: Int = 1) = "%.${digits}f%%".format(x * 100)
    private fun f0(x: Double) = "%.0f".format(x)
    private fun f1(x: Double) = "%.1f".format(x)
}

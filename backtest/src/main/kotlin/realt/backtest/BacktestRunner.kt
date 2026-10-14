package realt.backtest

import realt.core.learn.ContextMemory
import realt.core.model.Symbol
import java.io.File
import java.util.Collections
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger

data class RunResult(
    val records: List<DayRecord>,
    val gated: List<GatedTrade>,
    val memory: ContextMemory,
    val report: Report,
)

/** 多标的并行回测 → L6 时间顺序回放 → 输出报告 */
class BacktestRunner(
    private val data: MarketData,
    private val cfg: BacktestConfig,
    private val targetWinRate: Double = 0.8,
    private val threads: Int = Runtime.getRuntime().availableProcessors().coerceAtLeast(1),
    private val progress: Progress? = null,
) {
    fun run(symbols: List<Symbol>, outDir: File?): RunResult {
        val bt = Backtester(data, cfg)
        val records = Collections.synchronizedList(ArrayList<DayRecord>())
        val done = AtomicInteger()
        val pool = Executors.newFixedThreadPool(threads)
        progress?.start("回测", symbols.size)
        for (s in symbols) pool.submit {
            try {
                records += bt.run(s)
            } catch (e: Exception) {
                val msg = "  $s 回测失败: $e"
                if (progress != null) progress.log(msg) else System.err.println(msg)
            }
            progress?.step(done.incrementAndGet(), s.id)
        }
        pool.shutdown()
        pool.awaitTermination(7, TimeUnit.DAYS)
        progress?.finish()

        val sorted = records.sortedWith(compareBy({ it.date }, { it.symbol.id }))
        val (gated, memory) = if (cfg.model != null) {
            // 参数包模式:出手已由模型决定,不再叠加 L6 记忆门控
            sorted.flatMap { r -> r.trades.map { GatedTrade(r.symbol, it, true) } } to ContextMemory()
        } else MemoryReplay(cfg.costs, ContextMemory(targetWinRate = targetWinRate)).run(sorted)
        val report = Report(sorted, gated, memory, cfg.costs, targetWinRate, cfg.model)
        if (outDir != null) {
            outDir.mkdirs()
            File(outDir, "report.md").writeText(report.markdown())
            File(outDir, "trades.csv").writeText("﻿" + report.tradesCsv())
            File(outDir, "memory.csv").writeText("﻿" + memory.toCsv())
        }
        return RunResult(sorted, gated, memory, report)
    }
}

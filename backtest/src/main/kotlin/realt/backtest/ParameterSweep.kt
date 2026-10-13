package realt.backtest

import realt.core.engine.TConfig
import realt.core.engine.TTrade
import realt.core.model.Symbol
import java.io.File
import java.util.Collections
import java.util.Properties
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger

/**
 * 参数层面的"迷宫试错":在训练区间上尝试一组参数组合,挑出达标且收益最好的,
 * 再到时间上更晚、从未参与挑选的测试区间检验。测试区间的结果才是可信的。
 */
class ParameterSweep(
    private val data: MarketData,
    private val cfg: BacktestConfig,
    private val target: Double = 0.8,
    /** 训练区间占全部交易日的比例(按时间先后切分) */
    private val trainRatio: Double = 0.6,
    /** 训练区间至少要有多少次做T,参数才有资格入选 */
    private val minTrades: Int = 20,
    private val threads: Int = Runtime.getRuntime().availableProcessors().coerceIn(1, 8),
    private val log: (String) -> Unit = {},
) {
    data class Row(val config: TConfig, val train: TStats, val test: TStats)

    fun grid(base: TConfig): List<TConfig> {
        val out = ArrayList<TConfig>()
        out += base
        for (entry in listOf(0.5, 0.6, 0.7))
            for (tp in listOf(0.35, 0.5))
                for (sl in listOf(0.5, 0.8))
                    for (noLate in listOf(60, 90))
                        for (tb in listOf(0.45, 0.6))
                            for (rx in listOf(false, true))
                                out += base.copy(
                                    entryThreshold = entry, takeProfitAtr = tp, stopLossAtr = sl,
                                    noOpenBeforeCloseMin = noLate, trendBlock = tb, regimeExit = rx,
                                )
        return out.distinct()
    }

    fun run(symbols: List<Symbol>, outDir: File): List<Row> {
        val variants = grid(cfg.tConfig)
        log("参数组合 ${variants.size} 组,标的 ${symbols.size} 只")
        val bt = Backtester(data, cfg, variants)
        val records = Collections.synchronizedList(ArrayList<DayRecord>())
        val done = AtomicInteger()
        val pool = Executors.newFixedThreadPool(threads)
        for (s in symbols) pool.submit {
            try {
                records += bt.run(s)
            } catch (e: Exception) {
                log("  $s 失败: ${e.message}")
            }
            val k = done.incrementAndGet()
            if (k % 10 == 0 || k == symbols.size) log("  进度 $k / ${symbols.size}")
        }
        pool.shutdown()
        pool.awaitTermination(7, TimeUnit.DAYS)

        val dates = records.map { it.date }.distinct().sorted()
        require(dates.size >= 10) { "交易日太少(${dates.size}),无法切分训练/测试区间" }
        val split = dates[(dates.size * trainRatio).toInt().coerceIn(1, dates.size - 1)]
        log("训练区间 ${dates.first()} ~ <$split,测试区间 $split ~ ${dates.last()}")

        fun trades(k: Int, train: Boolean): List<Pair<Symbol, TTrade>> =
            records.filter { (it.date < split) == train }.flatMap { r -> r.variantTrades[k].map { r.symbol to it } }

        val rows = variants.indices.map { k ->
            Row(variants[k], TStats.of(trades(k, true), cfg.costs), TStats.of(trades(k, false), cfg.costs))
        }
        val ranked = rows.sortedWith(compareByDescending<Row> { rank(it.train) }.thenByDescending { it.train.netPnl })
        val best = ranked.first()

        outDir.mkdirs()
        File(outDir, "sweep.md").writeText(markdown(ranked, split, records.size))
        File(outDir, "best.properties").writer(Charsets.UTF_8).use {
            TConfigIO.toProperties(best.config).store(it, "realT 参数扫描最优(训练区间 < $split)")
        }
        return ranked
    }

    /** 排序档位:3 达标;2 盈利且胜率较高;1 盈利;0 其他。样本不足降一档 */
    private fun rank(s: TStats): Int {
        val r = when {
            s.meets(target) -> 3
            s.netPnl > 0 && s.winRate >= target - 0.1 -> 2
            s.netPnl > 0 -> 1
            else -> 0
        }
        return if (s.n < minTrades) minOf(r, 0) else r
    }

    private fun markdown(rows: List<Row>, split: Int, samples: Int) = buildString {
        appendLine("# realT 参数扫描")
        appendLine()
        appendLine("- 个股·日样本 $samples,按时间切分:训练 < $split ≤ 测试")
        appendLine("- 挑选只看训练区间;**测试区间是从未参与挑选的样本外结果,以它为准**")
        appendLine("- 目标:胜率 ≥ ${"%.0f".format(target * 100)}% 且净盈亏 > 0")
        appendLine()
        appendLine("| # | 开仓阈值 | 止盈ATR | 止损ATR | 尾盘禁开(分) | 逆势封锁 | 形态离场 | 训练次数 | 训练胜率 | 训练净盈亏 | 测试次数 | 测试胜率 | 测试净盈亏 | 测试达标 |")
        appendLine("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for ((i, r) in rows.take(20).withIndex()) {
            val c = r.config
            appendLine("| ${i + 1} | ${c.entryThreshold} | ${c.takeProfitAtr} | ${c.stopLossAtr} | ${c.noOpenBeforeCloseMin} | ${c.trendBlock} | ${if (c.regimeExit) "开" else "关"} | " +
                "${r.train.n} | ${pct(r.train.winRate)} | ${"%.0f".format(r.train.netPnl)} | ${r.test.n} | ${pct(r.test.winRate)} | ${"%.0f".format(r.test.netPnl)} | ${if (r.test.meets(target)) "✔" else "✘"} |")
        }
        appendLine()
        appendLine("最优参数已写入 best.properties,可用 `realt backtest --params best.properties` 复用。")
    }

    private fun pct(x: Double) = "%.1f%%".format(x * 100)
}

/** TConfig 与 properties 文件互转 */
object TConfigIO {
    fun toProperties(c: TConfig): Properties = Properties().apply {
        setProperty("tRatio", c.tRatio.toString())
        setProperty("maxTradesPerDay", c.maxTradesPerDay.toString())
        setProperty("noOpenAfterOpenMin", c.noOpenAfterOpenMin.toString())
        setProperty("noOpenBeforeCloseMin", c.noOpenBeforeCloseMin.toString())
        setProperty("forceCloseBeforeCloseMin", c.forceCloseBeforeCloseMin.toString())
        setProperty("entryThreshold", c.entryThreshold.toString())
        setProperty("exitThreshold", c.exitThreshold.toString())
        setProperty("takeProfitAtr", c.takeProfitAtr.toString())
        setProperty("stopLossAtr", c.stopLossAtr.toString())
        setProperty("momentumBlock", c.momentumBlock.toString())
        setProperty("minEdgePct", c.minEdgePct.toString())
        setProperty("edgeCostMultiple", c.edgeCostMultiple.toString())
        setProperty("trendBlock", c.trendBlock.toString())
        setProperty("flatBlock", c.flatBlock.toString())
        setProperty("maxHoldMin", c.maxHoldMin.toString())
        setProperty("regimeExit", c.regimeExit.toString())
        setProperty("cooldownMin", c.cooldownMin.toString())
        setProperty("allowSellFirst", c.allowSellFirst.toString())
        setProperty("allowBuyFirst", c.allowBuyFirst.toString())
    }

    fun fromProperties(p: Properties, base: TConfig = TConfig()): TConfig {
        fun d(k: String, v: Double) = p.getProperty(k)?.toDouble() ?: v
        fun i(k: String, v: Int) = p.getProperty(k)?.toInt() ?: v
        fun b(k: String, v: Boolean) = p.getProperty(k)?.toBoolean() ?: v
        return TConfig(
            tRatio = d("tRatio", base.tRatio),
            maxTradesPerDay = i("maxTradesPerDay", base.maxTradesPerDay),
            noOpenAfterOpenMin = i("noOpenAfterOpenMin", base.noOpenAfterOpenMin),
            noOpenBeforeCloseMin = i("noOpenBeforeCloseMin", base.noOpenBeforeCloseMin),
            forceCloseBeforeCloseMin = i("forceCloseBeforeCloseMin", base.forceCloseBeforeCloseMin),
            entryThreshold = d("entryThreshold", base.entryThreshold),
            exitThreshold = d("exitThreshold", base.exitThreshold),
            takeProfitAtr = d("takeProfitAtr", base.takeProfitAtr),
            stopLossAtr = d("stopLossAtr", base.stopLossAtr),
            momentumBlock = d("momentumBlock", base.momentumBlock),
            minEdgePct = d("minEdgePct", base.minEdgePct),
            edgeCostMultiple = d("edgeCostMultiple", base.edgeCostMultiple),
            trendBlock = d("trendBlock", base.trendBlock),
            flatBlock = d("flatBlock", base.flatBlock),
            maxHoldMin = i("maxHoldMin", base.maxHoldMin),
            regimeExit = b("regimeExit", base.regimeExit),
            cooldownMin = i("cooldownMin", base.cooldownMin),
            allowSellFirst = b("allowSellFirst", base.allowSellFirst),
            allowBuyFirst = b("allowBuyFirst", base.allowBuyFirst),
        )
    }
}

package realt.backtest

import realt.core.engine.TDirection
import realt.tdx.TdxReader
import java.nio.file.Files
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class EndToEndTest {

    @Test
    fun syntheticVipdocRunsThroughWholePipeline() {
        val dir = Files.createTempDirectory("vipdoc").toFile()
        SyntheticMarket(seed = 3).generate(dir, stocks = 3, days = 220, minuteDays = 20)
        val reader = TdxReader(dir)
        val symbols = reader.listMinuteSymbols()
        assertEquals(3, symbols.size)

        val out = Files.createTempDirectory("out").toFile()
        val data = MarketData(reader, out, useBreadth = true)
        assertTrue(data.breadth.isNotEmpty())
        val res = BacktestRunner(data, BacktestConfig()).run(symbols, out)

        assertEquals(3 * 20, res.records.size)
        for (r in res.records) {
            for (t in r.trades) {
                // 每笔 T 当天开平,且开仓早于平仓,股数为整百
                assertEquals(r.date, t.date)
                assertTrue(t.closeMinute >= t.openMinute)
                assertEquals(0, t.shares % 100)
                assertTrue(t.direction == TDirection.SELL_FIRST || t.direction == TDirection.BUY_FIRST)
            }
            assertTrue(r.trades.size <= 2)
        }
        assertTrue(out.resolve("report.md").readText().contains("总体对比"))
        assertTrue(out.resolve("trades.csv").isFile)
    }

    @Test
    fun observationsAreSharedAcrossVariantsWithoutChangingResults() {
        val dir = Files.createTempDirectory("vipdoc").toFile()
        SyntheticMarket(seed = 5).generate(dir, stocks = 1, days = 200, minuteDays = 10)
        val reader = TdxReader(dir)
        val sym = reader.listMinuteSymbols().single()
        val data = MarketData(reader, Files.createTempDirectory("out").toFile(), useBreadth = false)
        val cfg = BacktestConfig()
        val single = Backtester(data, cfg).run(sym)
        val multi = Backtester(data, cfg, listOf(cfg.tConfig, cfg.tConfig.copy(entryThreshold = 0.9))).run(sym)
        // 主参数在多组并行时的结果必须与单独运行完全一致
        assertEquals(single.map { it.trades }, multi.map { it.variantTrades[0] })
    }
}

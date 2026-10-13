package realt.backtest

import realt.core.ml.TradeModel
import realt.tdx.TdxReader
import java.nio.file.Files
import kotlin.test.Test
import kotlin.test.assertTrue

class TrainerTest {

    @Test
    fun trainProducesLoadablePackAndOutOfSampleBacktestRuns() {
        val dir = Files.createTempDirectory("vipdoc").toFile()
        SyntheticMarket(seed = 11).generate(dir, stocks = 4, days = 240, minuteDays = 30)
        val reader = TdxReader(dir)
        val symbols = reader.listMinuteSymbols()
        val out = Files.createTempDirectory("model").toFile()
        val data = MarketData(reader, out, useBreadth = false)

        val res = Trainer(data, BacktestConfig(), minTrades = 5).run(symbols, out)
        val packFile = out.resolve("realt-model.txt")
        assertTrue(packFile.isFile)
        assertTrue(out.resolve("train-report.md").readText().contains("测试段(样本外)"))

        val pack = TradeModel.parse(packFile.readText())
        assertTrue(pack.threshold in 0.5..0.95)
        assertTrue(pack.trainEnd >= pack.trainStart)

        // 用参数包回测:引擎走模型分支,且每笔都当日开平
        val bt = BacktestRunner(data, BacktestConfig(tConfig = pack.config, costs = pack.costs, model = pack))
            .run(symbols, null)
        for (r in bt.records) for (t in r.trades) assertTrue(t.date == r.date && t.closeMinute >= t.openMinute)
        assertTrue(bt.report.markdown().contains("样本外"))
        assertTrue(res.model.gbdt.trees.isNotEmpty())
    }
}

class WilsonTest {
    @Test
    fun wilsonBounds() {
        val lo = Trainer.wilson(30, 30, -1)
        assertTrue(lo in 0.88..0.89, "lo=$lo")
        val lo100 = Trainer.wilson(99, 100, -1)
        assertTrue(lo100 in 0.94..0.96, "lo100=$lo100")
    }
}

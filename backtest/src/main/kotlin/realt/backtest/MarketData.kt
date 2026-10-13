package realt.backtest

import realt.core.model.Bar
import realt.core.model.Symbol
import realt.core.regime.Breadth
import realt.core.regime.MarketRegime
import realt.core.regime.MarketRegimeDetector
import realt.tdx.BreadthScanner
import realt.tdx.TdxReader
import java.io.File
import java.util.concurrent.ConcurrentHashMap

/**
 * 回测所需的市场数据,来源于通达信 vipdoc。
 * 大盘环境按日期缓存(所有个股共用同一天的 L1 结论)。
 */
class MarketData(
    val reader: TdxReader,
    cacheDir: File,
    useBreadth: Boolean,
    private val indexSymbol: Symbol = TdxReader.SH_INDEX,
    log: (String) -> Unit = {},
) {
    val index: List<Bar> = reader.daily(indexSymbol)
    val breadth: List<Breadth> =
        if (useBreadth) BreadthScanner(reader, File(cacheDir, "breadth.csv")).load(progress = log) else emptyList()

    private val regimeCache = ConcurrentHashMap<Int, MarketRegime>()
    private val detector = MarketRegimeDetector()

    init {
        if (index.isEmpty()) log("警告:未找到指数日线 ${reader.dayFile(indexSymbol)},大盘环境将视为'震荡'")
    }

    /** 交易日 [date] 开盘前可知的大盘环境(只用 date 之前的数据) */
    fun regimeBefore(date: Int): MarketRegime = regimeCache.getOrPut(date) {
        val idx = index.filter { it.date < date }.takeLast(250)
        val br = breadth.filter { it.date < date }.takeLast(20)
        detector.detect(idx, br)
    }
}

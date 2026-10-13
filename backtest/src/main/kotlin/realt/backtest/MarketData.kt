package realt.backtest

import realt.core.model.Bar
import realt.core.model.Symbol
import realt.core.model.byDay
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

    /** 指数 1 分钟线(按日分组)与日线,懒加载;没有下载指数分钟线时为空 */
    private class IndexData(val minutes: Map<Int, List<Bar>>, val daily: List<Bar>)
    private val indexCache = ConcurrentHashMap<String, IndexData>()

    private fun indexData(s: Symbol): IndexData = indexCache.getOrPut(s.id) {
        IndexData(reader.minute1(s).byDay(), reader.daily(s))
    }

    /** 个股对应的参考指数:创业板→创业板指,科创板→科创50,其余→上证指数;缺数据时退回上证指数 */
    fun indexFor(stock: Symbol): Symbol {
        val pref = when {
            stock.market == "sz" && stock.code.startsWith("30") -> TdxReader.CYB_INDEX
            stock.market == "sh" && stock.code.startsWith("68") -> Symbol("sh", "000688")
            else -> TdxReader.SH_INDEX
        }
        return if (indexData(pref).minutes.isNotEmpty()) pref else TdxReader.SH_INDEX
    }

    /** 指数在 [date] 的 1 分钟线与昨收;无数据返回 (空, 0) */
    fun indexDay(stock: Symbol, date: Int): Pair<List<Bar>, Double> {
        val d = indexData(indexFor(stock))
        val bars = d.minutes[date] ?: return emptyList<Bar>() to 0.0
        val prev = d.daily.lastOrNull { it.date < date }?.close ?: return emptyList<Bar>() to 0.0
        return bars to prev
    }

    val hasIndexMinutes: Boolean get() = indexData(TdxReader.SH_INDEX).minutes.isNotEmpty()
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

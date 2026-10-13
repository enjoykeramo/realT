package realt.tdx

import realt.core.regime.Breadth
import java.io.File
import kotlin.math.round

/**
 * 扫描 vipdoc 全部 A 股日线,统计每日涨跌家数、涨停跌停家数(市场宽度)。
 * 结果缓存为 CSV,vipdoc 日线更新后(以文件最大修改时间判断)自动重算。
 *
 * 注:离线日线不含 ST 标记,ST 股按普通涨跌幅计,涨跌停家数会略有偏差,不影响宽度的方向判断。
 */
class BreadthScanner(private val reader: TdxReader, private val cacheFile: File) {

    fun load(days: Int = 400, progress: (String) -> Unit = {}): List<Breadth> {
        val symbols = TdxReader.MARKETS.flatMap { reader.listSymbols(it) }
        val stamp = symbols.maxOfOrNull { reader.dayFile(it).lastModified() } ?: 0L
        if (cacheFile.isFile) {
            val lines = cacheFile.readLines()
            if (lines.firstOrNull() == "#$stamp,$days") {
                return lines.drop(2).filter { it.isNotBlank() }.map { l ->
                    val p = l.split(',').map { it.trim().toInt() }
                    Breadth(p[0], p[1], p[2], p[3], p[4], p[5])
                }
            }
        }
        progress("统计市场宽度:扫描 ${symbols.size} 只 A 股日线…")
        data class Acc(var up: Int = 0, var down: Int = 0, var lu: Int = 0, var ld: Int = 0, var total: Int = 0)
        val acc = java.util.TreeMap<Int, Acc>()
        for ((i, s) in symbols.withIndex()) {
            if (i % 1000 == 0 && i > 0) progress("  $i / ${symbols.size}")
            val bars = reader.daily(s, days + 1)
            for (k in 1 until bars.size) {
                val pc = bars[k - 1].close
                val c = bars[k].close
                val a = acc.getOrPut(bars[k].date) { Acc() }
                a.total++
                if (c > pc) a.up++ else if (c < pc) a.down++
                val lim = s.limitPct
                if (c >= round(pc * (1 + lim) * 100) / 100 - 0.005) a.lu++
                if (c <= round(pc * (1 - lim) * 100) / 100 + 0.005) a.ld++
            }
        }
        val out = acc.map { (d, a) -> Breadth(d, a.up, a.down, a.lu, a.ld, a.total) }
        cacheFile.parentFile?.mkdirs()
        cacheFile.writeText(buildString {
            appendLine("#$stamp,$days")
            appendLine("date,up,down,limitUp,limitDown,total")
            for (b in out) appendLine("${b.date},${b.up},${b.down},${b.limitUp},${b.limitDown},${b.total}")
        })
        return out
    }
}

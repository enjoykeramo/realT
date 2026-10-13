package realt.app

import realt.backtest.BacktestConfig
import realt.backtest.BacktestRunner
import realt.backtest.Backtester
import realt.backtest.MarketData
import realt.backtest.ParameterSweep
import realt.backtest.TConfigIO
import realt.backtest.SyntheticMarket
import realt.core.engine.TConfig
import realt.core.model.Symbol
import realt.tdx.TdxReader
import java.io.File
import java.util.Properties
import kotlin.random.Random
import kotlin.system.exitProcess

private const val HELP = """
realT —— A 股分时做T算法验证工具

用法:
  realt config --vipdoc <通达信vipdoc目录>      保存数据目录,例如 D:\new_tdx\vipdoc
  realt check                                  检查数据目录与数据覆盖范围
  realt backtest [选项]                        多标的回测,输出 report.md / trades.csv / memory.csv
  realt sweep [backtest 选项]                  参数扫描:训练区间挑参数,测试区间样本外检验,输出 sweep.md / best.properties
  realt replay --symbol sh600000 --date 20250930  单只股票单日逐笔回放(看引擎每一步判断)
  realt demo [--out demo-vipdoc]               生成合成数据并跑通全流程(无实盘意义)

backtest 选项:
  --symbols sh600000,sz000001   指定股票(逗号分隔)
  --file list.txt               从文件读股票列表(每行一个,支持 600000 / sh600000)
  --sample 100                  未指定股票时,从有 1 分钟数据的 A 股中随机抽取 N 只(默认 60)
  --start 20250101 --end 20251231   日期范围
  --base 100000                 每只股票底仓市值(元)
  --t-ratio 0.3                 T 仓占底仓比例
  --max-t 2                     每日最多做T次数
  --target 0.8                  目标胜率
  --params best.properties      加载参数文件(sweep 产出),命令行参数优先
  --no-breadth                  不扫描全市场宽度(更快)
  --out realt-out               输出目录
  --vipdoc <目录>               临时指定数据目录(不保存)

配置文件位置:当前目录 realt.properties 优先,其次 用户目录\.realt\realt.properties
"""

fun main(args: Array<String>) {
    if (args.isEmpty() || args[0] in listOf("-h", "--help", "help")) {
        println(HELP.trimIndent()); return
    }
    val opts = parseOpts(args.drop(1))
    try {
        when (args[0]) {
            "config" -> config(opts)
            "check" -> check(opts)
            "backtest" -> backtest(opts)
            "sweep" -> sweep(opts)
            "replay" -> replay(opts)
            "demo" -> demo(opts)
            else -> {
                println("未知命令: ${args[0]}"); println(HELP.trimIndent()); exitProcess(2)
            }
        }
    } catch (e: IllegalArgumentException) {
        System.err.println("错误: ${e.message}")
        exitProcess(1)
    }
}

private fun parseOpts(a: List<String>): Map<String, String> {
    val m = HashMap<String, String>()
    var i = 0
    while (i < a.size) {
        val k = a[i].removePrefix("--")
        if (i + 1 < a.size && !a[i + 1].startsWith("--")) {
            m[k] = a[i + 1]; i += 2
        } else {
            m[k] = "true"; i++
        }
    }
    return m
}

private val homeConfig = File(System.getProperty("user.home"), ".realt/realt.properties")
private val localConfig = File("realt.properties")

private fun loadProps(): Properties {
    val p = Properties()
    for (f in listOf(homeConfig, localConfig)) if (f.isFile) f.reader(Charsets.UTF_8).use { p.load(it) }
    return p
}

private fun config(o: Map<String, String>) {
    val dir = o["vipdoc"] ?: throw IllegalArgumentException("请提供 --vipdoc <目录>")
    val f = File(dir)
    require(File(f, "sh").isDirectory || File(f, "sz").isDirectory) { "$dir 下没有 sh/sz 子目录,不像是通达信 vipdoc 目录" }
    val p = loadProps()
    p["vipdoc"] = f.absolutePath
    homeConfig.parentFile.mkdirs()
    homeConfig.writer(Charsets.UTF_8).use { p.store(it, "realT") }
    println("已保存: vipdoc = ${f.absolutePath}  ->  $homeConfig")
}

private fun reader(o: Map<String, String>): TdxReader {
    val dir = o["vipdoc"] ?: loadProps().getProperty("vipdoc")
        ?: throw IllegalArgumentException("尚未配置数据目录,请先运行: realt config --vipdoc <通达信vipdoc目录>")
    return TdxReader(File(dir))
}

private fun check(o: Map<String, String>) {
    val r = reader(o)
    println("数据目录: ${r.vipdoc.absolutePath}")
    for (m in TdxReader.MARKETS) {
        val lday = File(r.vipdoc, "$m/lday").list()?.size ?: 0
        val min1 = File(r.vipdoc, "$m/minline").list()?.count { it.endsWith(".lc1") } ?: 0
        val min5 = File(r.vipdoc, "$m/fzline").list()?.count { it.endsWith(".lc5") } ?: 0
        println("  $m: 日线 $lday 个文件, 1分钟 $min1 个, 5分钟 $min5 个")
    }
    val idx = r.daily(TdxReader.SH_INDEX)
    println("上证指数日线: ${idx.size} 条 ${idx.firstOrNull()?.date ?: "-"} ~ ${idx.lastOrNull()?.date ?: "-"}")
    val ms = r.listMinuteSymbols()
    println("有 1 分钟线的 A 股: ${ms.size} 只")
    ms.take(3).forEach { s ->
        val bars = r.minute1(s)
        val days = bars.map { it.date }.distinct()
        println("  例 $s: ${bars.size} 根, ${days.size} 个交易日 ${days.firstOrNull() ?: "-"} ~ ${days.lastOrNull() ?: "-"}")
    }
    if (ms.isEmpty()) println("提示:请在通达信「系统 → 盘后数据下载」中勾选「1分钟线数据」下载后再试。")
}

private fun symbolsFrom(o: Map<String, String>, r: TdxReader): List<Symbol> {
    fun norm(s: String): Symbol? {
        val t = s.trim().lowercase()
        if (t.isEmpty() || t.startsWith("#")) return null
        if (t.length >= 8 && t[0].isLetter()) return Symbol.parse(t)
        val code = t.take(6)
        val m = when {
            code.startsWith("6") -> "sh"
            code.startsWith("0") || code.startsWith("3") -> "sz"
            else -> "bj"
        }
        return Symbol(m, code)
    }
    o["symbols"]?.let { return it.split(',').mapNotNull(::norm) }
    o["file"]?.let { return File(it).readLines().mapNotNull(::norm) }
    val all = r.listMinuteSymbols()
    val n = o["sample"]?.toInt() ?: 60
    return if (all.size <= n) all else all.shuffled(Random(7)).take(n).sortedBy { it.id }
}

private fun backtestConfig(o: Map<String, String>): BacktestConfig {
    var t = TConfig()
    o["params"]?.let { f ->
        val p = Properties()
        File(f).reader(Charsets.UTF_8).use { p.load(it) }
        t = TConfigIO.fromProperties(p, t)
    }
    o["t-ratio"]?.let { t = t.copy(tRatio = it.toDouble()) }
    o["max-t"]?.let { t = t.copy(maxTradesPerDay = it.toInt()) }
    return BacktestConfig(
        startDate = o["start"]?.toInt() ?: 0,
        endDate = o["end"]?.toInt() ?: 99999999,
        baseValue = o["base"]?.toDouble() ?: 100_000.0,
        tConfig = t,
    )
}

private fun sweep(o: Map<String, String>) {
    val r = reader(o)
    val symbols = symbolsFrom(o, r)
    require(symbols.isNotEmpty()) { "没有可回测的股票(检查 1 分钟线是否已下载)" }
    val out = File(o["out"] ?: "realt-out")
    val data = MarketData(r, out, useBreadth = o["no-breadth"] == null, log = ::println)
    val target = o["target"]?.toDouble() ?: 0.8
    ParameterSweep(data, backtestConfig(o), target, log = ::println).run(symbols, out)
    println(File(out, "sweep.md").readText().lineSequence().take(16).joinToString("\n"))
    println("完整结果: ${File(out, "sweep.md").absolutePath}")
}

private fun backtest(o: Map<String, String>) {
    val r = reader(o)
    val symbols = symbolsFrom(o, r)
    require(symbols.isNotEmpty()) { "没有可回测的股票(检查 1 分钟线是否已下载)" }
    val out = File(o["out"] ?: "realt-out")
    println("回测 ${symbols.size} 只股票 → ${out.absolutePath}")
    val data = MarketData(r, out, useBreadth = o["no-breadth"] == null, log = ::println)
    val target = o["target"]?.toDouble() ?: 0.8
    val res = BacktestRunner(data, backtestConfig(o), target, log = ::println).run(symbols, out)
    println()
    println(res.report.markdown().lineSequence().takeWhile { !it.startsWith("## 2") && !it.startsWith("## 3") }.joinToString("\n"))
    println("完整报告: ${File(out, "report.md").absolutePath}")
}

private fun replay(o: Map<String, String>) {
    val r = reader(o)
    val sym = Symbol.parse(o["symbol"] ?: throw IllegalArgumentException("请提供 --symbol"))
    val date = o["date"]?.toInt() ?: throw IllegalArgumentException("请提供 --date yyyyMMdd")
    val data = MarketData(r, File(o["out"] ?: "realt-out"), useBreadth = o["no-breadth"] == null, log = ::println)
    val recs = Backtester(data, backtestConfig(o).copy(startDate = date, endDate = date)).run(sym)
    val rec = recs.firstOrNull() ?: throw IllegalArgumentException("$sym 在 $date 没有可用数据(需要该日 1 分钟线及之前 140 日日线)")
    println("$sym $date  大盘:${rec.regime.label}  主力阶段:${rec.stage.label}")
    println("分时预判  开盘前:${rec.prior.label}  10:00:${rec.pred30.label}  10:30:${rec.pred60.label}  11:30:${rec.pred120.label}  实际:${rec.actual.label}")
    println(if (rec.tradable) "" else "波动不足以覆盖成本,当日不做T")
    rec.log.forEach { println("  $it") }
    if (rec.trades.isEmpty()) println("  (realT 当日未出手)")
    println("原型规则:")
    rec.baselineTrades.forEach { t ->
        println("  卖 %.2f @%02d:%02d  买 %.2f @%02d:%02d".format(t.openPrice, t.openMinute / 60, t.openMinute % 60, t.closePrice, t.closeMinute / 60, t.closeMinute % 60))
    }
    if (rec.baselineTrades.isEmpty()) println("  (原型当日未出手)")
}

private fun demo(o: Map<String, String>) {
    val dir = File(o["out"] ?: "demo-vipdoc")
    println("生成合成数据 → ${dir.absolutePath}(仅用于验证流程,结果无实盘意义)")
    SyntheticMarket().generate(dir)
    backtest(o + mapOf("vipdoc" to dir.path, "out" to (o["report"] ?: "realt-demo-out"), "no-breadth" to "true"))
}

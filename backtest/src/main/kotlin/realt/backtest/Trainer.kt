package realt.backtest

import realt.core.daytype.DayTypePredictor
import realt.core.engine.ActionKind
import realt.core.engine.Observation
import realt.core.engine.Observer
import realt.core.engine.TAction
import realt.core.engine.TConfig
import realt.core.engine.TDirection
import realt.core.engine.TEngine
import realt.core.ml.Features
import realt.core.ml.Gbdt
import realt.core.ml.TradeModel
import realt.core.model.Session
import realt.core.model.Symbol
import java.io.File
import java.util.Collections
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger

/**
 * 训练:用历史分时数据学习"什么情况下做T能赢",输出参数包。
 *
 * 1. 候选生成:逐分钟回放,每出现一个转折结构(缠论/布林/RSI)就记一个候选点,
 *    同时记录当时的全部特征(L1~L4 + 分时状态)。
 * 2. 打标签:对每个候选点、每组离场参数(止盈×止损),用与实盘完全相同的离场逻辑模拟这一笔T,
 *    扣除佣金/印花税/过户费/滑点后盈利记为 1,否则为 0。
 * 3. 按时间切分:训练 60% / 验证 20% / 测试 20%。
 *    训练集拟合 GBDT 胜率模型;验证集上按"实盘规则"(同一时间只持一笔、每日上限、冷却)
 *    模拟出手,选择离场参数和出手阈值,要求胜率 ≥ 目标且净盈亏 > 0、交易数足够;
 *    测试集从未参与任何选择,其结果才是可信的样本外表现。
 * 4. 用训练+验证数据按选定参数重训最终模型,连同阈值、离场参数、费率写入参数包。
 */
class Trainer(
    private val data: MarketData,
    private val cfg: BacktestConfig,
    private val target: Double = 0.8,
    private val trainRatio: Double = 0.6,
    private val validRatio: Double = 0.2,
    private val minTrades: Int = 30,
    private val threads: Int = Runtime.getRuntime().availableProcessors().coerceIn(1, 8),
    private val log: (String) -> Unit = {},
) {
    /** 候选点:特征 + 每组离场参数下的结果 */
    class Cand(
        val symbol: Symbol,
        val date: Int,
        val elapsed: Int,
        val dir: TDirection,
        val x: DoubleArray,
        val pnl: DoubleArray,
        val ret: DoubleArray,
        val openE: IntArray,
        val closeE: IntArray,
    )

    /** 离场参数网格;开仓时间窗放宽到收盘前 20 分钟,由模型自己学习时间的影响 */
    fun exitGrid(base: TConfig): List<TConfig> {
        val b = base.copy(noOpenBeforeCloseMin = 20, regimeExit = false)
        return listOf(0.3, 0.45, 0.6).flatMap { tp ->
            listOf(0.5, 0.8, 1.2).map { sl -> b.copy(takeProfitAtr = tp, stopLossAtr = sl) }
        }
    }

    data class Eval(val n: Int, val wins: Int, val pnl: Double, val days: Int) {
        val winRate get() = if (n == 0) 0.0 else wins.toDouble() / n
        fun meets(target: Double, minN: Int) = n >= minN && winRate >= target && pnl > 0
    }

    class Result(val model: TradeModel, val report: String)

    fun run(symbols: List<Symbol>, outDir: File): Result {
        val grid = exitGrid(cfg.tConfig)
        log("第 1 步:生成候选点并打标签(${grid.size} 组离场参数)…")
        val cands = Collections.synchronizedList(ArrayList<Cand>())
        val dayCount = Collections.synchronizedMap(HashMap<Int, Int>())
        val done = AtomicInteger()
        val pool = Executors.newFixedThreadPool(threads)
        val bt = Backtester(data, cfg)
        for (s in symbols) pool.submit {
            try {
                val local = ArrayList<Cand>()
                for (su in bt.setups(s)) {
                    local += candidatesOf(su, grid)
                    dayCount.merge(su.date, 1, Int::plus)
                }
                cands += local
            } catch (e: Throwable) {
                log("  $s 失败: ${e.message}")
            }
            val k = done.incrementAndGet()
            if (k % 10 == 0 || k == symbols.size) log("  进度 $k / ${symbols.size},候选点 ${cands.size}")
        }
        pool.shutdown()
        pool.awaitTermination(7, TimeUnit.DAYS)

        val dates = dayCount.keys.sorted()
        require(dates.size >= 20) { "交易日太少(${dates.size}),至少需要 20 个交易日的 1 分钟数据" }
        require(cands.size >= 500) { "候选点太少(${cands.size}),请增加股票数量或数据天数" }
        val d1 = dates[(dates.size * trainRatio).toInt()]
        val d2 = dates[(dates.size * (trainRatio + validRatio)).toInt().coerceAtMost(dates.size - 1)]
        val train = cands.filter { it.date < d1 }
        val valid = cands.filter { it.date in d1 until d2 }
        val test = cands.filter { it.date >= d2 }
        log("第 2 步:按时间切分 训练 ${dates.first()}~<$d1(${train.size}) 验证 $d1~<$d2(${valid.size}) 测试 $d2~${dates.last()}(${test.size})")

        data class Choice(val k: Int, val thr: Double, val v: Eval, val trees: Int)
        val models = ArrayList<Gbdt>()
        var best: Choice? = null
        val curveRows = StringBuilder()
        for ((k, ec) in grid.withIndex()) {
            val m = Gbdt.train(
                train.map { it.x }, IntArray(train.size) { if (train[it].pnl[k] > 0) 1 else 0 },
                valid.map { it.x }, IntArray(valid.size) { if (valid[it].pnl[k] > 0) 1 else 0 },
            )
            models += m
            val pv = valid.map { m.predict(it.x) }
            val base = simulate(valid, pv, k, 0.0, ec)
            log("  离场组 ${k + 1}/${grid.size} 止盈${ec.takeProfitAtr}ATR 止损${ec.stopLossAtr}ATR:${m.trees.size} 棵树,不过滤时验证胜率 ${pct(base.winRate)}")
            for (thrI in 50..95) {
                val thr = thrI / 100.0
                val e = simulate(valid, pv, k, thr, ec)
                val c = Choice(k, thr, e, m.trees.size)
                if (better(c.v, best?.v)) best = c
            }
        }
        val ch = best!!
        val ec = grid[ch.k]
        val m = models[ch.k]
        val pTest = test.map { m.predict(it.x) }
        val testEval = simulate(test, pTest, ch.k, ch.thr, ec)
        val testAll = simulate(test, pTest, ch.k, 0.0, ec)
        log("第 3 步:选定 止盈${ec.takeProfitAtr}ATR 止损${ec.stopLossAtr}ATR 阈值${ch.thr}")
        log("  验证:${ch.v.n} 笔 胜率 ${pct(ch.v.winRate)} 净盈亏 ${"%.0f".format(ch.v.pnl)}")
        log("  测试(样本外):${testEval.n} 笔 胜率 ${pct(testEval.winRate)} 净盈亏 ${"%.0f".format(testEval.pnl)}")
        for (thrI in 50..95 step 5) {
            val thr = thrI / 100.0
            val v = simulate(valid, valid.map { m.predict(it.x) }, ch.k, thr, ec)
            val t = simulate(test, pTest, ch.k, thr, ec)
            curveRows.appendLine("| $thr | ${v.n} | ${pct(v.winRate)} | ${"%.0f".format(v.pnl)} | ${t.n} | ${pct(t.winRate)} | ${"%.0f".format(t.pnl)} |")
        }

        log("第 4 步:用训练+验证数据重训最终模型…")
        val tv = train + valid
        val finalModel = Gbdt.train(
            tv.map { it.x }, IntArray(tv.size) { if (tv[it].pnl[ch.k] > 0) 1 else 0 },
            p = Gbdt.Params(nTrees = maxOf(ch.trees, 20)),
        )
        val deployCfg = ec.copy(noOpenBeforeCloseMin = 20)
        val meta = linkedMapOf(
            "target" to target.toString(),
            "validWinRate" to "%.4f".format(ch.v.winRate), "validTrades" to ch.v.n.toString(),
            "testStart" to d2.toString(), "testEnd" to dates.last().toString(),
            "testWinRate" to "%.4f".format(testEval.winRate), "testTrades" to testEval.n.toString(),
            "testPnl" to "%.2f".format(testEval.pnl),
            "symbols" to symbols.size.toString(),
        )
        val model = TradeModel(deployCfg, cfg.costs, ch.thr, finalModel, dates.first(), dates[dates.indexOf(d2) - 1], meta)

        val report = buildString {
            appendLine("# realT 训练报告")
            appendLine()
            appendLine("- 标的 ${symbols.size} 只,交易日 ${dates.first()} ~ ${dates.last()}(${dates.size} 天),候选点 ${cands.size}")
            appendLine("- 费率:佣金 ${feeText(cfg.costs.commissionRate)}(最低 ${cfg.costs.minCommission} 元)、印花税 ${feeText(cfg.costs.stampDutyRate)}(仅卖出)、过户费 ${feeText(cfg.costs.transferRate)}、滑点单边 ${feeText(cfg.costs.slippage)}")
            appendLine("- 切分:训练 < $d1 ≤ 验证 < $d2 ≤ 测试(测试集从未参与任何选择)")
            appendLine("- 目标:胜率 ≥ ${pct(target)} 且净盈亏 > 0,验证集至少 $minTrades 笔")
            appendLine()
            appendLine("## 结论")
            appendLine()
            appendLine("| 数据段 | 做T次数 | 胜率 | 净盈亏(元) | 达标 |")
            appendLine("|---|---|---|---|---|")
            appendLine("| 验证(用于挑参数) | ${ch.v.n} | ${pct(ch.v.winRate)} | ${"%.0f".format(ch.v.pnl)} | ${if (ch.v.meets(target, minTrades)) "✔" else "✘"} |")
            appendLine("| **测试(样本外)** | ${testEval.n} | ${pct(testEval.winRate)} | ${"%.0f".format(testEval.pnl)} | ${if (testEval.meets(target, 1)) "✔" else "✘"} |")
            appendLine("| 测试·不用模型过滤 | ${testAll.n} | ${pct(testAll.winRate)} | ${"%.0f".format(testAll.pnl)} | - |")
            appendLine()
            appendLine("选定参数:止盈 ${ec.takeProfitAtr} ATR,止损 ${ec.stopLossAtr} ATR,出手阈值(模型胜率)${ch.thr},最终模型 ${finalModel.trees.size} 棵树。")
            appendLine()
            appendLine("## 阈值与胜率/出手次数的关系(选定离场参数)")
            appendLine()
            appendLine("阈值越高越挑剔:胜率上升、出手减少。测试列是样本外表现。")
            appendLine()
            appendLine("| 阈值 | 验证次数 | 验证胜率 | 验证净盈亏 | 测试次数 | 测试胜率 | 测试净盈亏 |")
            appendLine("|---|---|---|---|---|---|---|")
            append(curveRows)
            appendLine()
            appendLine("## 最重要的特征")
            appendLine()
            appendLine("| 特征 | 相对重要性 |")
            appendLine("|---|---|")
            val imp = m.importance
            val tot = imp.sum().takeIf { it > 0 } ?: 1.0
            imp.indices.sortedByDescending { imp[it] }.take(15).forEach {
                appendLine("| ${Features.names[it]} | ${pct(imp[it] / tot)} |")
            }
            appendLine()
            appendLine("参数包:`realt-model.txt`。回测:`realt backtest --model realt-model.txt --start $d2`(只看样本外区间)。")
        }
        outDir.mkdirs()
        File(outDir, "realt-model.txt").writeText(model.serialize())
        File(outDir, "train-report.md").writeText(report)
        return Result(model, report)
    }

    /** 选择规则:达标者取净盈亏最大;都不达标时取"盈利且胜率最高";再不行取净盈亏最大 */
    private fun better(a: Eval, b: Eval?): Boolean {
        if (b == null) return true
        fun tier(e: Eval) = when {
            e.meets(target, minTrades) -> 3
            e.n >= minTrades && e.pnl > 0 -> 2
            e.n >= minTrades -> 1
            else -> 0
        }
        val ta = tier(a); val tb = tier(b)
        if (ta != tb) return ta > tb
        return when (ta) {
            3 -> a.pnl > b.pnl
            2 -> a.winRate > b.winRate || (a.winRate == b.winRate && a.pnl > b.pnl)
            else -> a.pnl > b.pnl
        }
    }

    /** 按实盘规则模拟出手:同一时间只持一笔、每日上限、冷却、时间窗 */
    private fun simulate(cs: List<Cand>, p: List<Double>, k: Int, thr: Double, ec: TConfig): Eval {
        var n = 0; var wins = 0; var pnl = 0.0
        val days = HashSet<Pair<Symbol, Int>>()
        val idx = cs.indices.groupBy { cs[it].symbol to cs[it].date }
        for ((key, ids) in idx) {
            var busyUntil = -1
            var lastClose = -1000
            var count = 0
            for (i in ids.sortedWith(compareBy({ cs[it].elapsed }, { -p[it] }))) {
                if (p[i] < thr) continue
                val c = cs[i]
                if (count >= ec.maxTradesPerDay) break
                // 与实盘一致:持仓期间不开新仓,平仓后冷却
                if (c.elapsed <= busyUntil || c.elapsed - lastClose < ec.cooldownMin) continue
                n++; count++
                if (c.pnl[k] > 0) wins++
                pnl += c.pnl[k]
                busyUntil = c.closeE[k]; lastClose = c.closeE[k]
                days += key
            }
        }
        return Eval(n, wins, pnl, days.size)
    }

    /** 一个交易日内的全部候选点及其在各组离场参数下的结果 */
    private fun candidatesOf(su: DaySetup, grid: List<TConfig>): List<Cand> {
        val observer = Observer(su.ctx, su.history, DayTypePredictor())
        val bars = su.dayBars
        val obs = ArrayList<Observation>(bars.size)
        for (b in bars) obs += observer.observe(b)
        if ((su.shares * grid[0].tRatio / 100).toInt() <= 0) return emptyList()

        val out = ArrayList<Cand>()
        for (i in bars.indices) {
            val el = Session.elapsed(bars[i].minute)
            if (el < grid[0].noOpenAfterOpenMin || Session.MINUTES_PER_DAY - el <= grid[0].noOpenBeforeCloseMin) continue
            for (dir in Features.candidates(obs[i])) {
                val pnl = DoubleArray(grid.size); val ret = DoubleArray(grid.size)
                val oe = IntArray(grid.size); val ce = IntArray(grid.size)
                for ((k, g) in grid.withIndex()) {
                    val e = TEngine(g, cfg.costs, su.ctx, su.shares, null)
                    var pending: TAction? = e.forceOpen(dir, bars[i], obs[i])
                    var j = i + 1
                    while (j < bars.size && e.trades.isEmpty()) {
                        pending?.let { fill(e, it, bars[j].open, bars[j].minute) }
                        pending = if (e.trades.isEmpty()) e.decide(bars[j], obs[j]) else null
                        j++
                    }
                    val last = bars.last()
                    if (e.trades.isEmpty()) {
                        pending?.let { fill(e, it, last.close, last.minute) }
                        e.forceClose(last)?.let { fill(e, it, last.close, last.minute) }
                    }
                    val t = e.trades.firstOrNull() ?: continue
                    pnl[k] = t.netPnl(cfg.costs)
                    ret[k] = t.returnPct(cfg.costs)
                    oe[k] = Session.elapsed(t.openMinute)
                    ce[k] = Session.elapsed(t.closeMinute)
                }
                out += Cand(su.symbol, su.date, el, dir, Features.extract(su.ctx, obs[i], dir), pnl, ret, oe, ce)
            }
        }
        return out
    }

    private fun fill(e: TEngine, a: TAction, px: Double, minute: Int) {
        val slip = cfg.costs.slippage
        val buy = a.kind == ActionKind.OPEN_BUY || a.kind == ActionKind.CLOSE_BUY
        e.onFill(a, if (buy) px * (1 + slip) else px * (1 - slip), minute)
    }

    private fun pct(x: Double) = "%.1f%%".format(x * 100)

    companion object {
        /** 费率显示:0.00025 → 万2.5 */
        fun feeText(rate: Double): String = "万%s".format(trim(rate * 10000))
        private fun trim(x: Double) = if (x == Math.floor(x)) "%.0f".format(x) else "%.2f".format(x).trimEnd('0')
    }
}

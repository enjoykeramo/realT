package realt.backtest

import realt.core.engine.ActionKind
import realt.core.engine.CloseReason
import realt.core.engine.Observation
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
import kotlin.math.sqrt

/**
 * 训练:用历史分时数据学习"什么情况下做T能赢",输出参数包。
 *
 * 1. 候选生成:逐分钟回放,每出现一个转折结构(缠论/布林/RSI)就记一个候选点及其全部特征。
 * 2. 打标签:对每个候选点、每组离场参数,用与实盘完全相同的离场逻辑模拟这一笔(当日必须完成),
 *    扣全部费用后盈利为 1。
 * 3. 滚动前推(walk-forward):前 40% 交易日作为起始训练期,其余分成 [folds] 段;
 *    每段都只用它之前的数据训练模型并预测它 → 得到大量"样本外预测"。
 *    最后一段是最终测试集;其余各段的样本外预测用来挑选离场参数与出手阈值(交易笔数远多于单一验证集,结论更稳)。
 * 4. 用测试段之前的全部数据重训最终模型,与阈值、离场参数、费率一起写入参数包。
 *    测试段不进入最终模型,因此 `backtest --model` 在测试段上仍是样本外。
 */
class Trainer(
    private val data: MarketData,
    private val cfg: BacktestConfig,
    private val target: Double = 0.95,
    private val firstTrainRatio: Double = 0.4,
    private val folds: Int = 4,
    private val minTrades: Int = 30,
    private val threads: Int = Runtime.getRuntime().availableProcessors().coerceIn(1, 8),
    private val log: (String) -> Unit = {},
) {
    /** 候选点:特征 + 每组离场参数下的结果(均为当日完成的T) */
    class Cand(
        val symbol: Symbol,
        val date: Int,
        val elapsed: Int,
        val dir: TDirection,
        val x: DoubleArray,
        val pnl: DoubleArray,
        val reason: Array<CloseReason?>,
        val closeE: IntArray,
    )

    /** 离场参数网格:止盈偏小、止损偏宽,是高胜率的必要条件;开仓时间窗放宽到收盘前 20 分钟,由模型学习时间影响 */
    fun exitGrid(base: TConfig): List<TConfig> {
        val b = base.copy(noOpenBeforeCloseMin = 20, regimeExit = false)
        return listOf(0.15, 0.25, 0.4).flatMap { tp ->
            listOf(0.8, 1.5).map { sl -> b.copy(takeProfitAtr = tp, stopLossAtr = sl) }
        }
    }

    data class Eval(val n: Int, val wins: Int, val pnl: Double, val sumRet: Double) {
        val winRate get() = if (n == 0) 0.0 else wins.toDouble() / n
        val avgBp get() = if (n == 0) 0.0 else sumRet / n * 10000
        /** Wilson 95% 置信区间:样本少时胜率的保守/乐观估计 */
        val winLow: Double get() = wilson(wins, n, -1)
        val winHigh: Double get() = wilson(wins, n, 1)
        fun meets(target: Double, minN: Int) = n >= minN && winRate >= target && pnl > 0
    }

    class Result(val model: TradeModel, val report: String)

    fun run(symbols: List<Symbol>, outDir: File): Result {
        val grid = exitGrid(cfg.tConfig)
        if (!data.hasIndexMinutes) log("提示:未找到指数 1 分钟线(sh000001.lc1),指数分时特征将为 0。建议在通达信下载 1 分钟线时包含指数。")
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
        require(dates.size >= 25) { "交易日太少(${dates.size}),至少需要 25 个交易日的 1 分钟数据" }
        require(cands.size >= 1000) { "候选点太少(${cands.size}),请增加股票数量或数据天数" }

        // ---- 滚动前推切分 ----
        val startIdx = (dates.size * firstTrainRatio).toInt()
        val foldStarts = (0 until folds).map { dates[startIdx + (dates.size - startIdx) * it / folds] }
        val testStart = foldStarts.last()
        fun foldOf(d: Int): Int = foldStarts.indexOfLast { d >= it } // -1 = 起始训练期
        val byFold = cands.groupBy { foldOf(it.date) }
        log("第 2 步:滚动前推 起始训练 ${dates.first()}~<${foldStarts[0]},样本外 $folds 段:" +
            foldStarts.joinToString(" ") { "≥$it" } + ";最后一段为测试集")

        // 每个离场组:各段的样本外预测
        val oofP = Array(grid.size) { HashMap<Cand, Double>() }
        val treesUsed = ArrayList<Int>()
        val importance = DoubleArray(Features.size)
        for (f in 0 until folds) {
            val trainSet = cands.filter { it.date < foldStarts[f] }
            val trainDates = dates.filter { it < foldStarts[f] }
            val esStart = trainDates[(trainDates.size * 0.85).toInt()]
            val fit = trainSet.filter { it.date < esStart }
            val es = trainSet.filter { it.date >= esStart }
            val pred = byFold[f] ?: emptyList()
            for (k in grid.indices) {
                val m = Gbdt.train(
                    fit.map { it.x }, IntArray(fit.size) { if (fit[it].pnl[k] > 0) 1 else 0 },
                    es.map { it.x }, IntArray(es.size) { if (es[it].pnl[k] > 0) 1 else 0 },
                    gbdtParams(),
                )
                treesUsed += m.trees.size
                for (i in m.importance.indices) importance[i] += m.importance[i]
                for (c in pred) oofP[k][c] = m.predict(c.x)
            }
            log("  第 ${f + 1}/$folds 段:训练 ${fit.size} + 早停 ${es.size},预测 ${pred.size}")
        }

        // ---- 在选择段(除测试段外的样本外数据)上挑离场参数与阈值 ----
        val sel = cands.filter { foldOf(it.date) in 0 until folds - 1 }
        val test = byFold[folds - 1] ?: emptyList()
        val thresholds = (50..99).map { it / 100.0 }
        data class Choice(val k: Int, val thr: Double, val e: Eval)
        var best: Choice? = null
        val frontier = StringBuilder()
        for ((k, ec) in grid.withIndex()) {
            val p = sel.map { oofP[k][it] ?: 0.0 }
            var kBest: Choice? = null
            for (thr in thresholds) {
                val c = Choice(k, thr, simulate(sel, p, k, thr, ec))
                if (better(c.e, kBest?.e)) kBest = c
                if (better(c.e, best?.e)) best = c
            }
            val kb = kBest!!
            frontier.appendLine("| ${ec.takeProfitAtr} | ${ec.stopLossAtr} | ${kb.thr} | ${kb.e.n} | ${pct(kb.e.winRate)} | ${pct(kb.e.winLow)}~${pct(kb.e.winHigh)} | ${f1(kb.e.avgBp)} | ${f0(kb.e.pnl)} |")
        }
        val ch = best!!
        val ec = grid[ch.k]
        val pSel = sel.map { oofP[ch.k][it] ?: 0.0 }
        val pTest = test.map { oofP[ch.k][it] ?: 0.0 }
        val testEval = simulate(test, pTest, ch.k, ch.thr, ec)
        val testAll = simulate(test, pTest, ch.k, 0.0, ec)
        log("第 3 步:选定 止盈${ec.takeProfitAtr}ATR 止损${ec.stopLossAtr}ATR 阈值${ch.thr}")
        log("  选择段:${ch.e.n} 笔 胜率 ${pct(ch.e.winRate)}(95%置信 ${pct(ch.e.winLow)}~${pct(ch.e.winHigh)})净盈亏 ${f0(ch.e.pnl)}")
        log("  测试段:${testEval.n} 笔 胜率 ${pct(testEval.winRate)}(95%置信 ${pct(testEval.winLow)}~${pct(testEval.winHigh)})净盈亏 ${f0(testEval.pnl)}")

        val curve = StringBuilder()
        for (thr in thresholds.filter { (it * 100).toInt() % 5 == 0 || it >= 0.8 }) {
            val v = simulate(sel, pSel, ch.k, thr, ec)
            val t = simulate(test, pTest, ch.k, thr, ec)
            if (v.n == 0 && t.n == 0) continue
            curve.appendLine("| $thr | ${v.n} | ${pct(v.winRate)} | ${f1(v.avgBp)} | ${f0(v.pnl)} | ${t.n} | ${pct(t.winRate)} | ${f1(t.avgBp)} | ${f0(t.pnl)} |")
        }

        // 亏损单是怎么输的
        val lossReasons = StringBuilder()
        run {
            val taken = takenTrades(sel + test, pSel + pTest, ch.k, ch.thr, ec)
            val losers = taken.filter { it.pnl[ch.k] <= 0 }
            lossReasons.appendLine("出手 ${taken.size} 笔中亏损 ${losers.size} 笔:")
            lossReasons.appendLine()
            lossReasons.appendLine("| 方向 | 平仓原因 | 亏损笔数 | 平均亏损(元) |")
            lossReasons.appendLine("|---|---|---|---|")
            for ((key, g) in losers.groupBy { it.dir.label to (it.reason[ch.k]?.label ?: "-") }.toSortedMap(compareBy({ it.first }, { it.second }))) {
                lossReasons.appendLine("| ${key.first} | ${key.second} | ${g.size} | ${f0(g.map { it.pnl[ch.k] }.average())} |")
            }
        }

        log("第 4 步:用测试段之前的全部数据重训最终模型…")
        val finalSet = cands.filter { it.date < testStart }
        val nTrees = treesUsed.sorted()[treesUsed.size / 2].coerceAtLeast(20)
        val finalModel = Gbdt.train(
            finalSet.map { it.x }, IntArray(finalSet.size) { if (finalSet[it].pnl[ch.k] > 0) 1 else 0 },
            p = gbdtParams().copy(nTrees = nTrees),
        )
        val trainEnd = dates[dates.indexOf(testStart) - 1]
        val meta = linkedMapOf(
            "target" to target.toString(),
            "selTrades" to ch.e.n.toString(), "selWinRate" to "%.4f".format(ch.e.winRate),
            "testStart" to testStart.toString(), "testEnd" to dates.last().toString(),
            "testTrades" to testEval.n.toString(), "testWinRate" to "%.4f".format(testEval.winRate),
            "testPnl" to "%.2f".format(testEval.pnl), "symbols" to symbols.size.toString(),
        )
        val model = TradeModel(ec, cfg.costs, ch.thr, finalModel, dates.first(), trainEnd, meta)

        val report = buildString {
            appendLine("# realT 训练报告")
            appendLine()
            appendLine("- 标的 ${symbols.size} 只,交易日 ${dates.first()} ~ ${dates.last()}(${dates.size} 天),候选点 ${cands.size}")
            appendLine("- 费率:佣金 ${feeText(cfg.costs.commissionRate)}(最低 ${cfg.costs.minCommission} 元)、印花税 ${feeText(cfg.costs.stampDutyRate)}(仅卖出)、过户费 ${feeText(cfg.costs.transferRate)}、滑点单边 ${feeText(cfg.costs.slippage)}")
            appendLine("- 指数分时数据:${if (data.hasIndexMinutes) "有" else "**无**(指数特征为 0)"}")
            appendLine("- 滚动前推:起始训练 < ${foldStarts[0]};样本外 $folds 段,前 ${folds - 1} 段用于挑参数,**最后一段 ≥ $testStart 为测试集**")
            appendLine("- 目标:胜率 ≥ ${pct(target)} 且净盈亏 > 0,至少 $minTrades 笔")
            appendLine()
            appendLine("## 结论")
            appendLine()
            appendLine("| 数据段 | 做T次数 | 胜率 | 胜率95%置信区间 | 平均每笔(bp) | 净盈亏(元) | 达标 |")
            appendLine("|---|---|---|---|---|---|---|")
            appendLine("| 选择段(样本外预测,用于挑参数) | ${ch.e.n} | ${pct(ch.e.winRate)} | ${pct(ch.e.winLow)}~${pct(ch.e.winHigh)} | ${f1(ch.e.avgBp)} | ${f0(ch.e.pnl)} | ${mark(ch.e.meets(target, minTrades))} |")
            appendLine("| **测试段(样本外)** | ${testEval.n} | ${pct(testEval.winRate)} | ${pct(testEval.winLow)}~${pct(testEval.winHigh)} | ${f1(testEval.avgBp)} | ${f0(testEval.pnl)} | ${mark(testEval.meets(target, 1))} |")
            appendLine("| 测试段·不用模型过滤 | ${testAll.n} | ${pct(testAll.winRate)} | - | ${f1(testAll.avgBp)} | ${f0(testAll.pnl)} | - |")
            appendLine()
            appendLine("选定:止盈 ${ec.takeProfitAtr} ATR,止损 ${ec.stopLossAtr} ATR,出手阈值 ${ch.thr},最终模型 ${finalModel.trees.size} 棵树。")
            appendLine()
            appendLine("> 置信区间:30 笔全胜,真实胜率的 95% 置信下限只有约 89%;要让下限 ≥ 95%,大约需要 100 笔中最多错 1 笔。笔数少时不要只看胜率。")
            appendLine()
            appendLine("## 各组离场参数能达到的最好结果(选择段)")
            appendLine()
            appendLine("| 止盈ATR | 止损ATR | 阈值 | 次数 | 胜率 | 95%置信区间 | 平均(bp) | 净盈亏 |")
            appendLine("|---|---|---|---|---|---|---|---|")
            append(frontier)
            appendLine()
            appendLine("## 阈值—胜率—出手次数(选定离场参数)")
            appendLine()
            appendLine("| 阈值 | 选择段次数 | 选择段胜率 | 平均(bp) | 净盈亏 | 测试次数 | 测试胜率 | 平均(bp) | 净盈亏 |")
            appendLine("|---|---|---|---|---|---|---|---|---|")
            append(curve)
            appendLine()
            appendLine("## 亏损单分析")
            appendLine()
            append(lossReasons)
            appendLine()
            appendLine("## 最重要的特征(各段模型累计)")
            appendLine()
            appendLine("| 特征 | 相对重要性 |")
            appendLine("|---|---|")
            val tot = importance.sum().takeIf { it > 0 } ?: 1.0
            importance.indices.sortedByDescending { importance[it] }.take(20).forEach {
                appendLine("| ${Features.names[it]} | ${pct(importance[it] / tot)} |")
            }
            appendLine()
            appendLine("参数包:`realt-model.txt`(训练数据截止 $trainEnd)。样本外回测:`realt backtest --model realt-model.txt --start $testStart`")
        }
        outDir.mkdirs()
        File(outDir, "realt-model.txt").writeText(model.serialize())
        File(outDir, "train-report.md").writeText(report)
        return Result(model, report)
    }

    private fun gbdtParams() = Gbdt.Params(nTrees = 400, minLeaf = 30, colSample = 0.7, earlyStop = 40)

    /** 选择规则:达标者取净盈亏最大;都不达标时取"笔数够、盈利、胜率最高";再不行取净盈亏最大 */
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

    /** 按实盘规则挑出会真正出手的候选:同一时间只持一笔、每日上限、冷却 */
    private fun takenTrades(cs: List<Cand>, p: List<Double>, k: Int, thr: Double, ec: TConfig): List<Cand> {
        val out = ArrayList<Cand>()
        val idx = cs.indices.groupBy { cs[it].symbol to cs[it].date }
        for ((_, ids) in idx) {
            var busyUntil = -1
            var lastClose = -1000
            var count = 0
            for (i in ids.sortedWith(compareBy({ cs[it].elapsed }, { -p[it] }))) {
                if (p[i] < thr) continue
                val c = cs[i]
                if (count >= ec.maxTradesPerDay) break
                if (c.elapsed <= busyUntil || c.elapsed - lastClose < ec.cooldownMin) continue
                out += c
                count++
                busyUntil = c.closeE[k]; lastClose = c.closeE[k]
            }
        }
        return out
    }

    private fun simulate(cs: List<Cand>, p: List<Double>, k: Int, thr: Double, ec: TConfig): Eval {
        var n = 0; var wins = 0; var pnl = 0.0; var sumRet = 0.0
        val notional = cfg.baseValue * ec.tRatio
        for (c in takenTrades(cs, p, k, thr, ec)) {
            val v = c.pnl[k]
            n++
            if (v > 0) wins++
            pnl += v
            sumRet += v / notional
        }
        return Eval(n, wins, pnl, sumRet)
    }

    /** 一个交易日内的全部候选点及其在各组离场参数下的结果 */
    private fun candidatesOf(su: DaySetup, grid: List<TConfig>): List<Cand> {
        val observer = su.observer()
        val bars = su.dayBars
        val obs = ArrayList<Observation>(bars.size)
        for (b in bars) obs += observer.observe(b)
        if ((su.shares * grid[0].tRatio / 100).toInt() <= 0) return emptyList()

        val out = ArrayList<Cand>()
        for (i in bars.indices) {
            val el = Session.elapsed(bars[i].minute)
            if (el < grid[0].noOpenAfterOpenMin || Session.MINUTES_PER_DAY - el <= grid[0].noOpenBeforeCloseMin) continue
            for (dir in Features.candidates(obs[i])) {
                val pnl = DoubleArray(grid.size); val reason = arrayOfNulls<CloseReason>(grid.size)
                val ce = IntArray(grid.size)
                for ((k, g) in grid.withIndex()) {
                    val e = TEngine(g, cfg.costs, su.ctx, su.shares, null)
                    var pending: TAction? = e.forceOpen(dir, bars[i], obs[i])
                    var j = i + 1
                    while (j < bars.size && e.trades.isEmpty()) {
                        pending?.let { fill(e, it, bars[j].open, bars[j].minute) }
                        if (e.trades.isNotEmpty()) break
                        pending = e.decide(bars[j], obs[j])
                        j++
                    }
                    val last = bars.last()
                    if (e.trades.isEmpty()) {
                        pending?.let { fill(e, it, last.close, last.minute) }
                        e.forceClose(last)?.let { fill(e, it, last.close, last.minute) }
                    }
                    val t = e.trades.firstOrNull() ?: continue
                    pnl[k] = t.netPnl(cfg.costs)
                    ce[k] = Session.elapsed(t.closeMinute)
                    reason[k] = t.closeReason
                }
                out += Cand(su.symbol, su.date, el, dir, Features.extract(su.ctx, obs[i], dir), pnl, reason, ce)
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
    private fun f0(x: Double) = "%.0f".format(x)
    private fun f1(x: Double) = "%.1f".format(x)
    private fun mark(b: Boolean) = if (b) "✔" else "✘"

    companion object {
        /** 费率显示:0.00025 → 万2.5 */
        fun feeText(rate: Double): String = "万%s".format(trim(rate * 10000))
        private fun trim(x: Double) = if (x == Math.floor(x)) "%.0f".format(x) else "%.3f".format(x).trimEnd('0')

        /** Wilson 置信区间(95%),sign=-1 下限,+1 上限 */
        fun wilson(w: Int, n: Int, sign: Int): Double {
            if (n == 0) return 0.0
            val z = 1.96
            val p = w.toDouble() / n
            val d = 1 + z * z / n
            val c = p + z * z / (2 * n)
            val r = z * sqrt(p * (1 - p) / n + z * z / (4.0 * n * n))
            return ((c + sign * r) / d).coerceIn(0.0, 1.0)
        }
    }
}

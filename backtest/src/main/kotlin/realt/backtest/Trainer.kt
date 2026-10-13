package realt.backtest

import realt.core.engine.CloseReason
import realt.core.engine.Observation
import realt.core.engine.SimBroker
import realt.core.engine.TConfig
import realt.core.engine.TDirection
import realt.core.engine.TEngine
import realt.core.ml.BinnedData
import realt.core.ml.Calibration
import realt.core.ml.FeatureGroup
import realt.core.ml.Features
import realt.core.ml.Gbdt
import realt.core.ml.TradeModel
import realt.core.model.Session
import realt.core.model.Symbol
import realt.core.signal.Ind
import realt.core.signal.SignalFamily
import java.io.File
import java.util.concurrent.Callable
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicLong
import kotlin.math.sqrt
import kotlin.random.Random

/**
 * 训练:用历史分时数据学习"什么情况下做T能赢",输出参数包。
 *
 * 1. 候选生成(多线程,按股票并行):逐分钟回放,每出现转折结构(缠论 1/5 分钟买卖点、布林刺破收回、RSI 拐头)
 *    记一个候选点及其约 110 个特征;对每组"离场 × 挂单"参数,用与实盘完全相同的引擎与撮合规则模拟这一笔
 *    (当日必须完成),扣全部费用后盈利为 1。
 * 2. 特征分箱一次,所有模型共享。
 * 3. 滚动前推(walk-forward):前 40% 交易日为起始训练期,其余分 [folds] 段,每段只用它之前的数据训练、
 *    用留出集早停并做概率校准,然后预测该段。最后一段是测试集,不参与任何选择。
 * 4. 离场/挂单参数筛选:各组参数的样本外校准概率上,按实盘规则模拟出手,选出达标且净盈亏最大的组合与阈值。
 * 5. 模块消融:逐组去掉一层/一个模块(缠论、MACD、布林、L1、L2、L3、指数……)重新做滚动前推,
 *    衡量每层的贡献;去掉后反而更好的组被剔除,得到最优组合。
 * 6. 最后一段的模型(只用测试段之前的数据训练)即参数包模型,测试段结果为样本外表现。
 * 所有模型训练任务在线程池中并行。
 */
class Trainer(
    private val data: MarketData,
    private val cfg: BacktestConfig,
    private val target: Double = 0.95,
    private val firstTrainRatio: Double = 0.4,
    private val folds: Int = 4,
    private val minTrades: Int = 30,
    private val threads: Int = Runtime.getRuntime().availableProcessors().coerceAtLeast(1),
    private val progress: Progress = Progress(),
) {
    // ---------------- 候选点(列存) ----------------
    private class Block(val symbol: Int, cap: Int, val nf: Int, val nk: Int) {
        var n = 0
        var date = IntArray(cap); var elapsed = ShortArray(cap); var dir = ByteArray(cap); var family = ByteArray(cap)
        var x = FloatArray(cap * nf); var pnl = FloatArray(cap * nk); var reason = ByteArray(cap * nk); var closeE = ShortArray(cap * nk)
        fun ensure() {
            if (n < date.size) return
            val c = date.size * 2
            date = date.copyOf(c); elapsed = elapsed.copyOf(c); dir = dir.copyOf(c); family = family.copyOf(c)
            x = x.copyOf(c * nf); pnl = pnl.copyOf(c * nk); reason = reason.copyOf(c * nk); closeE = closeE.copyOf(c * nk)
        }
    }

    /** 全部候选点(合并后) */
    private class Cands(
        val n: Int, val nk: Int, val symbol: IntArray, val date: IntArray, val elapsed: ShortArray, val dir: ByteArray,
        val family: ByteArray, val pnl: FloatArray, val reason: ByteArray, val closeE: ShortArray,
    ) {
        fun pnl(r: Int, k: Int) = pnl[r * nk + k]
        fun closeE(r: Int, k: Int) = closeE[r * nk + k].toInt()
        fun reason(r: Int, k: Int) = reason[r * nk + k].toInt()
        /** 该参数下这笔是否成交(挂单可能未成交) */
        fun filled(r: Int, k: Int) = !pnl(r, k).isNaN()
    }

    data class Eval(val n: Int, val wins: Int, val pnl: Double, val sumRet: Double) {
        val winRate get() = if (n == 0) 0.0 else wins.toDouble() / n
        val avgBp get() = if (n == 0) 0.0 else sumRet / n * 10000
        val winLow: Double get() = wilson(wins, n, -1)
        val winHigh: Double get() = wilson(wins, n, 1)
        fun meets(target: Double, minN: Int) = n >= minN && winRate >= target && pnl > 0
    }

    class Result(val model: TradeModel, val report: String)

    /** 离场 × 挂单参数网格。开仓时间窗放宽到收盘前 20 分钟,由模型学习时间的影响 */
    fun exitGrid(base: TConfig): List<TConfig> {
        val b = base.copy(noOpenBeforeCloseMin = 20, regimeExit = false, tpLimit = true)
        return buildList {
            for (off in listOf(0.0, 0.1)) for (tp in listOf(0.12, 0.18, 0.25)) for (sl in listOf(1.0, 1.6))
                add(b.copy(entryOffsetAtr = off, takeProfitAtr = tp, stopLossAtr = sl))
        }
    }

    private fun gbdtParams(screen: Boolean) = Gbdt.Params(
        nTrees = 400, maxDepth = 4, learningRate = 0.05, minLeaf = 100, lambda = 2.0,
        rowSample = if (screen) 0.5 else 0.7, colSample = 0.7, earlyStop = 40,
    )

    fun run(symbols: List<Symbol>, outDir: File): Result {
        val t0 = System.nanoTime()
        val grid = exitGrid(cfg.tConfig)
        val nk = grid.size
        val nf = Features.size
        if (!data.hasIndexMinutes) progress.log("提示:未找到指数 1 分钟线(sh000001.lc1),指数分时特征将为 0。")
        progress.log("训练线程 $threads,标的 ${symbols.size} 只,特征 $nf 个,离场×挂单参数 $nk 组,目标胜率 ${pct(target)}")

        // ---------- 1. 候选生成 ----------
        val blocks = java.util.Collections.synchronizedList(ArrayList<Block>())
        val dayCount = java.util.concurrent.ConcurrentHashMap<Int, Int>()
        val done = AtomicInteger(); val candCount = AtomicLong()
        val bt = Backtester(data, cfg)
        progress.start("1/5 候选生成", symbols.size)
        runParallel(symbols.indices.map { si ->
            Callable {
                val blk = Block(si, 1024, nf, nk)
                try {
                    for (su in bt.setups(symbols[si])) {
                        candidatesOf(su, grid, blk)
                        dayCount.merge(su.date, 1, Int::plus)
                    }
                } catch (e: Throwable) {
                    progress.log("  ${symbols[si]} 失败: ${e.message}")
                }
                blocks += blk
                candCount.addAndGet(blk.n.toLong())
                progress.step(done.incrementAndGet(), "候选点 ${"%,d".format(candCount.get())}")
            }
        })
        progress.finish("候选点 ${"%,d".format(candCount.get())}")

        val dates = dayCount.keys.sorted()
        require(dates.size >= 25) { "交易日太少(${dates.size}),至少需要 25 个交易日的 1 分钟数据" }
        val startIdx = (dates.size * firstTrainRatio).toInt()
        val foldStarts = (0 until folds).map { dates[startIdx + (dates.size - startIdx) * it / folds] }
        val testStart = foldStarts.last()

        // ---------- 2. 合并 + 分箱 ----------
        progress.start("2/5 特征分箱", nf)
        val sorted = blocks.sortedBy { it.symbol }
        val n = sorted.sumOf { it.n }
        require(n >= 1000) { "候选点太少($n),请增加股票数量或数据天数" }
        val offs = IntArray(sorted.size + 1).also { for (i in sorted.indices) it[i + 1] = it[i] + sorted[i].n }
        val c = run {
            val sym = IntArray(n); val date = IntArray(n); val el = ShortArray(n); val dir = ByteArray(n); val fam = ByteArray(n)
            val pnl = FloatArray(n * nk); val rsn = ByteArray(n * nk); val ce = ShortArray(n * nk)
            for ((bi, b) in sorted.withIndex()) {
                val o = offs[bi]
                for (i in 0 until b.n) { sym[o + i] = b.symbol }
                System.arraycopy(b.date, 0, date, o, b.n); System.arraycopy(b.elapsed, 0, el, o, b.n)
                System.arraycopy(b.dir, 0, dir, o, b.n); System.arraycopy(b.family, 0, fam, o, b.n)
                System.arraycopy(b.pnl, 0, pnl, o * nk, b.n * nk); System.arraycopy(b.reason, 0, rsn, o * nk, b.n * nk)
                System.arraycopy(b.closeE, 0, ce, o * nk, b.n * nk)
            }
            Cands(n, nk, sym, date, el, dir, fam, pnl, rsn, ce)
        }
        // 分位点只用起始训练期的样本(最多 30 万)
        val initRows = (0 until n).filter { c.date[it] < foldStarts[0] }
        val edgeRows = (if (initRows.size > 300_000) initRows.shuffled(Random(1)).take(300_000) else initRows).toIntArray()
        fun xAt(r: Int, f: Int): Double {
            var lo = 0; var hi = sorted.size - 1
            while (lo < hi) { val m = (lo + hi + 1) ushr 1; if (offs[m] <= r) lo = m else hi = m - 1 }
            return sorted[lo].x[(r - offs[lo]) * nf + f].toDouble()
        }
        val edges = Array(nf) { f ->
            val vals = DoubleArray(edgeRows.size) { xAt(edgeRows[it], f) }.also { it.sort() }
            val e = ArrayList<Double>()
            for (b in 1 until 64) {
                if (vals.isEmpty()) break
                val v = vals[(b.toLong() * (vals.size - 1) / 64).toInt()]
                if (e.isEmpty() || v > e.last()) e += v
            }
            e.toDoubleArray()
        }
        val cols = Array(nf) { ByteArray(n) }
        val binDone = AtomicInteger()
        runParallel((0 until nf).map { f ->
            Callable {
                val e = edges[f]; val col = cols[f]
                for ((bi, b) in sorted.withIndex()) {
                    val o = offs[bi]
                    for (i in 0 until b.n) col[o + i] = BinnedData.binOf(e, b.x[i * nf + f].toDouble()).toByte()
                }
                progress.step(binDone.incrementAndGet())
            }
        })
        val bd = BinnedData(edges, cols)
        sorted.forEach { it.x = FloatArray(0) } // 释放原始特征
        progress.finish()

        val foldOf = IntArray(n) { r -> foldStarts.indexOfLast { c.date[r] >= it } } // -1 = 起始训练期
        val selRows = (0 until n).filter { foldOf[it] in 0 until folds - 1 }.toIntArray()
        val testRows = (0 until n).filter { foldOf[it] == folds - 1 }.toIntArray()
        progress.log("切分:起始训练 ${dates.first()}~<${foldStarts[0]};样本外 $folds 段 " +
            foldStarts.joinToString(" ") { "≥$it" } + ";最后一段(≥$testStart)为测试集")

        // ---------- 3. 离场/挂单参数筛选 ----------
        val oof = Array(nk) { DoubleArray(n) { Double.NaN } }
        val jobs3 = (0 until nk).flatMap { k -> (0 until folds).map { f -> k to f } }
        val done3 = AtomicInteger()
        progress.start("3/5 参数筛选", jobs3.size)
        runParallel(jobs3.map { (k, f) ->
            Callable {
                val fm = trainFold(bd, c, foldOf, dates, foldStarts, k, f, null, screen = true)
                for ((i, r) in fm.rows.withIndex()) oof[k][r] = fm.pred[i]
                progress.step(done3.incrementAndGet(), "离场组${k + 1} 第${f + 1}段")
            }
        })
        progress.finish()

        data class Choice(val k: Int, val thr: Double, val e: Eval)
        fun bestFor(k: Int, p: DoubleArray, rows: IntArray): Choice {
            var best: Choice? = null
            for (ti in 0..99) {
                val thr = (500 + ti * 5) / 1000.0
                val ch = Choice(k, thr, simulate(c, rows, p, k, thr, grid[k]))
                if (better(ch.e, best?.e)) best = ch
            }
            return best!!
        }
        val perK = (0 until nk).map { k -> bestFor(k, oof[k], selRows) }
        var choice = perK.reduce { a, b -> if (better(b.e, a.e)) b else a }
        val kStar = choice.k

        // ---------- 4. 模块消融 ----------
        val groups = FeatureGroup.entries
        val ablP = HashMap<FeatureGroup, DoubleArray>()
        val jobs4 = groups.flatMap { g -> (0 until folds - 1).map { f -> g to f } }
        val done4 = AtomicInteger()
        progress.start("4/5 模块消融", jobs4.size)
        runParallel(jobs4.map { (g, f) ->
            Callable {
                val cols4 = (0 until nf).filter { Features.specs[it].group != g }.toIntArray()
                val fm = trainFold(bd, c, foldOf, dates, foldStarts, kStar, f, cols4, screen = true)
                val arr = synchronized(ablP) { ablP.getOrPut(g) { DoubleArray(n) { Double.NaN } } }
                for ((i, r) in fm.rows.withIndex()) arr[r] = fm.pred[i]
                progress.step(done4.incrementAndGet(), "去掉「${g.label}」第${f + 1}段")
            }
        })
        progress.finish()
        val fullAuc = auc(c, selRows, oof[kStar], kStar)
        val fullBest = bestFor(kStar, oof[kStar], selRows)
        val selFolds = (0 until folds - 1).map { f -> selRows.filter { foldOf[it] == f }.toIntArray() }
        val fullFoldAuc = selFolds.map { auc(c, it, oof[kStar], kStar) }
        data class Abl(val g: FeatureGroup, val auc: Double, val best: Choice, val foldGain: List<Double>)
        val abl = groups.map { g ->
            val p = ablP[g]!!
            Abl(g, auc(c, selRows, p, kStar), bestFor(kStar, p, selRows),
                selFolds.mapIndexed { i, rows -> auc(c, rows, p, kStar) - fullFoldAuc[i] })
        }
        // 剔除条件(防止把噪声当结论):去掉后 AUC 平均提升 ≥ 0.002、且在每一段都提升,且可达最好结果不变差
        val dropped = abl.filter { a ->
            a.auc - fullAuc >= 0.002 && a.foldGain.all { it > 0 } && !better(fullBest.e, a.best.e)
        }.map { it.g }.toSet()
        val finalCols = if (dropped.isEmpty()) null else (0 until nf).filter { Features.specs[it].group !in dropped }.toIntArray()

        // ---------- 5. 最终组合 ----------
        progress.start("5/5 最终模型", folds)
        val finalP = DoubleArray(n) { Double.NaN }
        val done5 = AtomicInteger()
        val finalFolds = HashMap<Int, FoldModel>()
        runParallel((0 until folds).map { f ->
            Callable {
                // 最终组合在每一段都以完整强度重训,阈值选择与部署模型同一口径
                val fm = trainFold(bd, c, foldOf, dates, foldStarts, kStar, f, finalCols, screen = false)
                for ((i, r) in fm.rows.withIndex()) finalP[r] = fm.pred[i]
                synchronized(finalFolds) { finalFolds[f] = fm }
                progress.step(done5.incrementAndGet(), "第${f + 1}段")
            }
        })
        progress.finish()
        choice = bestFor(kStar, finalP, selRows)
        val ec = grid[kStar]
        val testEval = simulate(c, testRows, finalP, kStar, choice.thr, ec)
        val testAll = simulate(c, testRows, finalP, kStar, 0.0, ec)
        val deploy = finalFolds[folds - 1]!!
        val trainEnd = dates[dates.indexOf(testStart) - 1]
        val meta = linkedMapOf(
            "target" to target.toString(),
            "selTrades" to choice.e.n.toString(), "selWinRate" to "%.4f".format(choice.e.winRate),
            "testStart" to testStart.toString(), "testEnd" to dates.last().toString(),
            "testTrades" to testEval.n.toString(), "testWinRate" to "%.4f".format(testEval.winRate),
            "testPnl" to "%.2f".format(testEval.pnl), "symbols" to symbols.size.toString(),
            "droppedGroups" to dropped.joinToString("/") { it.name },
        )
        val model = TradeModel(ec, cfg.costs, choice.thr, deploy.model, dates.first(), trainEnd, meta, deploy.calib)

        val minutes = (System.nanoTime() - t0) / 6e10
        val report = report(
            symbols.size, dates, n, foldStarts, testStart, grid, perK.map { Triple(it.k, it.thr, it.e) },
            kStar, choice.thr, choice.e, testEval, testAll, fullAuc, fullBest.e,
            abl.map { Triple(it.g, it.auc, it.best.e) }, dropped, c, selRows, testRows, finalP,
            deploy.model, finalFolds.values.map { it.model }, trainEnd, minutes,
        )
        outDir.mkdirs()
        File(outDir, "realt-model.txt").writeText(model.serialize())
        File(outDir, "train-report.md").writeText(report)
        return Result(model, report)
    }

    // ---------------- 单段训练 ----------------
    private class FoldModel(val model: Gbdt, val calib: Calibration, val rows: IntArray, val pred: DoubleArray)

    /**
     * 第 [f] 段:用该段之前的数据训练(最后 15% 交易日留出做早停与校准),预测该段并校准。
     * 只使用在离场组 [k] 下成交了的候选点。
     */
    private fun trainFold(
        bd: BinnedData, c: Cands, foldOf: IntArray, dates: List<Int>, foldStarts: List<Int>,
        k: Int, f: Int, cols: IntArray?, screen: Boolean,
    ): FoldModel {
        val start = foldStarts[f]
        val end = if (f + 1 < foldStarts.size) foldStarts[f + 1] else Int.MAX_VALUE
        val trainDates = dates.filter { it < start }
        val esStart = trainDates[(trainDates.size * 0.85).toInt().coerceAtMost(trainDates.size - 1)]
        val y = ByteArray(c.n) { r -> if (c.filled(r, k) && c.pnl(r, k) > 0) 1 else 0 }
        val tr = ArrayList<Int>(); val es = ArrayList<Int>(); val pr = ArrayList<Int>()
        for (r in 0 until c.n) {
            if (!c.filled(r, k)) continue
            val d = c.date[r]
            when {
                d < esStart -> tr += r
                d < start -> es += r
                d < end -> pr += r
            }
        }
        val m = Gbdt.trainBinned(bd, y, tr.toIntArray(), es.toIntArray(), cols, gbdtParams(screen).copy(seed = 7 + k * 31 + f))
        val esScores = DoubleArray(es.size) { m.predictBinned(bd, es[it]) }
        val calib = Calibration.fit(esScores, IntArray(es.size) { y[es[it]].toInt() })
        val rows = pr.toIntArray()
        return FoldModel(m, calib, rows, DoubleArray(rows.size) { calib.apply(m.predictBinned(bd, rows[it])) })
    }

    // ---------------- 候选生成与打标签 ----------------
    private fun candidatesOf(su: DaySetup, grid: List<TConfig>, blk: Block) {
        if ((su.shares * grid[0].tRatio / 100).toInt() <= 0) return
        val observer = su.observer()
        val bars = su.dayBars
        val obs = ArrayList<Observation>(bars.size)
        for (b in bars) obs += observer.observe(b)
        val nf = blk.nf; val nk = blk.nk
        for (i in bars.indices) {
            val el = Session.elapsed(bars[i].minute)
            if (el < grid[0].noOpenAfterOpenMin || Session.MINUTES_PER_DAY - el <= grid[0].noOpenBeforeCloseMin) continue
            for (dir in Features.candidates(obs[i])) {
                blk.ensure()
                val row = blk.n
                blk.date[row] = su.date; blk.elapsed[row] = el.toShort()
                blk.dir[row] = if (dir == TDirection.SELL_FIRST) 1 else 0
                blk.family[row] = familyOf(obs[i], dir).toByte()
                val x = Features.extract(su.ctx, obs[i], dir)
                for (f in 0 until nf) blk.x[row * nf + f] = x[f].toFloat()
                for ((k, g) in grid.withIndex()) {
                    val e = TEngine(g, cfg.costs, su.ctx, su.shares, null)
                    val broker = SimBroker(e, cfg.costs.slippage)
                    val first = e.forceOpen(dir, bars[i], obs[i])
                    broker.onBar(bars[i]) { first }
                    var j = i + 1
                    while (j < bars.size && e.trades.isEmpty() && (e.isOpen || broker.working != null)) {
                        val jj = j
                        broker.onBar(bars[jj]) { if (e.isOpen) e.decide(bars[jj], obs[jj]) else null }
                        j++
                    }
                    if (e.trades.isEmpty() && (e.isOpen || broker.working != null)) broker.endOfDay(bars.last())
                    val t = e.trades.firstOrNull()
                    val idx = row * nk + k
                    if (t == null) {
                        blk.pnl[idx] = Float.NaN; blk.reason[idx] = -1; blk.closeE[idx] = -1
                    } else {
                        blk.pnl[idx] = t.netPnl(cfg.costs).toFloat()
                        blk.reason[idx] = t.closeReason.ordinal.toByte()
                        blk.closeE[idx] = Session.elapsed(t.closeMinute).toShort()
                    }
                }
                blk.n++
            }
        }
    }

    /** 候选点的信号来源:1 缠论1分钟 2 布林 3 RSI 4 仅缠论5分钟 */
    private fun familyOf(o: Observation, dir: TDirection): Int {
        val sell = dir == TDirection.SELL_FIRST
        val fam = if (sell) o.sig.sellFamily else o.sig.buyFamily
        return when (fam) {
            SignalFamily.CHAN -> 1
            SignalFamily.BOLL -> 2
            SignalFamily.VWAP -> 3
            SignalFamily.NONE -> if ((if (sell) o.sig.ind[Ind.C5_SELL] else o.sig.ind[Ind.C5_BUY]) > 0) 4 else 0
        }
    }

    // ---------------- 选择与模拟 ----------------
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

    /** 按实盘规则挑出会真正出手的候选:同一时间只持一笔、每日上限、冷却;挂单未成交的不占用 */
    private fun taken(c: Cands, rows: IntArray, p: DoubleArray, k: Int, thr: Double, ec: TConfig): IntArray {
        val out = ArrayList<Int>()
        val byDay = HashMap<Long, MutableList<Int>>()
        for (r in rows) {
            if (p[r].isNaN() || p[r] < thr || !c.filled(r, k)) continue
            byDay.getOrPut(c.symbol[r].toLong() * 100_000_000L + c.date[r]) { ArrayList(4) } += r
        }
        for (ids in byDay.values) {
            ids.sortWith(compareBy<Int>({ c.elapsed[it] }, { -p[it] }))
            var busyUntil = -1; var lastClose = -1000; var count = 0
            for (r in ids) {
                if (count >= ec.maxTradesPerDay) break
                val el = c.elapsed[r].toInt()
                if (el <= busyUntil || el - lastClose < ec.cooldownMin) continue
                out += r; count++
                busyUntil = c.closeE(r, k); lastClose = busyUntil
            }
        }
        return out.toIntArray()
    }

    private fun simulate(c: Cands, rows: IntArray, p: DoubleArray, k: Int, thr: Double, ec: TConfig): Eval {
        var nn = 0; var wins = 0; var pnl = 0.0; var sumRet = 0.0
        val notional = cfg.baseValue * ec.tRatio
        for (r in taken(c, rows, p, k, thr, ec)) {
            val v = c.pnl(r, k).toDouble()
            nn++; if (v > 0) wins++
            pnl += v; sumRet += v / notional
        }
        return Eval(nn, wins, pnl, sumRet)
    }

    /** 样本外 AUC(越高说明模型越能区分赢单与输单) */
    private fun auc(c: Cands, rows: IntArray, p: DoubleArray, k: Int): Double {
        val rs = rows.filter { !p[it].isNaN() && c.filled(it, k) }.sortedBy { p[it] }
        var pos = 0L; var neg = 0L; var rankSum = 0.0
        var i = 0
        while (i < rs.size) {
            var j = i
            while (j + 1 < rs.size && p[rs[j + 1]] == p[rs[i]]) j++
            val avgRank = (i + j) / 2.0 + 1
            for (t in i..j) if (c.pnl(rs[t], k) > 0) { pos++; rankSum += avgRank } else neg++
            i = j + 1
        }
        if (pos == 0L || neg == 0L) return 0.5
        return (rankSum - pos * (pos + 1) / 2.0) / (pos.toDouble() * neg)
    }

    private fun runParallel(tasks: List<Callable<Unit>>) {
        val pool = Executors.newFixedThreadPool(threads)
        try {
            pool.invokeAll(tasks).forEach { it.get() }
        } finally {
            pool.shutdown()
        }
    }

    // ---------------- 报告 ----------------
    private fun report(
        nSymbols: Int, dates: List<Int>, nCand: Int, foldStarts: List<Int>, testStart: Int, grid: List<TConfig>,
        perK: List<Triple<Int, Double, Eval>>, kStar: Int, thr: Double, sel: Eval, test: Eval, testAll: Eval,
        fullAuc: Double, fullBest: Eval, abl: List<Triple<FeatureGroup, Double, Eval>>, dropped: Set<FeatureGroup>,
        c: Cands, selRows: IntArray, testRows: IntArray, finalP: DoubleArray, deploy: Gbdt, all: List<Gbdt>,
        trainEnd: Int, minutes: Double,
    ): String = buildString {
        val ec = grid[kStar]
        fun cfgText(g: TConfig) = "${if (g.entryOffsetAtr > 0) "挂单${g.entryOffsetAtr}ATR" else "市价"} 止盈${g.takeProfitAtr} 止损${g.stopLossAtr}"
        appendLine("# realT 训练报告")
        appendLine()
        appendLine("- 标的 $nSymbols 只,交易日 ${dates.first()} ~ ${dates.last()}(${dates.size} 天),候选点 ${"%,d".format(nCand)},用时 ${"%.1f".format(minutes)} 分钟($threads 线程)")
        appendLine("- 费率:佣金 ${feeText(cfg.costs.commissionRate)}(最低 ${cfg.costs.minCommission} 元)、印花税 ${feeText(cfg.costs.stampDutyRate)}(仅卖出)、过户费 ${feeText(cfg.costs.transferRate)}、滑点单边 ${feeText(cfg.costs.slippage)};只做当日T")
        appendLine("- 指数分时数据:${if (data.hasIndexMinutes) "有" else "**无**(指数特征为 0)"}")
        appendLine("- 滚动前推:起始训练 < ${foldStarts[0]};样本外 $folds 段,前 ${folds - 1} 段用于挑参数,**最后一段 ≥ $testStart 为测试集**;每段模型都经过概率校准")
        appendLine("- 目标:胜率 ≥ ${pct(target)} 且净盈亏 > 0,至少 $minTrades 笔")
        appendLine()
        appendLine("## 结论")
        appendLine()
        appendLine("| 数据段 | 做T次数 | 胜率 | 胜率95%置信区间 | 平均每笔(bp) | 净盈亏(元) | 达标 |")
        appendLine("|---|---|---|---|---|---|---|")
        appendLine("| 选择段(样本外预测) | ${sel.n} | ${pct(sel.winRate)} | ${pct(sel.winLow)}~${pct(sel.winHigh)} | ${f1(sel.avgBp)} | ${f0(sel.pnl)} | ${mark(sel.meets(target, minTrades))} |")
        appendLine("| **测试段(样本外)** | ${test.n} | ${pct(test.winRate)} | ${pct(test.winLow)}~${pct(test.winHigh)} | ${f1(test.avgBp)} | ${f0(test.pnl)} | ${mark(test.meets(target, 1))} |")
        appendLine("| 测试段·不用模型过滤 | ${testAll.n} | ${pct(testAll.winRate)} | - | ${f1(testAll.avgBp)} | ${f0(testAll.pnl)} | - |")
        appendLine()
        appendLine("选定:${cfgText(ec)},出手阈值(校准胜率)$thr;参数包模型 ${deploy.trees.size} 棵树。")
        appendLine()
        val selDays = dates.count { it >= foldStarts[0] && it < testStart }
        appendLine("> 出手频率:选择段约 ${f1(sel.n.toDouble() / maxOf(1, selDays))} 笔/交易日(全部 $nSymbols 只合计)。")
        appendLine("> 置信区间:30 笔全胜,真实胜率的 95% 置信下限只有约 89%;要让下限 ≥ 95%,大约需要 100 笔中最多错 1 笔。")
        if (!sel.meets(target, minTrades)) {
            appendLine("> **未达成 ${pct(target)} 目标**:当前数据与特征下,满足笔数要求时能达到的最高样本外胜率见下文各表。")
        }
        appendLine()
        appendLine("## 1. 离场 × 挂单参数筛选(选择段,各组最优阈值)")
        appendLine()
        appendLine("| # | 开仓 | 止盈ATR | 止损ATR | 阈值 | 次数 | 胜率 | 95%置信区间 | 平均(bp) | 净盈亏 |")
        appendLine("|---|---|---|---|---|---|---|---|---|---|")
        for ((k, t, e) in perK) {
            val g = grid[k]
            appendLine("| ${k + 1}${if (k == kStar) "★" else ""} | ${if (g.entryOffsetAtr > 0) "挂单${g.entryOffsetAtr}" else "市价"} | ${g.takeProfitAtr} | ${g.stopLossAtr} | ${"%.3f".format(t)} | ${e.n} | ${pct(e.winRate)} | ${pct(e.winLow)}~${pct(e.winHigh)} | ${f1(e.avgBp)} | ${f0(e.pnl)} |")
        }
        appendLine()
        appendLine("## 2. 各层 / 模块贡献(消融:去掉该组后重新滚动前推)")
        appendLine()
        appendLine("AUC 衡量模型区分赢单/输单的能力(0.5 = 随机)。全部特征:AUC ${"%.4f".format(fullAuc)},最好结果 ${fullBest.n} 笔 / ${pct(fullBest.winRate)} / ${f0(fullBest.pnl)} 元。")
        appendLine()
        appendLine("| 去掉的层/模块 | 特征数 | AUC | ΔAUC | 最好结果:次数 | 胜率 | 净盈亏 | 结论 |")
        appendLine("|---|---|---|---|---|---|---|---|")
        for ((g, a, e) in abl.sortedBy { it.second }) {
            val d = a - fullAuc
            val verdict = when {
                g in dropped -> "剔除(去掉后各段一致更好)"
                d < -0.005 -> "**关键**"
                d < -0.001 -> "有贡献"
                else -> "贡献很小"
            }
            appendLine("| ${g.label} | ${Features.columnsOf(g).size} | ${"%.4f".format(a)} | ${"%+.4f".format(d)} | ${e.n} | ${pct(e.winRate)} | ${f0(e.pnl)} | $verdict |")
        }
        appendLine()
        appendLine("最终组合:${if (dropped.isEmpty()) "全部层与模块" else "剔除 " + dropped.joinToString("、") { it.label }}。")
        appendLine()
        appendLine("## 3. 信号来源效能(选定离场参数)")
        appendLine()
        appendLine("| 信号来源 | 候选点 | 不过滤胜率 | 不过滤平均(bp) | 模型选中次数 | 选中胜率 |")
        appendLine("|---|---|---|---|---|---|")
        val tk = (taken(c, selRows, finalP, kStar, thr, ec) + taken(c, testRows, finalP, kStar, thr, ec)).toSet()
        val notional = cfg.baseValue * ec.tRatio
        for ((code, name) in listOf(1 to "缠论 1 分钟买卖点", 4 to "缠论 5 分钟买卖点", 2 to "布林刺破收回", 3 to "RSI 极值拐头")) {
            val rows = (selRows + testRows).filter { c.family[it].toInt() == code && c.filled(it, kStar) }
            if (rows.isEmpty()) continue
            val w = rows.count { c.pnl(it, kStar) > 0 }
            val bp = rows.sumOf { c.pnl(it, kStar).toDouble() / notional } / rows.size * 10000
            val ts = rows.filter { it in tk }
            appendLine("| $name | ${rows.size} | ${pct(w.toDouble() / rows.size)} | ${f1(bp)} | ${ts.size} | ${pct(if (ts.isEmpty()) 0.0 else ts.count { c.pnl(it, kStar) > 0 }.toDouble() / ts.size)} |")
        }
        appendLine()
        appendLine("## 4. 阈值—胜率—出手次数(选定参数,校准胜率)")
        appendLine()
        appendLine("| 阈值 | 选择段次数 | 胜率 | 95%下限 | 平均(bp) | 净盈亏 | 测试次数 | 测试胜率 | 测试净盈亏 |")
        appendLine("|---|---|---|---|---|---|---|---|---|")
        for (ti in 0..99) {
            val t = (500 + ti * 5) / 1000.0
            if (ti % 10 != 0 && t < 0.85) continue
            val v = simulate(c, selRows, finalP, kStar, t, ec)
            val te = simulate(c, testRows, finalP, kStar, t, ec)
            if (v.n == 0 && te.n == 0) continue
            appendLine("| ${"%.3f".format(t)} | ${v.n} | ${pct(v.winRate)} | ${pct(v.winLow)} | ${f1(v.avgBp)} | ${f0(v.pnl)} | ${te.n} | ${pct(te.winRate)} | ${f0(te.pnl)} |")
        }
        appendLine()
        appendLine("## 5. 亏损单分析(选择段 + 测试段)")
        appendLine()
        val losers = tk.filter { c.pnl(it, kStar) <= 0 }
        appendLine("出手 ${tk.size} 笔中亏损 ${losers.size} 笔:")
        appendLine()
        appendLine("| 方向 | 平仓原因 | 亏损笔数 | 平均亏损(元) |")
        appendLine("|---|---|---|---|")
        for ((key, g) in losers.groupBy { (if (c.dir[it].toInt() == 1) "先卖后买" else "先买后卖") to CloseReason.entries[c.reason(it, kStar)].label }
            .toSortedMap(compareBy({ it.first }, { it.second }))) {
            appendLine("| ${key.first} | ${key.second} | ${g.size} | ${f0(g.map { c.pnl(it, kStar).toDouble() }.average())} |")
        }
        appendLine()
        appendLine("## 6. 最重要的特征")
        appendLine()
        appendLine("| 特征 | 所属层/模块 | 相对重要性 |")
        appendLine("|---|---|---|")
        val imp = DoubleArray(Features.size)
        for (m in all) for (i in m.importance.indices) imp[i] += m.importance[i]
        val tot = imp.sum().takeIf { it > 0 } ?: 1.0
        imp.indices.sortedByDescending { imp[it] }.take(25).forEach {
            appendLine("| ${Features.names[it]} | ${Features.specs[it].group.label} | ${pct(imp[it] / tot)} |")
        }
        val byGroup = FeatureGroup.entries.associateWith { g -> Features.columnsOf(g).sumOf { imp[it] } / tot }
        appendLine()
        appendLine("按层/模块汇总(最终组合各段模型):" + byGroup.entries.sortedByDescending { it.value }.joinToString(",") { "${it.key.label} ${pct(it.value)}" })
        appendLine()
        appendLine("参数包:`realt-model.txt`(训练数据截止 $trainEnd)。样本外回测:`realt backtest --model realt-model.txt --start $testStart`")
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
            val cc = p + z * z / (2 * n)
            val r = z * sqrt(p * (1 - p) / n + z * z / (4.0 * n * n))
            return ((cc + sign * r) / d).coerceIn(0.0, 1.0)
        }
    }
}

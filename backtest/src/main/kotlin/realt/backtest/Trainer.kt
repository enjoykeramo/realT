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
import realt.core.ml.StrategySpec
import realt.core.ml.TradeModel
import realt.core.model.Session
import realt.core.model.Symbol
import realt.core.regime.MarketLevel
import java.io.File
import java.util.concurrent.Callable
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicLong
import kotlin.math.sqrt
import kotlin.random.Random

/**
 * 训练 = 迭代寻优:配置策略 → 回测打分 → 保留好的 → 二次验证 → 在好策略附近继续变异 → 直到通过验证,最后一次性验收。
 *
 * 时间上切成互不重叠的四段(都只用更早的数据训练模型,不看未来):
 *   ① 模型训练期(前 40% 交易日):只用来训练胜率模型
 *   ② 寻优期(40%~70%):每一轮对几百条策略回测打分,保留精英并变异出下一代
 *   ③ 验证期(70%~85%):每轮的精英策略做"二次验证",通过者进入候选池
 *   ④ 验收期(最后 15%):迭代全部结束后,只对最终选中的一条策略验收一次
 * 验收期绝不参与迭代:如果反复用同一段数据"迭代到达标",试过的策略足够多时总能碰巧达标(过拟合),验收期就是防这个的。
 */
class Trainer(
    private val data: MarketData,
    private val cfg: BacktestConfig,
    private val target: Double = 0.95,
    private val minTrades: Int = 30,
    private val minValTrades: Int = 30,
    private val rounds: Int = 60,
    private val patience: Int = 12,
    private val population: Int = 400,
    private val ablation: Boolean = true,
    private val threads: Int = Runtime.getRuntime().availableProcessors().coerceAtLeast(1),
    private val progress: Progress = Progress(),
    private val seed: Int = 20261014,
) {
    // ---------------- 候选点(列存) ----------------
    private class Block(val symbol: Int, cap: Int, val nf: Int, val nk: Int) {
        var n = 0
        var date = IntArray(cap); var elapsed = ShortArray(cap); var dir = ByteArray(cap); var family = ByteArray(cap)
        var regime = ByteArray(cap); var atr = FloatArray(cap); var vdev = FloatArray(cap); var trend = FloatArray(cap); var score = FloatArray(cap)
        /** 特征分箱后的箱号(每值 1 字节) */
        var x = ByteArray(cap * nf); var pnl = FloatArray(cap * nk); var reason = ByteArray(cap * nk); var closeE = ShortArray(cap * nk)
        fun ensure() {
            if (n < date.size) return
            val c = date.size * 2
            date = date.copyOf(c); elapsed = elapsed.copyOf(c); dir = dir.copyOf(c); family = family.copyOf(c)
            regime = regime.copyOf(c); atr = atr.copyOf(c); vdev = vdev.copyOf(c); trend = trend.copyOf(c); score = score.copyOf(c)
            x = x.copyOf(c * nf); pnl = pnl.copyOf(c * nk); reason = reason.copyOf(c * nk); closeE = closeE.copyOf(c * nk)
        }
        fun release() {
            date = IntArray(0); elapsed = ShortArray(0); dir = ByteArray(0); family = ByteArray(0); regime = ByteArray(0)
            atr = FloatArray(0); vdev = FloatArray(0); trend = FloatArray(0); score = FloatArray(0)
            x = ByteArray(0); pnl = FloatArray(0); reason = ByteArray(0); closeE = ShortArray(0)
        }
    }

    private class Cands(
        val n: Int, val nk: Int, val symbol: IntArray, val date: IntArray, val elapsed: ShortArray, val dir: ByteArray,
        val family: ByteArray, val regime: ByteArray, val atr: FloatArray, val vdev: FloatArray, val trend: FloatArray,
        val score: FloatArray, val pnl: FloatArray, val reason: ByteArray, val closeE: ShortArray,
    ) {
        fun pnl(r: Int, k: Int) = pnl[r * nk + k]
        fun closeE(r: Int, k: Int) = closeE[r * nk + k].toInt()
        fun reason(r: Int, k: Int) = reason[r * nk + k].toInt()
        fun filled(r: Int, k: Int) = !pnl(r, k).isNaN()
        fun dirOf(r: Int) = if (dir[r].toInt() == 1) TDirection.SELL_FIRST else TDirection.BUY_FIRST
    }

    data class Eval(val n: Int, val wins: Int, val pnl: Double, val sumRet: Double) {
        val winRate get() = if (n == 0) 0.0 else wins.toDouble() / n
        val avgBp get() = if (n == 0) 0.0 else sumRet / n * 10000
        val winLow: Double get() = wilson(wins, n, -1)
        val winHigh: Double get() = wilson(wins, n, 1)
        fun meets(target: Double, minN: Int) = n >= minN && winRate >= target && pnl > 0
    }

    /** 一条候选策略:规则过滤 + 离场参数组 + 是否用模型(0 不用 / 1 模型A / 2 模型B) */
    data class Gene(val spec: StrategySpec, val k: Int, val model: Int)

    class Result(val model: TradeModel, val report: String)

    /** 离场 × 挂单参数网格(32 组),寻优在其中选择 */
    fun exitGrid(base: TConfig): List<TConfig> {
        val b = base.copy(noOpenBeforeCloseMin = 20, regimeExit = false, tpLimit = true)
        return buildList {
            for (off in OFFSETS) for (tp in TPS) for (sl in SLS) add(b.copy(entryOffsetAtr = off, takeProfitAtr = tp, stopLossAtr = sl))
        }
    }

    private fun gridIndex(off: Int, tp: Int, sl: Int) = (off * TPS.size + tp) * SLS.size + sl

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

        // ---------- 1. 抽样确定分箱 + 内存预算 ----------
        val bt = Backtester(data, cfg)
        val maxMem = Runtime.getRuntime().maxMemory()
        val sampleSyms = symbols.indices.filter { it % maxOf(1, symbols.size / 40) == 0 }.take(40)
        val sampleX = java.util.Collections.synchronizedList(ArrayList<FloatArray>())
        val sampleDays = AtomicLong(); val sampleCands = AtomicLong(); val sampleSetups = AtomicLong()
        val done0 = AtomicInteger()
        progress.start("1/6 抽样分箱", sampleSyms.size)
        runParallel(sampleSyms.map { si ->
            Callable {
                val setups = bt.setups(symbols[si]).toList()
                sampleSetups.addAndGet(setups.size.toLong())
                // 只用每只股票前 40% 的交易日(起始训练期)计算分位点,不看后面的数据
                for (su in setups.take((setups.size * 0.4).toInt())) {
                    val xs = featuresOnly(su, grid[0])
                    sampleDays.incrementAndGet(); sampleCands.addAndGet(xs.size.toLong())
                    for (x in xs) if (sampleX.size < 300_000) sampleX += x
                }
                progress.step(done0.incrementAndGet())
            }
        })
        require(sampleX.size >= 200) { "抽样得到的候选点太少(${sampleX.size}),请检查 1 分钟数据" }
        val edges = Array(nf) { f ->
            val vals = DoubleArray(sampleX.size) { sampleX[it][f].toDouble() }.also { it.sort() }
            val e = ArrayList<Double>()
            for (b in 1 until 64) {
                val v = vals[(b.toLong() * (vals.size - 1) / 64).toInt()]
                if (e.isEmpty() || v > e.last()) e += v
            }
            e.toDoubleArray()
        }
        sampleX.clear()
        // 内存预算:每个候选点约 = 特征 nf 字节 + 每组离场 7 字节 + 属性 34 字节 + 训练期工作区
        val perCand = nf + nk * 7 + 34 + 70 + 12L * minOf(threads, 8)
        val estDays = sampleSetups.get().toDouble() / sampleSyms.size * symbols.size
        val estCands = estDays * sampleCands.get() / maxOf(1L, sampleDays.get())
        val budget = maxMem * 0.7 - 150e6
        val keep = if (estCands * perCand <= budget) 1.0 else (budget / (estCands * perCand)).coerceIn(0.05, 1.0)
        progress.finish("预计候选点约 ${"%,d".format(estCands.toLong())},需内存约 ${gb(estCands * perCand + 150e6)},可用上限 ${gb(maxMem.toDouble())}")
        if (keep < 1.0) progress.log("内存不足以容纳全部数据:随机保留 ${pct(keep)} 的「股票·交易日」(整日保留,不影响当日规则)。" +
            "若想用全部数据,请减少 --sample 或增大可用内存(见 README)。")

        // ---------- 2. 候选生成(特征直接分箱存储)----------
        val blocks = java.util.Collections.synchronizedList(ArrayList<Block>())
        val dayCount = java.util.concurrent.ConcurrentHashMap<Int, Int>()
        val done = AtomicInteger(); val candCount = AtomicLong()
        progress.start("2/6 候选生成", symbols.size)
        runParallel(symbols.indices.map { si ->
            Callable {
                val blk = Block(si, 1024, nf, nk)
                try {
                    for (su in bt.setups(symbols[si])) {
                        dayCount.merge(su.date, 1, Int::plus)
                        if (keep < 1.0 && keepHash(si, su.date) >= keep) continue
                        candidatesOf(su, grid, blk, edges)
                    }
                } catch (e: OutOfMemoryError) {
                    throw e
                } catch (e: Exception) {
                    progress.log("  ${symbols[si]} 失败: $e")
                }
                blocks += blk
                candCount.addAndGet(blk.n.toLong())
                progress.step(done.incrementAndGet(), "候选点 ${"%,d".format(candCount.get())}")
            }
        })
        progress.finish("候选点 ${"%,d".format(candCount.get())}")

        val dates = dayCount.keys.sorted()
        require(dates.size >= 30) { "交易日太少(${dates.size}),至少需要 30 个交易日的 1 分钟数据" }
        val foldStarts = listOf(0.40, 0.55, 0.70, 0.85).map { dates[(dates.size * it).toInt()] }
        val searchStart = foldStarts[0]; val valStart = foldStarts[2]; val holdStart = foldStarts[3]

        val sorted = blocks.sortedBy { it.symbol }
        blocks.clear()
        val n = sorted.sumOf { it.n }
        require(n >= 2000) { "候选点太少($n),请增加股票数量或数据天数" }
        val offs = IntArray(sorted.size + 1).also { for (i in sorted.indices) it[i + 1] = it[i] + sorted[i].n }
        val cols = Array(nf) { ByteArray(n) }
        val c = merge(sorted, offs, n, nk, cols) // 合并时逐块释放,避免内存翻倍
        val bd = BinnedData(edges, cols)
        val foldOf = IntArray(n) { r -> foldStarts.indexOfLast { c.date[r] >= it } }
        progress.log("时间切分:模型训练 ${dates.first()}~<$searchStart | 寻优 $searchStart~<$valStart | 验证 $valStart~<$holdStart | 验收 $holdStart~${dates.last()}")

        // ---------- 3. 胜率模型(滚动前推,每段只用之前的数据)----------
        // 大盘环境是"同一天所有股票相同"的变量,样本只有几十天,放进模型容易记住"哪天好",改由寻优作为过滤条件并经验证期检验
        val modelCols = (0 until nf).filter { Features.specs[it].group != FeatureGroup.MARKET }.toIntArray()
        val modelK = intArrayOf(gridIndex(0, 1, 1), gridIndex(0, 3, 2)) // 模型A:市价/止盈0.12/止损1.0;模型B:市价/止盈0.25/止损1.6
        val prob = Array(2) { DoubleArray(n) { Double.NaN } }
        val deploy = arrayOfNulls<FoldModel>(2)
        val modelImp = Array(2) { DoubleArray(nf) }
        val jobs3 = (0..1).flatMap { m -> (0 until 4).map { f -> m to f } }
        val done3 = AtomicInteger()
        progress.start("3/6 胜率模型", jobs3.size)
        runParallel(jobs3.map { (m, f) ->
            Callable {
                val fm = trainFold(bd, c, dates, foldStarts, modelK[m], f, modelCols, screen = false)
                for ((i, r) in fm.rows.withIndex()) prob[m][r] = fm.pred[i]
                synchronized(modelImp) { for (i in 0 until nf) modelImp[m][i] += fm.model.importance[i] }
                if (f == 3) deploy[m] = fm
                progress.step(done3.incrementAndGet(), "模型${"AB"[m]} 第${f + 1}段")
            }
        })
        progress.finish()
        val searchRows = sortRows(c, (0 until n).filter { foldOf[it] in 0..1 })
        val valRows = sortRows(c, (0 until n).filter { foldOf[it] == 2 })
        val holdRows = sortRows(c, (0 until n).filter { foldOf[it] == 3 })
        val modelAuc = (0..1).map { m -> auc(c, searchRows + valRows, prob[m], modelK[m]) }

        // ---------- 4. 模块消融(仅在寻优期,模型A)----------
        val abl = ArrayList<Triple<FeatureGroup, Double, Double>>() // 组, AUC, ΔAUC
        if (ablation) {
            val groups = FeatureGroup.entries.filter { it != FeatureGroup.MARKET }
            val ablP = groups.associateWith { FloatArray(n) { Float.NaN } }
            val jobs4 = groups.flatMap { g -> (0..1).map { f -> g to f } }
            val done4 = AtomicInteger()
            progress.start("4/6 模块消融", jobs4.size)
            runParallel(jobs4.map { (g, f) ->
                Callable {
                    val cs = modelCols.filter { Features.specs[it].group != g }.toIntArray()
                    val fm = trainFold(bd, c, dates, foldStarts, modelK[0], f, cs, screen = true)
                    val arr = ablP.getValue(g)
                    for ((i, r) in fm.rows.withIndex()) arr[r] = fm.pred[i].toFloat()
                    progress.step(done4.incrementAndGet(), "去掉「${g.label}」第${f + 1}段")
                }
            })
            progress.finish()
            val full = auc(c, searchRows, prob[0], modelK[0])
            for (g in groups) {
                val arr = ablP.getValue(g)
                val a = auc(c, searchRows, DoubleArray(n) { arr[it].toDouble() }, modelK[0])
                abl += Triple(g, a, a - full)
            }
        }

        // ---------- 5. 迭代寻优 ----------
        val rnd = Random(seed)
        val searchCache = java.util.concurrent.ConcurrentHashMap<Gene, Eval>()
        val valCache = HashMap<Gene, Eval>()
        fun evalOn(g: Gene, rows: IntArray) = simulateGene(c, rows, g, prob, grid)
        fun fitness(e: Eval): Double = if (e.n < minTrades) -1.0 + 0.1 * e.n / minTrades
        else e.winLow + (if (e.pnl > 0) 0.0 else -0.3) + 1e-4 * (e.sumRet / e.n) * 10000
        fun valPass(e: Eval) = e.n >= minValTrades && e.winRate >= target && e.pnl > 0
        fun valKey(e: Eval) = e.winLow + (if (e.pnl > 0) 0.0 else -0.5)

        var pop = ArrayList<Gene>()
        // 种子:每组离场参数的"全开"策略,保证从最宽的配置起步
        for (k in 0 until nk) pop += Gene(StrategySpec(), k, 0)
        while (pop.size < population) pop += randomGene(rnd, nk)
        data class RoundLog(val round: Int, val evaluated: Int, val best: Gene, val bestSearch: Eval, val validatedTotal: Int, val bestVal: Eval?)
        val roundLogs = ArrayList<RoundLog>()
        var bestValidated: Pair<Gene, Eval>? = null
        var bestValKey = Double.NEGATIVE_INFINITY
        var sinceImprove = 0
        var validations = 0
        progress.start("5/6 迭代寻优", rounds)
        for (round in 1..rounds) {
            val todo = pop.filter { !searchCache.containsKey(it) }
            runParallel(todo.chunked(maxOf(1, todo.size / (threads * 4))).map { chunk ->
                Callable { for (g in chunk) searchCache[g] = evalOn(g, searchRows) }
            })
            val ranked = pop.distinct().sortedByDescending { fitness(searchCache.getValue(it)) }
            val elites = ranked.take(population / 10)
            // 二次验证:本轮前 10 名中尚未验证过的
            for (g in elites.take(10)) {
                if (g in valCache) continue
                val ve = evalOn(g, valRows)
                valCache[g] = ve
                validations++
                val key = valKey(ve) + if (valPass(ve)) 1.0 else 0.0
                if (key > bestValKey) { bestValKey = key; bestValidated = g to ve; sinceImprove = -1 }
            }
            sinceImprove++
            val passed = valCache.count { valPass(it.value) }
            roundLogs += RoundLog(round, searchCache.size, elites.first(), searchCache.getValue(elites.first()), passed,
                bestValidated?.second)
            val bv = bestValidated?.second
            progress.step(round, "已评估 ${searchCache.size} 条 | 通过验证 $passed 条 | 最佳验证 " +
                (if (bv == null) "-" else "${pct(bv.winRate)}(${bv.n}笔)"))
            if (passed > 0 && sinceImprove >= patience) break
            // 下一代:精英 + 精英变异 + 交叉 + 随机新个体
            val next = LinkedHashSet<Gene>(elites)
            while (next.size < population * 85 / 100) {
                val a = elites[rnd.nextInt(elites.size)]
                next += if (rnd.nextDouble() < 0.25) crossover(a, elites[rnd.nextInt(elites.size)], rnd) else mutate(a, rnd, nk)
            }
            while (next.size < population) next += randomGene(rnd, nk)
            pop = ArrayList(next)
        }
        progress.finish("共评估 ${searchCache.size} 条策略,二次验证 $validations 次")

        // ---------- 6. 最终验收(只做一次)----------
        progress.start("6/6 最终验收", 1)
        val (finalGene, finalVal) = bestValidated ?: run {
            val g = searchCache.entries.maxByOrNull { fitness(it.value) }!!.key
            g to evalOn(g, valRows)
        }
        val finalSearch = searchCache[finalGene] ?: evalOn(finalGene, searchRows)
        val finalHold = evalOn(finalGene, holdRows)
        val holdAll = evalOn(Gene(StrategySpec(), finalGene.k, 0), holdRows)
        progress.step(1)
        progress.finish()

        val ec = grid[finalGene.k].copy(maxTradesPerDay = finalGene.spec.maxTrades)
        val spec = finalGene.spec.copy(useModel = finalGene.model != 0)
        val dm = deploy[if (finalGene.model == 2) 1 else 0]!!
        val trainEnd = dates[dates.indexOf(holdStart) - 1]
        val meta = linkedMapOf(
            "target" to target.toString(),
            "valTrades" to finalVal.n.toString(), "valWinRate" to "%.4f".format(finalVal.winRate),
            "testStart" to holdStart.toString(), "testEnd" to dates.last().toString(),
            "testTrades" to finalHold.n.toString(), "testWinRate" to "%.4f".format(finalHold.winRate),
            "testPnl" to "%.2f".format(finalHold.pnl), "symbols" to symbols.size.toString(),
            "evaluated" to searchCache.size.toString(), "validations" to validations.toString(),
        )
        val model = TradeModel(ec, cfg.costs, if (spec.useModel) spec.modelThr else 0.0, dm.model, dates.first(), trainEnd,
            meta, dm.calib, spec)

        val minutes = (System.nanoTime() - t0) / 6e10
        val top = valCache.entries.sortedByDescending { valKey(it.value) + if (valPass(it.value)) 1 else 0 }.take(10)
        val report = buildString {
            fun gtext(g: Gene) = "${cfgText(grid[g.k])} | ${g.spec.copy(useModel = g.model != 0).describe()}" +
                (if (g.model != 0) "(模型${"AB"[g.model - 1]})" else "")
            appendLine("# realT 训练报告(迭代寻优)")
            appendLine()
            appendLine("- 标的 ${symbols.size} 只,交易日 ${dates.first()} ~ ${dates.last()}(${dates.size} 天),候选点 ${"%,d".format(n)},用时 ${"%.1f".format(minutes)} 分钟($threads 线程)")
            appendLine("- 费率:佣金 ${feeText(cfg.costs.commissionRate)}(最低 ${cfg.costs.minCommission} 元)、印花税 ${feeText(cfg.costs.stampDutyRate)}(仅卖出)、过户费 ${feeText(cfg.costs.transferRate)}、滑点单边 ${feeText(cfg.costs.slippage)};只做当日T")
            appendLine("- 指数分时数据:${if (data.hasIndexMinutes) "有" else "**无**"}")
            appendLine("- 时间切分:模型训练 < $searchStart | **寻优** $searchStart~ | **二次验证** $valStart~ | **最终验收** $holdStart~${dates.last()}(只用一次)")
            appendLine("- 目标:胜率 ≥ ${pct(target)} 且净盈亏 > 0;寻优期至少 $minTrades 笔,验证期至少 $minValTrades 笔")
            appendLine("- 迭代:共 ${roundLogs.size} 轮,评估 ${searchCache.size} 条策略,二次验证 $validations 次")
            appendLine()
            appendLine("## 结论")
            appendLine()
            appendLine("最终策略:**${gtext(finalGene)}**")
            appendLine()
            appendLine("| 数据段 | 做T次数 | 胜率 | 胜率95%置信区间 | 平均每笔(bp) | 净盈亏(元) | 达标 |")
            appendLine("|---|---|---|---|---|---|---|")
            appendLine("| 寻优期(用于挑选) | ${finalSearch.n} | ${pct(finalSearch.winRate)} | ${pct(finalSearch.winLow)}~${pct(finalSearch.winHigh)} | ${f1(finalSearch.avgBp)} | ${f0(finalSearch.pnl)} | ${mark(finalSearch.meets(target, minTrades))} |")
            appendLine("| 二次验证期 | ${finalVal.n} | ${pct(finalVal.winRate)} | ${pct(finalVal.winLow)}~${pct(finalVal.winHigh)} | ${f1(finalVal.avgBp)} | ${f0(finalVal.pnl)} | ${mark(valPass(finalVal))} |")
            appendLine("| **最终验收期(样本外,只用一次)** | ${finalHold.n} | ${pct(finalHold.winRate)} | ${pct(finalHold.winLow)}~${pct(finalHold.winHigh)} | ${f1(finalHold.avgBp)} | ${f0(finalHold.pnl)} | ${mark(finalHold.meets(target, 1))} |")
            appendLine("| 验收期·同离场参数不加过滤 | ${holdAll.n} | ${pct(holdAll.winRate)} | - | ${f1(holdAll.avgBp)} | ${f0(holdAll.pnl)} | - |")
            appendLine()
            val holdDays = dates.count { it >= holdStart }
            appendLine("> 出手频率:验收期约 ${f1(finalHold.n.toDouble() / maxOf(1, holdDays))} 笔/交易日(全部 ${symbols.size} 只合计)。")
            if (!valPass(finalVal)) appendLine("> **迭代未找到通过二次验证的策略**:下表给出验证期表现最好的策略;说明在当前数据与策略空间内达不到目标。")
            else if (!finalHold.meets(target, 1)) appendLine("> **通过了二次验证,但最终验收未达标**:说明验证期的达标有偶然成分,不能投入使用。")
            appendLine()
            appendLine("## 1. 迭代过程")
            appendLine()
            appendLine("| 轮次 | 累计评估 | 本轮寻优最佳(次数/胜率/95%下限) | 累计通过验证 | 最佳验证(次数/胜率) |")
            appendLine("|---|---|---|---|---|")
            for (l in roundLogs) {
                if (l.round > 5 && l.round % 5 != 0 && l.round != roundLogs.last().round) continue
                val bv = l.bestVal
                appendLine("| ${l.round} | ${l.evaluated} | ${l.bestSearch.n} / ${pct(l.bestSearch.winRate)} / ${pct(l.bestSearch.winLow)} | ${l.validatedTotal} | ${if (bv == null) "-" else "${bv.n} / ${pct(bv.winRate)}"} |")
            }
            appendLine()
            appendLine("## 2. 二次验证表现最好的 10 条策略")
            appendLine()
            appendLine("| # | 策略 | 寻优期 次数/胜率 | 验证期 次数/胜率 | 验证95%下限 | 验证净盈亏 | 通过 |")
            appendLine("|---|---|---|---|---|---|---|")
            for ((i, e) in top.withIndex()) {
                val s = searchCache.getValue(e.key); val v = e.value
                appendLine("| ${i + 1} | ${gtext(e.key)} | ${s.n} / ${pct(s.winRate)} | ${v.n} / ${pct(v.winRate)} | ${pct(v.winLow)} | ${f0(v.pnl)} | ${mark(valPass(v))} |")
            }
            appendLine()
            appendLine("## 3. 各维度单独看(寻优期 + 验证期,离场参数 = ${cfgText(grid[finalGene.k])},不加其他过滤)")
            appendLine()
            appendLine("| 维度 | 取值 | 次数 | 胜率 | 平均(bp) | 净盈亏 |")
            appendLine("|---|---|---|---|---|---|")
            val sv = searchRows + valRows
            fun dimRow(dim: String, name: String, spec: StrategySpec) {
                val e = simulateGene(c, sortRows(c, sv.toList()), Gene(spec, finalGene.k, 0), prob, grid)
                appendLine("| $dim | $name | ${e.n} | ${pct(e.winRate)} | ${f1(e.avgBp)} | ${f0(e.pnl)} |")
            }
            listOf("缠论1分钟" to 1, "布林刺破收回" to 2, "RSI拐头" to 4, "缠论5分钟" to 8).forEach { (nm, m) -> dimRow("信号来源", nm, StrategySpec(families = m)) }
            dimRow("方向", "只先卖后买", StrategySpec(dirs = 1)); dimRow("方向", "只先买后卖", StrategySpec(dirs = 2))
            listOf(5 to 30, 31 to 60, 61 to 120, 121 to 180, 181 to 220).forEach { (a, b) ->
                dimRow("时间", "${StrategySpec.clock(a)}~${StrategySpec.clock(b)}", StrategySpec(timeStart = a, timeEnd = b))
            }
            for (lv in MarketLevel.entries) dimRow("大盘", lv.label, StrategySpec(regimeMask = 1 shl lv.ordinal))
            appendLine()
            appendLine("## 4. 胜率模型")
            appendLine()
            appendLine("模型只作为策略的一个可选过滤条件,是否使用由寻优决定。样本外 AUC(0.5 = 随机):模型A ${"%.4f".format(modelAuc[0])},模型B ${"%.4f".format(modelAuc[1])}。")
            appendLine()
            if (abl.isNotEmpty()) {
                appendLine("各层/模块对模型A的贡献(寻优期,去掉该组后 AUC 的变化,负值越大越重要):")
                appendLine()
                appendLine("| 去掉的层/模块 | AUC | ΔAUC |")
                appendLine("|---|---|---|")
                for ((g, a, d) in abl.sortedBy { it.third }) appendLine("| ${g.label} | ${"%.4f".format(a)} | ${"%+.4f".format(d)} |")
                appendLine()
            }
            val imp = modelImp[0]; val tot = imp.sum().takeIf { it > 0 } ?: 1.0
            appendLine("模型A按层/模块的重要性:" + FeatureGroup.entries.filter { it != FeatureGroup.MARKET }
                .associateWith { g -> Features.columnsOf(g).sumOf { imp[it] } / tot }.entries.sortedByDescending { it.value }
                .joinToString(",") { "${it.key.label} ${pct(it.value)}" })
            appendLine()
            appendLine("## 5. 最终策略亏损单(寻优期 + 验证期)")
            appendLine()
            val tk = takenRows(c, sortRows(c, sv.toList()), finalGene, prob, grid)
            val losers = tk.filter { c.pnl(it, finalGene.k) <= 0 }
            appendLine("出手 ${tk.size} 笔中亏损 ${losers.size} 笔:")
            appendLine()
            appendLine("| 方向 | 平仓原因 | 亏损笔数 | 平均亏损(元) |")
            appendLine("|---|---|---|---|")
            for ((key, g) in losers.groupBy { c.dirOf(it).label to CloseReason.entries[c.reason(it, finalGene.k)].label }
                .toSortedMap(compareBy({ it.first }, { it.second }))) {
                appendLine("| ${key.first} | ${key.second} | ${g.size} | ${f0(g.map { c.pnl(it, finalGene.k).toDouble() }.average())} |")
            }
            appendLine()
            appendLine("## 说明")
            appendLine()
            appendLine("- 寻优期的胜率是\"挑出来的\",一定偏乐观;二次验证期用于淘汰偶然达标的策略;最终验收期只用一次,它的结果才代表真实能力。")
            appendLine("- 评估的策略越多,寻优期越容易出现虚高的胜率,所以达标判断以验证期和验收期为准。")
            appendLine("- 参数包:`realt-model.txt`(数据截止 $trainEnd)。样本外回测:`realt backtest --model realt-model.txt --start $holdStart`")
        }
        outDir.mkdirs()
        File(outDir, "realt-model.txt").writeText(model.serialize())
        File(outDir, "train-report.md").writeText(report)
        return Result(model, report)
    }

    // ---------------- 寻优:策略的生成、变异、交叉 ----------------
    private val starts = intArrayOf(5, 15, 30, 45, 60, 90, 120, 150, 180)
    private val ends = intArrayOf(30, 60, 90, 120, 150, 180, 200, 220)
    private val atrMins = doubleArrayOf(0.0, 0.015, 0.02, 0.03, 0.04)
    private val atrMaxs = doubleArrayOf(0.03, 0.04, 0.06, 0.08, 1.0)
    private val vdevs = doubleArrayOf(-9.0, -0.2, 0.0, 0.1, 0.2, 0.3, 0.5)
    private val trends = doubleArrayOf(-9.0, -0.5, -0.2, 0.0, 0.2, 0.4)
    private val thrs = (0..18).map { 0.5 + it * 0.025 }.toDoubleArray()

    private fun randomGene(r: Random, nk: Int): Gene {
        val st = starts[r.nextInt(starts.size)]
        val en = ends.filter { it > st }.let { it[r.nextInt(it.size)] }
        val amin = atrMins[r.nextInt(atrMins.size)]
        val amax = atrMaxs.filter { it > amin }.let { it[r.nextInt(it.size)] }
        val model = r.nextInt(3)
        return Gene(
            StrategySpec(
                dirs = r.nextInt(3), families = 1 + r.nextInt(15), timeStart = st, timeEnd = en,
                regimeMask = if (r.nextDouble() < 0.6) 31 else 1 + r.nextInt(31),
                atrMin = amin, atrMax = amax,
                vwapDevMin = vdevs[r.nextInt(vdevs.size)], withTrendMin = trends[r.nextInt(trends.size)],
                useModel = model != 0, modelThr = thrs[r.nextInt(thrs.size)], maxTrades = 1 + r.nextInt(2),
            ),
            r.nextInt(nk), model,
        )
    }

    private fun mutate(g: Gene, r: Random, nk: Int): Gene {
        var s = g.spec; var k = g.k; var model = g.model
        repeat(1 + r.nextInt(3)) {
            when (r.nextInt(11)) {
                0 -> { // 离场参数:相邻一档或随机
                    val off = k / (TPS.size * SLS.size); val tp = k / SLS.size % TPS.size; val sl = k % SLS.size
                    k = when (r.nextInt(4)) {
                        0 -> gridIndex(1 - off, tp, sl)
                        1 -> gridIndex(off, (tp + if (r.nextBoolean()) 1 else TPS.size - 1) % TPS.size, sl)
                        2 -> gridIndex(off, tp, (sl + if (r.nextBoolean()) 1 else SLS.size - 1) % SLS.size)
                        else -> r.nextInt(nk)
                    }
                }
                1 -> s = s.copy(dirs = r.nextInt(3))
                2 -> s = s.copy(families = (s.families xor (1 shl r.nextInt(4))).let { if (it == 0) s.families else it })
                3 -> { val st = starts[r.nextInt(starts.size)]; s = s.copy(timeStart = st, timeEnd = maxOf(s.timeEnd, ends.first { it > st })) }
                4 -> { val en = ends[r.nextInt(ends.size)]; s = s.copy(timeEnd = en, timeStart = minOf(s.timeStart, starts.last { it < en })) }
                5 -> s = s.copy(regimeMask = (s.regimeMask xor (1 shl r.nextInt(5))).let { if (it == 0) s.regimeMask else it })
                6 -> { val a = atrMins[r.nextInt(atrMins.size)]; s = s.copy(atrMin = a, atrMax = maxOf(s.atrMax, atrMaxs.first { it > a })) }
                7 -> s = s.copy(vwapDevMin = vdevs[r.nextInt(vdevs.size)])
                8 -> s = s.copy(withTrendMin = trends[r.nextInt(trends.size)])
                9 -> { model = r.nextInt(3); s = s.copy(useModel = model != 0) }
                else -> {
                    val i = thrs.indexOfFirst { it >= s.modelThr - 1e-9 }.coerceAtLeast(0)
                    s = s.copy(modelThr = thrs[(i + r.nextInt(5) - 2).coerceIn(0, thrs.size - 1)], maxTrades = if (r.nextInt(4) == 0) 3 - s.maxTrades else s.maxTrades)
                }
            }
        }
        return Gene(s, k, model)
    }

    private fun crossover(a: Gene, b: Gene, r: Random): Gene {
        fun <T> pick(x: T, y: T) = if (r.nextBoolean()) x else y
        val s = StrategySpec(
            pick(a.spec.dirs, b.spec.dirs), pick(a.spec.families, b.spec.families),
            minOf(a.spec.timeStart, b.spec.timeStart).let { pick(it, maxOf(a.spec.timeStart, b.spec.timeStart)) },
            maxOf(a.spec.timeEnd, b.spec.timeEnd), pick(a.spec.regimeMask, b.spec.regimeMask),
            pick(a.spec.atrMin, b.spec.atrMin), maxOf(a.spec.atrMax, b.spec.atrMax),
            pick(a.spec.vwapDevMin, b.spec.vwapDevMin), pick(a.spec.withTrendMin, b.spec.withTrendMin),
            false, pick(a.spec.modelThr, b.spec.modelThr), pick(a.spec.maxTrades, b.spec.maxTrades),
        )
        val model = pick(a.model, b.model)
        val fixed = if (s.timeEnd <= s.timeStart) s.copy(timeEnd = ends.first { it > s.timeStart }) else s
        return Gene(fixed.copy(useModel = model != 0, atrMax = maxOf(fixed.atrMax, atrMaxs.first { it > fixed.atrMin })),
            pick(a.k, b.k), model)
    }

    // ---------------- 模拟出手(与实盘引擎规则一致)----------------
    /** 行按 股票→日期→时间→方向 排序,便于线性扫描 */
    private fun sortRows(c: Cands, rows: List<Int>): IntArray =
        rows.sortedWith(compareBy<Int>({ c.symbol[it] }, { c.date[it] }, { c.elapsed[it] }, { c.dir[it] })).toIntArray()

    private fun passes(c: Cands, r: Int, g: Gene, prob: Array<DoubleArray>): Boolean {
        if (!c.filled(r, g.k)) return false
        if (!g.spec.allows(c.family[r].toInt(), c.dirOf(r), c.elapsed[r].toInt(), c.regime[r].toInt(),
                c.atr[r].toDouble(), c.vdev[r].toDouble(), c.trend[r].toDouble())) return false
        if (g.model != 0) {
            val p = prob[g.model - 1][r]
            if (p.isNaN() || p < g.spec.modelThr) return false
        }
        return true
    }

    /** 同一时刻多个候选:用模型取胜率高者,否则取 L4 综合分高者;同一时间只持一笔、每日上限、冷却 */
    private fun takenRows(c: Cands, rows: IntArray, g: Gene, prob: Array<DoubleArray>, grid: List<TConfig>): IntArray {
        val out = ArrayList<Int>()
        val cool = grid[g.k].cooldownMin
        var i = 0
        var day = -1L; var busy = -1; var lastClose = -1000; var count = 0
        while (i < rows.size) {
            val r0 = rows[i]
            val key = c.symbol[r0].toLong() * 100_000_000L + c.date[r0]
            if (key != day) { day = key; busy = -1; lastClose = -1000; count = 0 }
            val el = c.elapsed[r0].toInt()
            var j = i; var best = -1; var bestP = Double.NEGATIVE_INFINITY
            while (j < rows.size && c.symbol[rows[j]] == c.symbol[r0] && c.date[rows[j]] == c.date[r0] && c.elapsed[rows[j]].toInt() == el) {
                val q = rows[j]
                if (passes(c, q, g, prob)) {
                    val pr = if (g.model != 0) prob[g.model - 1][q] else c.score[q].toDouble()
                    if (pr > bestP) { bestP = pr; best = q }
                }
                j++
            }
            if (best >= 0 && count < g.spec.maxTrades && el > busy && el - lastClose >= cool) {
                out += best; count++
                busy = c.closeE(best, g.k); lastClose = busy
            }
            i = j
        }
        return out.toIntArray()
    }

    private fun simulateGene(c: Cands, rows: IntArray, g: Gene, prob: Array<DoubleArray>, grid: List<TConfig>): Eval {
        var n = 0; var wins = 0; var pnl = 0.0; var sumRet = 0.0
        val notional = cfg.baseValue * grid[g.k].tRatio
        for (r in takenRows(c, rows, g, prob, grid)) {
            val v = c.pnl(r, g.k).toDouble()
            n++; if (v > 0) wins++
            pnl += v; sumRet += v / notional
        }
        return Eval(n, wins, pnl, sumRet)
    }

    // ---------------- 模型训练(单段)----------------
    private class FoldModel(val model: Gbdt, val calib: Calibration, val rows: IntArray, val pred: DoubleArray)

    private fun trainFold(
        bd: BinnedData, c: Cands, dates: List<Int>, foldStarts: List<Int>,
        k: Int, f: Int, cols: IntArray?, screen: Boolean,
    ): FoldModel {
        val start = foldStarts[f]
        val end = if (f + 1 < foldStarts.size) foldStarts[f + 1] else Int.MAX_VALUE
        val trainDates = dates.filter { it < start }
        val esStart = trainDates[(trainDates.size * 0.85).toInt().coerceAtMost(trainDates.size - 1)]
        val y = ByteArray(c.n) { r -> if (c.filled(r, k) && c.pnl(r, k) > 0) 1 else 0 }
        val tr = ArrayList<Int>(); val es = ArrayList<Int>(); val pr = ArrayList<Int>()
        for (r in 0 until c.n) {
            val d = c.date[r]
            if (d >= start && d < end) { pr += r; continue }
            if (!c.filled(r, k)) continue
            if (d < esStart) tr += r else if (d < start) es += r
        }
        val m = Gbdt.trainBinned(bd, y, tr.toIntArray(), es.toIntArray(), cols, gbdtParams(screen).copy(seed = 7 + k * 31 + f))
        val esScores = DoubleArray(es.size) { m.predictBinned(bd, es[it]) }
        val calib = Calibration.fit(esScores, IntArray(es.size) { y[es[it]].toInt() })
        val rows = pr.toIntArray()
        return FoldModel(m, calib, rows, DoubleArray(rows.size) { calib.apply(m.predictBinned(bd, rows[it])) })
    }

    // ---------------- 候选生成与打标签 ----------------
    /** 只算特征(抽样确定分箱用),不打标签 */
    private fun featuresOnly(su: DaySetup, g0: TConfig): List<FloatArray> {
        if ((su.shares * g0.tRatio / 100).toInt() <= 0) return emptyList()
        val observer = su.observer()
        val out = ArrayList<FloatArray>()
        for (b in su.dayBars) {
            val o = observer.observe(b)
            val el = Session.elapsed(b.minute)
            if (el < g0.noOpenAfterOpenMin || Session.MINUTES_PER_DAY - el <= g0.noOpenBeforeCloseMin) continue
            for (dir in Features.candidates(o)) {
                val x = Features.extract(su.ctx, o, dir)
                out += FloatArray(x.size) { x[it].toFloat() }
            }
        }
        return out
    }

    /** 按 (股票, 日期) 的确定性哈希抽样,整日保留或整日跳过 */
    private fun keepHash(si: Int, date: Int): Double {
        var h = si.toLong() * 1_000_003L + date
        h = (h xor (h ushr 33)) * -0x61c8864680b583ebL
        h = h xor (h ushr 29)
        return (h ushr 11).toDouble() / (1L shl 53)
    }

    private fun candidatesOf(su: DaySetup, grid: List<TConfig>, blk: Block, edges: Array<DoubleArray>) {
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
                val o = obs[i]
                blk.date[row] = su.date; blk.elapsed[row] = el.toShort()
                blk.dir[row] = if (dir == TDirection.SELL_FIRST) 1 else 0
                blk.family[row] = StrategySpec.familyOf(o, dir).toByte()
                blk.regime[row] = su.ctx.regime.level.ordinal.toByte()
                blk.atr[row] = su.ctx.atrPct.toFloat()
                blk.vdev[row] = StrategySpec.vwapDevS(o, dir).toFloat()
                blk.trend[row] = StrategySpec.withTrend(o, dir).toFloat()
                blk.score[row] = StrategySpec.score(o, dir).toFloat()
                val x = Features.extract(su.ctx, o, dir)
                // 用与实盘完全相同的 double 值分箱(实盘按 x <= 分界值 走树)
                for (f in 0 until nf) blk.x[row * nf + f] = BinnedData.binOf(edges[f], x[f]).toByte()
                for ((k, g) in grid.withIndex()) {
                    val e = TEngine(g, cfg.costs, su.ctx, su.shares, null)
                    val broker = SimBroker(e, cfg.costs.slippage)
                    val first = e.forceOpen(dir, bars[i], o)
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

    /** 合并各股票的候选点;每合并完一块立即释放它,峰值内存不翻倍 */
    private fun merge(sorted: List<Block>, offs: IntArray, n: Int, nk: Int, cols: Array<ByteArray>): Cands {
        val nf = cols.size
        val sym = IntArray(n); val date = IntArray(n); val el = ShortArray(n); val dir = ByteArray(n); val fam = ByteArray(n)
        val reg = ByteArray(n); val atr = FloatArray(n); val vdev = FloatArray(n); val trend = FloatArray(n); val score = FloatArray(n)
        val pnl = FloatArray(n * nk); val rsn = ByteArray(n * nk); val ce = ShortArray(n * nk)
        for ((bi, b) in sorted.withIndex()) {
            val o = offs[bi]
            java.util.Arrays.fill(sym, o, o + b.n, b.symbol)
            System.arraycopy(b.date, 0, date, o, b.n); System.arraycopy(b.elapsed, 0, el, o, b.n)
            System.arraycopy(b.dir, 0, dir, o, b.n); System.arraycopy(b.family, 0, fam, o, b.n)
            System.arraycopy(b.regime, 0, reg, o, b.n); System.arraycopy(b.atr, 0, atr, o, b.n)
            System.arraycopy(b.vdev, 0, vdev, o, b.n); System.arraycopy(b.trend, 0, trend, o, b.n)
            System.arraycopy(b.score, 0, score, o, b.n)
            System.arraycopy(b.pnl, 0, pnl, o * nk, b.n * nk); System.arraycopy(b.reason, 0, rsn, o * nk, b.n * nk)
            System.arraycopy(b.closeE, 0, ce, o * nk, b.n * nk)
            for (i in 0 until b.n) for (f in 0 until nf) cols[f][o + i] = b.x[i * nf + f]
            b.release()
        }
        return Cands(n, nk, sym, date, el, dir, fam, reg, atr, vdev, trend, score, pnl, rsn, ce)
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
        if (tasks.isEmpty()) return
        val pool = Executors.newFixedThreadPool(threads)
        try {
            pool.invokeAll(tasks).forEach { it.get() }
        } finally {
            pool.shutdown()
        }
    }

    private fun cfgText(g: TConfig) = "${if (g.entryOffsetAtr > 0) "挂单${g.entryOffsetAtr}ATR" else "市价"} 止盈${g.takeProfitAtr} 止损${g.stopLossAtr}"
    private fun pct(x: Double) = "%.1f%%".format(x * 100)
    private fun gb(bytes: Double) = "%.1fGB".format(bytes / 1e9)
    private fun f0(x: Double) = "%.0f".format(x)
    private fun f1(x: Double) = "%.1f".format(x)
    private fun mark(b: Boolean) = if (b) "✔" else "✘"

    companion object {
        val OFFSETS = doubleArrayOf(0.0, 0.1)
        val TPS = doubleArrayOf(0.08, 0.12, 0.18, 0.25)
        val SLS = doubleArrayOf(0.6, 1.0, 1.6, 2.5)

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

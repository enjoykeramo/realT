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
    private val target: Double = 0.90,
    /** 每笔利润目标:扣除全部费用后平均每笔 ≥ minBp 个基点(1bp = 0.01%) */
    private val minBp: Double = 10.0,
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
        /** 实际成交金额(T 仓股数 × 开仓价),用于按成交金额计算每笔收益率(与回测报告一致) */
        var value = FloatArray(cap)
        /** 特征分箱后的箱号(每值 1 字节) */
        var x = ByteArray(cap * nf); var pnl = FloatArray(cap * nk); var reason = ByteArray(cap * nk); var closeE = ShortArray(cap * nk)
        fun ensure() {
            if (n < date.size) return
            val c = date.size * 2
            date = date.copyOf(c); elapsed = elapsed.copyOf(c); dir = dir.copyOf(c); family = family.copyOf(c)
            regime = regime.copyOf(c); atr = atr.copyOf(c); vdev = vdev.copyOf(c); trend = trend.copyOf(c); score = score.copyOf(c)
            value = value.copyOf(c)
            x = x.copyOf(c * nf); pnl = pnl.copyOf(c * nk); reason = reason.copyOf(c * nk); closeE = closeE.copyOf(c * nk)
        }
        fun release() {
            date = IntArray(0); elapsed = ShortArray(0); dir = ByteArray(0); family = ByteArray(0); regime = ByteArray(0)
            atr = FloatArray(0); vdev = FloatArray(0); trend = FloatArray(0); score = FloatArray(0); value = FloatArray(0)
            x = ByteArray(0); pnl = FloatArray(0); reason = ByteArray(0); closeE = ShortArray(0)
        }
    }

    /** 全局列存:各股票算完后追加进来(同一股票的行连续、按时间有序),块随即释放 */
    private class Store(val nf: Int, val nk: Int, cap0: Int) {
        var n = 0; private set
        private var cap = cap0
        var symbol = IntArray(cap); var date = IntArray(cap); var elapsed = ShortArray(cap); var dir = ByteArray(cap)
        var family = ByteArray(cap); var regime = ByteArray(cap); var atr = FloatArray(cap); var vdev = FloatArray(cap)
        var trend = FloatArray(cap); var score = FloatArray(cap); var value = FloatArray(cap)
        var pnl = FloatArray(cap * nk); var reason = ByteArray(cap * nk); var closeE = ShortArray(cap * nk)
        var cols = Array(nf) { ByteArray(cap) }

        /** 追加一块;[keepRow] 不为空时只追加满足条件的行(按日再抽样) */
        @Synchronized
        fun append(b: Block, keepRow: ((Int) -> Boolean)? = null) {
            if (keepRow != null) { appendRows(b, keepRow); return }
            if (n + b.n > cap) grow(n + b.n)
            val o = n
            java.util.Arrays.fill(symbol, o, o + b.n, b.symbol)
            System.arraycopy(b.date, 0, date, o, b.n); System.arraycopy(b.elapsed, 0, elapsed, o, b.n)
            System.arraycopy(b.dir, 0, dir, o, b.n); System.arraycopy(b.family, 0, family, o, b.n)
            System.arraycopy(b.regime, 0, regime, o, b.n); System.arraycopy(b.atr, 0, atr, o, b.n)
            System.arraycopy(b.vdev, 0, vdev, o, b.n); System.arraycopy(b.trend, 0, trend, o, b.n)
            System.arraycopy(b.score, 0, score, o, b.n); System.arraycopy(b.value, 0, value, o, b.n)
            System.arraycopy(b.pnl, 0, pnl, o * nk, b.n * nk); System.arraycopy(b.reason, 0, reason, o * nk, b.n * nk)
            System.arraycopy(b.closeE, 0, closeE, o * nk, b.n * nk)
            for (f in 0 until nf) { val col = cols[f]; for (i in 0 until b.n) col[o + i] = b.x[i * nf + f] }
            n += b.n
            b.release()
        }

        private fun appendRows(b: Block, keepRow: (Int) -> Boolean) {
            var m = 0
            for (i in 0 until b.n) if (keepRow(i)) m++
            if (n + m > cap) grow(n + m)
            for (i in 0 until b.n) {
                if (!keepRow(i)) continue
                val o = n
                symbol[o] = b.symbol; date[o] = b.date[i]; elapsed[o] = b.elapsed[i]; dir[o] = b.dir[i]
                family[o] = b.family[i]; regime[o] = b.regime[i]; atr[o] = b.atr[i]; vdev[o] = b.vdev[i]
                trend[o] = b.trend[i]; score[o] = b.score[i]; value[o] = b.value[i]
                System.arraycopy(b.pnl, i * nk, pnl, o * nk, nk); System.arraycopy(b.reason, i * nk, reason, o * nk, nk)
                System.arraycopy(b.closeE, i * nk, closeE, o * nk, nk)
                for (f in 0 until nf) cols[f][o] = b.x[i * nf + f]
                n++
            }
            b.release()
        }

        /** 容量按精确行数一次分配,正常不会扩容 */
        private fun grow(need: Int) {
            val c = maxOf(need, (cap * 1.25).toInt())
            symbol = symbol.copyOf(c); date = date.copyOf(c); elapsed = elapsed.copyOf(c); dir = dir.copyOf(c)
            family = family.copyOf(c); regime = regime.copyOf(c); atr = atr.copyOf(c); vdev = vdev.copyOf(c)
            trend = trend.copyOf(c); score = score.copyOf(c); value = value.copyOf(c)
            pnl = pnl.copyOf(c * nk); reason = reason.copyOf(c * nk); closeE = closeE.copyOf(c * nk)
            for (f in 0 until nf) cols[f] = cols[f].copyOf(c)
            cap = c
        }

        fun cands() = Cands(n, nk, symbol, date, elapsed, dir, family, regime, atr, vdev, trend, score, value, pnl, reason, closeE)
    }

    /**
     * 候选点临时文件:候选生成时每只股票算完立即写盘,结束后得到精确行数,再一次性按精确容量载入。
     * 这样内存里同时只有"每线程一只股票"的数据,不依赖抽样估算,也不会因扩容瞬间占用双倍内存。
     */
    private class Spill(val file: File, val nf: Int, val nk: Int) {
        private val ch = java.nio.channels.FileChannel.open(file.toPath(),
            java.nio.file.StandardOpenOption.CREATE, java.nio.file.StandardOpenOption.TRUNCATE_EXISTING,
            java.nio.file.StandardOpenOption.READ, java.nio.file.StandardOpenOption.WRITE)
        var rows = 0L; private set

        private fun bytes(n: Int) = n * (4 + 2 + 3 + 20 + nf + nk * 7)

        @Synchronized
        fun write(b: Block) {
            if (b.n == 0) return
            val buf = java.nio.ByteBuffer.allocate(8 + bytes(b.n))
            buf.putInt(b.symbol).putInt(b.n)
            val n = b.n
            buf.asIntBuffer().put(b.date, 0, n); buf.position(buf.position() + n * 4)
            buf.asShortBuffer().put(b.elapsed, 0, n); buf.position(buf.position() + n * 2)
            buf.put(b.dir, 0, n).put(b.family, 0, n).put(b.regime, 0, n)
            for (a in arrayOf(b.atr, b.vdev, b.trend, b.score, b.value)) { buf.asFloatBuffer().put(a, 0, n); buf.position(buf.position() + n * 4) }
            buf.put(b.x, 0, n * nf)
            buf.asFloatBuffer().put(b.pnl, 0, n * nk); buf.position(buf.position() + n * nk * 4)
            buf.put(b.reason, 0, n * nk)
            buf.asShortBuffer().put(b.closeE, 0, n * nk); buf.position(buf.position() + n * nk * 2)
            buf.flip()
            while (buf.hasRemaining()) ch.write(buf)
            rows += n
        }

        val sizeBytes get() = ch.size()

        /** 顺序读回每一块 */
        fun readAll(each: (Block) -> Unit) {
            ch.position(0)
            val hdr = java.nio.ByteBuffer.allocate(8)
            while (true) {
                hdr.clear()
                if (!readFully(hdr)) break
                hdr.flip()
                val sym = hdr.getInt(); val n = hdr.getInt()
                val buf = java.nio.ByteBuffer.allocate(bytes(n))
                check(readFully(buf)) { "候选点临时文件损坏: $file" }
                buf.flip()
                val b = Block(sym, n, nf, nk)
                b.n = n
                buf.asIntBuffer().get(b.date, 0, n); buf.position(buf.position() + n * 4)
                buf.asShortBuffer().get(b.elapsed, 0, n); buf.position(buf.position() + n * 2)
                buf.get(b.dir, 0, n).get(b.family, 0, n).get(b.regime, 0, n)
                for (a in arrayOf(b.atr, b.vdev, b.trend, b.score, b.value)) { buf.asFloatBuffer().get(a, 0, n); buf.position(buf.position() + n * 4) }
                buf.get(b.x, 0, n * nf)
                buf.asFloatBuffer().get(b.pnl, 0, n * nk); buf.position(buf.position() + n * nk * 4)
                buf.get(b.reason, 0, n * nk)
                buf.asShortBuffer().get(b.closeE, 0, n * nk)
                each(b)
            }
        }

        private fun readFully(buf: java.nio.ByteBuffer): Boolean {
            while (buf.hasRemaining()) if (ch.read(buf) < 0) return false
            return true
        }

        fun delete() {
            runCatching { ch.close() }
            file.delete()
        }
    }

    private class Cands(
        val n: Int, val nk: Int, val symbol: IntArray, val date: IntArray, val elapsed: ShortArray, val dir: ByteArray,
        val family: ByteArray, val regime: ByteArray, val atr: FloatArray, val vdev: FloatArray, val trend: FloatArray,
        val score: FloatArray, val value: FloatArray, val pnl: FloatArray, val reason: ByteArray, val closeE: ShortArray,
    ) {
        fun pnl(r: Int, k: Int) = pnl[r * nk + k]
        fun closeE(r: Int, k: Int) = closeE[r * nk + k].toInt()
        fun reason(r: Int, k: Int) = reason[r * nk + k].toInt()
        fun filled(r: Int, k: Int) = !pnl(r, k).isNaN()
        fun dirOf(r: Int) = if (dir[r].toInt() == 1) TDirection.SELL_FIRST else TDirection.BUY_FIRST
    }

    /**
     * 一条策略在一段数据上的表现。
     * [days] 有出手的交易日数;[ssr] 按交易日聚类的残差平方和 Σ(赢_d − p·次数_d)²,用于"按日聚类"的置信区间:
     * 同一天的多笔交易受同一个大盘影响、彼此相关,不能当作独立样本,只按笔数算的置信区间会偏窄。
     */
    data class Eval(
        val n: Int, val wins: Int, val pnl: Double, val sumRet: Double,
        val winPnl: Double = 0.0, val days: Int = 0, val ssr: Double = 0.0,
        val sumRet2: Double = 0.0, val ssrRet: Double = 0.0,
    ) {
        val winRate get() = if (n == 0) 0.0 else wins.toDouble() / n
        val avgBp get() = if (n == 0) 0.0 else sumRet / n * 10000
        private val clusterSe: Double get() =
            if (days < 2 || n == 0) Double.NaN else sqrt(days.toDouble() / (days - 1) * ssr) / n
        /** 胜率 95% 下限:取「按笔 Wilson」与「按日聚类」两者中更保守的 */
        val winLow: Double get() {
            if (n == 0) return 0.0
            val w = wilson(wins, n, -1)
            if (days < 2) return minOf(w, 0.5 * w)
            return minOf(w, (winRate - 1.96 * clusterSe).coerceIn(0.0, 1.0))
        }
        val winHigh: Double get() {
            if (n == 0) return 0.0
            val w = wilson(wins, n, 1)
            return if (days < 2) w else maxOf(w, (winRate + 1.96 * clusterSe).coerceIn(0.0, 1.0))
        }
        /**
         * 平均每笔利润(bp)的 95% 下限:取「按笔」与「按交易日聚类」两种标准误中较大者;只在 1 天出手时视为毫无把握。
         */
        val retLowBp: Double get() {
            if (n == 0) return -1e4
            val m = sumRet / n
            var se = if (n > 1) sqrt(maxOf(0.0, (sumRet2 - n * m * m) / (n - 1)) / n) else kotlin.math.abs(m)
            se = if (days >= 2) maxOf(se, sqrt(days.toDouble() / (days - 1) * ssrRet) / n) else maxOf(se, kotlin.math.abs(m))
            return (m - 1.96 * se) * 10000
        }

        /** 盈亏平衡胜率 = 平均亏损 /(平均盈利 + 平均亏损);没有亏损或没有盈利时为 NaN */
        val breakEven: Double get() {
            val losses = n - wins
            if (wins == 0 || losses == 0) return Double.NaN
            val aw = winPnl / wins; val al = (winPnl - pnl) / losses
            return if (aw + al <= 0) Double.NaN else al / (aw + al)
        }
        fun meets(target: Double, minBp: Double, minN: Int) = n >= minN && winRate >= target && avgBp >= minBp && pnl > 0
    }

    /** 模型分数:prob 校准胜率(同一时刻多个候选时取高者);pctl 在该段模型自己近期分数中的分位(用于阈值过滤) */
    private class Scores(val prob: Array<DoubleArray>, val pctl: Array<FloatArray>)

    /** 一条候选策略:规则过滤 + 离场参数组 + 是否用模型(0 不用 / 1 模型A / 2 模型B) */
    data class Gene(val spec: StrategySpec, val k: Int, val model: Int)

    class Result(val model: TradeModel, val report: String)

    /** 离场参数网格(32 组 = 离场方式 2 × 止盈 4 × 止损 4),寻优在其中选择 */
    fun exitGrid(base: TConfig): List<TConfig> {
        val b = base.copy(noOpenBeforeCloseMin = 20, regimeExit = false, tpLimit = true)
        return buildList {
            for (hold in HOLDS) for (tp in TPS) for (sl in SLS) add(b.copy(timeStopMin = hold, takeProfitAtr = tp, stopLossAtr = sl))
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
        progress.log("训练线程 $threads,标的 ${symbols.size} 只,特征 $nf 个,离场参数 $nk 组,目标:胜率 ≥ ${pct(target)} 且每笔 ≥ ${f1(minBp)}bp")

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
        // 内存预算(每个候选点):特征 nf 字节 + 每组离场 7 字节 + 属性约 30 字节
        //   + 训练期工作区(分段/两个模型概率/消融预测/标签/行号)约 100 字节;另留 30% 给并行训练的 GBDT 与 GC
        val perCand = nf + nk * 7 + 34 + 100L
        val estDays = sampleSetups.get().toDouble() / sampleSyms.size * symbols.size
        val estCands = estDays * sampleCands.get() / maxOf(1L, sampleDays.get())
        val budget = maxMem * 0.7 - 150e6
        // 预抽样只用于明显装不下时少算一些(留 30% 余量),最终保留比例按写盘后的精确行数确定
        val keep0 = if (estCands * perCand <= budget * 1.3) 1.0 else (budget * 1.3 / (estCands * perCand)).coerceIn(0.05, 1.0)
        progress.finish("预计候选点约 ${"%,d".format(estCands.toLong())},需内存约 ${gb(estCands * perCand + 150e6)},可用上限 ${gb(maxMem.toDouble())}")

        // ---------- 2. 候选生成(特征直接分箱;每只股票算完立即写入临时文件,内存只保留各线程当前那一只)----------
        outDir.mkdirs()
        val spill = Spill(File(outDir, "candidates.tmp"), nf, nk)
        val dayCount = java.util.concurrent.ConcurrentHashMap<Int, Int>()
        val dayRows = java.util.concurrent.ConcurrentHashMap<Long, Int>() // (股票, 日期) → 候选点数
        val done = AtomicInteger()
        val failed = AtomicInteger()
        try {
            progress.start("2/6 候选生成", symbols.size)
            runParallel(symbols.indices.map { si ->
                Callable {
                    val blk = Block(si, 1024, nf, nk)
                    try {
                        for (su in bt.setups(symbols[si])) {
                            dayCount.merge(su.date, 1, Int::plus)
                            if (keep0 < 1.0 && keepHash(si, su.date) >= keep0) continue
                            val before = blk.n
                            candidatesOf(su, grid, blk, edges)
                            if (blk.n > before) dayRows[si.toLong() shl 32 or su.date.toLong()] = blk.n - before
                        }
                    } catch (e: OutOfMemoryError) {
                        throw e
                    } catch (e: Exception) {
                        failed.incrementAndGet()
                        progress.log("  ${symbols[si]} 跳过: $e")
                    }
                    spill.write(blk)
                    blk.release()
                    progress.step(done.incrementAndGet(), "候选点 ${"%,d".format(spill.rows)} | 内存 ${memText()}")
                }
            })
            progress.finish("候选点 ${"%,d".format(spill.rows)} | 临时文件 ${gb(spill.sizeBytes.toDouble())} | 内存 ${memText()}")
            if (failed.get() > 0) progress.log("  共 ${failed.get()} 只股票数据异常已跳过")
        } catch (e: Throwable) {
            spill.delete()
            throw e
        }

        // 按精确行数确定最终保留比例(整日保留),一次性分配全局存储并载入
        val exact = spill.rows
        val keep = if (exact * perCand <= budget) keep0 else (keep0 * budget / (exact * perCand)).coerceIn(0.01, keep0)
        if (keep < 1.0) progress.log("内存不足以容纳全部数据:随机保留 ${pct(keep)} 的「股票·交易日」(整日保留,不影响当日规则)。" +
            "若想用全部数据,请减少 --sample 或增大可用内存(见 README)。")
        fun keepDay(si: Int, date: Int) = keep >= keep0 || keepHash(si, date) < keep
        var cap = 0L
        for ((key, cnt) in dayRows) if (keepDay((key ushr 32).toInt(), key.toInt())) cap += cnt
        require(cap * maxOf(nf, nk) < Int.MAX_VALUE) { "候选点过多(${"%,d".format(cap)}),请减少 --sample" }
        val store: Store
        try {
            System.gc()
            store = Store(nf, nk, maxOf(1L, cap).toInt())
            spill.readAll { b -> store.append(b, if (keep >= keep0) null else { r -> keepHash(b.symbol, b.date[r]) < keep }) }
        } finally {
            spill.delete()
        }
        progress.log("载入候选点 ${"%,d".format(store.n)} | 内存 ${memText()}")

        val dates = dayCount.keys.sorted()
        require(dates.size >= 30) { "交易日太少(${dates.size}),至少需要 30 个交易日的 1 分钟数据" }
        val foldStarts = listOf(0.40, 0.55, 0.70, 0.85).map { dates[(dates.size * it).toInt()] }
        val searchStart = foldStarts[0]; val valStart = foldStarts[2]; val holdStart = foldStarts[3]
        val n = store.n
        require(n >= 2000) { "候选点太少($n),请增加股票数量或数据天数" }
        val c = store.cands()
        val bd = BinnedData(edges, store.cols)
        val foldOf = IntArray(n) { r -> foldStarts.indexOfLast { c.date[r] >= it } }
        progress.log("时间切分:模型训练 ${dates.first()}~<$searchStart | 寻优 $searchStart~<$valStart | 验证 $valStart~<$holdStart | 验收 $holdStart~${dates.last()}")

        // 大盘环境是"同一天所有股票相同"的变量,样本只有几十天,放进模型容易记住"哪天好",改由寻优作为过滤条件并经验证期检验
        val baseCols = (0 until nf).filter { Features.specs[it].group != FeatureGroup.MARKET }.toIntArray()
        val modelK = intArrayOf(gridIndex(0, 1, 1), gridIndex(0, 3, 2)) // 模型A:市价/止盈0.12/止损1.0;模型B:市价/止盈0.25/止损1.6
        val searchRows = rowsWhere(n) { foldOf[it] in 0..1 }
        val valRows = rowsWhere(n) { foldOf[it] == 2 }
        val holdRows = rowsWhere(n) { foldOf[it] == 3 }

        // ---------- 3. 模块消融 → 选定模块组合(仅用寻优期,模型A)----------
        // 每次去掉一组重新训练;去掉后 AUC 反而升高(超过噪声 ABL_DROP)的组判为"拖后腿",正式模型不再使用
        data class Abl(val group: FeatureGroup, val auc: Double, val delta: Double, val dropped: Boolean)
        val abl = ArrayList<Abl>()
        var baseAuc = Double.NaN
        val droppedGroups = ArrayList<FeatureGroup>()
        if (ablation) {
            val groups = FeatureGroup.entries.filter { it != FeatureGroup.MARKET && Features.columnsOf(it).isNotEmpty() }
            val variants: List<FeatureGroup?> = listOf<FeatureGroup?>(null) + groups
            val ablP = variants.associateWith { FloatArray(n) { Float.NaN } }
            val jobs = variants.flatMap { g -> (0..1).map { f -> g to f } }
            val doneA = AtomicInteger()
            progress.start("3/6 模块消融", jobs.size)
            runParallel(parallelism = modelParallelism(n), tasks = jobs.map { (g, f) ->
                Callable {
                    val cs = if (g == null) baseCols else baseCols.filter { Features.specs[it].group != g }.toIntArray()
                    val fm = trainFold(bd, c, dates, foldStarts, modelK[0], f, cs, screen = true)
                    val arr = ablP.getValue(g)
                    for ((i, r) in fm.rows.withIndex()) arr[r] = fm.pred[i].toFloat()
                    progress.step(doneA.incrementAndGet(), if (g == null) "全部模块 第${f + 1}段" else "去掉「${g.label}」第${f + 1}段")
                }
            })
            progress.finish()
            val baseArr = ablP.getValue(null)
            baseAuc = auc(c, searchRows, modelK[0]) { baseArr[it].toDouble() }
            for (g in groups) {
                val arr = ablP.getValue(g)
                val a = auc(c, searchRows, modelK[0]) { arr[it].toDouble() }
                abl += Abl(g, a, a - baseAuc, a - baseAuc > ABL_DROP)
            }
            // 最多剔除一半的组,且至少保留 3 组
            for (x in abl.filter { it.dropped }.sortedByDescending { it.delta }.take(maxOf(0, minOf(groups.size / 2, groups.size - 3))))
                droppedGroups += x.group
            if (droppedGroups.isNotEmpty()) progress.log("模块选择:剔除 ${droppedGroups.joinToString("、") { it.label }}(去掉后寻优期 AUC 反而升高)")
        }
        val modelCols = baseCols.filter { Features.specs[it].group !in droppedGroups }.toIntArray()

        // ---------- 4. 胜率模型(滚动前推,每段只用之前的数据;用选定的模块组合)----------
        val prob = Array(2) { DoubleArray(n) { Double.NaN } }
        val pctl = Array(2) { FloatArray(n) { Float.NaN } }
        val sc = Scores(prob, pctl)
        val deploy = arrayOfNulls<FoldModel>(2)
        val modelImp = Array(2) { DoubleArray(nf) }
        val jobs3 = (0..1).flatMap { m -> (0 until 4).map { f -> m to f } }
        val done3 = AtomicInteger()
        progress.start("4/6 胜率模型", jobs3.size)
        runParallel(parallelism = modelParallelism(n), tasks = jobs3.map { (m, f) ->
            Callable {
                val fm = trainFold(bd, c, dates, foldStarts, modelK[m], f, modelCols, screen = false)
                for ((i, r) in fm.rows.withIndex()) { prob[m][r] = fm.pred[i]; pctl[m][r] = fm.pctl(fm.pred[i]) }
                synchronized(modelImp) { for (i in 0 until nf) modelImp[m][i] += fm.model.importance[i] }
                if (f == 3) deploy[m] = fm
                progress.step(done3.incrementAndGet(), "模型${"AB"[m]} 第${f + 1}段")
            }
        })
        progress.finish()
        val modelAuc = (0..1).map { m -> auc(c, searchRows + valRows, prob[m], modelK[m]) }

        // ---------- 5. 迭代寻优 ----------
        val rnd = Random(seed)
        val searchCache = java.util.concurrent.ConcurrentHashMap<Gene, Eval>()
        val valCache = HashMap<Gene, Eval>()
        fun evalOn(g: Gene, rows: IntArray) = simulateGene(c, rows, g, sc, grid)
        // 出手需分散在足够多的交易日上:只在少数几天出手的"高胜率"多半是碰上了好日子
        val searchDays = dates.count { it >= searchStart && it < valStart }
        val valDays = dates.count { it >= valStart && it < holdStart }
        val minSearchDays = maxOf(3, Math.ceil(searchDays * 0.3).toInt())
        val minValDays = maxOf(3, Math.ceil(valDays * 0.4).toInt())
        // 双目标:胜率 ≥ target 且 每笔利润 ≥ minBp。打分 = 两个目标"把握程度"中较差的那个:
        //   胜率 95% 下限超出目标多少(每 5 个百分点记 1 分)、每笔利润 95% 下限超出目标多少(每 10bp 记 1 分)
        // 只要有一个目标没把握,分数就上不去;两个都有把握时,余量越大越好
        fun margin(e: Eval) = minOf((e.winLow - target) / 0.05, (e.retLowBp - minBp) / 10.0)
        fun fitness(e: Eval): Double = if (e.n < minTrades || e.days < minSearchDays)
            -1000.0 + 10 * (minOf(1.0, e.n.toDouble() / minTrades) + minOf(1.0, e.days.toDouble() / minSearchDays))
        else margin(e)
        fun valPass(e: Eval) = e.n >= minValTrades && e.days >= minValDays && e.winRate >= target && e.avgBp >= minBp && e.pnl > 0
        fun valKey(e: Eval) = if (e.n == 0) -1e4 else margin(e)

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
                val key = valKey(ve) + if (valPass(ve)) 1000.0 else 0.0
                if (key > bestValKey) { bestValKey = key; bestValidated = g to ve; sinceImprove = -1 }
            }
            sinceImprove++
            val passed = valCache.count { valPass(it.value) }
            roundLogs += RoundLog(round, searchCache.size, elites.first(), searchCache.getValue(elites.first()), passed,
                bestValidated?.second)
            val bv = bestValidated?.second
            progress.step(round, "已评估 ${searchCache.size} 条 | 通过验证 $passed 条 | 最佳验证 " +
                (if (bv == null) "-" else "${pct(bv.winRate)}/${f1(bv.avgBp)}bp(${bv.n}笔)"))
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
        val dm = deploy[if (finalGene.model == 2) 1 else 0]!!
        // 寻优用的是分位阈值;部署时换算成该模型的校准胜率阈值(两者对同一模型完全等价)
        val spec = finalGene.spec.copy(useModel = finalGene.model != 0,
            modelThr = if (finalGene.model != 0) dm.probThreshold(finalGene.spec.modelThr) else finalGene.spec.modelThr)
        val trainEnd = dates[dates.indexOf(holdStart) - 1]
        val meta = linkedMapOf(
            "target" to target.toString(), "minBp" to minBp.toString(),
            "valAvgBp" to "%.2f".format(finalVal.avgBp), "testAvgBp" to "%.2f".format(finalHold.avgBp),
            "valTrades" to finalVal.n.toString(), "valWinRate" to "%.4f".format(finalVal.winRate),
            "testStart" to holdStart.toString(), "testEnd" to dates.last().toString(),
            "testTrades" to finalHold.n.toString(), "testWinRate" to "%.4f".format(finalHold.winRate),
            "testPnl" to "%.2f".format(finalHold.pnl), "symbols" to symbols.size.toString(),
            "evaluated" to searchCache.size.toString(), "validations" to validations.toString(),
            "modelPctl" to finalGene.spec.modelThr.toString(), "droppedGroups" to droppedGroups.joinToString("/") { it.name },
        )
        val model = TradeModel(ec, cfg.costs, if (spec.useModel) spec.modelThr else 0.0, dm.model, dates.first(), trainEnd,
            meta, dm.calib, spec)

        val minutes = (System.nanoTime() - t0) / 6e10
        val top = valCache.entries.sortedByDescending { valKey(it.value) + if (valPass(it.value)) 1000.0 else 0.0 }.take(10)
        val report = buildString {
            fun gtext(g: Gene) = "${cfgText(grid[g.k])} | " +
                g.spec.copy(useModel = g.model != 0).describe("模型${"AB"[maxOf(0, g.model - 1)]}分数前${f0((1 - g.spec.modelThr) * 100)}%")
            appendLine("# realT 训练报告(迭代寻优)")
            appendLine()
            appendLine("- 标的 ${symbols.size} 只,交易日 ${dates.first()} ~ ${dates.last()}(${dates.size} 天),候选点 ${"%,d".format(n)},用时 ${"%.1f".format(minutes)} 分钟($threads 线程)")
            appendLine("- 费率:佣金 ${feeText(cfg.costs.commissionRate)}(最低 ${cfg.costs.minCommission} 元)、印花税 ${feeText(cfg.costs.stampDutyRate)}(仅卖出)、过户费 ${feeText(cfg.costs.transferRate)}、滑点单边 ${feeText(cfg.costs.slippage)};只做当日T")
            if (keep < 1.0) appendLine("- 内存限制:按「股票·交易日」随机保留 ${pct(keep)} 的数据参与训练")
            appendLine("- 指数分时数据:${if (data.hasIndexMinutes) "有" else "**无**"}")
            appendLine("- 时间切分:模型训练 < $searchStart | **寻优** $searchStart~ | **二次验证** $valStart~ | **最终验收** $holdStart~${dates.last()}(只用一次)")
            appendLine("- **目标(双目标)**:胜率 ≥ ${pct(target)} **且** 扣费后平均每笔 ≥ ${f1(minBp)}bp(${"%.2f%%".format(minBp / 100)})")
            appendLine("- 样本要求:寻优期至少 $minTrades 笔且分布在 $minSearchDays 个交易日以上," +
                "验证期至少 $minValTrades 笔且分布在 $minValDays 个交易日以上")
            appendLine("- 交易日:寻优期 $searchDays 天,验证期 $valDays 天,验收期 ${dates.count { it >= holdStart }} 天")
            appendLine("- 迭代:共 ${roundLogs.size} 轮,评估 ${searchCache.size} 条策略,二次验证 $validations 次")
            appendLine()
            appendLine("## 结论")
            appendLine()
            appendLine("最终策略:**${gtext(finalGene)}**")
            appendLine()
            appendLine("| 数据段 | 做T次数 | 出手天数 | 胜率 | 胜率95%置信区间 | 盈亏平衡胜率 | 平均每笔(bp) | 每笔95%下限(bp) | 净盈亏(元) | 达标 |")
            appendLine("|---|---|---|---|---|---|---|---|---|---|")
            fun row(name: String, e: Eval, ok: String) = appendLine("| $name | ${e.n} | ${e.days} | ${pct(e.winRate)} | " +
                "${if (e.n == 0) "-" else "${pct(e.winLow)}~${pct(e.winHigh)}"} | ${if (e.breakEven.isNaN()) "-" else pct(e.breakEven)} | " +
                "${f1(e.avgBp)} | ${if (e.n == 0) "-" else f1(e.retLowBp)} | ${f0(e.pnl)} | $ok |")
            row("寻优期(用于挑选)", finalSearch, mark(finalSearch.meets(target, minBp, minTrades)))
            row("二次验证期", finalVal, mark(valPass(finalVal)))
            row("**最终验收期(样本外,只用一次)**", finalHold, mark(finalHold.meets(target, minBp, 1)))
            row("验收期·同离场参数不加过滤", holdAll, "-")
            appendLine()
            val holdDays = dates.count { it >= holdStart }
            appendLine("> 出手频率:验收期约 ${f1(finalHold.n.toDouble() / maxOf(1, holdDays))} 笔/交易日(全部 ${symbols.size} 只合计)。")
            appendLine("> 置信区间同时考虑笔数和**交易日聚类**(同一天的交易受同一大盘影响,不是独立样本),取两者中更保守的。" +
                "盈亏平衡胜率 = 平均亏损 ÷(平均盈利 + 平均亏损),实际胜率要明显高于它才有利润。" +
                "达标 = 胜率与平均每笔两个目标同时满足且净盈亏 > 0。")
            if (!valPass(finalVal)) appendLine("> **迭代未找到通过二次验证的策略**:下表给出验证期表现最好的策略;说明在当前数据与策略空间内达不到目标。")
            else if (!finalHold.meets(target, minBp, 1)) appendLine("> **通过了二次验证,但最终验收未达标**:说明验证期的达标有偶然成分,不能投入使用。")
            if (finalHold.n == 0) appendLine("> **验收期没有出手**:原因见第 6 节「出手漏斗」(候选点在哪一步被过滤掉)。")
            appendLine()
            appendLine("## 1. 迭代过程")
            appendLine()
            appendLine("| 轮次 | 累计评估 | 本轮寻优最佳(次数/胜率/平均bp) | 累计通过验证 | 最佳验证(次数/胜率/平均bp) |")
            appendLine("|---|---|---|---|---|")
            for (l in roundLogs) {
                if (l.round > 5 && l.round % 5 != 0 && l.round != roundLogs.last().round) continue
                val bv = l.bestVal
                appendLine("| ${l.round} | ${l.evaluated} | ${l.bestSearch.n} / ${pct(l.bestSearch.winRate)} / ${f1(l.bestSearch.avgBp)} | ${l.validatedTotal} | ${if (bv == null) "-" else "${bv.n} / ${pct(bv.winRate)} / ${f1(bv.avgBp)}"} |")
            }
            appendLine()
            appendLine("## 2. 二次验证表现最好的 10 条策略")
            appendLine()
            appendLine("| # | 策略 | 寻优期 次数/胜率/平均bp | 验证期 次数/天数/胜率/平均bp | 验证胜率95%下限 | 验证每笔95%下限(bp) | 验证净盈亏 | 通过 |")
            appendLine("|---|---|---|---|---|---|---|---|")
            for ((i, e) in top.withIndex()) {
                val s = searchCache.getValue(e.key); val v = e.value
                appendLine("| ${i + 1} | ${gtext(e.key)} | ${s.n} / ${pct(s.winRate)} / ${f1(s.avgBp)} | ${v.n} / ${v.days} / ${pct(v.winRate)} / ${f1(v.avgBp)} | ${pct(v.winLow)} | ${f1(v.retLowBp)} | ${f0(v.pnl)} | ${mark(valPass(v))} |")
            }
            appendLine()
            appendLine("## 3. 各维度单独看(寻优期 + 验证期,离场参数 = ${cfgText(grid[finalGene.k])},不加其他过滤)")
            appendLine()
            appendLine("| 维度 | 取值 | 次数 | 胜率 | 平均(bp) | 净盈亏 |")
            appendLine("|---|---|---|---|---|---|")
            val sv = (searchRows + valRows).also { it.sort() }
            fun dimRow(dim: String, name: String, spec: StrategySpec) {
                val e = simulateGene(c, sv, Gene(spec, finalGene.k, 0), sc, grid)
                appendLine("| $dim | $name | ${e.n} | ${pct(e.winRate)} | ${f1(e.avgBp)} | ${f0(e.pnl)} |")
            }
            listOf("缠论1分钟" to 1, "布林刺破收回" to 2, "RSI拐头" to 4, "缠论5分钟" to 8).forEach { (nm, m) -> dimRow("信号来源", nm, StrategySpec(families = m)) }
            dimRow("方向", "只先卖后买", StrategySpec(dirs = 1)); dimRow("方向", "只先买后卖", StrategySpec(dirs = 2))
            listOf(5 to 30, 31 to 60, 61 to 120, 121 to 180, 181 to 220).forEach { (a, b) ->
                dimRow("时间", "${StrategySpec.clock(a)}~${StrategySpec.clock(b)}", StrategySpec(timeStart = a, timeEnd = b))
            }
            for (lv in MarketLevel.entries) dimRow("大盘", lv.label, StrategySpec(regimeMask = 1 shl lv.ordinal))
            listOf(0.0 to 0.02, 0.02 to 0.03, 0.03 to 0.04, 0.04 to 0.06, 0.06 to 1.0).forEach { (a, b) ->
                dimRow("个股ATR", "${f1(a * 100)}%~${if (b >= 1) "不限" else "${f1(b * 100)}%"}", StrategySpec(atrMin = a, atrMax = b))
            }
            // 板块:按股票代码划分(主板/创业板/科创板/北交所),看优势是否集中在某类股票
            fun boardOf(si: Int): String {
                val sy = symbols[si]
                return when {
                    sy.market == "sz" && sy.code.startsWith("30") -> "创业板"
                    sy.market == "sh" && sy.code.startsWith("68") -> "科创板"
                    sy.market == "bj" -> "北交所"
                    else -> "主板"
                }
            }
            val boardRows = sv.groupBy { boardOf(c.symbol[it]) }
            for (bn in listOf("主板", "创业板", "科创板", "北交所")) {
                val rows = boardRows[bn]?.toIntArray() ?: continue
                val e = simulateGene(c, rows, Gene(StrategySpec(), finalGene.k, 0), sc, grid)
                appendLine("| 板块 | $bn | ${e.n} | ${pct(e.winRate)} | ${f1(e.avgBp)} | ${f0(e.pnl)} |")
            }
            appendLine()
            appendLine("### 离场参数对比(寻优期 + 验证期,不加任何过滤)")
            appendLine()
            val feeBp = cfg.costs.roundTripRateAt(cfg.baseValue * cfg.tConfig.tRatio) * 10000
            appendLine("看信号本身的优势:按 T 仓 ${f0(cfg.baseValue * cfg.tConfig.tRatio)} 元计,来回费用约 ${f1(feeBp)}bp(含滑点),\"扣费前\" = 平均每笔 + 费用。" +
                "扣费前接近 0 说明信号本身没有方向优势,过滤只是在噪声里挑。")
            appendLine()
            appendLine("| 离场参数 | 次数 | 胜率 | 平均每笔(bp) | 扣费前(bp) | 止盈 | 止损 | 限时离场 | 尾盘收尾 | 其他 |")
            appendLine("|---|---|---|---|---|---|---|---|---|---|")
            for (k in grid.indices) {
                val g0 = Gene(StrategySpec(), k, 0)
                val tk = takenRows(c, sv, g0, sc, grid)
                if (tk.isEmpty()) continue
                val e = simulateGene(c, sv, g0, sc, grid)
                val rc = IntArray(CloseReason.entries.size)
                for (r in tk) c.reason(r, k).let { if (it >= 0) rc[it]++ }
                fun share(vararg rs: CloseReason) = pct(rs.sumOf { rc[it.ordinal] }.toDouble() / tk.size)
                val other = tk.size - listOf(CloseReason.TARGET, CloseReason.STOP, CloseReason.TIME_STOP, CloseReason.TIME).sumOf { rc[it.ordinal] }
                appendLine("| ${cfgText(grid[k])} | ${e.n} | ${pct(e.winRate)} | ${f1(e.avgBp)} | ${f1(e.avgBp + feeBp)} | " +
                    "${share(CloseReason.TARGET)} | ${share(CloseReason.STOP)} | ${share(CloseReason.TIME_STOP)} | ${share(CloseReason.TIME)} | " +
                    "${pct(other.toDouble() / tk.size)} |")
            }
            appendLine()
            appendLine("## 4. 胜率模型")
            appendLine()
            appendLine("模型只作为策略的一个可选过滤条件,是否使用由寻优决定。样本外 AUC(0.5 = 随机):模型A ${"%.4f".format(modelAuc[0])},模型B ${"%.4f".format(modelAuc[1])}。")
            appendLine()
            if (abl.isNotEmpty()) {
                appendLine("各层/模块对模型A的贡献(寻优期,去掉该组重新训练后 AUC 的变化;基准 = 全部模块 ${"%.4f".format(baseAuc)})。" +
                    "负值越大越重要;去掉后 AUC 升高超过 ${ABL_DROP} 的组判为拖后腿,正式模型不再使用:")
                appendLine()
                appendLine("| 去掉的层/模块 | AUC | ΔAUC | 正式模型 |")
                appendLine("|---|---|---|---|")
                for (x in abl.sortedBy { it.delta }) appendLine("| ${x.group.label} | ${"%.4f".format(x.auc)} | ${"%+.4f".format(x.delta)} | " +
                    "${if (x.group in droppedGroups) "**剔除**" else "保留"} |")
                appendLine()
            }
            val imp = modelImp[0]; val tot = imp.sum().takeIf { it > 0 } ?: 1.0
            appendLine("模型A按层/模块的重要性:" + FeatureGroup.entries.filter { it != FeatureGroup.MARKET && it !in droppedGroups }
                .associateWith { g -> Features.columnsOf(g).sumOf { imp[it] } / tot }.entries.sortedByDescending { it.value }
                .joinToString(",") { "${it.key.label} ${pct(it.value)}" })
            appendLine()
            appendLine("## 5. 最终策略亏损单(寻优期 + 验证期)")
            appendLine()
            val tk = takenRows(c, sv, finalGene, sc, grid)
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
            appendLine("## 6. 出手漏斗与大盘环境")
            appendLine()
            appendLine("最终策略在各段的候选点逐步被哪个条件过滤(每行是被该条件过滤掉的候选点数):")
            appendLine()
            appendLine("| 步骤 | 寻优期 | 验证期 | 验收期 |")
            appendLine("|---|---|---|---|")
            val segs = listOf(searchRows, valRows, holdRows)
            val fails = segs.map { rows -> IntArray(10).also { cnt -> for (r in rows) cnt[failStep(c, r, finalGene, sc)]++ } }
            val takes = segs.map { takenRows(c, it, finalGene, sc, grid).size }
            appendLine("| 候选点总数 | ${segs.joinToString(" | ") { "%,d".format(it.size) }} |")
            for ((code, label) in listOf(8 to "该离场参数下未成交", 1 to "方向不符", 2 to "信号来源不符", 3 to "不在时间窗",
                    4 to "大盘环境不符", 5 to "ATR 不在范围", 6 to "均价偏离不够", 7 to "顺势程度不够", 9 to "模型分位不够"))
                appendLine("| − $label | ${fails.joinToString(" | ") { "%,d".format(it[code]) }} |")
            appendLine("| = 满足全部条件 | ${fails.joinToString(" | ") { "%,d".format(it[0]) }} |")
            appendLine("| 实际出手(同一时间只持一笔、每日上限、冷却) | ${takes.joinToString(" | ")} |")
            appendLine()
            appendLine("各段交易日的大盘环境(大盘环境是按日判定的,整天所有股票相同):")
            appendLine()
            appendLine("| 大盘环境 | 寻优期(天) | 验证期(天) | 验收期(天) | 最终策略允许 |")
            appendLine("|---|---|---|---|---|")
            val dayRegime = HashMap<Int, Int>()
            for (r in 0 until n) dayRegime.putIfAbsent(c.date[r], c.regime[r].toInt())
            val segBounds = listOf(searchStart to valStart, valStart to holdStart, holdStart to Int.MAX_VALUE)
            for (lv in MarketLevel.entries) {
                val cnt = segBounds.map { (a, b) -> dayRegime.count { (d, rg) -> d >= a && d < b && rg == lv.ordinal } }
                val allowed = (finalGene.spec.regimeMask and (1 shl lv.ordinal)) != 0
                appendLine("| ${lv.label} | ${cnt.joinToString(" | ")} | ${if (allowed) "√" else "×"} |")
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
    /** 模型阈值(分位):0.90 表示只做模型分数排在近期前 10% 的候选点 */
    private val thrs = doubleArrayOf(0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.92, 0.94, 0.96, 0.97, 0.98, 0.99)

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
    /** 满足条件的行号(按存储顺序:同一股票连续、时间先后有序,可直接线性扫描) */
    private inline fun rowsWhere(n: Int, pred: (Int) -> Boolean): IntArray {
        var cnt = 0
        for (r in 0 until n) if (pred(r)) cnt++
        val out = IntArray(cnt); var i = 0
        for (r in 0 until n) if (pred(r)) out[i++] = r
        return out
    }

    /** 候选点在哪一步被过滤:0 通过;1~7 见 [StrategySpec.firstFail];8 该离场参数下未成交;9 模型分位不够 */
    private fun failStep(c: Cands, r: Int, g: Gene, sc: Scores): Int {
        if (!c.filled(r, g.k)) return 8
        val f = g.spec.firstFail(c.family[r].toInt(), c.dirOf(r), c.elapsed[r].toInt(), c.regime[r].toInt(),
            c.atr[r].toDouble(), c.vdev[r].toDouble(), c.trend[r].toDouble())
        if (f != 0) return f
        if (g.model != 0) {
            val q = sc.pctl[g.model - 1][r]
            // 模型阈值按分位:分数在该段模型近期(早停集)分数中的位置 ≥ modelThr(寻优时 modelThr 存的是分位)
            if (q.isNaN() || q < g.spec.modelThr - PCTL_EPS) return 9
        }
        return 0
    }

    private fun passes(c: Cands, r: Int, g: Gene, sc: Scores): Boolean = failStep(c, r, g, sc) == 0

    /** 同一时刻多个候选:用模型取胜率高者,否则取 L4 综合分高者;同一时间只持一笔、每日上限、冷却 */
    private fun takenRows(c: Cands, rows: IntArray, g: Gene, sc: Scores, grid: List<TConfig>): IntArray {
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
                if (passes(c, q, g, sc)) {
                    val pr = if (g.model != 0) sc.prob[g.model - 1][q] else c.score[q].toDouble()
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

    private fun simulateGene(c: Cands, rows: IntArray, g: Gene, sc: Scores, grid: List<TConfig>): Eval {
        var n = 0; var wins = 0; var pnl = 0.0; var sumRet = 0.0; var winPnl = 0.0; var sumRet2 = 0.0
        val notional = cfg.baseValue * grid[g.k].tRatio
        val perDay = HashMap<Int, DoubleArray>() // 日期 → [次数, 赢, 收益和]
        for (r in takenRows(c, rows, g, sc, grid)) {
            val v = c.pnl(r, g.k).toDouble()
            val ret = v / (if (c.value[r] > 0f) c.value[r].toDouble() else notional)
            val d = perDay.getOrPut(c.date[r]) { DoubleArray(3) }
            n++; d[0]++; d[2] += ret
            if (v > 0) { wins++; d[1]++; winPnl += v }
            pnl += v; sumRet += ret; sumRet2 += ret * ret
        }
        var ssr = 0.0; var ssrRet = 0.0
        if (n > 0) {
            val p = wins.toDouble() / n; val m = sumRet / n
            for (d in perDay.values) {
                val e = d[1] - p * d[0]; ssr += e * e
                val er = d[2] - m * d[0]; ssrRet += er * er
            }
        }
        return Eval(n, wins, pnl, sumRet, winPnl, perDay.size, ssr, sumRet2, ssrRet)
    }

    // ---------------- 模型训练(单段)----------------
    /** [ref]:早停集(该段之前最近的交易日)上的校准分数,升序;用来把分数换算成分位 */
    private class FoldModel(val model: Gbdt, val calib: Calibration, val rows: IntArray, val pred: DoubleArray, val ref: DoubleArray) {
        /** 分数 x 在 ref 中的分位 = ref 中 ≤ x 的比例 */
        fun pctl(x: Double): Float {
            if (ref.isEmpty()) return Float.NaN
            var lo = 0; var hi = ref.size
            while (lo < hi) { val m = (lo + hi) ushr 1; if (ref[m] <= x) lo = m + 1 else hi = m }
            return (lo.toDouble() / ref.size).toFloat()
        }

        /** 与"分位 ≥ q"完全等价的校准胜率阈值(部署到实盘引擎用):ref 中最小的 v,使 pctl(v) ≥ q */
        fun probThreshold(q: Double): Double {
            if (ref.isEmpty()) return 1.0
            var i = (Math.ceil((q - PCTL_EPS) * ref.size).toInt() - 1).coerceIn(0, ref.size - 1)
            while (i > 0 && pctl(ref[i - 1]) >= q - PCTL_EPS) i--
            while (i < ref.size - 1 && pctl(ref[i]) < q - PCTL_EPS) i++
            return if (pctl(ref[i]) >= q - PCTL_EPS) ref[i] else Math.nextUp(ref.last())
        }
    }

    private val labelCache = java.util.concurrent.ConcurrentHashMap<Int, ByteArray>()

    /** 某离场组下的输赢标签(各段、各模型共享同一份) */
    private fun labels(c: Cands, k: Int): ByteArray = labelCache.getOrPut(k) {
        ByteArray(c.n) { r -> if (c.filled(r, k) && c.pnl(r, k) > 0) 1 else 0 }
    }

    private fun trainFold(
        bd: BinnedData, c: Cands, dates: List<Int>, foldStarts: List<Int>,
        k: Int, f: Int, cols: IntArray?, screen: Boolean,
    ): FoldModel {
        val start = foldStarts[f]
        val end = if (f + 1 < foldStarts.size) foldStarts[f + 1] else Int.MAX_VALUE
        val trainDates = dates.filter { it < start }
        val esStart = trainDates[(trainDates.size * 0.85).toInt().coerceAtMost(trainDates.size - 1)]
        val y = labels(c, k)
        val inPred = { r: Int -> c.date[r] >= start && c.date[r] < end }
        val tr = rowsWhere(c.n) { c.filled(it, k) && c.date[it] < esStart }
        val es = rowsWhere(c.n) { c.filled(it, k) && c.date[it] >= esStart && c.date[it] < start }
        val pr = rowsWhere(c.n) { inPred(it) }
        val m = Gbdt.trainBinned(bd, y, tr, es, cols, gbdtParams(screen).copy(seed = 7 + k * 31 + f))
        val esScores = DoubleArray(es.size) { m.predictBinned(bd, es[it]) }
        val calib = Calibration.fit(esScores, IntArray(es.size) { y[es[it]].toInt() })
        val ref = DoubleArray(es.size) { calib.apply(esScores[it]) }.also { it.sort() }
        return FoldModel(m, calib, pr, DoubleArray(pr.size) { calib.apply(m.predictBinned(bd, pr[it])) }, ref)
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
                blk.value[row] = 0f
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
                        if (blk.value[row] == 0f) blk.value[row] = (t.openPrice * t.shares).toFloat()
                        blk.reason[idx] = t.closeReason.ordinal.toByte()
                        blk.closeE[idx] = Session.elapsed(t.closeMinute).toShort()
                    }
                }
                blk.n++
            }
        }
    }

    /** 样本外 AUC(越高说明模型越能区分赢单与输单) */
    private fun auc(c: Cands, rows: IntArray, p: DoubleArray, k: Int): Double = auc(c, rows, k) { p[it] }

    private inline fun auc(c: Cands, rows: IntArray, k: Int, p: (Int) -> Double): Double {
        // 正负样本分数各自排序后归并计数(Mann-Whitney U),不装箱
        var np = 0; var nn = 0
        for (r in rows) if (!p(r).isNaN() && c.filled(r, k)) { if (c.pnl(r, k) > 0) np++ else nn++ }
        if (np == 0 || nn == 0) return 0.5
        val pos = DoubleArray(np); val neg = DoubleArray(nn); var i = 0; var j = 0
        for (r in rows) if (!p(r).isNaN() && c.filled(r, k)) { if (c.pnl(r, k) > 0) pos[i++] = p(r) else neg[j++] = p(r) }
        pos.sort(); neg.sort()
        var u = 0.0; var lo = 0; var hi = 0
        for (x in pos) {
            while (lo < nn && neg[lo] < x) lo++
            while (hi < nn && neg[hi] <= x) hi++
            u += lo + (hi - lo) * 0.5
        }
        return u / (np.toDouble() * nn)
    }

    /**
     * 训练任务可同时运行的个数,按剩余内存限制并发。每个 GBDT 任务约需 行数 × 28 字节:
     * 间隔/梯度/二阶导 12 + 训练/早停/预测行号 4 + 行抽样与节点划分 8 + 预测结果 4。
     */
    private fun modelParallelism(n: Int): Int {
        val rt = Runtime.getRuntime()
        System.gc()
        val free = rt.maxMemory() - (rt.totalMemory() - rt.freeMemory())
        val perJob = n * 28L + 60_000_000L
        return (free * 0.8 / perJob).toInt().coerceIn(1, threads)
    }

    private fun memText(): String {
        val rt = Runtime.getRuntime()
        return "%.1f/%.1fGB".format((rt.totalMemory() - rt.freeMemory()) / 1e9, rt.maxMemory() / 1e9)
    }

    private fun runParallel(tasks: List<Callable<Unit>>, parallelism: Int = threads) {
        if (tasks.isEmpty()) return
        val pool = Executors.newFixedThreadPool(parallelism.coerceIn(1, threads))
        try {
            pool.invokeAll(tasks).forEach { it.get() }
        } finally {
            pool.shutdown()
        }
    }

    private fun cfgText(g: TConfig) = "${if (g.entryOffsetAtr > 0) "挂单${g.entryOffsetAtr}ATR" else "市价"} 止盈${g.takeProfitAtr} 止损${g.stopLossAtr}" +
        if (g.timeStopMin > 0) " 限时${g.timeStopMin}分钟" else ""
    private fun pct(x: Double) = "%.1f%%".format(x * 100)
    private fun gb(bytes: Double) = "%.1fGB".format(bytes / 1e9)
    private fun f0(x: Double) = "%.0f".format(x)
    private fun f1(x: Double) = "%.1f".format(x)
    private fun mark(b: Boolean) = if (b) "√" else "×"

    companion object {
        /**
         * 离场方式:0 = 持有到止盈/止损/尾盘;45 = 45 分钟内未止盈就按市价离场。
         * (早期的"挂单 0.1ATR 开仓"在真实数据的历次训练中从未被选中,改为限时离场,参数组数保持 32)
         */
        val HOLDS = intArrayOf(0, 45)
        val TPS = doubleArrayOf(0.08, 0.12, 0.18, 0.25)
        val SLS = doubleArrayOf(0.6, 1.0, 1.6, 2.5)
        private const val PCTL_EPS = 1e-6
        /** 去掉某组后 AUC 升高超过此值,判为该组拖后腿 */
        private const val ABL_DROP = 0.002

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

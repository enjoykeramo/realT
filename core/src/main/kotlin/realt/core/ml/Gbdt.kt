package realt.core.ml

import kotlin.math.exp
import kotlin.math.ln
import kotlin.random.Random

/** 一棵回归树,数组形式存储:叶子 feature = -1。threshold 为原始特征值,bin 为分箱下标(训练期使用) */
class Tree(
    val feature: IntArray,
    val threshold: DoubleArray,
    val left: IntArray,
    val right: IntArray,
    val value: DoubleArray,
    val bin: IntArray = IntArray(feature.size),
) {
    fun predict(x: DoubleArray): Double {
        var n = 0
        while (feature[n] >= 0) n = if (x[feature[n]] <= threshold[n]) left[n] else right[n]
        return value[n]
    }

    fun predictBinned(d: BinnedData, row: Int): Double {
        var n = 0
        while (feature[n] >= 0) n = if ((d.cols[feature[n]][row].toInt() and 0xFF) <= bin[n]) left[n] else right[n]
        return value[n]
    }
}

/**
 * 分箱后的特征矩阵(列存,每值 1 字节)。全部候选点只分箱一次,所有模型(各折、各离场组、消融)共享,
 * 显著降低内存与训练时间。分位点只用起始训练期的样本计算(无监督,不使用标签)。
 */
class BinnedData(val edges: Array<DoubleArray>, val cols: Array<ByteArray>) {
    val nFeatures: Int get() = cols.size
    val nRows: Int get() = if (cols.isEmpty()) 0 else cols[0].size

    companion object {
        /** 由 [value](row, feature) 取值;[edgeRows] 用于计算分位点 */
        fun build(nRows: Int, nFeatures: Int, edgeRows: IntArray, bins: Int = 64, value: (Int, Int) -> Double): BinnedData {
            val edges = Array(nFeatures) { f ->
                val vals = DoubleArray(edgeRows.size) { value(edgeRows[it], f) }.also { it.sort() }
                val e = ArrayList<Double>()
                if (vals.isNotEmpty()) for (b in 1 until bins) {
                    val v = vals[(b.toLong() * (vals.size - 1) / bins).toInt()]
                    if (e.isEmpty() || v > e.last()) e += v
                }
                e.toDoubleArray()
            }
            val cols = Array(nFeatures) { f ->
                val e = edges[f]
                ByteArray(nRows) { r -> binOf(e, value(r, f)).toByte() }
            }
            return BinnedData(edges, cols)
        }

        fun binOf(e: DoubleArray, v: Double): Int {
            var lo = 0; var hi = e.size
            while (lo < hi) {
                val m = (lo + hi) ushr 1
                if (v <= e[m]) hi = m else lo = m + 1
            }
            return lo
        }
    }
}

/**
 * 梯度提升决策树(二分类,logloss),直方图分箱实现,纯 Kotlin,无第三方依赖。
 * 输出 = 原始胜率分数;经 [Calibration] 校准后才是可比较的胜率概率。
 */
class Gbdt(val base: Double, val learningRate: Double, val trees: List<Tree>) {

    fun margin(x: DoubleArray): Double {
        var m = base
        for (t in trees) m += learningRate * t.predict(x)
        return m
    }

    fun predict(x: DoubleArray): Double = 1.0 / (1.0 + exp(-margin(x)))

    fun predictBinned(d: BinnedData, row: Int): Double {
        var m = base
        for (t in trees) m += learningRate * t.predictBinned(d, row)
        return 1.0 / (1.0 + exp(-m))
    }

    /** 各特征被用于分裂的总增益,用于解释模型 */
    var importance: DoubleArray = DoubleArray(0)

    fun serialize(): String = buildString {
        appendLine("gbdt $base $learningRate ${trees.size}")
        for (t in trees) {
            append("tree ${t.feature.size}")
            for (i in t.feature.indices) append(" ${t.feature[i]},${t.threshold[i]},${t.left[i]},${t.right[i]},${t.value[i]}")
            appendLine()
        }
    }

    data class Params(
        val nTrees: Int = 300,
        val maxDepth: Int = 3,
        val learningRate: Double = 0.05,
        val minLeaf: Int = 40,
        val lambda: Double = 1.0,
        val rowSample: Double = 0.8,
        val colSample: Double = 0.8,
        val bins: Int = 64,
        /** 验证集 logloss 连续多少棵树不改善即停止 */
        val earlyStop: Int = 30,
        val seed: Int = 7,
    )

    companion object {
        fun deserialize(text: String): Gbdt {
            val lines = text.lineSequence().map { it.trim() }.filter { it.isNotEmpty() }.toList()
            val head = lines.first().split(' ')
            require(head[0] == "gbdt") { "不是 GBDT 模型" }
            val trees = lines.drop(1).filter { it.startsWith("tree") }.map { l ->
                val p = l.split(' ')
                val n = p[1].toInt()
                val f = IntArray(n); val th = DoubleArray(n); val le = IntArray(n); val ri = IntArray(n); val v = DoubleArray(n)
                for (i in 0 until n) {
                    val q = p[2 + i].split(',')
                    f[i] = q[0].toInt(); th[i] = q[1].toDouble(); le[i] = q[2].toInt(); ri[i] = q[3].toInt(); v[i] = q[4].toDouble()
                }
                Tree(f, th, le, ri, v)
            }
            return Gbdt(head[1].toDouble(), head[2].toDouble(), trees)
        }

        /** 便捷入口:直接用原始特征训练(内部分箱)。[vx]/[vy] 为早停验证集,可为空 */
        fun train(
            x: List<DoubleArray>, y: IntArray,
            vx: List<DoubleArray> = emptyList(), vy: IntArray = IntArray(0),
            p: Params = Params(),
        ): Gbdt {
            val all = x + vx
            val d = BinnedData.build(all.size, x[0].size, IntArray(x.size) { it }, p.bins) { r, f -> all[r][f] }
            val labels = ByteArray(all.size) { if (it < x.size) y[it].toByte() else vy[it - x.size].toByte() }
            return trainBinned(d, labels, IntArray(x.size) { it }, IntArray(vx.size) { x.size + it }, null, p)
        }

        /**
         * 在共享的分箱矩阵上训练。[trainRows]/[esRows] 为行下标;[allowed] 为允许使用的特征列(null = 全部)。
         * 线程安全:不修改 [d],可多个模型并行训练。
         */
        fun trainBinned(
            d: BinnedData, y: ByteArray, trainRows: IntArray, esRows: IntArray,
            allowed: IntArray?, p: Params = Params(),
        ): Gbdt {
            require(trainRows.isNotEmpty())
            val nf = d.nFeatures
            val feats = allowed ?: IntArray(nf) { it }
            val rnd = Random(p.seed)
            var pos = 0.0
            for (r in trainRows) pos += y[r]
            val prior = ((pos + 1) / (trainRows.size + 2)).coerceIn(1e-4, 1 - 1e-4)
            val base = ln(prior / (1 - prior))
            val n = d.nRows
            // 间隔/梯度均用 float 存储:候选点上百万时,每个并行任务的工作区约 12 字节/行
            val margin = FloatArray(n)
            for (r in trainRows) margin[r] = base.toFloat()
            for (r in esRows) margin[r] = base.toFloat()
            val grad = FloatArray(n)
            val hess = FloatArray(n)
            val trees = ArrayList<Tree>()
            val importance = DoubleArray(nf)
            var bestLoss = Double.MAX_VALUE
            var bestCount = 0
            var sinceBest = 0

            for (t in 0 until p.nTrees) {
                for (r in trainRows) {
                    val pr = 1.0 / (1.0 + exp(-margin[r].toDouble()))
                    grad[r] = (pr - y[r]).toFloat()
                    hess[r] = maxOf(pr * (1 - pr), 1e-6).toFloat()
                }
                val rows = sample(trainRows, p.rowSample, rnd)
                val cols = feats.filter { rnd.nextDouble() < p.colSample }.ifEmpty { listOf(feats[rnd.nextInt(feats.size)]) }.toIntArray()
                val tree = TreeBuilder(d, grad, hess, cols, p, importance).build(rows)
                trees += tree
                for (r in trainRows) margin[r] += (p.learningRate * tree.predictBinned(d, r)).toFloat()
                if (esRows.isNotEmpty()) {
                    var loss = 0.0
                    for (r in esRows) {
                        margin[r] += (p.learningRate * tree.predictBinned(d, r)).toFloat()
                        val pr = (1.0 / (1.0 + exp(-margin[r].toDouble()))).coerceIn(1e-9, 1 - 1e-9)
                        loss -= if (y[r].toInt() == 1) ln(pr) else ln(1 - pr)
                    }
                    loss /= esRows.size
                    if (loss < bestLoss - 1e-7) {
                        bestLoss = loss; bestCount = trees.size; sinceBest = 0
                    } else if (++sinceBest >= p.earlyStop) break
                }
            }
            val kept = if (esRows.isNotEmpty() && bestCount > 0) trees.subList(0, bestCount).toList() else trees
            return Gbdt(base, p.learningRate, kept).also { it.importance = importance }
        }

        private fun sample(rows: IntArray, ratio: Double, rnd: Random): IntArray {
            if (ratio >= 1.0) return rows
            val buf = IntArray(rows.size)
            var m = 0
            for (r in rows) if (rnd.nextDouble() < ratio) buf[m++] = r
            return buf.copyOf(m)
        }
    }

    private class TreeBuilder(
        val d: BinnedData,
        val grad: FloatArray,
        val hess: FloatArray,
        val cols: IntArray,
        val p: Params,
        val importance: DoubleArray,
    ) {
        val feature = ArrayList<Int>()
        val threshold = ArrayList<Double>()
        val bin = ArrayList<Int>()
        val left = ArrayList<Int>()
        val right = ArrayList<Int>()
        val value = ArrayList<Double>()

        /** 每个特征的直方图段长 = 最大箱数 */
        private val STRIDE = maxOf(2, cols.maxOf { d.edges[it].size + 1 })

        /** 节点直方图:每个候选特征一段长度为 STRIDE 的 g/h/count */
        private inner class Hist {
            val g = DoubleArray(cols.size * STRIDE)
            val h = DoubleArray(cols.size * STRIDE)
            val c = IntArray(cols.size * STRIDE)
        }

        private fun compute(rows: IntArray): Hist {
            val hs = Hist()
            for ((ci, f) in cols.withIndex()) {
                val base = ci * STRIDE
                val col = d.cols[f]
                for (r in rows) {
                    val b = base + (col[r].toInt() and 0xFF)
                    hs.g[b] += grad[r]; hs.h[b] += hess[r]; hs.c[b]++
                }
            }
            return hs
        }

        /** 直方图相减:大子节点 = 父节点 - 小子节点,省去一半统计量 */
        private fun subtract(parent: Hist, small: Hist): Hist {
            val hs = Hist()
            for (i in hs.g.indices) {
                hs.g[i] = parent.g[i] - small.g[i]; hs.h[i] = parent.h[i] - small.h[i]; hs.c[i] = parent.c[i] - small.c[i]
            }
            return hs
        }

        fun build(rows: IntArray): Tree {
            node(rows, 0, null)
            return Tree(feature.toIntArray(), threshold.toDoubleArray(), left.toIntArray(), right.toIntArray(),
                value.toDoubleArray(), bin.toIntArray())
        }

        private fun newNode(): Int {
            feature += -1; threshold += 0.0; bin += 0; left += -1; right += -1; value += 0.0
            return feature.size - 1
        }

        private fun splittable(n: Int, depth: Int) = depth < p.maxDepth && n >= 2 * p.minLeaf

        private fun node(rows: IntArray, depth: Int, given: Hist?): Int {
            val id = newNode()
            var g = 0.0; var h = 0.0
            for (r in rows) { g += grad[r]; h += hess[r] }
            value[id] = -g / (h + p.lambda)
            if (!splittable(rows.size, depth)) return id

            val hs = given ?: compute(rows)
            val parentScore = g * g / (h + p.lambda)
            var bestGain = 0.0; var bestF = -1; var bestBin = -1
            for ((ci, f) in cols.withIndex()) {
                val nb = d.edges[f].size + 1
                if (nb < 2) continue
                val base = ci * STRIDE
                var gl = 0.0; var hl = 0.0; var cl = 0
                for (b in 0 until nb - 1) {
                    gl += hs.g[base + b]; hl += hs.h[base + b]; cl += hs.c[base + b]
                    val cr = rows.size - cl
                    if (cl < p.minLeaf) continue
                    if (cr < p.minLeaf) break
                    val gr = g - gl; val hr = h - hl
                    val gain = gl * gl / (hl + p.lambda) + gr * gr / (hr + p.lambda) - parentScore
                    if (gain > bestGain) { bestGain = gain; bestF = f; bestBin = b }
                }
            }
            if (bestF < 0) return id
            importance[bestF] += bestGain
            val col = d.cols[bestF]
            var nl = 0
            for (x in rows) if ((col[x].toInt() and 0xFF) <= bestBin) nl++
            val l = IntArray(nl); val r = IntArray(rows.size - nl)
            var li = 0; var ri = 0
            for (x in rows) if ((col[x].toInt() and 0xFF) <= bestBin) l[li++] = x else r[ri++] = x
            feature[id] = bestF
            threshold[id] = d.edges[bestF][bestBin]
            bin[id] = bestBin

            var lh: Hist? = null; var rh: Hist? = null
            val needL = splittable(l.size, depth + 1); val needR = splittable(r.size, depth + 1)
            if (needL && needR) {
                if (l.size <= r.size) { lh = compute(l); rh = subtract(hs, lh) } else { rh = compute(r); lh = subtract(hs, rh) }
            }
            val lc = node(l, depth + 1, lh)
            val rc = node(r, depth + 1, rh)
            left[id] = lc; right[id] = rc
            return id
        }
    }
}

/**
 * 概率校准(保序回归 / PAV):把模型原始分数映射为"历史上真实的胜率"。
 * 用每个模型自己的留出集拟合,使不同时间段、不同数据量训练出的模型,其输出在同一刻度上可比,
 * 出手阈值(例如 0.95)才有"胜率 95%"的含义。
 */
class Calibration(val x: DoubleArray, val y: DoubleArray) {
    fun apply(p: Double): Double {
        if (x.isEmpty()) return p
        if (p <= x[0]) return y[0]
        if (p >= x[x.size - 1]) return y[y.size - 1]
        var lo = 0; var hi = x.size - 1
        while (hi - lo > 1) {
            val m = (lo + hi) ushr 1
            if (x[m] <= p) lo = m else hi = m
        }
        val t = (p - x[lo]) / (x[hi] - x[lo]).coerceAtLeast(1e-12)
        return y[lo] + t * (y[hi] - y[lo])
    }

    fun serialize(): String = x.indices.joinToString(";") { "${x[it]},${y[it]}" }

    companion object {
        val IDENTITY = Calibration(DoubleArray(0), DoubleArray(0))

        fun parse(s: String): Calibration {
            if (s.isBlank()) return IDENTITY
            val pts = s.split(';').map { it.split(',') }
            return Calibration(DoubleArray(pts.size) { pts[it][0].toDouble() }, DoubleArray(pts.size) { pts[it][1].toDouble() })
        }

        /** 保序回归;每个块至少 [minBlock] 个样本,块中心(分数均值)→ 块胜率,块间线性插值 */
        fun fit(scores: DoubleArray, labels: IntArray, minBlock: Int = 50): Calibration {
            if (scores.size < 2 * minBlock) return IDENTITY
            val idx = scores.indices.sortedBy { scores[it] }
            // 先按 minBlock 聚成初始块,再做 PAV 合并
            val sx = ArrayList<Double>(); val sy = ArrayList<Double>(); val sw = ArrayList<Double>()
            var i = 0
            while (i < idx.size) {
                val end = minOf(idx.size, i + minBlock).let { if (idx.size - it < minBlock) idx.size else it }
                var ax = 0.0; var ay = 0.0
                for (k in i until end) { ax += scores[idx[k]]; ay += labels[idx[k]] }
                val w = (end - i).toDouble()
                sx += ax / w; sy += ay / w; sw += w
                // PAV:违反单调则与前一块合并
                while (sy.size >= 2 && sy[sy.size - 2] >= sy[sy.size - 1]) {
                    val w2 = sw.removeAt(sw.size - 1); val y2 = sy.removeAt(sy.size - 1); val x2 = sx.removeAt(sx.size - 1)
                    val w1 = sw[sw.size - 1]
                    sy[sy.size - 1] = (sy[sy.size - 1] * w1 + y2 * w2) / (w1 + w2)
                    sx[sx.size - 1] = (sx[sx.size - 1] * w1 + x2 * w2) / (w1 + w2)
                    sw[sw.size - 1] = w1 + w2
                }
                i = end
            }
            return Calibration(sx.toDoubleArray(), sy.toDoubleArray())
        }
    }
}

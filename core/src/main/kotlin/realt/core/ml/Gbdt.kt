package realt.core.ml

import kotlin.math.exp
import kotlin.math.ln
import kotlin.random.Random

/** 一棵回归树,数组形式存储:叶子 feature = -1 */
class Tree(
    val feature: IntArray,
    val threshold: DoubleArray,
    val left: IntArray,
    val right: IntArray,
    val value: DoubleArray,
) {
    fun predict(x: DoubleArray): Double {
        var n = 0
        while (feature[n] >= 0) n = if (x[feature[n]] <= threshold[n]) left[n] else right[n]
        return value[n]
    }
}

/**
 * 梯度提升决策树(二分类,logloss),直方图分箱实现,纯 Kotlin,无第三方依赖。
 * 输出 = 胜率概率 P(这次做T扣费后盈利)。
 */
class Gbdt(val base: Double, val learningRate: Double, val trees: List<Tree>) {

    fun margin(x: DoubleArray): Double {
        var m = base
        for (t in trees) m += learningRate * t.predict(x)
        return m
    }

    fun predict(x: DoubleArray): Double = 1.0 / (1.0 + exp(-margin(x)))

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
        val bins: Int = 32,
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

        /**
         * 训练。[x] 行主序特征,[y] 0/1 标签;[vx]/[vy] 为验证集(用于早停,可为空)。
         */
        fun train(
            x: List<DoubleArray>, y: IntArray,
            vx: List<DoubleArray> = emptyList(), vy: IntArray = IntArray(0),
            p: Params = Params(),
        ): Gbdt {
            val n = x.size
            require(n > 0)
            val nf = x[0].size
            val rnd = Random(p.seed)
            // ---- 分箱:每个特征取分位点 ----
            val edges = Array(nf) { f ->
                val vals = DoubleArray(n) { x[it][f] }.also { it.sort() }
                val e = ArrayList<Double>()
                for (b in 1 until p.bins) {
                    val v = vals[(b.toLong() * (n - 1) / p.bins).toInt()]
                    if (e.isEmpty() || v > e.last()) e += v
                }
                e.toDoubleArray()
            }
            val binned = Array(nf) { f ->
                val e = edges[f]
                ByteArray(n) { i -> binOf(e, x[i][f]).toByte() }
            }
            val pos = y.sum().toDouble()
            val prior = ((pos + 1) / (n + 2)).coerceIn(1e-4, 1 - 1e-4)
            val base = ln(prior / (1 - prior))
            val margin = DoubleArray(n) { base }
            val vMargin = DoubleArray(vx.size) { base }
            val grad = DoubleArray(n)
            val hess = DoubleArray(n)
            val trees = ArrayList<Tree>()
            val importance = DoubleArray(nf)
            var bestLoss = Double.MAX_VALUE
            var bestCount = 0
            var sinceBest = 0

            for (t in 0 until p.nTrees) {
                for (i in 0 until n) {
                    val pr = 1.0 / (1.0 + exp(-margin[i]))
                    grad[i] = pr - y[i]
                    hess[i] = maxOf(pr * (1 - pr), 1e-6)
                }
                val rows = sampleRows(n, p.rowSample, rnd)
                val cols = (0 until nf).filter { rnd.nextDouble() < p.colSample }.ifEmpty { listOf(rnd.nextInt(nf)) }.toIntArray()
                val tree = TreeBuilder(binned, edges, grad, hess, cols, p, importance).build(rows)
                trees += tree
                for (i in 0 until n) margin[i] += p.learningRate * tree.predict(x[i])
                if (vx.isNotEmpty()) {
                    var loss = 0.0
                    for (i in vx.indices) {
                        vMargin[i] += p.learningRate * tree.predict(vx[i])
                        val pr = (1.0 / (1.0 + exp(-vMargin[i]))).coerceIn(1e-9, 1 - 1e-9)
                        loss -= if (vy[i] == 1) ln(pr) else ln(1 - pr)
                    }
                    loss /= vx.size
                    if (loss < bestLoss - 1e-6) {
                        bestLoss = loss; bestCount = trees.size; sinceBest = 0
                    } else if (++sinceBest >= p.earlyStop) break
                }
            }
            val kept = if (vx.isNotEmpty() && bestCount > 0) trees.subList(0, bestCount).toList() else trees
            return Gbdt(base, p.learningRate, kept).also { it.importance = importance }
        }

        private fun sampleRows(n: Int, ratio: Double, rnd: Random): IntArray {
            val buf = IntArray(n)
            var m = 0
            for (i in 0 until n) if (rnd.nextDouble() < ratio) buf[m++] = i
            return buf.copyOf(m)
        }

        private fun binOf(e: DoubleArray, v: Double): Int {
            var lo = 0; var hi = e.size
            while (lo < hi) {
                val m = (lo + hi) ushr 1
                if (v <= e[m]) hi = m else lo = m + 1
            }
            return lo
        }
    }

    private class TreeBuilder(
        val binned: Array<ByteArray>,
        val edges: Array<DoubleArray>,
        val grad: DoubleArray,
        val hess: DoubleArray,
        val cols: IntArray,
        val p: Params,
        val importance: DoubleArray,
    ) {
        val feature = ArrayList<Int>()
        val threshold = ArrayList<Double>()
        val left = ArrayList<Int>()
        val right = ArrayList<Int>()
        val value = ArrayList<Double>()

        fun build(rows: IntArray): Tree {
            node(rows, 0)
            return Tree(feature.toIntArray(), threshold.toDoubleArray(), left.toIntArray(), right.toIntArray(), value.toDoubleArray())
        }

        private fun newNode(): Int {
            feature += -1; threshold += 0.0; left += -1; right += -1; value += 0.0
            return feature.size - 1
        }

        private fun node(rows: IntArray, depth: Int): Int {
            val id = newNode()
            var g = 0.0; var h = 0.0
            for (r in rows) { g += grad[r]; h += hess[r] }
            value[id] = -g / (h + p.lambda)
            if (depth >= p.maxDepth || rows.size < 2 * p.minLeaf) return id

            val parentScore = g * g / (h + p.lambda)
            var bestGain = 0.0; var bestF = -1; var bestBin = -1
            for (f in cols) {
                val nb = edges[f].size + 1
                if (nb < 2) continue
                val gh = DoubleArray(nb); val hh = DoubleArray(nb); val ch = IntArray(nb)
                val col = binned[f]
                for (r in rows) {
                    val b = col[r].toInt() and 0xFF
                    gh[b] += grad[r]; hh[b] += hess[r]; ch[b]++
                }
                var gl = 0.0; var hl = 0.0; var cl = 0
                for (b in 0 until nb - 1) {
                    gl += gh[b]; hl += hh[b]; cl += ch[b]
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
            val col = binned[bestF]
            var nl = 0
            for (x in rows) if ((col[x].toInt() and 0xFF) <= bestBin) nl++
            val l = IntArray(nl); val r = IntArray(rows.size - nl)
            var li0 = 0; var ri0 = 0
            for (x in rows) if ((col[x].toInt() and 0xFF) <= bestBin) l[li0++] = x else r[ri0++] = x
            feature[id] = bestF
            threshold[id] = edges[bestF][bestBin]
            val li = node(l, depth + 1)
            val ri = node(r, depth + 1)
            left[id] = li; right[id] = ri
            return id
        }
    }
}

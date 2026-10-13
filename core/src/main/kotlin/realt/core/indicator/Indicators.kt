package realt.core.indicator

import realt.core.model.Bar
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.sqrt

/**
 * 常用技术指标。全部是"只看过去"的因果计算:第 i 个值只依赖 0..i 的数据,
 * 因此在回测中逐 bar 截取前缀计算与实时计算结果一致。
 * 预热期不足的位置填 NaN。
 */
object Indicators {

    fun sma(x: DoubleArray, n: Int): DoubleArray {
        val out = DoubleArray(x.size) { Double.NaN }
        var sum = 0.0
        for (i in x.indices) {
            sum += x[i]
            if (i >= n) sum -= x[i - n]
            if (i >= n - 1) out[i] = sum / n
        }
        return out
    }

    /** 指数平均,首值用第一个数据初始化(与通达信 EMA 一致) */
    fun ema(x: DoubleArray, n: Int): DoubleArray {
        val out = DoubleArray(x.size)
        if (x.isEmpty()) return out
        val k = 2.0 / (n + 1)
        out[0] = x[0]
        for (i in 1 until x.size) out[i] = out[i - 1] + k * (x[i] - out[i - 1])
        return out
    }

    /** 通达信 SMA(X,N,M):Y = (M*X + (N-M)*Y') / N */
    fun tdxSma(x: DoubleArray, n: Int, m: Int): DoubleArray {
        val out = DoubleArray(x.size)
        if (x.isEmpty()) return out
        out[0] = x[0]
        for (i in 1 until x.size) out[i] = (m * x[i] + (n - m) * out[i - 1]) / n
        return out
    }

    class Macd(val dif: DoubleArray, val dea: DoubleArray, val hist: DoubleArray)

    /** MACD,hist 采用国内习惯的 2*(DIF-DEA) */
    fun macd(close: DoubleArray, fast: Int = 12, slow: Int = 26, signal: Int = 9): Macd {
        val ef = ema(close, fast)
        val es = ema(close, slow)
        val dif = DoubleArray(close.size) { ef[it] - es[it] }
        val dea = ema(dif, signal)
        val hist = DoubleArray(close.size) { 2 * (dif[it] - dea[it]) }
        return Macd(dif, dea, hist)
    }

    class Boll(
        val mid: DoubleArray,
        val upper: DoubleArray,
        val lower: DoubleArray,
        /** 布林百分比 %b = (C - 下轨) / (上轨 - 下轨) */
        val percentB: DoubleArray,
        /** 带宽 (上轨 - 下轨) / 中轨 */
        val width: DoubleArray,
    )

    fun boll(close: DoubleArray, n: Int = 20, k: Double = 2.0): Boll {
        val mid = sma(close, n)
        val up = DoubleArray(close.size) { Double.NaN }
        val dn = DoubleArray(close.size) { Double.NaN }
        val pb = DoubleArray(close.size) { Double.NaN }
        val w = DoubleArray(close.size) { Double.NaN }
        for (i in n - 1 until close.size) {
            var s = 0.0
            for (j in i - n + 1..i) {
                val d = close[j] - mid[i]
                s += d * d
            }
            val sd = sqrt(s / n)
            up[i] = mid[i] + k * sd
            dn[i] = mid[i] - k * sd
            val band = up[i] - dn[i]
            pb[i] = if (band > 1e-12) (close[i] - dn[i]) / band else 0.5
            w[i] = if (mid[i] > 0) band / mid[i] else 0.0
        }
        return Boll(mid, up, dn, pb, w)
    }

    /** RSI,通达信算法(SMA 平滑) */
    fun rsi(close: DoubleArray, n: Int = 6): DoubleArray {
        val out = DoubleArray(close.size) { Double.NaN }
        if (close.size < 2) return out
        val up = DoubleArray(close.size)
        val all = DoubleArray(close.size)
        for (i in 1 until close.size) {
            val d = close[i] - close[i - 1]
            up[i] = max(d, 0.0)
            all[i] = abs(d)
        }
        val su = tdxSma(up, n, 1)
        val sa = tdxSma(all, n, 1)
        for (i in 1 until close.size) out[i] = if (sa[i] > 1e-12) 100 * su[i] / sa[i] else 50.0
        return out
    }

    /** 真实波幅均值 ATR */
    fun atr(bars: List<Bar>, n: Int = 14): DoubleArray {
        val tr = DoubleArray(bars.size)
        for (i in bars.indices) {
            val b = bars[i]
            tr[i] = if (i == 0) b.high - b.low else {
                val pc = bars[i - 1].close
                maxOf(b.high - b.low, abs(b.high - pc), abs(b.low - pc))
            }
        }
        return tdxSma(tr, n, 1)
    }

    /**
     * 分时均价线(VWAP)。只在同一交易日内累计。
     * 用典型价 (H+L+C)/3 加权成交量,避免不同数据源 amount/volume 单位不一致的问题。
     */
    fun vwap(bars: List<Bar>): DoubleArray {
        val out = DoubleArray(bars.size)
        var pv = 0.0
        var v = 0.0
        var day = Int.MIN_VALUE
        for (i in bars.indices) {
            val b = bars[i]
            if (b.date != day) {
                day = b.date; pv = 0.0; v = 0.0
            }
            val tp = (b.high + b.low + b.close) / 3
            pv += tp * b.volume
            v += b.volume
            out[i] = if (v > 0) pv / v else tp
        }
        return out
    }

    /** 能量潮 OBV */
    fun obv(bars: List<Bar>): DoubleArray {
        val out = DoubleArray(bars.size)
        for (i in 1 until bars.size) {
            val d = bars[i].close - bars[i - 1].close
            out[i] = out[i - 1] + when {
                d > 0 -> bars[i].volume
                d < 0 -> -bars[i].volume
                else -> 0.0
            }
        }
        return out
    }

    /** 蔡金资金流 CMF(n) */
    fun cmf(bars: List<Bar>, n: Int = 20): DoubleArray {
        val out = DoubleArray(bars.size) { Double.NaN }
        val mfv = DoubleArray(bars.size) { i ->
            val b = bars[i]
            val r = b.high - b.low
            if (r <= 1e-12) 0.0 else ((b.close - b.low) - (b.high - b.close)) / r * b.volume
        }
        for (i in n - 1 until bars.size) {
            var s = 0.0
            var v = 0.0
            for (j in i - n + 1..i) {
                s += mfv[j]; v += bars[j].volume
            }
            out[i] = if (v > 0) s / v else 0.0
        }
        return out
    }

    /** 简单线性回归斜率(按每根 bar 计),用于判断趋势方向 */
    fun slope(x: DoubleArray, from: Int, to: Int): Double {
        val n = to - from + 1
        if (n < 2) return 0.0
        var sx = 0.0; var sy = 0.0; var sxy = 0.0; var sxx = 0.0
        for (i in from..to) {
            val t = (i - from).toDouble()
            sx += t; sy += x[i]; sxy += t * x[i]; sxx += t * t
        }
        val den = n * sxx - sx * sx
        return if (abs(den) < 1e-12) 0.0 else (n * sxy - sx * sy) / den
    }

    /** x[i] 在 x[from..to] 中的百分位(0..1) */
    fun percentile(x: DoubleArray, from: Int, to: Int, value: Double): Double {
        var cnt = 0; var tot = 0
        for (i in from..to) {
            if (x[i].isNaN()) continue
            tot++
            if (x[i] <= value) cnt++
        }
        return if (tot == 0) 0.5 else cnt.toDouble() / tot
    }
}

fun List<Bar>.closes(): DoubleArray = DoubleArray(size) { this[it].close }
fun List<Bar>.volumes(): DoubleArray = DoubleArray(size) { this[it].volume }

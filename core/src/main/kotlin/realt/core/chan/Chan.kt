package realt.core.chan

import realt.core.indicator.Indicators
import realt.core.indicator.closes
import realt.core.model.Bar
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min

/** 经过包含处理后的K线 */
data class MergedBar(
    val high: Double,
    val low: Double,
    /** 原始 bar 起止下标(含) */
    val start: Int,
    val end: Int,
    /** 高点/低点所在原始 bar 下标 */
    val highIdx: Int,
    val lowIdx: Int,
)

enum class FractalType { TOP, BOTTOM }

/** 笔的端点(确认过的分型) */
data class Pivot(
    val type: FractalType,
    /** 合并K线下标 */
    val mIdx: Int,
    /** 原始 bar 下标 */
    val idx: Int,
    val price: Double,
)

/** 一笔:从 [from] 到 [to] */
data class Stroke(val from: Pivot, val to: Pivot) {
    val up: Boolean get() = to.type == FractalType.TOP
    val high: Double get() = max(from.price, to.price)
    val low: Double get() = min(from.price, to.price)
}

/** 中枢:至少三笔重叠区间 [zd, zg] */
data class Center(
    val zg: Double,
    val zd: Double,
    val gg: Double,
    val dd: Double,
    val firstStroke: Int,
    val lastStroke: Int,
)

enum class ChanPointType(val buy: Boolean, val label: String) {
    BUY1(true, "一买"), BUY2(true, "二买"), BUY3(true, "三买"),
    SELL1(false, "一卖"), SELL2(false, "二卖"), SELL3(false, "三卖"),
}

/** 买卖点 */
data class ChanPoint(
    val type: ChanPointType,
    /** 对应分型在原始 bar 中的下标 */
    val idx: Int,
    val price: Double,
    /** 0..1,背驰/结构的力度 */
    val strength: Double,
    val note: String,
)

class ChanResult(
    val merged: List<MergedBar>,
    val pivots: List<Pivot>,
    val strokes: List<Stroke>,
    val centers: List<Center>,
    /**
     * 由"最新一根合并K线"刚刚确认的买卖点(实时信号)。
     * 只有当分型恰好在本 bar 被确认时才非空,避免同一信号重复触发。
     */
    val freshPoint: ChanPoint?,
) {
    val lastCenter: Center? get() = centers.lastOrNull()
}

/**
 * 缠论结构识别:包含处理 → 分型 → 笔 → 中枢 → 背驰 → 一二三类买卖点。
 *
 * 这是一个"可实时"的简化实现:每次对完整前缀重新计算,结果不含未来数据。
 * 笔的定义采用新笔规则:顶底分型之间至少 [minGap] 根合并K线(含分型自身共 5 根)。
 */
class ChanAnalyzer(
    private val minGap: Int = 4,
    /** 背驰判定:离开段 MACD 面积 / 进入段面积 < 该值视为背驰 */
    private val divergenceRatio: Double = 0.85,
) {

    fun analyze(bars: List<Bar>): ChanResult {
        val merged = merge(bars)
        val pivots = strokesPivots(merged)
        val strokes = pivots.zipWithNext { a, b -> Stroke(a, b) }
        val centers = centers(strokes)
        val fresh = freshPoint(bars, merged, pivots, strokes, centers)
        return ChanResult(merged, pivots, strokes, centers, fresh)
    }

    // ---------- 1. 包含处理 ----------
    fun merge(bars: List<Bar>): List<MergedBar> {
        val out = ArrayList<MergedBar>(bars.size)
        for (i in bars.indices) {
            val b = bars[i]
            if (out.isEmpty()) {
                out += MergedBar(b.high, b.low, i, i, i, i); continue
            }
            val last = out.last()
            val contains = (b.high <= last.high && b.low >= last.low) || (b.high >= last.high && b.low <= last.low)
            if (!contains) {
                out += MergedBar(b.high, b.low, i, i, i, i); continue
            }
            // 方向由前一根合并K线决定;只有一根时默认向上
            val up = if (out.size >= 2) last.high > out[out.size - 2].high else true
            val nh: Double; val nl: Double
            if (up) {
                nh = max(last.high, b.high); nl = max(last.low, b.low)
            } else {
                nh = min(last.high, b.high); nl = min(last.low, b.low)
            }
            val hi = if (b.high >= last.high) i else last.highIdx
            val lo = if (b.low <= last.low) i else last.lowIdx
            out[out.size - 1] = MergedBar(nh, nl, last.start, i, hi, lo)
        }
        return out
    }

    // ---------- 2/3. 分型与笔 ----------
    private fun strokesPivots(m: List<MergedBar>): List<Pivot> {
        val pivots = ArrayList<Pivot>()
        for (k in 1 until m.size - 1) {
            val a = m[k - 1]; val b = m[k]; val c = m[k + 1]
            val top = b.high > a.high && b.high > c.high
            val bottom = b.low < a.low && b.low < c.low
            if (!top && !bottom) continue
            val p = if (top) Pivot(FractalType.TOP, k, b.highIdx, b.high)
            else Pivot(FractalType.BOTTOM, k, b.lowIdx, b.low)
            if (pivots.isEmpty()) {
                pivots += p; continue
            }
            val last = pivots.last()
            if (p.type == last.type) {
                // 同向分型:取更极端者延伸笔
                val better = if (p.type == FractalType.TOP) p.price > last.price else p.price < last.price
                if (better) pivots[pivots.size - 1] = p
            } else {
                val gapOk = p.mIdx - last.mIdx >= minGap
                val priceOk = if (p.type == FractalType.TOP) p.price > last.price else p.price < last.price
                if (gapOk && priceOk) {
                    pivots += p
                } else if (pivots.size >= 2) {
                    // 不成笔,但若突破了上一笔起点,则上一笔被破坏,改为延伸更早的那一笔
                    val prev = pivots[pivots.size - 2]
                    val breaks = if (p.type == FractalType.TOP) p.price > prev.price else p.price < prev.price
                    if (breaks) {
                        pivots.removeAt(pivots.size - 1)
                        pivots[pivots.size - 1] = p
                    }
                }
            }
        }
        return pivots
    }

    // ---------- 4. 中枢 ----------
    private fun centers(s: List<Stroke>): List<Center> {
        val out = ArrayList<Center>()
        var i = 0
        while (i + 2 < s.size) {
            val zg = minOf(s[i].high, s[i + 1].high, s[i + 2].high)
            val zd = maxOf(s[i].low, s[i + 1].low, s[i + 2].low)
            if (zg <= zd) {
                i++; continue
            }
            var gg = maxOf(s[i].high, s[i + 1].high, s[i + 2].high)
            var dd = minOf(s[i].low, s[i + 1].low, s[i + 2].low)
            var j = i + 3
            while (j < s.size && s[j].low < zg && s[j].high > zd) {
                gg = max(gg, s[j].high); dd = min(dd, s[j].low); j++
            }
            out += Center(zg, zd, gg, dd, i, j - 1)
            i = j
        }
        return out
    }

    // ---------- 5/6. 背驰与买卖点 ----------
    private fun freshPoint(
        bars: List<Bar>,
        m: List<MergedBar>,
        pivots: List<Pivot>,
        strokes: List<Stroke>,
        centers: List<Center>,
    ): ChanPoint? {
        if (pivots.size < 3 || m.size < 3) return null
        val last = pivots.last()
        // 仅当分型由最新一根合并K线确认时才视为"刚出现"的信号
        if (last.mIdx != m.size - 2) return null
        val hist = Indicators.macd(bars.closes()).hist
        val bottom = last.type == FractalType.BOTTOM

        // 当前这一笔(以 last 结束)与上一根同向笔
        val cur = strokes.last()
        val prevSame = if (strokes.size >= 3) strokes[strokes.size - 3] else null
        val center = centers.lastOrNull()

        if (prevSame != null) {
            val newExtreme = if (bottom) cur.to.price < prevSame.to.price else cur.to.price > prevSame.to.price
            if (newExtreme) {
                val aCur = area(hist, cur, bottom)
                val aPrev = area(hist, prevSame, bottom)
                val ratio = if (aPrev > 1e-12) aCur / aPrev else 1.0
                val belowCenter = center == null ||
                    (if (bottom) cur.to.price < center.zd else cur.to.price > center.zg)
                if (ratio < divergenceRatio && belowCenter) {
                    val strength = (1 - ratio).coerceIn(0.0, 1.0) * 0.6 + 0.4
                    return ChanPoint(
                        if (bottom) ChanPointType.BUY1 else ChanPointType.SELL1,
                        last.idx, last.price, strength,
                        "背驰 面积比 %.2f".format(ratio)
                    )
                }
            } else {
                // 未创新低/新高:二类买卖点(回抽不破前极值)
                val prior = prevSame.to
                val trendBefore = if (pivots.size >= 5) {
                    val p5 = pivots[pivots.size - 5]
                    if (bottom) p5.price > prior.price else p5.price < prior.price
                } else true
                if (trendBefore) {
                    val margin = abs(cur.to.price - prior.price) / prior.price
                    return ChanPoint(
                        if (bottom) ChanPointType.BUY2 else ChanPointType.SELL2,
                        last.idx, last.price, (0.5 + margin * 50).coerceAtMost(0.8),
                        if (bottom) "回踩不破前低" else "反抽不过前高"
                    )
                }
            }
        }

        // 三类买卖点:离开中枢后回抽不回到中枢
        if (center != null && center.lastStroke <= strokes.size - 3) {
            val leave = strokes[strokes.size - 2]
            if (bottom && leave.up && leave.high > center.gg && cur.to.price > center.zg) {
                return ChanPoint(ChanPointType.BUY3, last.idx, last.price, 0.7, "回踩不进中枢 ZG=%.2f".format(center.zg))
            }
            if (!bottom && !leave.up && leave.low < center.dd && cur.to.price < center.zd) {
                return ChanPoint(ChanPointType.SELL3, last.idx, last.price, 0.7, "反抽不进中枢 ZD=%.2f".format(center.zd))
            }
        }
        return null
    }

    /** 一笔区间内同向 MACD 柱面积(向下笔取绿柱、向上笔取红柱) */
    private fun area(hist: DoubleArray, s: Stroke, down: Boolean): Double {
        var a = 0.0
        val from = min(s.from.idx, s.to.idx)
        val to = min(max(s.from.idx, s.to.idx), hist.size - 1)
        for (i in from..to) {
            val h = hist[i]
            if (down && h < 0) a -= h
            if (!down && h > 0) a += h
        }
        return a
    }
}

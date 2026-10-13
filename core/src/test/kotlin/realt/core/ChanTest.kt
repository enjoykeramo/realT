package realt.core

import realt.core.chan.ChanAnalyzer
import realt.core.chan.ChanPointType
import realt.core.chan.FractalType
import realt.core.model.Bar
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class ChanTest {
    /** 用一串收盘价生成 K 线,高低点为收盘价 ± 0.1 */
    private fun bars(px: List<Double>) = px.mapIndexed { i, p ->
        Bar(20250102, 571 + i, p, p + 0.1, p - 0.1, p, 100.0, 100.0 * p)
    }

    private fun zigzag(vararg legs: Pair<Double, Int>): List<Double> {
        val out = ArrayList<Double>()
        var p = legs.first().first
        out += p
        for ((target, steps) in legs.drop(1)) {
            val d = (target - p) / steps
            repeat(steps) { p += d; out += p }
        }
        return out
    }

    @Test
    fun inclusionMergesContainedBars() {
        val b = listOf(
            Bar(1, 571, 10.0, 11.0, 9.0, 10.0, 1.0, 1.0),
            Bar(1, 572, 10.0, 10.5, 9.5, 10.0, 1.0, 1.0), // 被包含
            Bar(1, 573, 10.0, 12.0, 10.0, 11.0, 1.0, 1.0),
        )
        val m = ChanAnalyzer().merge(b)
        assertEquals(2, m.size)
        assertEquals(0, m[0].start); assertEquals(1, m[0].end)
    }

    @Test
    fun zigzagProducesAlternatingStrokes() {
        val px = zigzag(10.0 to 0, 12.0 to 10, 10.5 to 10, 13.0 to 10, 11.0 to 10, 14.0 to 10, 13.0 to 3)
        val r = ChanAnalyzer().analyze(bars(px))
        assertTrue(r.pivots.size >= 4, "pivots=${r.pivots}")
        for (i in 1 until r.pivots.size) assertTrue(r.pivots[i].type != r.pivots[i - 1].type)
        assertEquals(FractalType.TOP, r.pivots.first { it.price > 11.9 }.type)
    }

    @Test
    fun centerFromOverlappingStrokes() {
        val px = zigzag(10.0 to 0, 11.0 to 8, 10.2 to 8, 10.9 to 8, 10.3 to 8, 11.5 to 8, 11.2 to 3)
        val r = ChanAnalyzer().analyze(bars(px))
        assertTrue(r.centers.isNotEmpty(), "strokes=${r.strokes.size}")
        val c = r.centers.first()
        assertTrue(c.zg > c.zd)
    }

    @Test
    fun bottomDivergenceGivesBuy1() {
        // 第二段下跌创新低但跌速更慢(MACD 面积更小) → 一买
        val px = zigzag(
            14.0 to 0, 15.0 to 5, 12.0 to 6, 12.8 to 8, 11.6 to 12, 11.9 to 4,
        )
        val analyzer = ChanAnalyzer()
        val all = bars(px)
        // 逐 bar 扫描,记录出现过的实时买卖点
        val seen = (10..all.size).mapNotNull { analyzer.analyze(all.subList(0, it)).freshPoint }
        assertTrue(seen.any { it.type == ChanPointType.BUY1 }, "seen=$seen")
    }
}

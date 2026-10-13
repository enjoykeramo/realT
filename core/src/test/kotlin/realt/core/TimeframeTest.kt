package realt.core

import realt.core.indicator.Indicators
import realt.core.indicator.closes
import realt.core.model.Bar
import realt.core.model.Session
import realt.core.model.aggregate
import realt.core.signal.Timeframe
import kotlin.math.abs
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertTrue

class TimeframeTest {
    private fun bars(days: Int): List<Bar> {
        val rnd = Random(3)
        var p = 10.0
        val out = ArrayList<Bar>()
        for (d in 0 until days) for (i in 1..240) {
            val m = if (i <= 120) Session.OPEN + i else Session.AFTERNOON_START + i - 120
            val o = p
            p *= 1 + (rnd.nextDouble() - 0.5) * 0.004
            out += Bar(20250101 + d, m, o, maxOf(o, p) * 1.001, minOf(o, p) * 0.999, p, 100.0 + rnd.nextInt(100), 0.0)
        }
        return out
    }

    private fun close(a: Double, b: Double) = assertTrue(abs(a - b) < 1e-9 || (a.isNaN() && b.isNaN()), "$a != $b")

    @Test
    fun incrementalMatchesBatchOnEveryPrefix() {
        val all = bars(2)
        for (period in listOf(1, 5, 15)) {
            val tf = Timeframe(period)
            for ((i, b) in all.withIndex()) {
                tf.update(b)
                if (i % 7 != 0 && i != all.size - 1) continue
                val agg = all.subList(0, i + 1).aggregate(period)
                val c = agg.closes()
                val k = c.size - 1
                val m = Indicators.macd(c)
                val (d, de, h) = tf.macdNow()
                close(d, m.dif[k]); close(de, m.dea[k]); close(h, m.hist[k])
                if (k >= 1) close(tf.rsiNow(), Indicators.rsi(c, 6)[k])
                val bb = Indicators.boll(c, 20, 2.0)
                close(tf.boll().first, bb.percentB[k])
                if (k >= 1) close(tf.boll(1).first, bb.percentB[k - 1])
            }
        }
    }
}

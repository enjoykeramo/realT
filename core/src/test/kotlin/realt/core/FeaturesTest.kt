package realt.core

import realt.core.daytype.DayContext
import realt.core.engine.Observer
import realt.core.engine.TDirection
import realt.core.ml.Features
import realt.core.model.Bar
import realt.core.model.Session
import realt.core.regime.MarketRegime
import realt.core.stage.StageEstimate
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class FeaturesTest {
    private fun day(date: Int, base: Double) = (1..240).map { i ->
        val m = if (i <= 120) Session.OPEN + i else Session.AFTERNOON_START + i - 120
        val p = base * (1 + 0.01 * kotlin.math.sin(i / 15.0))
        Bar(date, m, p, p * 1.001, p * 0.999, p, 1000.0 + i, 1000.0 * p)
    }

    @Test
    fun featureVectorMatchesNamesAndIndexIsCausal() {
        val ctx = DayContext(10.0, 0.03, MarketRegime.UNKNOWN, StageEstimate.UNKNOWN)
        val idx = day(20250102, 3000.0)
        val obsr = Observer(ctx, day(20250101, 10.0), indexBars = idx, indexPrevClose = 2990.0)
        val bars = day(20250102, 10.0)
        var last = obsr.observe(bars[0])
        for (b in bars.drop(1).take(59)) last = obsr.observe(b)
        for (d in TDirection.entries) {
            val x = Features.extract(ctx, last, d)
            assertEquals(Features.names.size, x.size)
            assertTrue(x.all { !it.isNaN() && !it.isInfinite() }, "NaN in ${Features.names.zip(x.toList())}")
        }
        // 指数状态只读到第 60 分钟,不看未来
        assertTrue(last.day.hasIndex)
        assertEquals((idx[59].close / 2990.0 - 1) * 100, last.day.idxRet, 1e-9)
    }
}

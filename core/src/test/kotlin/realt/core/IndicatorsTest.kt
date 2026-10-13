package realt.core

import realt.core.indicator.Indicators
import realt.core.model.Bar
import realt.core.model.Session
import realt.core.model.aggregate
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class IndicatorsTest {
    private fun near(a: Double, b: Double, eps: Double = 1e-9) = assertTrue(abs(a - b) < eps, "$a != $b")

    @Test
    fun smaAndEma() {
        val x = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val s = Indicators.sma(x, 3)
        assertTrue(s[1].isNaN())
        near(s[2], 2.0); near(s[4], 4.0)
        val e = Indicators.ema(x, 3) // k = 0.5
        near(e[0], 1.0); near(e[1], 1.5); near(e[2], 2.25)
    }

    @Test
    fun macdOfConstantIsZero() {
        val m = Indicators.macd(DoubleArray(100) { 10.0 })
        near(m.dif.last(), 0.0); near(m.hist.last(), 0.0)
    }

    @Test
    fun bollPercentB() {
        val x = DoubleArray(20) { if (it % 2 == 0) 9.0 else 11.0 }
        val b = Indicators.boll(x, 20, 2.0)
        near(b.mid[19], 10.0)
        near(b.upper[19], 12.0); near(b.lower[19], 8.0)
        near(b.percentB[19], (11.0 - 8.0) / 4.0)
    }

    @Test
    fun rsiBounds() {
        val up = Indicators.rsi(DoubleArray(30) { it.toDouble() }, 6)
        near(up.last(), 100.0)
        val down = Indicators.rsi(DoubleArray(30) { -it.toDouble() }, 6)
        near(down.last(), 0.0)
    }

    @Test
    fun vwapResetsEachDay() {
        val bars = listOf(
            Bar(20250101, 571, 10.0, 10.0, 10.0, 10.0, 100.0, 1000.0),
            Bar(20250101, 572, 12.0, 12.0, 12.0, 12.0, 100.0, 1200.0),
            Bar(20250102, 571, 20.0, 20.0, 20.0, 20.0, 100.0, 2000.0),
        )
        val v = Indicators.vwap(bars)
        near(v[1], 11.0); near(v[2], 20.0)
    }

    @Test
    fun aggregateFiveMinutes() {
        val day = (1..240).map { i ->
            val m = if (i <= 120) Session.OPEN + i else Session.AFTERNOON_START + i - 120
            Bar(20250102, m, i.toDouble(), i + 0.5, i - 0.5, i + 0.2, 10.0, 100.0)
        }
        val five = day.aggregate(5)
        assertEquals(48, five.size)
        assertEquals(Session.OPEN + 5, five[0].minute)
        near(five[0].open, 1.0); near(five[0].close, 5.2); near(five[0].high, 5.5); near(five[0].low, 0.5)
        near(five[0].volume, 50.0)
        assertEquals(Session.AFTERNOON_START + 5, five[24].minute)
        assertEquals(Session.CLOSE, five.last().minute)
    }

    @Test
    fun sessionElapsed() {
        assertEquals(1, Session.elapsed(Session.OPEN + 1))
        assertEquals(120, Session.elapsed(Session.MORNING_END))
        assertEquals(121, Session.elapsed(Session.AFTERNOON_START + 1))
        assertEquals(240, Session.elapsed(Session.CLOSE))
    }
}

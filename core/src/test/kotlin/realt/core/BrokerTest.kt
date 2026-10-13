package realt.core

import realt.core.engine.ActionKind
import realt.core.engine.CloseReason
import realt.core.engine.Fills
import realt.core.engine.TAction
import realt.core.ml.Calibration
import realt.core.model.Bar
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNull
import kotlin.test.assertTrue

class BrokerTest {
    private fun bar(o: Double, h: Double, l: Double, c: Double) = Bar(20250102, 600, o, h, l, c, 100.0, 0.0)
    private fun act(kind: ActionKind, limit: Double?) = TAction(kind, 20250102, 599, 10.0, 100, "t", limitPrice = limit)

    @Test
    fun marketOrderFillsAtOpenWithSlippage() {
        assertEquals(10.0 * 1.0002, Fills.tryFill(act(ActionKind.OPEN_BUY, null), bar(10.0, 10.1, 9.9, 10.0), 0.0002)!!, 1e-12)
        assertEquals(10.0 * 0.9998, Fills.tryFill(act(ActionKind.OPEN_SELL, null), bar(10.0, 10.1, 9.9, 10.0), 0.0002)!!, 1e-12)
    }

    @Test
    fun limitSellNeedsPriceToTradeThrough() {
        val a = act(ActionKind.OPEN_SELL, 10.20)
        assertNull(Fills.tryFill(a, bar(10.0, 10.20, 9.9, 10.1), 0.0002))          // 仅触及,不算成交(排队)
        assertEquals(10.20, Fills.tryFill(a, bar(10.0, 10.21, 9.9, 10.1), 0.0002)!!, 1e-12)
        assertEquals(10.30, Fills.tryFill(a, bar(10.30, 10.4, 10.2, 10.3), 0.0002)!!, 1e-12) // 跳空高开按开盘价
    }

    @Test
    fun limitBuySymmetric() {
        val a = act(ActionKind.CLOSE_BUY, 9.80)
        assertNull(Fills.tryFill(a, bar(10.0, 10.1, 9.80, 9.9), 0.0002))
        assertEquals(9.80, Fills.tryFill(a, bar(10.0, 10.1, 9.79, 9.9), 0.0002)!!, 1e-12)
    }

    @Test
    fun tickRounding() {
        assertEquals(10.12, Fills.floorTick(10.1299), 1e-9)
        assertEquals(10.13, Fills.ceilTick(10.1201), 1e-9)
        assertEquals(10.12, Fills.ceilTick(10.12), 1e-9)
        assertTrue(CloseReason.entries.isNotEmpty())
    }

    @Test
    fun calibrationIsMonotoneAndMatchesFrequencies() {
        val rnd = Random(5)
        val s = DoubleArray(5000) { rnd.nextDouble() }
        // 真实胜率 = 0.5 + 0.4 * 分数
        val y = IntArray(s.size) { if (rnd.nextDouble() < 0.5 + 0.4 * s[it]) 1 else 0 }
        val cal = Calibration.fit(s, y)
        var prev = -1.0
        for (i in 0..100) {
            val v = cal.apply(i / 100.0)
            assertTrue(v >= prev - 1e-12); prev = v
        }
        assertTrue(kotlin.math.abs(cal.apply(0.9) - 0.86) < 0.06, "cal(0.9)=${cal.apply(0.9)}")
        val back = Calibration.parse(cal.serialize())
        assertEquals(cal.apply(0.42), back.apply(0.42), 1e-12)
    }
}

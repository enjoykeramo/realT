package realt.core

import realt.core.engine.Costs
import realt.core.engine.TConfig
import realt.core.ml.Features
import realt.core.ml.Gbdt
import realt.core.ml.TradeModel
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class GbdtTest {

    @Test
    fun learnsNonLinearRule() {
        // 标签 = (x0 > 0.5 且 x1 < 0.3) 或 x2 > 0.9,带 5% 噪声
        val rnd = Random(1)
        fun sample(n: Int): Pair<List<DoubleArray>, IntArray> {
            val x = List(n) { DoubleArray(5) { rnd.nextDouble() } }
            val y = IntArray(n) {
                val v = (x[it][0] > 0.5 && x[it][1] < 0.3) || x[it][2] > 0.9
                if (rnd.nextDouble() < 0.05) (if (v) 0 else 1) else (if (v) 1 else 0)
            }
            return x to y
        }
        val (tx, ty) = sample(4000)
        val (vx, vy) = sample(1000)
        val m = Gbdt.train(tx, ty, vx, vy)
        val acc = vx.indices.count { (m.predict(vx[it]) > 0.5) == (vy[it] == 1) }.toDouble() / vx.size
        assertTrue(acc > 0.88, "accuracy=$acc")
        // 重要性应集中在 x0/x1/x2
        val imp = m.importance
        assertTrue(imp[0] + imp[1] + imp[2] > 0.9 * imp.sum())

        val back = Gbdt.deserialize(m.serialize())
        for (x in vx.take(50)) assertEquals(m.predict(x), back.predict(x), 1e-12)
    }

    @Test
    fun tradeModelRoundTrip() {
        val rnd = Random(2)
        val x = List(300) { DoubleArray(Features.size) { rnd.nextDouble() } }
        val y = IntArray(300) { if (x[it][3] > 0.5) 1 else 0 }
        val g = Gbdt.train(x, y, p = Gbdt.Params(nTrees = 20))
        val cfg = TConfig(takeProfitAtr = 0.3, stopLossAtr = 1.2)
        val costs = Costs(commissionRate = 0.0001, minCommission = 0.0)
        val m = TradeModel(cfg, costs, 0.83, g, 20250101, 20250630, mapOf("testWinRate" to "0.81"))
        val text = m.serialize()
        val back = TradeModel.parse(text)
        assertEquals(0.83, back.threshold)
        assertEquals(cfg, back.config)
        assertEquals(costs, back.costs)
        assertEquals(20250630, back.trainEnd)
        assertEquals("0.81", back.meta["testWinRate"])
        assertEquals(m.winProb(x[0]), back.winProb(x[0]), 1e-12)
        // Windows 换行也能读
        assertEquals(0.83, TradeModel.parse(text.replace("\n", "\r\n")).threshold)
    }
}

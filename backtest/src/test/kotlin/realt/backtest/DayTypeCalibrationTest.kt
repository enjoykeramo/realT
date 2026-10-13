package realt.backtest

import realt.core.daytype.DayContext
import realt.core.daytype.DayType
import realt.core.daytype.DayTypeLabeler
import realt.core.daytype.DayTypePredictor
import realt.core.daytype.top
import realt.core.model.Session
import realt.core.regime.MarketRegime
import realt.core.stage.StageEstimate
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * 用合成的"干净"形态检验标注器与预判器:形态清晰时两者都应能大概率认出来。
 */
class DayTypeCalibrationTest {
    private val gen = SyntheticMarket()
    private val vol = 0.025
    private val ctx = DayContext(10.0, vol, MarketRegime.UNKNOWN, StageEstimate.UNKNOWN)

    private fun confusion(f: (DayType, Random) -> DayType): Map<DayType, Map<DayType, Int>> {
        val rnd = Random(1)
        return DayType.entries.associateWith { t ->
            val m = HashMap<DayType, Int>()
            repeat(60) { val p = f(t, rnd); m[p] = (m[p] ?: 0) + 1 }
            m
        }
    }

    private fun print(name: String, c: Map<DayType, Map<DayType, Int>>) {
        println(name)
        for ((t, m) in c) println("  ${t.label}: " + DayType.entries.joinToString(" ") { "${it.label}=${m[it] ?: 0}" })
    }

    @Test
    fun labelerRecognisesCleanShapes() {
        val c = confusion { t, rnd -> DayTypeLabeler.label(gen.intraday(rnd, 20250102, 10.0, vol, t), 10.0, vol) }
        print("标注器", c)
        val acc = c.entries.sumOf { (t, m) -> m[t] ?: 0 }.toDouble() / (60 * DayType.entries.size)
        assertTrue(acc > 0.7, "标注器准确率过低: $acc")
    }

    @Test
    fun predictorImprovesThroughTheDay() {
        val p = DayTypePredictor()
        fun acc(minutes: Int): Double {
            val c = confusion { t, rnd ->
                val day = gen.intraday(rnd, 20250102, 10.0, vol, t)
                val upto = day.filter { Session.elapsed(it.minute) <= minutes }
                p.posterior(p.prior(ctx, day.first().open), ctx, upto).top()
            }
            print("预判 @${minutes}min", c)
            return c.entries.sumOf { (t, m) -> m[t] ?: 0 }.toDouble() / (60 * DayType.entries.size)
        }
        val a60 = acc(60)
        val a120 = acc(120)
        val a180 = acc(180)
        println("准确率 60min=$a60 120min=$a120 180min=$a180")
        assertTrue(a120 > 0.45, "午盘预判准确率过低: $a120")
        assertTrue(a180 >= a60, "预判应随时间改善")
    }
}

package realt.backtest

import realt.core.daytype.DayType
import realt.core.model.Bar
import realt.core.model.Session
import realt.core.model.Symbol
import realt.tdx.TdxReader
import realt.tdx.TdxWriter
import java.io.File
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt
import kotlin.random.Random

/**
 * 合成行情:生成一个通达信格式的 vipdoc 目录,用于单元测试和无数据时体验流程。
 * 每天随机选择一种分时形态,按形态的漂移轮廓 + 随机噪声生成 240 根 1 分钟线。
 *
 * 注意:合成数据上的回测结果没有任何实盘参考意义,只用来验证程序链路正确。
 */
class SyntheticMarket(private val seed: Long = 42) {

    fun generate(vipdoc: File, stocks: Int = 8, days: Int = 320, minuteDays: Int = 60) {
        val rnd = Random(seed)
        val dates = tradingDates(20240102, days)
        // 指数
        val index = dailyWalk(rnd, dates, 3000.0, 0.011)
        TdxWriter.writeDaily(File(vipdoc, "sh/lday/sh000001.day"), TdxReader.SH_INDEX, index)

        for (k in 0 until stocks) {
            val sym = if (k % 2 == 0) Symbol("sh", "60%04d".format(1000 + k)) else Symbol("sz", "00%04d".format(2000 + k))
            val vol = 0.018 + rnd.nextDouble() * 0.015
            val daily = ArrayList(dailyWalk(rnd, dates.dropLast(minuteDays), 10.0 + rnd.nextDouble() * 30, vol))
            val minutes = ArrayList<Bar>()
            var prev = daily.last().close
            for (d in dates.takeLast(minuteDays)) {
                val type = DayType.entries[rnd.nextInt(DayType.entries.size)]
                val day = intraday(rnd, d, prev, vol, type)
                minutes += day
                daily += Bar(d, -1, day.first().open, day.maxOf { it.high }, day.minOf { it.low }, day.last().close,
                    day.sumOf { it.volume }, day.sumOf { it.amount })
                prev = day.last().close
            }
            TdxWriter.writeDaily(File(vipdoc, "${sym.market}/lday/${sym.id}.day"), sym, daily)
            TdxWriter.writeMinute(File(vipdoc, "${sym.market}/minline/${sym.id}.lc1"), minutes)
        }
    }

    private fun tradingDates(start: Int, n: Int): List<Int> {
        val out = ArrayList<Int>()
        var y = start / 10000; var m = start / 100 % 100; var d = start % 100
        var dow = 2 // 2024-01-02 为周二
        while (out.size < n) {
            if (dow in 1..5) out += y * 10000 + m * 100 + d
            dow = (dow + 1) % 7
            d++
            val dim = when (m) { 2 -> if (y % 4 == 0) 29 else 28; 4, 6, 9, 11 -> 30; else -> 31 }
            if (d > dim) { d = 1; m++ }
            if (m > 12) { m = 1; y++ }
        }
        return out
    }

    private fun dailyWalk(rnd: Random, dates: List<Int>, start: Double, vol: Double): List<Bar> {
        var p = start
        var drift = 0.0
        return dates.map { d ->
            if (rnd.nextDouble() < 0.03) drift = (rnd.nextDouble() - 0.5) * vol * 0.6
            val o = p * (1 + gauss(rnd) * vol * 0.3)
            val c = o * exp(drift + gauss(rnd) * vol)
            val h = max(o, c) * (1 + rnd.nextDouble() * vol * 0.6)
            val l = min(o, c) * (1 - rnd.nextDouble() * vol * 0.6)
            p = c
            val v = 1e6 * (0.5 + rnd.nextDouble())
            Bar(d, -1, r2(o), r2(h), r2(l), r2(c), v, v * c)
        }
    }

    /** 按形态生成一天 240 根 1 分钟线 */
    fun intraday(rnd: Random, date: Int, prevClose: Double, vol: Double, type: DayType): List<Bar> {
        val n = Session.MINUTES_PER_DAY
        val amp = vol * when (type) { DayType.FLAT -> 0.3; DayType.RANGE -> 0.8; else -> 1.2 }
        // 形态漂移轮廓 f(t),t∈[0,1],单位为 amp
        val shape: (Double) -> Double = when (type) {
            DayType.TREND_UP -> { t -> t }
            DayType.TREND_DOWN -> { t -> -t }
            DayType.FADE -> { t -> if (t < 0.3) t / 0.3 * 0.8 else 0.8 - (t - 0.3) / 0.7 * 1.3 }
            DayType.RECOVER -> { t -> if (t < 0.3) -t / 0.3 * 0.8 else -0.8 + (t - 0.3) / 0.7 * 1.3 }
            DayType.RANGE -> { t -> 0.4 * kotlin.math.sin(t * 2 * Math.PI * 2) }
            DayType.FLAT -> { _ -> 0.0 }
        }
        val gap = gauss(rnd) * vol * 0.3
        val open = prevClose * (1 + gap)
        // 噪声 = 均值回复(OU,半衰期约 10 分钟,稳态标准差 0.25×amp)+ 小幅随机游走
        val theta = 1.0 / 15
        val ouSigma = 0.25 * amp * sqrt(2 * theta)
        val rwSigma = amp * 0.1 / sqrt(n.toDouble())
        val noise = amp * 0.12 / sqrt(n / 10.0)
        val out = ArrayList<Bar>(n)
        var last = open
        var ou = 0.0
        var walk = 0.0
        for (i in 1..n) {
            val minute = if (i <= 120) Session.OPEN + i else Session.AFTERNOON_START + (i - 120)
            ou += -theta * ou + ouSigma * gauss(rnd)
            walk += gauss(rnd) * rwSigma
            val target = open * (1 + amp * shape(i.toDouble() / n) + ou + walk)
            val o = last
            val c = target
            val h = max(o, c) * (1 + rnd.nextDouble() * noise * 0.5)
            val l = min(o, c) * (1 - rnd.nextDouble() * noise * 0.5)
            val u = if (i <= 10 || i > n - 10) 3.0 else 1.0
            val v = (2e4 + rnd.nextDouble() * 4e4) * u * 100
            out += Bar(date, minute, r2(o), r2(h), r2(l), r2(c), v, v * c)
            last = c
        }
        return out
    }

    private fun gauss(rnd: Random): Double {
        var u: Double
        do { u = rnd.nextDouble() } while (u <= 1e-12)
        return sqrt(-2 * kotlin.math.ln(u)) * kotlin.math.cos(2 * Math.PI * rnd.nextDouble())
    }

    private fun r2(x: Double) = Math.round(x * 100) / 100.0
}

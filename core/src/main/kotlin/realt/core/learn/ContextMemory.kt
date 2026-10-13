package realt.core.learn

import realt.core.engine.ContextKey

/**
 * L6:"迷宫记忆"。
 *
 * 把每次做T的上下文(大盘档位 × 主力阶段 × 分时形态 × 方向 × 信号家族)当作迷宫里的一个路口,
 * 记录走过这里的结果。新的一天走到同一路口时:
 *  - 样本不足 → 允许探索(试错)
 *  - 胜率的贝叶斯下界(Beta 后验分位)低于目标 → 视为死路,不再出手
 *
 * 回测中按时间顺序在线更新,当天只能使用此前已结束交易日的统计,不存在未来函数。
 * 统计可以按"粗粒度"回退:精确路口样本不足时,参考去掉信号家族/阶段后的上级路口。
 */
class ContextMemory(
    /** 判定死路所需的最少样本 */
    private val minSamples: Int = 8,
    /** 目标胜率 */
    private val targetWinRate: Double = 0.8,
    /** 胜率下界容忍:后验均值低于 target - tolerance 视为死路 */
    private val tolerance: Double = 0.15,
) {
    class Stat {
        var n = 0; var wins = 0; var pnl = 0.0
        val winRate: Double get() = if (n == 0) 0.0 else wins.toDouble() / n
        /** Beta(1+w, 1+l) 后验均值 */
        val posteriorMean: Double get() = (wins + 1.0) / (n + 2.0)
    }

    private val stats = HashMap<String, Stat>()

    private fun keys(k: ContextKey): List<String> = listOf(
        k.toString(),
        "${k.regime}|*|${k.dayType.name}|${k.direction.name}|*",
        "*|*|${k.dayType.name}|${k.direction.name}|*",
    )

    fun record(k: ContextKey, win: Boolean, pnl: Double) {
        for (key in keys(k)) {
            val s = stats.getOrPut(key) { Stat() }
            s.n++
            if (win) s.wins++
            s.pnl += pnl
        }
    }

    /** 是否允许在该路口出手 */
    fun allow(k: ContextKey): Boolean {
        for (key in keys(k)) {
            val s = stats[key] ?: continue
            if (s.n < minSamples) continue
            // 找到第一个样本充足的层级就按它判定
            return s.posteriorMean >= targetWinRate - tolerance && s.pnl >= 0
        }
        return true
    }

    fun snapshot(): Map<String, Stat> = stats.toSortedMap()

    fun toCsv(): String = buildString {
        appendLine("context,n,wins,winRate,posteriorMean,pnl")
        for ((k, s) in snapshot()) appendLine("$k,${s.n},${s.wins},%.3f,%.3f,%.2f".format(s.winRate, s.posteriorMean, s.pnl))
    }

    fun loadCsv(text: String) {
        text.lineSequence().drop(1).filter { it.isNotBlank() }.forEach { line ->
            val p = line.split(',')
            if (p.size >= 6) {
                val s = Stat()
                s.n = p[1].toInt(); s.wins = p[2].toInt(); s.pnl = p[5].toDouble()
                stats[p[0]] = s
            }
        }
    }
}

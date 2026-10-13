package realt.backtest

import java.io.PrintStream

/**
 * 单行进度条:同一行原地刷新(\r),例如
 *   [2/5 训练模型] ██████████░░░░░░░░░░  52%  25/48 | 离场组3 第2段 | 用时 01:12 剩余约 01:06
 * 输出被重定向到文件时(无控制台),改为每 10% 打印一行。线程安全。
 */
class Progress(
    private val out: PrintStream = System.out,
    private val interactive: Boolean = System.console() != null,
) {
    private var stage = ""
    private var total = 1
    private var startNs = 0L
    private var lastPrintNs = 0L
    private var lastWidth = 0
    private var lastDecile = -1
    private var active = false

    @Synchronized
    fun start(stage: String, total: Int) {
        if (active) finish()
        this.stage = stage
        this.total = maxOf(1, total)
        startNs = System.nanoTime()
        lastPrintNs = 0L
        lastDecile = -1
        active = true
        render(0, "", force = true)
    }

    @Synchronized
    fun step(done: Int, info: String = "") {
        if (!active) return
        render(done, info, force = done >= total)
    }

    /** 结束当前阶段并换行;[summary] 追加在进度条后 */
    @Synchronized
    fun finish(summary: String = "") {
        if (!active) return
        render(total, summary, force = true, final = true)
        active = false
    }

    /** 在进度条上方打印一行普通日志 */
    @Synchronized
    fun log(line: String) {
        if (active && interactive) {
            out.print("\r" + " ".repeat(lastWidth) + "\r")
            out.println(line)
            lastWidth = 0
        } else out.println(line)
    }

    private fun render(done: Int, info: String, force: Boolean, final: Boolean = false) {
        val now = System.nanoTime()
        val frac = (done.toDouble() / total).coerceIn(0.0, 1.0)
        if (!interactive) {
            val dec = (frac * 10).toInt()
            if (dec == lastDecile && !final) return
            lastDecile = dec
        } else if (!force && now - lastPrintNs < 200_000_000L) return
        lastPrintNs = now
        val elapsed = (now - startNs) / 1e9
        val eta = if (frac > 0.01 && frac < 1) elapsed / frac * (1 - frac) else 0.0
        val barLen = 20
        val filled = (frac * barLen).toInt()
        val bar = "█".repeat(filled) + "░".repeat(barLen - filled)
        val sb = StringBuilder()
        sb.append("[").append(stage).append("] ").append(bar).append(" %3d%%  %d/%d".format((frac * 100).toInt(), done, total))
        if (info.isNotEmpty()) sb.append(" | ").append(info)
        sb.append(" | 用时 ").append(fmt(elapsed))
        if (!final && eta > 0) sb.append(" 剩余约 ").append(fmt(eta))
        val line = sb.toString()
        val width = displayWidth(line)
        if (interactive) {
            out.print("\r" + line + " ".repeat(maxOf(0, lastWidth - width)))
            if (final) out.println()
            out.flush()
            lastWidth = if (final) 0 else width
        } else {
            out.println(line)
        }
    }

    private fun fmt(sec: Double): String {
        val s = sec.toLong()
        return if (s >= 3600) "%d:%02d:%02d".format(s / 3600, s / 60 % 60, s % 60) else "%02d:%02d".format(s / 60, s % 60)
    }

    /** 中文及方块字符按 2 个字符宽度计算(宁多勿少),保证覆盖上一次输出 */
    private fun displayWidth(s: String) = s.sumOf { if (it.code >= 0x2500) 2L else 1L }.toInt()
}

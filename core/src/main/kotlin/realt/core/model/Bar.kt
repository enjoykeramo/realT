package realt.core.model

/**
 * 一根K线。日线 [minute] 为 -1;分钟线 [minute] 为当日自 00:00 起的分钟数,
 * 且按 A 股惯例标注在 bar 的结束时刻(09:31 表示 09:30–09:31 这一分钟)。
 *
 * [date] 采用 yyyyMMdd 整数,避免依赖 java.time,便于在 Android 低版本复用。
 */
data class Bar(
    val date: Int,
    val minute: Int,
    val open: Double,
    val high: Double,
    val low: Double,
    val close: Double,
    val volume: Double,
    val amount: Double,
) {
    val isDaily: Boolean get() = minute < 0

    /** HHmm 形式,便于打印 */
    val hhmm: Int get() = if (minute < 0) -1 else (minute / 60) * 100 + minute % 60

    override fun toString(): String =
        if (isDaily) "Bar($date O=$open H=$high L=$low C=$close V=$volume)"
        else "Bar($date ${"%04d".format(hhmm)} O=$open H=$high L=$low C=$close V=$volume)"
}

object Session {
    const val OPEN = 9 * 60 + 30        // 09:30
    const val MORNING_END = 11 * 60 + 30 // 11:30
    const val AFTERNOON_START = 13 * 60  // 13:00
    const val CLOSE = 15 * 60            // 15:00
    const val MINUTES_PER_DAY = 240

    /** 把 bar 结束时刻换算为开盘后已交易的分钟数(1..240) */
    fun elapsed(minute: Int): Int = when {
        minute <= MORNING_END -> minute - OPEN
        minute <= AFTERNOON_START -> 120
        else -> 120 + (minute - AFTERNOON_START)
    }

    /** 距收盘还剩的交易分钟数 */
    fun remaining(minute: Int): Int = MINUTES_PER_DAY - elapsed(minute)
}

/** 证券代码,如 sh600000 / sz000001 / bj430047 */
data class Symbol(val market: String, val code: String) {
    val id: String get() = market + code

    /** 是否为 A 股普通股票(排除指数、基金、债券) */
    val isAShare: Boolean
        get() = when (market) {
            "sh" -> code.startsWith("60") || code.startsWith("68")
            "sz" -> code.startsWith("00") || code.startsWith("30")
            "bj" -> code.startsWith("4") || code.startsWith("8") || code.startsWith("92")
            else -> false
        }

    /** 涨跌停幅度:主板 10%,创业板/科创板 20%,北交所 30% */
    val limitPct: Double
        get() = when {
            market == "sz" && code.startsWith("30") -> 0.20
            market == "sh" && code.startsWith("68") -> 0.20
            market == "bj" -> 0.30
            else -> 0.10
        }

    override fun toString(): String = id

    companion object {
        fun parse(s: String): Symbol {
            val t = s.trim().lowercase()
            require(t.length >= 8) { "无效代码: $s" }
            return Symbol(t.substring(0, 2), t.substring(2))
        }
    }
}

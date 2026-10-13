package realt.core.model

/**
 * 把 1 分钟线合成为 N 分钟线(同一交易日内,按开盘后经过的分钟数分组)。
 * 例如 5 分钟线的第一根 = 09:31..09:35 五根 1 分钟线,标注为 09:35。
 * 未走完的最后一组也会输出(实时场景下就是正在形成的那根 K 线)。
 */
fun List<Bar>.aggregate(period: Int): List<Bar> {
    if (period <= 1 || isEmpty()) return this
    val out = ArrayList<Bar>(size / period + 1)
    var i = 0
    while (i < size) {
        val first = this[i]
        val bucket = (Session.elapsed(first.minute) - 1) / period
        var h = first.high
        var l = first.low
        var v = 0.0
        var a = 0.0
        var j = i
        while (j < size && this[j].date == first.date &&
            (Session.elapsed(this[j].minute) - 1) / period == bucket
        ) {
            val b = this[j]
            if (b.high > h) h = b.high
            if (b.low < l) l = b.low
            v += b.volume
            a += b.amount
            j++
        }
        val last = this[j - 1]
        out += Bar(first.date, last.minute, first.open, h, l, last.close, v, a)
        i = j
    }
    return out
}

/** 按交易日分组,保持时间顺序 */
fun List<Bar>.byDay(): LinkedHashMap<Int, List<Bar>> {
    val m = LinkedHashMap<Int, MutableList<Bar>>()
    for (b in this) m.getOrPut(b.date) { ArrayList(240) }.add(b)
    @Suppress("UNCHECKED_CAST")
    return m as LinkedHashMap<Int, List<Bar>>
}

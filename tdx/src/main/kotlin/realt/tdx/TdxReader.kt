package realt.tdx

import realt.core.model.Bar
import realt.core.model.Symbol
import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * 通达信离线数据(vipdoc)读取。
 *
 * 目录结构:vipdoc/{sh,sz,bj}/{lday,minline,fzline}/
 *  - lday/sh600000.day      日线,每条 32 字节
 *      uint32 日期 yyyyMMdd | uint32 开高低收 ×4(价格 × 100,基金/ETF × 1000)
 *      float32 成交额(元) | uint32 成交量(股) | uint32 保留
 *  - minline/sh600000.lc1   1 分钟线,每条 32 字节
 *  - fzline/sh600000.lc5    5 分钟线,每条 32 字节
 *      uint16 日期:year = v/2048 + 2004, month = (v%2048)/100, day = (v%2048)%100
 *      uint16 时间:自 00:00 起的分钟数(bar 结束时刻)
 *      float32 开高低收 ×4 | float32 成交额 | uint32 成交量 | uint32 保留
 * 全部为小端序。
 */
class TdxReader(val vipdoc: File) {

    init {
        require(vipdoc.isDirectory) { "vipdoc 目录不存在: $vipdoc" }
    }

    fun dayFile(s: Symbol) = File(vipdoc, "${s.market}/lday/${s.id}.day")
    fun min1File(s: Symbol) = File(vipdoc, "${s.market}/minline/${s.id}.lc1")
    fun min5File(s: Symbol) = File(vipdoc, "${s.market}/fzline/${s.id}.lc5")

    /** 读取日线;[lastN] > 0 时只读文件末尾 N 条(用于全市场扫描时提速) */
    fun daily(s: Symbol, lastN: Int = 0): List<Bar> {
        val f = dayFile(s)
        if (!f.isFile) return emptyList()
        val scale = priceScale(s)
        return readRecords(f, lastN).map { b ->
            val date = b.getInt(0)
            Bar(
                date, -1,
                b.getInt(4).toUInt().toDouble() / scale,
                b.getInt(8).toUInt().toDouble() / scale,
                b.getInt(12).toUInt().toDouble() / scale,
                b.getInt(16).toUInt().toDouble() / scale,
                b.getInt(24).toUInt().toDouble(),
                b.getFloat(20).toDouble(),
            )
        }.filter { it.close > 0 }
    }

    fun minute1(s: Symbol, lastN: Int = 0): List<Bar> = readMinute(min1File(s), lastN)
    fun minute5(s: Symbol, lastN: Int = 0): List<Bar> = readMinute(min5File(s), lastN)

    private fun readMinute(f: File, lastN: Int): List<Bar> {
        if (!f.isFile) return emptyList()
        return readRecords(f, lastN).map { b ->
            val d = b.getShort(0).toInt() and 0xFFFF
            val year = d / 2048 + 2004
            val month = (d % 2048) / 100
            val day = (d % 2048) % 100
            val minute = b.getShort(2).toInt() and 0xFFFF
            Bar(
                year * 10000 + month * 100 + day, minute,
                b.getFloat(4).toDouble(), b.getFloat(8).toDouble(),
                b.getFloat(12).toDouble(), b.getFloat(16).toDouble(),
                (b.getInt(24).toLong() and 0xFFFFFFFFL).toDouble(),
                b.getFloat(20).toDouble(),
            )
        }.filter { it.close > 0 }
    }

    private fun readRecords(f: File, lastN: Int): List<ByteBuffer> {
        RandomAccessFile(f, "r").use { raf ->
            val total = (raf.length() / REC).toInt()
            val n = if (lastN in 1 until total) lastN else total
            val bytes = ByteArray(n * REC)
            raf.seek((total - n).toLong() * REC)
            raf.readFully(bytes)
            val all = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
            return List(n) { i -> all.duplicate().order(ByteOrder.LITTLE_ENDIAN).position(i * REC).slice().order(ByteOrder.LITTLE_ENDIAN) }
        }
    }

    /** 列出某市场下有日线数据的全部证券 */
    fun listSymbols(market: String, aShareOnly: Boolean = true): List<Symbol> {
        val dir = File(vipdoc, "$market/lday")
        val files = dir.listFiles { _, name -> name.endsWith(".day") } ?: return emptyList()
        return files.map { Symbol(market, it.name.substring(2, it.name.length - 4)) }
            .filter { !aShareOnly || it.isAShare }
            .sortedBy { it.id }
    }

    /** 有 1 分钟线数据的 A 股 */
    fun listMinuteSymbols(): List<Symbol> =
        MARKETS.flatMap { m ->
            val dir = File(vipdoc, "$m/minline")
            (dir.listFiles { _, name -> name.endsWith(".lc1") } ?: emptyArray())
                .map { Symbol(m, it.name.substring(2, it.name.length - 4)) }
        }.filter { it.isAShare }.sortedBy { it.id }

    companion object {
        const val REC = 32
        val MARKETS = listOf("sh", "sz", "bj")

        /** 基金/ETF/可转债日线价格是 ×1000 存储 */
        fun priceScale(s: Symbol): Double = when {
            s.market == "sh" && (s.code.startsWith("5") || s.code.startsWith("11")) -> 1000.0
            s.market == "sz" && (s.code.startsWith("15") || s.code.startsWith("16") || s.code.startsWith("12")) -> 1000.0
            else -> 100.0
        }

        /** 常用指数 */
        val SH_INDEX = Symbol("sh", "000001")
        val SZ_INDEX = Symbol("sz", "399001")
        val CYB_INDEX = Symbol("sz", "399006")
    }
}

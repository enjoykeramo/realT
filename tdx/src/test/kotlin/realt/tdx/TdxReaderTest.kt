package realt.tdx

import realt.core.model.Bar
import realt.core.model.Symbol
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.file.Files
import java.io.File
import kotlin.test.Test
import kotlin.test.assertEquals

class TdxReaderTest {

    @Test
    fun dailyRoundTrip() {
        val dir = Files.createTempDirectory("vipdoc").toFile()
        val s = Symbol("sh", "600000")
        val bars = listOf(
            Bar(20250102, -1, 10.01, 10.5, 9.88, 10.32, 1234500.0, 1.27e7),
            Bar(20250103, -1, 10.32, 10.4, 10.1, 10.2, 999900.0, 1.01e7),
        )
        TdxWriter.writeDaily(File(dir, "sh/lday/sh600000.day"), s, bars)
        val r = TdxReader(dir)
        val got = r.daily(s)
        assertEquals(2, got.size)
        assertEquals(20250102, got[0].date)
        assertEquals(10.32, got[0].close, 1e-9)
        assertEquals(1234500.0, got[0].volume, 1e-6)
        assertEquals(listOf(got[1]), r.daily(s, lastN = 1))
        assertEquals(listOf(s), r.listSymbols("sh"))
    }

    @Test
    fun minuteDateTimeEncoding() {
        // 手工构造一条 lc1 记录:2025-09-30 09:31
        val buf = ByteBuffer.allocate(32).order(ByteOrder.LITTLE_ENDIAN)
        buf.putShort(((2025 - 2004) * 2048 + 930).toShort())
        buf.putShort((9 * 60 + 31).toShort())
        buf.putFloat(10.0f); buf.putFloat(10.2f); buf.putFloat(9.9f); buf.putFloat(10.1f)
        buf.putFloat(1.0e6f); buf.putInt(100000); buf.putInt(0)
        val dir = Files.createTempDirectory("vipdoc").toFile()
        File(dir, "sz/minline").mkdirs()
        File(dir, "sz/minline/sz000001.lc1").writeBytes(buf.array())
        File(dir, "sz/lday").mkdirs()
        val r = TdxReader(dir)
        val b = r.minute1(Symbol("sz", "000001")).single()
        assertEquals(20250930, b.date)
        assertEquals(931, b.hhmm)
        assertEquals(10.1, b.close, 1e-5)
        assertEquals(listOf(Symbol("sz", "000001")), r.listMinuteSymbols())
    }

    @Test
    fun etfPriceScale() {
        assertEquals(1000.0, TdxReader.priceScale(Symbol("sh", "510300")))
        assertEquals(100.0, TdxReader.priceScale(Symbol("sh", "600000")))
    }
}

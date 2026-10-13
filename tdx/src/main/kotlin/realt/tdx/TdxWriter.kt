package realt.tdx

import realt.core.model.Bar
import realt.core.model.Symbol
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

/** 写出通达信格式文件。用于单元测试及生成演示数据。 */
object TdxWriter {

    fun writeDaily(file: File, s: Symbol, bars: List<Bar>) {
        val scale = TdxReader.priceScale(s)
        val buf = ByteBuffer.allocate(bars.size * TdxReader.REC).order(ByteOrder.LITTLE_ENDIAN)
        for (b in bars) {
            buf.putInt(b.date)
            buf.putInt(Math.round(b.open * scale).toInt())
            buf.putInt(Math.round(b.high * scale).toInt())
            buf.putInt(Math.round(b.low * scale).toInt())
            buf.putInt(Math.round(b.close * scale).toInt())
            buf.putFloat(b.amount.toFloat())
            buf.putInt(b.volume.toLong().toInt())
            buf.putInt(0)
        }
        file.parentFile.mkdirs()
        file.writeBytes(buf.array())
    }

    fun writeMinute(file: File, bars: List<Bar>) {
        val buf = ByteBuffer.allocate(bars.size * TdxReader.REC).order(ByteOrder.LITTLE_ENDIAN)
        for (b in bars) {
            val y = b.date / 10000
            val md = b.date % 10000
            buf.putShort(((y - 2004) * 2048 + md).toShort())
            buf.putShort(b.minute.toShort())
            buf.putFloat(b.open.toFloat())
            buf.putFloat(b.high.toFloat())
            buf.putFloat(b.low.toFloat())
            buf.putFloat(b.close.toFloat())
            buf.putFloat(b.amount.toFloat())
            buf.putInt(b.volume.toLong().toInt())
            buf.putInt(0)
        }
        file.parentFile.mkdirs()
        file.writeBytes(buf.array())
    }
}

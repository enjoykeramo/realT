package realt.core.ml

import realt.core.engine.Costs
import realt.core.engine.TConfig

/**
 * 训练产出的"参数包":决策参数 + 费率 + 出手阈值 + GBDT 胜率模型。
 * 以纯文本保存,stockAPP 可直接把文件放进 assets 用 [parse] 加载。
 *
 * 用法:候选做T点(见 [Features.candidates])的胜率概率 ≥ [threshold] 才出手。
 */
class TradeModel(
    val config: TConfig,
    val costs: Costs,
    val threshold: Double,
    val gbdt: Gbdt,
    /** 训练、验证所用数据的日期范围(回测时据此提示样本内/样本外) */
    val trainStart: Int,
    val trainEnd: Int,
    /** 训练报告摘要,原样保存 */
    val meta: Map<String, String> = emptyMap(),
    /** 概率校准:模型原始分数 → 历史真实胜率 */
    val calibration: Calibration = Calibration.IDENTITY,
    /** 迭代寻优得到的策略;为 null 时只按模型胜率 ≥ [threshold] 出手 */
    val strategy: StrategySpec? = null,
) {
    /** 校准后的胜率概率 */
    fun winProb(features: DoubleArray): Double = calibration.apply(gbdt.predict(features))

    fun serialize(): String = buildString {
        appendLine("# realT 参数包(训练产出,请勿手工修改模型段)")
        appendLine("format=1")
        appendLine("features=${Features.names.joinToString(",")}")
        appendLine("threshold=$threshold")
        appendLine("calibration=${calibration.serialize()}")
        strategy?.let { appendLine("strategy=${it.serialize()}") }
        appendLine("trainStart=$trainStart")
        appendLine("trainEnd=$trainEnd")
        for ((k, v) in configMap(config)) appendLine("config.$k=$v")
        appendLine("costs.commissionRate=${costs.commissionRate}")
        appendLine("costs.minCommission=${costs.minCommission}")
        appendLine("costs.stampDutyRate=${costs.stampDutyRate}")
        appendLine("costs.transferRate=${costs.transferRate}")
        appendLine("costs.slippage=${costs.slippage}")
        for ((k, v) in meta) appendLine("meta.$k=${v.replace('\n', ' ')}")
        appendLine("[gbdt]")
        append(gbdt.serialize())
    }

    companion object {
        fun parse(text: String): TradeModel {
            val norm = text.replace("\r\n", "\n")
            val marker = "\n[gbdt]\n"
            val head = norm.substringBefore(marker)
            val body = norm.substringAfter(marker, "")
            require(body.isNotBlank()) { "参数包缺少 [gbdt] 段" }
            val kv = head.lineSequence().map { it.trim() }
                .filter { it.isNotEmpty() && !it.startsWith("#") && it.contains('=') }
                .associate { it.substringBefore('=') to it.substringAfter('=') }
            val feats = kv["features"]?.split(',') ?: emptyList()
            require(feats == Features.names) {
                "参数包的特征定义与当前程序版本不一致,请用当前版本重新训练"
            }
            val cfg = configFrom(kv.filterKeys { it.startsWith("config.") }.mapKeys { it.key.removePrefix("config.") })
            val c = Costs()
            val costs = Costs(
                commissionRate = kv["costs.commissionRate"]?.toDouble() ?: c.commissionRate,
                minCommission = kv["costs.minCommission"]?.toDouble() ?: c.minCommission,
                stampDutyRate = kv["costs.stampDutyRate"]?.toDouble() ?: c.stampDutyRate,
                transferRate = kv["costs.transferRate"]?.toDouble() ?: c.transferRate,
                slippage = kv["costs.slippage"]?.toDouble() ?: c.slippage,
            )
            return TradeModel(
                cfg, costs, kv.getValue("threshold").toDouble(), Gbdt.deserialize(body),
                kv["trainStart"]?.toInt() ?: 0, kv["trainEnd"]?.toInt() ?: 0,
                kv.filterKeys { it.startsWith("meta.") }.mapKeys { it.key.removePrefix("meta.") },
                Calibration.parse(kv["calibration"] ?: ""),
                kv["strategy"]?.let { StrategySpec.parse(it) },
            )
        }

        fun configMap(c: TConfig): Map<String, String> = linkedMapOf(
            "tRatio" to c.tRatio.toString(),
            "maxTradesPerDay" to c.maxTradesPerDay.toString(),
            "noOpenAfterOpenMin" to c.noOpenAfterOpenMin.toString(),
            "noOpenBeforeCloseMin" to c.noOpenBeforeCloseMin.toString(),
            "forceCloseBeforeCloseMin" to c.forceCloseBeforeCloseMin.toString(),
            "entryThreshold" to c.entryThreshold.toString(),
            "exitThreshold" to c.exitThreshold.toString(),
            "takeProfitAtr" to c.takeProfitAtr.toString(),
            "stopLossAtr" to c.stopLossAtr.toString(),
            "momentumBlock" to c.momentumBlock.toString(),
            "minEdgePct" to c.minEdgePct.toString(),
            "edgeCostMultiple" to c.edgeCostMultiple.toString(),
            "trendBlock" to c.trendBlock.toString(),
            "flatBlock" to c.flatBlock.toString(),
            "maxHoldMin" to c.maxHoldMin.toString(),
            "lateLockMin" to c.lateLockMin.toString(),
            "regimeExit" to c.regimeExit.toString(),
            "cooldownMin" to c.cooldownMin.toString(),
            "allowSellFirst" to c.allowSellFirst.toString(),
            "allowBuyFirst" to c.allowBuyFirst.toString(),
            "entryOffsetAtr" to c.entryOffsetAtr.toString(),
            "timeStopMin" to c.timeStopMin.toString(),
            "entryWaitMin" to c.entryWaitMin.toString(),
            "tpLimit" to c.tpLimit.toString(),
        )

        fun configFrom(m: Map<String, String>, base: TConfig = TConfig()): TConfig {
            fun d(k: String, v: Double) = m[k]?.toDouble() ?: v
            fun i(k: String, v: Int) = m[k]?.toInt() ?: v
            fun b(k: String, v: Boolean) = m[k]?.toBoolean() ?: v
            return TConfig(
                tRatio = d("tRatio", base.tRatio),
                maxTradesPerDay = i("maxTradesPerDay", base.maxTradesPerDay),
                noOpenAfterOpenMin = i("noOpenAfterOpenMin", base.noOpenAfterOpenMin),
                noOpenBeforeCloseMin = i("noOpenBeforeCloseMin", base.noOpenBeforeCloseMin),
                forceCloseBeforeCloseMin = i("forceCloseBeforeCloseMin", base.forceCloseBeforeCloseMin),
                entryThreshold = d("entryThreshold", base.entryThreshold),
                exitThreshold = d("exitThreshold", base.exitThreshold),
                takeProfitAtr = d("takeProfitAtr", base.takeProfitAtr),
                stopLossAtr = d("stopLossAtr", base.stopLossAtr),
                momentumBlock = d("momentumBlock", base.momentumBlock),
                minEdgePct = d("minEdgePct", base.minEdgePct),
                edgeCostMultiple = d("edgeCostMultiple", base.edgeCostMultiple),
                trendBlock = d("trendBlock", base.trendBlock),
                flatBlock = d("flatBlock", base.flatBlock),
                maxHoldMin = i("maxHoldMin", base.maxHoldMin),
                lateLockMin = i("lateLockMin", base.lateLockMin),
                regimeExit = b("regimeExit", base.regimeExit),
                cooldownMin = i("cooldownMin", base.cooldownMin),
                allowSellFirst = b("allowSellFirst", base.allowSellFirst),
                allowBuyFirst = b("allowBuyFirst", base.allowBuyFirst),
                entryOffsetAtr = d("entryOffsetAtr", base.entryOffsetAtr),
                timeStopMin = i("timeStopMin", base.timeStopMin),
                entryWaitMin = i("entryWaitMin", base.entryWaitMin),
                tpLimit = b("tpLimit", base.tpLimit),
            )
        }
    }
}

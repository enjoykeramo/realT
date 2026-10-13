package realt.core.engine

import kotlin.math.max

/**
 * A 股交易成本。默认:佣金万 1(最低 5 元)、卖出印花税万 5、过户费十万分之一。
 */
data class Costs(
    val commissionRate: Double = 0.0001,
    val minCommission: Double = 5.0,
    val stampDutyRate: Double = 0.0005,
    val transferRate: Double = 0.00001,
    /** 成交滑点(按价格比例,单边) */
    val slippage: Double = 0.0002,
) {
    fun buyFee(value: Double): Double = max(value * commissionRate, minCommission) + value * transferRate
    fun sellFee(value: Double): Double = max(value * commissionRate, minCommission) + value * transferRate + value * stampDutyRate

    /** 成交额为 [notional] 时一买一卖的实际费用比例(含最低佣金与双边滑点) */
    fun roundTripRateAt(notional: Double): Double =
        if (notional <= 0) roundTripRate else (buyFee(notional) + sellFee(notional)) / notional + 2 * slippage

    /** 一买一卖的费用占成交额比例(不含最低佣金影响) */
    val roundTripRate: Double get() = 2 * commissionRate + stampDutyRate + 2 * transferRate + 2 * slippage
}

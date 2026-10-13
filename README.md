# realT —— A 股分时做T算法

面向**已有底仓**的自选持仓,在分时图上给出高抛低吸(做T)建议,并用通达信离线数据做多标的回测验证。
目标:**出手胜率 ≥ 80%,且扣除费用后期望为正**;看不清的日子宁可不做。

算法设计与取舍见 [docs/DESIGN.md](docs/DESIGN.md)。

## 模块

| 模块 | 内容 | 依赖 |
|---|---|---|
| `core` | 纯算法:指标、缠论、大盘环境、主力阶段、分时形态预判、做T决策引擎、迷宫记忆 | 仅 Kotlin 标准库,JVM 17 字节码,**可直接给 stockAPP(Android)使用** |
| `tdx` | 通达信 vipdoc 读取(`.day` / `.lc1` / `.lc5`)、市场宽度扫描 | core |
| `backtest` | 分钟级回放、原型规则对照组、L6 记忆回放、参数扫描、报告 | core, tdx |
| `app` | 命令行工具 `realt` | backtest |

## Windows 上使用

### 1. 准备数据(通达信)

在通达信客户端「系统 → 盘后数据下载」中勾选并下载:

- **日线和实时行情数据**(`vipdoc\sh|sz\lday`,用于大盘环境、主力阶段)
- **1 分钟线数据**(`vipdoc\sh|sz\minline\*.lc1`,用于分时回测)

1 分钟线下载的天数越多,回测越可信(建议至少半年)。

### 2. 获取程序

- 从 GitHub Releases 下载 `realt-<版本>-windows-x64.zip`(自带 Java,解压即用);或
- 自行构建:`gradlew.bat :app:installDist`,程序在 `app\build\install\realt\bin\realt.bat`(需要 JDK 17+)

### 3. 配置数据目录与费率(只需一次)

```bat
realt.exe config --vipdoc D:\new_tdx\vipdoc
realt.exe config --commission 0.0001 --min-commission 5
realt.exe check
```

费率按你券商的实际情况填(`check` 会显示当前费率):

| 项目 | 选项 | 默认 | 说明 |
|---|---|---|---|
| 佣金 | `--commission` | 0.0001(万1) | 买卖双向收取 |
| 最低佣金 | `--min-commission` | 5 元 | 免五账户填 0 |
| 印花税 | `--stamp` | 0.0005(万5) | 仅卖出收取,2023-08-28 起减半为万5 |
| 过户费 | `--transfer` | 0.00001(十万分之一) | 沪深双向收取 |
| 滑点 | `--slippage` | 0.0002(万2) | 回测成交价的不利偏移,单边 |

以 3 万元的 T 仓为例,一买一卖:佣金按最低 5 元 ×2 + 印花税 15 + 过户费 0.6 ≈ 25.6 元,约 0.085%;再加两次滑点约 0.04%。
T 仓达到 5 万元以上时,万1 佣金才超过 5 元最低线。

### 4. 训练(核心步骤)

```bat
realt.exe train --sample 200 --out model1
realt.exe train --sample 200 --target 0.9 --out model1   :: 也可以指定其他目标胜率
```

**数据准备**:在通达信下载 1 分钟线时,请确保包含**上证指数 000001、创业板指 399006、科创50 000688**
(文件 `vipdoc\sh\minline\sh000001.lc1` 等)。指数分时是重要特征,缺失时报告会提示。

训练分 5 个阶段,每个阶段在**同一行**显示进度条、完成数、当前任务、已用时间和预计剩余时间:

```
[3/5 参数筛选] ██████████░░░░░░░░░░  52%  25/48 | 离场组7 第2段 | 用时 01:12 剩余约 01:06
```

1. **候选生成**(按股票多线程):逐分钟回放,记录每个转折信号(缠论 1/5 分钟买卖点、布林刺破收回、RSI 拐头)及其约 110 个特征,
   并对 12 组「开仓方式(市价/挂单)× 止盈 × 止损」参数分别模拟这一笔T(当日完成,扣全部费用)
2. **特征分箱**:一次完成,后续所有模型共享
3. **参数筛选**(模型训练多线程并行):滚动前推 + 概率校准,选出达标且净盈亏最大的参数与出手阈值
4. **模块消融**:逐一去掉缠论、MACD、布林、L1 大盘、L2 主力阶段、L3 分时形态、指数分时等,衡量每层贡献;
   去掉后在每一段都更好的层会被剔除,得到最优组合
5. **最终模型**:按最优组合重训,输出参数包

输出:
- `model1\realt-model.txt`:**参数包**(胜率模型 + 概率校准 + 出手阈值 + 开仓/止盈/止损参数 + 费率)
- `model1\train-report.md`:结论(含 95% 置信区间)、参数筛选表、**各层/模块贡献表**、**信号来源效能表**、
  阈值—胜率—出手次数表、亏损单分析、特征重要性

说明:
- 默认目标胜率 95%(且净盈亏 > 0);出手阈值是**校准后的胜率**,0.95 即"历史上同类信号的胜率约 95%"
- **测试段从未参与任何选择,它的胜率才可信**;笔数少时看置信区间
- `--threads N` 指定线程数(默认 CPU 核数)。300 只股票、约 90 个交易日,8 核约 10~15 分钟;内存不足时减少 `--sample`

### 5. 用参数包回测

```bat
realt.exe backtest --sample 100 --model model1\realt-model.txt --out out2
```

报告第 1 节会把结果拆成「样本内」和「样本外」两行(以参数包的训练截止日为界),**只看样本外**。
可以换一批训练时没用过的股票(`--symbols` / `--file`)做交叉验证。

`backtest` 的其他用法:

```bat
realt.exe backtest --symbols sh601188,sz000001,600519 --out out1
realt.exe backtest --file 持仓.txt --out out3
```

输出目录里:`report.md`(总体对比、按分时形态/方向/平仓原因拆分、逐只股票达标情况)、`trades.csv`(每笔明细,Excel 可打开)。

`realt.exe sweep` 是早期的规则参数扫描,不训练模型,保留作对比用。

### 6. 单日回放(看算法每一步怎么想的)

```bat
realt.exe replay --symbol sh601188 --date 20250930
```

### 无数据体验

```bat
realt.exe demo
```

生成一个合成的 vipdoc 并跑完整流程。合成数据**没有任何实盘意义**,只用于确认程序可运行。

## 在 stockAPP 中使用 core

```kotlin
// 开盘前
val regime = MarketRegimeDetector().detect(indexDaily, breadth)      // L1
val stage = StageDetector().detect(stockDaily)                       // L2
val ctx = DayContext(prevClose, atrPct, regime, stage)
val pack = TradeModel.parse(assets.open("realt-model.txt").reader().readText())  // 训练产出的参数包
val engine = TEngine(pack.config, pack.costs, ctx, baseShares = 2000, history = yesterday1m, model = pack)

// 盘中每来一根 1 分钟 K 线
val advice: TAction? = engine.onBar(bar)          // 高抛 / 低吸 / 回补 / 卖出
advice?.let { push(it.kind.label, it.price, it.shares, it.reason) }
engine.dist                                        // 当前分时形态概率,可画在分时图上
// 用户确认成交后
engine.onFill(advice, fillPrice, minute)
```

## 构建与测试

```bash
./gradlew test               # 单元测试 + 端到端测试
./gradlew :app:installDist   # 构建命令行工具
```

推送 `v*` 标签(例如 `v0.1.0`)会触发 GitHub Actions 构建 Windows 免安装版并发布 Release。

## 免责声明

本项目用于算法研究与验证,不构成投资建议。历史回测不代表未来收益。

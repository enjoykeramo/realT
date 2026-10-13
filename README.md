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
realt.exe config --commission 0.00025 --min-commission 5
realt.exe check
```

费率按你券商的实际情况填(`check` 会显示当前费率):

| 项目 | 选项 | 默认 | 说明 |
|---|---|---|---|
| 佣金 | `--commission` | 0.00025(万2.5) | 买卖双向收取 |
| 最低佣金 | `--min-commission` | 5 元 | 免五账户填 0 |
| 印花税 | `--stamp` | 0.0005(万5) | 仅卖出收取,2023-08-28 起减半为万5 |
| 过户费 | `--transfer` | 0.00001(十万分之一) | 沪深双向收取 |
| 滑点 | `--slippage` | 0.0002(万2) | 回测成交价的不利偏移,单边 |

以 3 万元的 T 仓为例,一买一卖:佣金 7.5×2 + 印花税 15 + 过户费 0.6 ≈ 30.6 元,约 0.10%;再加两次滑点约 0.04%。

### 4. 训练(核心步骤)

```bat
realt.exe train --sample 100 --out model1
```

- 输出 `model1\realt-model.txt`(**参数包**:胜率模型 + 出手阈值 + 止盈止损 + 费率)和 `model1\train-report.md`
- 数据按时间切成 训练 60% / 验证 20% / 测试 20%:训练集学模型,验证集挑止盈止损和出手阈值,**测试集从未参与任何选择,它的胜率才可信**
- 报告里有「阈值—胜率—出手次数」对照表:阈值越高越挑剔,胜率越高、出手越少
- 100 只股票、几个月 1 分钟数据,训练约需十几到几十分钟

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

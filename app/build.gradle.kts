plugins {
    application
}

dependencies {
    implementation(project(":backtest"))
}

application {
    mainClass.set("realt.app.MainKt")
    applicationName = "realt"
    // 堆上限按物理内存的 75% 自动确定(固定 -Xmx 在股票多时容易不够);可用环境变量 JAVA_OPTS 覆盖
    applicationDefaultJvmArgs = listOf("-XX:MaxRAMPercentage=75", "-Dfile.encoding=UTF-8")
}

plugins {
    application
}

dependencies {
    implementation(project(":backtest"))
}

application {
    mainClass.set("realt.app.MainKt")
    applicationName = "realt"
    applicationDefaultJvmArgs = listOf("-Xmx2g", "-Dfile.encoding=UTF-8")
}

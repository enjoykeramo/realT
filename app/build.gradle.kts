import java.time.LocalDateTime

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

// 构建信息(版本、构建时间、提交号):运行 `realt version` 或训练开始时显示,用来确认运行的是新构建
val buildInfo = tasks.register("buildInfo") {
    val out = layout.buildDirectory.dir("generated/buildinfo")
    val ver = project.version.toString()
    val root = rootDir
    outputs.dir(out)
    outputs.upToDateWhen { false }
    doLast {
        val commit = runCatching {
            val p = ProcessBuilder("git", "rev-parse", "--short", "HEAD").directory(root).redirectErrorStream(true).start()
            p.inputStream.bufferedReader().readText().trim().takeIf { p.waitFor() == 0 } ?: ""
        }.getOrDefault("")
        val f = out.get().file("realt/build.properties").asFile
        f.parentFile.mkdirs()
        f.writeText("version=$ver\nbuilt=${LocalDateTime.now().withNano(0)}\ncommit=$commit\n")
    }
}
sourceSets.main { resources.srcDir(buildInfo) }

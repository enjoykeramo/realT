import org.jetbrains.kotlin.gradle.dsl.JvmTarget
import org.jetbrains.kotlin.gradle.dsl.KotlinJvmProjectExtension

plugins {
    kotlin("jvm") version "2.0.21" apply false
}

allprojects {
    group = "realt"
    version = (findProperty("realtVersion") as String?) ?: "0.1.0"
}

subprojects {
    apply(plugin = "org.jetbrains.kotlin.jvm")

    // JVM 17 字节码:与 Android (AGP 8+) 兼容,core 可直接被 stockAPP 依赖
    extensions.configure<KotlinJvmProjectExtension> {
        compilerOptions {
            jvmTarget.set(JvmTarget.JVM_17)
            freeCompilerArgs.add("-Xjdk-release=17")
        }
    }
    extensions.configure<JavaPluginExtension> {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    dependencies {
        "testImplementation"(kotlin("test"))
    }

    tasks.withType<JavaCompile>().configureEach {
        options.release.set(17)
    }

    tasks.withType<Test>().configureEach {
        useJUnitPlatform()
    }
}

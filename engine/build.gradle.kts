plugins {
    kotlin("jvm") version "2.2.0"
    kotlin("plugin.serialization") version "2.2.0"
    application
}

group = "com.voicescam"
version = "0.1.0"

kotlin {
    compilerOptions {
        jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_21)
    }
}

java {
    sourceCompatibility = JavaVersion.VERSION_21
    targetCompatibility = JavaVersion.VERSION_21
}


repositories {
    mavenCentral()
    maven { url = uri("https://jitpack.io") }
    maven {
        name = "TarsosDSP"
        url = uri("https://mvn.0110.be/releases")
    }
}

val sherpaOnnxVersion = "v1.13.7"
val osName = System.getProperty("os.name").lowercase()
val osArch = System.getProperty("os.arch").lowercase()
val sherpaNativeClassifier = when {
    osName.contains("mac") || osName.contains("darwin") -> {
        if (osArch == "aarch64" || osArch == "arm64") "osx-aarch64" else "osx-x64"
    }
    osName.contains("linux") -> {
        if (osArch == "aarch64" || osArch == "arm64") "linux-aarch64" else "linux-x64"
    }
    osName.contains("win") -> {
        if (osArch == "aarch64" || osArch == "arm64") "win-arm64" else "win-x64"
    }
    else -> "linux-x64"
}

val ktorVersion = "3.0.3"

dependencies {
    implementation("io.ktor:ktor-server-core:$ktorVersion")
    implementation("io.ktor:ktor-server-netty:$ktorVersion")
    implementation("io.ktor:ktor-server-websockets:$ktorVersion")
    implementation("io.ktor:ktor-server-content-negotiation:$ktorVersion")
    implementation("io.ktor:ktor-serialization-kotlinx-json:$ktorVersion")
    implementation("io.ktor:ktor-server-cors:$ktorVersion")
    implementation("io.ktor:ktor-server-status-pages:$ktorVersion")
    implementation("io.ktor:ktor-server-call-logging:$ktorVersion")

    implementation("org.jetbrains.kotlinx:kotlinx-serialization-json:1.7.3")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:1.9.0")

    implementation("com.microsoft.onnxruntime:onnxruntime:1.20.0")
    implementation("com.github.k2-fsa.sherpa-onnx:sherpa-onnx-jvm:$sherpaOnnxVersion")
    implementation("com.github.k2-fsa.sherpa-onnx:sherpa-onnx-native-lib-$sherpaNativeClassifier:$sherpaOnnxVersion")
    implementation("be.tarsos.dsp:core:2.5")
    implementation("be.tarsos.dsp:jvm:2.5")

    implementation("ch.qos.logback:logback-classic:1.5.16")
}

application {
    mainClass.set("com.voicescam.ApplicationKt")
}

tasks.named<JavaExec>("run") {
    workingDir = rootProject.projectDir
}

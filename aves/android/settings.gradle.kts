pluginManagement {
    val flutterSdkPath = run {
        val properties = java.util.Properties()
        file("local.properties").inputStream().use { properties.load(it) }
        val flutterSdkPath = properties.getProperty("flutter.sdk")
        require(flutterSdkPath != null) { "flutter.sdk not set in local.properties" }
        flutterSdkPath
    }

    includeBuild("$flutterSdkPath/packages/flutter_tools/gradle")

    repositories {
        // The mirrors come first because plugin resolution from
        // plugins.gradle.org is slow from a Chinese network and this build
        // pulls several plugins. Drop the three maven { } lines below if you
        // are not in China -- the rest works on its own.
        if (System.getenv("CI") == null) {
            maven { url = uri("https://maven.aliyun.com/repository/gradle-plugin") }
            maven { url = uri("https://maven.aliyun.com/repository/google") }
            maven { url = uri("https://maven.aliyun.com/repository/public") }
        }
        google()
        mavenCentral()
        // the mirrors above can lag behind on plugin marker artifacts
        gradlePluginPortal()
    }
}

// Settings plugins (`Plugin<Settings>`) must be applied in the settings script.
plugins {
    id("dev.flutter.flutter-plugin-loader") version "1.0.0"

    // provide a repository to download additional JDKs
    // e.g. for java/kotlin `jvmToolchain` defined in build scripts
    id("org.gradle.toolchains.foojay-resolver-convention") version "1.0.0"

    // define versions (Gradle version catalog cannot be referenced here)
    id("com.android.application") version "9.4.1" apply false
    id("org.jetbrains.kotlin.android") version "2.4.20" apply false
}

include(":app")
include(":exifinterface")

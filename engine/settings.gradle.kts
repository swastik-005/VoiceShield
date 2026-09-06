rootProject.name = "voicescam-engine"

pluginManagement {
    repositories {
        gradlePluginPortal()
        mavenCentral()
    }
}

dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.PREFER_PROJECT)
    repositories {
        mavenCentral()
        maven { url = uri("https://jitpack.io") }
        maven {
            name = "TarsosDSP"
            url = uri("https://mvn.0110.be/releases")
        }
    }
}

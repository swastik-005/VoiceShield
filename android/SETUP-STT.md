# Speech-to-text setup (sherpa-onnx)

`SpeechBranch.kt` is written against the official sherpa-onnx Kotlin API, but
sherpa-onnx itself is **not** on the Gradle dependency list. There is no
confirmed first-party Maven artifact — the `com.bihe0832.android:lib-sherpa-onnx`
entry on Maven Central is a third-party republish, deliberately not used here.
So sherpa is vendored the way its own docs describe.

Until you complete steps 1–3 the app still builds and runs, and the AI-vs-human
branch is unaffected. Only the transcript stays empty.

**How that works, because a runtime `try`/`catch` is not enough.** `SpeechBranch`
imports `com.k2fsa.sherpa.onnx.*`, and those imports are resolved by the
*compiler*. Catching `UnsatisfiedLinkError` rescues a missing `.so` at runtime,
but nothing can rescue a missing Kotlin API at build time — there is no binary
to run. So the choice is made by `app/build.gradle.kts`, which looks for `.kt`
files in `app/src/main/java/com/k2fsa/sherpa/onnx/` and adds one of two source
sets to the build:

| sherpa vendored? | source set compiled | behaviour |
|---|---|---|
| no  | `app/src/noStt/java`   | inert stub, `start()` returns false |
| yes | `app/src/withStt/java` | the real streaming recogniser |

Each build prints which one it chose. Complete step 2 below and the next build
switches over on its own — there is no flag to flip.

If you edit `SpeechBranch`, **keep the two versions' public surface identical**.
`CaptureService` is compiled against whichever is present and knows about
neither, so a method added to one and not the other turns a missing optional
feature into a broken build.

## 1. Native libraries

Download `sherpa-onnx-v<version>-android.tar.bz2` from
<https://github.com/k2-fsa/sherpa-onnx/releases> and copy **two** `.so` files
per ABI into `app/src/main/jniLibs/<abi>/`:

```
app/src/main/jniLibs/arm64-v8a/libonnxruntime.so
app/src/main/jniLibs/arm64-v8a/libsherpa-onnx-jni.so
app/src/main/jniLibs/armeabi-v7a/...
app/src/main/jniLibs/x86_64/...
app/src/main/jniLibs/x86/...
```

The directories already exist with a `PLACEHOLDER.txt` in each — delete those
once the real files are in. `arm64-v8a` alone is enough for a modern phone and
saves roughly 15 MB per unused ABI.

Note this ships a **second** `libonnxruntime.so`, separate from the
`onnxruntime-android` Maven artifact that `SpoofDetector` uses. They coexist
(different loaders), but it is the main size cost of this branch.

## 2. Kotlin API

Copy `sherpa-onnx/kotlin-api/*.kt` from the same repo into:

```
app/src/main/java/com/k2fsa/sherpa/onnx/
```

Keep the package `com.k2fsa.sherpa.onnx` — `SpeechBranch.kt` imports
`OnlineRecognizer`, `OnlineStream`, `OnlineRecognizerConfig`,
`OnlineModelConfig`, `OnlineTransducerModelConfig`, `FeatureConfig` and
`EndpointConfig` from it. Take the API files from the **same release tag** as
the `.so` files; a mismatch surfaces as a JNI crash, not a compile error.

## 3. Model

Download a **streaming zipformer transducer** (an offline/non-streaming model
will not work — `OnlineRecognizer` requires a streaming one) from
<https://github.com/k2-fsa/sherpa-onnx/releases> and place four files in
`app/src/main/assets/stt/`:

```
encoder.onnx
decoder.onnx
joiner.onnx
tokens.txt
```

Rename the release's `encoder-epoch-99-avg-1.onnx` etc. to these names, or
override them in `SpeechBranch.Config`. Prefer the `int8` variants on a phone.

`app/build.gradle.kts` already sets `noCompress += listOf("onnx")`, so these are
memory-mapped from the APK rather than unpacked.

## How it is wired

- `CaptureService` creates one `SpeechBranch` and calls `push(window)` from the
  capture callback, so **both branches share a single `AudioRecord`**.
  `android.speech.SpeechRecognizer` was rejected because it opens its own
  capture session and would fight `CallAudioCapture` for the microphone.
- `push()` forwards only the **fresh tail** of each window. `CallAudioCapture`
  emits a 1.0 s window every 0.5 s, so consecutive windows overlap by half —
  feeding whole windows would send every sample twice and the transcript would
  stutter.
- `push()` never decodes. It enqueues and returns; decoding runs on the
  `stt-decode` thread, because the caller is the thread that must keep draining
  `AudioRecord` for both branches. On overflow the **oldest** audio is dropped
  rather than blocking the reader.
- Transcripts are broadcast as `ACTION_TRANSCRIPT` and rendered by
  `MainActivity`, independently of the score — the two branches run on different
  clocks and neither waits for the other.

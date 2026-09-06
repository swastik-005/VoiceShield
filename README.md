# VoiceShield
Voice Anti-spoofing
# VoiceShield

VoiceShield is a live unknown-call monitor. The system Phone app still receives every call. When the number is not in the address book, VoiceShield asks permission to listen on speaker, transcribe the caller, score the speech against a scam lexicon, draw a live spectrogram, and optionally run an ONNX anti-spoofing model.

This repository is **live-call only**. There is no upload/batch product UI. The Kotlin engine still exposes `/predict` for debugging audio files.

---

## Product behaviour

1. The user opens VoiceShield and grants:
   - scam alerts (beep + optional notifications)
   - **all contacts** (one Android `READ_CONTACTS` grant; the whole address book is imported)
   - unknown-call monitoring (`ROLE_CALL_SCREENING` / Caller ID & spam — **not** the default Phone app)
2. Incoming calls ring in the **stock Phone / Samsung Phone** app.
3. If the caller is in contacts, VoiceShield stays silent.
4. If the caller is unknown, VoiceShield prompts **Monitor this call?**
5. The user answers in Phone, puts the call on **speaker**, and allows VoiceShield to use the microphone.
6. While monitoring:
   - sherpa-onnx produces a **live transcript**
   - lexicon phrases from `scam_voice_detection_lexicon.pdf` are **underlined**
   - a **match %** bar shows how much of the spoken content hit the lexicon
   - **≥ 75%** (with enough content words) **red-flags** the call
   - each **1.0 s** audio window (0.5 s hop) becomes a **104×63** log-linear spectrogram and, if present, an ONNX spoof score
7. Hang up / Stop monitoring only stops VoiceShield. The Phone app still owns the call.
8. **Caller history** is stored on the device (localStorage in the WebView). Red-flagged rows stay marked.

Desktop browser demo: after onboarding, a simulated unknown ring appears once (~1.8 s). Native Android waits for a real incoming number.

---

## Architecture

```text
┌─────────────────────┐     speaker / mic PCM      ┌──────────────────────────┐
│  Android Phone app  │ ─────────────────────────► │ VoiceShield APK          │
│  (default dialer)   │   unknown → prompt         │ WebView UI (Vite build)  │
└─────────────────────┘                            │ CallScreeningService     │
                                                   │ READ_CONTACTS import     │
                                                   └────────────┬─────────────┘
                                                                │ WebSocket /stream
                                                                │ PCM s16le 16 kHz
                                                   ┌────────────▼─────────────┐
                                                   │ Kotlin engine :8080      │
                                                   │  fan-out 16 kHz PCM      │
                                                   │  ├─ 100 ms → sherpa-onnx │
                                                   │  └─ 1.0 s / 0.5 s hop    │
                                                   │       → LogLinearSpec    │
                                                   │       → ONNX [1,1,104,63]│
                                                   │  lexicon score on ASR    │
                                                   └──────────────────────────┘
```

**Do not** feed overlapping spectrogram windows into sherpa-onnx. ASR only receives contiguous 100 ms chunks (`1600` samples at 16 kHz). Decode runs on a worker thread, never on the AudioRecord / WS ingest path.

---

## Repository layout

| Path | Role |
|------|------|
| `engine/` | Ktor 3 + ONNX Runtime + sherpa-onnx + spectrogram + lexicon |
| `frontend/` | React + Vite (JavaScript). Desktop Galaxy S23 chassis; `?native=1` is full-screen |
| `mobile/` | Android app `com.voiceshield.app` — WebView + capture library |
| `mobile/capture/` | `CallAudioCapture` (100 ms stream + 1.0 s / 0.5 s windows) |
| `android/` | Reference capture/ASR coordinator (not packaged into the APK) |
| `engine/keywords.txt` | Lexicon phrases extracted from the PDF |
| `scam_voice_detection_lexicon.pdf` | Source lexicon (urgency, OTP, remote access, impersonation, …) |
| `VoiceShield.apk` | Sideload build (also `frontend/public/VoiceShield.apk`) |

---

## Spectrogram (features.py port)

Implementation: [`engine/src/main/kotlin/com/voicescam/audio/LogLinearSpectrogram.kt`](engine/src/main/kotlin/com/voicescam/audio/LogLinearSpectrogram.kt)

This is a **port of `features.py`**, not a redesign. Order of operations:

```text
frame → periodic Hann → FFT → magnitude → log → BAND CROP → dB floor → standardise
```

| Constant | Value |
|----------|--------|
| Sample rate | 16 000 Hz |
| Capture window | 16 000 samples (1.0 s) |
| Window hop | 8 000 samples (0.5 s) |
| `n_fft` / `win_length` | 512 |
| STFT hop | 256 |
| Frames per window | `1 + 16000/256` = **63** |
| FFT bins | 257 |
| Band crop | bins **8..111** inclusive → **104** (≈ 250–3469 Hz) |
| Log | `ln(mag + 1e-6)` |
| Floor | 80 dB below peak of the **cropped** array (`TOP_DB / (20/ln(10))` nats) |
| Std | unbiased, whole 2-D array (`count - 1`), torch-style |
| Silence | if `std < 1e-4`, emit zeros |
| Layout | frequency-major `out[f * 63 + t]` |
| ONNX input | **`[1, 1, 104, 63]`** float |

Hann is **periodic** (`2πk/N`, N=512). STFT uses **reflect** pad of `n_fft/2`, excluding the edge sample (torch `center=true`, `pad_mode=reflect`). Short/long audio is centre-cropped or end-padded to 1.0 s (`fix_length`).

`LogLinearSpectrogram` is **not thread-safe**. The engine keeps one instance per thread via `ThreadLocal`.

Class logits are assumed **`[bonafide, spoof]`**. If an export is reversed, swap in `OnnxInference.probabilities`.

The older 80×401 log-mel extractor remains in the tree as unused reference (`MelSpectrogramExtractor.kt`). Live inference does **not** use it.

---

## Live ASR (sherpa-onnx)

- JNI: JitPack `sherpa-onnx-jvm` + `sherpa-onnx-native-lib-<os>-<arch>` **v1.13.7**
- Types live only under `engine/src/main/kotlin/com/voicescam/asr/`
- Default dir: `engine/models/sherpa-onnx-streaming-zipformer-en-2023-06-26`
- Feature rate must stay **16 kHz**; chunks **1600** samples

Download example:

```bash
cd engine/models
curl -L -o zf.tar.bz2 \
  https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-en-2023-06-26.tar.bz2
tar xvf zf.tar.bz2
rm zf.tar.bz2
```

`GET /health` includes `asrLoaded` and `asrError`. Missing files: the HTTP server still starts; transcripts are empty.

---

## Scam lexicon

Phrases in [`engine/keywords.txt`](engine/keywords.txt) (and `frontend/src/lexiconPhrases.js` for on-device highlighting if the engine omits spans).

Scoring (`ScamKeywordScorer`):

- Case-insensitive phrase match, **longest first**, word boundaries (so `pin` does not hit `shopping`)
- **Match %** = content words covered by hits / content words  
  (stopwords such as *the*, *you*, *from* are ignored)
- **Red flag** if match % **≥ 0.75** and there are at least **6** content words
- Engine WS `transcript` events include `keywordHits`, `keywordScore` (0–1), `keywordSpans`, `contentWords`, `scamAlert`

Frontend underlines `keywordSpans` in the live transcript and writes **Caller history** in `localStorage`.

---

## Engine (backend)

Requires **JDK 21** (Gradle Kotlin 2.2) and **ffmpeg** on `PATH` for compressed uploads. WAV can use Java Sound.

```bash
cd engine
./gradlew run
```

Listens on `http://0.0.0.0:8080`. Config: [`engine/engine.properties`](engine/engine.properties). Env overrides use `VOICESCAM_` + uppercase keys (`onnx.modelPath` → `VOICESCAM_ONNX_MODEL_PATH`).

The process **starts without** `voice_scam.onnx` or Zipformer files.

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health` | ONNX + ASR load status, window duration, notes |
| GET | `/models` | ONNX input/output names and shapes |
| GET | `/reference/spectrogram` | EMA bona-fide reference if any window classified bona fide |
| POST | `/predict` | Multipart audio → `DetectionResult` (debug) |
| POST | `/predict/batch` | Multiple files |
| WS | `/stream` | Live PCM → windows + transcripts |

### WebSocket `/stream`

1. Text `{ "type": "start", "callId": "...", "sampleRate": 16000, "format": "pcm_s16le" }` (`pcm_f32le` accepted).
2. Binary PCM frames. PCM is resampled **once** to 16 kHz and fanned out:
   - **ASR tap:** `ContiguousStreamSlicer` → 100 ms → `LiveCallTranscriber.offerStreamChunk`
   - **Classifier:** `CallWindowBuffer` → 1.0 s, hop 0.5 s → `LogLinearSpectrogram.process` → ONNX
3. `{ "type": "backfill" }` resends the last ~20 windows for that `callId`.
4. `{ "type": "end" }` closes the call.

Window message (JSON): `windowStartMs`, `windowEndMs`, `label`, `spoofProbability`, `confidence`, `latencyMs`, `spectrogram` (`rows`, `cols`, `values`), optional transcript fields.

Transcript message: `{ "type": "transcript", "callId", "text", "isFinal", "keywordHits", "keywordScore", "keywordSpans", "contentWords", "scamAlert" }`.

Place the ONNX file:

```bash
cp /path/to/your_model.onnx engine/models/voice_scam.onnx
```

Expected graph: float input **`[1, 1, 104, 63]`**. A different shape is logged as a warning; the engine still feeds 104×63.

---

## Frontend

```bash
cd frontend
npm install
npm run dev -- --host 0.0.0.0 --port 5173
```

- Vite, React 18, JavaScript only (`base: "./"` for Android assets)
- API origin: `frontend/src/api.js` → `http://127.0.0.1:8080` (override with `VITE_API_BASE`)
- Desktop: 360×780 Galaxy S23 frame
- Native WebView: `?native=1` or `window.AndroidBridge` skips the chassis
- CORS: add the LAN origin in `engine.properties` `cors.origins` if the UI is not on localhost

Stages: home icon → permissions → live monitor.

On the phone, `127.0.0.1` is the **phone**, not the Mac. For on-device analysis against the laptop engine, point `API_BASE` at the Mac LAN address (for example `http://10.92.1.30:8080`) and rebuild the APK. Otherwise spectrogram/ASR stay empty even though the UI runs.

---

## Android app

- Application id: `com.voiceshield.app`
- `minSdk` 26, `compileSdk` 35, current **versionCode 7**
- WebView loads `https://appassets.androidplatform.net/assets/www/index.html?native=1` (`WebViewAssetLoader`; `file://` would block ES modules)
- JS bridge `AndroidBridge`: `pickContact` (imports **all** contacts), `requestCallScreening`, `startCallMonitor` / `stopCallMonitor`

The app is **not** the default dialer. `VoiceShieldCallScreeningService` always **allows** the call through to Phone, then notifies VoiceShield only for unknown numbers. `PhoneStateReceiver` is a backup for `RINGING` / `IDLE`.

Permissions: `RECORD_AUDIO`, `READ_CONTACTS`, `READ_PHONE_STATE`, `READ_CALL_LOG`, `POST_NOTIFICATIONS`, microphone FGS.

### Build the APK

Needs JDK **17** for the Android Gradle plugin (this repo vendors `.tooling/jdk-17` and `.android-sdk`):

```bash
export JAVA_HOME="$PWD/.tooling/jdk-17/Contents/Home"
export ANDROID_HOME="$PWD/.android-sdk"
cd frontend && npm run build
rm -rf ../mobile/app/src/main/assets/www
mkdir -p ../mobile/app/src/main/assets/www
cp -R dist/. ../mobile/app/src/main/assets/www/
cd ../mobile && ./gradlew assembleDebug
```

Output: `mobile/app/build/outputs/apk/debug/app-debug.apk` (copied to `VoiceShield.apk` / `frontend/public/VoiceShield.apk`).

### Install on a Galaxy S23

Same Wi Fi as the Mac, Vite serving `public/VoiceShield.apk`:

```text
http://<mac-lan-ip>:5173/VoiceShield.apk
```

Example used in this project: `http://10.92.1.30:5173/VoiceShield.apk`.

Chrome → Allow from this source → Install. Uninstall the previous build if Android blocks the update.

**Default apps**

- Phone app: **Phone** (not VoiceShield)
- Caller ID & spam / call screening: **VoiceShield** (so unknown numbers can be seen)

---

## Configuration cheat sheet

| Key | Default | Meaning |
|-----|---------|---------|
| `onnx.modelPath` | `models/voice_scam.onnx` | Anti-spoof weights |
| `threshold` | `0.5` | Spoof probability cut |
| `audio.sampleRate` | `16000` | PCM / ASR / spectrogram |
| `audio.windowDurationS` | `1.0` | Classifier window |
| `audio.windowOverlapS` | `0.5` | Hop = 0.5 s |
| `asr.modelDir` | Zipformer folder | Streaming ASR |
| `keywords.path` | `keywords.txt` | Lexicon phrases |
| `alert.consecutiveSpoofWindows` | `3` | Extra spoof-window banner in the UI |

---

## Hard constraints (capture / ASR)

- `CallAudioCapture.onStream` → ASR only (contiguous, non-overlapping).
- `CallAudioCapture.onWindow` → spectrogram / ONNX only (1.0 s, 0.5 s hop, overlapping).
- Never call `acceptWaveform` / `decode` on the AudioRecord thread.
- Hang-up must not re-ring. Native monitor waits for real incoming events; the browser demo rings once.

---

## What is not in this product

- Recreating a Python SDK or PDF-based upload/batch console
- VoiceShield as the default Phone / InCall UI
- Natural-voice vs difference spectrogram panels
- Looping incoming-call UI after hang-up

---

## Health check

```bash
curl -s http://127.0.0.1:8080/health
```

Look for `modelLoaded`, `asrLoaded`, `windowDurationS` (should be `1.0`), and notes describing the 104×63 log-linear pipeline.


![Uploading image.png…]()

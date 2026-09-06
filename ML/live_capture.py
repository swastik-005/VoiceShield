"""
Bridge between the Android capture (app/android/CallAudioCapture.kt) and the
Python feature pipeline.

CallAudioCapture emits a FloatArray of 16000 mono samples in [-1, 1] every
0.5 s. This module turns those windows into model-ready tensors using
ml/features.py - the SAME extractor training uses, so there is no train/serve skew
on the Python side.

Three entry points:

  1. HTTP server - the phone POSTs each window to /ingest.
         python ml/live_capture.py --serve --port 8765
     Wire format: raw little-endian float32 bytes (16000 floats = 64000 bytes),
     Content-Type: application/octet-stream.

  2. Offline replay - feed a WAV through as if it were arriving live. Use this
     to test the whole path with no phone involved.
         python ml/live_capture.py --wav datasets/data/processed/human/demo.wav \
             --model ml/training/checkpoints/spoofcnn.pt

  3. Library - StreamingSpectrogram, NearFarGate, ScoreSmoother, Detector.

Pass --model with a ml/run_training.py checkpoint to score live windows. Without
one every window scores 0.5 - deliberately not faked, because a made-up score
makes a demo look like it works while measuring nothing.

TWO GATES BEFORE THE MODEL
--------------------------
Live audio is not a clean stream of the caller talking. Two things have to be
rejected first, or they poison the rolling average:

  SILENCE. Ringback, hold music, line noise, dead air. AdaptiveVad drops
  anything that does not stand clear of the running noise floor.

  A FIXED threshold used to be used here, on the grounds that it matched
  preprocess.SILENCE_RMS. It did not match anything: preprocess applies 0.005
  to the CLEAN CACHED SOURCE waveform when choosing training windows, while
  this path applies it to a POST-AEC ACOUSTIC RESIDUAL picked up off a
  speakerphone. On a handset whose echo canceller suppresses the caller by
  30 dB - which is an ordinary Samsung on speaker - a fixed 0.005 rejects the
  entire call as silence, and the detector reports nothing at all with no error
  to say why. --fixed-vad restores the old behaviour for clean offline audio.

  THE NEAR-END SPEAKER. This is the one that is easy to miss. The app listens
  on speakerphone, so the mic hears BOTH parties - and the phone's owner is
  much closer and much louder than the handset's own loudspeaker. Those windows
  are genuine human speech, they score "human", and they drag the caller's
  score toward human exactly when the user answers a question. NearFarGate
  rejects windows far louder than the running median.

  IMPORTANT: NearFarGate needs true levels. Construct CallAudioCapture with
  normalise=false for this path (now the default) - its applyGain() scales every
  window toward a fixed target RMS, which erases the loudness difference the
  gate depends on. The CNN does not need that gain anyway: ml/features.py
  standardises each window, so the model is already exactly scale-invariant.
  Keep the gain for the speech-to-text branch, which does care about level.

  CallAudioCapture's SESSION gain is a different thing and is safe here: one
  constant held for the whole call, so it scales every window identically and
  leaves every ratio - and therefore this gate - untouched.

THE OTHER BRANCH
----------------
Scam-phrase detection runs in parallel off the same capture, on its own clock -
speech-to-text is seconds behind, and it does not need to be fast, because scam
intent accumulates over a whole call. Nothing here waits for it. /ingest accepts
an optional scam score from that branch and reports the 2x2 verdict; if none
arrives, the AI-vs-human half stands on its own.
"""

from __future__ import annotations

import argparse
import struct
import sys
import time
from collections import deque
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import torch

from features import (
    DEFAULT_BAND_CROP,
    HOP_WINDOW_SAMPLES,
    SAMPLE_RATE,
    WINDOW_SAMPLES,
    LogLinearSpectrogram,
    as_mono_window,
    feature_shape,
    spectrogram_from_window,
)

# How many recent window scores to average before reporting anything. Single
# windows are noisy and produce flickering false alarms; ~5-8 s of context is
# the useful unit. At a 0.5 s hop, 12 windows = 6 s.
SMOOTHING_WINDOWS = 12

# Nominal speech level, and the threshold used when --fixed-vad is passed.
#
# This is NOT "matched to preprocess.SILENCE_RMS" - see the module docstring.
# It is the level CallAudioCapture's session gain aims to clear, so calibration
# can report up front when the caller cannot be lifted above the gate.
VAD_RMS_THRESHOLD = 0.005

# Adaptive silence gate. Mirrored exactly in Gates.kt GateConfig.
#
# The adaptive threshold is CLAMPED BETWEEN VAD_ABSOLUTE_FLOOR AND
# VAD_RMS_THRESHOLD, and the ceiling is the important half.
#
# This gate exists to LOWER the bar when the capture is quiet - to rescue an AEC
# residual that a fixed 0.005 would throw away - and for nothing else. Letting
# it rise above the fixed threshold turns it into a stricter VAD, which is a
# different feature and a worse one: the noise-floor estimator is a low
# percentile of recent window levels, and on a recording that is nearly all
# speech that percentile tracks QUIET SPEECH, not silence. Uncapped, this
# rejected 940 of 1200 windows of clean studio audio that the fixed gate passed
# in full.
#
# With the ceiling in place the gate is provably never stricter than the one it
# replaces: it accepts everything the fixed gate accepted, plus quiet audio the
# fixed gate could not see. That one-way property is what makes it safe to turn
# on by default.
VAD_ABSOLUTE_FLOOR = 5e-4
VAD_NOISE_MULT = 3.0            # speech must beat the floor by this much (~9.5 dB)
VAD_NOISE_PERCENTILE = 10       # which percentile of recent levels is "the floor"
VAD_NOISE_HISTORY = 120         # 60 s at a 0.5 s hop
VAD_NOISE_MIN_SAMPLES = 20      # windows before the estimate is trusted

# Near-end rejection. A window whose RMS exceeds this multiple of the running
# median is treated as the phone's owner talking, not the caller.
NEAR_FAR_RATIO = 3.0


def apply_relay_profile() -> None:
    """
    Switch the gates to the Bluetooth relay's profile, mirroring
    GateProfile.RELAY in app/mobile/.../detect/Gates.kt.

    Two different signals, two different gates. Relayed call audio is line
    level and contains only the far end, so the speakerphone constants are
    wrong twice over: the silence gate sits low enough for line noise to clear
    it, and near-end rejection would throw away the caller's loudest windows
    for resembling a talker who is not in the stream.

    A module-level rebind rather than a profile object threaded through
    replay_wav() and serve(): both read these constants directly, and one
    assignment keeps the Python and Kotlin sides provably reading the same
    numbers rather than two copies that can drift.
    """
    global VAD_RMS_THRESHOLD, VAD_ABSOLUTE_FLOOR
    VAD_RMS_THRESHOLD = 0.02
    VAD_ABSOLUTE_FLOOR = 0.002
    print("relay gate profile: vad_threshold=%.3f floor=%.4f near-far=off"
          % (VAD_RMS_THRESHOLD, VAD_ABSOLUTE_FLOOR))
NEAR_FAR_HISTORY = 120          # 60 s at a 0.5 s hop
NEAR_FAR_MIN_SAMPLES = 20       # 10 s before the gate is trusted at all

# Fallback decision thresholds, used ONLY when nothing better is available.
#
# The real P(AI) threshold is the EER operating point measured on the held-out
# set and written into the checkpoint by ml/run_training.py; Detector reads it and
# passes it to verdict(). 0.5 is where a sigmoid happens to cross, not where the
# false-accept and false-reject rates meet, and on an off-centre model the two
# differ enough to throw away separation the model genuinely has.
#
# P_SCAM_THRESHOLD stays 0.5 because the text branch has no calibrated model to
# derive an operating point from yet. When it does, it should follow P(AI).
P_AI_THRESHOLD = 0.5
P_SCAM_THRESHOLD = 0.5


def has_speech(samples: torch.Tensor, threshold: float = VAD_RMS_THRESHOLD) -> bool:
    """
    Crude FIXED level gate. Enough to reject silence and ringback on clean
    offline audio, and wrong on a real speakerphone capture - see AdaptiveVad.
    Reached via --fixed-vad.
    """
    if samples.numel() == 0:
        return False
    return rms(samples) >= threshold


def rms(samples: torch.Tensor) -> float:
    return float(samples.pow(2).mean().sqrt())


class AdaptiveVad:
    """
    Silence gate that learns the noise floor instead of assuming it.

    Tracks a low percentile of recent window levels - on any real call that is
    the line noise, room tone and dead air - and calls a window speech when it
    stands VAD_NOISE_MULT above it, subject to an absolute floor so that pure
    quantisation noise can never be promoted to speech, and to a CEILING at
    VAD_RMS_THRESHOLD so it can never be stricter than the fixed gate it
    replaces. Read the constants block for why the ceiling is not optional.

    EVERY window's level is recorded, including rejected ones. The floor is an
    estimate of the quiet end of the distribution, so feeding it only the loud
    windows would make it chase the signal and slowly close the gate.

    Before VAD_NOISE_MIN_SAMPLES windows there is no floor worth having, so the
    absolute minimum is used alone. That errs permissive on purpose: a window
    wrongly admitted costs one inference, while a window wrongly rejected is
    gone, and early rejections also starve the near/far median that runs next.

    Mirrors Gates.kt AdaptiveVad. The percentile index is deliberately plain
    integer arithmetic rather than torch.quantile or numpy.percentile: those
    interpolate between neighbouring samples, and the two languages would have
    to agree on the interpolation convention as well as the data. `n * p // 100`
    cannot disagree.
    """

    def __init__(self, absolute_floor: float | None = None,
                 ceiling: float | None = None,
                 mult: float = VAD_NOISE_MULT,
                 percentile: int = VAD_NOISE_PERCENTILE,
                 history: int = VAD_NOISE_HISTORY,
                 min_samples: int = VAD_NOISE_MIN_SAMPLES):
        # Resolved HERE, not in the signature. A default argument is bound
        # once, when the def executes, so a profile switch that rebinds the
        # module constants later (--relay-gates) would leave these two frozen
        # at the acoustic values - and the run would report the profile it was
        # not using.
        self.absolute_floor = (
            VAD_ABSOLUTE_FLOOR if absolute_floor is None else absolute_floor
        )
        self.ceiling = VAD_RMS_THRESHOLD if ceiling is None else ceiling
        self.mult = mult
        self.percentile = percentile
        self.history = history
        self.min_samples = min_samples
        self.levels: list[float] = []

    def noise_floor(self) -> float:
        """The current estimate, or 0.0 before enough windows have been seen."""
        n = len(self.levels)
        if n < self.min_samples:
            return 0.0
        s = sorted(self.levels)
        return s[min((n * self.percentile) // 100, n - 1)]

    def threshold(self) -> float:
        """
        The level a window must exceed right now, clamped to
        [absolute_floor, ceiling].

        The ceiling is what keeps this from becoming a stricter VAD than the
        fixed one - see the constants block. Only the downward half of the
        adaptation is wanted.
        """
        t = max(self.absolute_floor, self.noise_floor() * self.mult)
        return min(t, self.ceiling)

    def accept(self, samples: torch.Tensor) -> tuple[bool, float, float]:
        """(is speech, level, threshold). Threshold is read BEFORE the add."""
        level = rms(samples) if samples.numel() else 0.0
        t = self.threshold()

        self.levels.append(level)
        if len(self.levels) > self.history:
            self.levels.pop(0)

        return (samples.numel() > 0 and level >= t), level, t

    def reset(self) -> None:
        self.levels.clear()


class NearFarGate:
    """
    Reject windows that are probably the phone's owner rather than the caller.

    Crude by design: the near-end talker is metres closer to the mic than the
    handset's loudspeaker is, so they arrive far louder. Track the running
    median of recent speech levels - which the caller dominates on a call where
    the caller is doing the talking - and reject anything well above it.

    Deliberately conservative. Rejecting a caller window costs half a second of
    context; accepting a near-end window feeds real human speech into the
    caller's score and actively pulls the verdict the wrong way.
    """

    def __init__(self, ratio: float = NEAR_FAR_RATIO,
                 history: int = NEAR_FAR_HISTORY,
                 min_samples: int = NEAR_FAR_MIN_SAMPLES,
                 enabled: bool = True):
        self.ratio = ratio
        self.min_samples = min_samples
        self.enabled = enabled
        self._levels: deque[float] = deque(maxlen=history)

    def median(self) -> float:
        if not self._levels:
            return 0.0
        s = sorted(self._levels)
        return s[len(s) // 2]

    def accept(self, samples: torch.Tensor) -> tuple[bool, float]:
        """
        Returns (accept, level). Always records the level, gate or no gate, so
        the median keeps tracking the call rather than only its quiet half.
        """
        level = rms(samples)
        med = self.median()
        n = len(self._levels)
        self._levels.append(level)
        if not self.enabled or n < self.min_samples or med <= 0:
            return True, level
        return level <= self.ratio * med, level

    def reset(self) -> None:
        self._levels.clear()


# ------------------------------------------------------------------ decoding

def decode_float32_le(payload: bytes) -> torch.Tensor:
    """Raw little-endian float32 bytes -> [N] float tensor."""
    if len(payload) % 4 != 0:
        raise ValueError(
            f"payload of {len(payload)} bytes is not a whole number of float32s"
        )
    n = len(payload) // 4
    return torch.tensor(struct.unpack(f"<{n}f", payload), dtype=torch.float32)


def decode_pcm16_le(payload: bytes) -> torch.Tensor:
    """Raw little-endian int16 PCM -> [N] float tensor in [-1, 1]."""
    if len(payload) % 2 != 0:
        raise ValueError(f"payload of {len(payload)} bytes is not whole int16s")
    n = len(payload) // 2
    ints = struct.unpack(f"<{n}h", payload)
    return torch.tensor(ints, dtype=torch.float32) / 32768.0


# ----------------------------------------------------------------- streaming

class StreamingSpectrogram:
    """
    Continuous sample stream -> a spectrogram every `hop` samples.

    Use this when the source is a raw stream rather than pre-cut windows (the
    --wav replay, or a socket delivering arbitrary chunk sizes). If the phone
    is sending complete 1 s windows, call Detector.spectrogram() directly -
    re-windowing already-windowed data would double-count overlap.

    push() returns (spectrogram, samples) pairs so the caller can gate on the
    audio it came from without recomputing anything.
    """

    def __init__(self, window: int = WINDOW_SAMPLES, hop: int = HOP_WINDOW_SAMPLES,
                 band_crop: bool | None = None,
                 extractor: LogLinearSpectrogram | None = None):
        self.window = window
        self.hop = hop
        self.extractor = extractor or LogLinearSpectrogram(band_crop=band_crop)
        self._buf = torch.zeros(0, dtype=torch.float32)

    def push(self, samples) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Add samples; return a (possibly empty) list of (spectrogram, window)."""
        if not isinstance(samples, torch.Tensor):
            samples = torch.as_tensor(samples, dtype=torch.float32)
        self._buf = torch.cat([self._buf, samples.flatten().to(torch.float32)])

        out = []
        while self._buf.shape[0] >= self.window:
            chunk = self._buf[: self.window]
            out.append((self.extractor(chunk.unsqueeze(0)), chunk))
            self._buf = self._buf[self.hop :]
        return out

    def flush(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Emit a final zero-padded window if a partial tail remains."""
        if self._buf.shape[0] == 0:
            return []
        tail = self._buf
        self._buf = torch.zeros(0, dtype=torch.float32)
        padded = as_mono_window(tail)
        return [(self.extractor(padded), padded.squeeze(0))]


class ScoreSmoother:
    """Rolling mean over recent window scores, per the 5-8 s averaging rule."""

    def __init__(self, n: int = SMOOTHING_WINDOWS):
        self.n = n
        self._scores: deque[float] = deque(maxlen=n)

    def add(self, score: float) -> float:
        self._scores.append(float(score))
        return sum(self._scores) / len(self._scores)

    @property
    def value(self) -> float:
        return sum(self._scores) / len(self._scores) if self._scores else 0.5

    @property
    def ready(self) -> bool:
        """False until enough context has accumulated to report honestly."""
        return len(self._scores) >= self.n

    def reset(self) -> None:
        self._scores.clear()


# ----------------------------------------------------------------- inference

class Detector:
    """
    Loads a ml/run_training.py checkpoint and scores windows.

    The architecture is rebuilt by training.model.from_checkpoint(), which reads
    every switch out of the checkpoint's model_config. Nothing is assumed here -
    a model trained band-cropped, without the coord channel, or with a different
    time pooling cannot silently be rebuilt the other way. That mismatch is what
    produces a model which loads without error and then outputs confident
    nonsense.
    """

    def __init__(self, checkpoint_path: str | Path, device: str = "cpu"):
        from training.model import from_checkpoint

        ckpt = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
        self.model = from_checkpoint(ckpt).to(device)
        self.model.eval()
        self.device = device

        cfg = ckpt.get("model_config", {})
        self.n_freq_bins = int(cfg.get("n_freq_bins", self.model.n_freq_bins))
        self.band_crop = bool(ckpt.get("band_crop", DEFAULT_BAND_CROP))
        self.mfm = bool(cfg.get("mfm", True))
        self.time_pool = str(cfg.get("time_pool", "mean+max"))
        self.holdout = ckpt.get("holdout_groups") or ckpt.get("test_groups") or []
        self.epoch = ckpt.get("epoch")

        # Operating point measured on the held-out set, written back into the
        # checkpoint by ml/run_training.py. 0.5 is where the training loss stopped
        # caring, not where the two error types balance; on a model whose
        # sigmoid is off-centre the two are far apart, and cutting at 0.5 throws
        # away separation the model actually has. Checkpoints from before this
        # field existed fall back to 0.5, which is what the code did then.
        thr = ckpt.get("eer_threshold")
        self.threshold = float(thr) if thr is not None and thr == thr else P_AI_THRESHOLD
        self.threshold_source = ("EER on held-out set" if thr is not None and thr == thr
                                 else "default 0.5 (checkpoint predates the field)")

        self.extractor = LogLinearSpectrogram(band_crop=self.band_crop)
        # Guard: the checkpoint's bin count must match what the extractor emits.
        got = self.extractor(torch.zeros(1, WINDOW_SAMPLES)).shape[-2]
        if got != self.n_freq_bins:
            raise ValueError(
                f"checkpoint expects {self.n_freq_bins} frequency bins but "
                f"ml/features.py produces {got} with band_crop={self.band_crop}. "
                f"The checkpoint and ml/features.py disagree - retrain, or fix "
                f"BAND_CROP_LO/HI in ml/features.py."
            )

    def spectrogram(self, samples) -> torch.Tensor:
        return spectrogram_from_window(samples, self.extractor)

    def score(self, spec: torch.Tensor) -> float:
        """P(AI voice) for one [1, F, T] spectrogram."""
        with torch.no_grad():
            logit = self.model(spec.unsqueeze(0).to(self.device))
            return float(torch.sigmoid(logit))

    def describe(self) -> str:
        band = (f"band-cropped [{self.n_freq_bins} bins]" if self.band_crop
                else f"full band [{self.n_freq_bins} bins]")
        arch = "LCNN/MFM" if self.mfm else "ReLU CNN"
        held = f", held out {len(self.holdout)} group(s)" if self.holdout else ""
        ep = f", epoch {self.epoch}" if self.epoch else ""
        thr = f", threshold {self.threshold:.3f} ({self.threshold_source})"
        return f"{arch}, {band}, time_pool={self.time_pool}{held}{ep}{thr}"


def score_window(spec: torch.Tensor, detector: "Detector | None" = None) -> float:
    """P(AI voice) for one window, or 0.5 when no model is loaded."""
    return 0.5 if detector is None else detector.score(spec)


# ------------------------------------------------------------------- verdict

def verdict(p_ai: float | None, p_scam: float | None,
            t_ai: float = P_AI_THRESHOLD,
            t_scam: float = P_SCAM_THRESHOLD) -> dict:
    """
    The 2x2 the app displays: quadrant of (P(AI) >= t_ai, P(scam) >= t_scam).

    The two scores are deliberately independent - different modality, different
    clock, different model. They are NOT fused into one four-way classifier:
    that would quarter the training data and invent couplings between two things
    that are close to independent in reality. Report both, label the quadrant.

    THE THRESHOLDS ARE ARGUMENTS, NOT 0.5. t_ai defaults to 0.5 only because a
    caller with no model has nothing better; when a Detector is loaded, its
    checkpoint's EER operating point is passed in instead. The returned dict
    carries the thresholds it used, so a logged verdict can be re-derived later
    rather than being a number whose meaning has been lost.
    """
    voice = None if p_ai is None else ("ai" if p_ai >= t_ai else "human")
    scam = None if p_scam is None else ("scam" if p_scam >= t_scam else "legitimate")
    if voice is None or scam is None:
        quadrant = None
    else:
        quadrant = f"{voice}+{scam}"
    return {"p_ai": p_ai, "p_scam": p_scam, "voice": voice,
            "scam": scam, "quadrant": quadrant,
            "t_ai": t_ai, "t_scam": t_scam}


def _json(obj: dict) -> str:
    import json
    return json.dumps(obj, default=lambda o: None)


# -------------------------------------------------------------------- replay

def replay_wav(path: Path, verbose: bool = True, detector=None,
               near_far: bool = True, fixed_vad: bool = False) -> int:
    """Push a WAV through the streaming path as if it arrived from the phone."""
    import soundfile as sf

    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    audio = torch.from_numpy(data).mean(dim=1)   # mono-ise
    if sr != SAMPLE_RATE:
        import torchaudio
        audio = torchaudio.functional.resample(
            audio.unsqueeze(0), sr, SAMPLE_RATE).squeeze(0)
        if verbose:
            print(f"resampled {sr} -> {SAMPLE_RATE} Hz")

    extractor = detector.extractor if detector else LogLinearSpectrogram()
    stream = StreamingSpectrogram(extractor=extractor)
    smoother = ScoreSmoother()
    gate = NearFarGate(enabled=near_far)
    vad = None if fixed_vad else AdaptiveVad()
    n = scored = silent = near = 0

    def handle(spec, samples):
        nonlocal n, scored, silent, near
        n += 1
        speech = has_speech(samples) if vad is None else vad.accept(samples)[0]
        if not speech:
            silent += 1
            return
        ok, _ = gate.accept(samples)
        if not ok:
            near += 1
            return
        scored += 1
        avg = smoother.add(score_window(spec, detector))
        if verbose and scored % 10 == 0:
            state = f"{avg:.3f}" if smoother.ready else "warming up"
            print(f"  window {n:4d}  shape={tuple(spec.shape)}  smoothed={state}")

    # 4096-sample chunks, i.e. deliberately not aligned to the window size, so
    # the ring buffer's partial-chunk handling actually gets exercised.
    for i in range(0, audio.shape[0], 4096):
        for spec, samples in stream.push(audio[i : i + 4096]):
            handle(spec, samples)
    for spec, samples in stream.flush():
        handle(spec, samples)

    dur = audio.shape[0] / SAMPLE_RATE
    print(f"\n{path.name}: {dur:.1f}s -> {n} windows of "
          f"{tuple(feature_shape(detector.band_crop if detector else None))}")
    print(f"  scored {scored}, skipped {silent} silent, {near} near-end")
    # Printed in the same shape as WindowGates.describe() on the phone, so a
    # dumped capture can be compared line to line. Divergent counts on the same
    # audio mean the Kotlin and Python gates have drifted apart - which is the
    # single failure this replay path exists to catch.
    print(f"  gates total={n} silent={silent} nearEnd={near} scored={scored} "
          f"floor={0.0 if vad is None else vad.noise_floor():.5f} "
          f"nearFarMedian={gate.median():.5f} "
          f"vad={'fixed' if vad is None else 'adaptive'}")
    if detector is None:
        print("  NO MODEL LOADED - every window scored 0.5, so there is no "
              "verdict to report.\n  Pass --model with a ml/run_training.py checkpoint.")
    elif scored:
        v = verdict(smoother.value, None, t_ai=detector.threshold)
        print(f"  P(AI voice) = {v['p_ai']:.3f} -> {v['voice']}"
              f"   (threshold {v['t_ai']:.3f})"
              + ("" if smoother.ready else "   (short clip - not enough context)"))
    return n


# -------------------------------------------------------------------- server

def serve(host: str, port: int, save_dir: Path | None, detector=None,
          near_far: bool = True, fixed_vad: bool = False) -> None:
    """Minimal HTTP endpoint the phone POSTs capture windows to."""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    # Note "scam_score", not "scam": verdict() already returns a "scam" key
    # holding the LABEL, and merging a state dict that reuses the name would
    # silently overwrite the label with the raw number in every response.
    state = {"count": 0, "scored": 0, "silent": 0, "near": 0, "scam_score": None}

    # Resolved once per server rather than per request: the threshold belongs to
    # the loaded checkpoint, not to the call. Without a model there is nothing
    # calibrated to use, and every window scores 0.5 anyway.
    t_ai = detector.threshold if detector is not None else P_AI_THRESHOLD
    smoother = ScoreSmoother()
    gate = NearFarGate(enabled=near_far)
    vad = None if fixed_vad else AdaptiveVad()
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass  # too chatty at two windows a second

        def _reply(self, code: int, obj: dict):
            payload = _json(obj).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            route = urlparse(self.path).path
            if route == "/health":
                self._reply(200, {"ok": True, **state,
                                  "model": detector.describe() if detector else None})
            elif route == "/verdict":
                v = verdict(smoother.value if state["scored"] else None,
                            state["scam_score"], t_ai=t_ai)
                self._reply(200, {**v, "ready": smoother.ready, **state})
            else:
                self._reply(404, {"error": "try /health, /verdict or POST /ingest"})

        def do_POST(self):
            parsed = urlparse(self.path)
            route, query = parsed.path, parse_qs(parsed.query)

            # The speech-to-text branch posts its own score here, on its own
            # clock. Nothing in the audio path blocks on it.
            if route == "/scam":
                try:
                    state["scam_score"] = float(query.get("p", ["nan"])[0])
                except ValueError:
                    self._reply(400, {"error": "pass ?p=<0..1>"})
                    return
                self._reply(200, verdict(smoother.value if state["scored"] else None,
                                         state["scam_score"], t_ai=t_ai))
                return

            if route != "/ingest":
                self._reply(404, {"error": "POST to /ingest or /scam"})
                return
            try:
                length = int(self.headers.get("Content-Length", 0))
            except ValueError:
                self._reply(400, {"error": "bad Content-Length"})
                return
            if length <= 0:
                self._reply(400, {"error": "empty body"})
                return

            payload = b""
            while len(payload) < length:
                part = self.rfile.read(length - len(payload))
                if not part:
                    break
                payload += part

            try:
                fmt = query.get("fmt", ["float32"])[0]
                samples = (decode_pcm16_le(payload) if fmt == "pcm16"
                           else decode_float32_le(payload))
            except Exception as e:
                self._reply(400, {"error": f"{type(e).__name__}: {e}"})
                return

            state["count"] += 1

            speech = has_speech(samples) if vad is None else vad.accept(samples)[0]
            if not speech:
                # Silence, ringback or hold music. Scoring it would poison the
                # rolling average with a meaningless number.
                state["silent"] += 1
                self._reply(200, {"n": state["count"], "speech": False,
                                  "score": None, "reason": "silence"})
                return

            accepted, level = gate.accept(samples)
            if not accepted:
                # Almost certainly the phone's owner, not the caller.
                state["near"] += 1
                self._reply(200, {"n": state["count"], "speech": True,
                                  "score": None, "reason": "near_end",
                                  "level": round(level, 5)})
                return

            try:
                spec = (detector.spectrogram(samples) if detector
                        else spectrogram_from_window(samples))
            except Exception as e:
                self._reply(400, {"error": f"{type(e).__name__}: {e}"})
                return

            state["scored"] += 1
            avg = smoother.add(score_window(spec, detector))
            if save_dir:
                torch.save({"spectrogram": spec, "ts": time.time()},
                           save_dir / f"window_{state['count']:06d}.pt")
            if state["scored"] % 20 == 0:
                print(f"  {state['scored']} scored  P(ai)={avg:.3f}  "
                      f"({state['silent']} silent, {state['near']} near-end)")
            self._reply(200, {"n": state["count"], "speech": True,
                              "score": round(avg, 4), "ready": smoother.ready,
                              **verdict(avg, state["scam_score"], t_ai=t_ai)})

    srv = ThreadingHTTPServer((host, port), Handler)
    print(f"listening on http://{host}:{port}")
    print(f"  model: {detector.describe() if detector else 'NONE - every window scores 0.5'}")
    print(f"  feature: {feature_shape(detector.band_crop if detector else None)}")
    print(f"  near-end gate: {'on' if near_far else 'off'}"
          + ("   (send un-normalised audio: CallAudioCapture(normalise=false))"
             if near_far else ""))
    print(f"  silence gate:  {'fixed at %.4f' % VAD_RMS_THRESHOLD if fixed_vad else 'adaptive'}")
    print(f"  POST /ingest            raw float32-LE window "
          f"({WINDOW_SAMPLES} floats = {WINDOW_SAMPLES * 4} bytes)")
    print(f"  POST /ingest?fmt=pcm16  raw int16-LE instead")
    print(f"  POST /scam?p=0.83       scam score from the speech-to-text branch")
    print(f"  GET  /verdict           current 2x2 verdict")
    print(f"  GET  /health")
    if save_dir:
        print(f"  saving windows to {save_dir.resolve()}")
    print("Phone and laptop must be on the same network; use the laptop's LAN IP.")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
        srv.server_close()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="ml/live_capture.py",
        description="Bridge Android capture windows into the Python feature pipeline",
    )
    p.add_argument("--serve", action="store_true", help="Run the HTTP ingest server")
    p.add_argument("--host", default="0.0.0.0", help="Bind address (default: all interfaces)")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--save-dir", default=None, help="Persist each window as a .pt")
    p.add_argument("--wav", help="Offline: replay an audio file through the live path")
    p.add_argument("--model", help="Path to a ml/run_training.py checkpoint (.pt). "
                                   "Without it every window scores 0.5.")
    p.add_argument("--no-near-far", action="store_true",
                   help="Disable near-end speaker rejection. Do this if the "
                        "phone is sending gain-normalised windows, where the "
                        "level difference the gate needs has been erased.")
    p.add_argument("--relay-gates", action="store_true",
                   help="Score with the Bluetooth relay's gate profile instead "
                        "of the speakerphone one: near-end rejection off (a "
                        "relayed downlink contains no near-end talker) and the "
                        "silence gate raised for line-level audio. Match this "
                        "to GateProfile.RELAY in app/mobile/.../detect/Gates.kt, or "
                        "a replay of a relay recording will not reproduce the "
                        "counts the phone reported.")
    p.add_argument("--fixed-vad", action="store_true",
                   help="Use the old fixed %.3f RMS silence threshold instead "
                        "of the adaptive noise-floor gate. Reasonable on clean "
                        "studio audio; on a real speakerphone capture it "
                        "rejects the whole call as silence."
                        % VAD_RMS_THRESHOLD)
    a = p.parse_args(argv)

    detector = None
    if a.model:
        if not Path(a.model).is_file():
            print(f"checkpoint not found: {a.model}")
            return 1
        detector = Detector(a.model)
        print(f"loaded model: {detector.describe()}")

    near_far = not a.no_near_far
    if a.relay_gates:
        apply_relay_profile()
        near_far = False
    if a.wav:
        path = Path(a.wav)
        if not path.is_file():
            print(f"file not found: {path}")
            return 1
        replay_wav(path, detector=detector, near_far=near_far,
                   fixed_vad=a.fixed_vad)
        return 0
    if a.serve:
        serve(a.host, a.port, Path(a.save_dir) if a.save_dir else None,
              detector, near_far, fixed_vad=a.fixed_vad)
        return 0

    p.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())

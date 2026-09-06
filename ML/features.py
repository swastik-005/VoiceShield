"""
Canonical feature extraction for the voice-scam detector.

SINGLE SOURCE OF TRUTH. Preprocessing, training and live capture all import
from here, so they cannot drift apart. When the Kotlin on-device spectrogram is
written, it must reproduce THESE numbers exactly - see PARITY below.

Feature spec
------------
    sample rate   16000 Hz, mono
    window        1.0 s  = 16000 samples (matches CallAudioCapture.WINDOW_SAMPLES)
    n_fft         512    -> 257 frequency bins (n_fft // 2 + 1)
    hop_length    256
    win_length    512, Hann (periodic)
    center        True, reflect padding -> 63 frames (1 + 16000 // 256)
    magnitude     power=1.0 (LINEAR magnitude, NOT mel, NOT power)
    log           natural log(magnitude + 1e-6)
    floor         clamp TOP_DB below the window's own peak
    normalise     per-window: subtract mean, divide by std over the whole 2-D
                  spectrogram (not per-band, not per-frame)
    band crop     ON by default -> bins 8..111 inclusive = 104 bins

    output shape  [1, 104, 63]

Linear, not mel: mel spacing compresses the high frequencies, and the harmonic
fine structure this model keys on is evenly spaced in Hz, not in mel. Within
the telephone band, linear spacing keeps harmonic spacing constant across the
whole axis, which is what makes "too-regular harmonics" representable by a
single convolution kernel.

Band crop is ON by default. The caller's voice crosses the phone network before
reaching the mic, which band-limits it to roughly 300-3400 Hz. Measured on this
repo's sample audio, bins 112-256 hold 0.1% of the energy after the codec: 57%
of the tensor, almost none of the signal. Cropping triples the effective sample
efficiency on a small dataset and costs nothing.

DEFAULTS
--------
Every default in this file, in model.py and in ml/live_capture.py resolves through
DEFAULT_BAND_CROP. Do not hardcode 104 or 257 anywhere; call feature_shape().

PARITY
------
The Kotlin implementation must match on every one of: sample rate, n_fft, hop,
window function AND its periodic/symmetric flag, centering + pad mode, whether
magnitude or power is taken, the log epsilon, the dynamic-range floor, the band
crop bounds, and the ORDER of the three post-steps (log -> floor -> normalise).
A mismatch in any one of them feeds the deployed model inputs unlike anything it
trained on, while training metrics still look perfect.

To check parity once the Kotlin exists, run:
    python ml/features.py --emit-reference
which writes a reference WAV plus the expected tensor as CSV. Run the same WAV
through the Kotlin path and diff - they should agree to ~3 decimal places.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torchaudio.transforms import Spectrogram

# ---------------------------------------------------------------- constants

SAMPLE_RATE = 16000
WINDOW_SECONDS = 1.0
WINDOW_SAMPLES = 16000          # CallAudioCapture.WINDOW_SAMPLES
HOP_SECONDS = 0.5
HOP_WINDOW_SAMPLES = 8000       # CallAudioCapture.HOP_SAMPLES

N_FFT = 512
HOP_LENGTH = 256
WIN_LENGTH = 512

N_FREQ_BINS = N_FFT // 2 + 1                    # 257
N_FRAMES = 1 + WINDOW_SAMPLES // HOP_LENGTH     # 63
FEATURE_SHAPE = (1, N_FREQ_BINS, N_FRAMES)      # [1, 257, 63]

# Width of one FFT bin. model.py imports this to report receptive fields in Hz.
BIN_HZ = SAMPLE_RATE / N_FFT                    # 31.25
FRAME_MS = 1000.0 * HOP_LENGTH / SAMPLE_RATE    # 16.0

LOG_EPS = 1e-6
NORM_EPS = 1e-5

# Below this spread, the window is treated as degenerate and standardisation
# returns zeros. See the note in LogLinearSpectrogram.__call__ - this is not a
# tuning knob, it is a guard against dividing by numerical noise.
DEGENERATE_STD = 1e-4

# Dynamic-range floor, in dB below the window's own peak. Converted to natural
# log units, which is what the spectrogram is in: 1 nat = 20/ln(10) = 8.6859 dB.
TOP_DB = 80.0
DB_PER_NAT = 8.685889638065035

# Telephone passband in bin indices. Bin width is 31.25 Hz, so 250 Hz -> bin 8
# and 3500 Hz -> bin 112. The slice is [8:112], i.e. bins 8..111 inclusive,
# which is 104 bins spanning 250-3500 Hz - a little wider than the nominal
# 300-3400 Hz passband so the filter skirts are included rather than clipped.
BAND_CROP_LO = 8
BAND_CROP_HI = 112
N_CROPPED_BINS = BAND_CROP_HI - BAND_CROP_LO    # 104
CROPPED_SHAPE = (1, N_CROPPED_BINS, N_FRAMES)   # [1, 104, 63]

# The pipeline default. ml/preprocess.py, ml/run_training.py and ml/live_capture.py all
# resolve to this unless a checkpoint says otherwise.
DEFAULT_BAND_CROP = True


def feature_shape(band_crop: bool | None = None) -> tuple[int, int, int]:
    """
    Output shape for the given config. Use this instead of hardcoding.

    None means "the pipeline default", which is DEFAULT_BAND_CROP. Pass an
    explicit True/False only when a checkpoint pins it.
    """
    if band_crop is None:
        band_crop = DEFAULT_BAND_CROP
    return CROPPED_SHAPE if band_crop else FEATURE_SHAPE


def n_freq_bins(band_crop: bool | None = None) -> int:
    """Number of frequency bins for the given config."""
    return feature_shape(band_crop)[1]


class LogLinearSpectrogram:
    """
    Waveform [1, T] -> log linear-magnitude spectrogram [1, F, frames].

    For the standard 1 s window the output is [1, 104, 63] band-cropped, or
    [1, 257, 63] with band_crop=False. Frozen defaults - change them here and
    preprocessing, training and inference all change together, which is the
    point of this module.
    """

    def __init__(
        self,
        n_fft: int = N_FFT,
        hop_length: int = HOP_LENGTH,
        win_length: int = WIN_LENGTH,
        log_eps: float = LOG_EPS,
        normalise: bool = True,
        top_db: float | None = TOP_DB,
        band_crop: bool | None = None,
    ):
        self.log_eps = log_eps
        self.normalise = normalise
        self.top_db = top_db
        self.band_crop = DEFAULT_BAND_CROP if band_crop is None else bool(band_crop)
        self.spec = Spectrogram(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window_fn=torch.hann_window,   # periodic Hann; Kotlin must match
            power=1.0,                     # magnitude, not power
            center=True,
            pad_mode="reflect",
            normalized=False,
        )

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        mag = self.spec(waveform)
        out = torch.log(mag + self.log_eps)

        if self.band_crop:
            out = out[..., BAND_CROP_LO:BAND_CROP_HI, :]

        if self.top_db is not None:
            # Clamp to a fixed dynamic range below this window's own peak.
            # Without this, standardisation stretches the near-silent band
            # above 3.4 kHz - which the phone codec already emptied - into
            # full-scale noise, burying the speech detail underneath it.
            floor = out.max() - (self.top_db / DB_PER_NAT)
            out = torch.clamp(out, min=float(floor))

        if self.normalise:
            # Per-window standardisation. Costs a little on clean data, but
            # keeps wildly varying phone recording levels from shifting the
            # input distribution at inference time. Applied AFTER the log and
            # AFTER the floor - the Kotlin port must use this same order.
            #
            # Note this also makes the CNN exactly scale-invariant, so the
            # gain CallAudioCapture.applyGain() applies has no effect on the
            # model's input. Keep that gain for the speech-to-text branch,
            # which does care about level.
            #
            # DEGENERATE WINDOWS. On digital silence every magnitude is 0, so
            # every value here is exactly log(LOG_EPS) and the spread is zero.
            # Without the guard below, `out - out.mean()` is not zero but
            # float32 summation error, and dividing that by (0 + NORM_EPS)
            # amplifies pure noise into a confident-looking constant around
            # -0.32 - which the CNN would then classify. Worse, the value
            # depends on summation order, so it differs between CPU and GPU and
            # between torch and any port. Zeros are the correct z-score for a
            # constant array, and they are reproducible.
            std = out.std()                        # unbiased: divides by N-1
            if float(std) < DEGENERATE_STD:
                out = torch.zeros_like(out)
            else:
                out = (out - out.mean()) / (std + NORM_EPS)
        return out

    def config(self) -> dict:
        return {"band_crop": self.band_crop, "top_db": self.top_db,
                "normalise": self.normalise}


def fix_length(waveform: torch.Tensor, n_samples: int = WINDOW_SAMPLES) -> torch.Tensor:
    """Centre-crop or zero-pad a [1, T] waveform to exactly n_samples."""
    n = waveform.shape[-1]
    if n > n_samples:
        start = (n - n_samples) // 2
        return waveform[:, start : start + n_samples]
    if n < n_samples:
        return torch.nn.functional.pad(waveform, (0, n_samples - n))
    return waveform


def as_mono_window(window) -> torch.Tensor:
    """
    Anything tensor-like holding mono float samples -> [1, WINDOW_SAMPLES].

    Accepts the FloatArray CallAudioCapture emits (decoded on this side), a
    numpy array, or a list. Short or long input is centre-cropped/padded rather
    than rejected, so a ragged final chunk still scores.
    """
    if not isinstance(window, torch.Tensor):
        window = torch.as_tensor(window, dtype=torch.float32)
    window = window.to(torch.float32)
    if window.ndim == 1:
        window = window.unsqueeze(0)
    elif window.ndim == 2 and window.shape[0] > 1:
        window = window.mean(dim=0, keepdim=True)   # mono-ise defensively
    return fix_length(window)


def spectrogram_from_window(
    window,
    extractor: "LogLinearSpectrogram | None" = None,
    band_crop: bool | None = None,
) -> torch.Tensor:
    """One live capture window -> model-ready tensor [1, F, 63]."""
    extractor = extractor or LogLinearSpectrogram(band_crop=band_crop)
    return extractor(as_mono_window(window))


def _reference_cases() -> dict[str, torch.Tensor]:
    """
    Signals that between them catch every porting mistake worth catching.

    Each one is chosen to fail loudly on a specific class of bug rather than to
    sound like anything:

      sweep     broadband and non-stationary - catches window function,
                centering and pad-mode errors, which show up as whole frames
                being wrong
      tone      a single stationary partial - a wrong window function smears it
                across neighbouring bins in a very visible way
      noise     flat spectrum - catches magnitude-vs-power and log errors
      silence   all zeros. Every magnitude is 0, so the whole array is
                log(LOG_EPS), the standard deviation is 0, and the output is
                only finite because of NORM_EPS. If an implementation divides
                by a bare std it produces NaN here and nowhere else.
      quiet     40 dB down - exercises the dynamic-range floor, which is
                otherwise never the binding constraint
      clipped   hard-clipped square - dense harmonics right up to Nyquist,
                so an off-by-one in the band crop is obvious
    """
    t = torch.arange(WINDOW_SAMPLES, dtype=torch.float32) / SAMPLE_RATE
    torch.manual_seed(0)
    noise = torch.randn(WINDOW_SAMPLES)
    sweep = torch.sin(2 * torch.pi * (200 + 3000 * t) * t)
    return {
        "reference": (0.7 * sweep + 0.1 * noise).clamp(-1.0, 1.0),
        "tone": 0.6 * torch.sin(2 * torch.pi * 1000 * t),
        "noise": (0.3 * noise).clamp(-1.0, 1.0),
        "silence": torch.zeros(WINDOW_SAMPLES),
        "quiet": (0.007 * sweep + 0.001 * noise).clamp(-1.0, 1.0),
        "clipped": torch.sign(torch.sin(2 * torch.pi * 220 * t)) * 0.9,
    }


def _emit_reference(out_dir: Path) -> None:
    """Write reference WAVs + expected tensors, for Python<->Kotlin diffing."""
    import csv

    import soundfile as sf

    out_dir.mkdir(parents=True, exist_ok=True)

    for name, sig in _reference_cases().items():
        wav_path = out_dir / f"parity_{name}.wav"
        sf.write(str(wav_path), sig.numpy(), SAMPLE_RATE, subtype="PCM_16")

        # Read the file BACK before computing the expected output. The WAV is
        # 16-bit, so writing it quantises every sample to a multiple of 1/32768.
        # Computing the reference from the pre-quantisation float would hand the
        # Kotlin port a target it cannot hit - it only ever sees the quantised
        # samples - and the resulting ~1e-2 disagreement looks exactly like a
        # real porting bug. Both sides must start from identical numbers.
        read_back, sr = sf.read(str(wav_path), dtype="float32", always_2d=True)
        assert sr == SAMPLE_RATE
        got = torch.from_numpy(read_back).mean(dim=1)

        shapes = []
        for crop in (True, False):
            spec = spectrogram_from_window(got, band_crop=crop).squeeze(0)
            suffix = "cropped" if crop else "full"
            csv_path = out_dir / f"parity_{name}_{suffix}.csv"
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                for row in spec.tolist():
                    w.writerow([f"{v:.6f}" for v in row])
            shapes.append(f"{suffix} {tuple(spec.shape)}")
        # Report BOTH shapes. This used to print `spec.shape` after the loop,
        # which is always the full-band one - so the line claimed the cropped
        # CSV was 257 rows when it is 104, and a reader checking the parity
        # output against it would go looking for a bug that is not there.
        print(f"  {name:10s} -> {wav_path.name}  + {', '.join(shapes)}")

    print(f"\nwrote {len(_reference_cases())} case(s) to {out_dir}/")
    print("Run the same WAVs through the Kotlin spectrogram and diff to ~3dp:")
    print(f"  java -cp parity.jar com.yourapp.detect.SpectrogramParityTestKt {out_dir}")
    print(f"The pipeline default is band_crop={DEFAULT_BAND_CROP} "
          f"-> the *_{'cropped' if DEFAULT_BAND_CROP else 'full'}.csv files.")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Canonical feature extractor")
    p.add_argument("--emit-reference", action="store_true",
                   help="Write parity_reference.wav + .csv for Kotlin cross-checking")
    p.add_argument("--out-dir", default="parity", help="Where to write reference files")
    a = p.parse_args()

    print(f"sample_rate={SAMPLE_RATE} window={WINDOW_SAMPLES} n_fft={N_FFT} "
          f"hop={HOP_LENGTH} win={WIN_LENGTH}")
    print(f"bin = {BIN_HZ:.2f} Hz   frame = {FRAME_MS:.0f} ms")
    print(f"full band     {FEATURE_SHAPE}")
    print(f"band cropped  {CROPPED_SHAPE}   bins {BAND_CROP_LO}..{BAND_CROP_HI - 1} "
          f"= {BAND_CROP_LO * BIN_HZ:.0f}-{(BAND_CROP_HI - 1) * BIN_HZ:.0f} Hz")
    print(f"default       {feature_shape()}   (DEFAULT_BAND_CROP={DEFAULT_BAND_CROP})")
    if a.emit_reference:
        _emit_reference(Path(a.out_dir))

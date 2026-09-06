#!/usr/bin/env python3
"""
ml/phone_call_effect.py

Converts a clean audio file into "phone call quality" audio.

Two outputs are generated:
  1. <name>_bandlimited.wav  - bandpass filtered (300-3400Hz) + compressed,
                               resampled to 8kHz. Clean telephone bandwidth.
  2. <name>_mulaw.wav        - same as above, plus encoded through the real
                               G.711 mu-law codec (used by actual landline
                               and VoIP calls) for authentic codec character.

This is the STANDALONE tool, for making listenable examples and for sanity
checking by ear. The training pipeline does NOT use it: ml/training/augment.py
reimplements the same chain in pure torch (phone_channel) so it can be
re-randomised on every epoch without shelling out to ffmpeg per clip. If you
change FILTER_CHAIN here, change augment.phone_channel to match.

ml/preprocess.py imports find_ffmpeg() from this module, and only needs ffmpeg to
decode formats libsndfile cannot read (mp3, m4a).

Requires ffmpeg to be installed and available on PATH.
  macOS:   brew install ffmpeg
  Windows: https://ffmpeg.org/download.html (add to PATH)
  Linux:   sudo apt install ffmpeg

Usage:
    python ml/phone_call_effect.py input_audio.wav
    python ml/phone_call_effect.py input_audio.wav -o output_folder
"""

import argparse
import os
import shutil
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

# Filter chain applied before resampling, so there is nothing left above
# the new Nyquist frequency to alias/distort when we drop to 8kHz.
# ml/training/augment.py phone_channel() mirrors this exactly.
FILTER_CHAIN = (
    "highpass=f=300,"
    "lowpass=f=3400,"
    "acompressor=threshold=-18dB:ratio=3:attack=5:release=50,"
    "aresample=8000"
)


FFMPEG_MISSING_MSG = """ffmpeg was not found.

Install it:
  Windows: winget install --id Gyan.FFmpeg -e   (or: choco install ffmpeg)
  macOS:   brew install ffmpeg
  Linux:   sudo apt install ffmpeg

Already installed but not found? Your terminal is carrying a stale PATH from
before the install. Either open a NEW terminal, or in PowerShell run:
  $env:Path = [Environment]::GetEnvironmentVariable("Path","Machine") + ';' + `
              [Environment]::GetEnvironmentVariable("Path","User")"""


def _search_dirs() -> list[Path]:
    """Standard install locations, for when PATH is stale or was never updated."""
    home = Path.home()
    local = Path(os.environ.get("LOCALAPPDATA", home / "AppData/Local"))
    files = Path(os.environ.get("ProgramFiles", "C:/Program Files"))
    return [
        local / "Microsoft/WinGet/Links",
        *(local / "Microsoft/WinGet/Packages").glob("Gyan.FFmpeg*/*/bin"),
        files / "ffmpeg/bin",
        Path("C:/ffmpeg/bin"),
        Path("/opt/homebrew/bin"), Path("/usr/local/bin"), Path("/usr/bin"),
    ]


@lru_cache(maxsize=1)
def find_ffmpeg() -> str | None:
    """
    Absolute path to ffmpeg, or None.

    Checks PATH first, then known install locations - a freshly installed ffmpeg
    is invisible to any shell opened before the install, and that shell is
    usually the one you are standing in.
    """
    found = shutil.which("ffmpeg")
    if found:
        return found
    exe = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
    for d in _search_dirs():
        try:
            candidate = d / exe
            if candidate.is_file():
                return str(candidate)
        except OSError:
            continue
    return None


def require_ffmpeg() -> str:
    """Return the ffmpeg path, or exit with an actionable message."""
    ffmpeg = find_ffmpeg()
    if ffmpeg is None:
        sys.exit(FFMPEG_MISSING_MSG)
    return ffmpeg


def run_ffmpeg(args):
    ffmpeg = require_ffmpeg()
    result = subprocess.run([ffmpeg, "-y", *args], capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        raise RuntimeError("ffmpeg failed - see error output above")


def make_bandlimited(input_path: Path, output_path: Path):
    """Clean phone-bandwidth version: filtered + compressed + resampled."""
    run_ffmpeg([
        "-i", str(input_path),
        "-af", FILTER_CHAIN,
        "-c:a", "pcm_s16le",
        str(output_path),
    ])


def make_mulaw(input_path: Path, encoded_path: Path, final_path: Path):
    """Authentic version: same processing, passed through real G.711 mu-law."""
    run_ffmpeg([
        "-i", str(input_path),
        "-af", FILTER_CHAIN,
        "-ar", "8000",
        "-c:a", "pcm_mulaw",
        str(encoded_path),
    ])
    run_ffmpeg([
        "-i", str(encoded_path),
        "-c:a", "pcm_s16le",
        str(final_path),
    ])


def process_file(input_path: Path, outdir: Path):
    stem = input_path.stem
    bandlimited_path = outdir / f"{stem}_bandlimited.wav"
    mulaw_encoded_path = outdir / f"{stem}_mulaw_encoded.wav"
    mulaw_final_path = outdir / f"{stem}_mulaw.wav"

    print(f"\nProcessing: {input_path.name}")
    print("  -> Creating bandlimited version...")
    make_bandlimited(input_path, bandlimited_path)
    print(f"     Saved: {bandlimited_path}")

    print("  -> Creating mu-law (authentic codec) version...")
    make_mulaw(input_path, mulaw_encoded_path, mulaw_final_path)
    if mulaw_encoded_path.exists():
        mulaw_encoded_path.unlink()
    print(f"     Saved: {mulaw_final_path}")


def main():
    parser = argparse.ArgumentParser(description="Make audio sound like a phone call.")
    parser.add_argument("input", help="Path to input audio file or folder containing audio files")
    parser.add_argument("-o", "--outdir", default="phone_audio_out", help="Output directory (default: phone_audio_out)")
    parser.add_argument("--recursive", action="store_true", help="Recursively process subdirectories")
    args = parser.parse_args()

    require_ffmpeg()

    input_path = Path(args.input)
    if not input_path.exists():
        sys.exit(f"Input path not found: {input_path}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    supported_exts = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}

    if input_path.is_dir():
        pattern = "**/*" if args.recursive else "*"
        audio_files = [
            f for f in input_path.glob(pattern)
            if f.is_file() and f.suffix.lower() in supported_exts and not f.name.endswith(("_bandlimited.wav", "_mulaw.wav", "_mulaw_encoded.wav"))
        ]
        if not audio_files:
            sys.exit(f"No audio files found in directory: {input_path}")
        print(f"Found {len(audio_files)} audio file(s) in {input_path}. Output folder: {outdir}")
        for audio_file in sorted(audio_files):
            process_file(audio_file, outdir)
    else:
        process_file(input_path, outdir)

    print("\nDone! The mu-law version is the closest to a real phone call.")


if __name__ == "__main__":
    main()

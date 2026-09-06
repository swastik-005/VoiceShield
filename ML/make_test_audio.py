"""
Generate synthetic human/ai audio so the pipeline can be exercised end to end
without any real dataset. NOT training data - a smoke test.

    python ml/make_test_audio.py                     # -> datasets/data/raw_smoketest/
    python ml/preprocess.py --raw-dir datasets/data/raw_smoketest \
                         --out-dir datasets/data/smoketest
    python ml/run_training.py --manifest datasets/data/smoketest/manifest.csv \
                           --out ml/training/checkpoints/smoketest.pt \
                           --results-dir datasets/data/smoketest/results --epochs 8

Human: jittered F0, shimmering harmonic amplitudes, aspiration noise.
AI:    rigidly periodic F0, flat harmonic envelope, a faint block-boundary
       artifact - a caricature of what a vocoder gets wrong.

THIS WRITES TO ITS OWN DIRECTORY, NOT datasets/data/raw/
-----------------------------------------------
The default used to be datasets/data/raw/, which is where the real corpus lives. Running
the documented smoke test therefore scattered six synthetic 'speakers' and three
fake 'engines' through the actual dataset - and because the label comes from the
folder name and the group from the subfolder, nothing downstream could tell them
apart from real recordings. They would just quietly train as human and ai.

So the default is datasets/data/raw_smoketest/ and the target must be an empty or
previously-synthetic directory: writing into a tree that already holds audio
this script did not create is refused. Pass an explicit path to override the
location; pass --force only if you genuinely mean to mix.
"""
import argparse
import math
from pathlib import Path

import numpy as np
import soundfile as sf

DEFAULT_ROOT = Path("datasets/data/raw_smoketest")

# Dropped beside the audio so a later run can tell "this tree is mine" from
# "this tree is somebody's dataset". Cheap, and the alternative is guessing.
MARKER = ".synthetic-smoke-test"

SR = 16000
DUR = 6.0


def voice(f0, seconds, jitter, shimmer, breath, seed, seam_hz=0.0):
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    t = np.arange(n) / SR

    # F0 contour: slow drift plus per-period jitter
    drift = 1.0 + 0.06 * np.sin(2 * math.pi * 0.7 * t + rng.uniform(0, 6))
    wobble = 1.0 + jitter * rng.standard_normal(n).cumsum() / max(1, np.sqrt(n))
    f = f0 * drift * np.clip(wobble, 0.8, 1.25)
    phase = 2 * math.pi * np.cumsum(f) / SR

    sig = np.zeros(n)
    for k in range(1, 26):
        if f0 * k > 3400:
            break
        amp = 1.0 / k
        if shimmer:
            amp *= 1.0 + shimmer * np.sin(2 * math.pi * rng.uniform(2, 9) * t)
        sig += amp * np.sin(k * phase + rng.uniform(0, 6))

    sig /= np.abs(sig).max() + 1e-9
    sig += breath * rng.standard_normal(n)          # aspiration
    # speech envelope: syllable-rate amplitude modulation with real pauses
    env = 0.5 + 0.5 * np.sin(2 * math.pi * 3.2 * t + rng.uniform(0, 6))
    env = np.clip(env, 0.05, None) ** 1.5
    sig *= env

    if seam_hz:
        # faint periodic discontinuity, like a realtime converter's block edges
        seam = np.zeros(n)
        step = int(SR / seam_hz)
        seam[::step] = 1.0
        sig += 0.02 * np.convolve(seam, rng.standard_normal(48), mode="same")

    return (0.5 * sig / (np.abs(sig).max() + 1e-9)).astype(np.float32)


def guard(root: Path, force: bool) -> None:
    """
    Refuse to scatter synthetic audio through a real dataset.

    Safe to write if the tree does not exist, is empty, or carries this
    script's own marker file. Anything else is somebody's corpus.
    """
    if force or not root.exists():
        return
    if (root / MARKER).is_file():
        return
    existing = [p for p in root.rglob("*")
                if p.is_file() and p.suffix.lower() in
                (".wav", ".mp3", ".flac", ".m4a", ".ogg", ".opus")]
    if not existing:
        return
    raise SystemExit(
        f"refusing to write into {root}/ - it already holds {len(existing)} "
        f"audio file(s)\nthat this script did not create, e.g. "
        f"{existing[0].relative_to(root).as_posix()}\n\n"
        f"Synthetic audio mixed into a real corpus is indistinguishable from it "
        f"afterwards:\nthe label comes from the folder name and the group from "
        f"the subfolder, so it\nwould simply train as though it were real.\n\n"
        f"Write somewhere else:   python ml/make_test_audio.py {DEFAULT_ROOT}\n"
        f"Or override on purpose: python ml/make_test_audio.py {root} --force")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", nargs="?", default=str(DEFAULT_ROOT),
                    help=f"Where to write. Default: {DEFAULT_ROOT}")
    ap.add_argument("--force", action="store_true",
                    help="Write even if the target already holds real audio")
    a = ap.parse_args()

    root = Path(a.root)
    guard(root, a.force)
    root.mkdir(parents=True, exist_ok=True)
    (root / MARKER).write_text(
        "Synthetic audio from ml/make_test_audio.py. Not a dataset.\n",
        encoding="utf-8")

    # human: several speakers, each its own group
    for i, f0 in enumerate([98, 118, 145, 172, 205, 238]):
        d = root / "human" / f"spk{i:02d}"
        d.mkdir(parents=True, exist_ok=True)
        for take in range(2):
            sf.write(d / f"take{take}.wav",
                     voice(f0, DUR, jitter=0.35, shimmer=0.25, breath=0.05,
                           seed=1000 + i * 10 + take), SR)

    # ai: three "engines", each its own group, each with its own artifact
    engines = {"engineA": dict(seam_hz=0.0, breath=0.002),
               "engineB": dict(seam_hz=12.0, breath=0.003),
               "engineC": dict(seam_hz=25.0, breath=0.001)}
    for i, (name, kw) in enumerate(engines.items()):
        d = root / "ai" / name
        d.mkdir(parents=True, exist_ok=True)
        for j, f0 in enumerate([110, 150, 195, 225]):
            sf.write(d / f"utt{j}.wav",
                     voice(f0, DUR, jitter=0.0, shimmer=0.0,
                           seed=2000 + i * 10 + j, **kw), SR)

    out = root.as_posix()
    # Strip the "raw_" prefix so the generated cache lands in a SIBLING
    # directory (datasets/data/smoketest/) rather than inside the raw tree, where
    # waveforms/ and manifest.csv would sit among the audio they index.
    stem = out.rsplit("/", 1)[-1]
    stem = stem[4:] if stem.startswith("raw_") else f"{stem}_out"
    print(f"wrote synthetic audio under {out}/")
    print(f"\nSmoke-test the pipeline WITHOUT touching datasets/data/raw:")
    print(f"  python ml/preprocess.py --raw-dir {out} --out-dir datasets/data/{stem}")
    print(f"  python ml/run_training.py --manifest datasets/data/{stem}/manifest.csv \\")
    print(f"      --out ml/training/checkpoints/{stem}.pt \\")
    print(f"      --results-dir datasets/data/{stem}/results --epochs 8 --workers 0")


if __name__ == "__main__":
    main()

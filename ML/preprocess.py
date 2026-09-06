"""
Turn raw audio into CNN-ready training windows.

    python ml/preprocess.py

Reads every audio file under datasets/data/raw/<label>/ and indexes one 1-second window
per 0.5 s of audio. The LABEL COMES FROM THE FOLDER NAME, so adding a dataset is
just dropping files in:

    datasets/data/raw/human/    real human speech      -> label 0
    datasets/data/raw/ai/       AI / TTS / cloned      -> label 1

Subfolders are allowed and become the "group" used for splitting, e.g.
datasets/data/raw/ai/rvc/ groups as "ai/rvc". Keep one speaker or one engine per
subfolder - ml/run_training.py splits by group so no recording appears in both
train and test.

WHAT THIS WRITES, AND WHY IT IS WAVEFORMS
-----------------------------------------
This step caches AUDIO, not spectrograms. An earlier version baked one fixed
phone degradation into a spectrogram per window and stored that. It was faster
to train from and it was wrong: every epoch then saw the same single channel,
so the model could not learn to be invariant to the channel - which is the
whole problem, because the channel at inference is a speakerphone in a room and
never exactly the one that was baked in.

So the waveform is cached once per source file, and training applies
    WaveformAugment -> features.LogLinearSpectrogram -> SpecAugment
fresh on every epoch. Val and test use the deterministic EvalChannel instead,
so their numbers are stable and still represent the deployed condition.

BOTH CLASSES GO THROUGH ONE IDENTICAL PATH
------------------------------------------
    resample -> trim silence -> loudness-normalise to -23 LUFS -> jittered
    window grid

Cross-corpus data leaks the corpus, not the voice: leading silence, trim style,
mastering level and where the speech sits inside a window are all properties of
the dataset an example came from, and every one of them separates human from AI
perfectly when human is corpus X and AI is corpus Y. A model that reaches ~1.0
on a random split has usually learned that and nothing else. See
ml/training/loudness.py. Each stage can be switched off individually to measure
how much it was carrying.

Writes:
    datasets/data/waveforms/<label>/<stem>.pt         int16 mono 16 kHz, one per source
    datasets/data/processed/<label>/<stem>.wav        EvalChannel preview, listenable
    datasets/data/spectrograms/<label>/<stem>_####.png  preview images  <- for your eyes
    datasets/data/manifest.csv    path, label, group, source_file, window_idx, start_sample

The manifest is the training index; each row points at a waveform file plus an
offset, so the 50% window overlap costs no extra disk.

ffmpeg is optional here - it is only used to decode formats libsndfile cannot
read (mp3, m4a on some systems). WAV/FLAC/OGG need nothing.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

import torch

from features import (
    HOP_WINDOW_SAMPLES,
    SAMPLE_RATE,
    WINDOW_SAMPLES,
    DEFAULT_BAND_CROP,
    LogLinearSpectrogram,
    feature_shape,
)
from phone_call_effect import find_ffmpeg
from training.augment import EvalChannel
from training.loudness import TARGET_LUFS, normalise_loudness, trim_silence

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".opus"}
VALID_LABELS = {"human", "ai"}

# Windows quieter than this (RMS in [-1, 1]) are dropped as silence.
#
# Tuning this is a real trade-off, in both directions:
#
#   Keep quiet windows  - breaths, pauses and fricatives (/s/, /f/) are exactly
#     where a voice changer fails hardest, because RVC-family models are trained
#     on voiced speech and mangle everything else. Throwing them away discards
#     the best evidence.
#
#   Drop quiet windows  - but ONLY with paired data. When human and AI audio come
#     from different corpora, near-silence differs by recording noise floor, and
#     the model happily learns THAT instead of anything about voices. Then the
#     accuracy is a confound, not a detector.
#
# So: conservative default while the data is cross-corpus, exposed as
# --silence-rms so it can be dropped to ~1e-4 once paired data exists.
SILENCE_RMS = 0.005


def load_mono(path: Path, ffmpeg: str | None) -> torch.Tensor:
    """Any supported audio file -> [T] float32 mono at SAMPLE_RATE."""
    import soundfile as sf

    try:
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    except Exception:
        if ffmpeg is None:
            raise
        # libsndfile could not decode it (usually mp3/m4a). Transcode and retry.
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td) / "decoded.wav"
            subprocess.run(
                [ffmpeg, "-y", "-i", str(path), "-ar", str(SAMPLE_RATE),
                 "-ac", "1", "-c:a", "pcm_s16le", str(tmp)],
                capture_output=True, text=True, check=True)
            data, sr = sf.read(str(tmp), dtype="float32", always_2d=True)

    w = torch.from_numpy(data).mean(dim=1)
    if sr != SAMPLE_RATE:
        import torchaudio
        w = torchaudio.functional.resample(w.unsqueeze(0), sr, SAMPLE_RATE).squeeze(0)
    return w


def to_int16(waveform: torch.Tensor) -> torch.Tensor:
    """float32 [-1, 1] -> int16, halving the cache size with no audible loss."""
    return (waveform.clamp(-1.0, 1.0) * 32767.0).round().to(torch.int16)


def save_png(spec: torch.Tensor, path: Path, band_crop: bool) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from features import BAND_CROP_HI, BAND_CROP_LO, BIN_HZ, SAMPLE_RATE as SR

    lo_khz = (BAND_CROP_LO * BIN_HZ) / 1000 if band_crop else 0.0
    hi_khz = (BAND_CROP_HI * BIN_HZ) / 1000 if band_crop else SR / 2000

    fig, ax = plt.subplots(figsize=(4.2, 3.2), dpi=110)
    ax.imshow(spec.squeeze(0).numpy(), origin="lower", aspect="auto", cmap="magma",
              extent=[0, 1.0, lo_khz, hi_khz])
    if not band_crop:
        ax.axhline(3.4, color="cyan", lw=0.7, ls="--", alpha=0.8)
    ax.set_xlabel("time (s)", fontsize=8)
    ax.set_ylabel("kHz", fontsize=8)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def discover(raw_dir: Path) -> list[tuple[Path, str, str]]:
    """Find (file, label, group) for every audio file under datasets/data/raw/<label>/."""
    found = []
    for label_dir in sorted(p for p in raw_dir.iterdir() if p.is_dir()):
        label = label_dir.name.lower()
        if label not in VALID_LABELS:
            print(f"  skipping datasets/data/raw/{label_dir.name}/ "
                  f"(folder name must be one of {sorted(VALID_LABELS)})")
            continue
        for f in sorted(label_dir.rglob("*")):
            if f.is_file() and f.suffix.lower() in AUDIO_EXTS:
                # AppleDouble sidecars. Copying a folder off a Mac leaves a
                # 4 KB "._name.flac" resource fork beside every real file, with
                # the SAME extension - so they look like audio here. Decoding
                # one fails in libsndfile, falls through to the ffmpeg retry,
                # spawns a subprocess and fails again. On an ASVspoof-sized
                # corpus that is ~94,000 wasted process spawns and 94,000 lines
                # of error spam before a single real window is written.
                if f.name.startswith("._"):
                    continue
                # Group = subfolder path if nested, else the filename stem. One
                # group never spans a train/test split.
                rel = f.relative_to(label_dir)
                group = f"{label}/{rel.parent.as_posix()}" if rel.parent.as_posix() != "." \
                    else f"{label}/{f.stem}"
                found.append((f, label, group))
    return found


def subsample_per_label(sources: list[tuple[Path, str, str]], cap: int,
                        seed: int = 0) -> list[tuple[Path, str, str]]:
    """
    Keep at most `cap` files per LABEL, drawn round-robin across that label's
    groups. 0 or less means no cap.

    Round-robin, not a flat random sample, and that distinction is the whole
    point. ASVspoof's spoof side is eight generators of ~9,200 files each while
    its bonafide side is 776 speakers of ~27 - a flat sample of the human class
    would hand back a few hundred speakers with one file apiece, and a flat
    sample of the ai class would silently over-weight whichever generator won
    the coin flips. Taking one file from each group in turn keeps EVERY
    generator and as many speakers as the cap allows, which is what the
    group-wise split in ml/run_training.py needs to produce a meaningful test set.
    """
    if cap <= 0:
        return sources

    by_label: dict[str, dict[str, list]] = {}
    for item in sources:
        by_label.setdefault(item[1], {}).setdefault(item[2], []).append(item)

    rng = random.Random(seed)
    kept: list[tuple[Path, str, str]] = []
    for label in sorted(by_label):
        groups = by_label[label]
        for files in groups.values():
            rng.shuffle(files)
        order = sorted(groups)
        rng.shuffle(order)

        picked, i = [], 0
        # Sweep the groups in turn, taking one file per group per pass, until
        # the cap is met or every group is exhausted.
        while len(picked) < cap:
            progressed = False
            for g in order:
                if i < len(groups[g]):
                    picked.append(groups[g][i])
                    progressed = True
                    if len(picked) >= cap:
                        break
            if not progressed:
                break
            i += 1
        kept.extend(picked)

        total = sum(len(v) for v in groups.values())
        if len(picked) < total:
            n_groups = len({p[2] for p in picked})
            print(f"  {label:5s} capped {total} -> {len(picked)} file(s) "
                  f"across {n_groups}/{len(groups)} group(s)")
    return kept


def warn_on_coarse_groups(rows: list[dict]) -> None:
    """
    A group that holds hundreds of source files is almost certainly a mistake.

    Groups are the unit of splitting, so one enormous group means all of that
    audio lands entirely in train or entirely in test. With only one group per
    class you get a test set containing a single class and metrics that mean
    nothing. This is the failure mode when a corpus is dumped into one folder.
    """
    files_per_group: dict[str, set] = {}
    for r in rows:
        files_per_group.setdefault(r["group"], set()).add(r["source_file"])

    coarse = {g: len(f) for g, f in files_per_group.items() if len(f) > 50}
    if not coarse:
        return
    print("\nWARNING: these groups contain many source files:")
    for g, n in sorted(coarse.items(), key=lambda kv: -kv[1]):
        print(f"  {g:40s} {n} files")
    print("  Groups are split whole, so everything above lands entirely in train\n"
          "  OR entirely in test. Split these into per-speaker / per-engine\n"
          "  subfolders (ml/sort_asvspoof.py --by-speaker does this for ASVspoof).")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", default="datasets/data/raw")
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--png-per-file", type=int, default=6,
                    help="Preview images per source file (0 = none). These are "
                         "for your eyes; training re-derives its own features.")
    ap.add_argument("--no-preview-audio", action="store_true",
                    help="Skip writing the listenable EvalChannel .wav previews")
    ap.add_argument("--no-band-crop", action="store_true",
                    help="Preview images at the full 0-8 kHz band. Does not "
                         "affect training, which reads band_crop from ml/features.py.")
    ap.add_argument("--silence-rms", type=float, default=SILENCE_RMS,
                    help=f"Drop windows quieter than this RMS (default "
                         f"{SILENCE_RMS}). Lower to ~1e-4 to keep breaths and "
                         f"fricatives once you have PAIRED data.")
    ap.add_argument("--keep-silent", action="store_true",
                    help="Keep every window regardless of level")
    ap.add_argument("--target-lufs", type=float, default=TARGET_LUFS,
                    metavar="LUFS",
                    help=f"Normalise every file to this integrated loudness "
                         f"(default {TARGET_LUFS}). Both classes, one number - "
                         f"otherwise mastering level alone separates the "
                         f"corpora.")
    ap.add_argument("--no-loudness-norm", action="store_true",
                    help="Skip loudness normalisation. Diagnostic only: this "
                         "leaves per-corpus level as a free shortcut AND lets "
                         "the compressor and SNR in training.augment hit the "
                         "two classes at different operating points.")
    ap.add_argument("--no-trim-silence", action="store_true",
                    help="Keep leading/trailing room tone. Diagnostic only - "
                         "how much silence a corpus leaves at the top of a file "
                         "is a per-corpus constant and a perfect giveaway.")
    ap.add_argument("--trim-top-db", type=float, default=40.0, metavar="DB",
                    help="Trim threshold, dB below the loudest frame "
                         "(default 40).")
    ap.add_argument("--hop-samples", type=int, default=HOP_WINDOW_SAMPLES,
                    metavar="N",
                    help=f"Stride between windows (default {HOP_WINDOW_SAMPLES} "
                         f"= 50%% overlap). Set to {WINDOW_SAMPLES} for no "
                         f"overlap: overlapping windows are near-duplicates, so "
                         f"a long file becomes thousands of copies of one voice "
                         f"and the model memorises it.")
    ap.add_argument("--no-window-jitter", action="store_true",
                    help="Start the window grid at sample 0 for every file. "
                         "The default offsets it randomly per file so that "
                         "phase-within-the-window stops being a corpus tell.")
    ap.add_argument("--no-clean", action="store_true",
                    help="Append to existing output instead of wiping it")
    ap.add_argument("--resume", action="store_true",
                    help="Reuse any datasets/data/waveforms/<...>.pt already on disk "
                         "instead of decoding its source again, and never wipe "
                         "the cache (implies --no-clean). The cached file holds "
                         "the audio AFTER trim and loudness normalisation, and "
                         "the window grid is seeded on the path, so the windows "
                         "rebuilt from it are identical to a fresh run - this "
                         "is a restart after an interrupted pass, not a "
                         "different dataset. Delete the cache to force a "
                         "re-decode after changing --target-lufs, "
                         "--trim-top-db or the trim/normalise flags, which are "
                         "baked into the cached audio and CANNOT be changed by "
                         "a resumed run.")
    ap.add_argument("--include-prefix", action="append", default=[], metavar="PREFIX",
                    help="Keep only groups starting with PREFIX (repeatable), "
                         "e.g. --include-prefix ai/asvspoof --include-prefix "
                         "human/asvspoof. Use this to build a SAME-CORPUS "
                         "dataset: mixing extra human-only sources in lets the "
                         "model score them right by recognising the corpus.")
    ap.add_argument("--exclude-prefix", action="append", default=[], metavar="PREFIX",
                    help="Drop groups starting with PREFIX (repeatable). "
                         "Applied after --include-prefix.")
    ap.add_argument("--max-files-per-label", type=int, default=0, metavar="N",
                    help="Keep at most N source files per label (0 = all), "
                         "drawn round-robin across that label's groups so every "
                         "generator and as many speakers as possible survive. "
                         "Use this to cut a 90k-file corpus down to something "
                         "that trains in an hour.")
    ap.add_argument("--subsample-seed", type=int, default=0,
                    help="Seed for --max-files-per-label, so the subset is "
                         "reproducible across runs")
    a = ap.parse_args(argv)

    ffmpeg = find_ffmpeg()
    print(f"ffmpeg: {ffmpeg or 'not found (only needed for mp3/m4a decoding)'}")

    raw_dir = Path(a.raw_dir)
    if not raw_dir.is_dir():
        print(f"not found: {raw_dir}\nExpected datasets/data/raw/human/ and datasets/data/raw/ai/")
        return 1

    sources = discover(raw_dir)
    if not sources:
        print(f"no audio found under {raw_dir}/<label>/")
        print(f"Put files in datasets/data/raw/human/ and datasets/data/raw/ai/")
        return 1

    if a.include_prefix or a.exclude_prefix:
        before = len(sources)
        if a.include_prefix:
            sources = [s for s in sources
                       if any(s[2].startswith(p) for p in a.include_prefix)]
        if a.exclude_prefix:
            sources = [s for s in sources
                       if not any(s[2].startswith(p) for p in a.exclude_prefix)]
        kept_srcs = Counter(s[2].split("/")[1] if "/" in s[2] else s[2]
                            for s in sources)
        print(f"source filter: {before} -> {len(sources)} file(s)  "
              f"{dict(kept_srcs.most_common())}")
        if not sources:
            print("  nothing left after filtering - check your prefixes")
            return 1

    if a.max_files_per_label > 0:
        print(f"subsampling to {a.max_files_per_label} file(s) per label "
              f"(seed {a.subsample_seed}):")
        sources = subsample_per_label(sources, a.max_files_per_label,
                                      a.subsample_seed)
        print()

    out = Path(a.out_dir)
    wav_dir = out / "waveforms"
    proc_dir, png_dir = out / "processed", out / "spectrograms"
    if not (a.no_clean or a.resume):
        for d in (wav_dir, proc_dir, png_dir):
            if d.exists():
                shutil.rmtree(d)
    for d in (wav_dir, proc_dir, png_dir):
        d.mkdir(parents=True, exist_ok=True)

    band_crop = not a.no_band_crop
    extractor = LogLinearSpectrogram(band_crop=band_crop)
    eval_channel = EvalChannel()
    print(f"training feature: {feature_shape()}  (band_crop={DEFAULT_BAND_CROP}, "
          f"set in ml/features.py)")
    print(f"silence gate: {'off' if a.keep_silent else a.silence_rms}")
    hop = max(1, a.hop_samples)
    overlap = ('no overlap' if hop >= WINDOW_SAMPLES
               else f'{100 * (1 - hop / WINDOW_SAMPLES):.0f}% overlap')
    grid = 'aligned to sample 0' if a.no_window_jitter else 'jittered'
    print(f"window grid : {WINDOW_SAMPLES} samples every {hop} "
          f"({overlap}, {grid})")
    level = ('NO LOUDNESS NORM' if a.no_loudness_norm
             else f'{a.target_lufs:g} LUFS')
    print(f"identical path for both classes: resample -> "
          f"{'NO TRIM' if a.no_trim_silence else 'trim'} -> {level}")
    print("caching WAVEFORMS - the channel is applied per-epoch during training\n")

    rows, total, dropped_total = [], 0, 0
    # Measured loudness BEFORE normalisation, per label. If these two means are
    # far apart, level alone was separating the classes - which is exactly the
    # shortcut this pass removes, and worth seeing rather than assuming.
    lufs_before: dict[str, list[float]] = {}
    n_limited = 0
    by_label = Counter(lbl for _, lbl, _ in sources)
    print(f"{len(sources)} source file(s): {dict(by_label)}\n")

    n_resumed = 0
    for src, label, group in sources:
        stem = src.stem
        # Mirror the raw/ tree rather than flattening to <label>/<stem>.pt.
        # Flattening silently collides whenever two speakers have a file with
        # the same name - human/spk00/take0.wav and human/spk01/take0.wav both
        # become take0.pt, and every manifest row then points at whichever was
        # written last. That is a data-corruption bug with no error message.
        rel = src.relative_to(raw_dir)
        wf_path = wav_dir / rel.with_suffix(".pt")

        # --resume: the cached .pt already holds this file's audio AFTER trim
        # and loudness normalisation, so re-windowing it reproduces the rows a
        # fresh decode would have produced (the grid jitter below is seeded on
        # the path, not on the run). Decoding is the whole cost of this script,
        # so this is what makes an interrupted pass restartable.
        cached = a.resume and wf_path.is_file()
        if cached:
            try:
                audio = torch.load(wf_path, weights_only=True)["waveform"]
                audio = audio.to(torch.float32) / 32767.0
                n_resumed += 1
            except Exception as e:
                print(f"  !! unreadable cache {wf_path.name}: "
                      f"{type(e).__name__}: {e} - re-decoding")
                cached = False
        if not cached:
            try:
                audio = load_mono(src, ffmpeg)
            except Exception as e:
                print(f"  !! could not decode {src.name}: {type(e).__name__}: {e}")
                continue

            # ONE PATH, BOTH CLASSES. Nothing below this line may branch on label.
            if not a.no_trim_silence:
                audio = trim_silence(audio, top_db=a.trim_top_db)
            if not a.no_loudness_norm:
                audio, measured, limited = normalise_loudness(audio, a.target_lufs)
                if math.isfinite(measured):
                    lufs_before.setdefault(label, []).append(measured)
                    n_limited += int(limited)

        if audio.shape[0] < WINDOW_SAMPLES:
            print(f"  .. {src.name}: shorter than one window after trimming, "
                  f"skipped")
            continue

        if not cached:
            wf_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"waveform": to_int16(audio), "sample_rate": SAMPLE_RATE,
                        "label": label, "group": group,
                        "source_file": src.as_posix()}, wf_path)

        # Random start offset for the window grid, one per file. Without it
        # every window in the corpus begins at a multiple of the hop counted
        # from the first sample, so "where the speech sits inside the window"
        # is decided by how that corpus trims its files. Seeded on the path, so
        # a re-run reproduces the same manifest.
        jitter = 0
        room = audio.shape[0] - WINDOW_SAMPLES
        if not a.no_window_jitter and room > 0:
            jitter = random.Random(
                f"{a.subsample_seed}:{rel.as_posix()}").randrange(
                    0, min(hop, room) + 1)

        n_slots = max(0, (audio.shape[0] - jitter - WINDOW_SAMPLES) // hop + 1)
        kept = dropped = 0
        pdir = png_dir / rel.parent
        pdir.mkdir(parents=True, exist_ok=True)

        for i in range(n_slots):
            start = jitter + i * hop
            chunk = audio[start : start + WINDOW_SAMPLES]
            if not a.keep_silent and chunk.pow(2).mean().sqrt().item() < a.silence_rms:
                dropped += 1
                continue
            rows.append({"path": wf_path.as_posix(), "label": label, "group": group,
                         "source_file": src.as_posix(), "window_idx": i,
                         "start_sample": start})
            if kept < a.png_per_file:
                # Preview through the eval channel, so the picture shows what the
                # model is actually asked to classify, not the clean studio take.
                degraded = eval_channel(chunk.unsqueeze(0))
                save_png(extractor(degraded), pdir / f"{stem}_{i:04d}.png", band_crop)
            kept += 1

        if not a.no_preview_audio and n_slots:
            import soundfile as sf
            wav_out = proc_dir / rel.with_suffix(".wav")
            wav_out.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(wav_out),
                     eval_channel(audio.unsqueeze(0)).squeeze(0).numpy(),
                     SAMPLE_RATE, subtype="PCM_16")

        total += kept
        dropped_total += dropped
        print(f"  [{label:5s}] {rel.as_posix()[:48]:48s} -> {kept:4d} windows"
              f"{f' ({dropped} silent dropped)' if dropped else ''}")

    manifest = out / "manifest.csv"
    with open(manifest, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["path", "label", "group", "source_file",
                                          "window_idx", "start_sample"])
        w.writeheader()
        w.writerows(rows)

    counts = Counter(r["label"] for r in rows)
    n_groups = len({r["group"] for r in rows})
    if len(lufs_before) > 1:
        means = {k: sum(v) / len(v) for k, v in sorted(lufs_before.items())}
        spread = max(means.values()) - min(means.values())
        print(f"\nloudness BEFORE normalisation (every file now at "
              f"{a.target_lufs:g} LUFS):")
        for k, m in means.items():
            print(f"  {k:5s} {m:7.2f} LUFS  ({len(lufs_before[k])} files)")
        note = ("  <- that alone was a usable shortcut; it is gone now"
                if spread >= 2.0 else "")
        print(f"  the two classes differed by {spread:.2f} LU{note}")
    if n_limited:
        print(f"  {n_limited} file(s) peak-limited at -1 dBFS instead of "
              f"reaching the target")
    if n_resumed:
        print(f"\nresumed: {n_resumed}/{len(sources)} file(s) re-windowed "
              f"from datasets/data/waveforms/ without decoding the source again")
    print(f"\n{total} windows ({dropped_total} silent dropped) in {n_groups} group(s)")
    print(f"  {dict(counts)}")
    print(f"  waveforms -> {wav_dir}/")
    print(f"  previews  -> {proc_dir}/ and {png_dir}/")
    print(f"  manifest  -> {manifest}")

    warn_on_coarse_groups(rows)

    if len(counts) < 2:
        only = next(iter(counts), "nothing")
        print(f"\nONLY ONE CLASS ('{only}'). Training needs both - a one-class "
              f"model predicts '{only}' every time and scores 100% having learned "
              f"nothing.\nAdd audio to datasets/data/raw/{'ai' if only == 'human' else 'human'}/ "
              f"and re-run.")
    elif n_groups < 4:
        print(f"\nOnly {n_groups} group(s). The split needs at least two per class "
              f"to put both classes in the test set.")
    else:
        print(f"\nReady: python ml/run_training.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())

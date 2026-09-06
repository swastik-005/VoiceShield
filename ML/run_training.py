"""
Train the AI-vs-human detector on the windows indexed by ml/preprocess.py.

    python ml/preprocess.py       # datasets/data/raw/<label>/ -> waveform cache + manifest
    python ml/run_training.py     # train

Labels come from the datasets/data/raw/<label>/ folder name, set by ml/preprocess.py.
Refuses to run on one class rather than producing a model that looks trained
and detects nothing.

THE CHANNEL IS APPLIED HERE, NOT IN PREPROCESSING
-------------------------------------------------
ml/preprocess.py caches waveforms. This file turns them into spectrograms, and it
does so through a channel simulation that is re-randomised every epoch:

    train      WaveformAugment -> LogLinearSpectrogram -> SpecAugment
    val/test   EvalChannel     -> LogLinearSpectrogram

WaveformAugment stacks the phone network (bandpass, compression, G.711 mu-law)
and the speakerphone re-capture (small-speaker response, room impulse response,
ambient noise), because that is the path the real signal takes. Randomising it
per epoch is what makes the model invariant to the channel instead of to one
particular baked-in version of it.

EvalChannel is the fixed average-case version of the same path, so val and test
measure the deployed condition without the metric moving between runs.

SPLITTING - the thing that makes the number honest
--------------------------------------------------
Windows overlap by 50%, so window N and window N+1 share half a second of audio.
Split those randomly and near-identical windows land in both train and test; the
model scores brilliantly by recognising audio it has already seen. Worse, all
windows from one recording share a speaker, a microphone and a room.

So the split is BY GROUP, never by window. A recording is entirely in train or
entirely in test. That is a lower number and a real one.

--holdout-group-prefix goes further: leave-one-converter-out. Hold out an entire
generator, train on the others, and report on the one never seen. On a live call
the attacker's tool will not be in your training set, so that is the only number
worth quoting.

AND THE VALIDATION SET MUST BE HELD OUT THE SAME WAY
----------------------------------------------------
Holding a generator out of TEST is only half the job. If val is still drawn
from the generators that train on, then "best val epoch" means "the epoch that
memorised the training generators hardest", and checkpoint selection actively
rewards the overfitting it is supposed to catch. The test number then measures
a checkpoint chosen for the wrong reason.

So --val-holdout-group-prefix holds out a DIFFERENT generator for validation:

    --holdout-group-prefix ai/asvspoof/A09      <- never trained on, reported
    --val-holdout-group-prefix ai/asvspoof/A12  <- never trained on, selects

Early stopping and best-checkpoint selection both run off that val set, so what
is being maximised is generalisation to an unseen generator rather than fit to
the seen ones.

READING THE THREE NUMBERS AT THE END
------------------------------------
train loss ~0.01 with held-out balanced accuracy ~0.5 is not a capacity problem
and no amount of dropout will touch it: the model has learned something that
separates the two corpora perfectly and does not exist in the held-out one -
leading silence, loudness, resample history. That is a DATA problem; fix it in
ml/preprocess.py (which now puts both classes through one identical path) and by
pairing the sources so only the vocoder differs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

try:
    from tqdm.auto import tqdm
except ImportError:                                  # progress bars are a
    def tqdm(it=None, **kw):                         # convenience, never a
        return it if it is not None else _NullBar()  # dependency of training
    class _NullBar:
        def update(self, *a): pass
        def set_postfix(self, **kw): pass
        def close(self): pass

from features import (
    DEFAULT_BAND_CROP,
    WINDOW_SAMPLES,
    LogLinearSpectrogram,
    feature_shape,
    n_freq_bins as n_freq_bins_for,
)
from training.augment import EvalChannel, SpecAugment, WaveformAugment
from training.model import SpoofCNN

LABELS = {"human": 0, "ai": 1}


# --------------------------------------------------------------------- data

class WaveformStore:
    """
    Every cached waveform packed into ONE memory-mapped int16 file.

    WHY THIS EXISTS - it was 44% of training time. The previous design was a
    64-file LRU in front of torch.load. With 154 recordings and a sampler that
    jumps randomly between them that missed roughly 60% of reads, and every miss
    deserialised a whole multi-megabyte recording in order to use one second of
    it: 3.0 ms of the 6.9 ms it took to produce a single window.

    A memmap turns that read into a slice. It also fixes the thing that made
    DataLoader workers expensive on Windows: spawn pickles the dataset into each
    worker, so an in-RAM cache costs N x 1 GB for N workers, whereas memmapped
    pages are shared by the OS and cost 1 GB once no matter how many workers
    read them.

    The blob is rebuilt whenever any .pt in the manifest changes size or mtime,
    so it cannot silently serve stale audio after a re-run of ml/preprocess.py.
    """

    def __init__(self, paths: list[str], root: Path):
        self.bin_path = root / "waveforms.i16"
        self.index_path = root / "waveforms.index.json"
        self.offsets = self._ensure(sorted(set(paths)))
        self._mm = None            # opened per process; memmaps do not pickle

    def __getstate__(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if k != "_mm"}

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._mm = None

    @staticmethod
    def _fingerprint(paths: list[str]) -> list:
        return [[p, os.path.getsize(p), int(os.path.getmtime(p))] for p in paths]

    def _ensure(self, paths: list[str]) -> dict[str, tuple[int, int]]:
        want = self._fingerprint(paths)
        if self.index_path.is_file() and self.bin_path.is_file():
            try:
                idx = json.loads(self.index_path.read_text(encoding="utf-8"))
                if idx["fingerprint"] == want:
                    return {p: tuple(v) for p, v in idx["offsets"].items()}
            except (ValueError, KeyError, OSError):
                pass                                  # corrupt or stale: rebuild

        self.bin_path.parent.mkdir(parents=True, exist_ok=True)
        offsets: dict[str, tuple[int, int]] = {}
        at = 0
        with open(self.bin_path, "wb") as f:
            for path in tqdm(paths, desc="packing waveforms", unit="file"):
                wave = torch.load(path, weights_only=False)["waveform"]
                if wave.dtype != torch.int16:
                    wave = (wave.to(torch.float32).clamp(-1.0, 1.0)
                            * 32767.0).to(torch.int16)
                arr = wave.numpy().astype("<i2", copy=False)
                f.write(arr.tobytes())
                offsets[path] = (at, int(arr.size))
                at += int(arr.size)
        self.index_path.write_text(
            json.dumps({"fingerprint": want,
                        "offsets": {k: list(v) for k, v in offsets.items()}}),
            encoding="utf-8")
        print(f"packed {len(paths)} recordings -> {self.bin_path} "
              f"({at * 2 / 1e9:.2f} GB)")
        return offsets

    def window(self, path: str, start: int, n: int) -> torch.Tensor:
        """One [n]-sample float window, zero-padded if the file ends first."""
        if self._mm is None:
            self._mm = np.memmap(self.bin_path, dtype="<i2", mode="r")
        off, length = self.offsets[path]
        lo = off + min(start, length)
        hi = off + min(start + n, length)
        x = torch.from_numpy(np.asarray(self._mm[lo:hi], dtype=np.float32)) / 32768.0
        if x.shape[0] < n:                            # ragged tail
            x = torch.nn.functional.pad(x, (0, n - x.shape[0]))
        return x


class WaveformCache:
    """
    Per-file LRU fallback, used when no WaveformStore is supplied (export_onnx
    reads a handful of windows and should not build a blob for them).
    """

    def __init__(self, max_files: int = 64):
        self.max_files = max_files
        self._cache: dict[str, torch.Tensor] = {}

    def get(self, path: str) -> torch.Tensor:
        hit = self._cache.get(path)
        if hit is not None:
            return hit
        blob = torch.load(path, weights_only=False)
        wave = blob["waveform"]
        if wave.dtype == torch.int16:
            wave = wave.to(torch.float32) / 32768.0
        if len(self._cache) >= self.max_files:
            self._cache.pop(next(iter(self._cache)))
        self._cache[path] = wave
        return wave


class WindowDataset(Dataset):
    """
    One manifest row -> one [1, F, 63] spectrogram, built on the fly.

    `channel` is the waveform-domain transform: WaveformAugment for train,
    EvalChannel for val/test. `spec_aug` is applied afterwards, train only.
    """

    def __init__(self, rows: list[dict], channel=None, spec_aug=None,
                 extractor: LogLinearSpectrogram | None = None,
                 store: "WaveformStore | None" = None):
        self.rows = rows
        self.channel = channel
        self.spec_aug = spec_aug
        self.extractor = extractor or LogLinearSpectrogram()
        self.store = store
        self.cache = None if store is not None else WaveformCache()

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        r = self.rows[i]
        start = int(r["start_sample"])
        if self.store is not None:
            chunk = self.store.window(r["path"], start, WINDOW_SAMPLES)
        else:
            wave = self.cache.get(r["path"])
            chunk = wave[start : start + WINDOW_SAMPLES]
            if chunk.shape[0] < WINDOW_SAMPLES:      # ragged tail
                chunk = torch.nn.functional.pad(
                    chunk, (0, WINDOW_SAMPLES - chunk.shape[0]))
        x = chunk.unsqueeze(0)

        if self.channel is not None:
            x = self.channel(x)
        spec = self.extractor(x)
        if self.spec_aug is not None:
            spec = self.spec_aug(spec)

        return {"spectrogram": spec,
                "label": torch.tensor(LABELS[r["label"]], dtype=torch.float32)}

    def labels(self) -> list[int]:
        return [LABELS[r["label"]] for r in self.rows]


class FeatureDataset(Dataset):
    """
    Spectrograms already computed, held in RAM. Same batch shape as
    WindowDataset so evaluate() cannot tell the difference.
    """

    def __init__(self, specs: torch.Tensor, labels: torch.Tensor):
        self.specs = specs
        self.label_t = labels

    def __len__(self) -> int:
        return int(self.specs.shape[0])

    def __getitem__(self, i: int):
        return {"spectrogram": self.specs[i], "label": self.label_t[i]}


def precompute_features(rows, store, channel, extractor, batch_size, workers,
                        desc) -> FeatureDataset:
    """
    Run the eval channel ONCE and keep the result.

    EvalChannel is deterministic by construction - fixed RIR seed, fixed noise -
    so a given val window produces a bit-identical spectrogram on every epoch.
    The old code re-ran it anyway: 9.6 ms of biquads, resampling, mu-law and an
    FFT convolution per window, per epoch, for all 20 epochs. Building it once
    turns validation into a bare GPU forward pass and changes no number.

    Costs RAM: ~4 bytes x 104 x 63 per window, about 26 kB, so ~250 MB per
    10k windows. --no-cache-eval falls back to recomputing if that is too much.
    """
    ds = WindowDataset(rows, channel=channel, extractor=extractor, store=store)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=workers, worker_init_fn=seed_worker)
    specs, labels = [], []
    for b in tqdm(loader, desc=desc, unit="batch", leave=False,
                  dynamic_ncols=True):
        specs.append(b["spectrogram"])
        labels.append(b["label"])
    if not specs:
        return FeatureDataset(torch.empty(0), torch.empty(0))
    return FeatureDataset(torch.cat(specs), torch.cat(labels))


def seed_worker(worker_id: int) -> None:
    """Give each DataLoader worker its own augmentation stream, and one thread."""
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    # ONE thread per worker. The channel simulation is a chain of tiny ops on a
    # one-second buffer; torch's intra-op pool wins nothing on buffers that
    # small. But left at its default, every one of N workers opens a pool sized
    # for the whole machine, so 8 workers x 16 threads fight over 16 cores and
    # most of the time goes into scheduling. The parallelism should come from
    # the workers, which is where the actual independent work is.
    torch.set_num_threads(1)
    # One thread per worker. The channel simulation is a chain of tiny ops on a
    # one-second buffer, so torch's intra-op pool wins nothing on them - but
    # left at its default every one of N workers opens a pool sized for the
    # whole machine, and N x 16 threads fight over 16 cores. Pinning to 1 makes
    # the parallelism come from the workers, where it actually helps.
    torch.set_num_threads(1)


def load_manifest(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if rows and "start_sample" not in rows[0]:
        raise SystemExit(
            "This manifest was written by an older ml/preprocess.py that cached\n"
            "spectrograms. Re-run: python ml/preprocess.py"
        )
    for r in rows:
        r["label"] = r["label"].strip().lower()
    return rows


def check_labels(rows: list[dict]) -> str | None:
    """Return an error message if this manifest cannot train a classifier."""
    counts = Counter(r["label"] for r in rows)
    bad = set(counts) - set(LABELS)
    if bad:
        return (f"unrecognised labels {sorted(bad)}; expected 'human' or 'ai'.\n"
                f"Labels come from the datasets/data/raw/<label>/ folder name - re-run "
                f"ml/preprocess.py rather than editing the manifest by hand.")
    if len(counts) < 2:
        only = next(iter(counts))
        need = "ai" if only == "human" else "human"
        return (f"every row is '{only}'. A classifier needs two classes.\n"
                f"With one class the model predicts '{only}' every time, scores "
                f"100%, and has learned nothing.\n"
                f"Put audio in datasets/data/raw/{need}/ and re-run ml/preprocess.py.")
    return None


# -------------------------------------------------------------------- splits

def split_by_prefix(rows: list[dict], prefixes: list[str],
                    val_prefixes: list[str], val_frac: float, seed: int):
    """
    Leave-one-converter-out: every group whose name starts with one of
    `prefixes` goes to test, nothing from it is ever trained on. Groups matching
    `val_prefixes` are held out the same way for validation, so the epoch that
    gets checkpointed is the one that generalises to a generator it has never
    seen rather than the one that memorised the generators it has.

    The two holdouts must name DIFFERENT generators. Sharing one would let the
    test set pick its own checkpoint, which is the same leak one level up.

    Held-out groups are usually one class only (the AI side), so a slice of
    recordings from the other class is moved across too - otherwise the split
    has nothing to get wrong in the human direction and accuracy is meaningless.

    The balance is by WINDOW COUNT, not by group count, and the difference is not
    cosmetic. Groups are wildly uneven: one ASVspoof attack is ~9,000 files in a
    single group, while one bonafide speaker is a handful. Matching group-for-
    group therefore put 3,928 spoof windows against 54 human windows from ONE
    speaker - and human recall is half of balanced accuracy, so the headline
    number swung on that speaker's idiosyncrasies. Accumulating other-class
    groups until the window totals match costs nothing and makes both recalls
    rest on comparable evidence. Test is balanced first and val out of what is
    left, so no recording backs both numbers.
    """
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[r["group"]].append(r)

    held = [g for g in groups if any(g.startswith(p) for p in prefixes)]
    if not held:
        return None
    val_held = [g for g in groups
                if any(g.startswith(p) for p in val_prefixes)
                and g not in set(held)]

    rng = random.Random(seed)
    pool = [g for g in groups if g not in set(held) | set(val_held)]
    rng.shuffle(pool)

    def balance(names: list[str]) -> list[str]:
        """Other-class groups drawn from `pool` until the window totals match."""
        labels = {groups[g][0]["label"] for g in names}
        target = sum(len(groups[g]) for g in names)
        other = [g for g in pool if groups[g][0]["label"] not in labels]
        chosen, taken = [], 0
        for g in other:
            if taken >= target:
                break
            chosen.append(g)
            taken += len(groups[g])
        if not chosen and other:         # never hand back an empty other class
            chosen = other[:1]
        for g in chosen:
            pool.remove(g)
        return chosen

    test_groups = held + balance(held)
    if val_held:
        val_groups = val_held + balance(val_held)
    else:
        # No val holdout named: fall back to a slice of the training generators
        # and warn at the call site, because selecting on this is the leak the
        # docstring at the top of this file is about.
        n_val = int(len(pool) * val_frac)
        val_groups, pool = pool[:n_val], pool[n_val:]

    pick = lambda names: [r for g in names for r in groups[g]]
    return (pick(pool), pick(val_groups), pick(test_groups), held, val_held)


def split_by_recording(rows: list[dict], test_frac: float, val_frac: float, seed: int):
    """
    Group by source recording, then split whole groups.

    Both classes must appear in every split or the metrics are meaningless, so
    groups are drawn per label rather than from one shuffled pool.
    """
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[r["group"]].append(r)

    by_label: dict[str, list[str]] = defaultdict(list)
    for name, g in groups.items():
        by_label[g[0]["label"]].append(name)

    rng = random.Random(seed)
    train, val, test = [], [], []
    for label, names in by_label.items():
        rng.shuffle(names)
        n = len(names)
        n_test = max(1, round(n * test_frac)) if n > 1 else 0
        n_val = max(1, round(n * val_frac)) if n - n_test > 1 else 0
        for name in names[:n_test]:
            test += groups[name]
        for name in names[n_test:n_test + n_val]:
            val += groups[name]
        for name in names[n_test + n_val:]:
            train += groups[name]
    return train, val, test, groups


# ------------------------------------------------------------------ metrics

def eer_with_threshold(probs: torch.Tensor,
                       labels: torch.Tensor) -> tuple[float, float]:
    """
    EER and the probability threshold that achieves it.

    Balanced accuracy depends on where 0.5 happens to fall; EER is the operating
    point where false-accept and false-reject rates meet, so two models can be
    compared without either being tuned.

    The THRESHOLD is returned as well because something has to ship one. 0.5 is
    an arbitrary place to cut a sigmoid - it is where the training loss stopped
    caring, not where the two error types balance - and on an imbalanced or
    poorly-calibrated model the two can be far apart. live_capture.verdict() and
    the phone both take this number rather than assuming 0.5.
    """
    if labels.numel() == 0 or labels.min() == labels.max():
        return float("nan"), float("nan")
    order = torch.argsort(probs, descending=True)
    y = labels[order]
    p = probs[order]
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    tp = torch.cumsum((y == 1).float(), 0)
    fp = torch.cumsum((y == 0).float(), 0)
    far = fp / n_neg                  # human wrongly called AI
    frr = 1.0 - tp / n_pos            # AI wrongly called human
    idx = int(torch.argmin((far - frr).abs()))
    return float((far[idx] + frr[idx]) / 2), float(p[idx])


def equal_error_rate(probs: torch.Tensor, labels: torch.Tensor) -> float:
    """EER only. See eer_with_threshold() for the operating point too."""
    return eer_with_threshold(probs, labels)[0]


def evaluate(model, loader, device, desc: str = "eval") -> dict:
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for b in tqdm(loader, desc=desc, unit="batch", leave=False,
                      dynamic_ncols=True):
            x = b["spectrogram"].to(device, non_blocking=True)
            probs.append(torch.sigmoid(model(x)).float().cpu())
            labels.append(b["label"])
    if not probs:
        return {"n": 0}
    p, y = torch.cat(probs), torch.cat(labels)
    pred = (p > 0.5).float()
    eer, eer_thr = eer_with_threshold(p, y)
    ai, hu = y == 1, y == 0
    r_ai = (pred[ai] == 1).float().mean().item() if ai.any() else float("nan")
    r_hu = (pred[hu] == 0).float().mean().item() if hu.any() else float("nan")
    return {
        "n": int(y.numel()),
        "acc": (pred == y).float().mean().item(),
        "recall_ai": r_ai,
        "recall_human": r_hu,
        # Plain accuracy is dominated by whichever class has more windows, so a
        # model that simply leans toward the majority scores well while getting
        # worse at the minority. Balanced accuracy is the mean of the two
        # recalls and is immune to that - quote this one.
        "balanced_acc": (r_ai + r_hu) / 2 if (ai.any() and hu.any()) else float("nan"),
        "eer": eer,
        # The operating point to ship, not the 0.5 the accuracies above use.
        "eer_threshold": eer_thr,
        "probs": p,
    }


def smooth_targets(y: torch.Tensor, eps: float) -> torch.Tensor:
    """
    Pull the 0/1 targets `eps` of the way toward 0.5.

    BCE with hard targets is minimised by driving the logit to infinity, so a
    model that has found a shortcut is rewarded for becoming maximally confident
    about it. Smoothing caps the return on confidence, which both calibrates the
    probability that ships as the EER threshold and takes the edge off memorised
    examples.
    """
    return y * (1.0 - eps) + 0.5 * eps if eps > 0 else y


def mixup_batch(x: torch.Tensor, y: torch.Tensor, beta) -> tuple:
    """
    Convex-combine the batch with a shuffled copy of itself, targets included.

    This is the single biggest regularisation win for anti-spoofing CNNs, and
    the reason is specific rather than general: the interpolated spectrogram of
    a human and an AI window is not a real recording of anything, so a feature
    that only exists as a per-corpus constant - a noise floor, a fixed level, a
    resampling ripple - is no longer a straight line to the label. The model is
    forced onto features that vary continuously with how much AI is in the mix.
    """
    if beta is None:
        return x, y
    lam = float(beta.sample())
    perm = torch.randperm(x.shape[0], device=x.device)
    return lam * x + (1.0 - lam) * x[perm], lam * y + (1.0 - lam) * y[perm]


def collect_predictions(rows, probs: torch.Tensor):
    """
    One row per window: label, P(ai), prediction, correct.

    Takes the probabilities evaluate() already produced instead of pushing the
    test set through the channel simulation a second time. The test loader is
    shuffle=False with no sampler, so row i and prob i are the same window - the
    second pass was recomputing an identical answer at ~10 ms per window.
    """
    p = probs if probs.numel() else torch.empty(0)
    out = []
    for r, prob in zip(rows, p.tolist()):
        true = LABELS[r["label"]]
        pred = 1 if prob > 0.5 else 0
        out.append({
            "path": r["path"],
            "group": r["group"],
            "source_file": r.get("source_file", ""),
            "window_idx": r.get("window_idx", ""),
            "true_label": r["label"],
            "prob_ai": round(prob, 6),
            "predicted": "ai" if pred else "human",
            "correct": int(pred == true),
        })
    return out


def summarise_groups(preds: list[dict]) -> list[dict]:
    """Aggregate per-window predictions into one row per recording."""
    by_group: dict[str, list[dict]] = defaultdict(list)
    for p in preds:
        by_group[p["group"]].append(p)

    rows = []
    for group, ps in by_group.items():
        n = len(ps)
        acc = sum(p["correct"] for p in ps) / n
        mean_p = sum(p["prob_ai"] for p in ps) / n
        rows.append({
            "group": group,
            "true_label": ps[0]["true_label"],
            "n_windows": n,
            "accuracy": round(acc, 4),
            "mean_prob_ai": round(mean_p, 4),
            # A recording the model gets confidently BACKWARDS is the signature
            # of a wrong label, not of hard audio.
            "suspect_label": int(acc < 0.25),
        })
    rows.sort(key=lambda r: r["accuracy"])
    return rows


def per_group_report(group_rows: list[dict]) -> None:
    print(f"\n  {'recording':44s} {'label':6s} {'n':>4s} {'acc':>6s} {'mean P(ai)':>11s}")
    for r in group_rows:
        flag = "   <-- check this file" if r["suspect_label"] else ""
        print(f"  {r['group'][:44]:44s} {r['true_label']:6s} {r['n_windows']:4d} "
              f"{r['accuracy']:6.2f} {r['mean_prob_ai']:11.2f}{flag}")

    suspect = [r for r in group_rows if r["suspect_label"]]
    if suspect:
        print(f"\n  {len(suspect)} recording(s) scored below 25% - the model is confident "
              f"and WRONG on\n  these, which usually means the label is wrong rather than "
              f"the audio is hard.")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


# --------------------------------------------------------------------- main

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default="datasets/data/manifest.csv")
    ap.add_argument("--epochs", type=int, default=10,
                    help="Upper bound, not a target - early stopping usually "
                         "ends the run first (default 10).")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--test-frac", type=float, default=0.25)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--workers", type=int, default=-1, metavar="N",
                    help="DataLoader workers; -1 (default) picks "
                         "min(8, cpu_count-1). The channel simulation is CPU "
                         "work and is what the GPU waits on, so this is the "
                         "single biggest speed knob. 0 to debug.")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"],
                    help="'cuda' fails loudly if no GPU is visible rather than "
                         "silently training on the CPU for an hour.")
    ap.add_argument("--amp", action="store_true",
                    help="bfloat16 autocast on CUDA. This model is small and "
                         "the pipeline is data-bound, so expect little.")
    ap.add_argument("--no-cache-eval", action="store_true",
                    help="Recompute val/test features every epoch instead of "
                         "caching them. EvalChannel is deterministic so the "
                         "cache changes no number - use only if RAM is tight.")
    ap.add_argument("--no-progress", action="store_true",
                    help="Disable the progress bars (for logging to a file).")
    ap.add_argument("--no-augment", action="store_true",
                    help="Disable BOTH waveform augmentation and SpecAugment. "
                         "The model then trains on clean audio and will not "
                         "survive a real call - diagnostic use only.")
    ap.add_argument("--no-eval-channel", action="store_true",
                    help="Evaluate on clean audio instead of the deployed "
                         "channel. Inflates every number; do not quote it.")
    ap.add_argument("--holdout-group-prefix", action="append", default=[],
                    metavar="PREFIX",
                    help="Hold out every group starting with PREFIX (e.g. "
                         "'ai/asvspoof/A09' or 'ai/rvc'). Repeatable. This is "
                         "leave-one-converter-out - the honest metric.")
    ap.add_argument("--val-holdout-group-prefix", action="append", default=[],
                    metavar="PREFIX",
                    help="Hold out a DIFFERENT generator for validation, so "
                         "checkpoint selection and early stopping are scored on "
                         "a generator that was never trained on. Without this, "
                         "val comes from the training generators and 'best val "
                         "epoch' means 'most overfit epoch'. Repeatable.")
    ap.add_argument("--weight-decay", type=float, default=1e-2, metavar="WD",
                    help="AdamW decoupled weight decay (default 1e-2). 0 makes "
                         "the optimiser plain Adam.")
    ap.add_argument("--mixup", type=float, default=0.4, metavar="ALPHA",
                    help="Mixup Beta(a, a) on spectrograms and targets "
                         "(default 0.4, 0 disables).")
    ap.add_argument("--label-smoothing", type=float, default=0.05, metavar="EPS",
                    help="Pull BCE targets EPS toward 0.5 (default 0.05).")
    ap.add_argument("--pos-weight", default="auto", metavar="W",
                    help="Weight on the positive (ai) term of the BCE loss. "
                         "'auto' = n_human/n_ai over the TRAIN split, which "
                         "makes the two classes contribute equally to the "
                         "gradient; '1' or 'off' = unweighted; or a float. "
                         "Without this a 4:1 ai:human split is minimised by "
                         "leaning on the majority class, which is what the "
                         "flip-flopping val recalls in training_history.csv "
                         "were.")
    ap.add_argument("--patience", type=int, default=3, metavar="N",
                    help="Stop after N epochs with no improvement in val "
                         "balanced accuracy (default 3, 0 disables). Needs a "
                         "val set to mean anything.")
    ap.add_argument("--spec-freq-width", type=int, default=10, metavar="BINS",
                    help="SpecAugment frequency-mask width in bins (default "
                         "10 = ~310 Hz). Raise to ~16 only after the split and "
                         "the data path have been ruled out - a wider mask "
                         "cannot fix a corpus shortcut.")
    ap.add_argument("--no-mfm", action="store_true",
                    help="Use ReLU instead of Max-Feature-Map (disables LCNN)")
    ap.add_argument("--no-freq-coord", action="store_true",
                    help="Drop the absolute-frequency coordinate channel")
    ap.add_argument("--time-pool", default="mean+max",
                    choices=["mean", "max", "mean+max"])
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--no-per-group", action="store_true",
                    help="Skip the per-recording label sanity check")
    ap.add_argument("--max-per-group", type=int, default=0,
                    help="Cap windows taken from any one recording (0 = no cap). "
                         "Stops one long file dominating the training set.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="ml/training/checkpoints/spoofcnn.pt")
    ap.add_argument("--results-dir", default="datasets/data/results",
                    help="Where the results CSVs are written")
    a = ap.parse_args(argv)

    mf = Path(a.manifest)
    if not mf.is_file():
        print(f"manifest not found: {mf}\nRun: python ml/preprocess.py")
        return 1

    rows = load_manifest(mf)
    # Before any capping: the blob is keyed on the manifest's full path set, so
    # changing --max-per-group re-uses it instead of repacking.
    all_paths = [r["path"] for r in rows]
    if a.max_per_group > 0:
        capped, seen = [], Counter()
        for r in rows:
            if seen[r["group"]] < a.max_per_group:
                capped.append(r)
                seen[r["group"]] += 1
        if len(capped) < len(rows):
            print(f"capped at {a.max_per_group}/recording: "
                  f"{len(rows)} -> {len(capped)} windows")
        rows = capped
    print(f"{len(rows)} windows from {mf}")
    print("label counts:", dict(Counter(r['label'] for r in rows)), "\n")

    err = check_labels(rows)
    if err:
        print("CANNOT TRAIN\n")
        print(err)
        return 1

    torch.manual_seed(a.seed)
    random.seed(a.seed)

    if a.device == "cuda" and not torch.cuda.is_available():
        print("--device cuda, but torch.cuda.is_available() is False.")
        print(f"  torch {torch.__version__}   built against CUDA "
              f"{torch.version.cuda}")
        print("  A CPU-only wheel is the usual cause. Reinstall torch and")
        print("  torchaudio together from the CUDA index for your driver.")
        return 1
    use_cuda = a.device != "cpu" and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        # TF32 and cudnn autotuning are free on any modern NVIDIA card and cost
        # nothing in accuracy at this model size.
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        props = torch.cuda.get_device_properties(0)
        print(f"device: cuda:0  {props.name}  "
              f"{props.total_memory / 1e9:.1f} GB VRAM"
              f"{'  bf16 autocast' if a.amp else ''}")
    else:
        print("device: cpu   (no CUDA GPU in use)")

    workers = a.workers if a.workers >= 0 else min(8, max(0, (os.cpu_count() or 2) - 1))
    print(f"dataloader workers: {workers}"
          + ("   <- the channel simulation runs here, not on the GPU"
             if workers else "   <- single-threaded; the GPU will sit idle"))

    held_out: list[str] = []
    val_held_out: list[str] = []
    if a.val_holdout_group_prefix and not a.holdout_group_prefix:
        print("--val-holdout-group-prefix needs --holdout-group-prefix too; "
              "holding a generator out of val while test is a random slice of "
              "recordings measures nothing extra.")
        return 1
    overlap = [v for v in a.val_holdout_group_prefix
               for t in a.holdout_group_prefix
               if v.startswith(t) or t.startswith(v)]
    if overlap:
        print(f"--val-holdout-group-prefix {overlap} overlaps "
              f"--holdout-group-prefix {a.holdout_group_prefix}. They must name "
              f"different generators, or the test set picks its own checkpoint.")
        return 1

    if a.holdout_group_prefix:
        result = split_by_prefix(rows, a.holdout_group_prefix,
                                 a.val_holdout_group_prefix, a.val_frac, a.seed)
        if result is None:
            print(f"no group matches {a.holdout_group_prefix}. Available groups:")
            for g in sorted({r["group"] for r in rows})[:20]:
                print(f"  {g}")
            return 1
        train_rows, val_rows, test_rows, held_out, val_held_out = result
        print(f"LEAVE-ONE-CONVERTER-OUT: holding out {len(held_out)} group(s) "
              f"matching {a.holdout_group_prefix}")
        if a.val_holdout_group_prefix and not val_held_out:
            print(f"no group matches --val-holdout-group-prefix "
                  f"{a.val_holdout_group_prefix}. Available groups:")
            for g in sorted({r["group"] for r in rows})[:20]:
                print(f"  {g}")
            return 1
        if val_held_out:
            print(f"GENERATOR-DISJOINT VAL: {len(val_held_out)} group(s) "
                  f"matching {a.val_holdout_group_prefix} select the checkpoint")
        else:
            print("WARNING: val is a slice of the TRAINING generators, so the\n"
                  "  best-val epoch is the one that memorised them hardest and\n"
                  "  checkpoint selection is rewarding the overfitting it is\n"
                  "  meant to catch. Pass --val-holdout-group-prefix with a\n"
                  "  generator other than the one in --holdout-group-prefix.")
    else:
        train_rows, val_rows, test_rows, _ = split_by_recording(
            rows, a.test_frac, a.val_frac, a.seed)

    def summarise(name, rs):
        recs = sorted({r["group"] for r in rs})
        c = Counter(r["label"] for r in rs)
        print(f"  {name:5s} {len(rs):5d} windows  {dict(c)}  from {len(recs)} recording(s)")
        for r in recs[:12]:
            print(f"          - {r}")
        if len(recs) > 12:
            print(f"          ... and {len(recs) - 12} more")

    print("split BY GROUP (no recording appears in two splits):")
    summarise("train", train_rows)
    summarise("val", val_rows)
    summarise("test", test_rows)
    print()

    if not train_rows or not test_rows:
        print("split produced an empty train or test set - too few recordings.")
        return 1

    # A one-class TRAIN split is fatal and silent. check_labels() above proves
    # the manifest has both classes, but holding a generator out of test and
    # another out of val can take every AI group with them - and then the model
    # trains on human only, drives its loss down predicting one class, and
    # reports recall_ai 0.000 with balanced accuracy 0.500. That looks exactly
    # like a model that failed to learn rather than one that was never given
    # anything to learn, which is a day lost to the wrong question.
    train_labels = Counter(r["label"] for r in train_rows)
    if len(train_labels) < 2:
        only = next(iter(train_labels))
        missing = "ai" if only == "human" else "human"
        held_all = sorted(set(held_out) | set(val_held_out))
        print(f"the TRAINING split is all '{only}' - every '{missing}' group "
              f"went into the holdouts.\n")
        print(f"  held out: {', '.join(held_all) if held_all else '(none)'}")
        print(f"  left to train on: "
              f"{len({r['group'] for r in train_rows})} group(s), all '{only}'\n")
        print("A model trained on one class predicts that class for everything, "
              "reaches a\nlow loss doing it, and scores 0.500 balanced accuracy "
              "- which reads as 'it\nlearned nothing' rather than 'it was given "
              "nothing'. Refusing instead.\n")
        print(f"Hold out fewer groups, or add more '{missing}' groups so some "
              f"survive the split.")
        return 1

    if len(set(r["label"] for r in test_rows)) < 2:
        print("WARNING: the test set contains only one class, so test accuracy is\n"
              "not meaningful. You need at least two GROUPS per class - check the\n"
              "coarse-group warning from ml/preprocess.py.\n")
    if val_rows and len(set(r["label"] for r in val_rows)) < 2:
        print("WARNING: the validation set contains only one class, so balanced\n"
              "accuracy is undefined and epoch selection falls back to saving the\n"
              "FINAL epoch rather than the best one. Add more groups per class.\n")

    # ---- the channel. Random for train, fixed for val/test.
    augment_enabled = not a.no_augment
    train_channel = WaveformAugment(enabled=augment_enabled)
    eval_channel = EvalChannel(enabled=not a.no_eval_channel)
    spec_aug = SpecAugment(freq_width=a.spec_freq_width,
                           enabled=augment_enabled)
    extractor = LogLinearSpectrogram()

    print(f"train channel: {train_channel.config()}")
    print(f"eval  channel: {eval_channel.config()}")

    n_bins = n_freq_bins_for(None)
    model = SpoofCNN(n_freq_bins=n_bins, dropout=a.dropout,
                     mfm=not a.no_mfm, freq_coord=not a.no_freq_coord,
                     time_pool=a.time_pool).to(device)
    print(f"\nSpoofCNN {model.n_parameters():,} params, input {feature_shape()}, "
          f"device={device}")
    print(f"  config: {model.config()}\n")

    store = WaveformStore(all_paths, mf.parent)

    train_ds = WindowDataset(train_rows, channel=train_channel,
                             spec_aug=spec_aug, extractor=extractor, store=store)

    lb = train_ds.labels()
    counts = torch.bincount(torch.tensor(lb), minlength=2).float().clamp(min=1)
    weights = (1.0 / counts)[torch.tensor(lb)]
    loader_kw = dict(num_workers=workers, worker_init_fn=seed_worker,
                     persistent_workers=workers > 0,
                     pin_memory=use_cuda)
    if workers > 0:
        loader_kw["prefetch_factor"] = 4
    train_loader = DataLoader(
        train_ds, batch_size=a.batch_size,
        sampler=WeightedRandomSampler(weights, len(lb), replacement=True),
        **loader_kw)

    # Val and test go through the FIXED channel, so their features are identical
    # on every epoch. Build them once here; after this, evaluating is a bare GPU
    # forward pass over tensors already in RAM.
    def eval_loader(rs, desc):
        if not rs:
            return None
        if a.no_cache_eval:
            ds = WindowDataset(rs, channel=eval_channel, extractor=extractor,
                               store=store)
            return DataLoader(ds, batch_size=a.batch_size, **loader_kw)
        ds = precompute_features(rs, store, eval_channel, extractor,
                                 a.batch_size, workers, desc)
        return DataLoader(ds, batch_size=a.batch_size, shuffle=False,
                          num_workers=0, pin_memory=use_cuda)

    val_loader = eval_loader(val_rows, "caching val features")
    test_loader = eval_loader(test_rows, "caching test features")

    # AdamW, not Adam. Adam's weight decay is folded into the gradient and is
    # then divided by the per-parameter second-moment estimate, so parameters
    # with small gradients - exactly the ones a memorised shortcut lives in -
    # are barely decayed at all. AdamW decouples it and actually shrinks them.
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr,
                            weight_decay=a.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    # Class weighting. The counts come from the TRAIN split alone - deriving
    # them from the whole manifest would let the held-out generators' size
    # influence training, which is a leak, however small.
    train_counts = Counter(r["label"] for r in train_rows)
    n_ai = train_counts.get("ai", 0)
    n_human = train_counts.get("human", 0)
    if str(a.pos_weight).lower() in ("off", "none", ""):
        pos_w = 1.0
    elif str(a.pos_weight).lower() == "auto":
        pos_w = (n_human / n_ai) if n_ai else 1.0
    else:
        pos_w = float(a.pos_weight)
    if abs(pos_w - 1.0) < 1e-9:
        loss_fn = nn.BCEWithLogitsLoss()
        print(f"loss: BCE unweighted  (train ai={n_ai} human={n_human})")
    else:
        loss_fn = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(pos_w, device=device))
        print(f"loss: BCE pos_weight={pos_w:.3f}"
              f"  (train ai={n_ai} human={n_human})")

    # Mixup and label smoothing are regularisation, not channel simulation, but
    # --no-augment exists to answer one question - "is this a shortcut?" - by
    # training a model with nothing standing in its way. Leaving these on would
    # blunt exactly the overfitting that probe is trying to expose.
    mix_beta = (torch.distributions.Beta(torch.tensor(a.mixup),
                                         torch.tensor(a.mixup))
                if a.mixup > 0 and augment_enabled else None)
    smoothing = a.label_smoothing if augment_enabled else 0.0
    print(f"regularisation: AdamW wd={a.weight_decay:g}  "
          f"mixup={'off' if mix_beta is None else f'{a.mixup:g}'}  "
          f"label_smoothing={smoothing:g}  dropout={a.dropout:g}  "
          f"patience={a.patience if val_loader else 'n/a (no val set)'}")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    best, best_epoch, stale, epochs_run = -1.0, 0, 0, 0
    history = []

    def checkpoint(epoch: int) -> dict:
        return {
            "state_dict": model.state_dict(),
            # The whole architecture, in one place, read straight back by
            # training.model.from_checkpoint(). Nothing is inferred at load.
            "model_config": model.config(),
            "band_crop": DEFAULT_BAND_CROP,
            "n_freq_bins": n_bins,
            "feature_shape": list(feature_shape()),
            "train_channel": train_channel.config(),
            "eval_channel": eval_channel.config(),
            "spec_augment": spec_aug.config(),
            "epoch": epoch,
            "holdout_groups": sorted(held_out),
            "val_holdout_groups": sorted(val_held_out),
            "test_groups": sorted({r["group"] for r in test_rows}),
        }

    autocast = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) \
        if (use_cuda and a.amp) else nullcontext
    started = time.perf_counter()

    for ep in range(1, a.epochs + 1):
        model.train()
        tot = seen = 0
        bar = tqdm(train_loader, desc=f"epoch {ep:>3d}/{a.epochs}", unit="batch",
                   leave=False, dynamic_ncols=True, disable=a.no_progress)
        for b in bar:
            x = b["spectrogram"].to(device, non_blocking=True)
            y = b["label"].to(device, non_blocking=True)
            x, y = mixup_batch(x, smooth_targets(y, smoothing), mix_beta)
            opt.zero_grad(set_to_none=True)
            with autocast():
                loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            tot += loss.item() * y.numel()
            seen += y.numel()
            bar.set_postfix(loss=f"{tot / max(seen, 1):.4f}")
        bar.close()
        sched.step()

        epochs_run = ep
        train_loss = tot / max(seen, 1)
        line = (f"epoch {ep:3d}/{a.epochs}  loss {train_loss:.4f}"
                f"  [{time.perf_counter() - started:.0f}s]")
        v = evaluate(model, val_loader, device, f"epoch {ep:>3d} val") \
            if val_loader else {}
        if v and v.get("n"):
            line += (f"  val bal {v['balanced_acc']:.3f}  eer {v['eer']:.3f}")
            # Select on balanced accuracy so checkpointing cannot be gamed by
            # drifting toward the majority class.
            score = v["balanced_acc"]
        else:
            score = -train_loss
        print(line)
        history.append({
            "epoch": ep,
            "train_loss": round(train_loss, 6),
            "val_acc": round(v["acc"], 4) if v.get("n") else "",
            "val_balanced_acc": round(v["balanced_acc"], 4) if v.get("n") else "",
            "val_eer": round(v["eer"], 4) if v.get("n") else "",
            "val_recall_ai": round(v["recall_ai"], 4) if v.get("n") else "",
            "val_recall_human": round(v["recall_human"], 4) if v.get("n") else "",
            "lr": round(sched.get_last_lr()[0], 8),
        })

        # nan (a single-class val set) must not count as an improvement OR as
        # a stale epoch - otherwise patience fires after 3 epochs having never
        # saved anything.
        if math.isfinite(score) and score > best:
            best, best_epoch, stale = score, ep, 0
            torch.save(checkpoint(ep), a.out)
        elif math.isfinite(score):
            stale += 1
            if a.patience > 0 and stale >= a.patience:
                where = ("unseen generator" if val_held_out
                         else "held-out recordings")
                print(f"early stop: {stale} epochs with no improvement on the "
                      f"val set ({where}); best was epoch {best_epoch} at "
                      f"{best:.3f}")
                break

    if best < 0:                      # no val set: save the final epoch
        torch.save(checkpoint(epochs_run), a.out)
    print(f"\nsaved -> {a.out}")

    # Reload the best checkpoint so the reported numbers describe the saved file.
    from training.model import from_checkpoint
    ckpt = torch.load(a.out, map_location=device, weights_only=False)
    model = from_checkpoint(ckpt).to(device)

    t = evaluate(model, test_loader, device, "test")
    label = ("UNSEEN CONVERTER" if held_out else "held-out RECORDINGS")
    print(f"\n--- {label} (the number to quote) ---")
    print(f"  windows      {t['n']}")
    print(f"  accuracy     {t['acc']:.3f}")
    print(f"  recall AI    {t['recall_ai']:.3f}")
    print(f"  recall human {t['recall_human']:.3f}")
    print(f"  BALANCED acc {t['balanced_acc']:.3f}   <- quote this, not plain accuracy")
    print(f"  EER          {t['eer']:.3f}   <- threshold-free, comparable across models")
    print(f"  EER thresh   {t['eer_threshold']:.3f}   <- the operating point that "
          f"ships, not 0.5")

    # Write the operating point back into the checkpoint. ml/export_onnx.py copies
    # it into the .json beside the .onnx, and the phone reads it from there, so
    # the number the metric was computed at is the number that ships. Without
    # this the deployed threshold is 0.5 - an arbitrary place to cut a sigmoid
    # that happens to match nothing that was measured.
    # ---- the three numbers, side by side, with what they mean.
    best_val = best if best >= 0 else float("nan")
    print("\n--- is it learning the voice, or the corpus? ---")
    print(f"  train loss (last epoch)      {train_loss:.4f}")
    print(f"  val balanced acc (best)      "
          f"{best_val:.3f}   [{'unseen generator' if val_held_out else 'SAME generators as train - not a real check'}]")
    print(f"  held-out generator bal acc   {t['balanced_acc']:.3f}"
          if held_out else
          f"  held-out recording bal acc   {t['balanced_acc']:.3f}")
    if train_loss < 0.05 and t["balanced_acc"] < 0.65:
        print("\n  SHORTCUT. The model fits the training data essentially "
              "perfectly and\n  transfers nothing, which means it found "
              "something that separates the\n  two corpora and does not exist "
              "in the held-out one. This is a DATA\n  problem: dropout, weight "
              "decay and mixup will not touch it. Pair the\n  sources so only "
              "the vocoder differs between the classes, and re-run\n  "
              "ml/preprocess.py so both go through the identical path.")
    elif train_loss < 0.05 and t["balanced_acc"] < 0.85:
        print("\n  Overfitting, but it is generalising somewhat. Raise "
              "--dropout / --weight-decay,\n  cap windows per recording with "
              "--max-per-group, and re-run ml/preprocess.py\n  with "
              "--hop-samples 16000 so overlapping near-duplicates stop being "
              "counted\n  as independent evidence.")
    elif math.isfinite(best_val) and best_val - t["balanced_acc"] > 0.15:
        print("\n  Generalises within the training generators but not to the "
              "held-out one -\n  the features it uses are generator-specific. "
              "More generators in training\n  helps here; more regularisation "
              "does not.")
    if not val_held_out and held_out:
        print("\n  NB: val was not generator-disjoint, so the checkpoint was "
              "selected on an\n  epoch's fit to generators it trained on. "
              "Re-run with --val-holdout-group-prefix.")

    ckpt["eer_threshold"] = t["eer_threshold"]
    ckpt["test_metrics"] = {k: t[k] for k in
                            ("n", "acc", "balanced_acc", "eer", "eer_threshold",
                             "recall_ai", "recall_human")}
    torch.save(ckpt, a.out)
    if held_out:
        print(f"\n  Held out entirely: {', '.join(held_out)}")
    print("\nScored through the deployed channel (phone codec + speakerphone "
          "re-capture)." if not a.no_eval_channel else
          "\nScored on CLEAN audio - this number is inflated.")

    preds = collect_predictions(test_rows, t["probs"])
    group_rows = summarise_groups(preds)

    if not a.no_per_group:
        print("\n--- per-recording breakdown (label sanity check) ---")
        per_group_report(group_rows)

    res = Path(a.results_dir)
    summary = [{"metric": k, "value": v} for k, v in [
        ("test_windows", t["n"]),
        ("test_accuracy", round(t["acc"], 4)),
        ("test_balanced_accuracy", round(t["balanced_acc"], 4)),
        ("test_eer", round(t["eer"], 4)),
        ("test_eer_threshold", round(t["eer_threshold"], 4)),
        ("test_recall_ai", round(t["recall_ai"], 4)),
        ("test_recall_human", round(t["recall_human"], 4)),
        ("best_val_balanced_accuracy", round(best, 4)),
        ("train_windows", len(train_rows)),
        ("val_windows", len(val_rows)),
        ("n_train_recordings", len({r["group"] for r in train_rows})),
        ("n_test_recordings", len({r["group"] for r in test_rows})),
        ("suspect_recordings", sum(r["suspect_label"] for r in group_rows)),
        ("epochs", a.epochs),
        ("epochs_run", epochs_run),
        ("best_val_epoch", best_epoch),
        ("weight_decay", a.weight_decay),
        ("mixup_alpha", a.mixup if mix_beta is not None else 0.0),
        ("label_smoothing", smoothing),
        ("dropout", a.dropout),
        ("patience", a.patience),
        ("spec_freq_width", a.spec_freq_width),
        ("val_generator_disjoint", int(bool(val_held_out))),
        ("val_holdout_groups", ";".join(val_held_out) if val_held_out else ""),
        ("pos_weight", round(pos_w, 6)),
        ("max_per_group", a.max_per_group),
        ("augment", int(augment_enabled)),
        ("eval_channel", int(not a.no_eval_channel)),
        ("feature_shape", "x".join(str(int(d)) for d in feature_shape())),
        ("model_params", model.n_parameters()),
        ("model_config", json.dumps(model.config())),
        ("seed", a.seed),
        ("holdout_groups", ";".join(held_out) if held_out else ""),
    ]]
    write_csv(res / "summary.csv", summary)
    write_csv(res / "per_recording.csv", group_rows)
    write_csv(res / "per_window.csv", preds)
    write_csv(res / "training_history.csv", history)

    elapsed = time.perf_counter() - started
    print(f"\ntotal wall clock: {elapsed:.0f}s "
          f"({elapsed / max(epochs_run, 1):.1f}s/epoch over {epochs_run} epoch(s))")
    print(f"\nresults written to {res}/")
    for name in ("summary.csv", "per_recording.csv", "per_window.csv",
                 "training_history.csv"):
        print(f"  {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

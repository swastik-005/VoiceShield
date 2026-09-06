"""
Fit diagnostic: is SpoofCNN learning, memorising, under- or overfitting?

ml/run_training.py reports TRAIN LOSS (measured through the random augmentation
channel plus SpecAugment) against VAL BALANCED ACCURACY (measured through the
fixed EvalChannel, no SpecAugment). Those two numbers are not comparable, so
the usual train-vs-val gap cannot be read off the history at all.

This script fixes that: it scores the saved checkpoint on train, val and test
through the SAME fixed EvalChannel with augmentation off, so the three numbers
differ only in which recordings they came from. It also scores train through
the training channel, to separate "the model is weak" from "the augmentation
is hard".
"""
from __future__ import annotations

import argparse
import random
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from features import LogLinearSpectrogram
from run_training import (WaveformStore, evaluate, load_manifest,
                          precompute_features, split_by_prefix,
                          split_by_recording)
from training.augment import EvalChannel, SpecAugment, WaveformAugment
from training.model import from_checkpoint


def subsample(rows, cap, seed):
    """Cap windows per split, round-robin by group so no recording is lost."""
    if cap <= 0 or len(rows) <= cap:
        return rows
    by_group = defaultdict(list)
    for r in rows:
        by_group[r["group"]].append(r)
    rng = random.Random(seed)
    for v in by_group.values():
        rng.shuffle(v)
    out, i = [], 0
    while len(out) < cap:
        added = False
        for g in sorted(by_group):
            if i < len(by_group[g]):
                out.append(by_group[g][i])
                added = True
                if len(out) >= cap:
                    break
        if not added:
            break
        i += 1
    return out


def score(model, rows, store, channel, extractor, device, bs, workers, desc,
          spec_aug=None):
    if not rows:
        return {"n": 0}
    ds = precompute_features(rows, store, channel, extractor, bs, workers, desc)
    if spec_aug is not None:
        ds.specs = torch.stack([spec_aug(s) for s in ds.specs])
    loader = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=0)
    m = evaluate(model, loader, device, desc)
    # BCE on the same probabilities, so loss is comparable across splits too.
    y = torch.tensor([1.0 if r["label"] == "ai" else 0.0 for r in rows])
    p = m["probs"].clamp(1e-6, 1 - 1e-6)
    m["loss"] = float(nn.functional.binary_cross_entropy(p, y))
    m["rows"] = rows
    return m


def line(name, m):
    if not m.get("n"):
        return f"  {name:<22s}  (empty)"
    return (f"  {name:<22s} {m['n']:6d} win   loss {m['loss']:6.4f}   "
            f"bal-acc {m['balanced_acc']:.3f}   acc {m['acc']:.3f}   "
            f"EER {m['eer']:.3f}   R_ai {m['recall_ai']:.3f}  "
            f"R_hu {m['recall_human']:.3f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="datasets/data/manifest.csv")
    ap.add_argument("--ckpt", default="ml/training/checkpoints/spoofcnn.pt")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--test-frac", type=float, default=0.25)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--holdout-group-prefix", action="append", default=[])
    ap.add_argument("--val-holdout-group-prefix", action="append", default=[],
                    help="Must repeat what the training run used. The split is "
                         "recomputed here, so a different holdout means these "
                         "numbers describe splits the checkpoint never saw.")
    ap.add_argument("--cap", type=int, default=4000,
                    help="max windows scored per split (0 = all)")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()

    rows = load_manifest(Path(a.manifest))
    all_paths = [r["path"] for r in rows]
    random.seed(a.seed)
    torch.manual_seed(a.seed)

    if a.holdout_group_prefix:
        train_rows, val_rows, test_rows, _, _ = split_by_prefix(
            rows, a.holdout_group_prefix, a.val_holdout_group_prefix,
            a.val_frac, a.seed)
    else:
        train_rows, val_rows, test_rows, _ = split_by_recording(
            rows, a.test_frac, a.val_frac, a.seed)

    device = torch.device("cuda" if (a.device != "cpu" and torch.cuda.is_available())
                          else "cpu")
    ckpt = torch.load(a.ckpt, map_location=device, weights_only=False)
    model = from_checkpoint(ckpt).to(device).eval()

    print(f"checkpoint {a.ckpt}  epoch {ckpt.get('epoch')}  "
          f"{model.n_parameters():,} params  device={device}")
    print(f"split seed={a.seed} test_frac={a.test_frac} val_frac={a.val_frac}")
    for nm, rs in (("train", train_rows), ("val", val_rows), ("test", test_rows)):
        c = Counter(r["label"] for r in rs)
        print(f"  {nm:5s} {len(rs):6d} windows  {dict(c)}  "
              f"{len({r['group'] for r in rs})} group(s)")

    store = WaveformStore(all_paths, Path(a.manifest).parent)
    extractor = LogLinearSpectrogram()
    eval_ch = EvalChannel(enabled=True)
    train_ch = WaveformAugment(enabled=True)
    spec_aug = SpecAugment(enabled=True)

    tr = subsample(train_rows, a.cap, a.seed)
    va = subsample(val_rows, a.cap, a.seed)
    te = subsample(test_rows, a.cap, a.seed)

    print("\n--- scored through the FIXED eval channel, no SpecAugment "
          "(only the recordings differ) ---")
    m_tr = score(model, tr, store, eval_ch, extractor, device, a.batch_size,
                 a.workers, "train(eval-ch)")
    m_va = score(model, va, store, eval_ch, extractor, device, a.batch_size,
                 a.workers, "val")
    m_te = score(model, te, store, eval_ch, extractor, device, a.batch_size,
                 a.workers, "test")
    print(line("TRAIN (seen)", m_tr))
    print(line("VAL   (unseen)", m_va))
    print(line("TEST  (unseen)", m_te))

    print("\n--- the same TRAIN windows through the TRAINING channel "
          "(what the loss curve actually measures) ---")
    m_tr_aug = score(model, tr, store, train_ch, extractor, device, a.batch_size,
                     a.workers, "train(train-ch)", spec_aug=spec_aug)
    print(line("TRAIN (augmented)", m_tr_aug))

    print("\n--- verdict inputs ---")
    if m_tr.get("n") and m_va.get("n"):
        print(f"  generalisation gap train->val : "
              f"bal-acc {m_tr['balanced_acc'] - m_va['balanced_acc']:+.3f}   "
              f"loss {m_va['loss'] - m_tr['loss']:+.4f}")
    if m_tr.get("n") and m_te.get("n"):
        print(f"  generalisation gap train->test: "
              f"bal-acc {m_tr['balanced_acc'] - m_te['balanced_acc']:+.3f}   "
              f"loss {m_te['loss'] - m_tr['loss']:+.4f}")
    if m_tr.get("n") and m_tr_aug.get("n"):
        print(f"  augmentation cost on train    : bal-acc "
              f"{m_tr['balanced_acc'] - m_tr_aug['balanced_acc']:+.3f}   "
              f"loss {m_tr_aug['loss'] - m_tr['loss']:+.4f}")

    print("\n--- per-group accuracy (memorisation shows up here) ---")
    for nm, m in (("train", m_tr), ("val", m_va), ("test", m_te)):
        if not m.get("n"):
            continue
        agg = defaultdict(lambda: [0, 0])
        for r, p in zip(m["rows"], m["probs"].tolist()):
            pred = 1 if p > 0.5 else 0
            truth = 1 if r["label"] == "ai" else 0
            agg[r["group"]][0] += int(pred == truth)
            agg[r["group"]][1] += 1
        print(f"  [{nm}]")
        for g, (ok, n) in sorted(agg.items(), key=lambda kv: kv[1][0] / kv[1][1]):
            flag = "  <-- worse than chance" if ok / n < 0.5 else ""
            print(f"    {ok / n:5.3f}  {ok:5d}/{n:<5d}  {g[:70]}{flag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
Trained checkpoint -> ONNX, for ONNX Runtime on Android.

    python ml/export_onnx.py
    python ml/export_onnx.py --checkpoint ml/training/checkpoints/holdout_sia.pt

THE CONTRACT THE PHONE SEES
---------------------------
    input   "spectrogram"  float32 [batch, 1, F, T]   F, T from ml/features.py
    output  "logit"        float32 [batch]

Note the output rank: SpoofCNN squeezes the channel, so this is [batch], NOT
[batch, 1]. ONNX Runtime hands Kotlin a FloatBuffer of length `batch` - read
element 0, do not index [0][0].

ONE LOGIT, NOT A PROBABILITY. Kotlin must apply sigmoid itself:

    val pAi = 1f / (1f + exp(-logit))

Exporting the sigmoid into the graph would have been friendlier and is wrong
here: the threshold sweep that picks an operating point (and the EER in
datasets/data/results/summary.csv) works in logit space, and a saturated sigmoid at
float32 throws away the ordering the sweep depends on. Anything above ~16 or
below ~-16 collapses to exactly 1.0 or 0.0 and stops being rankable.

WHY from_checkpoint() AND NOT SpoofCNN()
----------------------------------------
The architecture switches (mfm, freq_coord, time_pool, n_freq_bins) live in the
checkpoint, written there by ml/run_training.py. Constructing a default SpoofCNN
and loading weights into it works silently when the switches happen to match and
produces confident nonsense when they do not. from_checkpoint() rebuilds the
architecture the weights were actually trained with, so a mismatch is a load
error instead of a wrong answer on a phone.

WHY THE PARITY CHECK IS NOT OPTIONAL
------------------------------------
FrequencyCoord builds its ramp from x.shape at forward time, and time_pool
"mean+max" concatenates two reductions. Both are the kind of thing a tracer can
constant-fold against the example batch it was handed. An export that folded the
batch size still loads, still runs, and is wrong for every batch but the one it
was traced with - so this script re-runs both engines on real windows from the
manifest and refuses to leave a file behind that does not match.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from features import (
    BAND_CROP_HI,
    BAND_CROP_LO,
    BIN_HZ,
    DEFAULT_BAND_CROP,
    FRAME_MS,
    HOP_LENGTH,
    HOP_WINDOW_SAMPLES,
    N_FFT,
    SAMPLE_RATE,
    WINDOW_SAMPLES,
    feature_shape,
)
from training.model import from_checkpoint

# Above this the two engines disagree by more than float32 accumulation order
# explains, and something structural is wrong with the graph.
TOLERANCE = 1e-4


def real_windows(manifest: Path, n: int) -> torch.Tensor | None:
    """
    n spectrograms built from actual cached audio, through the eval channel.

    Random noise exercises the graph shape but not its numerics: a real
    spectrogram is dominated by a few loud low bins, which is exactly where a
    folded constant or a transposed axis shows up as a large absolute error.
    """
    if not manifest.is_file():
        return None
    import csv
    import random

    # WindowDataset, not a reimplementation of its slicing: the ragged-tail pad
    # and the channel order are part of what is being verified, so the check has
    # to run the same code path evaluation does.
    from run_training import WindowDataset
    from training.augment import EvalChannel

    with open(manifest, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return None

    random.Random(0).shuffle(rows)
    ds = WindowDataset(rows[:n], channel=EvalChannel())

    specs = []
    for i in range(len(ds)):
        try:
            specs.append(ds[i]["spectrogram"])
        except Exception:
            continue
    return torch.stack(specs) if specs else None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default="ml/training/checkpoints/spoofcnn.pt")
    ap.add_argument("--out", default=None,
                    help="Output .onnx (default: checkpoint path with .onnx)")
    ap.add_argument("--manifest", default="datasets/data/manifest.csv",
                    help="Source of real windows for the parity check")
    ap.add_argument("--opset", type=int, default=17,
                    help="ONNX opset. 17 is what onnxruntime-android ships "
                         "against; raise it only if you know the runtime on "
                         "the phone is newer.")
    ap.add_argument("--check-batches", type=int, nargs="+", default=[1, 4, 16],
                    help="Batch sizes to verify. More than one is the point - "
                         "it is how a folded batch dimension gets caught.")
    ap.add_argument("--no-dynamic-batch", action="store_true",
                    help="Export a fixed batch of 1. Smaller and simpler, and "
                         "the phone only ever infers one window at a time.")
    a = ap.parse_args(argv)

    ckpt_path = Path(a.checkpoint)
    if not ckpt_path.is_file():
        print(f"not found: {ckpt_path}\nTrain first: python ml/run_training.py")
        return 1

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = from_checkpoint(ckpt).to("cpu").eval()

    c, f, t = feature_shape()
    cfg = ckpt.get("model_config") or {}
    n_params = sum(p.numel() for p in model.parameters())
    print(f"checkpoint : {ckpt_path}")
    print(f"epoch      : {ckpt.get('epoch', '?')}")
    print(f"config     : {cfg}")
    print(f"input      : [batch, {c}, {f}, {t}]   ({n_params} params)")

    if f != int(cfg.get("n_freq_bins", f)):
        print(f"\nMISMATCH: checkpoint trained on {cfg['n_freq_bins']} freq bins, "
              f"ml/features.py now says {f}.\nThe phone would feed the wrong shape. "
              f"Fix ml/features.py or retrain.")
        return 1

    out_path = Path(a.out) if a.out else ckpt_path.with_suffix(".onnx")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    example = torch.randn(1, c, f, t)
    dynamic = None if a.no_dynamic_batch else {"spectrogram": {0: "batch"},
                                               "logit": {0: "batch"}}
    # dynamo=False is deliberate, and torch warns about it since 2.9. The
    # TorchScript exporter emits a flat opset-17 graph that onnxruntime-android
    # loads without a fuss; the dynamo path produces newer ops and a shape
    # regime worth adopting only once there is a phone to test it on. The parity
    # check below is what actually licenses this choice - if the legacy tracer
    # ever folds something it should not, the export fails rather than ships.
    torch.onnx.export(
        model, (example,), str(out_path),
        input_names=["spectrogram"], output_names=["logit"],
        dynamic_axes=dynamic, opset_version=a.opset,
        do_constant_folding=True, dynamo=False)

    import onnx
    import onnxruntime as ort

    graph = onnx.load(str(out_path))
    onnx.checker.check_model(graph)

    # onnx.checker only validates that the graph is well-formed, not that any
    # runtime can execute it. A graph promoted to float64 passes the checker and
    # then fails on the phone with NOT_IMPLEMENTED, because ONNX Runtime ships
    # no double Conv kernel - so name that case explicitly rather than letting a
    # stack trace be the diagnosis.
    f64 = [n.name for n in graph.graph.node for at in n.attribute
           if at.name == "value" and at.t.data_type == onnx.TensorProto.DOUBLE]
    try:
        sess = ort.InferenceSession(str(out_path),
                                    providers=["CPUExecutionProvider"])
    except Exception as e:
        out_path.unlink(missing_ok=True)
        print(f"\nONNX Runtime refused the graph: {type(e).__name__}: {e}")
        if f64:
            print(f"\nThe graph contains {len(f64)} float64 constant(s) "
                  f"({', '.join(f64[:3])}). Something in the model built a "
                  f"tensor at double precision and promoted the network with "
                  f"it. Look for a dtype= argument to a tensor factory in "
                  f"ml/training/model.py and cast after construction instead.")
        print("\nremoved the .onnx rather than leave a file that cannot load.")
        return 1

    batch_desc = "fixed at 1" if a.no_dynamic_batch else "dynamic"
    print(f"\nexported   : {out_path}  ({out_path.stat().st_size / 1024:.0f} KB, "
          f"opset {a.opset}, batch {batch_desc})")
    for i in sess.get_inputs():
        print(f"  in  {i.name:12s} {i.type} {i.shape}")
    for o in sess.get_outputs():
        print(f"  out {o.name:12s} {o.type} {o.shape}")

    batches = [1] if a.no_dynamic_batch else a.check_batches
    real = real_windows(Path(a.manifest), max(batches))
    source = f"real windows from {a.manifest}" if real is not None \
        else "random input - manifest unreadable"
    print(f"\nparity check ({source}):")

    worst = 0.0
    for n in batches:
        if real is not None and real.shape[0] >= n:
            x = real[:n]
        else:
            x = torch.randn(n, c, f, t)
        with torch.no_grad():
            ref = model(x).numpy()
        got = sess.run(["logit"], {"spectrogram": x.numpy()})[0]
        if got.shape != ref.shape:
            print(f"  batch {n:3d}  SHAPE MISMATCH torch {ref.shape} "
                  f"vs onnx {got.shape}")
            out_path.unlink(missing_ok=True)
            print("\nremoved the .onnx - a folded batch dimension would ship a "
                  "model that is wrong for every batch but 1.")
            return 1
        d = float(abs(ref - got).max())
        worst = max(worst, d)
        flag = "   FAIL" if d > TOLERANCE else ""
        print(f"  batch {n:3d}  max|torch - onnx| = {d:.3e}{flag}")

    if worst > TOLERANCE:
        out_path.unlink(missing_ok=True)
        print(f"\nFAILED: {worst:.3e} > {TOLERANCE:.0e}. Removed the .onnx rather "
              f"than ship a graph that disagrees with the checkpoint.")
        return 1

    # The operating point measured on the held-out set, defaulting to 0.5.
    #
    # NaN needs its own test rather than `or 0.5`: NaN is truthy, so `nan or
    # 0.5` is nan, json.dumps writes a bare NaN token that is not valid JSON,
    # and org.json on the phone throws while parsing the metadata. A single-
    # class test set is exactly when eer_with_threshold() returns NaN, so this
    # is reachable, not theoretical.
    _raw_thr = ckpt.get("eer_threshold")
    _has_thr = _raw_thr is not None and float(_raw_thr) == float(_raw_thr)
    _threshold = float(_raw_thr) if _has_thr else 0.5
    _threshold_source = ("eer on held-out set" if _has_thr else
                         "DEFAULT 0.5 - checkpoint carries no usable EER "
                         "threshold, re-run ml/run_training.py")

    # The phone rebuilds the spectrogram itself, in Kotlin. If any constant here
    # drifts from ml/features.py the input is silently wrong - same shape, wrong
    # content - so the contract ships beside the weights instead of being
    # retyped into Spectrogram.kt from memory.
    meta = {
        "model": out_path.name,
        "input_name": "spectrogram",
        "output_name": "logit",
        "input_shape": [1 if a.no_dynamic_batch else None, c, f, t],
        "output_activation": "sigmoid",
        "labels": {"0": "human", "1": "ai"},
        # The operating point measured on the held-out set by ml/run_training.py,
        # not 0.5. Kotlin compares sigmoid(logit) against THIS. Older
        # checkpoints predate the field and fall back to 0.5, which is what the
        # code did before the number existed.
        "p_ai_threshold": _threshold,
        "p_ai_threshold_source": _threshold_source,
        "test_metrics": ckpt.get("test_metrics") or {},
        "sample_rate": SAMPLE_RATE,
        "window_samples": WINDOW_SAMPLES,
        "hop_samples": HOP_WINDOW_SAMPLES,
        "n_fft": N_FFT,
        "hop_length": HOP_LENGTH,
        "band_crop": DEFAULT_BAND_CROP,
        "band_crop_lo_bin": BAND_CROP_LO,
        "band_crop_hi_bin": BAND_CROP_HI,
        "bin_hz": BIN_HZ,
        "frame_ms": FRAME_MS,
        "opset": a.opset,
        "source_checkpoint": ckpt_path.as_posix(),
        "model_config": cfg,
    }
    meta_path = out_path.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"\nOK. Ship both:\n  {out_path}\n  {meta_path}")
    print("\nKotlin reads ONE logit - apply sigmoid there:\n"
          "  val pAi = 1f / (1f + exp(-logit))")
    return 0


if __name__ == "__main__":
    sys.exit(main())

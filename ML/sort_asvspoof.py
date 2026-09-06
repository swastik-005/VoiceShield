"""
ASVspoof protocol file -> correctly sharded datasets/data/raw/.

    python ml/sort_asvspoof.py ASVspoof5.dev.track_1.tsv --audio-dir datasets/data/raw/ai --dry-run
    python ml/sort_asvspoof.py ASVspoof5.dev.track_1.tsv --audio-dir datasets/data/raw/ai

Bonafide files go to   datasets/data/raw/human/asvspoof/<speaker>/
Spoof files go to      datasets/data/raw/ai/asvspoof/<attack_id>/

WHY THIS SCRIPT EXISTS
----------------------
ASVspoof ships as one flat directory of ~90,000 files plus a protocol file that
says which is which. Dropped into datasets/data/raw/ai/ as-is you get:

  * every file labelled 'ai', including the bonafide half - the model then
    learns nothing except that ASVspoof audio is ASVspoof audio;
  * ONE group for the whole corpus, so ml/run_training.py's group-wise split puts
    all of it in train or all of it in test.

ml/preprocess.py cannot fix either problem, because the label comes from the folder
name and the group comes from the subfolder. The protocol file has the
information; this script applies it to the directory tree.

The sharding is the point. One directory per SPEAKER on the bonafide side and
one per ATTACK ID on the spoof side is exactly what makes
`ml/run_training.py --holdout-group-prefix ai/asvspoof/A09` work - hold out one
generator entirely, train on the others, report on the one never seen. That is
the only number worth quoting, and it is impossible without this layout.

PROTOCOL FORMATS
----------------
The column order differs between ASVspoof editions and nothing in the file
declares it:

    2019 LA   LA_0079 LA_T_1138215 - -     bonafide
    2019 LA   LA_0079 LA_T_1272637 - A01   spoof
    2021      LA_0023 LA_E_9332881 - - -   bonafide -   eval
    5 (2024)  D_0953  D_1000000001 female - - -     bonafide
    5 (2024)  D_0953  D_1000000002 female - - A09   spoof

So the columns are SNIFFED rather than hardcoded: the key column is the one
holding bonafide/spoof, the attack column is the one holding A-tags, the file
column is the one that actually matches files on disk, and the speaker column is
what is left over that looks like an ID. Every one of them can be overridden
with --file-col / --key-col / --attack-col / --speaker-col if a future edition
defeats the sniffer.

Run --self-test to check the sniffer against all of the layouts above without
needing the corpus.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Same set ml/preprocess.py accepts, so anything this script places is readable by
# the next stage. flac first: that is what ASVspoof actually ships.
AUDIO_EXTS = (".flac", ".wav", ".mp3", ".m4a", ".ogg", ".opus")

BONAFIDE = "bonafide"
SPOOF = "spoof"

# Attack tags are A01..A19 in 2019, A01..A32 in ASVspoof5. '-' is what the
# bonafide rows carry in that column.
ATTACK_RE = re.compile(r"^A\d{1,3}$")
PLACEHOLDER = {"-", "", "none", "null"}

# ml/preprocess.py warns above this; warn here too, where it can still be fixed.
COARSE_GROUP = 50


# ------------------------------------------------------------------ protocol

def read_protocol(path: Path) -> list[list[str]]:
    """
    Protocol file -> rows of fields.

    Tab-separated in ASVspoof5, space-separated in 2019/2021. split() with no
    argument handles both and collapses runs of whitespace, which is what the
    2019 files actually contain in places.
    """
    rows = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split("\t") if "\t" in line else line.split()
            rows.append([f.strip() for f in fields])
    return rows


def disk_stems(audio_dir: Path) -> dict[str, Path]:
    """
    Every audio file under audio_dir, indexed by filename stem.

    Recursive, because the corpus is sometimes one flat dir and sometimes
    nested under flac/ or wav/. The stem is the join key with the protocol,
    which never carries an extension.
    """
    found: dict[str, Path] = {}
    for f in audio_dir.rglob("*"):
        if not f.is_file() or f.suffix.lower() not in AUDIO_EXTS:
            continue
        if f.name.startswith("._"):          # AppleDouble sidecar, see ml/preprocess.py
            continue
        found.setdefault(f.stem, f)
    return found


def sniff_columns(rows: list[list[str]], stems: dict[str, Path]) -> dict[str, int]:
    """
    Work out which column is which. Returns {'file','key','attack','speaker'}.

    Each is found by what its VALUES look like rather than by position, because
    position is the thing that changes between editions. The file column is
    resolved against the files actually on disk - that is the only check that
    cannot be fooled by two columns having a similar shape.
    """
    if not rows:
        raise ValueError("protocol file has no usable rows")
    width = min(len(r) for r in rows)
    if width < 2:
        raise ValueError(
            f"protocol rows have only {width} column(s); expected at least a "
            f"filename and a bonafide/spoof key")

    col_values = [[r[c] for r in rows] for c in range(width)]

    def frac(c: int, pred) -> float:
        vals = col_values[c]
        return sum(1 for v in vals if pred(v)) / max(1, len(vals))

    # key: the column that is bonafide/spoof and nothing else.
    key_col = None
    for c in range(width):
        if frac(c, lambda v: v.lower() in (BONAFIDE, SPOOF)) > 0.99:
            key_col = c
            break
    if key_col is None:
        raise ValueError(
            "no column holds only 'bonafide'/'spoof'. Is this a protocol file? "
            "Pass --key-col to name it explicitly.")

    # file: the column whose values match files on disk. Scored, not thresholded
    # - a partial download should still sort what it has, and the report at the
    # end says how many were missing.
    file_col, best = None, 0.0
    for c in range(width):
        if c == key_col:
            continue
        hit = frac(c, lambda v: v in stems)
        if hit > best:
            file_col, best = c, hit
    if file_col is None or best == 0.0:
        # Nothing matched. Fall back to shape so --dry-run can still show the
        # user what the script thinks the columns are.
        for c in range(width):
            if c != key_col and frac(c, lambda v: bool(re.search(r"\d", v))) > 0.9:
                file_col = c
                break
    if file_col is None:
        raise ValueError(
            "could not identify the filename column. Pass --file-col.")

    # attack: A-tags on the spoof rows, placeholder on the bonafide rows.
    attack_col = None
    for c in range(width):
        if c in (key_col, file_col):
            continue
        if frac(c, lambda v: bool(ATTACK_RE.match(v)) or v.lower() in PLACEHOLDER) > 0.99 \
                and any(ATTACK_RE.match(v) for v in col_values[c]):
            attack_col = c
            break

    # speaker: whatever is left that looks like an ID and repeats. Gender
    # columns are excluded by the male/female test; placeholder columns by the
    # '-' test.
    speaker_col = None
    for c in range(width):
        if c in (key_col, file_col, attack_col):
            continue
        vals = col_values[c]
        if frac(c, lambda v: v.lower() in ("male", "female", "m", "f")) > 0.5:
            continue
        if frac(c, lambda v: v.lower() in PLACEHOLDER) > 0.5:
            continue
        # More than one distinct value, but NO upper bound: a dev or eval split
        # legitimately has one row per speaker, and requiring the column to
        # repeat rejected exactly those files. If this does land on a per-file
        # id rather than a speaker id, the result is one group per file - the
        # documented --no-by-speaker fallback, not a broken split.
        if len(set(vals)) > 1 and frac(c, lambda v: bool(re.search(r"\w", v))) > 0.9:
            speaker_col = c
            break

    return {"file": file_col, "key": key_col,
            "attack": attack_col, "speaker": speaker_col}


def sanitise(name: str) -> str:
    """Make a protocol value safe as a directory name."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]", "_", name.strip())
    return cleaned or "unknown"


# --------------------------------------------------------------------- plan

def build_plan(rows, cols, stems, out_root: Path, by_speaker: bool,
               max_per_group: int) -> tuple[list[tuple[Path, Path]], dict]:
    """
    Rows -> a list of (source, destination) moves, plus a report.

    Nothing touches the filesystem here. build the whole plan, show it, and only
    then apply it - so --dry-run and the real run agree by construction rather
    than by two code paths that have to be kept in step.
    """
    plan: list[tuple[Path, Path]] = []
    stats = {"rows": len(rows), "missing": [], "unlabelled": 0,
             "capped": 0, "groups": defaultdict(int), "keys": Counter()}

    fc, kc = cols["file"], cols["key"]
    ac, sc = cols["attack"], cols["speaker"]
    per_group: Counter = Counter()

    for r in rows:
        if len(r) <= max(x for x in (fc, kc, ac, sc) if x is not None):
            stats["unlabelled"] += 1
            continue

        stem = r[fc]
        key = r[kc].lower()
        stats["keys"][key] += 1

        src = stems.get(stem)
        if src is None:
            stats["missing"].append(stem)
            continue

        if key == BONAFIDE:
            # One directory per speaker. Without a speaker column the whole
            # bonafide half would be a single group, which defeats the split -
            # so fall back to one directory per FILE rather than one for all.
            if by_speaker and sc is not None:
                group = f"human/asvspoof/{sanitise(r[sc])}"
            else:
                group = f"human/asvspoof/{sanitise(stem)}"
        else:
            # One directory per attack id. This is what leave-one-converter-out
            # holds out, so an unknown tag becomes its own group rather than
            # being merged into a catch-all.
            tag = r[ac] if ac is not None and len(r) > ac else "unknown"
            if tag in PLACEHOLDER or not ATTACK_RE.match(tag):
                tag = "unknown"
            group = f"ai/asvspoof/{sanitise(tag)}"

        if max_per_group > 0 and per_group[group] >= max_per_group:
            stats["capped"] += 1
            continue
        per_group[group] += 1

        dst = out_root / group / src.name
        plan.append((src, dst))
        stats["groups"][group] += 1

    return plan, stats


def report(cols, stats, plan, out_root: Path, dry_run: bool) -> None:
    names = {v: k for k, v in cols.items() if v is not None}
    print("columns detected:")
    for idx in sorted(names):
        print(f"  [{idx}] {names[idx]}")
    if cols["attack"] is None:
        print("  attack column NOT FOUND - every spoof file groups as "
              "'ai/asvspoof/unknown', which makes leave-one-converter-out "
              "impossible. Pass --attack-col if the protocol has one.")
    if cols["speaker"] is None:
        print("  speaker column NOT FOUND - bonafide files group one per file "
              "instead of one per speaker. Pass --speaker-col if it has one.")

    print(f"\nprotocol rows      {stats['rows']}")
    print(f"  {dict(stats['keys'])}")
    if stats["missing"]:
        n = len(stats["missing"])
        print(f"  {n} row(s) had no matching audio file, e.g. "
              f"{', '.join(stats['missing'][:3])}")
    if stats["unlabelled"]:
        print(f"  {stats['unlabelled']} short row(s) skipped")
    if stats["capped"]:
        print(f"  {stats['capped']} file(s) skipped by --max-per-group")

    groups = stats["groups"]
    human = {g: n for g, n in groups.items() if g.startswith("human/")}
    ai = {g: n for g, n in groups.items() if g.startswith("ai/")}
    print(f"\n{len(plan)} file(s) -> {len(groups)} group(s) under {out_root}/")
    print(f"  human  {len(human):5d} group(s)  {sum(human.values())} file(s)")
    print(f"  ai     {len(ai):5d} group(s)  {sum(ai.values())} file(s)")

    for title, sub in (("ai (one per generator)", ai),
                       ("human (one per speaker)", human)):
        if not sub:
            continue
        print(f"\n  {title}:")
        for g, n in sorted(sub.items(), key=lambda kv: -kv[1])[:12]:
            print(f"    {g:44s} {n:6d}")
        if len(sub) > 12:
            print(f"    ... and {len(sub) - 12} more")

    coarse = {g: n for g, n in groups.items() if n > COARSE_GROUP}
    if coarse:
        print(f"\n  {len(coarse)} group(s) hold more than {COARSE_GROUP} files. "
              f"That is expected for\n  an ASVspoof attack id (~9,000 files each) "
              f"and it is fine - the group is\n  split whole, which is the point. "
              f"Use ml/preprocess.py --max-files-per-label\n  to cut the corpus down "
              f"to a trainable size.")

    if len(ai) < 2:
        print("\n  WARNING: fewer than 2 generator groups. "
              "--holdout-group-prefix needs at least two to hold one out.")
    if len(human) < 2:
        print("\n  WARNING: fewer than 2 human groups. The test set will contain "
              "only one class.")

    if dry_run:
        print("\nDRY RUN - nothing moved. Re-run without --dry-run to apply.")
        for src, dst in plan[:5]:
            print(f"  {src}  ->  {dst}")


def apply_plan(plan: list[tuple[Path, Path]], copy: bool) -> int:
    """Move (or copy) every planned file. Returns the number of failures."""
    op = shutil.copy2 if copy else shutil.move
    verb = "copy" if copy else "move"
    failed = 0
    for i, (src, dst) in enumerate(plan, 1):
        if dst.exists():
            continue                      # already sorted; re-runnable
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            op(str(src), str(dst))
        except Exception as e:
            failed += 1
            if failed <= 5:
                print(f"  !! {verb} failed {src.name}: {type(e).__name__}: {e}")
        if i % 5000 == 0:
            print(f"  {i}/{len(plan)}")
    return failed


# ---------------------------------------------------------------- self-test

def _self_test() -> int:
    """
    Check the sniffer against every protocol layout this script claims to read.

    No corpus needed: the 'files on disk' are faked by handing sniff_columns a
    stem index built from the synthetic rows. If a future edition changes the
    column order, this is where it should be caught.
    """
    cases = {
        "2019 LA": [
            "LA_0079 LA_T_1138215 - - bonafide",
            "LA_0079 LA_T_1272637 - A01 spoof",
            "LA_0080 LA_T_1000001 - A02 spoof",
            "LA_0081 LA_T_1000002 - - bonafide",
        ],
        "2021": [
            "LA_0023 LA_E_9332881 - - - bonafide - eval",
            "LA_0024 LA_E_1111111 - - A07 spoof - eval",
            "LA_0025 LA_E_2222222 - - A08 spoof - eval",
            "LA_0026 LA_E_3333333 - - - bonafide - eval",
        ],
        "5 (2024) tab": [
            "D_0953\tD_1000000001\tfemale\t-\t-\t-\tbonafide",
            "D_0953\tD_1000000002\tfemale\t-\t-\tA09\tspoof",
            "D_0954\tD_1000000003\tmale\t-\t-\tA10\tspoof",
            "D_0955\tD_1000000004\tmale\t-\t-\t-\tbonafide",
        ],
    }

    failures = 0
    for name, lines in cases.items():
        rows = [ln.split("\t") if "\t" in ln else ln.split() for ln in lines]
        # Pretend every referenced file exists, so the file column resolves the
        # same way it would against a real download.
        expected_file_col = 1
        stems = {r[expected_file_col]: Path(f"{r[expected_file_col]}.flac")
                 for r in rows}

        try:
            cols = sniff_columns(rows, stems)
        except Exception as e:
            print(f"  {name:14s} FAIL  sniff raised {type(e).__name__}: {e}")
            failures += 1
            continue

        plan, stats = build_plan(rows, cols, stems, Path("datasets/data/raw"),
                                 by_speaker=True, max_per_group=0)
        groups = set(stats["groups"])
        ai_groups = {g for g in groups if g.startswith("ai/")}
        human_groups = {g for g in groups if g.startswith("human/")}

        ok = (cols["file"] == expected_file_col
              and cols["attack"] is not None
              and cols["speaker"] == 0
              and len(plan) == len(rows)
              and len(ai_groups) == 2
              and len(human_groups) == 2
              and all(re.match(r"^ai/asvspoof/A\d+$", g) for g in ai_groups))

        print(f"  {name:14s} {'ok  ' if ok else 'FAIL'}  cols={cols}  "
              f"ai={sorted(ai_groups)}  human={len(human_groups)}")
        if not ok:
            failures += 1

    # A protocol with no attack column must still sort, just without generator
    # groups - and must SAY so rather than silently merging everything.
    rows = [r.split() for r in ("s1 f1 bonafide", "s2 f2 spoof")]
    stems = {"f1": Path("f1.flac"), "f2": Path("f2.flac")}
    cols = sniff_columns(rows, stems)
    plan, stats = build_plan(rows, cols, stems, Path("datasets/data/raw"), True, 0)
    degraded_ok = (cols["attack"] is None
                   and "ai/asvspoof/unknown" in stats["groups"])
    print(f"  {'no attack col':14s} {'ok  ' if degraded_ok else 'FAIL'}  "
          f"-> {sorted(stats['groups'])}")
    failures += 0 if degraded_ok else 1

    print(f"\n{'all sniffer cases passed' if not failures else f'{failures} case(s) FAILED'}")
    return 1 if failures else 0


# --------------------------------------------------------------------- main

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("protocol", nargs="?",
                    help="ASVspoof protocol/metadata file (.tsv or .txt)")
    ap.add_argument("--audio-dir", default="datasets/data/raw/ai",
                    help="Where the unsorted audio currently sits (searched "
                         "recursively). Default: datasets/data/raw/ai")
    ap.add_argument("--out-root", default="datasets/data/raw",
                    help="Root of the sorted tree. Default: datasets/data/raw")
    ap.add_argument("--dry-run", action="store_true",
                    help="Show the plan and touch nothing")
    ap.add_argument("--copy", action="store_true",
                    help="Copy instead of moving. Needs twice the disk; use it "
                         "when the download is shared or read-only.")
    ap.add_argument("--by-speaker", dest="by_speaker", action="store_true",
                    default=True,
                    help="One bonafide directory per speaker (default)")
    ap.add_argument("--no-by-speaker", dest="by_speaker", action="store_false",
                    help="One bonafide directory per FILE instead of per speaker")
    ap.add_argument("--max-per-group", type=int, default=0, metavar="N",
                    help="Keep at most N files per group (0 = all). An ASVspoof "
                         "attack is ~9,000 files; this trims while keeping every "
                         "generator.")
    ap.add_argument("--file-col", type=int, help="Override the sniffed columns")
    ap.add_argument("--key-col", type=int)
    ap.add_argument("--attack-col", type=int)
    ap.add_argument("--speaker-col", type=int)
    ap.add_argument("--self-test", action="store_true",
                    help="Check the column sniffer against every known protocol "
                         "layout. Needs no corpus.")
    a = ap.parse_args(argv)

    if a.self_test:
        print("sniffer self-test\n")
        return _self_test()

    if not a.protocol:
        ap.error("protocol file is required (or pass --self-test)")

    proto = Path(a.protocol)
    audio_dir = Path(a.audio_dir)
    out_root = Path(a.out_root)

    if not proto.is_file():
        print(f"protocol not found: {proto}")
        return 1
    if not audio_dir.is_dir():
        print(f"audio dir not found: {audio_dir}")
        return 1

    rows = read_protocol(proto)
    if not rows:
        print(f"no usable rows in {proto}")
        return 1

    stems = disk_stems(audio_dir)
    print(f"protocol : {proto}  ({len(rows)} rows)")
    print(f"audio    : {audio_dir}  ({len(stems)} file(s) on disk)\n")
    if not stems:
        print(f"no audio under {audio_dir}/ - nothing to sort.")
        return 1

    try:
        cols = sniff_columns(rows, stems)
    except ValueError as e:
        print(f"could not read the protocol: {e}")
        return 1

    for name, override in (("file", a.file_col), ("key", a.key_col),
                           ("attack", a.attack_col), ("speaker", a.speaker_col)):
        if override is not None:
            cols[name] = override

    plan, stats = build_plan(rows, cols, stems, out_root, a.by_speaker,
                             a.max_per_group)
    report(cols, stats, plan, out_root, a.dry_run)

    if a.dry_run:
        return 0
    if not plan:
        print("\nnothing to do.")
        return 0

    print(f"\n{'copying' if a.copy else 'moving'} {len(plan)} file(s)...")
    failed = apply_plan(plan, a.copy)
    if failed:
        print(f"\n{failed} file(s) failed.")
    print(f"\ndone -> {out_root}/\n\nNext:\n  python ml/preprocess.py\n"
          f"  python ml/run_training.py --holdout-group-prefix ai/asvspoof/A09")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

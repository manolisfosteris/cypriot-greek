"""Prepare Standard Greek rehearsal data from Common Voice Scripted Speech, Greek (el).

Picks a deterministic sample of validated clips, at most --max-per-speaker clips per speaker, converts them to
16 kHz mono WAV and writes
a segments.tsv in the same format as data/segments, so cyasr.train_qwen can mix them into training
(`smg_segments_dir` in the config).

Training clips come from Common Voice's train AND test splits (--train-from): CV Greek 27.0 has only 20 speakers in
train but 323 in test, and CV is never an evaluation set in this project (Standard Greek is scored on FLEURS), so
its test split is free to use; many voices matter more for rehearsal than many clips per voice. Monitoring clips
come from CV dev (50 speakers, disjoint from train and test).

Usage (after downloading and extracting the Common Voice Greek archive from mozilladatacollective.com):
  python scripts/prep_cv_greek.py data/cv_el --out data/segments_cv_el
Then zip the output folder for Colab:  (cd data && zip -rq ../segments_cv_el.zip segments_cv_el)
"""
import argparse
import csv
import random
from collections import Counter
from pathlib import Path

import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
import sys  # noqa: E402
sys.path.insert(0, str(ROOT))
from cyasr.data import SAMPLE_RATE, load_audio  # noqa: E402

COLUMNS = ["id", "audio_id", "speaker", "split", "duration_s", "text"]


def find_split_tsv(cv_dir: Path, name: str) -> Path:  # noqa: D103
    hits = sorted(cv_dir.rglob(name))
    hits = [h for h in hits if (h.parent / "clips").is_dir()]
    if not hits:
        raise SystemExit(f"no {name} next to a clips/ folder under {cv_dir}")
    return hits[0]


def pick(rows: list[dict], n: int, max_per_speaker: int, seed: int) -> list[dict]:
    rows = [r for r in rows if r.get("sentence", "").strip()]
    random.Random(seed).shuffle(rows)
    per, out = Counter(), []
    for r in rows:
        if per[r["client_id"]] < max_per_speaker:
            per[r["client_id"]] += 1
            out.append(r)
            if len(out) == n:
                break
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cv_dir", help="extracted Common Voice Greek folder (searched for train.tsv / dev.tsv + clips/)")
    ap.add_argument("--out", default="data/segments_cv_el")
    ap.add_argument("--n-train", type=int, default=410, help="410 next to 820 Cypriot pieces = 1/3 Standard Greek")
    ap.add_argument("--n-dev", type=int, default=100)
    ap.add_argument("--max-per-speaker", type=int, default=5)
    ap.add_argument("--train-from", default="train,test", help="CV splits pooled for our training clips")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    out = Path(a.out)
    table = []
    for split, n, sources in [("train", a.n_train, a.train_from.split(",")), ("dev", a.n_dev, ["dev"])]:
        rows = []
        for src in sources:
            tsv = find_split_tsv(Path(a.cv_dir), f"{src}.tsv")
            with open(tsv, encoding="utf-8", newline="") as f:
                rows += list(csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE))
        chosen = pick(rows, n, a.max_per_speaker, a.seed)
        (out / split).mkdir(parents=True, exist_ok=True)
        for r in chosen:
            audio = load_audio(tsv.parent / "clips" / r["path"])
            pid = f"cv_el_{Path(r['path']).stem}"
            sf.write(out / split / f"{pid}.wav", audio, SAMPLE_RATE)
            table.append({"id": pid, "audio_id": r["path"], "speaker": r["client_id"], "split": split,
                          "duration_s": round(len(audio) / SAMPLE_RATE, 2), "text": r["sentence"].strip()})
        secs = sum(t["duration_s"] for t in table if t["split"] == split)
        print(f"{split}: {len(chosen)} clips from {len({r['client_id'] for r in chosen})} speakers, "
              f"{secs / 60:.1f} min  (CV splits {sources}, {tsv.parent})")
    with open(out / "segments.tsv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, delimiter="\t", quoting=csv.QUOTE_NONE, escapechar="\\")
        w.writeheader()
        w.writerows(table)
    print("wrote", out / "segments.tsv")


if __name__ == "__main__":
    main()

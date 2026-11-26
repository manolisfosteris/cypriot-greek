"""Load the Common Voice Spontaneous Speech TSV and audio."""
import csv
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

SAMPLE_RATE = 16_000


def load_split(tsv: str | Path, split: str, max_samples: int | None = None) -> list[dict]:
    """Rows of one split (rows with an empty split are never returned).

    With max_samples, picks that many clips evenly spaced over the duration-sorted list,
    which is deterministic and always includes the shortest and longest clip.
    """
    with open(tsv, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        rows = [r for r in reader if r["split"] == split]
    rows.sort(key=lambda r: (int(r["duration_ms"]), r["audio_id"]))
    if max_samples and max_samples < len(rows):
        idx = np.linspace(0, len(rows) - 1, max_samples).round().astype(int)
        rows = [rows[i] for i in idx]
    return rows


def load_audio(path: str | Path) -> np.ndarray:
    """Read an audio file as mono float32 at 16 kHz (no ffmpeg needed)."""
    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        g = gcd(sr, SAMPLE_RATE)
        audio = resample_poly(audio, SAMPLE_RATE // g, sr // g).astype("float32")
    return audio

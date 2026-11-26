"""Cut clips longer than 30 s into Whisper-sized pieces at sentence boundaries.

Whisper trains on windows of at most 30 s, but some clips are longer and have one transcript.
We force-align the transcript to the audio (CTC, MMS aligner) to get a time for every word,
then cut at sentence ends. Pieces keep the exact original spelling; the simplified text is only
used to find word times. Clips of 30 s or less are not aligned and become one piece.

Usage: python -m cyasr.segment --split train [--max-clips 3] [--out-dir data/segments]
The test split is refused: it must never be processed.
"""
import argparse
import csv
import html
import re
import time
import unicodedata
from pathlib import Path
from typing import NamedTuple

import numpy as np
import soundfile as sf
import torch

from cyasr.data import SAMPLE_RATE, load_audio, load_split

ROOT = Path(__file__).resolve().parent.parent
ALIGN_MODEL = "MahmoudAshraf/mms-300m-1130-forced-aligner"
MAX_S = 30.0
MARGIN_S = 0.2  # padding kept around words when silence has to be trimmed
LATIN_TO_GREEK = {"ž": "ζ", "š": "σ"}
SENTENCE_END_RE = re.compile(r"[.?;!…][\"'»”’)\]]*$")
COLUMNS = ["id", "audio_id", "speaker", "split", "start_s", "end_s", "duration_s", "text",
           "align_score", "chars_per_sec"]


class Piece(NamedTuple):
    start_s: float
    end_s: float
    text: str
    score: float  # mean alignment score of its words (log-probability, closer to 0 is better)


def prepare_words(transcript: str) -> tuple[list[str], list[str]]:
    """Original words and simplified aligner words, same length, index-aligned.

    A token that becomes empty after simplifying (a lone "-" or "...") is merged into the
    previous original word, or into the next one if it comes first.
    """
    orig_words, align_words = [], []
    pending = []  # empty tokens waiting for a next word to join (only at the start)
    for word in transcript.split():
        simple = _simplify(word)
        if simple:
            orig_words.append(" ".join(pending + [word]))
            align_words.append(simple)
            pending = []
        elif orig_words:
            orig_words[-1] += " " + word
        else:
            pending.append(word)
    return orig_words, align_words


def _simplify(word: str) -> str:
    """Lowercase letters only: no punctuation, no combining marks, Latin ž/š become ζ/σ."""
    word = unicodedata.normalize("NFC", html.unescape(word)).lower()
    word = "".join(LATIN_TO_GREEK.get(ch, ch) for ch in word).replace("ς", "σ")
    word = unicodedata.normalize("NFD", word)
    word = "".join(ch for ch in word if unicodedata.category(ch)[0] not in "MPSC")
    return unicodedata.normalize("NFC", word)


def load_aligner(device: str | None = None):
    """Load the MMS aligner (float16 on GPU, float32 on CPU). Returns (model, tokenizer)."""
    from ctc_forced_aligner import load_alignment_model

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device == "cuda" else torch.float32
    return load_alignment_model(device, ALIGN_MODEL, dtype=dtype)


def align(audio: np.ndarray, align_words: list[str], aligner) -> list[tuple[float, float, float]]:
    """(start_s, end_s, score) for each word. Times are tight: pauses between words stay out."""
    from ctc_forced_aligner import (generate_emissions, get_alignments, get_spans,
                                    postprocess_results, preprocess_text)

    model, tokenizer = aligner
    waveform = torch.from_numpy(audio).to(model.dtype).to(model.device)
    emissions, stride = generate_emissions(model, waveform)
    # "edges" puts one <star> (any speech) before and after the text, so extra speech is absorbed.
    tokens, text = preprocess_text(" ".join(align_words), romanize=True, language="ell",
                                   star_frequency="edges")
    segments, scores, blank = get_alignments(emissions, tokens, tokenizer)
    spans = get_spans(tokens, segments, blank)
    # get_spans pads each word with half of the neighbouring pauses; drop that blank padding.
    for i, span in enumerate(spans):
        spans[i] = [s for s in span if s.label != blank] or span
    words = postprocess_results(text, spans, stride, scores)
    assert len(words) == len(align_words), "aligner returned a different number of words"
    return [(w["start"], w["end"], w["score"]) for w in words]


def make_pieces(orig_words: list[str], times: list[tuple[float, float, float]],
                max_s: float = MAX_S, clip_s: float | None = None) -> list[Piece]:
    """Group words into pieces of at most max_s, cutting at sentence ends where possible.

    clip_s is the clip length (default: end of the last word). Cut points between pieces lie
    in the middle of the pause between the two words. A sentence that is too long is split at
    its longest pauses.
    """
    n = len(orig_words)
    clip_s = clip_s if clip_s is not None else times[-1][1]

    def bounds(i: int, j: int) -> tuple[float, float]:
        """Start and end time of a piece made of words i..j."""
        start = 0.0 if i == 0 else (times[i - 1][1] + times[i][0]) / 2
        end = clip_s if j == n - 1 else (times[j][1] + times[j + 1][0]) / 2
        if end - start > max_s:  # too much silence: keep the words plus a small margin
            start = max(start, times[i][0] - MARGIN_S)
            end = min(end, times[j][1] + MARGIN_S)
        return start, end

    def fits(i: int, j: int) -> bool:
        start, end = bounds(i, j)
        return end - start <= max_s

    def split_long(i: int, j: int) -> list[tuple[int, int]]:
        """Split words i..j at the longest pause until every part fits."""
        if fits(i, j) or i == j:
            return [(i, j)]
        k = max(range(i, j), key=lambda k: times[k + 1][0] - times[k][1])
        return split_long(i, k) + split_long(k + 1, j)

    units = []  # sentences, or parts of sentences that were too long
    first = 0
    for i, word in enumerate(orig_words):
        if SENTENCE_END_RE.search(word) or i == n - 1:
            units += split_long(first, i)
            first = i + 1

    groups = []  # greedily add whole units while the piece still fits
    i, j = units[0]
    for a, b in units[1:]:
        if fits(i, b):
            j = b
        else:
            groups.append((i, j))
            i, j = a, b
    groups.append((i, j))

    pieces = []
    for i, j in groups:
        start, end = bounds(i, j)
        score = float(np.mean([t[2] for t in times[i:j + 1]]))
        pieces.append(Piece(start, end, " ".join(orig_words[i:j + 1]), score))
    return pieces


def segment_clip(audio: np.ndarray, transcript: str, aligner, max_s: float = MAX_S) -> list[Piece]:
    """Pieces for one clip. Short clips pass through as one piece with no alignment score."""
    duration = len(audio) / SAMPLE_RATE
    if duration <= max_s:
        return [Piece(0.0, duration, " ".join(transcript.split()), float("nan"))]
    orig_words, align_words = prepare_words(transcript)
    times = align(audio, align_words, aligner)
    return make_pieces(orig_words, times, max_s, clip_s=duration)


def pick_rows(rows: list[dict], max_clips: int | None) -> list[dict]:
    """All rows, or with max_clips that many long clips evenly spaced over the duration order."""
    if not max_clips:
        return rows
    long_rows = [r for r in rows if int(r["duration_ms"]) > MAX_S * 1000]
    idx = np.linspace(0, len(long_rows) - 1, min(max_clips, len(long_rows))).round().astype(int)
    return [long_rows[i] for i in idx]


def write_table(path: Path, split: str, new_rows: list[dict]) -> None:
    """Replace the rows of `split` in segments.tsv, keeping the other splits."""
    kept = []
    if path.exists():
        with open(path, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
            kept = [r for r in reader if r["split"] != split]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, COLUMNS, delimiter="\t", quoting=csv.QUOTE_NONE, escapechar="\\")
        writer.writeheader()
        writer.writerows(kept + new_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", required=True, choices=["train", "dev", "test"])
    parser.add_argument("--max-clips", type=int, default=None, help="only this many long clips")
    parser.add_argument("--out-dir", default="data/segments")
    parser.add_argument("--tsv", default="data/ss-corpus-el-CY.tsv")
    parser.add_argument("--audio-dir", default="data/audios")
    args = parser.parse_args()
    if args.split == "test":
        parser.error("the test split must never be processed")

    root = Path(args.out_dir) if Path(args.out_dir).is_absolute() else ROOT / args.out_dir
    wav_dir = root / args.split
    wav_dir.mkdir(parents=True, exist_ok=True)
    rows = pick_rows(load_split(ROOT / args.tsv, args.split), args.max_clips)
    aligner = None  # loaded only if a long clip needs it

    out_rows, hours_in, aligned_s = [], 0.0, 0.0
    t0 = time.time()
    for n, row in enumerate(rows, 1):
        audio = load_audio(ROOT / args.audio_dir / row["audio_file"])
        hours_in += len(audio) / SAMPLE_RATE / 3600
        if len(audio) / SAMPLE_RATE > MAX_S and aligner is None:
            aligner = load_aligner()
            t0 = time.time()  # do not count model loading in the speed figure
        if len(audio) / SAMPLE_RATE > MAX_S:
            aligned_s += len(audio) / SAMPLE_RATE
        pieces = segment_clip(audio, row["transcription"], aligner)
        for k, p in enumerate(pieces):
            piece_id = f"{row['audio_id']}_{k}"
            sf.write(wav_dir / f"{piece_id}.wav", audio[round(p.start_s * SAMPLE_RATE):
                                                         round(p.end_s * SAMPLE_RATE)],
                     SAMPLE_RATE, subtype="PCM_16")
            duration = p.end_s - p.start_s
            out_rows.append(dict(
                id=piece_id, audio_id=row["audio_id"], speaker=row["client_id"], split=args.split,
                start_s=f"{p.start_s:.3f}", end_s=f"{p.end_s:.3f}", duration_s=f"{duration:.3f}",
                text=p.text, align_score="" if np.isnan(p.score) else f"{p.score:.4f}",
                chars_per_sec=f"{len(p.text) / duration:.2f}"))
        if n % 20 == 0:
            print(f"{n}/{len(rows)} clips", flush=True)
    elapsed = time.time() - t0

    write_table(root / "segments.tsv", args.split, out_rows)
    hours_out = sum(float(r["duration_s"]) for r in out_rows) / 3600
    odd = [r for r in out_rows if not 3 <= float(r["chars_per_sec"]) <= 30]
    scored = sorted((r for r in out_rows if r["align_score"]), key=lambda r: float(r["align_score"]))
    print(f"{args.split}: {len(rows)} clips -> {len(out_rows)} pieces, "
          f"{hours_in:.2f} h in, {hours_out:.2f} h out")
    print(f"pieces with chars_per_sec < 3 or > 30: {len(odd)}")
    if aligned_s:
        print(f"aligned {aligned_s:.0f} s of audio in {elapsed:.0f} s ({aligned_s / elapsed:.2f}x real time)")
    for r in scored[:5]:
        print(f"low score {r['align_score']}  {r['id']}  {r['text'][:60]}")


if __name__ == "__main__":
    main()

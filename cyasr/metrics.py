"""Corpus-level WER/CER at every normalization level, with breakdowns."""
from collections import defaultdict

import jiwer

from cyasr.normalize import LEVELS, normalize

DURATION_BUCKETS = [("<10s", 0, 10), ("10-30s", 10, 30), (">30s", 30, float("inf"))]
HALLUCINATION_RATIO = 2.0


def error_counts(ref: str, hyp: str, unit: str = "word") -> tuple[int, int]:
    """Return (substitutions + deletions + insertions, reference length) for one pair."""
    if unit == "word":
        ref_len, hyp_len = len(ref.split()), len(hyp.split())
    else:
        ref_len, hyp_len = len(ref), len(hyp)
    if ref_len == 0 or hyp_len == 0:  # jiwer rejects empty strings
        return max(ref_len, hyp_len), ref_len
    process = jiwer.process_words if unit == "word" else jiwer.process_characters
    out = process(ref, hyp)
    return out.substitutions + out.deletions + out.insertions, ref_len


def corpus_rate(refs: list[str], hyps: list[str], unit: str = "word") -> dict:
    """Total errors divided by total reference length (not a mean of per-utterance rates)."""
    errors = total = 0
    for ref, hyp in zip(refs, hyps):
        e, n = error_counts(ref, hyp, unit)
        errors += e
        total += n
    return {"rate": errors / total if total else float("nan"), "errors": errors, "ref_len": total}


def score(records: list[dict]) -> dict:
    """WER and CER for every level over records with `ref` and `hyp` keys."""
    result = {}
    for level in LEVELS:
        refs = [normalize(r["ref"], level) for r in records]
        hyps = [normalize(r["hyp"], level) for r in records]
        w, c = corpus_rate(refs, hyps, "word"), corpus_rate(refs, hyps, "char")
        result[level] = {"wer": w["rate"], "cer": c["rate"],
                         "word_errors": w["errors"], "ref_words": w["ref_len"],
                         "char_errors": c["errors"], "ref_chars": c["ref_len"]}
    return result


def compute_metrics(records: list[dict]) -> dict:
    """Full metrics.json content for a list of prediction records."""
    by_speaker, by_bucket = defaultdict(list), defaultdict(list)
    for r in records:
        by_speaker[r["speaker"]].append(r)
        for name, lo, hi in DURATION_BUCKETS:
            if lo <= r["duration_s"] < hi:
                by_bucket[name].append(r)
    hallucinations = 0
    for r in records:
        ref_words = len(normalize(r["ref"], "nofill").split())
        hyp_words = len(normalize(r["hyp"], "nofill").split())
        hallucinations += ref_words > 0 and hyp_words / ref_words > HALLUCINATION_RATIO
    return {
        "n_utterances": len(records),
        "hours": sum(r["duration_s"] for r in records) / 3600,
        "empty_hypotheses": sum(not normalize(r["hyp"], "raw") for r in records),
        "hallucinations": int(hallucinations),
        "levels": score(records),
        "per_speaker": {k: {"n": len(v), "levels": score(v)} for k, v in sorted(by_speaker.items())},
        "by_duration": {name: {"n": len(by_bucket[name]), "levels": score(by_bucket[name])}
                        for name, _, _ in DURATION_BUCKETS if by_bucket[name]},
    }

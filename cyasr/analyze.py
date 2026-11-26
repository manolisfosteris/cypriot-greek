"""Error analysis for one run: top word substitutions, worst utterances, markdown summary.

Usage: python -m cyasr.analyze runs/<run_name> [--top 25]
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import jiwer

from cyasr.metrics import error_counts
from cyasr.normalize import normalize

LEVEL = "nodiac"


def load_predictions(run_dir: Path) -> list[dict]:
    with open(run_dir / "predictions.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def substitution_pairs(records: list[dict]) -> Counter:
    """Count (reference word, hypothesis word) substitutions at the nodiac level."""
    pairs: Counter = Counter()
    for r in records:
        ref, hyp = normalize(r["ref"], LEVEL), normalize(r["hyp"], LEVEL)
        if not ref or not hyp:
            continue
        out = jiwer.process_words(ref, hyp)
        ref_w, hyp_w = ref.split(), hyp.split()
        for chunk in out.alignments[0]:
            if chunk.type == "substitute":
                for k in range(chunk.ref_end_idx - chunk.ref_start_idx):
                    pairs[(ref_w[chunk.ref_start_idx + k], hyp_w[chunk.hyp_start_idx + k])] += 1
    return pairs


def worst_utterances(records: list[dict], n: int, min_words: int = 5) -> list[dict]:
    """Utterances with the highest nodiac WER (ignoring very short references)."""
    scored = []
    for r in records:
        ref, hyp = normalize(r["ref"], LEVEL), normalize(r["hyp"], LEVEL)
        errors, ref_len = error_counts(ref, hyp)
        if ref_len >= min_words:
            scored.append({**r, "wer": errors / ref_len, "ref_n": ref, "hyp_n": hyp})
    return sorted(scored, key=lambda r: -r["wer"])[:n]


def clip(text: str, limit: int = 400) -> str:
    return text if len(text) <= limit else text[:limit] + " ..."


def build_report(run_dir: Path, top: int) -> str:
    records = load_predictions(run_dir)
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    lines = [f"# Error analysis: {run_dir.name}", "",
             f"Model `{metrics['model']}`, split `{metrics['split']}`, {metrics['n_utterances']} utterances, "
             f"{metrics['empty_hypotheses']} empty hypotheses, {metrics['hallucinations']} flagged as hallucination.", "",
             "## WER / CER by normalization level", "", "| level | WER | CER |", "|---|---|---|"]
    for level, m in metrics["levels"].items():
        lines.append(f"| {level} | {m['wer']:.1%} | {m['cer']:.1%} |")
    lines += ["", f"## Top {top} word substitutions (reference -> hypothesis, level {LEVEL})", "",
              "| count | reference | hypothesis |", "|---|---|---|"]
    for (ref, hyp), count in substitution_pairs(records).most_common(top):
        lines.append(f"| {count} | {ref} | {hyp} |")
    lines += ["", f"## Worst {min(top, 10)} utterances by WER (level {LEVEL})", ""]
    for r in worst_utterances(records, min(top, 10)):
        lines += [f"- `{r['audio_id']}` ({r['duration_s']:.0f} s, WER {r['wer']:.0%})",
                  f"  - ref: {clip(r['ref_n'])}", f"  - hyp: {clip(r['hyp_n'])}"]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--top", type=int, default=25)
    args = parser.parse_args()
    report = build_report(args.run_dir, args.top)
    (args.run_dir / "analysis.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()

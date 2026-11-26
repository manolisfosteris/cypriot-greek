"""Print (and save) a markdown table of WER/CER across all runs in runs/.

Usage: python scripts/summarize_runs.py [runs_dir]
"""
import json
import sys
from pathlib import Path

LEVELS = ["raw", "basic", "nofill", "nodiac", "phonetic"]


def main() -> None:
    runs_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "runs")
    header = ["run", "model", "split", "n"] + [f"WER {l}" for l in LEVELS] + ["CER raw", "CER nodiac"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for path in sorted(runs_dir.glob("*/metrics.json")):
        m = json.loads(path.read_text(encoding="utf-8"))
        lv = m["levels"]
        cells = [path.parent.name, m["model"], m["split"], str(m["n_utterances"])]
        cells += [f"{lv[l]['wer']:.1%}" for l in LEVELS]
        cells += [f"{lv['raw']['cer']:.1%}", f"{lv['nodiac']['cer']:.1%}"]
        lines.append("| " + " | ".join(cells) + " |")
    table = "\n".join(lines) + "\n"
    (runs_dir / "summary.md").write_text(table, encoding="utf-8")
    print(table)


if __name__ == "__main__":
    main()

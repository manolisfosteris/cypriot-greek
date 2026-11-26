"""Score a Qwen3-ASR / Rumble checkpoint on cached FLEURS clips (Standard Greek forgetting check).

Same decoding and scoring as notebook 09 part B and scripts/wiseft_sweep.py: Rumble's options
{digits: false, filler: true, verbatim: true}, greedy, batch 1, all normalization levels, plus counts of
Cypriot forms in the output. `--prompt` sets the system-context tag (e.g. "Cypriot Greek"); default none.
Skips the run if <out>/<name>/metrics.json exists.

Usage:
  python scripts/eval_fleurs.py --model /content/drive/MyDrive/cyasr/runs/finetune_rumble-2-flash_mix_tag/best \
      --name mix_tag_notag --fleurs /content/fleurs_el_gr_200.pkl --out /content/drive/MyDrive/cyasr/runs_mix
"""
import argparse
import json
import pickle
import sys
import time
from collections import Counter
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from cyasr.metrics import score  # noqa: E402
from cyasr.normalize import normalize  # noqa: E402
from cyasr.transcribe_qwen import QwenTranscriber  # noqa: E402

RUMBLE_KW = {"digits": False, "filler": True, "verbatim": True}
MARKERS = ["τζαι", "τζ", "εν", "εννα", "ουλλοι", "ουλλα", "τουτο", "τουτη", "ποττε", "και", "ειναι"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--name", required=True)
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--fleurs", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d = Path(a.out) / a.name
    if (d / "metrics.json").exists():
        print(a.name, "already done")
        return
    torch.manual_seed(42)
    clips = pickle.load(open(a.fleurs, "rb"))
    t = QwenTranscriber(a.model, a.revision, "el", 1, 1024, "auto", RUMBLE_KW, prompt=a.prompt)
    t0 = time.time()
    outs = t.transcribe_detailed([c["audio"] for c in clips])
    records = [{"audio_id": c["audio_id"], "utt_id": c["utt_id"], "duration_s": c["duration_s"],
                "ref": c["ref"], **o} for c, o in zip(clips, outs)]
    c = Counter(w for r in records for w in normalize(r["hyp"], "nodiac").split())
    m = {"n_utterances": len(records), "levels": score(records), "markers": {k: c[k] for k in MARKERS},
         "model": a.model, "revision": a.revision, "prompt": a.prompt, "seconds": time.time() - t0,
         "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "dtype": str(t.dtype)}
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "predictions.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (d / "metrics.json").write_text(json.dumps(m, ensure_ascii=False, indent=1), encoding="utf-8")
    lv = m["levels"]
    print(f"{a.name}: FLEURS WER basic {lv['basic']['wer']:.3f} nodiac {lv['nodiac']['wer']:.3f} "
          f"CER {lv['basic']['cer']:.3f}  τζαι {m['markers']['τζαι']} εν {m['markers']['εν']} "
          f"({m['seconds']/60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()

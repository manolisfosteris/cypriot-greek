"""Weight blending (WiSE-FT) between zero-shot Rumble-2-Flash and our merged Cypriot LoRA fine-tune.

For each lambda, the model weights are W0 + lambda * (W1 - W0): lambda 0 = original Rumble, 1 = our fine-tune.
Each blend is scored on the Cypriot dev set (280 full-length clips, same settings as eval_rumble-2-flash_lora.yaml)
and on FLEURS el_gr test clips (Standard Greek, cached by notebook 10), with the same prompt options for both.

One process can run several lambdas; run several processes in parallel to fill a large GPU
(batch-1 greedy decoding uses only a small part of it). Finished (lambda, dataset) pairs are skipped.

Usage:
  python scripts/wiseft_sweep.py --ft /content/drive/MyDrive/cyasr/runs/finetune_rumble-2-flash_lora/best \
      --fleurs /content/fleurs_el_gr_200.pkl --out /content/drive/MyDrive/cyasr/runs_wiseft --lams 0 0.5
"""
import argparse
import json
import os
import pickle
import sys
import time
from collections import Counter
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from cyasr.data import load_audio, load_split  # noqa: E402
from cyasr.metrics import compute_metrics, score  # noqa: E402
from cyasr.normalize import normalize  # noqa: E402
from cyasr.transcribe_qwen import QwenTranscriber  # noqa: E402

RUMBLE_KW = {"digits": False, "filler": True, "verbatim": True}  # the options the LoRA was trained with
MARKERS = ["τζαι", "τζ", "εν", "εννα", "ουλλοι", "ουλλα", "τουτο", "τουτη", "ποττε", "και", "ειναι"]


def blend_setup(t: QwenTranscriber, base: str, revision: str, ft: str):
    """Return (W0, W1) for the tensors that differ, read straight from the two safetensors files in their
    stored dtype (bf16 for both checkpoints, so nothing is rounded). ~1.6 GB of CPU RAM instead of ~7 GB with
    full float32 state dicts, so two processes fit on a 12 GB Colab runtime."""
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    p0 = hf_hub_download(base, "model.safetensors", revision=revision)
    p1 = str(Path(ft) / "model.safetensors")
    model_keys = set(t.model.state_dict())
    w0, w1 = {}, {}
    with safe_open(p0, "pt") as f0, safe_open(p1, "pt") as f1:
        if set(f0.keys()) != set(f1.keys()):
            raise ValueError("key mismatch between base and fine-tuned checkpoint files")
        for k in f0.keys():
            a, b = f0.get_tensor(k), f1.get_tensor(k)
            if a.shape != b.shape:
                raise ValueError(f"shape mismatch {k}: {a.shape} vs {b.shape}")
            if not torch.equal(a, b):
                if k not in model_keys:
                    raise ValueError(f"changed tensor {k} is not a model parameter")
                w0[k], w1[k] = a, b
    groups = Counter(k.split(".")[1] if k.startswith("model.") else k.split(".")[0] for k in w0)
    n = sum(v.numel() for v in w0.values())
    print(f"{len(w0)} tensors differ ({n/1e6:.0f}M values), by top module: {dict(groups)}", flush=True)
    return w0, w1


@torch.no_grad()
def set_lambda(t: QwenTranscriber, w0: dict, w1: dict, lam: float) -> None:
    """W = W0 + lam * (W1 - W0), computed in float32 one tensor at a time, then cast to the model dtype."""
    params = dict(t.model.named_parameters()) | dict(t.model.named_buffers())
    for k in w0:
        a = w0[k].float()
        params[k].copy_((a + lam * (w1[k].float() - a)).to(params[k].dtype))


def run_cypriot(t, out: Path, tsv: Path, audio_dir: Path) -> dict:
    rows = load_split(tsv, "dev")
    t.max_new_tokens = 4096  # as in eval_rumble-2-flash_lora.yaml
    outs = []
    for i, r in enumerate(rows):
        outs += t.transcribe_detailed([load_audio(audio_dir / r["audio_file"])])
        if (i + 1) % 40 == 0:
            print(f"  cy {i+1}/{len(rows)}", flush=True)
    records = [{"audio_id": r["audio_id"], "speaker": r["client_id"], "duration_s": int(r["duration_ms"]) / 1000,
                "ref": r["transcription"], **o} for r, o in zip(rows, outs)]
    return records, compute_metrics(records)


def run_fleurs(t, clips: list[dict]) -> dict:
    t.max_new_tokens = 1024  # as in notebook 09 part B
    outs = t.transcribe_detailed([c["audio"] for c in clips])
    records = [{"audio_id": c["audio_id"], "utt_id": c["utt_id"], "speaker": "fleurs", "duration_s": c["duration_s"],
                "ref": c["ref"], **o} for c, o in zip(clips, outs)]
    m = {"n_utterances": len(records), "levels": score(records)}
    c = Counter(w for r in records for w in normalize(r["hyp"], "nodiac").split())
    m["markers"] = {k: c[k] for k in MARKERS}
    return records, m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Reson8/Rumble-2-Flash")
    ap.add_argument("--revision", default="95bd5f9133a2398e5e48a98c089fe48e122bde97")
    ap.add_argument("--ft", required=True, help="merged fine-tuned checkpoint dir")
    ap.add_argument("--fleurs", required=True, help="pickle of FLEURS clips (list of dicts with audio, ref, ...)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--lams", type=float, nargs="+", required=True)
    ap.add_argument("--tsv", default=str(ROOT / "data/ss-corpus-el-CY.tsv"))
    ap.add_argument("--audio-dir", default=str(ROOT / "data/audios"))
    a = ap.parse_args()
    torch.manual_seed(42)

    out = Path(a.out)
    todo = [(lam, ds) for lam in a.lams for ds in ("fleurs_el_gr", "cy_dev")
            if not (out / f"lam{lam:.2f}_{ds}" / "metrics.json").exists()]
    if not todo:
        print("all done")
        return
    t = QwenTranscriber(a.ft, None, "el", 1, 4096, "auto", RUMBLE_KW)  # processor + chat template of the fine-tune
    w0, w1 = blend_setup(t, a.base, a.revision, a.ft)
    clips = pickle.load(open(a.fleurs, "rb"))
    for lam, ds in todo:
        set_lambda(t, w0, w1, lam)
        t0 = time.time()
        if ds == "cy_dev":
            records, m = run_cypriot(t, out, Path(a.tsv), Path(a.audio_dir))
        else:
            records, m = run_fleurs(t, clips)
        m.update(lam=lam, dataset=ds, seconds=time.time() - t0, gpu=torch.cuda.get_device_name(0),
                 dtype=str(t.dtype), base=f"{a.base}@{a.revision}", ft=a.ft)
        d = out / f"lam{lam:.2f}_{ds}"
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "predictions.jsonl", "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        (d / "metrics.json").write_text(json.dumps(m, ensure_ascii=False, indent=1), encoding="utf-8")
        lv = m["levels"]
        print(f"lambda {lam:.2f} {ds}: WER basic {lv['basic']['wer']:.3f} nodiac {lv['nodiac']['wer']:.3f} "
              f"CER nodiac {lv['nodiac']['cer']:.3f} ({m['seconds']/60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()

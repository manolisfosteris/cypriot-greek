"""Run zero-shot ASR (Whisper, or Qwen3-ASR with `backend: qwen3_asr`) on one or more splits and write
a run directory.

Usage: python -m cyasr.run_baseline --config configs/baseline_whisper-small.yaml
"""
import argparse
import json
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import transformers
import yaml

from cyasr.data import load_audio, load_split
from cyasr.metrics import compute_metrics
from cyasr.transcribe import Transcriber

ROOT = Path(__file__).resolve().parent.parent
DEFAULTS = dict(backend="whisper", revision=None, max_new_tokens=4096, dtype="auto", template_kwargs=None, prompt=None, language="el", task="transcribe", num_beams=1, long_form="sequential",
                decoding="greedy", no_speech_threshold=None, batch_size=4, seed=42, max_samples=None, output_dir="runs", mlflow=False,
                tsv="data/ss-corpus-el-CY.tsv", audio_dir="data/audios")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def git_hash() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                             text=True, check=True)
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def environment() -> dict:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "git_hash": git_hash(),
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def make_transcriber(cfg: dict):
    if cfg["backend"] == "whisper":
        return Transcriber(cfg["model"], cfg["language"], cfg["task"], cfg["num_beams"],
                           cfg["long_form"], cfg["batch_size"], cfg["decoding"],
                           cfg["no_speech_threshold"])
    if cfg["backend"] == "qwen3_asr":
        if cfg["decoding"] != "greedy" or cfg["no_speech_threshold"] is not None or cfg["num_beams"] != 1:
            raise ValueError("qwen3_asr supports greedy decoding only (no fallback, no-speech rule or beams)")
        from cyasr.transcribe_qwen import QwenTranscriber  # imported here: Whisper runs don't need it
        return QwenTranscriber(cfg["model"], cfg["revision"], cfg["language"], cfg["batch_size"],
                               cfg["max_new_tokens"], cfg["dtype"], cfg["template_kwargs"], cfg["prompt"])
    raise ValueError(f"unknown backend {cfg['backend']!r}")


def run_split(cfg: dict, split: str, transcriber) -> Path:
    """Transcribe one split, write all run files, and return the run directory."""
    name = f"{cfg['run_name']}_{cfg['decoding']}"
    if cfg["no_speech_threshold"] is not None:
        name += f"_ns{cfg['no_speech_threshold']}"  # e.g. whisper-small_lora_fallback_ns0.6_dev
    name += f"_{split}"
    if cfg["max_samples"]:
        name += f"_n{cfg['max_samples']}"  # smoke runs never overwrite full runs
    run_dir = ROOT / cfg["output_dir"] / name
    run_dir.mkdir(parents=True, exist_ok=True)
    rows = load_split(ROOT / cfg["tsv"], split, cfg["max_samples"])
    audios = [load_audio(ROOT / cfg["audio_dir"] / r["audio_file"]) for r in rows]

    # Whisper returns text only; Qwen also returns extra fields per clip (raw output, detected language,
    # token count), which are saved in predictions.jsonl next to hyp.
    detailed = getattr(transcriber, "transcribe_detailed", None)
    outs: list[dict] = []
    for i in range(0, len(audios), 16):  # progress every 16 clips
        chunk = audios[i:i + 16]
        outs += detailed(chunk) if detailed else [{"hyp": h} for h in transcriber.transcribe(chunk)]
        print(f"[{split}] {len(outs)}/{len(audios)}", flush=True)

    records = [{"audio_id": r["audio_id"], "speaker": r["client_id"],
                "duration_s": int(r["duration_ms"]) / 1000,
                "ref": r["transcription"], **o} for r, o in zip(rows, outs)]
    with open(run_dir / "predictions.jsonl", "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    metrics = compute_metrics(records)
    metrics.update(model=cfg["model"], split=split)
    if detailed:
        metrics["hit_max_tokens"] = sum(o.get("hit_max_tokens", False) for o in outs)
    (run_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "config.yaml").write_text(yaml.safe_dump({**cfg, "split": split}, allow_unicode=True), encoding="utf-8")
    (run_dir / "env.json").write_text(json.dumps(environment(), indent=2), encoding="utf-8")
    if cfg["mlflow"]:
        log_mlflow(cfg, split, metrics, run_dir)
    lv = metrics["levels"]
    print(f"[{split}] WER raw {lv['raw']['wer']:.3f} nodiac {lv['nodiac']['wer']:.3f} | "
          f"CER raw {lv['raw']['cer']:.3f} nodiac {lv['nodiac']['cer']:.3f} -> {run_dir}")
    return run_dir


def log_mlflow(cfg: dict, split: str, metrics: dict, run_dir: Path) -> None:
    import mlflow
    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
    mlflow.set_experiment("cypriot-asr-baseline")
    with mlflow.start_run(run_name=run_dir.name):
        mlflow.log_params({**{k: v for k, v in cfg.items() if k != "splits"}, "split": split})
        for level, m in metrics["levels"].items():
            mlflow.log_metrics({f"wer_{level}": m["wer"], f"cer_{level}": m["cer"]})
        mlflow.log_metrics({"empty_hypotheses": metrics["empty_hypotheses"],
                            "hallucinations": metrics["hallucinations"]})
        mlflow.log_artifacts(str(run_dir))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-samples", type=int, help="override config (smoke tests)")
    parser.add_argument("--decoding", choices=["greedy", "fallback"], help="override config")
    parser.add_argument("--no-speech-threshold", type=float, help="override config (fallback only)")
    parser.add_argument("--model", help="override config: local checkpoint folder or Hub id")
    parser.add_argument("--splits", nargs="+", choices=["dev", "test"], help="override config")
    args = parser.parse_args()
    cfg = {**DEFAULTS, **yaml.safe_load(open(args.config, encoding="utf-8"))}
    if args.max_samples:
        cfg["max_samples"] = args.max_samples
    if args.decoding:
        cfg["decoding"] = args.decoding
    if args.no_speech_threshold is not None:
        cfg["no_speech_threshold"] = args.no_speech_threshold
    if args.model:
        cfg["model"] = args.model
    if args.splits:
        cfg["splits"] = args.splits
    set_seed(cfg["seed"])
    transcriber = make_transcriber(cfg)
    for split in cfg["splits"]:
        run_split(cfg, split, transcriber)


if __name__ == "__main__":
    sys.exit(main())

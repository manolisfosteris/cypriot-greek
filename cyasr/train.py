"""Fine-tune Whisper on the Cypriot Greek segments made by cyasr.segment.
With `backend: qwen3_asr` in the config, Qwen3-ASR / Rumble-2-Flash are trained instead (cyasr/train_qwen.py).

Usage: python -m cyasr.train --config configs/finetune_whisper-small.yaml
Writes runs/finetune_<run_name>/ with best/ (plain Whisper model + processor, loadable by
run_baseline as `model:`), config.yaml, env.json and train_log.json.

Label format (the decoder start token is added by the model when it shifts the labels):
    with timestamps:     <|el|><|transcribe|><|0.00|> text<|T|><|endoftext|>
    without timestamps:  <|el|><|transcribe|><|notimestamps|> text<|endoftext|>
"""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import yaml
from torch.utils.data import Dataset
from transformers import (EarlyStoppingCallback, Seq2SeqTrainer, Seq2SeqTrainingArguments,
                          WhisperForConditionalGeneration, WhisperProcessor, set_seed)
from transformers.trainer_utils import get_last_checkpoint

from cyasr.metrics import corpus_rate
from cyasr.normalize import clean_target, normalize
from cyasr.run_baseline import environment

ROOT = Path(__file__).resolve().parent.parent
MAX_LABELS = 448  # Whisper's max_target_positions
DEFAULTS = dict(backend="whisper", revision=None, language="el", template_kwargs=None,
                max_new_tokens=512,  # qwen3_asr: generation limit for dev WER during training (pieces are <= 30 s)
                output_dir="runs", segments_dir="data/segments", mode="full",
                learning_rate=1e-5, warmup_steps=50, max_steps=1000, batch_size=16, grad_accum=1,
                eval_batch_size=32,  # generation keeps no activations for backprop, so more fits
                resume=False,  # true: continue from the latest checkpoint in the run folder
                eval_steps=100, early_stopping_patience=3, seed=42, timestamp_fraction=0.5,
                spec_augment=False, mask_time_prob=0.05, mask_feature_prob=0.05,
                gradient_checkpointing=False, lora_r=32, lora_alpha=64, lora_dropout=0.05,
                num_workers=2, max_train_samples=None, max_eval_samples=None,
                eval_on_train=False,  # eval_on_train: smoke tests only, when there are no dev rows
                # Qwen3-ASR only: dialect tag in the system prompt, Standard Greek rehearsal, cheaper dev checks
                dialect_prompt=None, smg_segments_dir=None, smg_max_samples=None, smg_dev_samples=None,
                eval_subset=None)


def read_rows(segments_dir: Path, split: str, limit: int | None) -> list[dict]:
    with open(segments_dir / "segments.tsv", encoding="utf-8", newline="") as f:
        rows = [r for r in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE) if r["split"] == split]
    return rows[:limit] if limit else rows


def use_timestamps(piece_id: str, seed: int, fraction: float) -> bool:
    """Deterministic per piece: the same pieces get timestamps in every epoch and every run."""
    h = int(hashlib.md5(f"{seed}:{piece_id}".encode()).hexdigest(), 16)
    return (h % 10_000) / 10_000 < fraction


class SegmentDataset(Dataset):
    """One split of segments: log-mel features plus label token ids."""

    def __init__(self, segments_dir: Path, split: str, processor: WhisperProcessor, seed: int,
                 timestamp_fraction: float, limit: int | None = None):
        self.dir, self.split, self.processor = segments_dir, split, processor
        tok = processor.tokenizer
        self.ids = {t: tok.convert_tokens_to_ids(t) for t in
                    ["<|startoftranscript|>", "<|el|>", "<|transcribe|>", "<|notimestamps|>"]}
        self.eot = tok.eos_token_id
        self.rows, self.labels = [], []
        all_rows = read_rows(segments_dir, split, limit)
        n_rows = len(all_rows)
        for row in all_rows:
            labels = self.build_labels(row, seed, timestamp_fraction)
            if len(labels) <= MAX_LABELS:
                self.rows.append(row)
                self.labels.append(labels)
        print(f"[{split}] {len(self.rows)} pieces, {n_rows - len(self.rows)} dropped (labels > {MAX_LABELS} tokens)")

    def token_id(self, token: str) -> int:
        """Id of a special token; fails loudly if the tokenizer would treat it as plain text."""
        i = self.processor.tokenizer.convert_tokens_to_ids(token)
        assert i not in (None, self.processor.tokenizer.unk_token_id), f"{token} is not a special token"
        return i

    def build_labels(self, row: dict, seed: int, fraction: float) -> list[int]:
        ids, tok = self.ids, self.processor.tokenizer
        text = " " + clean_target(row["text"])  # leading space, as Whisper itself writes its text
        text_ids = tok.encode(text, add_special_tokens=False)
        prefix = [ids["<|startoftranscript|>"], ids["<|el|>"], ids["<|transcribe|>"]]
        if use_timestamps(row["id"], seed, fraction):
            end = min(round(float(row["duration_s"]) / 0.02) * 0.02, 30.0)
            return (prefix + [self.token_id("<|0.00|>")] + text_ids
                    + [self.token_id(f"<|{end:.2f}|>"), self.eot])
        return prefix + [ids["<|notimestamps|>"]] + text_ids + [self.eot]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        row = self.rows[i]
        audio, _ = sf.read(self.dir / self.split / f"{row['id']}.wav", dtype="float32")
        feats = self.processor.feature_extractor(audio, sampling_rate=16_000).input_features[0]
        return {"input_features": feats, "labels": self.labels[i]}


class Collator:
    """Stacks features and pads labels with -100 (ignored by the loss)."""

    def __init__(self, start_token_id: int):
        self.start = start_token_id

    def __call__(self, batch: list[dict]) -> dict:
        feats = torch.from_numpy(np.stack([b["input_features"] for b in batch]))
        labels = [b["labels"] for b in batch]
        # The model shifts labels right and puts the start token in front itself,
        # so a start token already at the front would be duplicated.
        labels = [l[1:] if l[0] == self.start else l for l in labels]
        width = max(len(l) for l in labels)
        padded = torch.tensor([l + [-100] * (width - len(l)) for l in labels])
        return {"input_features": feats, "labels": padded}


def make_metrics(processor: WhisperProcessor):
    tok = processor.tokenizer

    def compute_metrics(pred) -> dict:
        pred_ids = pred.predictions.copy()
        label_ids = pred.label_ids.copy()
        pred_ids[pred_ids == -100] = tok.pad_token_id  # -100 also pads predictions when batches are joined
        label_ids[label_ids == -100] = tok.pad_token_id
        hyps = tok.batch_decode(pred_ids, skip_special_tokens=True)
        refs = tok.batch_decode(label_ids, skip_special_tokens=True)
        out = {}
        for level in ["basic", "nofill", "nodiac"]:
            r, h = [normalize(x, level) for x in refs], [normalize(x, level) for x in hyps]
            out[f"wer_{level}"] = corpus_rate(r, h, "word")["rate"]
        out["cer_nodiac"] = corpus_rate([normalize(x, "nodiac") for x in refs],
                                        [normalize(x, "nodiac") for x in hyps], "char")["rate"]
        return out

    return compute_metrics


def count_params(model: torch.nn.Module, label: str) -> None:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[{label}] trainable parameters: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")


def build_model(cfg: dict) -> WhisperForConditionalGeneration:
    model = WhisperForConditionalGeneration.from_pretrained(cfg["model"])
    # Generation settings used for eval and saved with the model (greedy Greek transcription).
    model.generation_config.language = "el"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None
    model.generation_config.num_beams = 1
    model.generation_config.do_sample = False
    model.config.forced_decoder_ids = None
    if cfg["spec_augment"]:
        model.config.apply_spec_augment = True
        model.config.mask_time_prob = cfg["mask_time_prob"]
        model.config.mask_feature_prob = cfg["mask_feature_prob"]
    if cfg["gradient_checkpointing"]:
        model.config.use_cache = False

    if cfg["mode"] == "frozen_encoder":
        for p in model.model.encoder.parameters():
            p.requires_grad = False
    elif cfg["mode"] == "lora":
        from peft import LoraConfig, get_peft_model  # only needed for this mode
        lora = LoraConfig(r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"], lora_dropout=cfg["lora_dropout"],
                          target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"])
        model = get_peft_model(model, lora)
    elif cfg["mode"] != "full":
        raise ValueError("mode must be full, frozen_encoder or lora")
    count_params(model, cfg["mode"])
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = {**DEFAULTS, **yaml.safe_load(open(args.config, encoding="utf-8"))}
    set_seed(cfg["seed"])
    run_dir = ROOT / cfg["output_dir"] / f"finetune_{cfg['run_name']}"
    run_dir.mkdir(parents=True, exist_ok=True)
    seg_dir = ROOT / cfg["segments_dir"]
    if cfg["backend"] == "qwen3_asr":
        from cyasr import train_qwen  # Whisper runs don't import it
        return train_qwen.main(cfg, run_dir, seg_dir, read_rows, count_params)
    if cfg["backend"] != "whisper":
        raise ValueError(f"unknown backend {cfg['backend']!r}")

    processor = WhisperProcessor.from_pretrained(cfg["model"])
    model = build_model(cfg)
    ds_args = (processor, cfg["seed"], cfg["timestamp_fraction"])
    train_ds = SegmentDataset(seg_dir, "train", *ds_args, cfg["max_train_samples"])
    eval_ds = SegmentDataset(seg_dir, "train" if cfg["eval_on_train"] else "dev", *ds_args,
                             cfg["max_eval_samples"])
    if len(eval_ds) == 0:
        raise SystemExit("no dev rows to evaluate on (set eval_on_train: true for smoke tests only)")
    sot = processor.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
    for use_ts in (True, False):  # show one decoded label of each kind
        for ids in train_ds.labels:
            if (processor.tokenizer.convert_tokens_to_ids("<|notimestamps|>") not in ids) == use_ts:
                print("example label:", processor.tokenizer.decode(ids, decode_with_timestamps=True))
                break

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(run_dir / "checkpoints"), seed=cfg["seed"],
        learning_rate=cfg["learning_rate"], warmup_steps=cfg["warmup_steps"], max_steps=cfg["max_steps"],
        per_device_train_batch_size=cfg["batch_size"], per_device_eval_batch_size=cfg["eval_batch_size"],
        gradient_accumulation_steps=cfg["grad_accum"],
        gradient_checkpointing=cfg["gradient_checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": False},
        fp16=torch.cuda.is_available(),  # T4 has no bf16
        eval_strategy="steps", eval_steps=cfg["eval_steps"],
        save_strategy="steps", save_steps=cfg["eval_steps"], save_total_limit=2,
        load_best_model_at_end=True, metric_for_best_model="wer_basic", greater_is_better=False,
        predict_with_generate=True, generation_max_length=225,
        logging_steps=min(25, cfg["eval_steps"]), report_to="none",
        dataloader_num_workers=cfg["num_workers"], remove_unused_columns=False,
        label_names=["labels"])
    trainer = Seq2SeqTrainer(
        model=model, args=training_args, train_dataset=train_ds, eval_dataset=eval_ds,
        data_collator=Collator(sot), compute_metrics=make_metrics(processor),
        processing_class=processor,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=cfg["early_stopping_patience"])])
    ckpt_dir = run_dir / "checkpoints"
    last = get_last_checkpoint(str(ckpt_dir)) if cfg["resume"] and ckpt_dir.exists() else None
    if last:
        print("resuming from", last)  # timing.json then covers only the resumed part
    result = trainer.train(resume_from_checkpoint=last)

    best_model = trainer.model
    if cfg["mode"] == "lora":
        best_model = best_model.merge_and_unload()  # plain Whisper model, no adapters
    best_dir = run_dir / "best"
    best_model.save_pretrained(best_dir)
    processor.save_pretrained(best_dir)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    (run_dir / "env.json").write_text(json.dumps(environment(), indent=2), encoding="utf-8")
    log = trainer.state.log_history
    (run_dir / "train_log.json").write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")

    step = trainer.state.best_global_step
    # Trainer's runtime covers the whole loop, so dev evaluations are subtracted to get pure training time.
    total_s = result.metrics["train_runtime"]
    eval_s = sum(e["eval_runtime"] for e in log if "eval_runtime" in e)
    steps_done = trainer.state.global_step
    timing = {
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "mode": cfg["mode"],
        "total_time_min": round(total_s / 60, 2),
        "train_time_min": round((total_s - eval_s) / 60, 2),
        "eval_time_min": round(eval_s / 60, 2),
        "steps_done": steps_done,
        "best_step": step,
        "seconds_per_step": round((total_s - eval_s) / max(steps_done, 1), 3),
        "peak_gpu_memory_gb": (round(torch.cuda.max_memory_allocated() / 1e9, 2)
                               if torch.cuda.is_available() else None),
    }
    (run_dir / "timing.json").write_text(json.dumps(timing, indent=2), encoding="utf-8")
    print("timing:", timing)
    best_eval = next((e for e in log if e.get("step") == step and "eval_wer_basic" in e), {})
    print(f"best step {step}: " + ", ".join(f"{k[5:]} {v:.4f}" for k, v in best_eval.items()
                                           if k.startswith("eval_") and k.split("_")[1] in ("wer", "cer")))
    print(f"saved to {best_dir}")


if __name__ == "__main__":
    sys.exit(main())

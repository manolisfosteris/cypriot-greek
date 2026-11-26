"""Fine-tune a Qwen3-ASR checkpoint (Qwen/Qwen3-ASR-0.6B-hf, Reson8/Rumble-2-Flash) on the Cypriot segments.

Called by cyasr.train when the config has `backend: qwen3_asr`. Same recipe as Whisper where it carries over:
LoRA on the attention and feed-forward layers of encoder and decoder, dev WER every eval_steps, early stopping,
best checkpoint by dev WER, SpecAugment-style masking, adapters merged at the end.

Qwen3-ASR is a decoder-only language model fed with audio embeddings, so a training example is one sequence:
    <|im_start|>system\\n{controls}<|im_end|>\\n<|im_start|>user\\n<audio...><|im_end|>\\n
    <|im_start|>assistant\\nlanguage Greek<asr_text>{target text}<|im_end|>
The prompt is exactly what inference uses (apply_transcription_request with the language forced and the same
template options); only the target text and the final <|im_end|> are trained (labels -100 elsewhere).
Differences from the Whisper path: no timestamps (Qwen3-ASR has none), bf16 autocast instead of fp16.
"""
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import yaml
from torch.utils.data import ConcatDataset, Dataset
from transformers import (AutoProcessor, EarlyStoppingCallback, Qwen3ASRForConditionalGeneration, Trainer,
                          TrainingArguments)
from transformers.models.whisper.modeling_whisper import _compute_mask_indices
from transformers.trainer_utils import get_last_checkpoint

from cyasr.metrics import corpus_rate
from cyasr.normalize import clean_target, normalize
from cyasr.run_baseline import environment

MAX_TARGET_TOKENS = 1024  # a 30 s piece needs ~300; longer targets would mean a broken transcript
# LoRA on the same kinds of layers as Whisper: attention and feed-forward of the audio encoder and the decoder
# (not the convolutions, the audio-to-text projector or the output layer).
LORA_TARGETS = (r".*audio_tower\.layers\.\d+\.(self_attn\.(q|k|v|out)_proj|fc1|fc2)"
                r"|.*language_model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)")
# SpecAugment settings of Whisper's config, so both paths mask the same way
MASK_TIME_LENGTH, MASK_TIME_MIN, MASK_FEATURE_LENGTH, MASK_FEATURE_MIN = 10, 2, 10, 0


class QwenSegmentDataset(Dataset):
    """One split of segments: audio plus target token ids (the prompt is built per batch by the collator).

    `prompt` is the free-text context put into the system prompt after Rumble's control tokens (None = no
    context, the default prompt). Used as a dialect tag: Cypriot pieces get e.g. "Cypriot Greek", Standard
    Greek pieces none, so untagged inference keeps the standard behaviour.
    `subset` picks that many rows evenly spaced over the split (deterministic), for cheaper dev checks."""

    def __init__(self, segments_dir: Path, split: str, tokenizer, limit: int | None, read_rows,
                 prompt: str | None = None, subset: int | None = None):
        self.dir, self.split, self.prompt = segments_dir, split, prompt
        self.rows, self.targets = [], []
        all_rows = read_rows(segments_dir, split, limit)
        if subset and subset < len(all_rows):
            all_rows = [all_rows[i] for i in np.linspace(0, len(all_rows) - 1, subset).round().astype(int)]
        for row in all_rows:
            ids = tokenizer(clean_target(row["text"]), add_special_tokens=False)["input_ids"]
            if len(ids) <= MAX_TARGET_TOKENS:
                self.rows.append(row)
                self.targets.append(ids)
        print(f"[{segments_dir.name}/{split}] {len(self.rows)} pieces, {len(all_rows) - len(self.rows)} dropped "
              f"(targets > {MAX_TARGET_TOKENS} tokens), prompt {prompt!r}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        audio, sr = sf.read(self.dir / self.split / f"{self.rows[i]['id']}.wav", dtype="float32")
        assert sr == 16_000, sr
        return {"audio": audio, "target": self.targets[i], "prompt": self.prompt}


class QwenCollator:
    """Builds the prompts with the processor (identical to inference), then two views of each batch:
    - training: prompt + target + <|im_end|>, right-padded, labels only on target and <|im_end|>;
    - generation (gen_*): the prompt alone, left-padded, for dev WER.
    The audio features are shared. `augment` applies SpecAugment-style masking (training batches only)."""

    def __init__(self, processor, language: str, template_kwargs: dict, augment: bool = False,
                 mask_time_prob: float = 0.0, mask_feature_prob: float = 0.0):
        self.processor, self.language, self.template_kwargs = processor, language, template_kwargs
        tok = processor.tokenizer
        self.im_end = tok.convert_tokens_to_ids("<|im_end|>")
        self.pad = tok.pad_token_id
        self.augment, self.mask_time_prob, self.mask_feature_prob = augment, mask_time_prob, mask_feature_prob

    def __call__(self, batch: list[dict]) -> dict:
        contexts = [b.get("prompt") for b in batch]  # per piece: dialect tag or None (mixed batches allowed)
        prompt = self.processor.apply_transcription_request(
            [b["audio"] for b in batch], language=[self.language] * len(batch),
            prompt=contexts if any(c is not None for c in contexts) else None, **self.template_kwargs)
        ids, att = prompt["input_ids"], prompt["attention_mask"]  # left-padded
        seqs, labels = [], []
        for row_ids, row_att, b in zip(ids.tolist(), att.tolist(), batch):
            p = [t for t, a in zip(row_ids, row_att) if a]  # drop the left padding
            target = b["target"] + [self.im_end]
            seqs.append(p + target)
            labels.append([-100] * len(p) + target)
        width = max(len(s) for s in seqs)
        out = {
            "input_ids": torch.tensor([s + [self.pad] * (width - len(s)) for s in seqs]),
            "attention_mask": torch.tensor([[1] * len(s) + [0] * (width - len(s)) for s in seqs]),
            "labels": torch.tensor([l + [-100] * (width - len(l)) for l in labels]),
            "input_features": prompt["input_features"],
            "input_features_mask": prompt["input_features_mask"],
            "gen_input_ids": ids, "gen_attention_mask": att,
        }
        if self.augment:
            out["input_features"] = self.spec_augment(out["input_features"].clone(), out["input_features_mask"])
        return out

    def spec_augment(self, feats: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Zero random time spans and mel bands, as Whisper's _mask_input_features does (numpy RNG, seeded)."""
        b, n_mels, n_frames = feats.shape
        if self.mask_time_prob > 0:
            t = _compute_mask_indices((b, n_frames), self.mask_time_prob, MASK_TIME_LENGTH,
                                      attention_mask=mask, min_masks=MASK_TIME_MIN)
            feats[torch.from_numpy(t)[:, None].expand(-1, n_mels, -1)] = 0
        if self.mask_feature_prob > 0:
            f = _compute_mask_indices((b, n_mels), self.mask_feature_prob, MASK_FEATURE_LENGTH,
                                      min_masks=MASK_FEATURE_MIN)
            feats[torch.from_numpy(f)] = 0
        return feats


class QwenASRTrainer(Trainer):
    """Trainer whose evaluation generates from the prompt alone (so dev WER is a real transcription score),
    while the loss still uses the full sequence. Training ignores the gen_* views."""

    def __init__(self, *args, eval_collator=None, max_new_tokens: int = 512, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_collator, self.max_new_tokens = eval_collator, max_new_tokens

    def get_eval_dataloader(self, eval_dataset=None):
        train_collator, self.data_collator = self.data_collator, self.eval_collator  # no SpecAugment on dev
        try:
            return super().get_eval_dataloader(eval_dataset)
        finally:
            self.data_collator = train_collator

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        inputs = {k: v for k, v in inputs.items() if not k.startswith("gen_")}
        return super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        gen = {k[4:]: inputs.pop(k) for k in list(inputs) if k.startswith("gen_")}
        with torch.no_grad(), self.compute_loss_context_manager():
            loss = model(**inputs).loss.detach()
            if prediction_loss_only:
                return loss, None, None
            out = model.generate(input_ids=gen["input_ids"], attention_mask=gen["attention_mask"],
                                 input_features=inputs["input_features"],
                                 input_features_mask=inputs["input_features_mask"],
                                 max_new_tokens=self.max_new_tokens, do_sample=False)
        return loss, out[:, gen["input_ids"].shape[1]:], inputs["labels"]


def make_metrics(tokenizer):
    def compute_metrics(pred) -> dict:
        pred_ids, label_ids = pred.predictions.copy(), pred.label_ids.copy()
        pred_ids[pred_ids == -100] = tokenizer.pad_token_id  # -100 pads predictions when batches are joined
        label_ids[label_ids == -100] = tokenizer.pad_token_id
        hyps = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        refs = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        out = {}
        for level in ["basic", "nofill", "nodiac"]:
            r, h = [normalize(x, level) for x in refs], [normalize(x, level) for x in hyps]
            out[f"wer_{level}"] = corpus_rate(r, h, "word")["rate"]
        out["cer_nodiac"] = corpus_rate([normalize(x, "nodiac") for x in refs],
                                        [normalize(x, "nodiac") for x in hyps], "char")["rate"]
        return out
    return compute_metrics


def build_model(cfg: dict, count_params):
    # fp32 master weights, bf16 autocast: the LoRA updates are small, so they are kept in full precision.
    model = Qwen3ASRForConditionalGeneration.from_pretrained(cfg["model"], revision=cfg["revision"],
                                                             dtype=torch.float32)
    model.generation_config.do_sample = False
    if cfg["gradient_checkpointing"]:
        model.config.use_cache = False
    if cfg["mode"] == "lora":
        from peft import LoraConfig, get_peft_model
        lora = LoraConfig(r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"], lora_dropout=cfg["lora_dropout"],
                          target_modules=LORA_TARGETS)
        model = get_peft_model(model, lora)
    elif cfg["mode"] == "frozen_encoder":
        for p in model.model.audio_tower.parameters():
            p.requires_grad = False
    elif cfg["mode"] != "full":
        raise ValueError("mode must be full, frozen_encoder or lora")
    count_params(model, cfg["mode"])
    return model


def main(cfg: dict, run_dir: Path, seg_dir: Path, read_rows, count_params) -> None:
    processor = AutoProcessor.from_pretrained(cfg["model"], revision=cfg["revision"])
    tok = processor.tokenizer
    model = build_model(cfg, count_params)
    tag = cfg["dialect_prompt"]  # None = plain prompt (all earlier runs)
    train_ds = QwenSegmentDataset(seg_dir, "train", tok, cfg["max_train_samples"], read_rows, prompt=tag)
    eval_ds = QwenSegmentDataset(seg_dir, "train" if cfg["eval_on_train"] else "dev", tok,
                                 cfg["max_eval_samples"], read_rows, prompt=tag, subset=cfg["eval_subset"])
    if len(eval_ds) == 0:
        raise SystemExit("no dev rows to evaluate on (set eval_on_train: true for smoke tests only)")
    metric = "wer_basic"
    if cfg["smg_segments_dir"]:
        # Rehearsal: Standard Greek pieces (no tag) mixed into training. The trainer samples uniformly from the
        # concatenation, so the Standard Greek share of each batch is smg / (cypriot + smg) on average.
        smg_dir = Path(cfg["smg_segments_dir"])
        smg_dir = smg_dir if smg_dir.is_absolute() else Path(__file__).resolve().parent.parent / smg_dir
        smg_train = QwenSegmentDataset(smg_dir, "train", tok, cfg["smg_max_samples"], read_rows, prompt=None)
        smg_dev = QwenSegmentDataset(smg_dir, "dev", tok, None, read_rows, prompt=None,
                                     subset=cfg["smg_dev_samples"])
        print(f"training mix: {len(train_ds)} Cypriot + {len(smg_train)} Standard Greek pieces "
              f"({len(smg_train) / (len(train_ds) + len(smg_train)):.0%} Standard Greek)")
        train_ds = ConcatDataset([train_ds, smg_train])
        eval_ds = {"cy": eval_ds, "smg": smg_dev}  # both logged at every check; best checkpoint by Cypriot WER
        metric = "cy_wer_basic"
    collator_args = (processor, cfg["language"], cfg["template_kwargs"] or {})
    train_collator = QwenCollator(*collator_args, augment=cfg["spec_augment"],
                                  mask_time_prob=cfg["mask_time_prob"], mask_feature_prob=cfg["mask_feature_prob"])
    eval_collator = QwenCollator(*collator_args)
    examples = [train_ds[0]] + ([train_ds[len(train_ds) - 1]] if cfg["smg_segments_dir"] else [])
    for ex in examples:  # show a training sequence (Cypriot, and Standard Greek when mixed) and what is trained
        example = train_collator([ex])
        print("example input :", tok.decode(example["input_ids"][0]).replace("<|audio_pad|>", "")[:400])
        print("example target:", tok.decode([t for t in example["labels"][0].tolist() if t != -100]))

    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported(including_emulation=False)
    args = TrainingArguments(
        output_dir=str(run_dir / "checkpoints"), seed=cfg["seed"],
        learning_rate=cfg["learning_rate"], warmup_steps=cfg["warmup_steps"], max_steps=cfg["max_steps"],
        per_device_train_batch_size=cfg["batch_size"], per_device_eval_batch_size=cfg["eval_batch_size"],
        gradient_accumulation_steps=cfg["grad_accum"],
        gradient_checkpointing=cfg["gradient_checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=bf16, fp16=torch.cuda.is_available() and not bf16,
        eval_strategy="steps", eval_steps=cfg["eval_steps"],
        save_strategy="steps", save_steps=cfg["eval_steps"], save_total_limit=2,
        load_best_model_at_end=True, metric_for_best_model=metric, greater_is_better=False,
        logging_steps=min(25, cfg["eval_steps"]), report_to="none",
        dataloader_num_workers=cfg["num_workers"], remove_unused_columns=False, label_names=["labels"])
    trainer = QwenASRTrainer(
        model=model, args=args, train_dataset=train_ds, eval_dataset=eval_ds,
        data_collator=train_collator, eval_collator=eval_collator, compute_metrics=make_metrics(tok),
        processing_class=processor, max_new_tokens=cfg["max_new_tokens"],
        callbacks=[EarlyStoppingCallback(early_stopping_patience=cfg["early_stopping_patience"])])
    ckpt_dir = run_dir / "checkpoints"
    last = get_last_checkpoint(str(ckpt_dir)) if cfg["resume"] and ckpt_dir.exists() else None
    if last:
        print("resuming from", last)
    result = trainer.train(resume_from_checkpoint=last)

    best_model = trainer.model
    if cfg["mode"] == "lora":
        best_model = best_model.merge_and_unload()  # plain Qwen3-ASR model, loadable by run_baseline
    best_dir = run_dir / "best"
    best_model.to(torch.bfloat16).save_pretrained(best_dir)  # same dtype as the released checkpoints
    processor.save_pretrained(best_dir)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    (run_dir / "env.json").write_text(json.dumps(environment(), indent=2), encoding="utf-8")
    log = trainer.state.log_history
    (run_dir / "train_log.json").write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")

    total_s = result.metrics["train_runtime"]
    eval_s = sum(v for e in log for k, v in e.items() if k.startswith("eval_") and k.endswith("_runtime"))
    steps_done = trainer.state.global_step
    timing = {
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "mode": cfg["mode"], "resumed_from": last,
        "total_time_min": round(total_s / 60, 2), "train_time_min": round((total_s - eval_s) / 60, 2),
        "eval_time_min": round(eval_s / 60, 2), "steps_done": steps_done,
        "best_step": trainer.state.best_global_step,
        "seconds_per_step": round((total_s - eval_s) / max(steps_done, 1), 3),
        "peak_gpu_memory_gb": (round(torch.cuda.max_memory_allocated() / 1e9, 2)
                               if torch.cuda.is_available() else None),
    }
    (run_dir / "timing.json").write_text(json.dumps(timing, indent=2), encoding="utf-8")
    print("timing:", timing)
    print(f"saved to {best_dir}")

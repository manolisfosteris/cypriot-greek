"""Qwen3-ASR inference (native transformers implementation), with the same interface as Transcriber.

Unlike Whisper, Qwen3-ASR takes up to 20 minutes of audio in one pass, so there are no 30 s windows,
no timestamps and no window-boundary or silent-tail effects. Decoding is greedy (the checkpoint's
generation config); the language is forced by prefilling "language Greek<asr_text>".
"""
import numpy as np
import torch
from transformers import AutoProcessor, Qwen3ASRForConditionalGeneration
from transformers.models.qwen3_asr.processing_qwen3_asr import _parse_single_output

from cyasr.data import SAMPLE_RATE

MAX_AUDIO_S = 1200  # model limit per call (Qwen3-ASR technical report)


class QwenTranscriber:
    """Wraps one Qwen3-ASR checkpoint. `language` is an ISO code ("el") or None for auto-detection.
    `dtype` "auto" means bfloat16 where the GPU supports it (L4, A100), else float32 (T4, CPU):
    fp16 is avoided because the model was trained in bf16 and fp16 can overflow.
    `template_kwargs` are passed to the checkpoint's chat template, e.g. Reson8/Rumble-2-Flash's
    {"digits": False, "filler": True} (numbers as words, keep fillers). Qwen's own template ignores them."""

    def __init__(self, model_name: str, revision: str | None = None, language: str | None = "el",
                 batch_size: int = 8, max_new_tokens: int = 4096, dtype: str = "auto",
                 template_kwargs: dict | None = None, prompt: str | None = None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if dtype == "auto":
            bf16 = self.device == "cuda" and torch.cuda.is_bf16_supported(including_emulation=False)
            self.dtype = torch.bfloat16 if bf16 else torch.float32
        else:
            self.dtype = getattr(torch, dtype)
        self.processor = AutoProcessor.from_pretrained(model_name, revision=revision)
        self.model = Qwen3ASRForConditionalGeneration.from_pretrained(
            model_name, revision=revision, dtype=self.dtype).to(self.device).eval()
        self.language, self.batch_size, self.max_new_tokens = language, batch_size, max_new_tokens
        self.template_kwargs = template_kwargs or {}
        self.prompt = prompt  # free-text system context, e.g. a dialect tag ("Cypriot Greek"); None = none

    def transcribe(self, audios: list[np.ndarray]) -> list[str]:
        return [d["hyp"] for d in self.transcribe_detailed(audios)]

    @torch.inference_mode()
    def transcribe_detailed(self, audios: list[np.ndarray]) -> list[dict]:
        """Transcribe 16 kHz mono arrays. Order is preserved. Each result has `hyp` (the official output,
        after the processor's repetition clean-up), `hyp_raw` (before clean-up), `language` (detected,
        or None when forced), `n_tokens` and `hit_max_tokens` (the output was cut off)."""
        for a in audios:
            if len(a) > MAX_AUDIO_S * SAMPLE_RATE:
                raise ValueError(f"clip longer than {MAX_AUDIO_S} s")
        # Batches hold clips of similar length (less padding). Padding is masked, but the smoke test
        # in notebook 07 checks that batch size does not change the output.
        order = sorted(range(len(audios)), key=lambda i: len(audios[i]))
        results: list[dict] = [{}] * len(audios)
        for j in range(0, len(order), self.batch_size):
            idx = order[j:j + self.batch_size]
            for i, r in zip(idx, self._batch([audios[i] for i in idx])):
                results[i] = r
        return results

    def _batch(self, batch: list[np.ndarray]) -> list[dict]:
        # The processor expects 16 kHz (its default), which is what load_audio returns.
        inputs = self.processor.apply_transcription_request(
            batch, language=[self.language] * len(batch) if self.language else None,
            prompt=self.prompt, **self.template_kwargs)
        inputs = inputs.to(self.device, self.dtype)  # casts only the float tensors (audio features)
        out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        gen = out[:, inputs["input_ids"].shape[1]:]
        raw = self.processor.tokenizer.batch_decode(gen, skip_special_tokens=True)
        eos = self.model.generation_config.eos_token_id
        eos = set(eos if isinstance(eos, list) else [eos])
        results = []
        for text, ids in zip(raw, gen.tolist()):
            n = next((k for k, t in enumerate(ids) if t in eos), len(ids))  # text tokens before end-of-turn
            parsed = _parse_single_output(text)
            raw_text = text.split("<asr_text>", 1)[-1].strip()
            results.append({"hyp": parsed["transcription"], "hyp_raw": raw_text,
                            "language": parsed["language"], "n_tokens": n,
                            "hit_max_tokens": n >= self.max_new_tokens})
        return results

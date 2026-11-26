"""Whisper inference with deterministic (greedy by default) decoding and long-form support."""
import numpy as np
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor, pipeline

from cyasr.data import SAMPLE_RATE

WHISPER_WINDOW_S = 30

# OpenAI-style fallback: if a window looks stuck (too repetitive or too unsure), re-decode it
# at a higher temperature. Thresholds follow the transformers long-form defaults.
FALLBACK_KWARGS = dict(temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
                       compression_ratio_threshold=1.35, logprob_threshold=-1.0,
                       condition_on_prev_tokens=False)


class Transcriber:
    """Wraps one Whisper checkpoint. `long_form` is "sequential" or "chunked";
    `decoding` is "greedy" (fully deterministic) or "fallback" (temperature fallback, seeded).
    `no_speech_threshold` (fallback only): a window whose no-speech probability is above it AND whose
    text has average log-probability below logprob_threshold is treated as silent and dropped."""

    def __init__(self, model_name: str, language: str = "el", task: str = "transcribe",
                 num_beams: int = 1, long_form: str = "sequential", batch_size: int = 4,
                 decoding: str = "greedy", no_speech_threshold: float | None = None):
        if long_form not in ("sequential", "chunked"):
            raise ValueError("long_form must be 'sequential' or 'chunked'")
        if decoding not in ("greedy", "fallback"):
            raise ValueError("decoding must be 'greedy' or 'fallback'")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=self.dtype).to(self.device).eval()
        self.long_form, self.batch_size = long_form, batch_size
        self.gen_kwargs = dict(language=language, task=task, num_beams=num_beams)
        if decoding == "fallback":
            self.gen_kwargs.update(FALLBACK_KWARGS)
        else:
            self.gen_kwargs["do_sample"] = False
        if no_speech_threshold is not None:
            if decoding != "fallback":
                raise ValueError("no_speech_threshold needs decoding='fallback' (it uses its logprob_threshold)")
            self.gen_kwargs["no_speech_threshold"] = no_speech_threshold
        if long_form == "chunked":
            self.pipe = pipeline("automatic-speech-recognition", model=self.model,
                                 tokenizer=self.processor.tokenizer,
                                 feature_extractor=self.processor.feature_extractor,
                                 chunk_length_s=WHISPER_WINDOW_S, batch_size=batch_size,
                                 device=self.device, torch_dtype=self.dtype)

    @torch.inference_mode()
    def transcribe(self, audios: list[np.ndarray]) -> list[str]:
        """Transcribe a list of 16 kHz mono arrays. Order is preserved."""
        if self.long_form == "chunked":
            outs = self.pipe([{"raw": a, "sampling_rate": SAMPLE_RATE} for a in audios],
                             generate_kwargs=self.gen_kwargs)
            return [o["text"].strip() for o in outs]
        # Short and long clips are never batched together: mixing them changes how short clips
        # are padded, which changes their output. This keeps results independent of batch_size.
        texts = [""] * len(audios)
        for long in (False, True):
            idx = [i for i, a in enumerate(audios) if (len(a) > WHISPER_WINDOW_S * SAMPLE_RATE) == long]
            for j in range(0, len(idx), self.batch_size):
                chunk = idx[j:j + self.batch_size]
                for i, text in zip(chunk, self._sequential_batch([audios[i] for i in chunk], long)):
                    texts[i] = text
        return texts

    def _sequential_batch(self, batch: list[np.ndarray], is_long: bool) -> list[str]:
        # Timestamps are always on (as in OpenAI's transcribe()), so short and long clips are
        # decoded the same way.
        if is_long:
            # Keep the full audio; Whisper's sequential algorithm slides a 30 s window over it.
            inputs = self.processor(batch, sampling_rate=SAMPLE_RATE, return_tensors="pt",
                                    truncation=False, padding="longest", return_attention_mask=True)
        else:
            inputs = self.processor(batch, sampling_rate=SAMPLE_RATE, return_tensors="pt",
                                    return_attention_mask=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        inputs["input_features"] = inputs["input_features"].to(self.dtype)
        ids = self.model.generate(**inputs, return_timestamps=True, **self.gen_kwargs)
        # With no_speech_threshold, skipped windows can turn the token ids into floats; ids must be
        # integers (< 52k, so the conversion is exact).
        ids = ids.long()
        return [t.strip() for t in self.processor.batch_decode(ids, skip_special_tokens=True)]

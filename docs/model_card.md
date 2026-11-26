---
license: apache-2.0
language:
- el
base_model:
- Reson8/Rumble-2-Flash
pipeline_tag: automatic-speech-recognition
library_name: transformers
tags:
- qwen3-asr
- cypriot-greek
- dialect
- lora
---

# Rumble-2-Flash, fine-tuned for Cypriot Greek 🇨🇾

[Reson8/Rumble-2-Flash](https://huggingface.co/Reson8/Rumble-2-Flash) (509M parameters) fine-tuned with LoRA on
Cypriot Greek spontaneous speech, with a small amount of Standard Greek mixed into training to limit forgetting.
The LoRA adapters are merged, so this is a plain checkpoint with the same architecture and processor as the base
model.

Code, configs, all predictions and the full experiment log: https://github.com/manolisfosteris/cypriot-greek

## Results

WER at the `basic` normalization level (NFC, lowercase, final sigma, no punctuation), computed over the whole corpus.
Greedy decoding, batch size 1, full-length clips.

| | Cypriot dev | Cypriot test | FLEURS Greek (200 clips) |
|---|---|---|---|
| Rumble-2-Flash, zero-shot | 47.3% | 55.7% | 14.1% |
| Rumble-2-Flash + LoRA, Cypriot only | 17.3% | 26.9% | 39.7% |
| **this model** | 17.9% | **24.7%** | 29.6% |

Test: 371 clips, 2.89 h, 3 speakers not seen in training or dev. On test this model is significantly better than
the Cypriot-only LoRA (−2.1 points, paired bootstrap 95% CI [−3.6, −0.7]). Test CER (accents ignored): 8.7%.

## Usage

```python
import torch
import soundfile as sf
from transformers import AutoProcessor, Qwen3ASRForConditionalGeneration

model_id = "MODEL_ID"
processor = AutoProcessor.from_pretrained(model_id)
model = Qwen3ASRForConditionalGeneration.from_pretrained(model_id, dtype=torch.bfloat16).to("cuda").eval()

audio, sr = sf.read("clip.wav", dtype="float32")  # mono, 16 kHz (resample first if needed)
inputs = processor.apply_transcription_request(
    [audio], language=["el"], prompt="Cypriot Greek",
    digits=False, filler=True, verbatim=True,
).to(model.device, model.dtype)
out = model.generate(**inputs, max_new_tokens=4096, do_sample=False)
print(processor.decode(out[:, inputs["input_ids"].shape[1]:], return_format="transcription_only")[0])
```

- Settings used in all our evaluations: `digits=False` (numbers as words), `filler=True` (keep hesitations),
  `verbatim=True`, Greek forced, greedy decoding, batch size 1.
- `prompt="Cypriot Greek"` is the tag used in training. In our experiments it made no measurable difference, with or
  without it.
- Clips up to 20 minutes are decoded in one pass. Use bf16 on GPUs that support it, float32 otherwise (not fp16).

## Training

- **Data**: [Mozilla Common Voice](https://mozilladatacollective.com) Spontaneous Speech, Cypriot Greek (el-CY) v5.0,
  CC0. Official speaker-disjoint splits: train 566 clips (5 speakers), cut into 820 pieces of 30 s or less by forced
  alignment (about 4.7 h); dev 280 clips; test 371 clips. Plus 410 clips (206 speakers, 30 min) of Common Voice
  Scripted Speech Greek 27.0 (CC0) for rehearsal, with no tag.
- **Method**: LoRA r = 32, α = 64, dropout 0.05 on the attention and feed-forward layers of the audio encoder and the
  decoder. Learning rate 1e-4, 50 warmup steps, up to 1,000 steps, effective batch 16 (4 × 4), SpecAugment, seed 42.
  Best checkpoint by Cypriot dev WER; adapters merged.
- **Base revision**: `Reson8/Rumble-2-Flash@95bd5f9133a2398e5e48a98c089fe48e122bde97`.
- **Hardware**: one NVIDIA L4 (Google Colab), bf16, 32 min of training, peak memory 13.1 GB.
- Config: `configs/finetune_rumble-2-flash_mix_tag.yaml` in the GitHub repository.

## Limitations

- Trained on 5 speakers and evaluated on 5 more (2 dev, 3 test). One test speaker accounts for 146 of the 371 test
  clips and is much harder (about 35% WER).
- Spontaneous speech from Common Voice only; not tested on other domains, noisy or telephone audio.
- Output follows the Common Voice transcription conventions: Cypriot spelling (`τζαι`, `εν`), fillers kept, numbers
  as words. Cypriot Greek has no standard spelling, so other conventions will count as errors.
- Still worse at Standard Greek than the base model (FLEURS 29.6% vs 14.1%). For Standard Greek audio, use
  Rumble-2-Flash.
- One training seed. Only Greek was evaluated; other languages the base model supports were not tested after
  fine-tuning.
- The base model's auxiliary CTC head is not part of this checkpoint and was not fine-tuned.

## License

Apache-2.0, the license of the base model [Reson8/Rumble-2-Flash](https://huggingface.co/Reson8/Rumble-2-Flash),
which is itself built on [Qwen/Qwen3-ASR-0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) (Apache-2.0). Training data
is CC0.

## Acknowledgements

Mozilla Common Voice and its Cypriot Greek contributors for the data, Reson8 for Rumble-2-Flash, and the Qwen team for
Qwen3-ASR. Earlier work on this dataset: Stamou et al. 2026,
[First Steps in ASR for Cypriot Greek](https://aclanthology.org/2026.dialres-1.30/).

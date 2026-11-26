# cypriot-greek

Speech recognition for Cypriot Greek, a Greek dialect with no standard spelling and very little transcribed audio.
We fine-tuned [Reson8/Rumble-2-Flash](https://huggingface.co/Reson8/Rumble-2-Flash) (509M) and Whisper with LoRA
on about 4.7 h of spontaneous speech from Common Voice. Every model was chosen on the dev split and then scored
once on a test split of 3 speakers it had never heard. The best model reaches **24.7% WER on test**, down from
55.7% zero-shot. Fine-tuning on Cypriot data alone breaks Standard Greek (FLEURS WER 14.1% → 39.4%). Mixing 30 min
of Standard Greek into training brings that down to 29.6% and also gives the best Cypriot test score.

**About the base model.** Rumble-2-Flash is Reson8's speech recognition model, released in November 2026. Its
model card names [Qwen3-ASR-0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) (from Alibaba's Qwen team) as the base
model. Comparing the two checkpoints' files shows that Rumble keeps Qwen's audio encoder architecture (18 layers)
but has half the text decoder layers (14 instead of 28). Its vocabulary is cut from 151,643 to 99,562 tokens, with
nothing new added. It also adds six control tokens for transcription options (digits, fillers, verbatim) and an
extra CTC output head on the encoder. The training was more than pruning: zero-shot Greek WER on FLEURS drops from
52.9% (Qwen) to 13.5% (Rumble) in our measurement on 100 clips. Reson8 does not publish how the model was trained, which decoder layers were kept,
or how they were chosen. Details: [docs/tokenizer_diff.md](docs/tokenizer_diff.md) and PROJECT_LOG sections 10, 13
and 15.

## Links

- Model weights: TODO (Hugging Face upload pending)
- Dataset: [Common Voice Spontaneous Speech, Cypriot Greek (el-CY) v5.0](https://mozilladatacollective.com), CC0
- Rehearsal data: Common Voice Scripted Speech, Greek (el) 27.0, CC0, same site
- Baseline paper: [Stamou et al. 2026, First Steps in ASR for Cypriot Greek](https://aclanthology.org/2026.dialres-1.30/)
- Full experiment log: [docs/PROJECT_LOG.md](docs/PROJECT_LOG.md)

## Results

Test split: 371 full-length clips, 2.89 h, 3 speakers not seen in training or dev. Greedy decoding, batch 1.
WER at the `basic` normalization level (lowercase, no punctuation; see [Scoring](#scoring)).

| model | dev WER | test WER | test WER, accents ignored | test CER, accents ignored |
|---|---|---|---|---|
| Rumble-2-Flash, zero-shot | 47.3% | 55.7% | 53.6% | 20.8% |
| Rumble-2-Flash + LoRA | 17.3% | 26.9% | 23.8% | 10.9% |
| **Rumble-2-Flash + LoRA + Standard Greek rehearsal** (released model) | 17.9% | **24.7%** | **21.4%** | 8.7% |
| whisper-medium, zero-shot | 50.3% | 58.3% | 55.5% | 21.9% |
| whisper-medium + LoRA | 19.9% | 25.4% | 21.9% | 8.3% |
| whisper-large-v3, zero-shot | 47.3% | 52.6% | 50.1% | 18.5% |

Paired bootstrap on test (10,000 resamples): the rehearsal model beats the plain Rumble LoRA by 2.1 points (95% CI
[−3.6, −0.7]). The three fine-tuned models are not significantly different from each other on test, although Rumble
led on dev. Also tried, dev only: whisper-small (58.8% → 28.6%) and Qwen3-ASR-0.6B (73.1% → 31.6%).

**Standard Greek after fine-tuning** (first 200 clips of FLEURS `el_gr` test, WER):

| Rumble-2-Flash | Cypriot dev WER | FLEURS Greek WER |
|---|---|---|
| zero-shot | 47.3% | 14.1% |
| + LoRA | 17.3% | 39.7% |
| + LoRA, dialect tag in the prompt | 17.6% | 39.6% |
| + LoRA, Standard Greek rehearsal | 18.2% | 30.3% |
| + LoRA, rehearsal + dialect tag (released model) | 17.9% | 29.6% |

We also tried blending the original and fine-tuned weights (WiSE-FT, [figure](docs/figures/wiseft_tradeoff.png)).
It trades one language against the other and does not fix both.

**Comparison with Stamou et al. (2026).** They report 33.93% (whisper-large-v3) and 37.85% (whisper-medium). Their
numbers come from an earlier corpus version, a random split that is not speaker-independent, segments of 30 s or
less and their own normalization. Ours come from the official speaker-disjoint split, full-length clips and the
normalization below. The numbers are not directly comparable. Details: PROJECT_LOG section 14.

## Quick start

```bash
git clone https://github.com/TODO/cypriot-greek.git && cd cypriot-greek
python3 -m venv .venv && . .venv/bin/activate
pip install torch            # pick the CUDA or CPU build for your machine from pytorch.org
pip install -r requirements.txt
```

Transcribe a file with the fine-tuned model:

```python
from cyasr.data import load_audio
from cyasr.transcribe_qwen import QwenTranscriber

asr = QwenTranscriber(
    "TODO/hub-id-of-the-released-model",   # or a local folder, e.g. runs/finetune_rumble-2-flash_mix_tag/best
    language="el",
    batch_size=1,
    template_kwargs={"digits": False, "filler": True, "verbatim": True},
    prompt="Cypriot Greek",                # the tag used in training; it made no measurable difference
)
print(asr.transcribe([load_audio("clip.mp3")])[0])
```

Clips can be up to 20 minutes long and are decoded in one pass. On a GPU with bf16 support (L4, A100) the model
runs in bf16, otherwise in float32. Use `batch_size=1` to reproduce our numbers: in bf16, the batch size alone
changes the outputs.

## Reproducing the training

All GPU work ran on Google Colab. The notebooks in `notebooks/` are the exact runners we used. They mount Google
Drive, unpack a code bundle (`bash scripts/make_colab_bundle.sh`) and keep checkpoints on Drive so a run can
resume. The commands below are what they call.

**1. Data.** Download from the Mozilla Data Collective (account needed) and lay it out as:

```
data/ss-corpus-el-CY.tsv     # Spontaneous Speech el-CY v5.0 (sps-corpus-5.0-2026-09-11)
data/audios/*.mp3
data/cv_el/                  # Common Voice Greek 27.0, extracted (rehearsal runs only)
```

The `split` column in the TSV gives the speaker-disjoint splits: train 566 clips (5 speakers), dev 280 (2 speakers),
test 371 (3 speakers).

**2. Cut training clips into pieces of 30 s or less** with forced alignment (566 train clips → 820 pieces; dev
→ 407 pieces, used for checks during training):

```bash
python -m cyasr.segment --split train
python -m cyasr.segment --split dev
```

**3. Prepare the Standard Greek rehearsal set** (410 clips from 206 speakers, 30 min; 100 dev clips):

```bash
python scripts/prep_cv_greek.py data/cv_el --out data/segments_cv_el
```

**4. Train** (released model: [configs/finetune_rumble-2-flash_mix_tag.yaml](configs/finetune_rumble-2-flash_mix_tag.yaml),
notebook 11):

```bash
python -m cyasr.train --config configs/finetune_rumble-2-flash_mix_tag.yaml
```

Recipe, the same for every model: LoRA r = 32, α = 64, dropout 0.05 on the attention and feed-forward layers of
encoder and decoder; learning rate 1e-4, 50 warmup steps, up to 1,000 steps, effective batch 16, SpecAugment,
early stopping on dev WER, seed 42. Adapters are merged at the end, and the result is saved to
`runs/finetune_<run_name>/best/`. Base model revisions are pinned in the configs.

Hardware and time: the released model took 32 min of training plus 18 min of dev checks on a Colab L4 (24 GB),
bf16, peak 13.1 GB. whisper-medium took 61 min of training on an L4; whisper-small trained on a T4.

Other configs: `finetune_rumble-2-flash.yaml` (plain LoRA), `_tag`, `_mix`, `finetune_whisper-{small,medium}.yaml`,
`finetune_qwen3-asr-0.6b.yaml`.

**5. Evaluate** on full-length clips (writes `runs/<run_name>_<decoding>_<split>/` with `predictions.jsonl`,
`metrics.json`, `config.yaml`, `env.json`):

```bash
python -m cyasr.run_baseline --config configs/eval_rumble-2-flash_mix_tag.yaml \
    --model runs/finetune_rumble-2-flash_mix_tag/best --splits dev test
python -m cyasr.run_baseline --config configs/baseline_rumble-2-flash_matched.yaml   # zero-shot
python scripts/summarize_runs.py
```

FLEURS checks: `scripts/eval_fleurs.py` (notebooks 09 and 11). Weight blending: `scripts/wiseft_sweep.py` (notebook
10). Predictions and metrics of every reported dev and test run are already in `runs/`, so the tables can be
checked without a GPU. TODO: a single script that rebuilds all the tables above from `runs/`. The FLEURS, blending
and rehearsal results are not yet in `runs/` either.

### Scoring

WER and CER are computed over the whole corpus, not averaged per clip. Text is compared at several normalization
levels:

| level | adds |
|---|---|
| raw | whitespace collapse |
| basic | NFC, lowercase, final sigma, removes brackets, punctuation, apostrophes; hyphens split words |
| nofill | drops hesitations (`εεε`, `εμ`, `μμμ`) |
| nodiac | strips accents and the Cypriot breve (`τζ̆` → `τζ`) |
| phonetic | merges homophone spellings (ι/η/υ/ει/οι, ο/ω, αι/ε). Diagnostic only |

`basic` is the headline number. Tests: `pytest`.

## Limitations and intended use

Intended for research on Cypriot Greek and dialect ASR, and as a starting point for further fine-tuning. Not tested
for production use.

- Trained and evaluated on 10 speakers in total; the test set has 3, and one of them has 146 of the 371 clips
  (about 35% WER after fine-tuning).
- Spontaneous speech from Common Voice only. No tests on other domains, noisy audio or telephone audio.
- One training seed per configuration.
- Output follows the Common Voice transcription conventions: Cypriot spelling (`τζαι`, `εν`), fillers kept,
  numbers written as words. There is no standard Cypriot orthography, so other conventions will count as errors.
- Standard Greek accuracy is still worse than the base model's (FLEURS 29.6% vs 14.1%). For Standard Greek audio,
  use the base model.
- The Standard Greek check uses only 200 FLEURS clips of read news speech.

## License

- Code in this repository: no license yet (all rights reserved by default).
- Released model weights: Apache-2.0, inherited from the base model Reson8/Rumble-2-Flash (Apache-2.0).
- Whisper (MIT), Qwen3-ASR-0.6B (Apache-2.0).
- Data: Common Voice (CC0), FLEURS (CC-BY 4.0). Transcripts in `runs/` come from Common Voice (CC0).

## Citation and acknowledgements

TODO: citation entry for this repository / model.

Baseline:

```bibtex
@inproceedings{stamou-etal-2026-first,
  title     = {First Steps in {ASR} for {C}ypriot {G}reek: Challenges and Insights},
  author    = {Stamou and others},  % TODO: full author list from the ACL Anthology page
  booktitle = {Proceedings of DialRes 2026},
  pages     = {308--314},
  year      = {2026},
  url       = {https://aclanthology.org/2026.dialres-1.30/}
}
```

Thanks to Mozilla Common Voice and its Cypriot Greek contributors for the data, Reson8 for Rumble-2-Flash, the Qwen
team for Qwen3-ASR, OpenAI for Whisper, and the authors of
[ctc-forced-aligner](https://github.com/MahmoudAshraf97/ctc-forced-aligner), used for segmentation.

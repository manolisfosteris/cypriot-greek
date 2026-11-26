# Project log: fine-tuning Whisper on Cypriot Greek

A running record of what was done, the decisions and why, the problems and how they were solved, and the
numbers. Source material for the README, the interview notes and a possible blog post. Newest phase last.

---

## 1. Data and splits

**What**
- Mozilla Common Voice Spontaneous Speech, Cypriot Greek (el-CY), SPS v5.0: 1,221 clips, 10.37 h,
  10 speakers. Spontaneous answers to prompts, so fillers, false starts and English words are common.
- Splits come with the corpus and are **speaker-disjoint**: train 5 speakers, dev 2, test 3.
  Dev (280 clips, 2.73 h) and test (371 clips, 2.89 h).

**Decisions**
- **Dev for every decision, test only once at the end.** Choosing settings by test score would quietly fit
  the test set and make the final number optimistic.
- Keep the corpus splits (speaker-disjoint) rather than re-splitting. Harder than the paper's setup
  (its splits were not speaker-independent), but it measures what matters: new speakers.

---

## 2. Scoring

**What**
- WER = (substitutions + deletions + insertions) / reference words, computed over the whole corpus
  (not averaged per clip). CER is the same on characters.
- A normalization ladder, each level adding one step:
  raw → basic (NFC, lowercase, final sigma, punctuation) → nofill (drop εεε/εμ fillers)
  → nodiac (drop accents and the Cypriot breve) → phonetic (collapse homophone spellings).

**Decisions**
- **Report basic WER as the headline**, nodiac and CER alongside.
- **phonetic is a diagnostic only, never a headline.** Greek homophones are different words: φύλο (gender),
  φίλο (friend) and φύλλο (leaf) sound the same but are written differently. A transcript that people will
  read must spell them correctly, so collapsing them would hide real errors.
- Report CER as well as WER: one wrong letter makes a whole word wrong in WER, so WER exaggerates near-misses
  (a lesson from the pilot with Greek medical terms).

---

## 3. Zero-shot baseline (no fine-tuning)

**What**
- `cyasr/run_baseline.py`: off-the-shelf Whisper on the full-length clips, long-form sequential decoding
  (Whisper slides over the audio in 30 s windows using its timestamps), Greek forced (`language: el`).
- Two decodings: **greedy** (always the most likely token) and **fallback** (re-decode a window at higher
  temperature when the output looks like a loop or has low confidence).

**Results (WER basic / CER nodiac)**

| model | dev greedy | dev fallback | test greedy | test fallback |
|---|---|---|---|---|
| whisper-small | 69.1 / 28.8 | 58.8 / 20.4 | 73.1 / 32.8 | 66.9 / 27.1 |
| whisper-medium | 51.2 / 17.3 | 50.3 / 16.4 | 63.2 / 24.8 | 58.3 / 21.9 |
| whisper-large-v3 | 47.4 / 16.7 | 47.3 / 16.6 | 53.3 / 19.2 | 52.6 / 18.5 |

Paper (Stamou et al. 2026) zero-shot: small 81.3, medium 69.0, large-v3 58.3 (different splits and test setup).

**Findings**
- **The main error is dialect → Standard Greek "correction"**: τζαι → και 405 times, dropped final -ν,
  single instead of double consonants. The model hears Cypriot but writes Standard Greek.
- **Fallback helps small the most** (dev 69.1 → 58.8): small falls into repetition loops more often.
- **Tail hallucinations**: clips of ~31-32 s leave an almost silent last window, and small invents text to
  fill it. large-v3 sometimes inserts "Υπότιτλοι AUTHORWAVE" (a subtitle credit from its training data).
- Digits vs number words ("2" vs "δύο") count as fully wrong. Postponed: digit normalization.

**Problems**
- **Results depended on the batch size.** Mixing short and long clips in one batch changed the padding,
  which changed the outputs, so models were not compared on equal terms. Fix: batch short (≤30 s) and long
  clips separately, always ask for timestamps. Verified identical output at batch size 1 and 4.

---

## 4. Segmentation (making training pieces of ≤ 30 s)

**Why**: Whisper takes at most 30 s per input, and many clips are longer. Cutting the audio is easy, but
the transcript has no times, so we don't know where to cut the text.

**What**
- **Forced alignment** (`cyasr/segment.py`): a CTC model (MMS, `MahmoudAshraf/mms-300m-1130-forced-aligner`)
  gets the audio and the known transcript and finds when each word is spoken. Greek is romanized with uroman
  first. Then the clip is cut at word boundaries into pieces of ≤ 30 s.
- train: 566 clips → 820 pieces (4.68 h); dev: 280 clips → 407 pieces (2.65 h). All pieces ≤ 30 s.
  Ran at 68× real time on a T4. Test is never segmented by this script.

**Decisions**
- **Run the aligner on Colab**, not locally: it needs a GPU-sized model and the audio was already there.
- **Alignment scores are used to choose what to listen to, not to filter.** The lowest-scoring piece
  (40425_1, score −2.11) sounded fine and its transcript was correct: low scores came mostly from
  disfluencies (false starts, fillers), which are real speech and should stay.

**Problems**
- The PyPI package `ctc-forced-aligner` 1.0.2 is a different (ONNX) fork. Installed the GitHub version,
  pinned to a tested commit.

---

## 5. Training targets

**Choice**: option (a): keep the **Cypriot spelling**, made consistent, and **keep fillers**.
Option (b), training on accent-free text, was set aside as a possible later experiment.

**Why**: the output should be readable Cypriot text. Fillers and false starts are in the audio, so removing
them from the targets would teach the model to skip audible speech.

**What** (`clean_target` in `cyasr/normalize.py`)
- The Cypriot mark was written three ways (breve, caron, Latin ž/š): unified to letter + breve (τζ̆).
- Stripped CSV quotes, HTML codes, bracket markers; unified apostrophes; moved accents typed before their
  vowel onto the vowel. 702 of 1,221 transcripts changed. 6 unit tests added.

---

## 6. Fine-tuning whisper-small with LoRA

**Setup** (`cyasr/train.py`, `configs/finetune_whisper-small.yaml`)
- **LoRA** (r 32, alpha 64): small trainable matrices added to the attention projections (q, k, v, out)
  and the feed-forward layers (fc1, fc2) of encoder and decoder; 13M of 255M
  parameters trained (5.1%). Adapters merged into the weights after training, so the result is a plain
  Whisper model.
- Learning rate 1e-4 (LoRA needs a higher rate than full fine-tuning, ~1e-5), 50 warmup steps, max 1,000
  steps, effective batch 16 (8 × 2 gradient accumulation), fp16.
- **Teacher forcing** with cross-entropy on the tokens; SpecAugment (random time and frequency masks)
  against overfitting.
- **Timestamps on half the pieces** (`timestamp_fraction 0.5`), so the model keeps the ability to predict
  timestamps, which long-form decoding needs.
- **Dev evaluation every 100 steps, early stopping** after 3 checks without improvement, best checkpoint
  chosen by dev WER (not dev loss).

**Decisions**
- **Start with small, then larger models**; LoRA first because it is cheaper.
- **Effective batch kept at 16 when moving to a bigger GPU.** Physical batch (what fits in memory) does not
  change the update; effective batch is a hyperparameter tied to the learning rate. Changing it would have
  mixed two changes and broken comparability with the first run. Effective batch size is a candidate
  experiment for later.
- Paper's setup for comparison: frozen encoder, LR 5e-6, effective batch 64, 2,048 steps, segmented test set.

**Problems**
- **Out of memory on the T4 at batch 16**, even with LoRA. LoRA saves memory on gradients and optimizer
  state, but the large cost is the **activations** stored for the backward pass (1,500 encoder frames,
  24 layers, all needed because gradients flow back to the adapters in the first layers). Fix: batch 8 ×
  gradient accumulation 2 (same update). Peak memory 11 GB.
- **peft crashed on Colab's preinstalled torchao 0.10** (peft requires > 0.16, though we don't use it).
  Fix: uninstall torchao in the notebook.
- **Free Colab disconnected at step ~470 and the run was lost**: checkpoints were on the Colab disk.
  Fix: write checkpoints straight to Google Drive and add `resume: true` to continue from the latest one.
  Tested in the smoke test (stop at step 4, resume to 6).
- **Evaluation was slower than training** (7 min per dev check vs 4 min per 100 steps on the T4).
  Fix: separate `eval_batch_size` 32 (generation stores no activations). 3.1× faster.
- **Copying 1,227 small files from Drive took 15+ min** (each file is a separate network request).
  Fix: one `segments.zip` (625 MB) on Drive.
- Moved to Colab Pro, L4 GPU: 1.39 s/step vs 2.5 s on the T4.
- The L4 run disconnected too after step 200 and was resumed from `checkpoint-200`.
- **Bug: `timing.json` is wrong after a resume** (the restored log contains the earlier evaluations, the
  runtime only the resumed part, giving a negative training time). Not fixed yet.

**Reproducibility check**: the L4 run reproduced the lost T4 run's training loss to 3 decimals
(2.754, 1.807, 1.268, 1.099) and dev WER at step 100 within 0.2 points (fixed seed, same effective batch).

**Training curve (dev = segmented pieces)**

| step | epoch | train loss | dev loss | dev WER basic | dev CER nodiac |
|---|---|---|---|---|---|
| 100 | 1.9 | 1.10 | 0.443 | 32.9% | 9.7% |
| 200 | 3.9 | 0.68 | 0.391 | 27.9% | 8.1% |
| 300 | 5.8 | 0.48 | 0.380 | 27.1% | 7.9% |
| 400 | 7.7 | 0.35 | 0.387 | 26.3% | 7.7% |
| **500** | 9.7 | 0.28 | 0.397 | **25.8%** | 7.6% |
| 600 | 11.7 | 0.22 | 0.408 | 26.1% | 7.7% |
| 700 | 13.6 | 0.19 | 0.419 | 26.0% | 7.6% |
| 800 | 15.5 | 0.17 | 0.430 | 26.1% | 7.7% |

- Most of the gain comes in the first 2 epochs: Whisper already knows Greek and only adapts to the dialect.
- Dev loss is lowest at step 300 but WER keeps improving until 500: the model gets more words right while
  becoming over-confident on its mistakes. That is why the best checkpoint is chosen by WER.
- Early stopping ended at step 800 and kept step 500.
- Time on the L4: ~1.39 s/step, ~2.2 min per dev evaluation; a clean 800-step run ≈ 37 min.

---

## 7. Fine-tuned vs zero-shot, fair comparison (dev, full-length clips)

Same script, same 280 clips, same long-form decoding and scoring as the baseline; only the model differs.

| decoding | model | WER basic | WER nodiac | CER nodiac | hallucinations |
|---|---|---|---|---|---|
| greedy | zero-shot small | 69.1% | 67.4% | 28.8% | 12 |
| greedy | fine-tuned small | 40.6% | 38.2% | 22.8% | 18 |
| fallback | zero-shot small | 58.8% | 56.9% | 20.4% | 3 |
| **fallback** | **fine-tuned small** | **31.2%** | **28.7%** | **12.0%** | 5 |

**Findings**
- **Dev WER 58.8 → 31.2% on unseen speakers** with 4.7 h of training audio. Fine-tuned small is far better
  than zero-shot large-v3 (47.3% dev).
- **With greedy decoding the fine-tuned model loops more than the zero-shot one** (18 vs 12 clips), which
  is why greedy CER improved so little. Fallback catches most loops. Likely cause: training only on ≤ 30 s
  pieces, half without timestamps, while long-form decoding chains several windows by timestamps.
  → **Use fallback decoding for the fine-tuned model.**
- Long-form is still worse than the segmented pieces (WER nodiac 28.7 vs 23.8, CER 12.0 vs 7.6): window
  boundaries, the remaining loops and the silent-tail problem.
- **Learned**: τζ̆αι instead of και, εν, final -ν, double consonants, Cypriot words (θκυο, έρκουνταν,
  δυσκολεύκουμαι, καλοτζ̆αίρκα, βουρήσω).
- **Remaining**: sound-alike consonants (ρούγου/ρούχου, σύννεβα/σύννεφα, κάφσονες/καύσωνες);
  **English code-switching** written in Greek letters ("standing order" → "σταν την όρτερ");
  elision variants (τζ̆' vs τζ̆αι).
- **Not yet comparable to the paper's 52.4%** (fine-tuned small): different splits (theirs not
  speaker-independent), unknown normalization, dev not test, and their test set was segmented.

---

## 8. Error analysis and the silent-tail fix (dev)

**Errors by clip length** (WER nodiac, fallback decoding):

| model | all | ≤ 30 s (128 clips) | 30-33 s (38 clips) | > 33 s (114 clips) |
|---|---|---|---|---|
| zero-shot | 56.9% | 56.3% | 67.4% | 55.3% |
| fine-tuned | 28.7% | 22.5% | **51.4%** | 27.7% |

**Finding: one problem dominated.** Clips that fit in one window are at 22.5% (about the same as the
segmented pieces during training, and identical for greedy and fallback). Clips of 30-33 s leave a last
window with well under a second of near-silence, and the fine-tuned model invents text for it: the first 30 s
are transcribed almost perfectly, then lines of made-up words follow. With greedy decoding 15 of the 18
looping clips are 30.2-32 s long.

**Why fine-tuning made it worse**: every training piece was full of speech, so the model never learned to
output nothing. The gap between 30-33 s and short clips grew from 11 points (zero-shot) to 29 (fine-tuned).

**Diagnosis before fixing**: rather than retraining straight away, first test the explanation with Whisper's
built-in no-speech rule, which is cheap (decoding only). For each 30 s window Whisper estimates how likely the
window has no speech (the probability of its `<|nospeech|>` token at the start); the window's text is dropped
when that probability is above 0.6 AND the text's average log-probability is below -1 (the model is unsure).
The second condition protects quiet but real speech. It is on by default in OpenAI's `whisper` package but off
by default in Hugging Face `transformers` (`no_speech_threshold=None`), which we use: a difference worth noting
when comparing with results obtained with OpenAI's package. Threshold 0.6 (OpenAI's default) was the only value
tried, to keep it a diagnosis and not tuning on dev.

**Result** (WER nodiac, fallback decoding):

| model | rule | all | ≤ 30 s | 30-33 s | > 33 s | insertions | hallucinations |
|---|---|---|---|---|---|---|---|
| zero-shot | off | 56.9% | 56.3% | 67.4% | 55.3% | 756 | 3 |
| zero-shot | 0.6 | 56.1% | 57.3% | 56.7% | 55.4% | 587 | 1 |
| fine-tuned | off | 28.7% | 22.5% | 51.4% | 27.7% | 1150 | 5 |
| **fine-tuned** | **0.6** | **26.1%** | 22.6% | **28.6%** | 27.5% | 719 | 1 |

- **Explanation confirmed**: for the fine-tuned model the 30-33 s clips drop from 51.4% to 28.6%, the other
  clips are unchanged, insertions fall by 431. Fine-tuned + rule: dev WER nodiac 26.1%, CER nodiac 9.4%.
- **Zero-shot gains little and loses 1 point on short clips**: it is often unsure on real Cypriot speech, so
  the "unsure" condition sometimes drops real words.
- 30-33 s and > 33 s clips stay ~5-6 points worse than short ones: the general cost of long-form decoding
  (sentences cut at window boundaries).
- **Decision**: keep the rule (0.6, fallback decoding) as the standard decoding for fine-tuned models, and
  move on to a bigger model. Training on silent windows (labelled `<|nospeech|>`) would fix the cause, but the
  rule already recovers most of the loss, so the expected gain (~1-2 points) is smaller than from a bigger model.

**Problem**: with the rule on, Hugging Face's long-form decoding returned the token ids as floats for some
batches (when windows were skipped), and the tokenizer crashed. Fix: convert the ids back to integers
(exact, since ids are < 52k).

**Other findings from the substitution list** (fine-tuned, nodiac):
- Most remaining word errors are small function words: τον ↔ το (final -ν, often barely audible), elided
  τζ' written as τζαι, στο → το (the σ merges into the previous sound), homophones των/τον and οι/η.
  The phonetic level (25.7%) vs nodiac (28.7%) puts homophone spelling at about 3 points of WER.
- Greeklish output in one clip ("xero na odigom"), probably from Whisper's internet pretraining data.
- Digits vs number words ("3 4" vs "τρία τέσσερα") still count as errors.
- English words written in Greek letters: "turnover" → ττερνοβερ, "standing order" → σταν την όρτερ.

---

## 9. Fine-tuning whisper-medium with LoRA (2026-11-13)

**Setup**: same recipe as small (LoRA r32/alpha64, LR 1e-4, warmup 50, max 1000 steps, eval every 100,
patience 3, seed 42, SpecAugment), so only the model size changes. Batch 4 x accumulation 4 = effective batch
16, the same as small (batch 8 would not fit). Config `configs/finetune_whisper-medium.yaml`, notebook 05.
LoRA trains 34.6M of 798M parameters (4.3%; small: 13M). Training pieces are copied from `segments.zip`
(seconds, instead of 15+ min for the 1,227 single wavs).

**Training** (L4, dev = segmented pieces, as in section 6):

| step | train loss | dev loss | WER basic | WER nodiac | CER nodiac |
|---|---|---|---|---|---|
| 100 | 1.429 | 0.298 | 20.6% | 18.2% | 5.4% |
| 200 | 0.739 | **0.283** | 18.2% | 16.2% | 4.9% |
| 300 | 0.458 | 0.287 | 17.9% | 15.6% | 4.8% |
| 400 | 0.303 | 0.303 | 17.5% | 15.4% | 4.7% |
| 500 | 0.222 | 0.331 | 17.3% | 15.2% | 4.7% |
| 600 | 0.168 | 0.345 | 17.4% | 15.3% | 4.8% |
| 700 | 0.140 | 0.359 | 17.6% | 15.5% | 5.1% |
| 800 | 0.112 | 0.369 | 17.1% | 14.9% | 4.6% |
| **900 (best)** | 0.110 | 0.376 | **17.0%** | 14.8% | 4.6% |
| 1000 | 0.108 | 0.377 | 17.1% | 14.8% | 4.5% |

- **Medium at step 100 already beats small's best** (25.8% WER basic at step 500).
- **No early stop**: WER kept making tiny new bests (800, 900), so patience 3 never ran out. From step 400 on,
  WER stays within 17.0-17.6%, which is noise on ~410 pieces: stopping at ~500 would have saved ~40 min.
- **Dev loss rises from step 200 while WER stays flat**: the model gets the same words right but becomes
  overconfident when wrong (loss punishes confidence, WER does not). Same pattern as small.
- **Training loss is not comparable with dev loss or across models** (unverified hypothesis): at step 100 it is
  ~4.8x the dev loss with accumulation 4, and ~2.7x for small with accumulation 2, which fits the logged
  training loss being multiplied by the accumulation steps. It would not affect training (AdamW and gradient
  clipping remove a constant scale), only how the number reads. To check later.
- **Cost**: 131 min in total, of which evaluation 71 min and training 61 min (3.6 s/step); peak GPU memory
  17.1 GB of 24. No resume, so `timing.json` is valid for this run. The evals during training cost more than
  the training: for later runs, evaluate every 200 steps or on a dev subset.
- My speed estimate from the smoke test (2.4 s/step, 4.5 min per eval) was too optimistic: the smoke test's
  ratio to small did not carry over to the real run.

**Fair evaluation** (full-length dev clips, 280 clips, fallback decoding; notebook 06, config
`configs/eval_whisper-medium_lora.yaml`). Small from sections 7-8, for comparison:

| model | rule | WER basic | WER nodiac | CER nodiac | hallucinations |
|---|---|---|---|---|---|
| zero-shot small | off | 58.8% | 56.9% | 20.4% | 3 |
| zero-shot small | 0.6 | 58.0% | 56.1% | 20.0% | 1 |
| fine-tuned small | off | 31.2% | 28.7% | 12.0% | 5 |
| fine-tuned small | 0.6 | 28.6% | 26.1% | 9.4% | 1 |
| zero-shot medium | off | 50.3% | 48.2% | 16.4% | 2 |
| zero-shot medium | 0.6 | 49.7% | 47.7% | 16.0% | 1 |
| fine-tuned medium | off | 20.9% | 18.3% | 7.9% | 5 |
| **fine-tuned medium** | **0.6** | **19.9%** | **17.3%** | **6.8%** | 2 |

By clip length (medium, WER nodiac):

| model | rule | all | ≤ 30 s | 30-33 s | > 33 s | insertions | hallucinations |
|---|---|---|---|---|---|---|---|
| zero-shot | off | 48.2% | 47.6% | 53.5% | 47.6% | 456 | 2 |
| zero-shot | 0.6 | 47.7% | 47.7% | 49.7% | 47.4% | 386 | 1 |
| fine-tuned | off | 18.3% | 13.6% | 39.4% | 16.8% | 786 | 5 |
| fine-tuned | 0.6 | 17.3% | 13.6% | 28.7% | 17.1% | 623 | 2 |

- **Fine-tuning cuts medium's WER by 60% relative** (49.7% → 19.9% basic) and medium ends ~9 points below
  small (28.6%). Medium starts from a better zero-shot model and gains more from the same data.
- **The silent tail is now the main weakness**: with the rule, ≤ 30 s clips are at 13.6% but 30-33 s clips at
  28.7%, more than double (small: 22.6% vs 28.6%). The rule still helps (39.4% → 28.7%) but fixes only part
  of it: the better the model, the more these tails dominate what is left.
- The rule changes little for zero-shot medium, and nothing beyond noise for > 33 s clips, as with small.
- **Not comparable with the paper yet**: our 19.9% is far below the paper's whisper-medium 37.85%, but splits,
  normalization and segmented vs full clips differ. A gap this large is a reason to double-check our setup
  before claiming anything; the segmented test set (open items) is what makes a fair comparison.

**Problem: a run that looked hung.** The third evaluation (zero-shot medium + rule) showed no progress past
272/280 for 30+ min, while the first two runs took 10 min each, and then the VS Code connection to Colab
dropped. Its results turned out to be complete on Drive: the run had kept going on Colab while the output in
VS Code stopped updating. Lesson: check Drive for `metrics.json` before assuming a run died. Zero-shot is
still plausibly slower (its repetition loops trigger more temperature-fallback retries), but this run's real
duration is unknown.

---

## 10. Zero-shot Qwen3-ASR-0.6B and Reson8 Rumble-2-Flash (2026-11-20)

**What**: two more zero-shot baselines on dev, through the same `cyasr.run_baseline` script and scoring.
- `Qwen/Qwen3-ASR-0.6B-hf` (Apache-2.0; Greek is one of its 30 languages).
- `Reson8/Rumble-2-Flash`: Reson8's own fine-tune of Qwen3-ASR-0.6B (Apache-2.0, 45 languages incl. Greek),
  released 2026-11-19. Same architecture, its own tokenizer, plus control options in the chat template.
- New backend `cyasr/transcribe_qwen.py` (`backend: qwen3_asr` in the config), configs
  `baseline_qwen3-asr-0.6b.yaml`, `baseline_rumble-2-flash.yaml`, `baseline_rumble-2-flash_matched.yaml`,
  notebook 07.

**Decisions**
- **Native transformers implementation, not the `qwen-asr` package.** The package pins transformers 4.57.6, which
  would downgrade the rest of the project (5.22). transformers 5.22 has `Qwen3ASRForConditionalGeneration`, and Qwen
  publishes converted `-hf` checkpoints. Both model revisions are pinned by commit.
- **Whole clips in one pass.** Qwen3-ASR takes up to 20 min of audio (our longest clip is 271 s), so there are no
  30 s windows: no silent-tail problem and no sentences cut at window boundaries. This is a real difference in
  design, not only in model quality, so compare by clip length as well (≤ 30 s clips are the fairest comparison).
- **Greedy decoding, Greek forced** (prefill `language Greek<asr_text>`), like `language: el` for Whisper.
  There is no temperature fallback or no-speech rule for this backend.
- **Rumble-2-Flash runs twice**: product defaults (numbers as digits, fillers dropped, repeats kept) and options
  matched to our references (`digits: false, filler: true`). Our references write numbers as words and keep
  fillers, so the defaults lose points for convention alone; the gap between the two runs measures that.
  Also a talking point: the customisation is exactly what Reson8 sells.
- The processor's output parser removes long repetitions (part of the official pipeline). The run keeps both
  `hyp` (official) and `hyp_raw` (before clean-up) to see how often it matters.
- `max_new_tokens` 4096 (the checkpoint default is 512): Greek costs ~5.4 Qwen tokens per word on dev, and the longest
  dev reference is ~1,530 tokens. Cut-off outputs are counted (`hit_max_tokens` in `metrics.json`).
- dtype bfloat16 on L4/A100, float32 on T4 (no fp16: the model was trained in bf16).

**Checked locally** (no model loaded, the laptop has too little free RAM): the processor builds the right prompt for
a batch of a 13 s and a 143 s clip (left padding, Greek prefill, 100 mel frames per second), and Rumble's
`digits`/`filler` options reach its system prompt (`<nodigits><filler><verbatim>`).

**Problem: batch size changed the output** (Colab L4, 8 dev clips, Qwen config):

| check | identical clips |
|---|---|
| bf16, batch 1 vs batch 8 | 0/8 |
| bf16, batch 1, run twice | 8/8 |
| fp32, batch 1 vs batch 8 | 7/8 (the 8th differs only after 150+ characters) |
| fp32 vs bf16, both batch 1 | 1/8 |

Padding is masked correctly (fp32 nearly agrees), so unlike Whisper's batch problem (section 3) this is not a
padding bug: bf16 rounds differently for different batch shapes, and greedy decoding turns a tiny difference
into a different word. **Fix: batch size 1** (deterministic, and what a single-clip user gets). Cost: speed,
~50 min for Qwen and ~28 min for each Rumble run on the 280 dev clips. Precision alone changes 7 of 8 outputs,
so these models' scores carry some numerical noise; a fair cross-model comparison should use the same dtype
(all runs here: bf16).

**Results** (dev, 280 full-length clips, same scoring; Whisper rows from section 3):

| model | params | decoding | WER raw | WER basic | WER nodiac | WER phonetic | CER nodiac |
|---|---|---|---|---|---|---|---|
| whisper-small | 0.24B | fallback | 64.4% | 58.8% | 56.9% | 52.6% | 20.4% |
| whisper-medium | 0.77B | fallback | 57.1% | 50.3% | 48.2% | 44.3% | 16.4% |
| whisper-large-v3 | 1.55B | greedy | 54.1% | 47.4% | 45.8% | 42.8% | 16.7% |
| whisper-large-v3 | 1.55B | fallback | 53.9% | 47.3% | 45.7% | 42.7% | 16.6% |
| Qwen3-ASR-0.6B | 0.6B | greedy | 76.6% | 73.1% | 71.5% | 68.7% | 28.9% |
| Rumble-2-Flash, defaults | 0.6B | greedy | 54.1% | 47.7% | 46.2% | 43.0% | 15.0% |
| **Rumble-2-Flash, matched** | 0.6B | greedy | 53.8% | **47.3%** | 45.9% | 42.8% | **14.6%** |

By clip length (WER nodiac) and error types:

| model | all | ≤ 30 s | 30-33 s | > 33 s | CER ≤ 30 s | insertions | deletions | substitutions |
|---|---|---|---|---|---|---|---|---|
| whisper-large-v3 fallback | 45.7% | **45.4%** | 52.2% | 44.6% | 17.0% | 306 | 1000 | 6116 |
| Qwen3-ASR-0.6B | 71.5% | 73.1% | 72.1% | 70.6% | 29.6% | 265 | 2884 | 8473 |
| Rumble-2-Flash matched | 45.9% | 47.1% | **46.5%** | 45.2% | **15.3%** | 345 | 766 | 6349 |

**Findings**
- **Base Qwen3-ASR-0.6B is the weakest model tested**, worse even than zero-shot whisper-small (73.1% vs 58.8% basic).
  It glues words together and drops many (2,884 deletions, 2-3x the others): "εν παίζω κανέναν μουσικόν
  όργανον" → "εμπεζοκανένας μουσικονοργανον". Greek is in its language list but clearly a weaker language for it.
- **Reson8's fine-tune changes that completely**: same architecture and size, 73.1% → 47.3% WER basic, CER 28.9% →
  14.6%. Rumble-2-Flash (509M, see section 11) matches whisper-large-v3 (1.55B, 3x bigger) on WER and has the **best zero-shot
  CER of all models** (14.6% vs 16.6%): its errors are closer misses, more often a wrong letter than a wrong word.
- **No silent-tail problem**: Rumble's 30-33 s clips (46.5%) are as good as the rest, while every Whisper model is
  ~7 points worse on them. On clips that fit in one window (≤ 30 s), large-v3 is still ahead on WER (45.4% vs
  47.1%) but behind on CER (17.0% vs 15.3%). So Rumble's overall tie with large-v3 partly comes from whole-clip
  decoding, not only from better recognition.
- **Matching the conventions barely matters here** (defaults 47.7% → matched 47.3%). With defaults, 82 of 280
  outputs contain digits (references: 9); with `digits: false`, none. Few numbers are spoken in this data, so the
  effect is small, but the option works exactly as documented.
- **Same main error as Whisper: dialect → Standard Greek.** Top substitutions (Rumble matched, nodiac): τζαι → και
  468 times, εν → είναι 87, μιαν → μια 67, τες → τις 61, που → από 58 (Cypriot "from"), εναν → ενα 48,
  έσει → έχει 40, πράματα → πράγματα 39, εννα → να 37. It does produce some Cypriot forms (μιτσής, μπουζούκκιν,
  δυσκολεύκομαι, έρκουνταν, εκαθούμασταν). Zero-shot, no model writes Cypriot reliably; fine-tuning is what
  fixes this (sections 7 and 9).
- Rumble sometimes wraps its output in « » quotes (removed by `basic`, so only raw WER is affected) and writes
  English in Greek letters: "standing order" → «σταντινόρτερ» (large-v3 keeps "Standing order").
- No cut-off outputs (longest: 1,431 tokens, well under 4,096), no empty outputs, 1 hallucination each. The
  processor's repetition clean-up changed 1 clip out of 840.

**Next**: fine-tune Rumble-2-Flash with LoRA, same data and recipe as Whisper. It starts at large-v3's level with a
third of the parameters and no 30 s windows, so it is the most promising candidate; fine-tuned whisper-medium
(19.9% WER basic) is the number to beat. Native transformers supports training (the processor builds labels with
`output_labels=True`); `cyasr/train.py` is Whisper-only and needs a Qwen path.

## 11. Fine-tuning Rumble-2-Flash with LoRA (2026-11-22)

**What**: a Qwen3-ASR training path, `cyasr/train_qwen.py`, called by `cyasr/train.py` when the config has
`backend: qwen3_asr`; configs `finetune_rumble-2-flash.yaml` (training) and `eval_rumble-2-flash_lora.yaml`
(fair evaluation, identical to the zero-shot matched run); notebook 08 (training, then evaluation, in the background,
then the GPU is released).

**How a training example is built**: Qwen3-ASR is a decoder-only language model fed with audio embeddings, so the
example is one sequence: system prompt with Rumble's control tokens, user turn with the audio, then the assistant turn
`language Greek<asr_text>` + target text + `<|im_end|>`. The prompt comes from the same processor call as at
inference (`apply_transcription_request`, Greek forced, `digits: false, filler: true, verbatim: true`), so training and
evaluation see identical input. **Only the target text and `<|im_end|>` are trained** (labels -100 on the prompt and
the audio): the model should learn to transcribe, not to predict its own prompt.

**Decisions**
- **Same recipe as Whisper** where it carries over: LoRA r32/alpha64/dropout 0.05, LR 1e-4, warmup 50, max 1000 steps,
  effective batch 16 (4 × 4, see the smoke test below), dev WER every 100 steps, patience 3, seed 42, best checkpoint by dev WER basic,
  SpecAugment with Whisper's settings (time and mel-band spans of 10 set to zero, probability 0.05).
- **LoRA targets**: attention (q, k, v, out/o) and feed-forward layers of the 18-layer audio encoder and the 14-layer
  decoder; not the convolutions, the audio-to-text projector or the output layer. The same kinds of layers as for
  Whisper.
- **Dev WER during training is a real transcription score**: evaluation generates from the prompt alone, while dev
  loss uses the full sequence (a custom `prediction_step`). Without this, generation would start after the target.
- **fp32 weights with bf16 autocast** (Whisper used fp16 on the T4); the merged model is saved in bf16, like the
  released checkpoints. No timestamps (Qwen3-ASR has none), so `timestamp_fraction` does not apply.
- Rumble has **509M parameters**, base Qwen3-ASR-0.6B about 780M: Rumble is a compressed Qwen3-ASR with the same
  18-layer audio encoder but a **14-layer text decoder instead of 28** and a pruned vocabulary (99.6k vs 152k tokens;
  Greek tokenization is unchanged, 5.30 tokens per word on our corpus for both).

**Tested on the laptop's CPU** with a tiny randomly initialised Qwen3-ASR (2 + 2 layers, Rumble's real tokenizer and
prompt), because the real model does not fit in the free RAM: 4 LoRA steps run, evaluation generates, LoRA is merged,
`best/` is saved and loads in `run_baseline`'s Qwen backend. Starting loss 11.5 = ln(99,631), as expected for a
random model; overfitting 2 pieces for 60 steps reproduced the first transcript exactly. Not tested yet: memory and
speed of the real model on the L4 (the smoke test in notebook 08 checks both).

**Smoke test on the L4** (4 steps, 8 dev pieces): runs end to end; 19.4M trainable LoRA parameters (3.67% of 528M
with the adapters); example target `Για να φτάσει... α, σόρι.<|im_end|>`. With batch 8 × 2 peak memory was
**20.4 of 23 GB**, too little margin for batches of long pieces, so the real run uses **batch 4 × grad_accum 4**
(same effective batch 16) and eval batch 8. Even so, the allocator hit two near-OOM moments during dev evaluation
(it freed its cache and carried on). Speed: ~2.1 s per training step, ~11 min per dev evaluation (407 pieces,
generation), so ~14 min per 100 steps.

**First dev check** (step 100, segmented pieces): WER basic 19.2%, nodiac 17.3%, CER nodiac 5.5%, dev loss 0.17.
Started 20:15 UTC, finished 23:10 UTC; the GPU was released automatically.

**Training curve** (dev = 407 segmented pieces; train loss = mean over the preceding steps)

| step | train loss | dev loss | WER basic | WER nodiac | CER nodiac |
|---|---|---|---|---|---|
| 100 | 0.177 | 0.172 | 19.2% | 17.2% | 5.5% |
| 200 | 0.082 | 0.176 | 17.5% | 15.5% | 5.0% |
| 400 | 0.022 | 0.217 | 17.3% | 15.4% | 5.2% |
| 600 | 0.010 | 0.247 | 16.9% | 15.0% | 5.0% |
| 800 | 0.007 | 0.260 | 16.7% | 15.0% | 5.0% |
| 1000 | 0.005 | 0.263 | 16.5% | 14.7% | 4.9% |

Most of the gain comes in the first 200 steps (~19 epochs in total over 820 pieces). Dev loss rises from step 200
while dev WER keeps falling slowly: the model becomes over-confident on the tokens it gets wrong (higher loss) but
its greedy choices still improve. Best checkpoint = step 1000, the last one, so early stopping never triggered;
more steps might still help a little. Timing: 34 min of training (2.0 s/step), **110 min of dev evaluation**
(10 × 11 min of generation), peak memory 13.1 GB with batch 4. Lesson: for the next run, evaluate every 200 steps
or on a fixed subset of dev pieces.

**Fair evaluation** (280 full-length dev clips, one pass per clip, greedy, batch 1; same as all other rows)

| model | WER basic | WER nodiac | CER nodiac | ≤30 s | 30-33 s | >33 s | halluc. |
|---|---|---|---|---|---|---|---|
| zero-shot whisper-medium | 50.3% | 48.2% | 16.4% | 47.6% | 53.5% | 47.6% | 2 |
| fine-tuned whisper-medium + no-speech rule | 19.9% | 17.3% | 6.8% | 13.6% | 28.7% | 17.1% | 2 |
| zero-shot Rumble-2-Flash (matched prompt) | 47.3% | 45.9% | 14.6% | 47.1% | 46.5% | 45.2% | 1 |
| **fine-tuned Rumble-2-Flash** | **17.3%** | **15.0%** | **5.2%** | 14.8% | 15.0% | 15.1% | 1 |

(clip-length columns: WER nodiac.) Analysis: `scripts/compare_finetuned.py`, report saved as
`runs/rumble-2-flash_lora_greedy_dev/compare_vs_whisper-medium_lora.txt`.

**Findings**
- **Fine-tuned Rumble beats fine-tuned whisper-medium by 2.6 points WER basic** (17.3% vs 19.9%; nodiac 15.0% vs
  17.3%; CER 5.2% vs 6.8%), with a smaller model (509M vs 769M) and no decoding rules. A paired bootstrap over clips
  (10,000 resamples) gives a difference of -2.6 points, 95% CI [-4.0, -1.3]; Rumble is better on 147 clips,
  whisper-medium on 101, tie on 32.
- **The win comes from long clips, not from better hearing on short ones.** On clips ≤30 s whisper-medium is
  slightly better (13.6% vs 14.8%, CI of the difference [0.0, +2.3] points). On 30-33 s clips Rumble is far better
  (15.0% vs 28.7%), on >33 s clearly better (15.1% vs 17.1%, CI [-3.3, -0.7]). Rumble's error rate is flat across
  lengths because it reads the whole clip in one pass; Whisper's 30 s windows still cost it on clips just over 30 s,
  even with the no-speech rule. Error types agree: whisper-medium has 623 insertions vs 329 (repeated or invented
  text at window boundaries); Rumble has slightly more deletions (267 vs 219).
- **Fine-tuning removes most of the gap to the reference spelling**: zero-shot 47.3% → 17.3%. Dialect forms are now
  produced: τζαι 707 times (reference 670), και only twice (reference once), where zero-shot Rumble wrote και for
  τζαι 468 times (section 10). Part of the 30-point gain is therefore learning the corpus's orthography, not
  better hearing; the phonetic level (13.4% vs 15.2% for whisper-medium) shows the remaining errors are mostly
  real mishearings and morphology.
- **Remaining errors are shared by both models**: τζ/τζι vs τζαι (the reference keeps elided forms), το/τον
  (Cypriot final ν), οι/η, αθθυμούμαι vs θθυμούμαι, μιτσ̆ιοί vs μιτσ̆οί, εννά vs να. These are dialect
  morphology and the corpus's elision conventions, where the reference itself is a choice.
- **New failure mode**: on a few long clips Rumble skips a stretch or drifts (clip 46996: drops "καλός
  άνθρωπος ... Επίσης" and continues with a later part; clip 47586: 30 vs 18 errors). Code-switching is kept
  (English words in 27 hypotheses vs 30 references), with a few mixed-script words (Vizantinόν, Όταan).
- Caveat: one seed, dev only. The test set is still untouched.

---

## 12. Fine-tuning base Qwen3-ASR-0.6B with LoRA (2026-11-23)

**Why**: Rumble-2-Flash is a compressed Qwen3-ASR-0.6B (same 18-layer audio encoder, 14 instead of 28 decoder layers,
pruned vocabulary; section 11). Fine-tuning the original with the same recipe asks whether Rumble's half-depth decoder
(and Reson8's own training) costs or gains accuracy once both are adapted to Cypriot.

**Setup**: `configs/finetune_qwen3-asr-0.6b.yaml` / `eval_qwen3-asr-0.6b_lora.yaml`; recipe identical to Rumble
(LoRA r32/alpha64, LR 1e-4, 1000 steps, effective batch 16, dev check every 100 steps, patience 3, seed 42,
SpecAugment), plain Qwen prompt (no control tokens). 29.5M trainable LoRA parameters of 812M (Rumble: 19.4M of 528M).
Memory: the smoke test peaked at 19.5 GB with batch 4, so **batch 2 × grad_accum 8** and eval batch 4. Speed:
3.8 s/step (Rumble 2.0), dev check ~38 min (Rumble 11 min), with repeated near-OOM moments during generation. The
slower checks are explained by the doubled decoder and the halved eval batch (102 batches instead of 51), not by
runaway outputs: the final evaluation had no output cut off at the token limit.

**Step 100** (segmented dev pieces): WER basic 38.5%, nodiac 36.2%, CER nodiac 13.9%, dev loss 0.26; train loss
0.27. Rumble at the same step: 19.2% / 17.2% / 5.5%. Started 00:26 UTC, finished 08:49 UTC (7.4 h, of which
6.4 h dev checks); GPU released automatically.

**Training curve** (segmented dev pieces)

| step | train loss | dev loss | WER basic | WER nodiac | CER nodiac |
|---|---|---|---|---|---|
| 100 | 0.267 | 0.261 | 38.5% | 36.2% | 13.9% |
| 200 | 0.126 | 0.246 | 32.7% | 30.4% | 11.6% |
| 400 | 0.033 | 0.292 | 32.2% | 30.2% | 11.8% |
| 600 | 0.014 | 0.343 | 31.9% | 30.0% | 11.6% |
| 900 | 0.009 | 0.355 | **30.9%** | 29.0% | 10.7% |
| 1000 | 0.008 | 0.357 | 31.8% | 29.9% | 11.8% |

Plateau from step 200; best checkpoint step 900. Same pattern as Rumble (dev loss rising while WER creeps down),
but at about twice Rumble's error level throughout.

**Fair evaluation** (280 full-length dev clips, greedy, batch 1; length columns WER nodiac)

| model | decoder layers | WER basic | WER nodiac | CER nodiac | phonetic WER | ≤30 s | 30-33 s | >33 s |
|---|---|---|---|---|---|---|---|---|
| zero-shot Qwen3-ASR-0.6B | 28 | 73.1% | 71.5% | 28.9% | | 73.1% | 72.1% | 70.6% |
| **fine-tuned Qwen3-ASR-0.6B** | 28 | **31.6%** | 29.3% | 11.4% | 27.2% | 27.8% | 26.8% | 30.5% |
| zero-shot Rumble-2-Flash | 14 | 47.3% | 45.9% | 14.6% | | 47.1% | 46.5% | 45.2% |
| **fine-tuned Rumble-2-Flash** | 14 | **17.3%** | 15.0% | 5.2% | 13.4% | 14.8% | 15.0% | 15.1% |
| fine-tuned whisper-medium + rule | 24 | 19.9% | 17.3% | 6.8% | 15.2% | 13.6% | 28.7% | 17.1% |

Report: `runs/qwen3-asr-0.6b_lora_greedy_dev/compare_vs_rumble-2-flash_lora.txt`.

**Findings**
- **The compressed model wins by a wide margin**: fine-tuned Qwen is 14.3 points worse than fine-tuned Rumble (95% CI
  [+13.2, +15.3]); Rumble is better on 257 of 280 clips, Qwen on 10. Fine-tuning closed only part of the zero-shot gap
  (26 points → 14). Halving the decoder clearly did not cost accuracy here; whatever Reson8 did to the model
  (presumably training on more Greek / European speech, plus distillation) gives a far better starting point for a
  new dialect.
- **Both learned the corpus's orthography equally well**: τζαι 704 (Qwen) vs 707 (Rumble) vs 670 in the reference;
  και 7 vs 2. So Qwen's extra errors are not about dialect spelling conventions but about hearing and spelling Greek
  words: phonetic WER is still 27.2% vs 13.4%, i.e. the gap survives when all spelling differences of the same
  sound are ignored. Examples: επαγγελματικής → απαντηγελματικής, ευρωεκλογές → ευρευργικές, αθθυμούμαι → αφημούμαι.
- Like Rumble, it has no long-clip penalty (one pass per clip): 27.8% / 26.8% / 30.5% across lengths.
- What this cannot separate: Rumble differs from Qwen in training data, compression (14 layers, pruned vocabulary)
  and possibly distillation all at once; the card does not document the recipe. The fair statement is "Reson8's
  model is a much better base for Cypriot than its parent", not why.
- Cost: Qwen trains at half Rumble's speed and needs half the batch, for twice the error.

## 13. How Rumble's vocabulary was trimmed; FLEURS checks set up (2026-11-23)

**What**: `scripts/tokenizer_diff.py` compares the tokenizer files of Qwen3-ASR-0.6B and Rumble-2-Flash (revision
95bd5f9). No weights are needed. Full output: `docs/tokenizer_diff.md`.

**Findings**
- **Pure deletion, nothing new learned**: Rumble keeps 99,562 of Qwen's 151,643 BPE tokens, adds none, and keeps the
  kept tokens in their original order, renumbered 0..99,561. Its 99,306 merge rules are a subset of Qwen's, in Qwen's
  order. The trimmed vocabulary is consistent: every kept token can still be built from kept parts. All 256 byte
  tokens are kept, so any text can still be written, only with more tokens.
- **Chosen by usage, not by language**: removed tokens are spread over every script. Latin is 46% of all removed
  tokens: code pieces (`#include`, `\tif`, `(self`), whitespace runs (97% removed) and symbol/punctuation tokens (96%).
  Writing systems of languages outside the 45 are removed almost completely (Ethiopic, Tibetan, Lao, Myanmar 100%;
  Georgian 95%). Scripts of listed languages keep most tokens (Cyrillic 96%, Hiragana 99%), while CJK loses 53% and
  Thai 62%. Rare letters that listed languages need are kept, such as Maltese Ħ ċ, Latvian Ļ Ņ and Romanian Ț. The
  pattern fits "tokenize our multilingual training text and keep every token that occurs", not a list of blocked
  scripts. The model cannot learn this during training, because the embedding rows are deleted beforehand.
- **Unchanged for most languages**: the same sentence costs exactly the same number of tokens in Greek, German, Dutch,
  Russian, Arabic, Hindi, Chinese, Japanese, Korean and Thai. It costs more only in languages Rumble does not list:
  Hebrew +19%, Armenian +11%, Georgian +53%.
- **Greek was already character-level in Qwen**: Qwen has only 245 Greek tokens, and the only multi-letter ones are
  ου and αι. Rumble removed 148 of them. A sample of the removed ones is all polytonic or archaic forms (Ἦ, ᾧ, ϗ),
  and every modern Greek letter, accented or not, upper or lower case, is still a single token. So ~5.3 tokens per word for Greek
  (section 11) is inherited from Qwen, not caused by trimming. A Greek-aware tokenizer could cut decoder steps by
  about 3-4×, a possible idea of my own to mention.
- **Two additions the card does not mention**:
  1. Six new special tokens, `<digits> <nodigits> <filler> <nofiller> <verbatim> <noverbatim>`. These are the
     transcription options, trained in as control tokens.
  2. An auxiliary **CTC head** (`ctc_head.py`, `ctc/ctc_head.safetensors`): one linear layer, 896 → 99,631,
     on the audio encoder output, trained jointly with the decoder ("decoder + CTC loss"). It gives a fast
     encoder-only greedy transcript. Its docstring says it is "much less accurate" than normal decoding. Joint CTC
     training probably also helps the encoder, but that is not tested here.

**FLEURS checks**: notebook `09_fleurs_multilingual_colab.ipynb` has two parts.
- (A) Zero-shot Qwen vs Rumble, 100 test clips each in 12 languages, chosen from the diff above. This tests whether
  the compression hurt other languages.
- (B) The forgetting check from the open items: FLEURS `el_gr` test, zero-shot vs LoRA for whisper-medium, Rumble
  and Qwen, with a paired bootstrap and counts of Cypriot forms leaking into the output.

Results: section 15.

---

## 14. The paper's evaluation protocol (Stamou et al. 2026, read 2026-11-23)

Read from the PDF (DialRes 2026, pp. 308-314), to decide how to compare fairly.

- **Data**: "version 23.0" of Common Voice Spontaneous Speech el-CY, 1,284 clips, 10.92 h, 10 speakers (ours: SPS
  v5.0, 1,221 clips, 10.37 h, 10 speakers). Transcripts were "graphematically homogenized" by two linguists.
- **Split**: their own random **80/20 train/evaluation split, not speaker-independent** (stated in section 4). No
  separate dev set is described; results are on this "test set". No code, split or segment lists released.
- **Segmentation**: forced alignment with ctc-forced-aligner and `mms-300m-1130` (the same aligner as our
  `cyasr/segment.py`), 30 s windows with 2 s context, sentence-level pieces, pieces > 30 s removed. WER is
  reported on these pieces.
- **Normalization** (7 rule-based steps, applied to training and evaluation text): lowercase + NFC; quotes and
  apostrophes unified, dashes removed, whitespace collapsed; noise markers, URLs etc. removed, laughter mapped to a
  placeholder; elongations collapsed ("εεεεν" → "εν"); polytonic → monotonic; Latin letters in Greek words mapped
  to Greek; Cypriot variants mapped to canonical Cypriot forms (τζαι, τζε → τζ̆αι). Normalized vs original text
  differ by 15.8% WER / 3.8% CER. **Not stated: whether punctuation is removed before scoring**, and the WER tool.
- **Training**: full decoder fine-tuning with frozen encoder, LR 5e-6, effective batch 64, 2,048 steps, max 30 s,
  225 output tokens, fp16, gradient checkpointing. Decoding settings not given.
- **Results (test, WER / CER)**, normalized transcripts, before → after fine-tuning: small 81.27 → 52.38 / 32.33,
  medium 69.03 → 37.85 / 26.71, large-v3 58.33 → **33.93** / 21.15, large-v3-turbo 61.93 → 38.12 / 22.24;
  XLS-R-greek 83.14 → 46.96 / 17.88. On original transcripts medium reaches 43.64, large-v3 ~35.7.

**What this means for us**
- An exact reproduction is impossible (different corpus version, random non-released split). The closest fair
  comparison: our **speaker-disjoint test set, segmented the same way** (same aligner, ≤ 30 s pieces), scored at
  several normalization levels, next to our full-clip test numbers. Our split is harder (unseen speakers), so a
  lower WER on it is conservative evidence, not proof.
- Their zero-shot WERs are 10-22 points higher than ours (e.g. large-v3 58.3 vs our 47.3 on dev). A likely cause is
  scoring with punctuation kept (our "raw" level is far above "basic"); unverified.
- `cyasr/segment.py` refuses the test split on purpose. Segmenting test (alignment uses the reference text, which is
  allowed for evaluation) needs an explicit flag, used once, at the final evaluation.
- Their CERs (21-33%) are high compared with their WERs, unlike ours; another sign that the scoring differs.

## 15. FLEURS results: other languages and forgetting on Standard Greek (2026-11-23)

**Setup**: notebook 09 run on a Colab **T4**, so the Qwen backend ran in float32. Code is `cyasr_code.zip`
(md5 e57edda…, the same as the local code). FLEURS test audio was streamed from the Hub, first N clips in tar order.
Scoring is `basic` level, corpus-level; CER without spaces for zh/ja/th. CIs are a paired bootstrap, 10,000
resamples, seed 42. Results on Drive: `MyDrive/cyasr/runs_fleurs/` (predictions, metrics, `summary_part_a.json`,
`summary_part_b.json`).

**A. Zero-shot Qwen3-ASR-0.6B vs Rumble-2-Flash, 100 clips per language** (Δ = Rumble − Qwen; negative = Rumble better)

| language | metric | Qwen | Rumble | Δ | 95% CI | decode time Qwen / Rumble (s) |
|---|---|---|---|---|---|---|
| Greek | WER | 52.9% | 13.5% | −39.4 | [−42.9, −36.0] | 491 / 269 |
| Polish | WER | 27.6% | 8.5% | −19.1 | [−21.8, −16.5] | 185 / 102 |
| Hindi | WER | 18.5% | 13.0% | −5.4 | [−7.5, −3.5] | 451 / 254 |
| Dutch | WER | 14.1% | 9.2% | −5.0 | [−6.7, −3.3] | 169 / 94 |
| Arabic (Egypt) | WER | 21.0% | 17.2% | −3.8 | [−5.7, −1.9] | 162 / 94 |
| Vietnamese | WER | 9.7% | 6.7% | −3.0 | [−4.5, −1.5] | 156 / 84 |
| German | WER | 6.4% | 5.0% | −1.4 | [−2.6, −0.3] | 184 / 102 |
| Russian | WER | 9.0% | 7.7% | −1.2 | [−2.9, +0.3] | 180 / 103 |
| Thai | CER | 11.2% | 10.7% | −0.5 | [−1.7, +0.6] | 262 / 144 |
| Japanese | CER | 9.3% | 9.2% | −0.1 | [−1.1, +1.0] | 145 / 80 |
| Korean | WER | 13.9% | 14.8% | +0.9 | [−1.0, +2.9] | 159 / 88 |
| Chinese | CER | 7.2% | 9.4% | **+2.2** | [+0.8, +3.5] | 100 / 59 |

- **Sanity check**: Rumble's German 5.0% on 100 clips is close to its card (FLEURS de 4.77%), so the pipeline
  reproduces their number.
- **Rumble is ~1.8× faster everywhere**: 14 vs 28 decoder layers. Greek and Hindi take 2.5× longer than other
  languages in *both* models, which is the character-level tokenization of section 13 showing up as decode time.
- **No sign the vocabulary trimming hurt anything**: Thai lost 62% of its tokens and is unchanged.
- **Chinese is the one real regression** (Qwen better on 48 clips, Rumble on 10). The errors are same-sound wrong
  characters (大坝建成 → 大罢继承, 驯养 → 信仰, 沉积物 → 趁机雾), so it is a weaker Chinese language model, not a
  convention change. Plausible causes: less Chinese in their training data, or half the decoder layers. These cannot
  be separated without the recipe.
- So Rumble trades a little Chinese for large gains in Greek, Polish and the other European languages. That fits a
  European-focused product.

**B. Forgetting: FLEURS `el_gr` test, first 200 clips (37 min of audio)**

| system | WER basic | phonetic | dialect-forgiving* | CER | τζαι | εν | και | είναι |
|---|---|---|---|---|---|---|---|---|
| (reference) | | | | | 0 | 2 | 157 | 52 |
| whisper-medium | 22.1% | 18.1% | 17.3% | 8.3% | 0 | 7 | 153 | 53 |
| whisper-medium + LoRA | 46.9% | 42.0% | 30.1% | 17.5% | 31 | 57 | 123 | 28 |
| Rumble-2-Flash (matched prompt) | 14.1% | 12.6% | 12.0% | 7.1% | 0 | 2 | 152 | 52 |
| Rumble-2-Flash + LoRA | 39.4% | 36.7% | 25.4% | 14.8% | 25 | 43 | 129 | 35 |
| Qwen3-ASR-0.6B | 51.9% | 46.9% | 45.9% | 23.6% | 0 | 1 | 149 | 53 |
| Qwen3-ASR-0.6B + LoRA | 74.9% | 70.9% | 62.8% | 38.3% | 30 | 47 | 100 | 31 |

\*Diagnostic only, applied to ref and hyp: phonetic level, word-final ν dropped, τζαι/τζ → και, εν → είναι,
ούλλοι/ούλλα → όλοι/όλα.

- **All three fine-tunes forget Standard Greek badly**: WER rises by +24.8 (whisper-medium, CI [+22.7, +26.8]),
  +25.3 (Rumble, [+23.4, +27.1]) and +23.0 points (Qwen, [+17.1, +32.2]). Each is worse on 188-193 of 200 clips
  (Qwen: 159).
- **About half of it is the model writing Standard Greek in Cypriot form**: adding final ν (κόσμο → κόσμον,
  νερό → νερόν, δόγμα → δόγμαν), είναι → εν, από → που, και → τζαι, σήμερα → σήμμερα, χόκεϊ → χόκκιν.
  The dialect-forgiving score removes 48% (whisper-medium), 47% (Rumble) and 26% (Qwen) of the increase. The model
  learned "write what you hear in Cypriot orthography", which is the right behavior for our task and the wrong one
  for Standard Greek audio.
- **The other half is real degradation**: misheard words even with the dialect forgiven, e.g. έδρα → έτρεπεν,
  Ωκεανό → Αννό. Rumble + LoRA is still 13.4 points worse than zero-shot after forgiving dialect forms.
- Rumble keeps its lead after fine-tuning (39.4% vs whisper-medium 46.9%).
- **Not caused by numbers**: on the 157 clips without digits, the increases are about the same (e.g. Rumble
  11.9% → 36.8%).
- **What follows**:
  1. A Cypriot LoRA should not be served for Standard Greek audio. Keep the adapter unmerged and route by dialect,
     or mix SMG data into fine-tuning to keep both. Testing the mix is a natural next experiment.
  2. The LoRA rank (r = 32, attention + FFN in encoder and decoder) is a lot of capacity for 10 h. A lower rank or
     decoder-only LoRA may forget less. Untested.
- **Limits**:
  - 200 of 1,000+ test clips, one seed, T4 float32 for the Qwen models.
  - Rumble uses the `{digits: false, filler: true}` prompt in both runs (FLEURS writes digits), which costs a
    little in both runs alike.
  - FLEURS is read news-style speech, unlike our spontaneous data.

## 16. Weight blending (WiSE-FT): trading Cypriot gain against Standard Greek loss (2026-11-25)

**Why**: section 15 showed that the Rumble LoRA (Cypriot dev 47.3% → 17.3%) breaks Standard Greek (FLEURS 14.1% →
39.4%). The cheapest fix needs no training: blend the weights, **W = W0 + λ·(W1 − W0)**, with W0 = zero-shot Rumble
(revision 95bd5f9) and W1 = our merged LoRA fine-tune, and score each λ on both test beds (WiSE-FT, Wortsman et al.
2022).

**Setup**: `scripts/wiseft_sweep.py`, notebook 10.
- Both checkpoints are bf16 safetensors with identical keys. 206 tensors differ (394M values): 108 in the audio
  encoder (18 layers × q, k, v, out, fc1, fc2) and 98 in the decoder (14 layers × q, k, v, o, gate, up, down),
  exactly the LoRA targets. Blending is done in float32 per tensor, then cast to the model dtype.
- Cypriot dev: 280 full-length clips, as `eval_rumble-2-flash_lora.yaml`. FLEURS `el_gr`: the same first 200 test
  clips as section 15. Prompt `{digits: false, filler: true, verbatim: true}` for both, greedy, batch 1.
- Colab T4, float32, two processes in parallel (~5.3 GB RAM each; the 2 CPU cores limited the gain to ~12%).
  ~50 min per λ on Cypriot dev, ~17 min on FLEURS; 6 λ values in about 4.5 h. Results:
  `MyDrive/cyasr/runs_wiseft/` (λ = 0.625 is saved as `lam0.62_*`), plot `docs/figures/wiseft_tradeoff.png`.
- **Checks**: at λ = 0 and λ = 1 the FLEURS outputs are identical, 200/200 clips, to the section 15 runs (also T4
  float32). λ = 1 on Cypriot dev in float32 gives 17.3% / 15.0% / 5.3%, against 17.3% / 15.0% / 5.2% in bf16
  (section 11), and λ = 0 gives 47.3% / 45.9% / 14.5% against 47.3% / 45.9% / 14.6%.

**Results** (WER basic unless noted; Cypriot-form counts in the FLEURS output; reference: τζαι 0, εν 2)

| λ | Cypriot WER | nodiac | CER nodiac | FLEURS WER | FLEURS CER | τζαι | εν | Cypriot gain kept | SMG damage left |
|---|---|---|---|---|---|---|---|---|---|
| 0 (zero-shot) | 47.3% | 45.9% | 14.5% | 14.1% | 7.1% | 0 | 2 | 0% | 0% |
| 0.25 | 41.2% | 39.6% | 11.9% | 14.2% | 7.1% | 0 | 4 | 20% | 0% |
| 0.5 | 29.5% | 27.2% | 7.9% | 17.2% | 7.9% | 2 | 13 | 59% | 12% |
| 0.625 | 23.0% | 20.7% | 6.3% | 20.6% | 8.6% | 3 | 19 | 81% | 26% |
| **0.75** | **19.3%** | 16.9% | 5.4% | **25.6%** | 10.2% | 9 | 26 | **93%** | 45% |
| 1 (fine-tune) | 17.3% | 15.0% | 5.3% | 39.4% | 14.8% | 25 | 43 | 100% | 100% |

"Gain kept" = (47.3 − WER) / 30.0 on Cypriot dev; "damage left" = (WER − 14.1) / 25.3 on FLEURS. Paired bootstrap
(10,000 resamples, seed 42), each vs its endpoint: λ = 0.75 is +2.0 points on Cypriot vs λ = 1 (95% CI [+1.4, +2.5])
and −13.8 on FLEURS ([−15.0, −12.5]); λ = 0.5 is +3.1 on FLEURS vs zero-shot ([+2.0, +4.2]); λ = 0.25 is +0.1
([−0.7, +1.0]), no measurable loss.

**Findings**
- **The trade-off curve bends in our favour**: a straight line between the two models would give, at λ = 0.75,
  24.8% Cypriot and 33.1% FLEURS; the blend gives 19.3% and 25.6%, better on both. The last quarter of the way to
  the fine-tune buys 2 Cypriot points and costs 14 Standard Greek points.
- **λ = 0.75 keeps 93% of the Cypriot gain and removes 55% of the Standard Greek damage.** At 19.3% it still beats
  fine-tuned whisper-medium on Cypriot (19.9%), while its Standard Greek error (25.6%) is far below any full
  fine-tune (39.4-74.9%, section 15). λ = 0.5-0.625 are the "balanced" settings (Cypriot 23-30%, FLEURS 17-21%).
- **Spelling switches sharply, accuracy changes gradually.** τζαι in the Cypriot dev output (reference 670): 0, 3,
  518, 662, 690, 706 for λ = 0 … 1; και falls from 630 to 68 between λ = 0.25 and 0.5. The orthography is a
  near-binary switch between λ = 0.25 and 0.5. On FLEURS, Cypriot forms leak in only from λ ≈ 0.5 (εν 13) and
  mostly at λ = 1 (τζαι 25), so at λ = 0.75 the Standard Greek damage is mostly real mishearing, not dialect
  spelling.
- **With LoRA, blending = scaling the adapter.** The merged fine-tune is W1 = W0 + (α/r)·B·A, so W1 − W0 is exactly
  the LoRA update and the blend at λ equals the unmerged adapter run at λ × its normal scale. In practice:
  no second model is needed, and the scale can be set per request (λ = 1 for audio known to be Cypriot, ~0.5 for
  mixed or unknown audio) from one base model plus one adapter. The same knob applies to any LoRA fine-tune, e.g.
  our whisper-medium LoRA (Standard Greek 22.1% → 46.9%, section 15). The method itself is not new (WiSE-FT;
  adapter scaling is common practice); what is measured here is the trade-off curve for a dialect.
- **No single λ is right for both**: even λ = 0.75 is 11.5 points worse than zero-shot on Standard Greek
  ([+10.0, +13.1]). Blending reduces forgetting; it does not remove it. Fixing both at once needs Standard Greek
  in training (rehearsal) and, given the switch-like orthography, probably an explicit dialect tag in the prompt.

**Limits**
- λ is picked on the same dev set it is reported on, so the chosen point is slightly optimistic; a final λ should
  be confirmed on test.
- FLEURS: 200 of 1,000+ clips, read speech; one seed, one fine-tune.
- Only Rumble; whisper-medium's forgetting (section 15) is not blended yet.

## 17. Dialect tag and Standard Greek rehearsal (2026-11-25/26)

**Why**: blending (section 16) cannot keep both varieties at once, and the spelling flips like a switch, so the
model needs (1) Standard Greek in training and (2) an explicit signal for which spelling to use.

**Design**
- **Data**: 410 Common Voice Scripted Speech Greek clips (CC0; different source from the FLEURS test, so the
  Standard Greek score stays out-of-domain like section 16) from CV's own train split, max 10 per speaker, seed 42;
  100 from CV dev for monitoring. `scripts/prep_cv_greek.py` → `data/segments_cv_el/`. 410 next to 820 Cypriot
  pieces = 1/3 Standard Greek per batch on average (uniform sampling over the concatenation).
- **Tag**: Rumble's chat template puts free-text system context after its control tokens. Cypriot pieces get
  "Cypriot Greek" there (`<nodigits><filler><verbatim>Cypriot Greek`), Standard Greek pieces get nothing, so the
  default (untagged) prompt keeps standard output and existing users are unaffected. No new tokens. Inference:
  `prompt` in the eval config / `QwenTranscriber(prompt=...)`.
- **Recipe**: as section 11 (LoRA r32, LR 1e-4, 1000 steps, batch 4 × 4), dev checks every 250 steps on 150 evenly
  spaced Cypriot dev pieces + 100 CV dev clips, best checkpoint by Cypriot WER.
- **Ablation**: the same mix without the tag (`finetune_rumble-2-flash_mix.yaml`): does the tag matter?
- **Evaluation**: Cypriot dev full clips with the tag; FLEURS 200 clips without it (zero-shot and plain LoRA re-run
  on the same GPU); diagnostics: Cypriot untagged, FLEURS tagged. Compare with the λ = 0.5-0.75 blends.
- Code: `cyasr/train_qwen.py` (`dialect_prompt`, `smg_segments_dir`, `smg_max_samples`, `smg_dev_samples`,
  `eval_subset`; old configs unchanged), `cyasr/transcribe_qwen.py` (`prompt`), `scripts/eval_fleurs.py`,
  notebook 11. CPU smoke test with a tiny random Rumble-shaped model (2 + 2 layers): mixed batches train, the
  Cypriot example has the tag and the Standard Greek one not, both dev sets are logged, best = Cypriot WER, the
  saved model's inference prompt matches training.

**Run 1: tag only, no Standard Greek data** (`finetune_rumble-2-flash_tag.yaml`, 2026-11-25). Question: Rumble
was trained on untagged Greek before, so is a tag on the Cypriot data alone enough to keep the untagged
behaviour? Prediction beforehand: no, because a tag present on every training example carries no information and
nothing in the loss penalises Cypriot spelling on untagged input.
- Colab T4: fp16 autocast (no bf16; losses stayed finite), batch 2 × 8 (same effective 16), peak 9.4 GB,
  4.1 s/step, 69 min training + 20 min dev checks. Evaluation in float32. FLEURS rows for zero-shot and plain LoRA
  are the section 16 λ = 0 / λ = 1 runs (same 200 clips, T4 float32), copied to `MyDrive/cyasr/runs_mix/`.
- Dev checks (150 pieces): 17.1% (step 250), 16.8%, 16.1%, 15.8% (step 1000 = best).

| system | Cypriot WER basic | nodiac | CER nodiac | FLEURS WER basic | FLEURS CER | τζαι | εν |
|---|---|---|---|---|---|---|---|
| zero-shot Rumble | 47.3% | 45.9% | 14.5% | 14.1% | 7.1% | 0 | 2 |
| plain LoRA (section 11) | 17.3% | 15.0% | 5.2% | 39.4% | 14.8% | 25 | 43 |
| tag only, tagged input | 17.6% | 15.1% | 5.2% | 39.9% | 15.3% | 32 | 49 |
| tag only, untagged input | 17.5% | 15.1% | 5.2% | **39.8%** | 15.3% | 31 | 45 |

(Cypriot columns: dev, 280 full clips; FLEURS τζαι/εν: counts in the output, reference 0 / 2.)

- **The tag alone does nothing for Standard Greek**: untagged FLEURS 39.8% vs 39.4% for the plain LoRA; Cypriot forms
  leak in just as much (τζαι 31 vs 25).
- **The model ignores the tag**: tagged vs untagged gives the same scores and the same spelling on both test sets
  (Cypriot dev τζαι 707 vs 706; FLEURS 39.9% vs 39.8%). The outputs are not identical (101/280 Cypriot and 72/200
  FLEURS clips word-for-word the same), so the tag tokens change some decoding paths, but not systematically.
- **The tag costs nothing on Cypriot**: 17.6% / 15.1% vs 17.3% / 15.0% (one seed; within noise).
- Interpretation: being trained on untagged Greek before does not protect the untagged behaviour; the LoRA update
  changes weights shared by both prompts, and with the tag on every example the model has no reason to condition on
  it. A tag only becomes a switch when training contains both sides (tagged Cypriot and untagged Standard Greek),
  which is run 2 (`mix_tag`) and its ablation (`mix`).

**Runs 2 and 3: Standard Greek rehearsal with the tag (`mix_tag`) and without (`mix`)** (2026-11-26)
- Data: Common Voice Scripted Speech 27.0 Greek (CC0). CV Greek has only 20 speakers in its train split and 323 in
  test; CV is never an evaluation set here, so training clips were pooled from CV train + test, max 5 per speaker:
  **410 clips, 206 speakers, 30 min**; monitoring: 100 CV dev clips, 42 speakers (`scripts/prep_cv_greek.py`).
  Uploaded to Drive as three MP3 zips (10 MB upload limit) and converted on Colab with the same `load_audio`; all
  510 WAVs byte-identical to the local ones (md5).
- Colab L4, bf16, batch 4 × 4, 1.9 s/step, 32 min training + 18 min dev checks each, peak 13.1 GB. The runtime
  dropped once after `mix_tag` training had finished (checkpoints and `best/` were already on Drive); the
  evaluations were re-run on a new L4 runtime. FLEURS rows for zero-shot, plain LoRA and tag-only were re-measured
  on the same L4 in bf16 (`*_bf16`), so all FLEURS numbers below share GPU and dtype.
- Dev checks (150 Cypriot pieces / 100 CV dev clips), `mix_tag`: 17.3/17.1%, 17.2/18.0%, 16.1/18.7%, 15.9/18.5%
  (steps 250-1000, best 1000); `mix`: 17.4/19.5%, 16.6/20.5%, 16.4/18.7%, 16.5/20.5% (best 750).

**The 2 × 2** (Cypriot dev: 280 full clips, with the tag where trained with it; FLEURS: 200 clips, untagged;
τζαι/και counts in the Cypriot output, reference 670/1; FLEURS reference τζαι 0, εν 2)

| system | tag | SMG data | Cypriot WER | nodiac | CER nodiac | τζαι/και | FLEURS WER | FLEURS CER | τζαι | εν |
|---|---|---|---|---|---|---|---|---|---|---|
| zero-shot Rumble | | | 47.3% | 45.9% | 14.6% | 0/626 | 14.1% | 7.2% | 0 | 2 |
| plain LoRA (s. 11) | no | no | 17.3% | 15.0% | 5.2% | 707/2 | 39.7% | 15.0% | 25 | 41 |
| tag only | yes | no | 17.6%* | 15.1% | 5.2% | 707/6 | 39.6% | 15.0% | 33 | 48 |
| data only (`mix`) | no | yes | 18.2% | 15.8% | 5.5% | 688/4 | 30.3% | 11.7% | 6 | 23 |
| **data + tag (`mix_tag`)** | yes | yes | **17.9%** | 15.5% | 5.3% | 699/5 | **29.6%** | 11.6% | 9 | 24 |
| `mix_tag`, tag swapped (CY untagged / FLEURS tagged) | | | 17.9% | 15.5% | 5.4% | 704/4 | 30.2% | 11.9% | 9 | 27 |

\*Tag-only Cypriot row evaluated on a T4 in float32 (section 17 run 1); the rest on L4 bf16 (precision changed
the plain LoRA's Cypriot score by 0.0 points in section 16).

Paired bootstrap (10,000 resamples, seed 42), WER basic:
- FLEURS: `mix` vs plain LoRA −9.5 [−10.9, −8.0]; `mix_tag` vs plain LoRA −10.1 [−11.7, −8.6]; `mix_tag` vs `mix`
  −0.7 [−1.8, +0.4]; tag only vs plain LoRA −0.1 [−1.3, +1.1]; `mix_tag` vs zero-shot +15.5 [+13.9, +17.0].
- Cypriot: `mix` vs plain LoRA +0.9 [+0.3, +1.4]; `mix_tag` vs plain LoRA +0.6 [−0.1, +1.3]; `mix_tag` vs `mix`
  −0.3 [−0.8, +0.3].

**Findings**
- **The data does the work, the tag does nothing measurable.** Standard Greek rehearsal cuts the FLEURS damage by
  ~10 points (39.7% → 30.3% / 29.6%) at a cost of 0.6-0.9 Cypriot points. Adding the tag changes neither score
  significantly, with or without rehearsal (tag vs no tag: −0.1 and −0.7 points on FLEURS, CIs include 0).
- **The tag is not used even when it could be**: swapping it at inference changes nothing (Cypriot untagged 17.9%
  with τζαι 704; FLEURS tagged 30.2%). Likely reason: the audio already separates the two training domains
  perfectly (spontaneous Cypriot speech vs read Standard Greek, different speakers, microphones and style), so the
  model can choose the spelling from the audio and never needs the tag. A tag would only be needed where audio is
  ambiguous (e.g. Cypriot speakers reading standard text), which this data does not contain.
- **Spelling leak mostly fixed, mishearing not**: Cypriot forms in the FLEURS output drop from τζαι 25 / εν 41 to
  6-9 / 23-24 (reference 0 / 2), but WER is still 15.5 points above zero-shot and CER 11.6% vs 7.2%. Most of what is
  left is acoustic/lexical degradation on read news speech, not orthography.
- **Compared with blending (section 16)**: at about the same Cypriot WER as `mix_tag` (17.9%), the blending curve
  between λ = 0.75 (19.3% / 25.6%) and λ = 1 (17.3% / 39.4%) gives roughly 35% on FLEURS, so rehearsal is ~5 points
  better at equal Cypriot accuracy (interpolated, T4 fp32 vs L4 bf16; indicative). The best points of the two
  methods are close; they are complementary (blending `mix_tag` with the base is an untested next step).
- 30 minutes of read Standard Greek (1/3 of each batch) is a small amount; more or more varied Standard Greek, or
  pseudo-labels from zero-shot Rumble on unlabelled Greek audio, are the obvious levers.

**Limits**: one seed per variant; FLEURS 200 clips; the CV monitoring set is read speech like FLEURS (so
"in-domain-ish" for the rehearsal data, but different speakers and sentences from the FLEURS test); λ and the tag
text were not tuned.

## 18. Final test evaluation (2026-11-26)

**What**: the models chosen on dev, evaluated once on the **test** split (371 full-length clips, 2.89 h, 3 speakers
never seen in training or dev), with exactly their dev settings (Rumble/Qwen: greedy, batch 1, bf16, matched
prompt; `mix_tag` with its tag; whisper-medium LoRA: fallback decoding + no-speech rule 0.6). Colab L4, ~35 min per
Rumble model, 16 min for whisper-medium. Nothing was tuned on test. The model to publish (`mix_tag`) was chosen
before this run (docs/RELEASE_PLAN.md, D5), for its smaller Standard Greek loss.
Qwen LoRA did not finish (the runtime stopped at 112/371 clips); it is left out (it was 14 points behind Rumble on
dev). Results were copied from Drive to `runs/*_test/`.

| model | test WER basic | nodiac | phonetic | CER nodiac | dev WER basic | dev nodiac | dev CER nodiac |
|---|---|---|---|---|---|---|---|
| zero-shot whisper-medium (fallback) | 58.3% | 55.5% | 52.2% | 21.9% | 50.3% | 48.2% | 16.4% |
| zero-shot whisper-large-v3 (fallback) | 52.6% | 50.1% | 47.0% | 18.5% | 47.3% | 45.7% | 16.6% |
| zero-shot Rumble-2-Flash (matched) | 55.7% | 53.6% | 50.6% | 20.8% | 47.3% | 45.9% | 14.6% |
| whisper-medium LoRA + no-speech rule | 25.4% | 21.9% | 20.0% | **8.3%** | 19.9% | 17.3% | 6.8% |
| Rumble-2-Flash LoRA | 26.9% | 23.8% | 22.6% | 10.9% | **17.3%** | **15.0%** | **5.2%** |
| **Rumble-2-Flash `mix_tag`** (data + tag) | **24.7%** | **21.4%** | 20.2% | 8.7% | 17.9% | 15.5% | 5.3% |

Per test speaker (WER nodiac; clips 145 / 146 / 80): zero-shot Rumble 50.3 / 75.0 / 43.7%; whisper-medium LoRA
19.5 / 36.6 / 15.8%; Rumble LoRA 22.4 / 36.8 / 13.1%; `mix_tag` 19.4 / 35.2 / 14.4%. By duration (<10 s / 10-30 s /
>30 s): whisper-medium LoRA 30.6 / 25.2 / 20.0%; Rumble LoRA 27.6 / 25.9 / 22.7%; `mix_tag` 30.7 / 25.3 / 19.3%.

Paired bootstrap on test (10,000 resamples, seed 42), WER basic (nodiac similar):
- `mix_tag` vs Rumble LoRA: −2.1 points [−3.6, −0.7] (better on 138 clips, worse on 135).
- Rumble LoRA vs whisper-medium LoRA: +1.4 [−0.9, +4.3]; `mix_tag` vs whisper-medium LoRA: −0.7 [−2.3, +1.4].
- `mix_tag` vs zero-shot Rumble: −30.9 [−32.2, −29.6] (better on 363 of 371 clips).

**Findings**
- **Fine-tuning holds up on unseen speakers**: every fine-tuned model more than halves zero-shot error on test
  (Rumble 55.7% → 24.7%).
- **Test is harder than dev for everyone** (+5-9 points), mostly one speaker (d7725a: 35-37% after fine-tuning,
  75% zero-shot) with 146 of the 371 clips.
- **Rumble's dev lead over whisper-medium does not survive on test**: on dev Rumble LoRA was 2.6 points better
  ([−4.0, −1.3]); on test it is 1.4 points worse, not significant. Fine-tuned whisper-medium, Rumble LoRA and
  `mix_tag` are statistically tied on test. Rumble is still 1.5× smaller (509M vs 769M), needs no decoding rules and
  runs one pass per clip. Report this plainly; "Rumble wins" is a dev-only claim.
- **`mix_tag` is the best test model and significantly better than the plain Rumble LoRA** (−2.1 points), although
  it was 0.6 points worse on dev (n.s.). Plausible reading: the Standard Greek rehearsal regularises the fine-tune
  and generalises better to new speakers. One seed each, so treat it as a first result; it supports publishing
  `mix_tag` (chosen before the test run for its smaller Standard Greek loss).
- **CER**: whisper-medium LoRA has the lowest test CER (8.3%) with `mix_tag` close (8.7%); the plain Rumble LoRA is
  higher (10.9%), consistent with its occasional skipped/drifting stretches on long clips (section 11).
- **Comparison with the paper** (Stamou et al. 2026: whisper-large-v3 33.93% / medium 37.85%): our 24.7% (basic)
  / 21.4% (nodiac) is on a speaker-independent split and full-length clips with our own normalization, theirs on a
  random (not speaker-independent) split, ≤30 s forced-aligned pieces and their 7-step normalization. Not directly
  comparable (section 14); state both protocols, claim no "win".

---

## Open items
- Next for forgetting (after section 16): rehearsal (mix 20-50% Standard Greek from Common Voice Greek or FLEURS
  `el_gr` *train* into the LoRA run) with a dialect tag in the prompt (Cypriot vs Greek), cheaper training (dev check
  every 200 steps on a fixed subset, ~400 steps). Compare against the λ = 0.5-0.75 blends on both test beds.
- Done: LoRA fine-tuning of Rumble-2-Flash (section 11) and base Qwen3-ASR-0.6B (section 12). Next: one table of small / medium / Rumble, zero-shot vs
  fine-tuned, with time and memory; a second seed for Rumble and whisper-medium; final test evaluation. Consider Trackio for live training
  curves, synced to a Hugging Face Space so data survives Colab disconnects.
- Trailing-silence trimming before decoding (simple energy threshold): a cheap fix to test for the 30-33 s
  clips, simpler than training on silent windows.
- Fix the `timing.json` resume bug; use `segments.zip` in notebook 03.
- Later, each a separate experiment: full fine-tuning and frozen encoder vs LoRA; effective batch size;
  training on silent windows (`<|nospeech|>`); digit normalization; accent-free targets (option b);
  segmented test set for a paper-comparable number (protocol in section 14); final test evaluation.
- Done (section 15, first 200 clips): forgetting on FLEURS `el_gr` for medium, Rumble and Qwen. Still open: the
  full test set, whisper-small, and an SMG-mixed or lower-rank LoRA to reduce forgetting. Original plan:
  Forgetting control on FLEURS (read speech, CC-BY 4.0): Standard Modern Greek (`el_gr` test), zero-shot vs
  fine-tuned, for every model (small, medium, Qwen). Same pipeline: normalization, fallback + no-speech rule,
  forced language token. Look for Cypriot forms leaking into standard Greek (τζαι for και). Optional: a small
  English subset (`en_us`). Domain differs from ours (read vs spontaneous), so compare only zero-shot vs
  fine-tuned on the same FLEURS data. Rumble-2-Flash's card (revision 95bd5f9) no longer lists a FLEURS Greek score; check before citing one. Talking point: an unmerged LoRA adapter can be switched off for non-Cypriot audio.

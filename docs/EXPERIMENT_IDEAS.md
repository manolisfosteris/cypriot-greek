# Experiment ideas tracker

Ideas that could make the project stand out, beyond "fine-tune Whisper and beat the paper".
This file tracks what we might try. Results go in `docs/PROJECT_LOG.md` and the run folders.

Status values: `idea` → `planned` → `running` → `done` / `dropped` (say why)

## Overview

| # | Idea | Priority | Status | Depends on |
|---|------|----------|--------|------------|
| 0 | Baselines (Whisper, Qwen, other models), zero-shot | core | running | data |
| 1 | Fine-tune (reproduce paper range) | core | idea | 0 |
| 2 | Rigorous evaluation (seeds, bootstrap CIs, data checks) | core | idea | 0 |
| 3 | Text-only adaptation for the dialect | main idea | idea | 0, Cypriot text corpus |
| 4 | Spelling vs mishearing (normalised scoring) | analysis, end | idea | 1 |
| 5 | Data curve (1 / 2 / 4 / 9 h) | nice to have | idea | 1 |
| 6 | Catastrophic forgetting on Standard Greek | analysis, end | idea | 1 |
| 7 | Error taxonomy with counts | analysis, end | idea | 1, 3 |
| 8 | Hallucinations on spontaneous speech | optional | idea | 0 |
| 9 | Medium article + one-command reproduction | deliverable | idea | all |

---

## 2. Rigorous evaluation

Why: ~9 h of data and a small test set, so a 1–2 point WER gap may be noise.

- [ ] 2–3 seeds for the main fine-tuning runs; report mean ± std.
- [ ] Bootstrap 95% confidence intervals on test WER/CER (resample utterances).
- [ ] Paired bootstrap when comparing two systems on the same test set.
- [ ] Check for the same speaker in train and test; check duplicate sentences across splits.
- [ ] Confirm the paper's splits and text normalisation so the comparison is fair (see CLAUDE.md TODO).

## 3. Text-only adaptation (main idea)

Question: how much of the gain from audio fine-tuning can we get from Cypriot **text alone**?
This matches Reson8's product (customisable ASR, text-only adaptation).

Headline metric: **gap closed** = (WER_zero-shot − WER_text-only) / (WER_zero-shot − WER_audio-finetuned).
"Text-only closes X% of the gap at ~0 audio cost."

Second question: does text-only adaptation **stack on top of** the best fine-tuned model?
Expect smaller gains there (the fine-tune already saw the transcripts), but if it still helps, that's a strong result.

Grid:

| | zero-shot model | best fine-tuned model |
|---|---|---|
| no text adaptation | baseline | fine-tune result |
| + text adaptation | text-only result | stacked result |

Methods to try (cheap first):
- [ ] Prompt / vocabulary biasing with frequent Cypriot words (pilot showed this barely helps on whisper-small; test on larger models).
- [ ] Shallow fusion with an n-gram LM (KenLM) trained on Cypriot text, during beam search.
- [ ] TTS-generated audio from Cypriot text, then fine-tune (only if time allows; TTS for Cypriot may not exist).

Text sources to look for (check licences):
- [ ] Training-split transcripts (fair baseline for "in-domain text"; never test transcripts).
- [ ] Cypriot dialect dictionaries / word lists.
- [ ] Forums, social media, blogs written in Cypriot.
- [ ] Song lyrics, literature, theatre scripts in dialect.

Pitfall: no test-set text may leak into the LM. Deduplicate the text corpus against test transcripts.

## 4. Spelling vs mishearing

Question: how much of the WER is penalising valid alternative spellings rather than real recognition errors?
Cypriot has no standard orthography, so this matters more than for most languages.

Normalisation levels (report WER/CER at each, cumulative):
1. Basic: NFC, lowercase, final sigma, strip punctuation.
2. + remove accents.
3. + merge homophone vowels (ι/η/υ/ει/οι → one symbol, ο/ω, αι/ε).
4. + map known Cypriot spelling variants (e.g. τζ/τσ variants, palatal spellings) to one form.

Result: a chart showing WER dropping per level. The final level is roughly "real mishearing" error.
Also apply to the paper's numbers if their outputs/normalisation allow.

## 5. Data curve

Question: how much Cypriot audio does a customer need to record?

- [ ] Train on 1, 2, 4 and all ~9 hours (subsample **by speaker**, fixed seed, nested subsets).
- [ ] Same hyperparameters and evaluation for every point.
- [ ] Plot WER vs hours (log x-axis), with seed error bars if budget allows.
- [ ] Optional: same curve with text-only adaptation added, to show if text substitutes for audio at low data.

## 6. Catastrophic forgetting

Question: does fine-tuning on Cypriot make the model worse at Standard Modern Greek?

- [ ] Pick a standard Greek benchmark: FLEURS `el_gr` test and/or Common Voice Greek test.
- [ ] Evaluate zero-shot model vs each fine-tuned model on it (same normalisation).
- [ ] Compare LoRA vs full fine-tune (LoRA expected to forget less).
- [ ] Optional: check English too (code-switching relies on it).

Result: a two-axis plot: Cypriot WER (gain) vs Standard Greek WER (loss) per method.

## 7. Error taxonomy

Classify a sample of errors (e.g. 100–200 utterances) into categories, with counts per system:
- dialect word unknown to the model
- spelling variant (valid word, different spelling)
- English code-switch
- hallucination / repetition
- word split or merge ("μετοπρολόλη" → "με το προλόλι")
- genuine mishearing

Show which categories each method (fine-tune, text-only) actually fixes.

## 8. Hallucinations (optional)

- [ ] Count insertions on silence/pauses and repeated n-gram loops in spontaneous speech.
- [ ] Compare models and decoding settings (temperature 0, no_repeat, VAD).

## 9. Deliverables

- [ ] One-command reproduction (config-driven, fixed seeds, saved predictions per run).
- [ ] Medium article: one main chart, honest negative results, "what I'd do next with a GPU cluster".
- [ ] README with results table and links to configs/runs.

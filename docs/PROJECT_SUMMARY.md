# Project summary: what was done and what was learned

A short overview for interviews, the blog post and later chats (as of 2026-11-23). Details and numbers: 
`docs/PROJECT_LOG.md` (section numbers in brackets). Blog plan: `docs/BLOG_NOTES.md`.

## What was done

1. **Data and splits** [1]: Common Voice Spontaneous Speech el-CY v5.0, 1,221 clips, 10.4 h, 10 speakers. Kept the
   official speaker-disjoint splits (train 5 speakers, dev 2, test 3). Dev for every decision; test kept for the end.
2. **Scoring** [2]: corpus-level WER and CER with a normalization ladder (raw → basic → nofill → nodiac → phonetic),
   so spelling differences can be separated from real mishearing.
3. **Zero-shot baselines** [3, 10]: whisper-small, medium, large-v3, Qwen3-ASR-0.6B and Reson8's Rumble-2-Flash.
   Best zero-shot: large-v3 and Rumble, both 47.3% dev WER.
4. **Segmentation** [4]: forced alignment (MMS aligner) to cut long clips into ≤ 30 s training pieces at word
   boundaries (566 train clips → 820 pieces).
5. **Training targets** [5]: kept Cypriot spelling and fillers; unified the three ways the Cypriot mark was typed.
6. **LoRA fine-tuning, same recipe for every model** [6, 9, 11, 12]: r32/alpha64, LR 1e-4, effective batch 16,
   dev check every 100 steps, early stopping, SpecAugment. Built a second training path for Qwen-style models
   (prompt-masked labels, generation-based dev WER).
7. **Results on dev** (full-length clips, WER basic): whisper-small 58.8 → 28.6%, whisper-medium 50.3 → 19.9%,
   Qwen3-ASR 73.1 → 31.6%, **Rumble 47.3 → 17.3%** (best).
8. **Error analysis** [8, 11, 12]: errors by clip length, substitution pairs, dialect markers, paired bootstrap.
9. **Fixes found by diagnosis** [3, 8, 10]: Whisper's silent last window on 30-33 s clips (no-speech rule);
   batch size changing outputs (batch 1 for evaluation in bf16).
10. **Model forensics** [13]: how Rumble was made from Qwen (half the decoder layers, trimmed vocabulary, added
    control tokens and a CTC head).
11. **Paper read** [14]: their split is random and not speaker-independent, WER on segments, their own
    normalization, so numbers are not directly comparable.
12. **Infrastructure**: config-driven runs, fixed seeds, pinned model revisions, saved predictions and metrics per
    run, background training on Colab with resume and automatic GPU release, unit tests for normalization.

## What was learned

**About the dialect and the models**
- Off-the-shelf models hear Cypriot but write Standard Greek (τζαι → και hundreds of times). Much of the zero-shot
  error is orthography, not hearing.
- Fine-tuning on ~4.7 h teaches the Cypriot spelling quickly; what remains is dialect morphology (final -ν, elided
  forms, οι/η) and real mishearing.
- Whisper's 30 s windows cause a specific failure on clips just over 30 s (invented text for a near-silent tail).
  Models that read the whole clip at once (Qwen, Rumble) do not have it, and that is where Rumble wins.
- A smaller, compressed model (Rumble, 14 decoder layers) can be a far better base than its larger parent
  (Qwen, 28 layers): 17.3% vs 31.6% after identical fine-tuning. Pretraining matters more than depth here.

**About doing experiments**
- Change one thing at a time; keep the recipe fixed so comparisons are about the model.
- Decide on dev, report on test once; one seed is a first result, not a conclusion.
- Corpus-level WER, CER next to WER, several normalization levels, and paired bootstrap for differences.
- Diagnose before fixing (the no-speech rule tested the silent-tail explanation without retraining).
- Check assumptions against the files: Rumble's size turned out different from what its name and tokenizer suggested.
- Numerical details matter: in bf16, batch size alone changes outputs.

**About the engineering**
- GPU memory is mostly activations, not weights; LoRA does not fix that. Keep the effective batch and change the
  physical batch.
- Generation-based dev checks can cost more GPU time than training (110 of 144 min for Rumble).
- Long Colab runs need: background processes, logs and checkpoints on Drive, resume, and automatic disconnect.
- Pin model revisions and library versions; one library (`qwen-asr`) pinned an old transformers and was replaced by
  the native implementation.

## Main ideas not yet done (docs/EXPERIMENT_IDEAS.md)
- Text-only adaptation (idea 3): improve the model for Cypriot using text alone, e.g. a Cypriot language model
  during decoding. Closest to Reson8's product.
- Data curve (how many hours a customer needs), forgetting of Standard Greek (FLEURS notebook ready), error taxonomy.

import math

from cyasr.metrics import compute_metrics, corpus_rate, error_counts


def test_corpus_wer_is_total_errors_over_total_ref_words():
    refs = ["α β", "α β γ δ α β γ δ"]  # 2 and 8 words
    hyps = ["α", "α β γ δ α β γ δ"]    # 1 error and 0 errors
    assert math.isclose(corpus_rate(refs, hyps)["rate"], 1 / 10)  # a mean of rates would be 0.25


def test_empty_hypothesis_counts_all_words_as_deletions():
    assert error_counts("α β γ", "") == (3, 3)


def test_empty_reference_counts_insertions():
    assert error_counts("", "α β") == (2, 0)


def test_cer_counts_characters():
    errors, n = error_counts("κατι", "κατα", unit="char")
    assert (errors, n) == (1, 4)


def _rec(ref, hyp, speaker="s1", dur=5.0):
    return {"ref": ref, "hyp": hyp, "speaker": speaker, "duration_s": dur}


def test_compute_metrics_breakdowns_and_flags():
    records = [_rec("τζαι ναι", "τζαι ναι"),
               _rec("εεε καλά", "", speaker="s2", dur=40),
               _rec("ένα δύο", "ένα δύο τρία τέσσερα πέντε", dur=15)]
    m = compute_metrics(records)
    assert m["n_utterances"] == 3
    assert m["empty_hypotheses"] == 1
    assert m["hallucinations"] == 1
    assert set(m["per_speaker"]) == {"s1", "s2"}
    assert set(m["by_duration"]) == {"<10s", "10-30s", ">30s"}
    # nofill drops the filler: 5 reference words in total (2 + 1 + 2), 1 + 3 errors
    assert m["levels"]["nofill"]["ref_words"] == 5
    assert m["levels"]["nofill"]["word_errors"] == 4

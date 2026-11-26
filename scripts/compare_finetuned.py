"""Compare two runs' predictions on the same dev clips: corpus WER/CER per normalization level, a paired
bootstrap of the WER difference (resampling clips), WER by clip length, error types, most frequent
substitutions, dialect markers, and the clips where the two models differ most.

Usage: python scripts/compare_finetuned.py runs/rumble-2-flash_lora_greedy_dev runs/whisper-medium_lora_fallback_ns0.6_dev
"""
import json
import random
import sys
from collections import Counter
from pathlib import Path

import jiwer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cyasr.normalize import normalize  # noqa: E402

LEVELS = ["basic", "nofill", "nodiac", "phonetic"]
MARKERS = ["τζαι", "εν", "που", "ειμαστε", "ηταν", "ητουν", "τζιαι"]  # nodiac forms


def load(run):
    rows = [json.loads(l) for l in open(Path(run) / "predictions.jsonl", encoding="utf-8")]
    return {r["audio_id"]: r for r in rows}


def counts(ref, hyp):
    o = jiwer.process_words(ref, hyp) if ref else None
    if o is None:
        return 0, 0, 0, len(hyp.split()), 0
    return o.substitutions, o.deletions, o.insertions, 0, len(ref.split())


def per_clip(rows, level):
    out = {}
    for k, r in rows.items():
        ref, hyp = normalize(r["ref"], level), normalize(r["hyp"], level)
        s, d, i, extra, n = counts(ref, hyp)
        out[k] = (s + d + i + extra, n, s, d, i)
    return out


def cer(rows, level):
    e = n = 0
    for r in rows.values():
        ref, hyp = normalize(r["ref"], level), normalize(r["hyp"], level)
        if ref:
            o = jiwer.process_characters(ref, hyp)
            e += o.substitutions + o.deletions + o.insertions
            n += len(ref)
    return e / n


def bootstrap(a, b, keys, n_boot=10000, seed=0):
    rnd = random.Random(seed)
    diffs = []
    for _ in range(n_boot):
        sample = [rnd.choice(keys) for _ in keys]
        ea = sum(a[k][0] for k in sample); eb = sum(b[k][0] for k in sample)
        n = sum(a[k][1] for k in sample)
        diffs.append((ea - eb) / n)
    diffs.sort()
    return diffs[int(0.025 * n_boot)], diffs[int(0.975 * n_boot)], sum(d < 0 for d in diffs) / n_boot


def subs(rows, level="nodiac", top=15):
    c = Counter()
    for r in rows.values():
        ref, hyp = normalize(r["ref"], level), normalize(r["hyp"], level)
        if not ref:
            continue
        o = jiwer.process_words(ref, hyp)
        rw, hw = o.references[0], o.hypotheses[0]
        for ch in o.alignments[0]:
            if ch.type == "substitute":
                for x, y in zip(rw[ch.ref_start_idx:ch.ref_end_idx], hw[ch.hyp_start_idx:ch.hyp_end_idx]):
                    c[(x, y)] += 1
    return c.most_common(top)


def main(run_a, run_b):
    A, B = load(run_a), load(run_b)
    keys = sorted(set(A) & set(B))
    A = {k: A[k] for k in keys}; B = {k: B[k] for k in keys}
    na, nb = Path(run_a).name, Path(run_b).name
    print(f"A = {na}\nB = {nb}\n{len(keys)} shared clips\n")

    print(f"{'level':9} {'WER A':>7} {'WER B':>7} {'A-B':>7}  95% CI (paired bootstrap)  P(A better)")
    for lv in LEVELS:
        a, b = per_clip(A, lv), per_clip(B, lv)
        n = sum(v[1] for v in a.values())
        wa = sum(v[0] for v in a.values()) / n; wb = sum(v[0] for v in b.values()) / n
        lo, hi, p = bootstrap(a, b, keys)
        print(f"{lv:9} {wa:7.1%} {wb:7.1%} {wa - wb:+7.1%}  [{lo:+.1%}, {hi:+.1%}]            {p:.3f}")
    print(f"CER nodiac: A {cer(A, 'nodiac'):.1%}  B {cer(B, 'nodiac'):.1%}\n")

    a, b = per_clip(A, "nodiac"), per_clip(B, "nodiac")
    for name, d in (("A", a), ("B", b)):
        s = sum(v[2] for v in d.values()); dl = sum(v[3] for v in d.values()); i = sum(v[4] for v in d.values())
        print(f"{name}: substitutions {s}, deletions {dl}, insertions {i}")

    print("\nWER nodiac by clip length")
    bucket = lambda x: "<=30s" if x <= 30 else ("30-33s" if x <= 33 else ">33s")
    for bk in ("<=30s", "30-33s", ">33s"):
        ks = [k for k in keys if bucket(A[k]["duration_s"]) == bk]
        n = sum(a[k][1] for k in ks)
        print(f"  {bk:7} {len(ks):3} clips  A {sum(a[k][0] for k in ks) / n:6.1%}  B {sum(b[k][0] for k in ks) / n:6.1%}")

    wins = sum(a[k][0] < b[k][0] for k in keys); losses = sum(a[k][0] > b[k][0] for k in keys)
    print(f"\nper clip (nodiac errors): A better on {wins}, B better on {losses}, tie {len(keys) - wins - losses}")

    print("\ndialect markers in hypotheses (nodiac token counts; ref = reference count)")
    tok = lambda rows, f: Counter(w for r in rows.values() for w in normalize(r[f], "nodiac").split())
    ra, ha, hb = tok(A, "ref"), tok(A, "hyp"), tok(B, "hyp")
    for m in MARKERS + ["και"]:
        print(f"  {m:8} ref {ra[m]:4}  A {ha[m]:4}  B {hb[m]:4}")

    for name, rows in (("A", A), ("B", B)):
        print(f"\ntop substitutions {name} (ref -> hyp, nodiac)")
        print("  " + ", ".join(f"{x}->{y} {c}" for (x, y), c in subs(rows)))

    print("\nclips where A is much better / much worse than B (nodiac errors)")
    diff = sorted(keys, key=lambda k: a[k][0] - b[k][0])
    for k in diff[:3] + diff[-3:]:
        print(f"--- {k} ({A[k]['duration_s']:.1f}s) errors A {a[k][0]} B {b[k][0]} / {a[k][1]} words")
        print("  REF:", A[k]["ref"][:300]); print("  A  :", A[k]["hyp"][:300]); print("  B  :", B[k]["hyp"][:300])


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

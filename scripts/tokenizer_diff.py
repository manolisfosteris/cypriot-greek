"""Compare the Qwen3-ASR-0.6B tokenizer with the Reson8/Rumble-2-Flash tokenizer.

Questions:
  1. Which tokens were removed, grouped by writing system?
  2. How were they chosen? (order kept? removed tokens = late/rare BPE merges?)
  3. How many tokens per word does each tokenizer need, per language?

Usage:
  python scripts/tokenizer_diff.py <qwen_dir> <rumble_dir> [--out docs/tokenizer_diff.md]
The dirs hold vocab.json / merges.txt / tokenizer_config.json (download with hf_hub_download, no weights needed).
"""
import argparse
import json
import unicodedata
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer


# ---------------------------------------------------------------- byte-level BPE token string -> raw bytes
def _bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


U2B = _bytes_to_unicode()


def token_text(tok):
    """Decoded text of one vocab entry, or None if it is a partial UTF-8 sequence."""
    try:
        raw = bytes(U2B[ch] for ch in tok)
    except KeyError:
        return tok  # not a byte-level string (special token)
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return None


SCRIPTS = ["CJK", "HIRAGANA", "KATAKANA", "HANGUL", "THAI", "ARABIC", "HEBREW", "CYRILLIC", "GREEK", "LATIN",
           "DEVANAGARI", "BENGALI", "TAMIL", "TELUGU", "GEORGIAN", "ARMENIAN", "ETHIOPIC", "KHMER", "LAO", "MYANMAR",
           "GUJARATI", "GURMUKHI", "KANNADA", "MALAYALAM", "SINHALA", "TIBETAN"]


def char_script(ch):
    name = unicodedata.name(ch, "")
    for s in SCRIPTS:
        if name.startswith(s) or f" {s} " in f" {name} ":
            return "CJK" if s == "CJK" else s
    if "IDEOGRAPH" in name or name.startswith("FULLWIDTH") or name.startswith("IDEOGRAPHIC"):
        return "CJK"
    if ch.isdigit():
        return "DIGIT"
    if ch.isspace():
        return None
    return "SYMBOL/PUNCT"


def token_script(tok):
    text = token_text(tok)
    if text is None:
        return "partial UTF-8 bytes"
    c = Counter(s for s in map(char_script, text) if s)
    if not c:
        return "whitespace"
    letters = {k: v for k, v in c.items() if k not in ("DIGIT", "SYMBOL/PUNCT")}
    if letters:
        return max(letters, key=letters.get)
    return max(c, key=c.get)


# ---------------------------------------------------------------- same sentence in many languages (my own translations)
# "The weather is very nice today, and many people are walking in the park."
SENTENCES = {
    "English": "The weather is very nice today, and many people are walking in the park.",
    "Greek": "Ο καιρός είναι πολύ ωραίος σήμερα και πολλοί άνθρωποι περπατούν στο πάρκο.",
    "Cypriot Greek (approx.)": "Ο καιρός εν πολλά ωραίος σήμμερα τζιαι πολλοί άνθρωποι περπατούν στο πάρκο.",
    "German": "Das Wetter ist heute sehr schön, und viele Leute gehen im Park spazieren.",
    "Dutch": "Het weer is vandaag erg mooi en veel mensen wandelen in het park.",
    "French": "Il fait très beau aujourd'hui et beaucoup de gens se promènent dans le parc.",
    "Polish": "Pogoda jest dziś bardzo ładna i wiele osób spaceruje po parku.",
    "Russian": "Сегодня очень хорошая погода, и много людей гуляет в парке.",
    "Turkish": "Bugün hava çok güzel ve birçok insan parkta yürüyor.",
    "Arabic": "الطقس جميل جدا اليوم، والكثير من الناس يمشون في الحديقة.",
    "Hindi": "आज मौसम बहुत अच्छा है, और बहुत से लोग पार्क में टहल रहे हैं।",
    "Chinese": "今天天气很好，很多人在公园里散步。",
    "Japanese": "今日はとても良い天気で、たくさんの人が公園を散歩しています。",
    "Korean": "오늘은 날씨가 아주 좋아서 많은 사람들이 공원에서 산책하고 있습니다.",
    "Thai": "วันนี้อากาศดีมาก และมีคนจำนวนมากเดินเล่นในสวนสาธารณะ",
    # not in Rumble's 45-language list:
    "Hebrew (not listed)": "מזג האוויר נאה מאוד היום, ואנשים רבים מטיילים בפארק.",
    "Georgian (not listed)": "დღეს ამინდი ძალიან კარგია და ბევრი ადამიანი პარკში სეირნობს.",
    "Armenian (not listed)": "Այսօր եղանակը շատ լավ է, և շատ մարդիկ զբոսնում են այգում։",
    "Tamil (not listed)": "இன்று வானிலை மிகவும் நன்றாக உள்ளது, பலர் பூங்காவில் நடந்து செல்கிறார்கள்.",
}

# Cypriot / medical words relevant to this project
WORDS = ["τζιαι", "σήμμερα", "εν", "κόρη", "Λευκωσία", "μετοπρολόλη", "υπέρταση", "καρδιολόγος", "ευχαριστώ"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("qwen_dir")
    ap.add_argument("rumble_dir")
    ap.add_argument("--out", default="docs/tokenizer_diff.md")
    a = ap.parse_args()

    qv = json.load(open(Path(a.qwen_dir) / "vocab.json", encoding="utf-8"))
    rv = json.load(open(Path(a.rumble_dir) / "vocab.json", encoding="utf-8"))
    qm = Path(a.qwen_dir, "merges.txt").read_text(encoding="utf-8").splitlines()
    rm = Path(a.rumble_dir, "merges.txt").read_text(encoding="utf-8").splitlines()
    qm = [m for m in qm if m and not m.startswith("#version")]
    rm = [m for m in rm if m and not m.startswith("#version")]

    out = []
    p = out.append
    p("# Tokenizer diff: Qwen3-ASR-0.6B vs Reson8/Rumble-2-Flash\n")
    p("Generated by `scripts/tokenizer_diff.py` from each repo's `vocab.json` / `merges.txt`.\n")

    removed = sorted(set(qv) - set(rv), key=qv.get)
    added = sorted(set(rv) - set(qv), key=rv.get)
    kept = sorted(set(qv) & set(rv), key=qv.get)
    p("## 1. Counts\n")
    p("| | count |\n|---|---|")
    p(f"| Qwen base vocab (vocab.json) | {len(qv):,} |")
    p(f"| Rumble base vocab (vocab.json) | {len(rv):,} |")
    p(f"| kept (same token string in both) | {len(kept):,} |")
    p(f"| removed (only in Qwen) | {len(removed):,} |")
    p(f"| added (only in Rumble) | {len(added):,} |")
    p(f"| Qwen merges | {len(qm):,} |")
    p(f"| Rumble merges | {len(rm):,} |")
    rm_set, qm_set = set(rm), set(qm)
    p(f"| Rumble merges that are also Qwen merges | {len(rm_set & qm_set):,} |\n")
    if added:
        p("Added tokens (first 30): " + ", ".join(repr(token_text(t)) for t in added[:30]) + "\n")

    # ---- order / renumbering
    p("## 2. How were tokens chosen?\n")
    rum_ids = [rv[t] for t in kept]  # kept sorted by Qwen id
    monotone = all(x < y for x, y in zip(rum_ids, rum_ids[1:]))
    dense = sorted(rum_ids) == list(range(len(kept))) if kept else False
    p(f"- Kept tokens keep their relative order (Qwen id order == Rumble id order): **{monotone}**")
    p(f"- Rumble ids of kept tokens are exactly 0..{len(kept)-1} (a dense renumbering): **{dense}**")
    # merges order
    q_rank = {m: i for i, m in enumerate(qm)}
    shared = [m for m in rm if m in q_rank]
    m_mono = all(q_rank[x] < q_rank[y] for x, y in zip(shared, shared[1:]))
    p(f"- Rumble's merges appear in the same order as in Qwen's merge list: **{m_mono}**")
    # all 256 byte tokens kept?
    byte_toks = list(U2B.keys())
    p(f"- All 256 single-byte tokens kept (any text can still be encoded): **{all(b in rv for b in byte_toks)}**\n")

    # removed vs id position: BPE ids follow merge order, late merges = rarer pieces in Qwen's training data
    p("Share of tokens removed, by position in Qwen's vocabulary (low id = early, frequent merge; high id = late, rarer merge):\n")
    p("| Qwen id range | tokens | removed | % removed |\n|---|---|---|---|")
    rset = set(removed)
    step = 15000
    for lo in range(0, max(qv.values()) + 1, step):
        toks = [t for t, i in qv.items() if lo <= i < lo + step]
        r = sum(t in rset for t in toks)
        p(f"| {lo:,}-{lo+step-1:,} | {len(toks):,} | {r:,} | {100*r/max(1,len(toks)):.1f}% |")
    p("")

    # ---- by script
    p("## 3. Removed vs kept by writing system\n")
    q_scr = {t: token_script(t) for t in qv}
    tot = Counter(q_scr.values())
    rem = Counter(q_scr[t] for t in removed)
    p("| script | Qwen tokens | removed | % of that script removed | share of all removed |\n|---|---|---|---|---|")
    for s, n in tot.most_common():
        r = rem.get(s, 0)
        p(f"| {s} | {n:,} | {r:,} | {100*r/n:.1f}% | {100*r/max(1,len(removed)):.1f}% |")
    p("")
    for s in ["CJK", "LATIN", "GREEK", "CYRILLIC", "HANGUL", "THAI", "ARABIC"]:
        ex_r = [token_text(t) for t in removed if q_scr[t] == s][:15]
        ex_k = [token_text(t) for t in kept if q_scr[t] == s][-15:]
        if ex_r:
            p(f"- **{s}** removed examples: " + ", ".join(repr(x) for x in ex_r))
            p(f"  - kept examples (latest merges kept): " + ", ".join(repr(x) for x in ex_k))
    p("")

    # ---- fertility
    p("## 4. Tokens needed for the same sentence\n")
    qt = AutoTokenizer.from_pretrained(a.qwen_dir)
    rt = AutoTokenizer.from_pretrained(a.rumble_dir)
    p("Same sentence (\"The weather is very nice today, and many people are walking in the park.\") in each language. "
      "Translations are mine and approximate; the point is the token count, not the wording.\n")
    p("| language | words | Qwen tokens | Rumble tokens | change | round-trip OK |\n|---|---|---|---|---|---|")
    for lang, s in SENTENCES.items():
        qi = qt.encode(s, add_special_tokens=False)
        ri = rt.encode(s, add_special_tokens=False)
        ok = rt.decode(ri) == s
        w = len(s.split())
        p(f"| {lang} | {w} | {len(qi)} | {len(ri)} | {100*(len(ri)-len(qi))/len(qi):+.0f}% | {ok} |")
    p("")
    p("Single words (Cypriot dialect + medical terms):\n")
    p("| word | Qwen pieces | Rumble pieces |\n|---|---|---|")
    for wd in WORDS:
        s = " " + wd
        qp = [qt.decode([i]) for i in qt.encode(s, add_special_tokens=False)]
        rp = [rt.decode([i]) for i in rt.encode(s, add_special_tokens=False)]
        p(f"| {wd} | {' · '.join(qp)} | {' · '.join(rp)} |")
    p("")

    text = "\n".join(out)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()

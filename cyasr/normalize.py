"""Text normalization ladder for scoring Cypriot Greek transcripts.

Levels are cumulative: each level applies everything below it plus one more step.
    raw      whitespace collapse only
    basic    NFC, lowercase, final sigma, brackets/punctuation/apostrophes removed
    nofill   basic + hesitation fillers (εεε, εμ, εεεμ, ...) dropped
    nodiac   nofill + all combining marks stripped, Latin look-alikes mapped to Greek
    phonetic nodiac + homophone spellings collapsed (lenient diagnostic, not a headline metric)
"""
import html
import re
import unicodedata

LEVELS = ["raw", "basic", "nofill", "nodiac", "phonetic"]

# Latin letters the corpus uses inside Greek words (e.g. "τžαι" for "τζαι").
LATIN_TO_GREEK = {"ž": "ζ", "š": "σ"}

# Hesitations seen in the corpus: ε, εε, εεε, εμ, εεεμ, μμμ, ααα. Single "α" and "μ" are kept.
FILLER_RE = re.compile(r"ε+μ*|ε*μ{2,}|α{2,}μ*")

# Hyphens and dashes separate words ("σιγά-σιγά"); other punctuation is deleted.
_DASHES = "-–—‐"
CONSONANTS = "βγδζθκλμνξπρστφχψ"


def _basic(text: str) -> str:
    text = html.unescape(text)  # the TSV contains literal "&#x27;"
    text = unicodedata.normalize("NFC", text).lower().replace("ς", "σ")
    out = []
    for ch in text:
        cat = unicodedata.category(ch)
        if ch in _DASHES:
            out.append(" ")
        elif cat[0] in "PSC" or ch in "΄´":
            # Letters inside brackets are kept because only the bracket itself is removed.
            # A stray tonos like "κοντιν΄ον" is dropped too.
            out.append(" " if ch in ".,;:!?…" else "")
        else:
            out.append(ch)
    return " ".join("".join(out).split())


def _nofill(text: str) -> str:
    return " ".join(w for w in _basic(text).split() if not FILLER_RE.fullmatch(w))


def _nodiac(text: str) -> str:
    text = "".join(LATIN_TO_GREEK.get(ch, ch) for ch in _nofill(text))
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return unicodedata.normalize("NFC", text)


def _phonetic(text: str) -> str:
    text = _nodiac(text)
    # Protect diphthongs that are not homophones of the vowels below.
    for i, d in enumerate(["αυ", "ευ", "ου"]):
        text = text.replace(d, chr(1 + i))
    text = text.replace("αι", "ε")
    for pattern in ["ει", "οι", "υι", "η", "υ"]:
        text = text.replace(pattern, "ι")
    text = text.replace("ω", "ο")
    for i, d in enumerate(["αυ", "ευ", "ου"]):
        text = text.replace(chr(1 + i), d)
    return re.sub(f"([{CONSONANTS}])\\1", r"\1", text)  # double consonants -> single


def normalize(text: str, level: str = "nodiac") -> str:
    """Normalize `text` at one of LEVELS."""
    if level == "raw":
        return " ".join(text.split())
    funcs = {"basic": _basic, "nofill": _nofill, "nodiac": _nodiac, "phonetic": _phonetic}
    if level not in funcs:
        raise ValueError(f"unknown level {level!r}; choose from {LEVELS}")
    return funcs[level](text)


# --- Training targets: Cypriot orthography kept, but made consistent ---

BREVE = "̆"
# Latin letters standing in for the Cypriot mark. Mapped before NFD, which would split "ž" into z + caron.
LATIN_TO_CYPRIOT = {"ž": "ζ" + BREVE, "š": "σ" + BREVE}
# Combining marks (after NFD): the caron becomes the breve, grave becomes acute, dot-above is dropped.
MARK_MAP = {"̌": BREVE, "̀": "́", "̀": "́", "͘": ""}
APOSTROPHES = {"’": "'", "‘": "'"}
LOOSE_TONOS = "΄´"
GREEK_VOWELS = "αεηιουωΑΕΗΙΟΥΩ"


def _strip_csv_quotes(text: str) -> str:
    # 35 transcripts are wrapped in quotes with inner quotes doubled: "... την ""μέτταλ""."
    text = text.strip()
    if len(text) > 1 and text[0] == text[-1] == '"':
        text = text[1:-1].replace('""', '"')
    return text


def _fix_loose_tonos(text: str) -> str:
    # A tonos typed before its vowel ("ο΄υλλα", "΄Ε") goes onto that vowel; any other is dropped.
    text = re.sub(f"[{LOOSE_TONOS}]+([{GREEK_VOWELS}])", lambda m: m.group(1) + "́", text)
    return re.sub(f"[{LOOSE_TONOS}]", "", text)


def clean_target(text: str) -> str:
    """Clean a transcript for use as a training target (option a: readable Cypriot spelling).

    Keeps accents, capitals, punctuation, Cypriot marks, fillers and false starts. Only fixes
    things that are inconsistent or not speech: the three spellings of the Cypriot mark,
    CSV quotes, HTML codes, apostrophe variants, misplaced accents and bracket markers.
    """
    text = _strip_csv_quotes(html.unescape(text))
    text = text.replace("[...]", " ")
    text = re.sub(r"[()\[\]]", "", text)  # "(Σ)τα" -> "Στα": keep the letters, drop the brackets
    text = "".join(LATIN_TO_CYPRIOT.get(ch, APOSTROPHES.get(ch, ch)) for ch in text)
    text = _fix_loose_tonos(unicodedata.normalize("NFC", text))
    text = unicodedata.normalize("NFD", text)
    text = "".join(MARK_MAP.get(ch, ch) for ch in text)
    return " ".join(unicodedata.normalize("NFC", text).split())

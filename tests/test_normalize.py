import pytest

from cyasr.normalize import LEVELS, clean_target, normalize


def test_breve_caron_and_latin_z_collapse_at_nodiac():
    variants = ["τζ̆αι", "τžαι", "τζαι", "Τžαι"]
    assert len({normalize(v, "nodiac") for v in variants}) == 1
    assert normalize("τžαι", "nodiac") == "τζαι"


def test_basic_keeps_letters_in_brackets():
    assert normalize("(Σ)τα", "basic") == "στα"
    assert normalize("τού(τ)ον", "basic") == "τούτον"


def test_basic_punctuation_apostrophes_and_final_sigma():
    assert normalize("Εν' ΟΔΟΣ, «ναι»!", "basic") == "εν οδοσ ναι"
    assert normalize("'ντάξει.", "basic") == "ντάξει"
    assert normalize("εν&#x27; πολλά", "basic") == "εν πολλά"
    assert normalize("σιγά-σιγά", "basic") == "σιγά σιγά"


def test_stray_tonos_modifier_is_removed():
    assert normalize("κοντιν΄ον", "basic") == "κοντινον"
    assert normalize("ενν΄΄΄΄α", "basic") == "εννα"


def test_basic_keeps_accents_nodiac_strips_them():
    assert normalize("ευχαριστώ", "basic") == "ευχαριστώ"
    assert normalize("ευχαριστώ ΐ", "nodiac") == "ευχαριστω ι"


def test_fillers_dropped_at_nofill_only():
    assert normalize("εεεμ", "nofill") == ""
    assert normalize("Εεε, ναι εμ, ε", "nofill") == "ναι"
    assert normalize("εεεμ ναι", "basic") == "εεεμ ναι"


def test_real_words_are_not_fillers():
    text = "με ο α μα ένας"
    assert normalize(text, "nofill") == "με ο α μα ένασ"


def test_english_words_survive_lowercased():
    assert normalize("Online banking, ChatGPT.", "phonetic") == "online banking chatgpt"
    assert normalize("COVID", "nodiac") == "covid"


def test_phonetic_keeps_diphthongs_ev_av_ou():
    out = normalize("ευχαριστώ", "phonetic")
    assert out.startswith("ευ") and out == "ευχαριστο"
    assert normalize("αυτός", "phonetic") == "αυτοσ"
    assert normalize("ούτε", "phonetic").startswith("ου")


def test_phonetic_collapses_homophones():
    assert normalize("πηγαίνω", "phonetic") == normalize("πιγένο", "phonetic")
    assert normalize("κοιτάζω", "phonetic") == normalize("κιτάζο", "phonetic")
    assert normalize("λλίον", "phonetic") == normalize("λίον", "phonetic")


def test_levels_are_cumulative_on_a_real_string():
    s = "Εεε, τžαι (Σ)τα έντž' ευχαριστώ."
    outs = [normalize(s, lv) for lv in LEVELS]
    assert outs[0] == s
    assert outs[1].startswith("εεε")
    assert outs[2].startswith("τžαι")
    assert outs[3] == "τζαι στα εντζ ευχαριστω"


def test_unknown_level_raises():
    with pytest.raises(ValueError):
        normalize("x", "bogus")


# --- clean_target: training text ---

def test_clean_target_unifies_cypriot_mark_to_breve():
    for variant in ["τζ̆αι", "τζ̌αι", "τžαι"]:
        assert clean_target(variant) == "τζ̆αι"
    assert clean_target("σ̌ει") == "σ̆ει"


def test_clean_target_strips_csv_quotes_keeps_inner_quotes():
    assert clean_target('"Την ""ιντάστριαλ μέτταλ""."') == 'Την "ιντάστριαλ μέτταλ".'


def test_clean_target_moves_loose_tonos_onto_vowel():
    assert clean_target("ο΄υλλα") == "ούλλα"
    assert clean_target("΄Ελα") == "Έλα"
    assert clean_target("ενν΄΄΄΄α") == "εννά"


def test_clean_target_brackets_and_omission_marker():
    assert clean_target("(Σ)τα γενέθλια") == "Στα γενέθλια"
    assert clean_target("τούτ- [...] τούτα") == "τούτ- τούτα"


def test_clean_target_apostrophes_and_html():
    assert clean_target("&#x27;ντάξει εν’") == "'ντάξει εν'"


def test_clean_target_keeps_spelling_fillers_and_false_starts():
    text = "Εεεμ, που επέ- που εκά- θθυμούμαι την μάμμα μου."
    assert clean_target(text) == text

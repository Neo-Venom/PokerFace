"""Unit tests for Step C: Unicode normalization, zero-width removal, transliteration, and offset maps."""
from raksha.detect.identifiers import detect_identifier_spans, redact_spans
from raksha.detect.normalize import (
    build_normalized_view,
    deva_word_to_latin,
    latin_word_to_deva,
    phonetic_key,
)


def test_step_c_zero_width_and_homoglyph_offset_map():
    # Phone split with zero-width spaces + Devanagari digits + Cyrillic 'А' in PAN
    orig = "Keep this. Mob: +91\u200b ७२८६४\u200c ९०६५८ PAN: АBCDE1234F end."
    nv = build_normalized_view(orig)
    assert "\u200b" not in nv.clean
    assert "72864 90658" in nv.clean
    assert "ABCDE1234F" in nv.clean

    out = redact_spans(orig, detect_identifier_spans(orig, view=nv))
    assert out == "Keep this. Mob: [PHONE] PAN: [PAN] end."


def test_step_c_transliteration_and_folded_offset_map():
    orig = "नाम: मोहम्मद शाहिद को बुखार है"
    nv = build_normalized_view(orig)
    assert "mohammad" in nv.latin_folded
    assert "shahid" in nv.latin_folded
    idx = nv.latin_folded.index("mohammad")
    os_, oe_ = nv.map_folded_span(idx, idx + len("mohammad"))
    assert orig[os_:oe_] == "मोहम्मद"


def test_step_c_cross_script_phonetic_key():
    assert phonetic_key("मोहम्मद शाहिद") == phonetic_key("Mohd. Shahid")
    assert phonetic_key("अनीता शर्मा") == phonetic_key("Aneeta Sharma")
    assert phonetic_key("रुख़साना सिद्दीक़ी") == phonetic_key("Rukhsana Siddiqi")
    assert deva_word_to_latin("कमल") == "kamal"
    assert "कमल" in latin_word_to_deva("kamal")

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
    orig = "Keep this. Mob: +91\u200b ९८७६५\u200c ४३२१० PAN: АBCDE1234F end."
    nv = build_normalized_view(orig)
    assert "\u200b" not in nv.clean
    assert "98765 43210" in nv.clean
    assert "ABCDE1234F" in nv.clean

    out = redact_spans(orig, detect_identifier_spans(orig, view=nv))
    assert out == "Keep this. Mob: [PHONE] PAN: [PAN] end."


def test_step_c_transliteration_and_folded_offset_map():
    orig = "नाम: राजीव कुलकर्णी को बुखार है"
    nv = build_normalized_view(orig)
    assert "rajiv" in nv.latin_folded
    assert "kulakarni" in nv.latin_folded
    idx = nv.latin_folded.index("kulakarni")
    os_, oe_ = nv.map_folded_span(idx, idx + len("kulakarni"))
    assert orig[os_:oe_] == "कुलकर्णी"


def test_step_c_cross_script_phonetic_key():
    assert phonetic_key("मोहम्मद तनवीर") == phonetic_key("Mohd. Tanveer")
    assert phonetic_key("प्रियंका देसाई") == phonetic_key("Priyanka Desai")
    assert phonetic_key("सौरभ मुखर्जी") == phonetic_key("Saurabh Mukherjee") or phonetic_key("सौरभ") == phonetic_key("Saurabh")
    assert deva_word_to_latin("रोहन") == "rohan"
    assert "रोह" in latin_word_to_deva("rohan")

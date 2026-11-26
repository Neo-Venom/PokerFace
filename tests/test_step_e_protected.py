"""Unit tests for Step E: protected clinical spans and overlap conflict resolution."""
from raksha.detect.identifiers import detect_identifier_spans, redact_spans
from raksha.detect.names_context import detect_context_name_and_address_spans
from raksha.detect.protected import ClinicalLexicon, build_clinical_lexicon, resolve_pii_vs_protected


def test_step_e_nursing_note_temp_and_paracetamol_preserved():
    txt = (
        "NURSING NOTES - Ward 8 (Medicine), Bed 2 - PAT000991\n"
        "01/13/2026 22:00 - Tanveer sahab ka BP 117/82, pulse 106. Bukhar 103 F. Paracetamol 650 mg diya.\n"
        "14:00 - Mohd. Tanveer - K/C/O addison's disease. Attendant (S/o Farhan) informed about diet."
    )
    lex = ClinicalLexicon()
    prot = lex.find_spans(txt)
    pii = detect_identifier_spans(txt) + detect_context_name_and_address_spans(txt, protected_spans=prot)
    resolved = resolve_pii_vs_protected(pii, prot)
    out = redact_spans(txt, resolved)

    assert "103 F. Paracetamol 650 mg" in out
    assert "BP 117/82" in out
    assert "pulse 106" in out
    assert "addison's disease" in out
    assert "Ward 8 (Medicine)" in out
    assert "Tanveer" not in out
    assert "Farhan" not in out
    assert "PAT000991" not in out


def test_step_e_formulary_lexicon_and_protected_overlap_wins():
    parsed = [
        ("clean/drug_formulary.csv", "csv", [["drug", "strength"], ["Dolo", "650 mg"], ["Amoxycillin", "500 mg"]]),
        ("clean/lab_reference_ranges.csv", "csv", [["test", "unit"], ["Platelet Count", "/uL"]]),
    ]
    lex = build_clinical_lexicon(parsed)
    txt = "Dolo 650 BD x 5 days. Platelet Count 240124 /uL. Dr. Vivek Menon reviewed."
    prot = lex.find_spans(txt)
    pii = detect_identifier_spans(txt) + detect_context_name_and_address_spans(txt, protected_spans=prot)
    out = redact_spans(txt, resolve_pii_vs_protected(pii, prot))
    assert out == txt

"""Unit tests for Step D: context-aware free-text names, C/O ambiguity, doctors, and eponyms."""
from raksha.detect.identifiers import redact_spans
from raksha.detect.names_context import detect_context_name_and_address_spans


def _scrub(text: str) -> str:
    return redact_spans(text, detect_context_name_and_address_spans(text))


def test_step_d_care_of_disambiguation():
    complaint = "Presenting complaints: C/O fever for 3 days, C/O pain abdomen. K/C/O addison's disease."
    assert _scrub(complaint) == complaint

    care_of = "Address (attendant): C/O Devendra Kulkarni, H.No. 364, Ward 20, Mohalla Shastri Nagar"
    out = _scrub(care_of)
    assert "Devendra Kulkarni" not in out
    assert "[RELATIVE_NAME]" in out
    assert "[HOUSE]" in out
    assert "[LOCALITY]" in out


def test_step_d_doctors_eponyms_and_asha_preserved():
    txt = (
        "Consultant: Dr. Vivek Menon (MD Medicine). "
        "K/C/O Parkinson's disease and Bell's palsy. "
        "Referred by ASHA worker. Attendant: Vivek Gokhale. "
        "Dr. डॉ प्रदीप reviewed the patient."
    )
    out = _scrub(txt)
    assert "Dr. Vivek Menon" in out
    assert "Parkinson's disease" in out
    assert "Bell's palsy" in out
    assert "ASHA worker" in out
    assert "Vivek Gokhale" not in out
    assert "डॉ प्रदीप" in out


def test_step_d_multilingual_triggers_hinglish_and_hindi():
    txt = (
        "नाम: प्रवीण जोशी\n"
        "प्रवीण जोशी को 9 दिन से बुखार है।\n"
        "Tanveer sahab ka BP 117/82. Pallavi ji ki condition stable hai.\n"
        "[11/09/26, 8:13 pm] Saurabh Bhaiya: Report of Nandini behenji 65 Y / F\n"
        "S/o Raghuveer Deshmukh. Smt. Meera Iyer visited."
    )
    out = _scrub(txt)
    for leaked in ("प्रवीण जोशी", "Tanveer", "Pallavi", "Saurabh Bhaiya", "Nandini", "Raghuveer Deshmukh", "Meera Iyer"):
        assert leaked not in out
    assert "117/82" in out


def test_step_d_pt_prefix_and_unseen_labels_preserve_lowercase_prose():
    txt = "Pt. Zorath Qvendal (MRN: CAN-99) arrived. Pt allergic to sulfa drugs."
    out = _scrub(txt)
    assert "Zorath Qvendal" not in out
    assert "Pt allergic to sulfa drugs." in out


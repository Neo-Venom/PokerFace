"""Unit tests for Step D: context-aware free-text names, C/O ambiguity, doctors, and eponyms."""
from raksha.detect.identifiers import redact_spans
from raksha.detect.names_context import detect_context_name_and_address_spans


def _scrub(text: str) -> str:
    return redact_spans(text, detect_context_name_and_address_spans(text))


def test_step_d_care_of_disambiguation():
    complaint = "Presenting complaints: C/O fever for 3 days, C/O pain abdomen. K/C/O addison's disease."
    assert _scrub(complaint) == complaint

    care_of = "Address (attendant): C/O Ramesh Kumar, H.No. 364, Ward 20, Mohalla Raja Bazar"
    out = _scrub(care_of)
    assert "Ramesh Kumar" not in out
    assert "[RELATIVE_NAME]" in out
    assert "[HOUSE]" in out
    assert "[LOCALITY]" in out


def test_step_d_doctors_eponyms_and_asha_preserved():
    txt = (
        "Consultant: Dr. Ramesh Srivastava (MD Medicine). "
        "K/C/O Parkinson's disease and Bell's palsy. "
        "Referred by ASHA worker. Attendant: Ramesh Patel. "
        "Dr. डॉ रमेश reviewed the patient."
    )
    out = _scrub(txt)
    assert "Dr. Ramesh Srivastava" in out
    assert "Parkinson's disease" in out
    assert "Bell's palsy" in out
    assert "ASHA worker" in out
    assert "Ramesh Patel" not in out
    assert "डॉ रमेश" in out


def test_step_d_multilingual_triggers_hinglish_and_hindi():
    txt = (
        "नाम: कमल ठाकुर\n"
        "कमल ठाकुर को 9 दिन से बुखार है।\n"
        "Shahid sahab ka BP 117/82. Sunita ji ki condition stable hai.\n"
        "[11/09/26, 8:13 pm] Ajay Bhaiya: Report of Reena behenji 65 Y / F\n"
        "S/o Naresh Chauhan. Smt. Meera Devi visited."
    )
    out = _scrub(txt)
    for leaked in ("कमल ठाकुर", "Shahid", "Sunita", "Ajay Bhaiya", "Reena", "Naresh Chauhan", "Meera Devi"):
        assert leaked not in out
    assert "117/82" in out

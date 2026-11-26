"""Unit tests for Step A (structured fields) and Step B (identifier regex & numeric context)."""
from raksha.detect.identifiers import detect_identifier_spans, redact_spans
from raksha.detect.structured import (
    infer_column_type,
    sanitize_header_or_key,
    sanitize_json_data,
)


def test_step_a_value_first_column_inference_ignores_misspelled_headers():
    assert infer_column_type(["902115225016", "1234 5678 9012"], header="col_x") == "AADHAAR"
    assert infer_column_type(["+91 72864 90658", "8081120924"], header="samprk_no") == "PHONE"
    assert infer_column_type(["1977-04-12", "1985-01-09"], header="x_dt") == "DOB"
    assert infer_column_type(["2026-04-12", "2026-05-09"], header="visit_date") is None


def test_step_a_numeric_aadhaar_in_json():
    doc = {"patients": [{"uhid": "ABC-12345", "ids": {"aadhar_typo": 902115225016}}]}
    out = sanitize_json_data(doc)
    assert out["patients"][0]["uhid"] == "[MRN]"
    assert out["patients"][0]["ids"]["aadhar_typo"] == "[AADHAAR]"


def test_step_a_key_collision_and_facility_preserved():
    doc = {
        "facility": {"name": "District Hospital"},
        "by_name": {"Alice Kumar": "HOSP-10001", "Bob Nair": "HOSP-10002"},
    }
    out = sanitize_json_data(doc)
    assert out["facility"]["name"] == "District Hospital"
    assert "[PATIENT_NAME]" in out["by_name"]
    assert "[PATIENT_NAME]_2" in out["by_name"]


def test_step_b_phone_formats_including_devanagari_and_ext():
    txt = "Mob: +91 72864 90658, alt ८५९९२५३५२५, office 9876543210 ext 204"
    out = redact_spans(txt, detect_identifier_spans(txt))
    assert out == "Mob: [PHONE], alt [PHONE], office [PHONE]"


def test_step_b_numeric_context_preserves_platelets_doses_and_vitals():
    txt = (
        "Platelet Count 240124 /uL (range 150000-410000). "
        "BP 150/90 mmHg. Dolo 650 mg 1-0-1 x 5 days. "
        "Aadhaar 3972 0583 9628, ABHA 91-7938-3729-8403, PAN ABCDE1234F."
    )
    out = redact_spans(txt, detect_identifier_spans(txt))
    assert "240124 /uL" in out
    assert "150000-410000" in out
    assert "150/90" in out
    assert "650 mg" in out
    assert "1-0-1" in out
    assert "[AADHAAR]" in out
    assert "[ABHA_NUMBER]" in out
    assert "[PAN]" in out

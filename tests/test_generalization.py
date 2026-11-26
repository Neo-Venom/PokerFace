"""Generalization tests: contextual canaries & FP guard, compositional schema inference, NFC/NFD Unicode, and content-based clean classification."""
from __future__ import annotations

import json
import unicodedata
from pathlib import Path

from raksha.detect.crf_model import CRFPredictor
from raksha.detect.identifiers import detect_identifier_spans, redact_spans
from raksha.detect.names_context import detect_context_name_and_address_spans
from raksha.detect.protected import build_clinical_lexicon, resolve_pii_vs_protected
from raksha.detect.registry import build_alias_registry
from raksha.detect.structured import (
    infer_table_schema,
    is_non_patient_json_content,
    is_non_patient_table_content,
    process_table_rows,
    sanitize_json_data,
)
from raksha.export.filenames import sanitize_relative_path
from raksha.identity.evidence import extract_all_evidences, is_non_patient_text_content
from raksha.identity.linker import resolve_clusters
from raksha.ingest.loader import decode_bytes
from raksha.pipeline import run_pipeline


def _scrub(text: str, extra_items=None) -> str:
    lex = build_clinical_lexicon(extra_items or [])
    crf = CRFPredictor()
    prot = lex.find_spans(text)
    ids = detect_identifier_spans(text)
    ctx = detect_context_name_and_address_spans(text, protected_spans=prot)
    c_sp = crf.predict_spans(text) if crf.available else []
    merged = resolve_pii_vs_protected(ids + ctx + c_sp, prot)
    return redact_spans(text, merged)


def test_contextual_canaries_caught_while_clinical_alphanumeric_preserved():
    note = (
        "रेफरल नोट: Rapihivx0 Xibojoqz (मोबाइल: 9800000000)\n"
        "Patient: Zorath Qvendal came for follow up.\n"
        "Attendant Xipewizx21 Fadaziqz (CAN-31-900877) ne consent sign kiya.\n"
        "Signed consent of Lufewucx8 Surofezq witnessed by Dr. Kavita Rao.\n"
        "Rx: Dolo 650 mg SOS, Tab Paracetamol 500 mg BD, Vitamin B12 310 pg/mL.\n"
        "Labs: HbA1c 7.4%, SpO2 98%, HER2 negative, CA125 18 U/mL, BRCA1 wild-type.\n"
        "Admitted Ward 5, Bed 12. Covid 19 negative. ICD A15.0 Stage 3 CKD Phase 2."
    )
    out = _scrub(note)
    for secret_tok in ("Rapihivx0", "Xibojoqz", "Zorath Qvendal", "Xipewizx21", "Fadaziqz", "Lufewucx8", "Surofezq", "9800000000", "CAN-31-900877"):
        assert secret_tok not in out, f"Leaked: {secret_tok}\n{out}"
    assert "witnessed" in out
    assert "Dr. Kavita Rao" in out
    for clinical in ("Dolo 650 mg", "Paracetamol 500 mg", "Vitamin B12", "HbA1c 7.4%", "SpO2 98%", "HER2", "CA125", "BRCA1", "Ward 5", "Bed 12", "Covid 19", "A15.0", "Stage 3", "Phase 2"):
        assert clinical in out, f"Clinical false positive in: {clinical}\n{out}"


def test_unfamiliar_csv_columns_and_nested_json_inference_preserves_clinical():
    header = [
        "member_full_label", "client_enroll_tag", "home_quarter", "living_colony",
        "residency_pin", "triage_recorded_dt", "clinician_author", "custom_opaque_ref",
        "clinical_status", "hba1c_val",
    ]
    body = [
        ["Meera Kulkarni", "XYZ-29-104421", "Qtr-14", "Sector 7 Colony", "411038", "2026-04-10", "Dr. Neeraj Shah", "opaque_alpha_one", "stable", "7.2"],
        ["Rajan Deshmukh", "XYZ-29-104422", "Qtr-29", "Sector 3 Colony", "411039", "2026-04-11", "Dr. Sunita Bose", "opaque_beta_two", "stable", "6.8"],
        ["Farhan Siddiqui", "XYZ-29-104423", "Qtr-88", "Sector 9 Colony", "411040", "2026-04-12", "Dr. Neeraj Shah", "opaque_gamma_three", "improved", "5.9"],
    ]
    ts = infer_table_schema(header, body)
    assert ts.col_types[0] == "PATIENT_NAME"
    assert ts.col_types[1] == "MRN"
    assert ts.col_types[2] == "HOUSE"
    assert ts.col_types[3] == "LOCALITY"
    assert ts.col_types[4] == "PIN"
    assert ts.col_types[6] is None  # clinician preserved
    assert ts.col_types[7] == "REDACTED_PII"  # uncertain non-clinical string masked
    assert ts.col_types[8] is None and ts.col_types[9] is None  # clinical preserved

    _ch, clean_body = process_table_rows([header] + body, text_scrubber=_scrub)
    assert clean_body[0][0] == "[PATIENT_NAME]"
    assert clean_body[0][1] == "[MRN]"
    assert clean_body[0][2] == "[HOUSE]"
    assert clean_body[0][3] == "[LOCALITY]"
    assert clean_body[0][6] == "Dr. Neeraj Shah"
    assert clean_body[0][7] == "[REDACTED_PII]"
    assert clean_body[0][8] == "stable" and clean_body[0][9] == "7.2"

    nested = {
        "_meta": {"source_file": r"D:\Exports\opd\Shahid_NA_XYZ-29-104421\visits.json"},
        "beneficiary_name": {"first_tok": "Meera", "last_tok": "Kulkarni"},
        "panels": [{"name": "Routine", "results": [{"test": "HbA1c", "value": "7.2"}]}],
        "diagnosis": [{"code": "E11.9", "display": "Wilson's disease"}],
        "by_beneficiary": {"Zubin0 CanaryOne": "XYZ-29-999999"},
    }
    clean_json = sanitize_json_data(nested, text_scrubber=_scrub)
    dumped = json.dumps(clean_json, ensure_ascii=False)
    for leaked in ("Shahid", "XYZ-29-104421", "Meera", "Kulkarni", "Zubin0 CanaryOne", "XYZ-29-999999"):
        assert leaked not in dumped
    assert clean_json["panels"][0]["name"] == "Routine"
    assert clean_json["diagnosis"][0]["display"] == "Wilson's disease"


def test_unfamiliar_schema_links_same_patient_without_wrong_homonym_merge():
    parsed = [
        (
            "intake_a.csv",
            "csv",
            [
                ["member_full_label", "client_enroll_tag", "birth_dt", "age_at_visit", "sex_code"],
                ["Élodie Bernard", "NVP27-110421", "1986-05-10", "40", "F"],
                ["Rahul Verma", "NVP27-110888", "1990-01-01", "36", "M"],
                ["Rahul Verma", "NVP27-110999", "1964-08-12", "62", "M"],
            ],
        ),
        (
            "visit_b.json",
            "json",
            {
                "records": [
                    {
                        "beneficiary": {"given_nm": "E\u0301lodie", "sur_nm": "Bernard"},
                        "ref_code": "NVP27-110421",
                        "gender_val": "F",
                        "visit_dt": "2026-04-12",
                    },
                    {
                        "beneficiary": {"given_nm": "Rahul", "sur_nm": "Verma"},
                        "ref_code": "NVP27-110888",
                        "gender_val": "M",
                        "visit_dt": "2026-04-12",
                    },
                ]
            },
        ),
    ]
    evs = extract_all_evidences(parsed)
    res = resolve_clusters(evs)
    assert res.record_to_cluster[("intake_a.csv", "row:0")] == res.record_to_cluster[("visit_b.json", "/records/0")]
    assert res.record_to_cluster[("intake_a.csv", "row:1")] == res.record_to_cluster[("visit_b.json", "/records/1")]
    # Homonym Rahul Verma with distinct MRN + 26-year age gap MUST NOT merge
    assert res.record_to_cluster[("intake_a.csv", "row:2")] != res.record_to_cluster[("intake_a.csv", "row:1")]


def test_unicode_composed_and_decomposed_accents_and_cp1252_decoding():
    nfd_name = unicodedata.normalize("NFD", "Amélie Françoise")
    nfc_name = unicodedata.normalize("NFC", "Amélie Françoise")
    assert len(nfd_name) != len(nfc_name)

    parsed = [("pt.csv", "csv", [["patient_name", "mrn"], [nfc_name, "ABC-27-555123"]])]
    reg = build_alias_registry(parsed)

    txt = f"Patient: {nfd_name} visited today. Report of {nfc_name} 34 Y/F."
    out_txt = redact_spans(txt, resolve_pii_vs_protected(detect_context_name_and_address_spans(txt) + reg.propose_spans(txt), []))
    assert "Amélie" not in unicodedata.normalize("NFC", out_txt)
    assert "Françoise" not in unicodedata.normalize("NFC", out_txt)

    out_fn = sanitize_relative_path(f"notes/hinglish_{nfd_name.replace(' ', '_')}_01.txt", input_sha256="0123456789abcdef", registry=reg)
    folded_fn = unicodedata.normalize("NFKD", out_fn).casefold()
    assert "amelie" not in folded_fn and "francoise" not in folded_fn
    assert "PATIENT_NAME" in out_fn

    # Valid cp1252 accented French characters decode cleanly, while binary mojibake still fails
    valid_cp1252 = "Patient: Élodie Payet".encode("cp1252")
    assert "Élodie Payet" in decode_bytes(valid_cp1252)
    try:
        decode_bytes(b"OPD note \xff\xfe\xc3( broken")
        assert False, "Expected ValueError on corrupt bytes"
    except ValueError as exc:
        assert str(exc) == "invalid_encoding"


def test_content_based_clean_classification_replaces_path_heuristic(tmp_path: Path, monkeypatch):
    # Static non-patient files OUTSIDE clean/
    assert is_non_patient_table_content(["drug", "category"], [["Metformin", "Antidiabetic"]])
    assert is_non_patient_json_content({"hospital": "District Hosp", "wards": [{"ward": "ICU", "beds": 10}]})
    assert is_non_patient_text_content("Sepsis protocol: inj Ceftriaxone 1g IV BD within 1 hour.")

    # Patient files INSIDE clean/
    assert not is_non_patient_table_content(["pt_full", "age", "uhid"], [["Aarti Mehta", "45", "KMM/27/112233"]])
    assert not is_non_patient_text_content("Patient: Aarti Mehta, Age: 45 Y / F. Date: 2026-03-10.")

    inp = tmp_path / "in"
    out = tmp_path / "out"
    (inp / "protocols").mkdir(parents=True)
    (inp / "clean").mkdir(parents=True)

    proto = "Protocol guideline: Give Tab Metformin 500 mg BD for Type 2 diabetes."
    (inp / "protocols" / "diabetes_guide.txt").write_text(proto, encoding="utf-8")
    (inp / "clean" / "sneaky_patients.csv").write_text(
        "patient_name,uhid,age,district,visit_date\n"
        "Kavita Tiwari,SNJ-27-101010,34,Lucknow,2026-04-10\n",
        encoding="utf-8",
    )
    (inp / "clean" / "sneaky_note.txt").write_text(
        "Patient: Kavita Tiwari (UHID: SNJ-27-101010), Age: 34 Y/F.\n"
        "District: Lucknow. Date: 2026-04-10.\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("RAKSHA_SECRET", "content-rule-secret-xyz")
    assert run_pipeline(inp, out) == 0

    # Non-patient protocol outside clean/ stayed untouched (no PSN, no date shift, identical text)
    assert (out / "sanitized" / "protocols" / "diabetes_guide.txt").read_text(encoding="utf-8") == proto
    # Patient files inside clean/ WERE redacted, linked with same PSN, and generalized
    csv_out = (out / "sanitized" / "clean" / "sneaky_patients.csv").read_text(encoding="utf-8")
    note_out = (out / "sanitized" / "clean" / "sneaky_note.txt").read_text(encoding="utf-8")
    assert "Kavita Tiwari" not in csv_out and "SNJ-27-101010" not in csv_out
    assert "34," not in csv_out and ("20-39" in csv_out or "30-34" in csv_out)
    assert "Kavita Tiwari" not in note_out and "SNJ-27-101010" not in note_out and "2026-04-10" not in note_out
    recs = [json.loads(line) for line in (out / "records.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    by_f = {r["file"]: r["patient"] for r in recs}
    assert by_f["sanitized/protocols/diabetes_guide.txt"] is None
    assert by_f["sanitized/clean/sneaky_patients.csv"] is not None
    assert by_f["sanitized/clean/sneaky_patients.csv"] == by_f["sanitized/clean/sneaky_note.txt"]

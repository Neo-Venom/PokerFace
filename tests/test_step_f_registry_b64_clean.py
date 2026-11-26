"""Unit tests for Step F: alias registry, filename sanitization, base64 attachments, and clean/ preservation."""
import base64

from raksha.detect.identifiers import redact_spans
from raksha.detect.protected import ClinicalLexicon, resolve_pii_vs_protected
from raksha.detect.registry import build_alias_registry
from raksha.export.filenames import sanitize_base64_string, sanitize_relative_path


def test_step_f_alias_registry_cross_script_and_doctor_protection():
    parsed = [
        (
            "registration/patients.csv",
            "csv",
            [
                ["UHID", "Patient Name", "Relative Name", "Mobile", "Doctor"],
                ["HOSP-10001", "Pradeep Kulkarni", "Devendra", "9876543210", "Dr. Pradeep Menon"],
            ],
        )
    ]
    reg = build_alias_registry(parsed)
    txt = "प्रदीप कुलकर्णी visited. Dr. Pradeep Menon prescribed Paracetamol 650 mg."
    lex = ClinicalLexicon()
    prot = lex.find_spans(txt)
    cand = reg.propose_spans(txt)
    out = redact_spans(txt, resolve_pii_vs_protected(cand, prot))
    assert "प्रदीप कुलकर्णी" not in out
    assert "Dr. Pradeep Menon" in out
    assert "Paracetamol 650 mg" in out


def test_step_f_filename_and_base64_sanitization():
    reg = build_alias_registry([("reg.csv", "csv", [["UHID", "Patient Name"], ["KGM-0088123", "Nandini Deshmukh"]])])
    out_rel = sanitize_relative_path(
        "discharge/Discharge_Summary_Nandini_Deshmukh_KGM-0088123.txt",
        input_sha256="abcdef123456",
        registry=reg,
    )
    assert "Nandini" not in out_rel
    assert "Deshmukh" not in out_rel
    assert "KGM-0088123" not in out_rel
    assert out_rel.endswith(".txt")

    raw_b64 = base64.b64encode(b"Consent witnessed by Pema Dorjee, phone 8010993510.").decode("ascii")
    cleaned_b64 = sanitize_base64_string(
        raw_b64,
        lambda s: s.replace("Pema Dorjee", "[RELATIVE_NAME]").replace("8010993510", "[PHONE]"),
    )
    decoded = base64.b64decode(cleaned_b64).decode("utf-8")
    assert "Pema Dorjee" not in decoded
    assert "8010993510" not in decoded

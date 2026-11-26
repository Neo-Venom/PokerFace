"""Unit tests for Step F: alias registry, filename sanitization, base64 attachments, and clean/ preservation."""
import base64
from pathlib import Path

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
                ["UHID-10001", "Mohammad Arif", "Zubair", "9876543210", "Dr. Arif Khan"],
            ],
        )
    ]
    reg = build_alias_registry(parsed)
    txt = "मोहम्मद आरिफ visited. Dr. Arif Khan prescribed Paracetamol 650 mg."
    lex = ClinicalLexicon()
    prot = lex.find_spans(txt)
    cand = reg.propose_spans(txt)
    out = redact_spans(txt, resolve_pii_vs_protected(cand, prot))
    assert "मोहम्मद आरिफ" not in out
    assert "Dr. Arif Khan" in out
    assert "Paracetamol 650 mg" in out


def test_step_f_filename_and_base64_sanitization():
    reg = build_alias_registry([("reg.csv", "csv", [["UHID", "Patient Name"], ["JJH-0076928", "Reena Nishad"]])])
    out_rel = sanitize_relative_path(
        "discharge/Discharge_Summary_Reena_Nishad_JJH-0076928.txt",
        input_sha256="abcdef123456",
        registry=reg,
    )
    assert "Reena" not in out_rel
    assert "Nishad" not in out_rel
    assert "JJH-0076928" not in out_rel
    assert out_rel.endswith(".txt")

    raw_b64 = base64.b64encode(b"Consent witnessed by Tsering Wangmo, phone 8010993510.").decode("ascii")
    cleaned_b64 = sanitize_base64_string(raw_b64, lambda s: s.replace("Tsering Wangmo", "[RELATIVE_NAME]").replace("8010993510", "[PHONE]"))
    decoded = base64.b64decode(cleaned_b64).decode("utf-8")
    assert "Tsering Wangmo" not in decoded
    assert "8010993510" not in decoded

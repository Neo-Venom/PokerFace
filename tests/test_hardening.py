"""Regression tests for hostile-export hardening (synthetic fixtures only).

Each test pins a failure found with adversarial stress data: NUL bytes inside valid text, JSON NaN,
unsafe file names, spreadsheet formulas, integer-overflow ages, "C/O <complaint>" columns, audit-token
canaries, mixed-script names, PINs with no cue, and relatives that only conflict as single given names.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import Policy  # noqa: E402
from src.ingest.csv_handler import neutralize_formula  # noqa: E402
from src.ingest.json_handler import parse_json_entry  # noqa: E402
from src.ingest.profiler import _rel_value, name_like_value  # noqa: E402
from src.ingest.router import decode_strict, maybe_rename_rel_path  # noqa: E402
from src.linkage.scorer import rel_conflict, rel_types_comparable  # noqa: E402
from src.models import FileEntry  # noqa: E402
from src.nlp.medical_guard import MedicalGuard  # noqa: E402
from src.nlp.pii_ner import resolve_and_scrub  # noqa: E402
from src.nlp.pii_regex import extract_regex_spans  # noqa: E402
from src.privacy.kanon import generalize_age_value  # noqa: E402
from src.security.keys import KeyRing  # noqa: E402


class TestIngest(unittest.TestCase):
    def test_sparse_nul_bytes_are_spaces_not_quarantine(self) -> None:
        text, err = decode_strict(b"name,ref\nRavi,Ref: Kiran\x009876543210\n" + b"x" * 3000)
        self.assertIsNone(err)
        self.assertIn("Kiran 9876543210", text)

    def test_nul_only_file_is_still_quarantined(self) -> None:
        self.assertEqual(decode_strict(b"\x00" * 4096)[1], "corrupt_binary_bytes")

    def test_json_nan_is_quarantined(self) -> None:
        e = FileEntry(input_sha256="0" * 64, rel_input_path="v.json", rel_output_path="v.json", status="sanitized",
                      file_type="json", text_decoded='{"visits": [{"age": NaN, "los": Infinity}]}')
        parse_json_entry(e, Policy())
        self.assertEqual(e.status, "quarantined")

    def test_unsafe_file_names_are_quarantined(self) -> None:
        import tempfile
        from src.ingest.router import scan_input_tree
        with tempfile.TemporaryDirectory() as td:
            for name in ("note;rm -rf tmp;.csv", "$(whoami)_opd.json", ".._.._etc_passwd.txt", "ok_notes.txt"):
                (Path(td) / name).write_text("Patient Ravi Kumar 9876543210\n", encoding="utf-8")
            got = {rel: e.status for _p, rel, e in scan_input_tree(Path(td))}
        self.assertEqual(got.pop("ok_notes.txt"), "sanitized")
        self.assertEqual(set(got.values()), {"quarantined"})

    def test_unsafe_file_names_are_never_re_emitted(self) -> None:
        kr = KeyRing(b"test-secret")
        for name in ("note;rm -rf tmp;.csv", "$(whoami)_opd.json", ".._.._etc_passwd_x.txt", "`id`_lab.csv"):
            new, renamed = maybe_rename_rel_path(f"misc/{name}", "a" * 64, kr, lambda s: False)
            self.assertTrue(renamed, name)
            self.assertNotRegex(new, r"[;`$]|\.\._", name)
        self.assertEqual(maybe_rename_rel_path("misc/report.v2.txt", "a" * 64, kr, lambda s: False), ("misc/report.v2.txt", False))


class TestOutputHygiene(unittest.TestCase):
    def test_formula_cells_are_neutralised_numbers_are_not(self) -> None:
        for cell in ("=CMD|' /C calc'!A0", "+SUM(1+1)", "-2+3+cmd", "@HYPERLINK(\"x\")", "\t=1+1"):
            self.assertTrue(neutralize_formula(cell).startswith("'"), cell)
        for cell in ("-2.5", "+91", "-", "normal", "1.5e3"):
            self.assertEqual(neutralize_formula(cell), cell)

    def test_implausible_ages_fail_closed(self) -> None:
        for raw in ("-1", "0x7fffffff", "9223372036854775807", "-9223372036854775809", "130"):
            self.assertEqual(generalize_age_value(raw, "65+"), "[AGE]", raw)
        self.assertEqual(generalize_age_value("80", "65+"), "65+")
        self.assertEqual(generalize_age_value("8 months", "0-4"), "0-4")
        self.assertEqual(generalize_age_value("NaN", None), "NaN")


class TestMeaning(unittest.TestCase):
    def test_care_of_complaint_is_not_a_relative(self) -> None:
        self.assertFalse(_rel_value("C/O fever and weakness; Bell's palsy; HbA1c 6.5%"))
        self.assertFalse(_rel_value("C/O Fever"))
        self.assertTrue(_rel_value("C/O Prakash (PAN: ABCPK1234Z)"))
        self.assertTrue(_rel_value("D/o Late Ram Prasad"))

    def test_hinglish_name_cells_are_names(self) -> None:
        for v in ("Pooja didi", "Asha ji", "I. Devi"):
            self.assertTrue(name_like_value(v, None), v)
        self.assertFalse(name_like_value("ji", None))


class TestDetection(unittest.TestCase):
    def test_audit_token_and_long_segment_ids(self) -> None:
        for tok in ("CANARY-TXT-013-LEAK", "CANARY-BIGFIELD-9941"):
            self.assertIn(("MRN", tok), [(s.kind, s.text) for s in extract_regex_spans(f"Token: {tok} end")], tok)
        clinical = "SARS-CoV-2 RT-PCR negative, ICD-10 E11.9, CA-125 high, NS-500-ML-IV, ICU-BED-012"
        self.assertEqual([s for s in extract_regex_spans(clinical) if s.kind == "MRN"], [])

    def test_mixed_script_name_and_same_surname_doctor(self) -> None:
        out, _ = resolve_and_scrub("Pt लखन Lepcha (24M) seen by Dr. Kavita Lepcha today.", MedicalGuard())
        self.assertNotIn("लखन", out)
        self.assertIn("Dr. Kavita Lepcha", out)

    def test_pin_right_after_removed_village(self) -> None:
        from src.nlp.pii_ner import _pin_after_address
        self.assertEqual(_pin_after_address("[NAME] (W/o [RELATIVE]), [ADDRESS] 247451</div>"),
                         "[NAME] (W/o [RELATIVE]), [ADDRESS] [PIN]</div>")
        self.assertEqual(_pin_after_address("Platelets 245000 /uL"), "Platelets 245000 /uL")


class TestLinkage(unittest.TestCase):
    def test_single_given_name_relatives_conflict(self) -> None:
        self.assertTrue(rel_conflict([("salim",)], [("dinesh",)]))
        self.assertFalse(rel_conflict([("ramesh",)], [("ramesh", "kumar")]))
        self.assertFalse(rel_conflict([("r", "k", "sharma")], [("ramesh",)]))   # initials: undecidable

    def test_care_of_is_compared_with_guardian_types_only(self) -> None:
        self.assertTrue(rel_types_comparable("C/O", "W/o"))
        self.assertFalse(rel_types_comparable("S/o", "W/o"))
        self.assertFalse(rel_types_comparable(None, "W/o"))


if __name__ == "__main__":
    unittest.main()

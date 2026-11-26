"""Regression tests for unseen-format generalisation (synthetic fixtures only, no practice/dev data).

Each test pins a failure that was found with the adversarial "trap gauntlet" and fixed:
unknown column headers, unseen ID / phone / date shapes, hospital names taken for patients,
Hinglish verbs taken for names, encoded payloads, and prefix false positives ("village", "ग्रामीण").
"""
from __future__ import annotations

import csv
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import Policy  # noqa: E402
from src.linkage.evidence import RecordEvidence, _extract_from_text_blob  # noqa: E402
from src.nlp.date_shifter import shift_dates_in_text  # noqa: E402
from src.nlp.medical_guard import MedicalGuard  # noqa: E402
from src.nlp.pii_ner import PiiHarvester, resolve_and_scrub  # noqa: E402
from src.nlp.pii_regex import extract_regex_spans  # noqa: E402
from src.pipeline import run_pipeline  # noqa: E402


def kinds(text: str) -> list[tuple[str, str]]:
    return [(s.kind, s.text) for s in extract_regex_spans(text)]


class TestShapes(unittest.TestCase):
    def test_unseen_hospital_id_shapes(self) -> None:
        for tok in ("BCT22376516", "VZC.25.934542", "SSH-R-25-055336", "KR-6361141", "KRM/IP/2025/7743", "ADM-25-3397"):
            self.assertIn(("MRN", tok), kinds(f"ref {tok} seen"), tok)

    def test_clinical_codes_are_not_ids(self) -> None:
        for tok in ("E11.9", "A90", "HbA1c 7.2", "PKG-465", "COVID-19", "SpO2 96"):
            self.assertFalse([k for k in kinds(tok) if k[0] == "MRN"], tok)

    def test_phone_groupings(self) -> None:
        for ph in ("(751) 898-8707", "063951-20142", "+91-700-063-5448", "8188-391777", "98450.12345", "0091 9845012345"):
            self.assertTrue(any(k == "PHONE" for k, _ in kinds(f"call {ph}.")), ph)

    def test_plus91_phone_is_not_aadhaar(self) -> None:
        self.assertFalse(any(k == "AADHAAR" for k, _ in kinds("call back on +918989715707.")))

    def test_word_prefix_false_positives(self) -> None:
        self.assertEqual(kinds("village"), [])
        self.assertEqual(kinds("कोंकण ग्रामीण मिशन अस्पताल"), [])
        self.assertIn(("VILLAGE", "मोतीपुर"), kinds("ग्राम मोतीपुर, तहसील वाई"))

    def test_pin_after_nukta_zila_and_line_end(self) -> None:
        self.assertIn(("PIN", "333654"), kinds("ज़िला झुंझुनू - 333654"))
        self.assertIn(("PIN", "415673"), kinds("Vill. Bakhrikhera, Dist. Ratnagiri - 415673"))
        self.assertFalse(any(k == "PIN" for k, _ in kinds("Platelet count 362332 /cumm")))


class TestContext(unittest.TestCase):
    def setUp(self) -> None:
        self.g = MedicalGuard()
        self.g._add_doctor("Dr. Ramesh Iyer")
        self.h = PiiHarvester(self.g)

    def scrub(self, t: str) -> str:
        return resolve_and_scrub(t, self.g, self.h)[0]

    def test_unlabelled_names(self) -> None:
        out = self.scrub("Pt Thiruvenkadam (62M) brought by son Arivazhagan. Bed 7: Kezhokhoto Sema (aged 44).")
        for leak in ("Thiruvenkadam", "Arivazhagan", "Kezhokhoto"):
            self.assertNotIn(leak, out)

    def test_hinglish_relation_and_verbs(self) -> None:
        out = self.scrub("Lalrinpuii didi ka sugar 210 hai, unke pati Zothanmawia ne bataya ki dawai nahi li.")
        self.assertNotIn("Zothanmawia", out)
        self.assertNotIn("Lalrinpuii", out)
        self.assertIn("bataya", out)
        self.assertIn("sugar 210", out)

    def test_consent_signed_by_patient_is_redacted(self) -> None:
        self.assertNotIn("Vimala", self.scrub("Consent for procedure signed by Vimala Sandhu, D/o Ramesh Sandhu."))

    def test_doctor_kept_relative_removed(self) -> None:
        out = self.scrub("Seen by Dr. Ramesh Iyer. Attendant Ramesh Kale counselled. Reviewed by Ramesh Iyer.")
        self.assertIn("Dr. Ramesh Iyer", out)
        self.assertIn("Reviewed by Ramesh Iyer", out)
        self.assertNotIn("Kale", out)

    def test_clinical_text_untouched(self) -> None:
        t = "Review in Medicine Unit 2 OPD. Ward 7 (Surgery). K/C/O DM, HTN. H/O Koch's. Dolo 650 BD x 5 days. 1-0-1"
        self.assertEqual(self.scrub(t), t)

    def test_encoded_payloads_decoded(self) -> None:
        ent = "".join(f"&#{ord(c)};" for c in "पांडुरंग उरांव")
        h = PiiHarvester(self.g)
        h.add_name("Pandurang Oraon")
        h.add_name("पांडुरंग उरांव")
        h.compile()
        out = resolve_and_scrub(f"GET /search?name=Pandurang%20Oraon <td>{ent}</td>", self.g, h)[0]
        self.assertNotIn("Oraon", out)
        self.assertNotIn("&#2346;", out)

    def test_hospital_name_is_not_a_patient(self) -> None:
        self.g.hospital_names.add("brahmaputra care trust hospital")
        ev = RecordEvidence("x", None, None)
        _extract_from_text_blob(ev, "Parchi - Brahmaputra Care Trust Hospital\nJitendra ji (65Y/M) ko giddiness hai.\nRef no BCT84227930", self.g, True)
        self.assertNotIn(("brahmaputra", "care", "trust"), ev.full_names)
        ev2 = RecordEvidence("y", None, None)
        _extract_from_text_blob(ev2, "Patient: ORAON, Ishani    Age/Sex: 75Y/F", self.g, True)
        self.assertNotIn(("ishani", "age"), ev2.full_names)


class TestDates(unittest.TestCase):
    def test_named_month_formats_shift_and_keep_los(self) -> None:
        out = shift_dates_in_text("Admitted: 30-Oct-2025   Discharged: 09-Nov-2025", delta_days=30, dmy=True, guard=MedicalGuard())
        self.assertEqual(out, "Admitted: 29-Nov-2025   Discharged: 09-Dec-2025")
        out = shift_dates_in_text("admitted on June 8, 2025", delta_days=30, dmy=True)
        self.assertEqual(out, "admitted on July 8, 2025")

    def test_two_digit_year_and_doses(self) -> None:
        out = shift_dates_in_text("Date: 14/03/25. Rx 1-0-1 x 5 days. 0-0-1", delta_days=30, dmy=True, guard=MedicalGuard())
        self.assertEqual(out, "Date: 13/04/25. Rx 1-0-1 x 5 days. 0-0-1")


class TestEndToEnd(unittest.TestCase):
    def test_opaque_headers_and_amounts(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            inp, out = Path(td) / "in", Path(td) / "out"
            (inp / "reg").mkdir(parents=True)
            rows = [["holder", "reach_at", "abode", "gov_no", "born_on", "amount_inr", "dx"],
                    ["Thiruvenkadam Subramaniam", "+91 98450 12345", "14 Kovil St, Kuttanad 688501", "4567 8912 3456", "1962-07-14", "145000", "Essential hypertension"],
                    ["Lalrinpuii Hmar", "09436 155 872", "Bawngkawn, Aizawl 796012", "7812 3344 5566", "1990-02-03", "9000", "Type 2 diabetes mellitus"],
                    ["Ngawang Dorjee Bhutia", "7001-998877", "Rumtek Gompa Rd, Gangtok 737135", "2233 4455 6677", "1975-11-30", "18500", "Pulmonary tuberculosis"]]
            buf = io.StringIO()
            csv.writer(buf, lineterminator="\n").writerows(rows)
            (inp / "reg" / "intake.csv").write_text(buf.getvalue(), encoding="utf-8")
            (inp / "reg" / "visits.json").write_text(json.dumps({"visits": [
                {"visitor": "Ngawang Dorjee Bhutia", "contact": "7001998877", "village": "Rumtek", "complaint": "cough"}]}), encoding="utf-8")
            os.environ["RAKSHA_SECRET"] = "unit-test-secret"
            self.assertEqual(run_pipeline(inp, out, Policy()), 0)
            txt = (out / "sanitized" / "reg" / "intake.csv").read_text(encoding="utf-8")
            vis = json.loads((out / "sanitized" / "reg" / "visits.json").read_text(encoding="utf-8"))
            for leak in ("Thiruvenkadam", "Lalrinpuii", "Ngawang", "Dorjee", "98450", "7001", "4567", "Kovil", "Rumtek", "688501"):
                self.assertNotIn(leak, txt, leak)
            for amt in ("145000", "9000", "18500"):
                self.assertIn(amt, txt, "claim amounts are not PIN codes")
            self.assertIn("Essential hypertension", txt)
            self.assertIn("village", vis["visits"][0], "JSON keys must survive")
            recs = [json.loads(l) for l in (out / "records.jsonl").read_text(encoding="utf-8").splitlines()]
            by_loc = {(r["file"].rsplit("/", 1)[-1], r["loc"]): r["patient"] for r in recs}
            self.assertEqual(by_loc[("intake.csv", "row:2")], by_loc[("visits.json", "/visits/0")], "same person across files")
            self.assertNotEqual(by_loc[("intake.csv", "row:0")], by_loc[("intake.csv", "row:1")])


if __name__ == "__main__":
    unittest.main()

"""Synthetic unit & adversarial regression suite for Project RAKSHA (no dev/truth fixtures)."""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from src.config import Policy
from src.linkage.scorer import _tok_sim
from src.nlp.date_shifter import shift_dates_in_text
from src.nlp.indic_norm import name_tokens, phonetic_skeleton
from src.nlp.medical_guard import MedicalGuard
from src.nlp.pii_ner import PiiHarvester, resolve_and_scrub
from src.pipeline import run_pipeline
from src.privacy.kanon import format_age_band
from src.security.keys import KeyRing
from src.security.vault import EncryptedVault


class TestRakshaCore(unittest.TestCase):
    def test_hkdf_determinism_and_rotation(self) -> None:
        k1 = KeyRing(b"secret-alpha")
        k2 = KeyRing(b"secret-alpha")
        k3 = KeyRing(b"secret-beta")
        self.assertEqual(k1.pseudonym("cluster-1"), k2.pseudonym("cluster-1"))
        self.assertNotEqual(k1.pseudonym("cluster-1"), k3.pseudonym("cluster-1"))
        self.assertEqual(k1.date_shift_days("cluster-1"), k2.date_shift_days("cluster-1"))
        self.assertNotEqual(k1.date_shift_days("cluster-1"), k3.date_shift_days("cluster-1"))
        k1.clear()
        self.assertEqual(k1.k_psn, b"")

    def test_cross_script_name_canonicalization(self) -> None:
        self.assertEqual(name_tokens("Mohammad Shahid"), ["mohamad", "shahid"])
        self.assertEqual(name_tokens("Md. Shahid"), ["mohamad", "shahid"])
        self.assertEqual(name_tokens("मोहम्मद शाहिद"), ["mohamad", "shahid"])
        deva_toks = name_tokens("रुख़साना सिद्दीक़ी")
        lat_toks = name_tokens("Rukhsana Siddiqui")
        self.assertEqual(deva_toks[1], lat_toks[1])
        self.assertEqual(phonetic_skeleton(deva_toks[0]), phonetic_skeleton(lat_toks[0]))
        self.assertGreaterEqual(_tok_sim(deva_toks[0], lat_toks[0]), 0.88)

    def test_meaning_traps_and_clinical_shield(self) -> None:
        guard = MedicalGuard()
        harvester = PiiHarvester(guard)
        harvester.add_name("Asha Rajput")
        harvester.compile()
        text = (
            "Patient Name: Asha Rajput  Age/Sex: 48 Yrs/F\n"
            "Referred by ASHA worker. Consulted Dr. Ramesh Srivastava.\n"
            "K/C/O Addison's disease, Parkinson's disease.\n"
            "Ward 7 (Medicine) Bed 12. Platelet Count 137130 /uL.\n"
            "Aadhaar: 397205839628  Tab Paracetamol 650 mg 1-0-1\n"
        )
        cleaned, _ = resolve_and_scrub(text, guard, harvester)
        self.assertNotIn("Asha Rajput", cleaned)
        self.assertNotIn("397205839628", cleaned)
        self.assertIn("ASHA worker", cleaned)
        self.assertIn("Dr. Ramesh Srivastava", cleaned)
        self.assertIn("Addison's disease", cleaned)
        self.assertIn("Parkinson's disease", cleaned)
        self.assertIn("Ward 7 (Medicine)", cleaned)
        self.assertIn("137130", cleaned)
        self.assertIn("1-0-1", cleaned)

    def test_date_shift_preserves_los_and_yearless(self) -> None:
        txt = "Admitted: 10-02-2026\nDischarged: 20-02-2026\nVisit: 5 September"
        shifted = shift_dates_in_text(txt, delta_days=40, dmy=True, anchor_year=2026)
        self.assertIn("22-03-2026", shifted)
        self.assertIn("01-04-2026", shifted)
        self.assertIn("15 October", shifted)
        self.assertEqual(len(txt.splitlines()), len(shifted.splitlines()))

    def test_truthful_age_band(self) -> None:
        self.assertEqual(format_age_band(48, 48, width=5), "45-49")
        self.assertEqual(format_age_band(49, 50, width=5), "45-54")
        self.assertEqual(format_age_band(39, 40, width=20), "30-49")
        self.assertEqual(format_age_band(59, 60, width=20, top_code=80), "50-69")
        self.assertEqual(format_age_band(89, 91, width=5, top_code=85), "85+")

    def test_fhir_uuid_performed_whatsapp_and_vault(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            inp = root / "in"
            out = root / "out"
            inp.mkdir()
            (inp / "opd.csv").write_text(
                "UHID,Patient Name,Age,Gender,Mobile\n"
                "SNJ/26/0009,Kavita Verma,48,F,9811223344\n",
                encoding="utf-8",
            )
            (inp / "whatsapp_forward_0001.txt").write_text(
                "Forwarded message\nKavita Verma 48 Y / F\nUHID: SNJ/26/0009\nDx: Viral fever\n",
                encoding="utf-8",
            )
            bundle = {
                "resourceType": "Bundle",
                "type": "collection",
                "entry": [
                    {
                        "fullUrl": "urn:uuid:93368d52-86de-4263-8902-419343ba0fb6",
                        "resource": {
                            "resourceType": "Patient",
                            "id": "93368d52-86de-4263-8902-419343ba0fb6",
                            "identifier": [{"system": "http://hospital.in/uhid", "value": "SNJ/26/0009"}],
                            "name": [{"text": "Kavita Verma"}],
                        },
                    },
                    {
                        "fullUrl": "urn:uuid:b6ae9c60-4286-4730-8390-bd257f0acd4c",
                        "resource": {
                            "resourceType": "Procedure",
                            "id": "b6ae9c60-4286-4730-8390-bd257f0acd4c",
                            "status": "completed",
                            "code": {"text": "IV Fluids"},
                            "subject": {"reference": "urn:uuid:93368d52-86de-4263-8902-419343ba0fb6"},
                            "performedDateTime": "2026-03-14T09:00:00+05:30",
                        },
                    },
                ],
            }
            (inp / "bundle.json").write_text(json.dumps(bundle), encoding="utf-8")
            (inp / "patient_index.json").write_text(
                json.dumps({"by_uhid": {"SNJ/26/0009": {"visits": 2}}, "by_name": {"Kavita Verma": "SNJ/26/0009"}}),
                encoding="utf-8",
            )
            pdf_bytes = b"%PDF-1.4 quarantined"
            (inp / "scan.pdf").write_bytes(pdf_bytes)
            import hashlib

            pdf_sha = hashlib.sha256(pdf_bytes).hexdigest()
            os.environ["RAKSHA_SECRET"] = "vault-audit-secret"
            rc = run_pipeline(inp, out, Policy(encrypt_quarantine_archive=True))
            self.assertEqual(rc, 0)

            out_bundle = json.loads((out / "sanitized" / "bundle.json").read_text(encoding="utf-8"))
            self.assertEqual(out_bundle["entry"][0]["resource"]["id"], "93368d52-86de-4263-8902-419343ba0fb6")
            self.assertEqual(out_bundle["entry"][1]["resource"]["id"], "b6ae9c60-4286-4730-8390-bd257f0acd4c")
            self.assertNotIn("2026-03-14", out_bundle["entry"][1]["resource"]["performedDateTime"])

            wa = (out / "sanitized" / "whatsapp_forward_0001.txt").read_text(encoding="utf-8")
            self.assertNotIn(" 48 ", wa)
            self.assertIn("40-59 Y / F", wa)

            idx_out = json.loads((out / "sanitized" / "patient_index.json").read_text(encoding="utf-8"))
            uhid_keys = list(idx_out["by_uhid"].keys())
            self.assertTrue(len(uhid_keys) == 1 and uhid_keys[0].startswith("PSN-"))
            self.assertEqual(idx_out["by_name"].get(uhid_keys[0]), uhid_keys[0])

            enc_blob = (out / "quarantine" / "scan.pdf.enc").read_bytes()
            kr = KeyRing(b"vault-audit-secret")
            self.assertEqual(EncryptedVault(kr.k_vault).unseal_bytes(enc_blob, pdf_sha), pdf_bytes)

    def test_end_to_end_quarantine_and_linkage(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            inp = root / "in"
            out = root / "out"
            inp.mkdir()
            (inp / "patients.csv").write_text(
                "UHID,Patient Name,Age,District,Mobile\nSNJ/26/0001,Meenakshi Iyer,42,Unnao,9876543210\n",
                encoding="utf-8",
            )
            (inp / "note_Meenakshi_Iyer_SNJ_26_0001.txt").write_text(
                "Discharge note\nPatient Name: Meenakshi Iyer  Age: 42\nUHID: SNJ/26/0001\nDx: Typhoid fever\n",
                encoding="utf-8",
            )
            (inp / "scan.pdf").write_bytes(b"%PDF-1.4 binary payload")
            (inp / "xxe.xml").write_text(
                '<?xml version="1.0"?><!DOCTYPE bomb [<!ENTITY a "lol"><!ENTITY b "&a;&a;">]><r>&b;</r>',
                encoding="utf-8",
            )
            os.environ["RAKSHA_SECRET"] = "unit-test-secret-key"
            rc = run_pipeline(inp, out, Policy())
            self.assertEqual(rc, 0)
            self.assertFalse((out / "sanitized" / "scan.pdf").exists())
            self.assertFalse((out / "sanitized" / "xxe.xml").exists())
            recs = (out / "records.jsonl").read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(recs), 2)
            import json

            r0, r1 = json.loads(recs[0]), json.loads(recs[1])
            self.assertIsNotNone(r0["patient"])
            self.assertEqual(r0["patient"], r1["patient"])
            out_txt = "\n".join(p.read_text(encoding="utf-8") for p in (out / "sanitized").rglob("*") if p.is_file())
            self.assertNotIn("Meenakshi", out_txt)
            self.assertNotIn("9876543210", out_txt)
            self.assertNotIn("unit-test-secret-key", out_txt)
            self.assertIn("Typhoid fever", out_txt)


if __name__ == "__main__":
    unittest.main()

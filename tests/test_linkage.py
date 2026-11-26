"""Linkage unit tests (synthetic fixtures only): name comparison, vetoes, family phones, weak-record attachment."""
from __future__ import annotations

import datetime as dt
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import Policy  # noqa: E402
from src.linkage import scorer as S  # noqa: E402
from src.linkage.cluster import is_weak_identity, link_and_assign_pseudonyms  # noqa: E402
from src.linkage.evidence import (  # noqa: E402
    RecordEvidence, _add_id_evidence, _add_name_evidence, _detect_gender_text, _fhir_human_name,
    _reorder_surname_first, is_episode_id,
)
from src.security.keys import KeyRing  # noqa: E402

_N = 0


def ev(name: str | None = None, rel: str | None = None, rel_type: str | None = None, gender: str | None = None,
       phones: tuple[str, ...] = (), mrn: str | None = None, dob: dt.date | None = None,
       byear: int | None = None) -> RecordEvidence:
    global _N
    _N += 1
    rec = SimpleNamespace(loc=f"row:{_N}", cluster_id=None, pseudonym=None)
    e = RecordEvidence(f"u{_N:05d}", SimpleNamespace(input_sha256=f"{_N:064x}"), rec)
    if name:
        _add_name_evidence(e, name)
    if rel:
        _add_name_evidence(e, rel, is_relative=True)
        e.rel_type = rel_type
    e.gender = gender
    e.phones.update(phones)
    if mrn:
        _add_id_evidence(e, "MRN", mrn)
    if dob:
        e.dobs.add(dob)
        e.birth_years.add(dob.year)
    if byear:
        e.birth_years.add(byear)
    return e


def link(evs: list[RecordEvidence]) -> list[str | None]:
    link_and_assign_pseudonyms(evs, KeyRing(b"unit-test-secret"), Policy())
    return [e.record.pseudonym for e in evs]


class TestEvidence(unittest.TestCase):
    def test_episode_ids(self) -> None:
        for s in ("KRM/IP/2025/7743", "ADM-25-3397", "VZC-IP-10462", "OPD/26/001", "IPD-2026-00432"):
            self.assertTrue(is_episode_id(s), s)
        for s in ("KR-7261757", "UHID-010678", "SNJ/26/064098"):
            self.assertFalse(is_episode_id(s), s)

    def test_text_gender_ignores_honorifics(self) -> None:
        self.assertIsNone(_detect_gender_text("Patient Sunita, attendant Ramesh bhaiya. Sahab ne bola f/u in 5 m."))
        self.assertEqual(_detect_gender_text("Age/Sex: 45 Y / F"), "F")
        self.assertEqual(_detect_gender_text("a 60-year-old man with fever"), "M")
        self.assertIsNone(_detect_gender_text("Sex: M ... 45 Y / F"))
        # "पुत्र" inside a hospital name is not "son of"
        self.assertIsNone(_detect_gender_text("ब्रह्मपुत्र केयर ट्रस्ट अस्पताल — पर्चा\nरोगी का नाम: सरोज सेमा"))
        self.assertEqual(_detect_gender_text("रोगी: सरोज, पुत्री रमेश"), "F")

    def test_fhir_name_given_family(self) -> None:
        self.assertEqual(_fhir_human_name([{"given": ["Asha", "K"], "family": "Rajput"}]), "Asha K Rajput")
        self.assertEqual(_fhir_human_name({"text": "Asha Rajput", "family": "X"}), "Asha Rajput")
        self.assertEqual(_fhir_human_name({"use": "official"}), "")

    def test_surname_first(self) -> None:
        self.assertEqual(_reorder_surname_first("Iyer, Rajesh"), "Rajesh Iyer")
        self.assertEqual(_reorder_surname_first("Rajesh, S/o Mohan"), "Rajesh, S/o Mohan")

    def test_weak_identity(self) -> None:
        self.assertTrue(is_weak_identity(ev("Meera ji", "Mahesh", "D/o")))
        self.assertFalse(is_weak_identity(ev("Meera Joshi")))
        self.assertFalse(is_weak_identity(ev("Meera", phones=("9876543210",))))


class TestScorer(unittest.TestCase):
    def test_one_consonant_apart_is_not_a_match(self) -> None:
        self.assertLess(S._tok_sim("rajesh", "ramesh"), S.MATCH)

    def test_swapped_dob(self) -> None:
        self.assertTrue(S.dob_compatible(dt.date(1980, 5, 7), dt.date(1980, 7, 5)))
        self.assertFalse(S.dob_compatible(dt.date(1980, 5, 7), dt.date(1981, 7, 5)))
        a, b = ev("Kiran Bhatt", dob=dt.date(1980, 5, 7)), ev("Kiran Bhatt", dob=dt.date(1980, 7, 5))
        self.assertFalse(S.has_hard_veto(a, b))

    def test_episode_numbers_never_conflict(self) -> None:
        a, b = ev("Kiran Bhatt", mrn="KRM/IP/2025/7743"), ev("Kiran Bhatt", mrn="KRM/IP/2025/9135")
        self.assertFalse(S.conflicting_system_ids(a, b))
        c, d = ev("Kiran Bhatt", mrn="KRM-0072617"), ev("Kiran Bhatt", mrn="KRM-0088412")
        self.assertTrue(S.conflicting_system_ids(c, d))

    def test_family_phone_needs_distinct_people(self) -> None:
        ph = ("9876543210",)
        # one patient, two spellings under one registration number: not a family phone
        same = [ev("Grace Kulkarni", phones=ph, mrn="KRM-0072617"), ev("Gracy Kulkarni", phones=ph, mrn="KRM-0072617")]
        self.assertEqual(S.detect_shared_family_phones(same), set())
        # spouses sharing a phone
        fam = [ev("Sunil Verma", gender="M", phones=ph), ev("Anita Verma", gender="F", phones=ph)]
        self.assertEqual(S.detect_shared_family_phones(fam), set(ph))


class TestClusters(unittest.TestCase):
    def test_homonyms_with_different_dob_stay_apart(self) -> None:
        evs = [ev("Ramesh Kumar", dob=dt.date(1970, 1, 2), mrn="AB-1234567"),
               ev("Ramesh Kumar", dob=dt.date(1995, 3, 4), mrn="AB-7654321")]
        p = link(evs)
        self.assertNotEqual(p[0], p[1])

    def test_rare_fragment_attaches_common_one_does_not(self) -> None:
        core = [ev("Asha Rajput", "Kamal Rajput", "W/o", "F", ("9811111111",), "AB-1000001"),
                ev("Asha Rajput", "Kamal Rajput", "W/o", "F", ("9811111111",), "AB-1000001")]
        others = [ev(f"{g} Sharma", f"{r} Sharma", "W/o", "F", (f"98222{i:05d}",), f"AB-20{i:05d}")
                  for i, (g, r) in enumerate([("Pooja", "Vikas"), ("Neha", "Rohit"), ("Kavya", "Arun"),
                                              ("Ritu", "Manoj"), ("Seema", "Ajay"), ("Lata", "Gopal")])]
        frag = ev("Asha", "Kamal", "W/o", "F")
        p = link(core + others + [frag])
        self.assertIsNotNone(p[0])
        self.assertEqual(p[-1], p[0])

    def test_ambiguous_fragment_abstains(self) -> None:
        evs = [ev("Meera Joshi", "Mahesh Joshi", "D/o", "F", ("9811111111",), "AB-1000001"),
               ev("Meera Patel", "Mahesh Patel", "D/o", "F", ("9822222222",), "AB-1000002")]
        frag = ev("Meera ji", "Mahesh", "D/o", "F")
        p = link(evs + [frag])
        self.assertIsNone(p[-1])


if __name__ == "__main__":
    unittest.main()

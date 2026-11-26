"""Tests for Milestone M2: identity resolution, anti-merge constraints, >=80-bit PSNs, and shuffle invariance."""
from __future__ import annotations

import csv
import io
import json
import random
import re
from pathlib import Path

from raksha.config import derive_subkey
from raksha.identity.evidence import (
    NameForm,
    RecordEvidence,
    RelativeClue,
    extract_all_evidences,
    make_name_form,
)
from raksha.identity.linker import resolve_clusters, score_pair
from raksha.identity.pseudonymizer import issue_cluster_pseudonyms
from raksha.pipeline import run_pipeline


def _ev(
    rel: str,
    loc: str = "row:0",
    *,
    name: str | None = None,
    relative: tuple[str, str] | None = None,
    sex: str | None = None,
    dob: str | None = None,
    age: int | None = None,
    village: str | None = None,
    phone: str | None = None,
    aadhaar: str | None = None,
    abha: str | None = None,
    mrn: tuple[str, str] | None = None,
    hosp: str = "h1",
    is_frag: bool = False,
) -> RecordEvidence:
    content = json.dumps(
        {
            "rel": rel,
            "loc": loc,
            "name": name,
            "rel": relative,
            "sex": sex,
            "dob": dob,
            "age": age,
            "v": village,
            "ph": phone,
            "aad": aadhaar,
            "abh": abha,
            "mrn": mrn,
        },
        sort_keys=True,
    )
    import hashlib

    ev = RecordEvidence(
        rel=rel,
        loc=loc,
        content_hash=hashlib.sha256(content.encode("utf-8")).hexdigest(),
        hospital=hosp,
        sex=sex,
        dob=dob,
        age=age,
        is_fragmentary=is_frag,
    )
    if name:
        nf = make_name_form(name, is_fragment=is_frag)
        if nf:
            ev.names.append(nf)
    if relative:
        rtok, rp = (
            make_name_form(relative[1]).norm_tokens,
            make_name_form(relative[1]).phonetic_tokens,
        )
        ev.relatives.append(RelativeClue(relation=relative[0], norm_tokens=rtok, phonetic_tokens=rp))
    if village:
        vf = make_name_form(village)
        if vf:
            ev.villages.add(" ".join(vf.norm_tokens))
            ev.village_phonetics.add(" ".join(vf.phonetic_tokens))
    if phone:
        ev.phones.add(phone)
    if aadhaar:
        ev.aadhaars.add(aadhaar)
        ev.has_strong_id = True
    if abha:
        ev.abhas.add(abha)
        ev.has_strong_id = True
    if mrn:
        ev.scoped_mrns.add(mrn)
        ev.has_strong_id = True
    return ev


def test_cross_script_and_corroborated_md_alias() -> None:
    e_en = _ev("a.csv", "row:0", name="Mohammad Tariq", dob="1985-04-12", sex="M", village="Sitapur")
    e_hi = _ev("b.csv", "row:0", name="मोहम्मद तारिक", dob="1985-04-12", sex="M", village="Sitapur")
    e_abbr = _ev("c.csv", "row:0", name="Md. Tariq", dob="1985-04-12", sex="M", village="Sitapur")
    e_other_md = _ev("d.csv", "row:0", name="Md. Aslam", dob="1985-04-12", sex="M", village="Sitapur")

    res = resolve_clusters([e_en, e_hi, e_abbr, e_other_md])
    c_en = res.record_to_cluster[("a.csv", "row:0")]
    c_hi = res.record_to_cluster[("b.csv", "row:0")]
    c_ab = res.record_to_cluster[("c.csv", "row:0")]
    c_aslam = res.record_to_cluster[("d.csv", "row:0")]

    assert c_en == c_hi == c_ab
    assert c_aslam != c_en


def test_homonyms_never_merge_and_shared_family_phone_kept_separate() -> None:
    h1 = _ev("h1.csv", "row:0", name="Ramesh Kumar", relative=("S/o", "Babu Ram"), dob="1980-01-10", village="Rampur", phone="9876543210")
    h2 = _ev("h2.csv", "row:0", name="Ramesh Kumar", relative=("S/o", "Hari Lal"), dob="1992-08-20", village="Hardoi")
    son = _ev("rel.csv", "row:0", name="Vikas Kumar", dob="2005-03-04", phone="9876543210")

    res = resolve_clusters([h1, h2, son])
    c1 = res.record_to_cluster[("h1.csv", "row:0")]
    c2 = res.record_to_cluster[("h2.csv", "row:0")]
    c_son = res.record_to_cluster[("rel.csv", "row:0")]
    assert len({c1, c2, c_son}) == 3


def test_anti_transitive_merge_with_exclusive_id_conflict() -> None:
    a = _ev("a.csv", "row:0", name="Suresh Patel", dob="1975-05-05", sex="M", aadhaar="222233334444")
    b = _ev("b.csv", "row:0", name="Suresh Patel", dob="1975-05-05", sex="M", mrn=("h1", "MRN-1001"))
    c = _ev("c.csv", "row:0", name="Suresh Patel", dob="1975-05-05", sex="M", mrn=("h1", "MRN-1001"), aadhaar="555566667777")

    sc_ac, conf_ac = score_pair(a, c)
    assert conf_ac is True
    res = resolve_clusters([a, b, c])
    ca = res.record_to_cluster.get(("a.csv", "row:0"))
    cc = res.record_to_cluster.get(("c.csv", "row:0"))
    assert ca != cc


def test_fragmentary_attachment_unique_vs_ambiguous() -> None:
    meera1 = _ev("m1.csv", "row:0", name="Meera Devi", relative=("W/o", "Rajan Lal"), sex="F", village="Barabanki")
    frag_unique = _ev("n1.txt", "doc", name="Meera", relative=("W/o", "Rajan"), is_frag=True)
    res_ok = resolve_clusters([meera1, frag_unique])
    assert res_ok.record_to_cluster[("m1.csv", "row:0")] == res_ok.record_to_cluster[("n1.txt", "doc")]

    # Add an ambiguous second Meera married to another Rajan -> fragmentary must go to null!
    meera2 = _ev("m2.csv", "row:0", name="Meera Sharma", relative=("W/o", "Rajan Kumar"), sex="F", village="Unnao")
    res_amb = resolve_clusters([meera1, meera2, frag_unique])
    assert ("n1.txt", "doc") not in res_amb.record_to_cluster


def test_psn_entropy_secret_sensitivity_and_collision_avoidance(monkeypatch) -> None:
    e1 = _ev("r1.csv", "row:0", name="Kavita Devi", abha="91123456789012")
    e2 = _ev("r2.csv", "row:0", name="Anil Verma", abha="91987654321098")
    res = resolve_clusters([e1, e2])
    k1 = derive_subkey(b"secret-alpha-1", b"pseudonym")
    k2 = derive_subkey(b"secret-beta-2", b"pseudonym")

    p1 = issue_cluster_pseudonyms(res.clusters, k1)
    p2 = issue_cluster_pseudonyms(res.clusters, k2)
    for psn in p1.values():
        assert re.fullmatch(r"PSN-[0-9A-HJKMNP-TV-Z]{16}", psn)
    assert set(p1.values()) != set(p2.values())
    assert len(set(p1.values())) == len(p1)

    # Forced digest collision -> nonce branch must guarantee unique PSNs
    import hmac as _hmac_mod
    import raksha.identity.pseudonymizer as _psn_mod

    orig_new = _hmac_mod.new

    class _CollidingHMAC:
        def __init__(self, key, msg, digestmod):
            self._msg = msg
            self._real = orig_new(key, msg, digestmod)

        def digest(self):
            return b"\x00" * 32 if b"#c" not in self._msg else self._real.digest()

    monkeypatch.setattr(_psn_mod.hmac, "new", _CollidingHMAC)
    p_col = issue_cluster_pseudonyms(res.clusters, k1)
    assert len(set(p_col.values())) == 2


def _write_sample_dataset(root: Path, *, shuffle_seed: int | None = None) -> None:
    rows = [
        ["patient_name", "uhid", "dob", "sex", "mobile", "village"],
        ["Aarav Verma", "AR-2026-0001", "12/05/1988", "M", "9811223344", "Barabanki"],
        ["आरव वर्मा", "AR-2026-0001", "12/05/1988", "M", "9811223344", "Barabanki"],
        ["Meera Singh", "AR-2026-0002", "03/11/1991", "F", "9822334455", "Sitapur"],
    ]
    header, body = rows[0], rows[1:]
    if shuffle_seed is not None:
        rng = random.Random(shuffle_seed)
        rng.shuffle(body)

    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(header)
    for r in body:
        w.writerow(r)

    json_recs = [
        {"patient_name": "Aarav Verma", "hospital_uhid": "AR-2026-0001", "test": "Hb", "val": "13.2 g/dL"},
        {"patient_name": "Meera Singh", "hospital_uhid": "AR-2026-0002", "test": "TSH", "val": "2.4 mIU/L"},
    ]
    if shuffle_seed is not None:
        random.Random(shuffle_seed + 1).shuffle(json_recs)

    fhir_bundle = {
        "resourceType": "Bundle",
        "type": "collection",
        "entry": [
            {
                "resource": {
                    "resourceType": "Patient",
                    "id": "p1",
                    "identifier": [{"system": "http://arogya.example/mrn", "value": "AR-2026-0002"}],
                    "name": [{"text": "Meera Singh"}],
                    "gender": "female",
                    "birthDate": "1991-11-03",
                }
            }
        ],
    }

    files = [
        ("arogya_patients.csv", buf.getvalue()),
        ("arogya_labs.json", json.dumps({"hospital": "Arogya", "results": json_recs})),
        ("arogya_fhir.json", json.dumps(fhir_bundle)),
        (
            "notes/arogya_opd_note.txt",
            "Patient: Aarav Verma\nUHID: AR-2026-0001\nDx: Viral fever. Tab Paracetamol 500 mg BD.\n",
        ),
    ]
    if shuffle_seed is not None:
        random.Random(shuffle_seed + 2).shuffle(files)
        files = [(f"sub_{len(files) - i}/{name}", txt) for i, (name, txt) in enumerate(files)]

    for rel, content in files:
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8", newline="")


def test_pipeline_psn_injection_and_shuffle_invariance(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("RAKSHA_SECRET", "m2-shuffle-test-secret-xyz")

    in1 = tmp_path / "in1"
    out1 = tmp_path / "out1"
    _write_sample_dataset(in1, shuffle_seed=None)
    assert run_pipeline(in1, out1) == 0

    in2 = tmp_path / "in2"
    out2 = tmp_path / "out2"
    _write_sample_dataset(in2, shuffle_seed=42)
    assert run_pipeline(in2, out2) == 0

    def _summary_clusters(out_dir: Path) -> tuple[dict[str, int], set[str]]:
        recs = [
            json.loads(line)
            for line in (out_dir / "records.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        counts: dict[str, int] = {}
        psns: set[str] = set()
        for r in recs:
            psn = r["patient"]
            assert psn is not None and psn.startswith("PSN-")
            psns.add(psn)
            counts[psn] = counts.get(psn, 0) + 1
            file_txt = (out_dir / r["file"]).read_text(encoding="utf-8")
            assert psn in file_txt
        return counts, psns

    c1, p1 = _summary_clusters(out1)
    c2, p2 = _summary_clusters(out2)
    assert p1 == p2
    assert len(p1) == 2
    assert sorted(c1.values()) == sorted(c2.values()) == [3, 4]

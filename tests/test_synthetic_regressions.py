"""Regression tests for multi-patient keyed dicts, repeated IDs, MRNs, ragged CSVs, Hindi clinical text, and high-frequency cluster bridging."""
from __future__ import annotations

import base64
import csv
import io
import json
from pathlib import Path

from raksha.detect.structured import (
    infer_table_schema,
    process_table_rows,
    sanitize_header_or_key,
    sanitize_json_data,
)
from raksha.identity.evidence import (
    NameForm,
    RecordEvidence,
    extract_all_evidences,
    has_multi_patient_keyed_dict,
)
from raksha.identity.linker import resolve_clusters
from raksha.pipeline import run_pipeline


def test_multi_patient_keyed_dict_not_merged_and_keys_redacted(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("RAKSHA_SECRET", "reg-test-secret-01")
    inp = tmp_path / "in"
    out = tmp_path / "out"
    (inp / "lab").mkdir(parents=True)
    (inp / "registration").mkdir(parents=True)

    reg_csv = (
        "MRN,Patient_Name,Age,Sex,Mobile\n"
        "XYZ/26/1001,Deepak Kulkarni,44,M,9811100011\n"
        "XYZ/26/1002,Smita Sawant,38,F,9811100022\n"
    )
    (inp / "registration" / "reg.csv").write_text(reg_csv, encoding="utf-8")

    keyed = {
        "batch_date": "2026-02-14",
        "results_by_patient": {
            "Deepak Kulkarni (9811100011)": {"HbA1c": "7.1%", "code": "XYZ/26/1001"},
            "Smita Sawant (9811100022)": {"HbA1c": "5.8%", "code": "XYZ/26/1002"},
            "Xyloqen0 Zindalor (9811100033)": {"HbA1c": "6.2%", "code": "XYZ/26/1003"},
        },
    }
    assert has_multi_patient_keyed_dict(keyed) is True
    (inp / "lab" / "keyed.json").write_text(json.dumps(keyed), encoding="utf-8")

    assert run_pipeline(inp, out) == 0

    san_keyed = (out / "sanitized" / "lab" / "keyed.json").read_text(encoding="utf-8")
    for leak in ("Deepak", "Kulkarni", "Smita", "Sawant", "Xyloqen0", "Zindalor", "9811100011", "9811100022", "XYZ/26/1001"):
        assert leak not in san_keyed
    assert "7.1%" in san_keyed

    recs = [json.loads(line) for line in (out / "records.jsonl").read_text(encoding="utf-8").splitlines()]
    keyed_recs = [r for r in recs if r["file"].endswith("keyed.json")]
    assert len(keyed_recs) == 1
    assert keyed_recs[0]["patient"] is None
    records_blob = (out / "records.jsonl").read_text(encoding="utf-8")
    assert "Xyloqen0" not in records_blob and "9811100011" not in records_blob


def test_conflicting_identities_and_repeated_shared_phone_pruned():
    evs = []
    for i, nm in enumerate(["Rohan Deshpande", "Meera Nair", "Arvind Kulkarni", "Neha Joshi"]):
        evs.append(
            RecordEvidence(
                rel=f"f{i}.csv",
                loc="row:0",
                content_hash=f"h{i}",
                hospital="hosp1",
                names=[NameForm(nm, tuple(nm.lower().split()), tuple(nm.lower().split()))],
                phones={"9899000001"},  # Shared desk phone across 4 mutually disagreeing people
                global_mrns={f"HSP-26-{1000 + i}"},
                has_strong_id=True,
            )
        )
    res = resolve_clusters(evs)
    cluster_ids = {res.record_to_cluster[("f" + str(i) + ".csv", "row:0")] for i in range(4)}
    assert len(cluster_ids) == 4


def test_bilingual_paren_alias_operator_base64_filename_and_fhir_url(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("RAKSHA_SECRET", "reg-test-secret-02")
    inp = tmp_path / "in"
    out = tmp_path / "out"
    (inp / "notes").mkdir(parents=True)
    (inp / "lab").mkdir(parents=True)
    (inp / "fhir").mkdir(parents=True)

    (inp / "notes" / "hi_note.txt").write_text(
        "मरीज़ का नाम: नंदिनी साठे (Nandini Sathe), उम्र/लिंग: 34/F, पंजीयन क्रमांक: ABC:UHID:26:77123\n"
        "Duty Sister: Vital chart updated at 09:00.\n"
        "Address: 12B, Azad Chowk, behind City Hospital, Gram: Rampur, Distt: Pune, PIN: 411001\n",
        encoding="utf-8",
    )

    inner_b64 = base64.b64encode(
        "Patient: Chetan Phadke\nMobile: 9822334455\nHb: 13.4 g/dL".encode("utf-8")
    ).decode("ascii")
    payload = {"operator_account": "Kiran Kulkarni", "raw_blob": inner_b64, "test": "CBC"}
    (inp / "lab" / "9822334455_Chetan_Phadke_cbc.json").write_text(json.dumps(payload), encoding="utf-8")

    bundle = {
        "resourceType": "Bundle",
        "type": "collection",
        "meta": {"source": "https://emr.example.org/fetch?patient=Gaurav%20Nimbalkar&uhid=ZPM:UHID:26:55441"},
        "entry": [
            {
                "resource": {
                    "resourceType": "Patient",
                    "id": "p1",
                    "text": {"status": "generated", "div": "<div xmlns=\"http://www.w3.org/1999/xhtml\">Gaurav Nimbalkar DOB 1988-03-14 Gram Shirur</div>"},
                    "name": [{"text": "Gaurav Nimbalkar"}],
                    "gender": "male",
                    "birthDate": "1988-03-14",
                }
            }
        ],
    }
    (inp / "fhir" / "b1.json").write_text(json.dumps(bundle), encoding="utf-8")

    assert run_pipeline(inp, out) == 0

    hi_out = (out / "sanitized" / "notes" / "hi_note.txt").read_text(encoding="utf-8")
    for secret_str in ("नंदिनी साठे", "Nandini Sathe", "ABC:UHID:26:77123", "Azad Chowk", "behind City Hospital", "Rampur", "411001"):
        assert secret_str not in hi_out
    assert "Duty Sister" in hi_out

    lab_files = list((out / "sanitized" / "lab").glob("*.json"))
    assert len(lab_files) == 1
    assert "Chetan" not in lab_files[0].name and "Phadke" not in lab_files[0].name and "9822334455" not in lab_files[0].name
    lab_data = json.loads(lab_files[0].read_text(encoding="utf-8"))
    assert "Kiran Kulkarni" not in json.dumps(lab_data)
    decoded = base64.b64decode(lab_data["raw_blob"]).decode("utf-8")
    assert "Chetan Phadke" not in decoded and "9822334455" not in decoded and "13.4 g/dL" in decoded

    fhir_out = (out / "sanitized" / "fhir" / "b1.json").read_text(encoding="utf-8")
    assert "Gaurav" not in fhir_out and "Nimbalkar" not in fhir_out and "ZPM:UHID:26:55441" not in fhir_out and "Shirur" not in fhir_out


def test_mrn_variants_visit_uid_gram_sabha_and_headers_intact():
    header = [
        "Encounter_Code", "Beneficiary_Nm", "Gram_Sabha", "Domicile_Addr",
        "Clinical_Remarks", "ABHA_Health_Acc", "Birth_MDY", "Visit_MDY",
        "Verifier_Raghav_Bhatia_9811223344",
    ]
    body = [
        ["XYZ-B26-441122", "Pradeep Joshi", "Rampur", "H.No. 14, Main Bazaar", "Fever x 2 days", "91-1234-5678-9012", "03/15/1984", "07/22/2026", "verified"],
        ["KLM:UHID:26:10982", "Sunita Rao", "Alipur", "H.No. 22, Station Road", "Dry cough", "91-2234-5678-9012", "08/19/1979", "07/25/2026", "verified"],
        ["ZYN_REG_26_440912", "Tariq Sheikh", "Bharthana", "H.No. 9, Gandhi Chowk", "Headache", "91-3234-5678-9012", "11/02/1991", "08/01/2026", "verified"],
    ]
    ts = infer_table_schema(header, body)
    assert ts.col_types[0] == "MRN"
    assert ts.col_types[2] == "VILLAGE"
    assert ts.col_types[4] is None
    assert ts.col_types[5] == "ABHA_NUMBER"
    assert ts.col_types[6] == "DOB"
    assert ts.part_hints[7] == "CLINICAL_DATE"

    ch, cb = process_table_rows([header] + body)
    assert ch[2] == "Gram_Sabha"
    assert ch[3] == "Domicile_Addr"
    assert ch[4] == "Clinical_Remarks"
    assert ch[5] == "ABHA_Health_Acc"
    assert ch[6] == "Birth_MDY"
    assert ch[7] == "Visit_MDY"
    assert "Raghav" not in ch[8] and "9811223344" not in ch[8]
    assert cb[0][0] == "[MRN]" and cb[1][0] == "[MRN]" and cb[2][0] == "[MRN]"
    assert sanitize_header_or_key("order_id") == "order_id"


def test_ragged_csv_first_last_landmark_age_and_district_preserved(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("RAKSHA_SECRET", "reg-test-secret-03")
    inp = tmp_path / "in"
    out = tmp_path / "out"
    inp.mkdir()

    csv_text = (
        "Reg_ID,Given_Nm,Family_Nm,Kin_Nm,Age_Yrs,Sex,Birth_MDY,Address_Line,District,PIN,Tel_No,Visit_MDY\n"
        "XYZ_REG_26_100001,Kishor,Gokhale,Vasant Gokhale,71,M,05/04/1955,\"H.No. 88, Tola Bazaar, near Bus Stand\",Jhansi,284001,9876500001,06/14/2026\n"
        "XYZ_REG_26_100002,Suresh,Panchal,Manoj Panchal,68,M,04/03/1958,Azad Chowk, behind Old Temple,Jhansi,284002,9876500002,06/18/2026\n"
    )
    (inp / "reg.csv").write_text(csv_text, encoding="utf-8")
    assert run_pipeline(inp, out) == 0

    rows = list(csv.reader((out / "sanitized" / "reg.csv").read_text(encoding="utf-8").splitlines()))
    rag = rows[2]
    assert "Suresh" not in " ".join(rag) and "Panchal" not in " ".join(rag) and "Manoj" not in " ".join(rag)
    assert "Azad Chowk" not in " ".join(rag) and "behind Old Temple" not in " ".join(rag)
    assert "68" not in rag  # banded age on ragged row
    assert "Jhansi" in rag  # District preserved


def test_hinglish_comma_linking_and_hindi_clinical_preservation(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("RAKSHA_SECRET", "reg-test-secret-04")
    inp = tmp_path / "in"
    out = tmp_path / "out"
    (inp / "reg").mkdir(parents=True)
    (inp / "disch").mkdir(parents=True)

    (inp / "reg" / "r.csv").write_text(
        "MRN,Patient_Name,Rel_Type,Rel_Name,Village,Age,Sex,Mobile\n"
        "ABC/26/9001,Harishankar Verma,S/o,Omprakash Verma,Sitapur,52,M,9810010099\n",
        encoding="utf-8",
    )
    (inp / "disch" / "frag.txt").write_text(
        "Short Slip (01/05/2026 - 04/05/2026)\n"
        "Pt: Harishankar Verma, S/o Omprakash Verma, Gram Sitapur.\n"
        "शिकायत: पैरों में सूजन, कमज़ोरी और चक्कर आना | पूर्व इतिहास: मधुमेह ६ वर्षों से, दमा रोग\n"
        "उपचार: सेफ्ट्रियाक्सोन १ ग्राम आईवी दिन में दो बार\n",
        encoding="utf-8",
    )
    assert run_pipeline(inp, out) == 0

    frag_out = (out / "sanitized" / "disch" / "frag.txt").read_text(encoding="utf-8")
    assert "Harishankar" not in frag_out and "Omprakash" not in frag_out and "Sitapur" not in frag_out
    for med in ("पैरों में सूजन", "कमज़ोरी और चक्कर आना", "मधुमेह ६ वर्षों से", "दमा रोग", "सेफ्ट्रियाक्सोन १ ग्राम आईवी दिन में दो बार"):
        assert med in frag_out, f"Clinical phrase lost: {med}\n{frag_out}"

    recs = [json.loads(l) for l in (out / "records.jsonl").read_text(encoding="utf-8").splitlines()]
    psns = {r["patient"] for r in recs}
    assert len(psns) == 1 and None not in psns


def test_high_visit_count_patient_links_beyond_50_without_homonym_merge():
    items_data = []
    # 55 visits at hospB + 2 visits at hospA for same patient + 2 homonyms with different phone/age
    evs: list[RecordEvidence] = []
    for i in range(55):
        evs.append(
            RecordEvidence(
                rel="bulk.json",
                loc=f"/b/{i}",
                content_hash=f"b{i:03d}",
                hospital="hospB",
                names=[NameForm("Mahesh Kulkarni", ("mahesh", "kulkarni"), ("mhs", "kulkrni"))],
                phones={"9812000111"},
                age=41,
                sex="M",
                global_mrns={"HB-26-5001"},
                scoped_mrns={("hospb", "HB-26-5001")},
                has_strong_id=True,
            )
        )
    for j in range(2):
        evs.append(
            RecordEvidence(
                rel="reg.csv",
                loc=f"row:{j}",
                content_hash=f"a{j:03d}",
                hospital="hospA",
                names=[NameForm("Mahesh Kulkarni", ("mahesh", "kulkarni"), ("mhs", "kulkrni"))],
                phones={"9812000111"},
                age=41,
                sex="M",
                global_mrns={"HA/26/9001"},
                scoped_mrns={("hospa", "HA/26/9001")},
                has_strong_id=True,
            )
        )
    evs.append(
        RecordEvidence(
            rel="hom.csv",
            loc="row:0",
            content_hash="h999",
            hospital="hospC",
            names=[NameForm("Mahesh Kulkarni", ("mahesh", "kulkarni"), ("mhs", "kulkrni"))],
            phones={"9899999888"},
            age=65,
            sex="M",
            global_mrns={"HC-26-7777"},
            scoped_mrns={("hospc", "HC-26-7777")},
            has_strong_id=True,
        )
    )
    res = resolve_clusters(evs)
    c_bulk0 = res.record_to_cluster[("bulk.json", "/b/0")]
    c_reg0 = res.record_to_cluster[("reg.csv", "row:0")]
    c_hom = res.record_to_cluster[("hom.csv", "row:0")]
    assert c_bulk0 == c_reg0
    assert c_hom != c_bulk0


def test_hindi_father_label_and_local_alias_do_not_drop_record_evidence():
    txt = (
        "मरीज़ का नाम: कमला देवी\n"
        "पिता का नाम: मोहन लाल\n"
        "पंजीकरण सं.: XYZ/26/10099\n"
    )
    json_rec = {
        "roster": [
            {
                "patient_ref": "NVP27-550099",
                "name": {"given_nm": "Claire", "sur_nm": "Moreau", "local": "मीरा शुक्ला"},
                "uidai_int": 710158385699,
            }
        ]
    }
    evs = extract_all_evidences([
        ("disch/d1.txt", "txt", txt),
        ("roster/r1.json", "json", json_rec),
    ])
    assert len(evs) == 2
    assert len(evs[0].names) == 1
    assert evs[0].relatives and "mohan" in evs[0].relatives[0].norm_tokens


def test_short_first_name_variant_with_shared_mrn_links():
    evs = [
        RecordEvidence(
            rel="reg.csv",
            loc="row:0",
            content_hash="h_reg",
            hospital="hosp1",
            names=[NameForm("VIJAY YADAV", ("vijay", "yadav"), ("wijy", "ydw"))],
            global_mrns={"HSP/26/001122"},
            scoped_mrns={("hosp1", "HSP/26/001122")},
            age=42,
            sex="M",
            has_strong_id=True,
        ),
        RecordEvidence(
            rel="lab.csv",
            loc="row:0",
            content_hash="h_lab",
            hospital="lab1",
            names=[NameForm("Vijai Yadav", ("vijai", "yadav"), ("wijai", "ydw"))],
            global_mrns={"HSP/26/001122"},
            scoped_mrns={("lab1", "HSP/26/001122")},
            age=44,
            sex="M",
            has_strong_id=True,
        ),
        RecordEvidence(
            rel="opd.json",
            loc="/visits/0",
            content_hash="h_opd",
            hospital="hosp1",
            names=[NameForm("Vijay", ("vijay",), ("wijy",), is_fragment_first_only=True)],
            global_mrns={"HSP/26/001122"},
            scoped_mrns={("hosp1", "HSP/26/001122")},
            has_strong_id=True,
        ),
    ]
    res = resolve_clusters(evs)
    cids = {res.record_to_cluster[(e.rel, e.loc)] for e in evs}
    assert len(cids) == 1

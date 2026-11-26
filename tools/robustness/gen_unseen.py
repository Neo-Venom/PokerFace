"""Generate a brand-new unseen synthetic hospital dataset with ground truth (Faker en_IN/hi_IN).

Deliberately differs from dev/input in hospital prefixes, MRN formats, column/key names,
note templates, homonyms, doctor-patient name overlaps, and planted canaries.
"""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import random
import shutil
from pathlib import Path

from faker import Faker


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_file(path: Path, data: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return _sha256_bytes(data)


def generate_unseen_dataset(out_dir: Path, seed: int = 2027, n_patients: int = 24) -> dict:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    inp = out_dir / "input"
    inp.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    fake_en = Faker("en_IN")
    fake_hi = Faker("hi_IN")
    Faker.seed(seed)

    # Unseen hospital MRN styles
    mrn_fmts = [
        lambda i: f"KGMU/27/{10010 + i}",
        lambda i: f"VHC-27-{80100 + i}",
        lambda i: f"ARC27-{300100 + i}",
        lambda i: f"SYD-{500100 + i}",
    ]

    patients = []
    for i in range(n_patients):
        fn, ln = fake_en.first_name(), fake_en.last_name()
        rfn, rln = fake_en.first_name(), ln
        dfn, dln = fake_hi.first_name(), fake_hi.last_name()
        drfn, drln = fake_hi.first_name(), dln
        # Make patient 0 & 1 homonyms (same name, different identity/village)
        if i == 1:
            fn, ln = patients[0]["fn"], patients[0]["ln"]
            dfn, dln = patients[0]["dfn"], patients[0]["dln"]
        phone = f"{rng.choice('6789')}{rng.randint(100000000, 999999999)}"
        aadhaar = "".join(str(rng.randint(2, 9)) if j == 0 else str(rng.randint(0, 9)) for j in range(12))
        abha = f"91-{rng.randint(1000,9999)}-{rng.randint(1000,9999)}-{rng.randint(1000,9999)}"
        pan = "".join(rng.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ", k=5)) + f"{rng.randint(1000,9999)}" + rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        dob = f"{rng.randint(1945, 2010)}-{rng.randint(1,12):02d}-{rng.randint(1,28):02d}"
        vill = f"{fake_en.city_suffix().title()}pur {rng.choice(['Kalan', 'Khurd', 'Buzurg'])}".strip()
        pin = f"{rng.randint(1,8)}{rng.randint(10000,99999)}"
        patients.append({
            "idx": i,
            "mrn": mrn_fmts[i % len(mrn_fmts)](i),
            "fn": fn, "ln": ln, "name": f"{fn} {ln}",
            "rfn": rfn, "rln": rln, "rel": f"{rfn} {rln}",
            "deva_name": f"{dfn} {dln}", "dfn": dfn, "dln": dln,
            "deva_rel": f"{drfn} {drln}",
            "phone": phone,
            "aadhaar": aadhaar,
            "abha": abha,
            "abha_addr": f"{fn.lower()}.{i}@abdm",
            "pan": pan,
            "email": f"{fn.lower()}.{ln.lower()}{i}@mailtest.example",
            "dob": dob,
            "house": f"H.No. {rng.randint(10, 499)}",
            "locality": f"Mohalla {fake_en.last_name()} Nagar",
            "landmark": f"near {fake_en.first_name()} Chowk",
            "village": vill,
            "tehsil": f"Tehsil {fake_en.city()}",
            "district": fake_en.state(),
            "pin": pin,
            "age": 2026 - int(dob[:4]),
            "sex": rng.choice(["M", "F"]),
        })

    # Plant 2 doctors that share exact names with patients[2] and patients[3] (hard overlap trap!)
    doctors = [
        {"en": f"Dr. {patients[2]['name']}", "hi": f"डॉ. {patients[2]['deva_name']}"},
        {"en": f"Dr. {patients[3]['name']}", "hi": f"डॉ. {patients[3]['deva_name']}"},
        {"en": f"Dr. {fake_en.first_name()} {fake_en.last_name()}", "hi": f"डॉ. {fake_hi.first_name()} {fake_hi.last_name()}"},
    ]

    # 3 planted canaries with completely unseen gibberish names
    canaries = [
        {"name": "Zorath Qvendal", "mrn": "CAN-99-887766", "phone": "9988776655"},
        {"name": "Krixel Vaelorian", "mrn": "KGMU/27/99991", "phone": "8877665544"},
        {"name": "Mivren Trelok", "mrn": "SYD-999111", "phone": "7766554433"},
    ]

    truth_files: dict[str, dict] = {}

    def add_truth(rel: str, sha: str, units: dict, expect: str = "sanitize"):
        truth_files[rel] = {"sha256": sha, "expect": expect, "units": units}

    # 1. Unseen Registration CSV (UTF-8 BOM, completely renamed columns!)
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["pt_uid", "beneficiary_full_name", "ling", "umar_yrs", "janm_dinank", "guardian_full", "ghar_pata", "gram_panchayat", "block_tehsil", "dist_name", "dak_code", "sampark_sutra", "uidai_num", "abdm_no"])
    csv_units = {}
    for i, pt in enumerate(patients[:14]):
        addr = f"C/O {pt['rel']}, {pt['house']}, {pt['locality']}, {pt['landmark']}, {pt['tehsil']}"
        w.writerow([pt["mrn"], pt["name"], pt["sex"], pt["age"], pt["dob"], f"S/o {pt['rel']}", addr, pt["village"], pt["tehsil"], pt["district"], pt["pin"], f"+91 {pt['phone'][:5]} {pt['phone'][5:]}", f"{pt['aadhaar'][:4]} {pt['aadhaar'][4:8]} {pt['aadhaar'][8:]}", pt["abha"]])
        csv_units[f"row:{i}"] = {
            "entities": [
                {"type": "MRN", "script": "latin", "channel": "content", "surface": pt["mrn"]},
                {"type": "PATIENT_NAME", "script": "latin", "channel": "content", "surface": pt["name"]},
                {"type": "DOB", "script": "latin", "channel": "content", "surface": pt["dob"]},
                {"type": "RELATIVE_NAME", "script": "latin", "channel": "content", "surface": pt["rel"]},
                {"type": "HOUSE", "script": "latin", "channel": "content", "surface": pt["house"]},
                {"type": "LOCALITY", "script": "latin", "channel": "content", "surface": pt["locality"]},
                {"type": "LANDMARK", "script": "latin", "channel": "content", "surface": pt["landmark"]},
                {"type": "VILLAGE", "script": "latin", "channel": "content", "surface": pt["village"]},
                {"type": "PIN", "script": "latin", "channel": "content", "surface": pt["pin"]},
                {"type": "PHONE", "script": "latin", "channel": "content", "surface": pt["phone"]},
                {"type": "AADHAAR", "script": "latin", "channel": "content", "surface": pt["aadhaar"]},
                {"type": "ABHA_NUMBER", "script": "latin", "channel": "content", "surface": pt["abha"]},
            ],
            "preserve": [],
        }
    sha = _write_file(inp / "reg/metro_beneficiaries.csv", b"\xef\xbb\xbf" + buf.getvalue().encode("utf-8"))
    add_truth("reg/metro_beneficiaries.csv", sha, csv_units)

    # 2. Unseen TSV (UTF-16, Hindi/Hinglish columns + ABHA addresses + emails)
    tbuf = io.StringIO()
    tw = csv.writer(tbuf, delimiter="\t", lineterminator="\n")
    tw.writerow(["record_code", "mariz_naam", "pita_pati", "gaon_name", "pin_zip", "contact_phone", "abdm_addr", "mail_id"])
    tsv_units = {}
    for i, pt in enumerate(patients[14:]):
        tw.writerow([pt["mrn"], pt["deva_name"], pt["deva_rel"], pt["village"], pt["pin"], pt["phone"], pt["abha_addr"], pt["email"]])
        tsv_units[f"row:{i}"] = {
            "entities": [
                {"type": "MRN", "script": "latin", "channel": "content", "surface": pt["mrn"]},
                {"type": "PATIENT_NAME", "script": "deva", "channel": "content", "surface": pt["deva_name"]},
                {"type": "RELATIVE_NAME", "script": "deva", "channel": "content", "surface": pt["deva_rel"]},
                {"type": "VILLAGE", "script": "latin", "channel": "content", "surface": pt["village"]},
                {"type": "PIN", "script": "latin", "channel": "content", "surface": pt["pin"]},
                {"type": "PHONE", "script": "latin", "channel": "content", "surface": pt["phone"]},
                {"type": "ABHA_ADDRESS", "script": "latin", "channel": "content", "surface": pt["abha_addr"]},
                {"type": "EMAIL", "script": "latin", "channel": "content", "surface": pt["email"]},
            ],
            "preserve": [],
        }
    sha = _write_file(inp / "reg/rural_roster.tsv", tbuf.getvalue().encode("utf-16"))
    add_truth("reg/rural_roster.tsv", sha, tsv_units)

    # 3. JSON with numeric int IDs under misspelled keys, PII keys, metadata path, and canary
    pt0 = patients[0]
    pt5 = patients[5]
    can0 = canaries[0]
    j_obj = {
        "_meta": {"export_src": f"D:\\Clinics\\Unseen\\{pt0['fn']}_{pt0['ln']}\\records.json"},
        "by_beneficiary": {pt5["name"]: pt5["mrn"]},
        "entries": [
            {
                "pt_uid": pt0["mrn"],
                "beneficiary_full_name": pt0["name"],
                "adhaar_int": int(pt0["aadhaar"]),
                "mob_num": int(pt0["phone"]),
                "doc_consultant": doctors[0]["en"],
                "pan_ref": pt0["pan"],
                "note": f"Pt. {can0['name']} (MRN: {can0['mrn']}, Mob: {can0['phone']}) referred for C/O fever for 3 days. Platelet Count 235000 /uL. Tab Paracetamol 650 mg 1-0-1.",
            }
        ],
    }
    sha = _write_file(inp / "records/unseen_dump.json", json.dumps(j_obj, ensure_ascii=False, indent=2).encode("utf-8"))
    add_truth(
        "records/unseen_dump.json",
        sha,
        {
            "doc": {
                "entities": [
                    {"type": "PATIENT_NAME", "script": "latin", "channel": "metadata", "surface": f"{pt0['fn']}_{pt0['ln']}"},
                    {"type": "PATIENT_NAME", "script": "latin", "channel": "key", "surface": pt5["name"]},
                    {"type": "MRN", "script": "latin", "channel": "key", "surface": pt5["mrn"]},
                    {"type": "MRN", "script": "latin", "channel": "content", "surface": pt0["mrn"]},
                    {"type": "PATIENT_NAME", "script": "latin", "channel": "content", "surface": pt0["name"]},
                    {"type": "AADHAAR", "script": "latin", "channel": "content", "surface": pt0["aadhaar"]},
                    {"type": "PHONE", "script": "latin", "channel": "content", "surface": pt0["phone"]},
                    {"type": "PAN", "script": "latin", "channel": "content", "surface": pt0["pan"]},
                    {"type": "CANARY", "script": "latin", "channel": "content", "surface": can0["name"]},
                    {"type": "CANARY", "script": "latin", "channel": "content", "surface": can0["mrn"]},
                    {"type": "CANARY", "script": "latin", "channel": "content", "surface": can0["phone"]},
                ],
                "preserve": [
                    {"text": doctors[0]["en"]},
                    {"text": "C/O fever for 3 days"},
                    {"text": "Platelet Count 235000 /uL"},
                    {"text": "Paracetamol 650 mg"},
                ],
            }
        },
    )

    # 4. Multilingual notes (EN, HI, Hinglish) with filename PII, canary, doctor overlap, eponyms
    pt6 = patients[6]
    can1 = canaries[1]
    fn_en = f"notes/Summary_{pt6['fn']}_{pt6['ln']}_{pt6['mrn'].replace('/', '_')}.txt"
    txt_en = (
        f"Beneficiary: {pt6['name']}\n"
        f"Caretaker: {pt6['rel']}\n"
        f"Consultant: {doctors[0]['en']} (MD)\n"
        f"Address: C/O {pt6['rel']}, {pt6['house']}, {pt6['locality']}, {pt6['village']}, {pt6['tehsil']} - {pt6['pin']}\n"
        f"Presenting complaints: C/O chest pain for 4 days. K/C/O Parkinson's disease.\n"
        f"{pt6['fn']} ji ka BP 128/82, SpO2 97%. Tab Atorvastatin 10 mg 0-0-1 x 14 days.\n"
        f"Followup note for {can1['name']} mob {can1['phone']}.\n"
    )
    sha = _write_file(inp / fn_en, txt_en.encode("utf-8"))
    add_truth(
        fn_en,
        sha,
        {
            "doc": {
                "entities": [
                    {"type": "PATIENT_NAME", "script": "latin", "channel": "filename", "surface": f"{pt6['fn']}_{pt6['ln']}"},
                    {"type": "PATIENT_NAME", "script": "latin", "channel": "content", "surface": pt6["name"]},
                    {"type": "RELATIVE_NAME", "script": "latin", "channel": "content", "surface": pt6["rel"]},
                    {"type": "HOUSE", "script": "latin", "channel": "content", "surface": pt6["house"]},
                    {"type": "LOCALITY", "script": "latin", "channel": "content", "surface": pt6["locality"]},
                    {"type": "VILLAGE", "script": "latin", "channel": "content", "surface": pt6["village"]},
                    {"type": "PIN", "script": "latin", "channel": "content", "surface": pt6["pin"]},
                    {"type": "PATIENT_NAME", "script": "hinglish", "channel": "content", "surface": f"{pt6['fn']} ji"},
                    {"type": "CANARY", "script": "latin", "channel": "content", "surface": can1["name"]},
                    {"type": "CANARY", "script": "latin", "channel": "content", "surface": can1["phone"]},
                ],
                "preserve": [
                    {"text": doctors[0]["en"]},
                    {"text": "Parkinson's disease"},
                    {"text": "BP 128/82"},
                    {"text": "Atorvastatin 10 mg"},
                ],
            }
        },
    )

    # Hindi note + phone filename + base64 attachment with canary
    pt7 = patients[7]
    can2 = canaries[2]
    txt_hi = (
        f"रोगी का नाम: {pt7['deva_name']}\n"
        f"अभिभावक: {pt7['deva_rel']}\n"
        f"{pt7['deva_name']} को 4 दिन से बुखार है।\n"
        f"{doctors[1]['hi']} ने जांच की।\n"
        f"निदान: डेंगू (A90)\n"
        f"Tab Paracetamol 650 mg 1-1-1 x 5 days\n"
    )
    sha = _write_file(inp / f"notes/{pt7['phone']}_hindi_note.txt", txt_hi.encode("utf-8"))
    add_truth(
        f"notes/{pt7['phone']}_hindi_note.txt",
        sha,
        {
            "doc": {
                "entities": [
                    {"type": "PHONE", "script": "latin", "channel": "filename", "surface": pt7["phone"]},
                    {"type": "PATIENT_NAME", "script": "deva", "channel": "content", "surface": pt7["deva_name"]},
                    {"type": "RELATIVE_NAME", "script": "deva", "channel": "content", "surface": pt7["deva_rel"]},
                ],
                "preserve": [
                    {"text": doctors[1]["hi"]},
                    {"text": "Paracetamol 650 mg"},
                    {"text": "A90"},
                ],
            }
        },
    )

    b64_payload = base64.b64encode(f"Signed by {can2['name']}, UHID {can2['mrn']}, phone {can2['phone']}.".encode("utf-8")).decode("ascii")
    b64_doc = {"doc_type": "consent", "attachment": {"contentType": "text/plain", "data": b64_payload}}
    sha = _write_file(inp / "attachments/consent_b64.json", json.dumps(b64_doc, indent=2).encode("utf-8"))
    add_truth(
        "attachments/consent_b64.json",
        sha,
        {
            "doc": {
                "entities": [
                    {"type": "CANARY", "script": "latin", "channel": "base64", "surface": can2["name"]},
                    {"type": "CANARY", "script": "latin", "channel": "base64", "surface": can2["mrn"]},
                    {"type": "CANARY", "script": "latin", "channel": "base64", "surface": can2["phone"]},
                ],
                "preserve": [],
            }
        },
    )

    # 5. Clean folder (zero PII)
    clean_proto = "HYPERTENSION PROTOCOL\n1. C/O headache without red flags -> check BP 140/90 mmHg.\n2. Tab Amlodipine 5 mg OD.\n3. Nodal doctor: Dr. Anirudh Kulkarni.\n"
    sha = _write_file(inp / "clean/htn_protocol.txt", clean_proto.encode("utf-8"))
    add_truth("clean/htn_protocol.txt", sha, {"doc": {"entities": [], "preserve": [{"text": "Amlodipine 5 mg"}, {"text": "Dr. Anirudh Kulkarni"}]}})

    # 6. Corrupted & unsupported files
    sha = _write_file(inp / "misc/bad_bytes.txt", b"Patient: " + can0["name"].encode() + b" \xff\xfe\xc3( mobile 9988776655\n")
    add_truth("misc/bad_bytes.txt", sha, {"doc": {"entities": [{"type": "CANARY", "script": "latin", "channel": "content", "surface": can0["name"]}], "preserve": []}}, expect="either")

    sha = _write_file(inp / "misc/broken.json", b'{"visits": [{"name": "Broken Record"')
    add_truth("misc/broken.json", sha, {"doc": {"entities": [], "preserve": []}}, expect="either")

    sha = _write_file(inp / "misc/empty.csv", b"")
    add_truth("misc/empty.csv", sha, {"doc": {"entities": [], "preserve": []}}, expect="either")

    sha = _write_file(inp / "misc/scan.pdf", b"%PDF-1.4\n% Fake binary PDF " + can1["name"].encode() + b"\n")
    add_truth("misc/scan.pdf", sha, {"doc": {"entities": [{"type": "CANARY", "script": "latin", "channel": "content", "surface": can1["name"]}], "preserve": []}}, expect="either")

    truth = {"files": truth_files, "canaries": [c["name"] for c in canaries]}
    (out_dir / "truth.json").write_text(json.dumps(truth, ensure_ascii=False, indent=2), encoding="utf-8")
    return truth


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate synthetic unseen dataset with ground truth")
    ap.add_argument("--output", default=".unseen_data", help="Destination folder (gitignored)")
    ap.add_argument("--seed", type=int, default=2027)
    ap.add_argument("--patients", type=int, default=24)
    a = ap.parse_args()
    truth = generate_unseen_dataset(Path(a.output), seed=a.seed, n_patients=a.patients)
    print(f"Generated unseen dataset at {a.output}: {len(truth['files'])} files, {len(truth['canaries'])} canaries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

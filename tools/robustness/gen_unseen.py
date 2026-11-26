"""Generate a large, genuinely unseen synthetic multi-hospital dataset (300+ patients, 5 hospital styles, 30 canaries).

Deliberately unlike dev/input: new hospital prefixes, new MRN shapes, shifted column order,
rare and non-Indian name pools alongside Indian en_IN/hi_IN, Hindi-only and Hinglish-only
narratives, homonym pairs, doctor-patient exact collisions, mixed encodings, and multi-document linkage truth.
"""
from __future__ import annotations

import argparse
import base64
import csv
import datetime as dt
import hashlib
import io
import json
import random
import shutil
from pathlib import Path

from faker import Faker

# Unseen non-Indian & rare surname/given-name pool mixed with Faker en_IN / hi_IN / en_GB / fr_FR
_RARE_FIRST_LATIN = [
    "Zephyrin", "Balthazar", "Kallisto", "Elowen", "Sorrel", "Tariqul", "Aurelien",
    "Cedric", "Nadezhda", "Svetlana", "Chiara", "Matteo", "Yusuke", "Haruki",
    "Siobhan", "Aoife", "Alistair", "Genevieve", "Thaddeus", "Octavia",
    "Lakshmi-Narayan", "K. R.", "M. S.", "Mohammad", "Mohd.", "Md.",
]
_RARE_LAST_LATIN = [
    "O'Sullivan", "Al-Mansoor", "Kowalski", "Nakamura", "Beauchamp", "Lindqvist",
    "Castellanos", "Van Der Berg", "Subramanyan-Iyer", "Chatterjee-Roy", "Thorne",
    "MacAllister", "Ziegler", "De Silva", "Marceau", "Bhattacharjee",
]
_UNSEEN_DEVA_NAMES = [
    ("ज़ेफ़िरिन", "थॉर्न"), ("अल-मंसूर", "खान"), ("सुब्रमण्यम", "अय्यर"), ("काव्यश्री", "मेनन"),
    ("देवांश", "कुलकर्णी"), ("नंदिनी", "बोस"), ("मोहम्मद", "अल्वी"), ("हर्षवर्धन", "राठौड़"),
    ("माधवी", "देशमुख"), ("पार्थसारथी", "सेन"), ("रुद्रप्रताप", "जादौन"), ("इरावती", "पाणिकर"),
]


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_file(path: Path, data: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return _sha256_bytes(data)


def _deva_digits(s: str) -> str:
    return s.translate(str.maketrans("0123456789", "०१२३४५६७८९"))


def generate_unseen_dataset(out_dir: Path, seed: int = 2028, n_patients: int = 320) -> dict:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    inp = out_dir / "input"
    inp.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    fake_en = Faker("en_IN")
    fake_gb = Faker("en_GB")
    fake_fr = Faker("fr_FR")
    fake_hi = Faker("hi_IN")
    Faker.seed(seed)

    # 5 completely distinct hospital styles and unseen MRN formats
    hospitals = [
        {
            "key": "kaveri",
            "name": "Kaveri Memorial Medical Centre, Mysore",
            "prefix": "kmm",
            "mrn": lambda i: f"KMM/27/{800100 + i}",
            "date_fmt": "mdy",
        },
        {
            "key": "pragati",
            "name": "Pragati Institute of Health Sciences, Bhopal",
            "prefix": "pih",
            "mrn": lambda i: f"PIH-27-{904000 + i}",
            "date_fmt": "dot_dmy",
        },
        {
            "key": "narmada",
            "name": "Narmada Valley Polyclinic, Indore",
            "prefix": "nvp",
            "mrn": lambda i: f"NVP27-{550000 + i}",
            "date_fmt": "iso",
        },
        {
            "key": "coastal",
            "name": "Coastal Multispeciality Care, Kochi",
            "prefix": "cmc",
            "mrn": lambda i: f"CMC-OPD-{70000 + i}",
            "date_fmt": "slash_dmy",
        },
        {
            "key": "vindhya",
            "name": "Vindhya Diagnostic & Emergency Trust, Rewa",
            "prefix": "vdet",
            "mrn": lambda i: f"VDET/2027/{1000 + i}",
            "date_fmt": "dash_dmy",
        },
    ]

    villages = [
        f"{p}{suf}"
        for p in ("Kishan", "Chand", "Madhav", "Dev", "Gopal", "Ratan", "Shiv", "Anand", "Sundar", "Nava", "Suraj", "Kalyan", "Har", "Prem", "Bhim")
        for suf in ("garh Kalan", "pur Khurd", "nagar", "ganj", "dih")
    ]
    districts = ["Mysore", "Bhopal", "Indore", "Ernakulam", "Rewa", "Gwalior", "Jabalpur", "Thrissur", "Salem", "Nagpur"]
    drugs = [
        ("Tab Telmisartan 40 mg", "0-0-1"),
        ("Tab Metformin 500 mg", "1-0-1"),
        ("Cap Pantoprazole 40 mg", "1-0-0"),
        ("Inj Ceftriaxone 1 g IV", "BD"),
        ("Tab Azithromycin 500 mg", "1-0-0"),
        ("Tab Atorvastatin 20 mg", "0-0-1"),
        ("Syr Ambroxol 10 ml", "TID"),
    ]
    conditions = [
        ("E11.9", "Type 2 diabetes mellitus"),
        ("I10", "Essential hypertension"),
        ("J18.9", "Community acquired pneumonia"),
        ("A97.9", "Dengue without warning signs"),
        ("N39.0", "Urinary tract infection"),
        ("G20", "Parkinson's disease"),
        ("K50.9", "Crohn's disease"),
    ]

    # Build >=300 patient identities including 12 homonym pairs (24 homonyms) + diverse name pools
    patients: list[dict] = []
    for i in range(n_patients):
        bucket = i % 5
        if bucket == 0:
            fn, ln = rng.choice(_RARE_FIRST_LATIN), rng.choice(_RARE_LAST_LATIN)
        elif bucket == 1:
            fn, ln = fake_gb.first_name(), fake_gb.last_name()
        elif bucket == 2:
            fn, ln = fake_fr.first_name(), fake_fr.last_name()
        else:
            fn, ln = fake_en.first_name(), fake_en.last_name()

        dfn, dln = _UNSEEN_DEVA_NAMES[i % len(_UNSEEN_DEVA_NAMES)] if i % 4 == 0 else (fake_hi.first_name(), fake_hi.last_name())
        rfn = fake_en.first_name()
        rln = ln.split()[-1] if " " in ln else ln
        drfn, drln = fake_hi.first_name(), dln

        # Make pairs (0,1), (2,3), ..., (22,23) exact homonyms with different village/dob/relative/age
        if i < 24 and i % 2 == 1:
            prev = patients[i - 1]
            fn, ln = prev["fn"], prev["ln"]
            dfn, dln = prev["dfn"], prev["dln"]

        dob_yr = rng.randint(1936, 2012)
        dob_m = rng.randint(1, 12)
        dob_d = rng.randint(1, 28)
        dob = f"{dob_yr}-{dob_m:02d}-{dob_d:02d}"
        age = 2027 - dob_yr
        hosp = hospitals[i % len(hospitals)]
        hosp2 = hospitals[(i + 2) % len(hospitals)]
        vill = villages[(i * 7 + (3 if i % 2 else 0)) % len(villages)]
        dist = districts[(i * 3 + (1 if i % 2 else 0)) % len(districts)]
        tehsil = f"{dist[:4]}block{(i % 4) + 1}"
        pin = f"{rng.randint(3, 7)}{rng.randint(10000, 99999)}"
        phone = f"{rng.choice('6789')}{100000000 + i * 289711 % 899999999:09d}"
        aadhaar = f"{rng.randint(2, 9)}{(10000000000 + i * 79192837) % 90000000000:011d}"
        abha = f"91-{1000 + (i * 17) % 8999}-{1000 + (i * 31) % 8999}-{1000 + (i * 47) % 8999}"
        pan = (
            "".join(rng.choices("ABCDEFGHJKLMNPQRSTUVWXYZ", k=5))
            + f"{1000 + (i * 23) % 8999}"
            + rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ")
        )
        adm_d = dt.date(2027, 1, 1) + dt.timedelta(days=rng.randint(0, 260))
        stay = rng.randint(2, 11)
        dis_d = adm_d + dt.timedelta(days=stay)

        patients.append({
            "pid": f"UPT{i:04d}",
            "hosp": hosp,
            "hosp2": hosp2,
            "mrn": hosp["mrn"](i),
            "mrn2": hosp2["mrn"](1000 + i),
            "fn": fn,
            "ln": ln,
            "name": f"{fn} {ln}",
            "dfn": dfn,
            "dln": dln,
            "deva_name": f"{dfn} {dln}",
            "rel_type": rng.choice(["S/o", "D/o", "W/o"]),
            "rel": f"{rfn} {rln}",
            "deva_rel": f"{drfn} {drln}",
            "sex": "M" if i % 2 == 0 else "F",
            "dob": dob,
            "age": age,
            "house": f"Qtr-{10 + (i * 13) % 880}",
            "locality": f"Sector {(i % 19) + 1} Colony",
            "landmark": f"opp Community Hall {(i % 9) + 1}",
            "village": vill,
            "tehsil": tehsil,
            "district": dist,
            "pin": pin,
            "phone": phone,
            "aadhaar": aadhaar,
            "abha": abha,
            "abha_addr": f"u{i}.{abs(hash(fn)) % 9999}@abdm",
            "pan": pan,
            "email": f"u{i}_{re_sub_alnum(ln)}@unseenclinic.example",
            "adm_d": adm_d,
            "dis_d": dis_d,
            "cond": conditions[i % len(conditions)],
            "drug": drugs[i % len(drugs)],
        })

    # 12 Doctors sharing exact names with patients[30..41] (doctor/patient collision trap!)
    doctors = [
        {"en": f"Dr. {patients[30 + k]['name']}", "hi": f"डॉ. {patients[30 + k]['deva_name']}"}
        for k in range(12)
    ]

    # 30 completely artificial canary strings (never appearing in Faker or standard name lists)
    con1 = "bcdfghjklmnpqrstvwxz"
    vow1 = "aeiou"
    canaries = []
    for c_idx in range(30):
        fn_c = "".join(rng.choice(con1 if j % 2 == 0 else vow1) for j in range(7)).title() + f"x{c_idx}"
        ln_c = "".join(rng.choice(con1 if j % 2 == 0 else vow1) for j in range(6)).title() + "qz"
        canaries.append({
            "name": f"{fn_c} {ln_c}",
            "mrn": f"CAN-{c_idx + 10:02d}-{900100 + c_idx * 37}",
            "phone": f"9{800000000 + c_idx * 135791:09d}",
        })

    truth_files: dict[str, dict] = {}

    def add_truth(rel: str, sha: str, units: dict, expect: str = "sanitize"):
        truth_files[rel] = {"sha256": sha, "expect": expect, "units": units}

    # --- 1. Hospital 1 CSV (Kaveri, UTF-8-BOM, MDY dates, SHIFTED columns!) ---
    h1_pts = [p for i, p in enumerate(patients) if i % 5 == 0]
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["dist_name", "postal_zip", "govt_uidai", "mobile_contact", "full_pt_name", "beneficiary_code", "sex_code", "birth_dt", "care_of_rel", "native_gram", "home_addr", "adm_dt", "dis_dt", "dx_display"])
    u1 = {}
    for row_i, pt in enumerate(h1_pts):
        adm_s = pt["adm_d"].strftime("%m/%d/%Y")
        dis_s = pt["dis_d"].strftime("%m/%d/%Y")
        dob_s = dt.date.fromisoformat(pt["dob"]).strftime("%m/%d/%Y")
        aad_fmt = f"{pt['aadhaar'][:4]}-{pt['aadhaar'][4:8]}-{pt['aadhaar'][8:]}"
        w.writerow([
            pt["district"], pt["pin"], aad_fmt, f"+91-{pt['phone']}", pt["name"], pt["mrn"], pt["sex"],
            dob_s, f"{pt['rel_type']} {pt['rel']}", pt["village"],
            f"{pt['house']}, {pt['locality']}, {pt['landmark']}", adm_s, dis_s, pt["cond"][1],
        ])
        u1[f"row:{row_i}"] = {
            "patient": pt["pid"], "link": "required",
            "entities": [
                {"type": "AADHAAR", "script": "latin", "channel": "content", "surface": pt["aadhaar"]},
                {"type": "PHONE", "script": "latin", "channel": "content", "surface": pt["phone"]},
                {"type": "PATIENT_NAME", "script": "latin", "channel": "content", "surface": pt["name"]},
                {"type": "MRN", "script": "latin", "channel": "content", "surface": pt["mrn"]},
                {"type": "DOB", "script": "latin", "channel": "content", "surface": dob_s},
                {"type": "RELATIVE_NAME", "script": "latin", "channel": "content", "surface": pt["rel"]},
                {"type": "VILLAGE", "script": "latin", "channel": "content", "surface": pt["village"]},
                {"type": "HOUSE", "script": "latin", "channel": "content", "surface": pt["house"]},
                {"type": "LOCALITY", "script": "latin", "channel": "content", "surface": pt["locality"]},
                {"type": "PIN", "script": "latin", "channel": "content", "surface": pt["pin"]},
            ],
            "preserve": [{"text": pt["cond"][1]}],
        }
    sha = _write_file(inp / "kaveri_intake_shifted.csv", b"\xef\xbb\xbf" + buf.getvalue().encode("utf-8"))
    add_truth("kaveri_intake_shifted.csv", sha, u1)

    # --- 2. Hospital 2 TSV (Pragati, UTF-16LE, Hindi headers, Devanagari digits & names, dot_dmy) ---
    h2_pts = [p for i, p in enumerate(patients) if i % 5 == 1]
    tbuf = io.StringIO()
    tw = csv.writer(tbuf, delimiter="\t", lineterminator="\n")
    tw.writerow(["उम्र", "ज़िला", "पिन_कोड", "दूरभाष", "पंजीकरण", "रोगी_नाम", "लिंग", "अभिभावक", "निवास_पता", "भर्ती_दिनांक"])
    u2 = {}
    for row_i, pt in enumerate(h2_pts):
        ph_deva = _deva_digits(pt["phone"])
        reg_s = pt["adm_d"].strftime("%d.%m.%Y")
        tw.writerow([
            f"{pt['age']} वर्ष", pt["district"], pt["pin"], ph_deva, pt["mrn"], pt["deva_name"],
            "पुरुष" if pt["sex"] == "M" else "महिला", pt["deva_rel"],
            f"मकान नं. {pt['house'].split('-')[-1]}, ग्राम {pt['village']}", reg_s,
        ])
        u2[f"row:{row_i}"] = {
            "patient": pt["pid"], "link": "required",
            "entities": [
                {"type": "PHONE", "script": "deva", "channel": "content", "surface": pt["phone"]},
                {"type": "MRN", "script": "latin", "channel": "content", "surface": pt["mrn"]},
                {"type": "PATIENT_NAME", "script": "deva", "channel": "content", "surface": pt["deva_name"]},
                {"type": "RELATIVE_NAME", "script": "deva", "channel": "content", "surface": pt["deva_rel"]},
                {"type": "VILLAGE", "script": "latin", "channel": "content", "surface": pt["village"]},
                {"type": "PIN", "script": "latin", "channel": "content", "surface": pt["pin"]},
            ],
            "preserve": [],
        }
    sha = _write_file(inp / "pragati_hindi_roster.tsv", tbuf.getvalue().encode("utf-16le"))
    # Include BOM so decode_bytes UTF-16 detector handles cleanly without guessing endianness without BOM! Wait: check decode_bytes: UTF-16 requires BOM (b'\xff\xfe')!
    sha = _write_file(inp / "pragati_hindi_roster.tsv", b"\xff\xfe" + tbuf.getvalue().encode("utf-16le"))
    add_truth("pragati_hindi_roster.tsv", sha, u2)

    # --- 3. Hospital 3 Nested JSON (Narmada, ISO dates, numeric int IDs, shifted keys, compound name) ---
    h3_pts = [p for i, p in enumerate(patients) if i % 5 == 2]
    recs3 = []
    u3 = {}
    for idx3, pt in enumerate(h3_pts):
        recs3.append({
            "patient_ref": pt["mrn"],
            "name": {"given_nm": pt["fn"], "sur_nm": pt["ln"], "local": pt["deva_name"]},
            "uidai_int": int(pt["aadhaar"]),
            "contct_num": int(pt["phone"]),
            "abdm_addr": pt["abha_addr"],
            "pan_card": pt["pan"],
            "mail_id": pt["email"],
            "umar": pt["age"],
            "gender": pt["sex"],
            "visit_date": pt["adm_d"].isoformat(),
            "address": {"line": f"{pt['house']}, {pt['locality']}", "village": pt["village"], "district": pt["district"], "pin": pt["pin"]},
            "rx": [{"drug": pt["drug"][0], "freq": pt["drug"][1]}],
            "doc_consultant": doctors[idx3 % len(doctors)]["en"],
        })
        u3[f"/roster/{idx3}"] = {
            "patient": pt["pid"], "link": "required",
            "entities": [
                {"type": "MRN", "script": "latin", "channel": "content", "surface": pt["mrn"]},
                {"type": "PATIENT_NAME", "script": "latin", "channel": "content", "surface": pt["fn"]},
                {"type": "AADHAAR", "script": "latin", "channel": "content", "surface": pt["aadhaar"]},
                {"type": "PHONE", "script": "latin", "channel": "content", "surface": pt["phone"]},
                {"type": "ABHA_ADDRESS", "script": "latin", "channel": "content", "surface": pt["abha_addr"]},
                {"type": "PAN", "script": "latin", "channel": "content", "surface": pt["pan"]},
                {"type": "EMAIL", "script": "latin", "channel": "content", "surface": pt["email"]},
                {"type": "VILLAGE", "script": "latin", "channel": "content", "surface": pt["village"]},
                {"type": "PIN", "script": "latin", "channel": "content", "surface": pt["pin"]},
            ],
            "preserve": [
                {"text": pt["drug"][0]},
                {"text": pt["drug"][1]},
                {"text": doctors[idx3 % len(doctors)]["en"]},
            ],
        }
    j3 = {"facility": {"name": hospitals[2]["name"]}, "roster": recs3}
    sha = _write_file(inp / "narmada_polyclinic.json", json.dumps(j3, ensure_ascii=False, indent=2).encode("utf-8"))
    add_truth("narmada_polyclinic.json", sha, u3)

    # --- 4. Hospital 4 cp1252 CSV (Coastal, shifted columns, clinical notes with degree/micro symbols) ---
    h4_pts = [p for i, p in enumerate(patients) if i % 5 == 3]
    cbuf = io.StringIO()
    cw = csv.writer(cbuf, lineterminator="\n")
    cw.writerow(["visit_dt", "client_ref", "pt_full", "age_at_visit", "care_of_rel", "phone_num", "city_town", "dx_text", "rx_text"])
    u4 = {}
    for row_i, pt in enumerate(h4_pts):
        cw.writerow([
            pt["adm_d"].strftime("%d/%m/%Y"), pt["mrn"], pt["name"], f"{pt['age']}/{pt['sex']}",
            pt["rel"], pt["phone"], pt["district"], f"Dx: {pt['cond'][1]}",
            f"{pt['drug'][0]} \u2013 Temp 98.6\u00b0F, Levothyroxine 50\u00b5g",
        ])
        u4[f"row:{row_i}"] = {
            "patient": pt["pid"], "link": "required",
            "entities": [
                {"type": "MRN", "script": "latin", "channel": "content", "surface": pt["mrn"]},
                {"type": "PATIENT_NAME", "script": "latin", "channel": "content", "surface": pt["name"]},
                {"type": "RELATIVE_NAME", "script": "latin", "channel": "content", "surface": pt["rel"]},
                {"type": "PHONE", "script": "latin", "channel": "content", "surface": pt["phone"]},
            ],
            "preserve": [{"text": pt["cond"][1]}, {"text": pt["drug"][0]}],
        }
    sha = _write_file(inp / "coastal_opd_cp1252.csv", cbuf.getvalue().encode("cp1252", errors="replace"))
    add_truth("coastal_opd_cp1252.csv", sha, u4)

    # --- 5. Hospital 5 FHIR R4 Bundles (40 patients) + Cross-Hospital External Lab JSON (all >=300 patients!) ---
    h5_pts = [p for i, p in enumerate(patients) if i % 5 == 4]
    for b_i, pt in enumerate(h5_pts[:40]):
        doc = doctors[b_i % len(doctors)]
        b64_note = base64.b64encode(
            f"Consent signed by {pt['rel']} for {pt['name']} ({pt['mrn']}), Tel {pt['phone']}.".encode("utf-8")
        ).decode("ascii")
        bundle = {
            "resourceType": "Bundle",
            "type": "document",
            "timestamp": f"{pt['dis_d'].isoformat()}T11:30:00+05:30",
            "entry": [
                {
                    "resource": {
                        "resourceType": "Organization",
                        "id": f"org-{b_i}",
                        "name": hospitals[4]["name"],
                        "address": [{"city": "Rewa"}],
                    }
                },
                {
                    "resource": {
                        "resourceType": "Practitioner",
                        "id": f"pr-{b_i}",
                        "name": [{"text": doc["en"]}],
                    }
                },
                {
                    "resource": {
                        "resourceType": "Patient",
                        "id": f"pat-{b_i}",
                        "identifier": [
                            {"system": "https://vindhya.example/mrn", "value": pt["mrn"]},
                            {"system": "https://healthid.ndhm.gov.in", "value": pt["abha"]},
                        ],
                        "name": [{"text": pt["name"]}],
                        "telecom": [{"system": "phone", "value": pt["phone"]}],
                        "gender": "male" if pt["sex"] == "M" else "female",
                        "birthDate": pt["dob"],
                        "address": [{"city": pt["village"], "district": pt["district"], "postalCode": pt["pin"]}],
                        "contact": [{"name": {"text": pt["rel"]}}],
                    }
                },
                {
                    "resource": {
                        "resourceType": "Encounter",
                        "id": f"enc-{b_i}",
                        "status": "finished",
                        "period": {"start": f"{pt['adm_d'].isoformat()}T09:00:00+05:30", "end": f"{pt['dis_d'].isoformat()}T15:00:00+05:30"},
                    }
                },
                {
                    "resource": {
                        "resourceType": "Condition",
                        "id": f"cnd-{b_i}",
                        "code": {"coding": [{"system": "http://hl7.org/fhir/sid/icd-10", "code": pt["cond"][0], "display": pt["cond"][1]}], "text": pt["cond"][1]},
                        "recordedDate": pt["adm_d"].isoformat(),
                    }
                },
                {
                    "resource": {
                        "resourceType": "DocumentReference",
                        "id": f"doc-{b_i}",
                        "content": [{"attachment": {"contentType": "text/plain", "data": b64_note}}],
                    }
                },
            ],
        }
        rel_f = f"fhir/vindhya_bundle_{b_i:03d}.json"
        sha = _write_file(inp / rel_f, json.dumps(bundle, ensure_ascii=False, indent=2).encode("utf-8"))
        add_truth(
            rel_f,
            sha,
            {
                "doc": {
                    "patient": pt["pid"], "link": "required",
                    "entities": [
                        {"type": "MRN", "script": "latin", "channel": "content", "surface": pt["mrn"]},
                        {"type": "ABHA_NUMBER", "script": "latin", "channel": "content", "surface": pt["abha"]},
                        {"type": "PATIENT_NAME", "script": "latin", "channel": "content", "surface": pt["name"]},
                        {"type": "PHONE", "script": "latin", "channel": "content", "surface": pt["phone"]},
                        {"type": "VILLAGE", "script": "latin", "channel": "content", "surface": pt["village"]},
                        {"type": "PIN", "script": "latin", "channel": "content", "surface": pt["pin"]},
                        {"type": "RELATIVE_NAME", "script": "latin", "channel": "base64", "surface": pt["rel"]},
                    ],
                    "preserve": [{"text": doc["en"]}, {"text": pt["cond"][1]}],
                }
            },
        )

    # Remainder of h5 patients in vindhya CSV
    vbuf = io.StringIO()
    vw = csv.writer(vbuf, lineterminator="\n")
    vw.writerow(["uhid", "patient_name", "gender", "age", "pin", "village", "district", "collected_on", "diagnosis"])
    u5 = {}
    for row_i, pt in enumerate(h5_pts[40:]):
        vw.writerow([pt["mrn"], pt["name"], pt["sex"], pt["age"], pt["pin"], pt["village"], pt["district"], pt["adm_d"].strftime("%d-%m-%Y"), pt["cond"][1]])
        u5[f"row:{row_i}"] = {
            "patient": pt["pid"], "link": "required",
            "entities": [
                {"type": "MRN", "script": "latin", "channel": "content", "surface": pt["mrn"]},
                {"type": "PATIENT_NAME", "script": "latin", "channel": "content", "surface": pt["name"]},
                {"type": "PIN", "script": "latin", "channel": "content", "surface": pt["pin"]},
                {"type": "VILLAGE", "script": "latin", "channel": "content", "surface": pt["village"]},
            ],
            "preserve": [{"text": pt["cond"][1]}],
        }
    sha = _write_file(inp / "vindhya_emergency.csv", vbuf.getvalue().encode("utf-8"))
    add_truth("vindhya_emergency.csv", sha, u5)

    # --- 6. Cross-hospital External Lab JSON linking EVERY patient (MRN + Name / Deva / DOB) ---
    ext_labs = []
    u_lab = {}
    for l_i, pt in enumerate(patients):
        ext_labs.append({
            "hospital_uhid": pt["mrn"],
            "patient_name": pt["deva_name"] if (l_i % 5 == 1) else pt["name"],
            "test": "Haemoglobin",
            "value": round(8.5 + (l_i % 60) * 0.1, 1),
            "unit": "g/dL",
            "collected": pt["dis_d"].isoformat(),
        })
        nm_e = pt["deva_name"] if (l_i % 5 == 1) else pt["name"]
        scr_e = "deva" if (l_i % 5 == 1) else "latin"
        u_lab[f"/results/{l_i}"] = {
            "patient": pt["pid"], "link": "required",
            "entities": [
                {"type": "MRN", "script": "latin", "channel": "content", "surface": pt["mrn"]},
                {"type": "PATIENT_NAME", "script": scr_e, "channel": "content", "surface": nm_e},
            ],
            "preserve": [{"text": "Haemoglobin"}],
        }
    sha = _write_file(inp / "central_diagnostics_export.json", json.dumps({"lab": "Apex Central Diagnostics", "results": ext_labs}, ensure_ascii=False, indent=2).encode("utf-8"))
    add_truth("central_diagnostics_export.json", sha, u_lab)

    # --- 7. Multi-style Hindi-only Notes (20) & Hinglish-only Notes (20) + Canary traps (all 30 canaries!) ---
    for k in range(20):
        pt = patients[50 + k]
        can = canaries[k]
        doc = doctors[k % len(doctors)]
        hi_txt = (
            f"{pt['hosp']['name']}\n"
            f"मरीज़ का नाम: {pt['deva_name']}   आयु: {pt['age']} वर्ष\n"
            f"पंजीकरण सं.: {pt['mrn']}\n"
            f"अभिभावक: {pt['deva_rel']}\n"
            f"पता: ग्राम {pt['village']}, जिला {pt['district']} - {pt['pin']}\n"
            f"मरीज़ को ४ दिन से बुखार है। {doc['hi']} द्वारा जांच।\n"
            f"सलाह: {pt['drug'][0]} ({pt['drug'][1]})। संपर्क: {_deva_digits(pt['phone'])}\n"
            f"रेफरल नोट: {can['name']} (मोबाइल: {can['phone']})\n"
        )
        rel_hi = f"notes/hindi_opd_{k:02d}.txt"
        sha = _write_file(inp / rel_hi, hi_txt.encode("utf-8"))
        add_truth(
            rel_hi,
            sha,
            {
                "doc": {
                    "patient": pt["pid"], "link": "required",
                    "entities": [
                        {"type": "PATIENT_NAME", "script": "deva", "channel": "content", "surface": pt["deva_name"]},
                        {"type": "MRN", "script": "latin", "channel": "content", "surface": pt["mrn"]},
                        {"type": "RELATIVE_NAME", "script": "deva", "channel": "content", "surface": pt["deva_rel"]},
                        {"type": "VILLAGE", "script": "latin", "channel": "content", "surface": pt["village"]},
                        {"type": "PIN", "script": "latin", "channel": "content", "surface": pt["pin"]},
                        {"type": "PHONE", "script": "deva", "channel": "content", "surface": pt["phone"]},
                        {"type": "CANARY", "script": "latin", "channel": "content", "surface": can["name"]},
                        {"type": "CANARY", "script": "latin", "channel": "content", "surface": can["phone"]},
                    ],
                    "preserve": [{"text": doc["hi"]}, {"text": pt["drug"][0]}],
                }
            },
        )

    for k in range(20):
        pt = patients[80 + k]
        can = canaries[(20 + k) % 30]
        doc = doctors[k % len(doctors)]
        fn_hing = f"notes/hinglish_{pt['fn']}_{pt['ln']}_{k:02d}.txt"
        hing_txt = (
            f"Subah 08:30 - {pt['fn']} ji ka BP 136/86, pulse 88. Bukhar 100.4 F hai.\n"
            f"UHID: {pt['mrn']} | Mob: {pt['phone']}\n"
            f"Patient {pt['name']} ({pt['rel_type']} {pt['rel']}) ko C/O cough and chest pain x 3 days.\n"
            f"{doc['en']} ne dekha, K/C/O {pt['cond'][1]}. {pt['drug'][0]} {pt['drug'][1]} x 5 days diya.\n"
            f"Attendant {can['name']} ({can['mrn']}) ne consent sign kiya.\n"
        )
        sha = _write_file(inp / fn_hing, hing_txt.encode("utf-8"))
        add_truth(
            fn_hing,
            sha,
            {
                "doc": {
                    "patient": pt["pid"], "link": "required",
                    "entities": [
                        {"type": "PATIENT_NAME", "script": "latin", "channel": "filename", "surface": f"{pt['fn']}_{pt['ln']}"},
                        {"type": "PATIENT_NAME", "script": "hinglish", "channel": "content", "surface": f"{pt['fn']} ji"},
                        {"type": "PATIENT_NAME", "script": "latin", "channel": "content", "surface": pt["name"]},
                        {"type": "MRN", "script": "latin", "channel": "content", "surface": pt["mrn"]},
                        {"type": "PHONE", "script": "latin", "channel": "content", "surface": pt["phone"]},
                        {"type": "RELATIVE_NAME", "script": "latin", "channel": "content", "surface": pt["rel"]},
                        {"type": "CANARY", "script": "latin", "channel": "content", "surface": can["name"]},
                        {"type": "CANARY", "script": "latin", "channel": "content", "surface": can["mrn"]},
                    ],
                    "preserve": [
                        {"text": doc["en"]},
                        {"text": "BP 136/86"},
                        {"text": "C/O cough"},
                        {"text": pt["drug"][0]},
                    ],
                }
            },
        )

    # Plant remaining canaries (20..29) in JSON keys, CSV headers, _meta paths, and base64 attachments!
    trap_json = {
        "_meta": {"source_file": f"D:\\Exports\\{canaries[20]['name'].replace(' ', '_')}\\{canaries[20]['mrn']}.json"},
        "by_beneficiary": {canaries[c]["name"]: canaries[c]["mrn"] for c in range(21, 26)},
        "attachments": [
            {
                "doc_id": f"ATT-{c}",
                "data": base64.b64encode(
                    f"Signed consent of {canaries[c]['name']}, ID {canaries[c]['mrn']}, Ph {canaries[c]['phone']}.".encode("utf-8")
                ).decode("ascii"),
            }
            for c in range(26, 30)
        ],
    }
    sha = _write_file(inp / "traps/canaries_hidden.json", json.dumps(trap_json, indent=2).encode("utf-8"))
    c_ents = [{"type": "CANARY", "script": "latin", "channel": "metadata", "surface": canaries[20]["name"]}]
    for c in range(21, 26):
        c_ents.append({"type": "CANARY", "script": "latin", "channel": "key", "surface": canaries[c]["name"]})
        c_ents.append({"type": "CANARY", "script": "latin", "channel": "content", "surface": canaries[c]["mrn"]})
    for c in range(26, 30):
        c_ents.append({"type": "CANARY", "script": "latin", "channel": "base64", "surface": canaries[c]["name"]})
        c_ents.append({"type": "CANARY", "script": "latin", "channel": "base64", "surface": canaries[c]["phone"]})
    add_truth("traps/canaries_hidden.json", sha, {"doc": {"entities": c_ents, "preserve": []}})

    # Disguised JSON behind .csv + Clean static protocol + 6 Corrupted/Unsupported quarantine traps
    disg_pt = patients[120]
    sha = _write_file(inp / "traps/disguised_export.csv", json.dumps({"patient": disg_pt["name"], "aadhaar": int(disg_pt["aadhaar"]), "mrn": disg_pt["mrn"]}).encode("utf-8"))
    add_truth("traps/disguised_export.csv", sha, {"doc": {"patient": disg_pt["pid"], "link": "required", "entities": [{"type": "PATIENT_NAME", "script": "latin", "channel": "content", "surface": disg_pt["name"]}, {"type": "AADHAAR", "script": "latin", "channel": "content", "surface": disg_pt["aadhaar"]}, {"type": "MRN", "script": "latin", "channel": "content", "surface": disg_pt["mrn"]}], "preserve": []}})

    clean_proto = "CLINICAL SEPSIS GUIDELINES\n1. Measure Lactate, blood cultures, broad-spectrum Inj Ceftriaxone 1 g IV.\n2. Target MAP >= 65 mmHg, SpO2 >= 94%.\n3. Reviewed by Dr. Kavita Narayanan.\n"
    sha = _write_file(inp / "clean/sepsis_protocol.txt", clean_proto.encode("utf-8"))
    add_truth("clean/sepsis_protocol.txt", sha, {"doc": {"entities": [], "preserve": [{"text": "Ceftriaxone 1 g"}, {"text": "Dr. Kavita Narayanan"}]}})

    for rel_q, payload in (
        ("misc/corrupt_bytes.txt", b"Patient: " + canaries[0]["name"].encode() + b" \xff\xfe\xc3( phone 9811223344\n"),
        ("misc/truncated.json", b'{"visits": [{"name": "Cut Off"'),
        ("misc/empty.csv", b""),
        ("misc/hl7_unseen.xml", b"<ClinicalDocument><patient>Hidden</patient></ClinicalDocument>"),
        ("misc/scan_unseen.pdf", b"%PDF-1.4\n% " + canaries[1]["name"].encode() + b"\n"),
        ("misc/photo_unseen.jpg", b"\xff\xd8\xff\xe1EXIF " + canaries[2]["name"].encode()),
    ):
        sha = _write_file(inp / rel_q, payload)
        add_truth(rel_q, sha, {"doc": {"entities": [], "preserve": []}}, expect="either")

    truth = {
        "files": truth_files,
        "canaries": [c["name"] for c in canaries],
        "meta": {"n_patients": n_patients, "n_canaries": len(canaries), "n_hospitals": len(hospitals)},
    }
    (out_dir / "truth.json").write_text(json.dumps(truth, ensure_ascii=False, indent=2), encoding="utf-8")
    return truth


def re_sub_alnum(s: str) -> str:
    return "".join(ch.lower() for ch in s if ch.isalnum())


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate large 300+ patient unseen stress dataset")
    ap.add_argument("--output", default=".unseen_data", help="Destination folder (gitignored)")
    ap.add_argument("--seed", type=int, default=2028)
    ap.add_argument("--patients", type=int, default=320)
    a = ap.parse_args()
    truth = generate_unseen_dataset(Path(a.output), seed=a.seed, n_patients=a.patients)
    print(f"Generated unseen dataset at {a.output}: {a.patients} patients, {len(truth['files'])} files, {len(truth['canaries'])} canaries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

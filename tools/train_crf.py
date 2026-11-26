"""Step G offline trainer: generates synthetic Faker en_IN/hi_IN clinical notes (disjoint by patient identity), trains sklearn-crfsuite, and saves compact model."""
from __future__ import annotations

import random
import re
import sys
from pathlib import Path

import sklearn_crfsuite
from faker import Faker

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from raksha.detect.crf_model import MODEL_PATH, sent_to_features, tokenize_with_spans

# Rare/canary-style synthetic names disjoint from Faker pools and unseen in dev data
_CANARY_POOLS = [
    ("Zubin", "Khosravi"), ("Tashi", "Dorjee"), ("Maitreyi", "Senapati"), ("Qvornak", "Zelenski"),
    ("Vaelor", "Trenzok"), ("Kripan", "Narasimhan"), ("Srinivas", "Kulkarni"), ("Prabhakar", "Deshmukh"),
    ("Niharika", "Chattopadhyay"), ("Subramanian", "Iyer"), ("Devika", "Panicker"), ("Harshvardhan", "Rathore"),
]

_DRUGS = ["Paracetamol 650 mg", "Dolo 650", "Metformin 500 mg", "Amlodipine 5 mg", "Ceftriaxone 1 g", "Pantoprazole 40 mg", "Azithromycin 500 mg"]
_DISEASES = ["Typhoid fever", "Dengue fever", "Addison's disease", "Parkinson's disease", "Bell's palsy", "Crohn's disease", "Essential hypertension"]
_COMPLAINTS = ["fever for 3 days", "pain abdomen", "cough and cold", "headache x 2 days", "generalised weakness"]


def _make_patients(rng: random.Random, n_patients: int, use_canary: bool = False) -> list[dict]:
    fake_en = Faker("en_IN")
    fake_hi = Faker("hi_IN")
    Faker.seed(42 if not use_canary else 999)
    patients = []
    for i in range(n_patients):
        if use_canary:
            fn, ln = _CANARY_POOLS[i % len(_CANARY_POOLS)]
            rel_fn, rel_ln = _CANARY_POOLS[(i + 5) % len(_CANARY_POOLS)]
            deva_name = f"{fake_hi.first_name()} {fake_hi.last_name()}"
            deva_rel = f"{fake_hi.first_name()} {fake_hi.last_name()}"
        else:
            fn = fake_en.first_name()
            ln = fake_en.last_name()
            rel_fn = fake_en.first_name()
            rel_ln = ln if rng.random() < 0.7 else fake_en.last_name()
            deva_name = f"{fake_hi.first_name()} {fake_hi.last_name()}"
            deva_rel = f"{fake_hi.first_name()} {fake_hi.last_name()}"
        patients.append({
            "pid": i,
            "name": f"{fn} {ln}",
            "first": fn,
            "last": ln,
            "rel": f"{rel_fn} {rel_ln}",
            "deva_name": deva_name,
            "deva_rel": deva_rel,
            "doc": f"Dr. {fake_en.first_name()} {fake_en.last_name()}",
            "doc_hi": f"डॉ. {fake_hi.first_name()} {fake_hi.last_name()}",
        })
    return patients


def _annotate_text(text: str, targets: list[tuple[str, str]]) -> list[tuple[list[dict], list[str]]]:
    """Align (entity_surface, label) spans onto each tokenized line's BIO tags."""
    sents = tokenize_with_spans(text)
    if not sents:
        return []
    char_labels: list[tuple[int, int, str]] = []
    for surface, lbl in targets:
        if not surface:
            continue
        for m in re.finditer(re.escape(surface), text):
            char_labels.append((m.start(), m.end(), lbl))

    out: list[tuple[list[dict], list[str]]] = []
    for sent in sents:
        tags = ["O"] * len(sent)
        for idx, (_tok, ts, te) in enumerate(sent):
            for cs, ce, lbl in char_labels:
                if ts >= cs and te <= ce:
                    prev_same = idx > 0 and sent[idx - 1][1] >= cs and sent[idx - 1][2] <= ce and tags[idx - 1].endswith(lbl)
                    tags[idx] = f"I-{lbl}" if prev_same else f"B-{lbl}"
                    break
        out.append((sent_to_features(sent), tags))
    return out


def build_synthetic_dataset(patients: list[dict], rng: random.Random, samples_per_patient: int = 8):
    X, y = [], []
    for pt in patients:
        name, first, rel = pt["name"], pt["first"], pt["rel"]
        dn, dr = pt["deva_name"], pt["deva_rel"]
        doc, doc_hi = pt["doc"], pt["doc_hi"]
        drug = rng.choice(_DRUGS)
        dis = rng.choice(_DISEASES)
        comp = rng.choice(_COMPLAINTS)

        templates: list[tuple[str, list[tuple[str, str]]]] = [
            (f"Patient Name: {name}   Age/Sex: {rng.randint(18, 85)} Yrs/M", [(name, "PATIENT_NAME")]),
            (f"Name: {name}   Reg No: HOSP-{rng.randint(10000,99999)}", [(name, "PATIENT_NAME")]),
            (f"Pt. {name} presented with C/O {comp}.", [(name, "PATIENT_NAME")]),
            (f"Pt. {name} (MRN: HOSP-{rng.randint(10000,99999)}) referred for {comp}.", [(name, "PATIENT_NAME")]),
            (f"Beneficiary: {name} | Caretaker: {rel}", [(name, "PATIENT_NAME"), (rel, "RELATIVE_NAME")]),
            (f"Client Name: {name}, S/o {rel}", [(name, "PATIENT_NAME"), (rel, "RELATIVE_NAME")]),
            (f"S/o {rel}, Attendant: {rel}, Mob: 9876543210", [(rel, "RELATIVE_NAME")]),
            (f"{first}, W/o {rel}, discharged in stable condition.", [(first, "PATIENT_NAME"), (rel, "RELATIVE_NAME")]),
            (f"08:30 - {first} sahab ka BP 130/80, pulse 88. {drug} diya.", [(first, "PATIENT_NAME")]),
            (f"22:00 - {first} ji ki condition ab stable hai.", [(first, "PATIENT_NAME")]),
            (f"14:00 - {name} - K/C/O {dis}. Attendant (S/o {rel}) informed.", [(name, "PATIENT_NAME"), (rel, "RELATIVE_NAME")]),
            (f"Report of {first} behenji 48 Y / F | Haemoglobin 12.1 g/dL", [(f"{first} behenji", "PATIENT_NAME")]),
            (f"[11/09/26, 8:13 pm] {rel} Bhaiya: <attachment: report.pdf>", [(f"{rel} Bhaiya", "RELATIVE_NAME")]),
            (f"Consent witnessed by {name}, phone 8010993510.", [(name, "PATIENT_NAME")]),
            (f"Sample received from {first}'s attendant (S/o {rel}).", [(first, "PATIENT_NAME"), (rel, "RELATIVE_NAME")]),
            (f"18:40 - {name} (attendant, bed 12), mob 9123456780", [(name, "RELATIVE_NAME")]),
            (f"AMB-410,North Town,South City,called by {name} 98111-22334", [(name, "PATIENT_NAME")]),
            (f"Address (attendant): C/O {rel}, Ward 4", [(rel, "RELATIVE_NAME")]),
            (f"मरीज़ का नाम: {dn}   आयु: 45 वर्ष", [(dn, "PATIENT_NAME")]),
            (f"नाम: {dn}   पिता का नाम: {dr}", [(dn, "PATIENT_NAME"), (dr, "RELATIVE_NAME")]),
            (f"{dn} को 5 दिन से बुखार है।", [(dn, "PATIENT_NAME")]),
            # Negative / hard clinical trap lines (0 entities!)
            (f"Consultant: {doc} (MD Medicine). Reviewed by {doc}.", []),
            (f"OPD: C/O {comp}. K/C/O {dis}.", []),
            (f"1. Tab {drug} 1-0-1 x 5 days. Platelet Count 240124 /uL.", []),
            (f"Referred by health worker. {doc_hi} took evening round.", []),
            (f"Ward 5 (Medicine) / Bed 12. Diagnosis: {dis}.", []),
            ("Metro Multispeciality Hospital, North Zone\nDISCHARGE SUMMARY", []),
            ("Apex Charitable Trust, West Zone\nडिस्चार्ज सारांश", []),
            ("भर्ती की तारीख: 26.10.2026\nछुट्टी की तारीख: 30.10.2026", []),
            ("निदान: फेफड़ों की टीबी (A15.0); निमोनिया (J18.9); मलेरिया (B54);", []),
            ("Procedure: Laparoscopic cholecystectomy done on day 2.", []),
            ("Presenting complaints: C/O fever for 3 days, generalised weakness.", []),
            ("Pathology Labs Diagnostics - Diagnostic report", []),
        ]
        chosen = rng.sample(templates, min(samples_per_patient, len(templates)))
        for txt, tgts in chosen:
            for feats, tags in _annotate_text(txt, tgts):
                if feats:
                    X.append(feats)
                    y.append(tags)
    return X, y


def evaluate_f1(y_true: list[list[str]], y_pred: list[list[str]]) -> tuple[float, float, float]:
    tp = fp = fn = 0
    for ts, ps in zip(y_true, y_pred):
        for t, p in zip(ts, ps):
            if p != "O" and p == t:
                tp += 1
            elif p != "O" and p != t:
                fp += 1
            elif t != "O" and p != t:
                fn += 1
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return prec, rec, f1


def main() -> int:
    rng = random.Random(2026)
    all_patients = _make_patients(rng, 500, use_canary=False)
    # Disjoint split by patient identity: 400 train / 100 val + 12 unseen canary identities
    train_pts = all_patients[:400]
    val_pts = all_patients[400:]
    canary_pts = _make_patients(rng, 24, use_canary=True)

    X_train, y_train = build_synthetic_dataset(train_pts, rng, samples_per_patient=14)
    X_val, y_val = build_synthetic_dataset(val_pts, rng, samples_per_patient=14)
    X_can, y_can = build_synthetic_dataset(canary_pts, rng, samples_per_patient=14)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    crf = sklearn_crfsuite.CRF(
        algorithm="lbfgs",
        c1=0.25,
        c2=0.1,
        max_iterations=100,
        all_possible_transitions=True,
        model_filename=str(MODEL_PATH),
    )
    crf.fit(X_train, y_train)

    y_val_pred = crf.predict(X_val)
    vp, vr, vf1 = evaluate_f1(y_val, y_val_pred)
    y_can_pred = crf.predict(X_can)
    cp, cr, cf1 = evaluate_f1(y_can, y_can_pred)

    size_kb = MODEL_PATH.stat().st_size / 1024
    print(f"Trained CRF saved to {MODEL_PATH} ({size_kb:.1f} KB)")
    print(f"Disjoint-patient validation: P={vp:.3f} R={vr:.3f} F1={vf1:.3f}")
    print(f"Unseen canary-context test:  P={cp:.3f} R={cr:.3f} F1={cf1:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

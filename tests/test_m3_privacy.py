"""Tests for Milestone M3: keyed date shifts, weekdays, truthful age bands, risk escalation, and unlinkability."""
from __future__ import annotations

import csv
import datetime as dt
import json
from pathlib import Path

from raksha.config import derive_subkey
from raksha.pipeline import run_pipeline
from raksha.privacy.dates import infer_month_first, keyed_shift_days, shift_date_str, shift_dates_and_weekdays_in_text
from raksha.privacy.risk import PolicySettings, PrivacyPlan
from raksha.privacy.transform import format_truthful_age_band, generalize_text


def test_keyed_date_shift_magnitude_and_secret_sensitivity() -> None:
    k1 = derive_subkey(b"secret-alpha", b"date_shift")
    k2 = derive_subkey(b"secret-beta", b"date_shift")
    shifts = [keyed_shift_days(f"PSN-TEST{i}", k1, max_days=180, min_days=15) for i in range(50)]
    for s in shifts:
        assert 15 <= abs(s) <= 180
    # Different secrets must produce different offsets across a cohort
    shifts_b = [keyed_shift_days(f"PSN-TEST{i}", k2, max_days=180, min_days=15) for i in range(50)]
    assert shifts != shifts_b


def test_date_format_interval_and_weekday_preservation() -> None:
    # Check interval + weekday recomputation across ISO, DMY, MDY, and WhatsApp timestamps
    txt = "Monday, 10/05/2026: Admission\n15/05/2026 (Friday): Discharge\n[15/05/26, 10:00 am] note"
    shifted_txt, changed = shift_dates_and_weekdays_in_text(txt, 25, month_first=False)
    assert changed is True
    # 10/05/2026 + 25 days = 04/06/2026 (Thursday), 15/05/2026 + 25 days = 09/06/2026 (Tuesday)
    assert "Thursday, 04/06/2026: Admission" in shifted_txt
    assert "09/06/2026 (Tuesday): Discharge" in shifted_txt
    assert "[09/06/26, 10:00 am] note" in shifted_txt

    # Original Monday/Friday must NOT remain next to shifted dates
    assert "Monday" not in shifted_txt
    assert "Friday" not in shifted_txt

    # MDY preservation & inference
    assert infer_month_first(["04/20/1979", "09/26/2026"], "dob") is True
    assert infer_month_first(["26/09/2026", "10/08/2026"], "date") is False
    out_mdy, ok = shift_date_str("03/23/2026", 10, month_first=True)
    assert ok and out_mdy == "04/02/2026"
    out_iso, ok2 = shift_date_str("2026-04-15T10:45:00+05:30", 20)
    assert ok2 and out_iso == "2026-05-05T10:45:00+05:30"


def test_truthful_age_bands_and_clinical_duration_preservation() -> None:
    for width in (5, 10, 20):
        for age in (1, 19, 20, 44, 55, 79):
            band = format_truthful_age_band(age, width)
            lo, hi = (int(x) for x in band.split("-"))
            assert lo <= age <= hi
            assert (hi - lo + 1) == width
    assert format_truthful_age_band(95, 10, top_cap=80) == "80+"
    assert format_truthful_age_band(95, 0) == "[AGE]"

    plan = PrivacyPlan(
        profile_key="PSN-TEST",
        shift_days=30,
        age_band_width=10,
        top_age_cap=80,
        keep_district=False,
        keep_tehsil=False,
        keep_gender=True,
        keep_fhir_birthdate=False,
    )
    note = (
        "Patient Name: [PATIENT_NAME]  Age/Sex: 54 Yrs/F\n"
        "Address: [HOUSE], Tehsil Bilgram, Distt. Hardoi - [PIN]\n"
        "History: H/O TB 8 yrs back, cough x 12 yrs. Tab Paracetamol 500 mg BD x 5 days. Ref: 70-140 mg/dL.\n"
    )
    out, changed = generalize_text(note, plan)
    assert changed is True
    assert "Age/Sex: 50-59 Yrs/F" in out
    assert "Tehsil [TEHSIL]" in out
    assert "Distt. [DISTRICT]" in out
    assert "8 yrs back" in out
    assert "12 yrs" in out
    assert "Paracetamol 500 mg BD x 5 days" in out
    assert "70-140 mg/dL" in out


def test_end_to_end_m3_intervals_cross_release_and_clean_preservation(tmp_path: Path, monkeypatch) -> None:
    inp = tmp_path / "input"
    (inp / "clean").mkdir(parents=True)
    (inp / "clean" / "protocol.txt").write_text("Paracetamol 650 mg TID x 5 days. Normal Hb 12.0-16.0 g/dL.\n", encoding="utf-8")

    # Patient with 2 linked files (billing CSV + FHIR Bundle + TXT note) + rare condition + old age
    rows = [
        ["patient_name", "uhid", "age", "gender", "district", "tehsil", "admission_date", "discharge_date", "procedure"],
        ["Kavita Singh", "HOSP/26/1001", "88", "F", "Sitapur", "Biswan", "10/04/2026", "16/04/2026", "Deep brain stimulation"],
    ]
    with (inp / "claims.csv").open("w", encoding="utf-8", newline="") as f:
        csv.writer(f, lineterminator="\n").writerows(rows)

    (inp / "ds.txt").write_text(
        "Name: Kavita Singh\nUHID: HOSP/26/1001\nAge/Sex: 88 Yrs/F\n"
        "Address: House 12, Vill. Rampur, Tehsil Biswan, Distt. Sitapur - 261001\n"
        "Date of Admission: 10/04/2026\nDate of Discharge: 16/04/2026\n"
        "Diagnosis: Deep brain stimulation. Tab Metformin 500 mg BD.\n",
        encoding="utf-8",
    )

    fhir_b = {
        "resourceType": "Bundle",
        "type": "document",
        "timestamp": "2026-04-16T09:00:00+05:30",
        "entry": [
            {
                "resource": {
                    "resourceType": "Patient",
                    "id": "p1",
                    "identifier": [{"value": "HOSP/26/1001"}],
                    "name": [{"text": "Kavita Singh"}],
                    "gender": "female",
                    "birthDate": "1938-05-01",
                }
            },
            {
                "resource": {
                    "resourceType": "Encounter",
                    "id": "e1",
                    "status": "finished",
                    "period": {"start": "2026-04-10T09:00:00+05:30", "end": "2026-04-16T14:00:00+05:30"},
                }
            },
        ],
    }
    (inp / "fhir.json").write_text(json.dumps(fhir_b), encoding="utf-8")

    out_a = tmp_path / "out_a"
    out_b = tmp_path / "out_b"
    monkeypatch.setenv("RAKSHA_SECRET", "m3-secret-one-abcdef")
    assert run_pipeline(inp, out_a) == 0
    monkeypatch.setenv("RAKSHA_SECRET", "m3-secret-two-uvwxyz")
    assert run_pipeline(inp, out_b) == 0

    # clean/ untouched
    assert (out_a / "sanitized" / "clean" / "protocol.txt").read_text(encoding="utf-8") == (inp / "clean" / "protocol.txt").read_text(encoding="utf-8")

    # Cross-release unlinkability: different PSN + different shifted dates
    ds_a = (out_a / "sanitized" / "ds.txt").read_text(encoding="utf-8")
    ds_b = (out_b / "sanitized" / "ds.txt").read_text(encoding="utf-8")
    assert "Kavita" not in ds_a and "Biswan" not in ds_a
    assert "Metformin 500 mg BD" in ds_a
    assert ds_a != ds_b

    # Exact 6-day LoS preserved across shifted dates in FHIR Bundle & claims CSV
    b_out = json.loads((out_a / "sanitized" / "fhir.json").read_text(encoding="utf-8"))
    pt_res = b_out["entry"][0]["resource"]
    enc_res = b_out["entry"][1]["resource"]
    assert "birthDate" not in pt_res  # omitted when generalized/risky to keep R4 valid & block birth_year
    d_start = dt.date.fromisoformat(enc_res["period"]["start"][:10])
    d_end = dt.date.fromisoformat(enc_res["period"]["end"][:10])
    assert (d_end - d_start).days == 6
    assert d_start != dt.date(2026, 4, 10)  # actually shifted!

    # Same shifted dates inside claims.csv and ds.txt
    csv_out = (out_a / "sanitized" / "claims.csv").read_text(encoding="utf-8")
    assert d_start.strftime("%d/%m/%Y") in csv_out
    assert d_start.strftime("%d/%m/%Y") in ds_a

    # Report counts match records.jsonl
    rpt = json.loads((out_a / "report.json").read_text(encoding="utf-8"))
    assert rpt["counts"]["records_generalized"] == 3
    assert rpt["counts"]["records_suppressed"] == 0

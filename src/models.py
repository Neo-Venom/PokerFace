"""In-memory RAM-only data models for Project RAKSHA.

None of these dataclasses are serialized raw to logs, errors, or temp files.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Span:
    start: int
    end: int
    kind: str  # PATIENT_NAME, RELATIVE_NAME, PHONE, AADHAAR, ABHA_NUMBER, MRN, PAN, DOB, HOUSE, LOCALITY, LANDMARK, VILLAGE, PIN, EMAIL, INSURANCE_ID, CANARY, PROTECTED_*
    text: str
    channel: str = "content"
    script: str = "latin"
    confidence: float = 1.0
    protected: bool = False


@dataclass(slots=True)
class PatientRecord:
    record_uid: str
    rel_input_path: str
    rel_output_path: str
    file_type: str  # csv | tsv | json | fhir | text | xml
    loc: str
    raw_payload: Any = None
    names_raw: list[str] = field(default_factory=list)
    guardian_names_raw: list[str] = field(default_factory=list)
    guardian_relation: str | None = None
    gov_ids: dict[str, list[str]] = field(default_factory=dict)
    uhids: list[str] = field(default_factory=list)
    phones: list[str] = field(default_factory=list)
    emails: list[str] = field(default_factory=list)
    gender: str | None = None  # M | F | O | None
    dob: dt.date | None = None
    reported_ages: list[int] = field(default_factory=list)
    reference_date: dt.date | None = None
    pin_code: str | None = None
    village: str | None = None
    tehsil: str | None = None
    district: str | None = None
    state: str | None = None
    hospital: str | None = None
    address_raw: str | None = None
    clinical_dates: list[dt.date] = field(default_factory=list)
    diagnoses: list[str] = field(default_factory=list)
    procedures: list[str] = field(default_factory=list)
    cluster_id: str | None = None
    pseudonym: str | None = None
    date_shift_days: int = 0
    age_band: str | None = None
    geo_generalized: str | None = None
    was_generalized: bool = False
    was_suppressed: bool = False


@dataclass(slots=True)
class FileEntry:
    input_sha256: str
    rel_input_path: str
    rel_output_path: str | None
    status: str  # sanitized | quarantined | suppressed
    reason: str | None = None
    file_type: str = "unknown"
    delimiter: str | None = None
    text_decoded: str | None = None
    parsed_data: Any = None
    column_roles: dict[Any, str] = field(default_factory=dict)
    date_order_hint: str = "dmy"
    has_pii_filename: bool = False
    is_clean_doc: bool = False
    records: list[PatientRecord] = field(default_factory=list)
    key_roles: dict[str, str] = field(default_factory=dict)   # JSON key -> role inferred from values
    multi_patient: bool = False                                # PII in >=2 headers/keys: a multi-patient table

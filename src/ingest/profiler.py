"""Dual-signal schema-agnostic column/key profiler & date-order inference."""
from __future__ import annotations

import re
from collections import Counter

from rapidfuzz import fuzz

_DIGIT_MAP = str.maketrans("०१२३४५६७८९", "0123456789")

_ROLE_SYNONYMS: dict[str, tuple[str, ...]] = {
    "PATIENT_NAME": (
        "patient name", "patient's name", "patients name", "naam", "name",
        "pt name", "pt_name", "pt fullname", "mrz naam", "mrz nm", "first name", "last name",
        "firstname", "lastname", "first", "last", "local", "subjectname",
        "carerecipient", "member name", "beneficiary name", "नाम", "मरीज का नाम", "मरीज़ का नाम", "रोगी नाम",
    ),
    "RELATIVE_NAME": (
        "relative name", "guardian", "guardian full", "pita pati ka naam", "pita ka naam",
        "pati ka naam", "father name", "husband name", "caregiver", "caregiver name", "relation name",
        "attendant", "पिता का नाम", "पति का नाम", "अभिभावक",
    ),
    "RELATION": ("relation", "relationship", "रिश्ता"),
    "PHONE": (
        "mobile", "mob", "mob no", "mobile no", "contact", "contact#", "contact num", "phone",
        "callback", "telecom", "sampark", "sampark num", "tel", "cell phone", "फोन", "फोन नंबर", "मोबाइल", "दूरभाष",
    ),
    "EMAIL": ("email", "e-mail", "mail"),
    "AADHAAR": (
        "aadhaar", "aadhar", "aadhaar no", "aadhar no", "uid", "uid 12", "uid num",
        "national id", "national id num", "govt id", "govtid a", "आधार", "आधार संख्या",
    ),
    "ABHA": ("abha", "abha no", "abha number", "abha address", "health id"),
    "PAN": ("pan", "payer pan", "pan no"),
    "INSURANCE_ID": ("pmjay", "pmjay id", "policy", "policy no", "insurance", "tpa"),
    "MRN": (
        "uhid", "reg no", "reg code", "patientid", "patient id", "patient ref",
        "mrn", "mrn link", "ipd", "ipd no", "opd no", "hospital uhid", "hosp id", "hosp ref",
        "pt code", "card id", "subject ref", "care id",
        "bill no", "sample id", "पंजीकरण", "पंजीकरण सं", "आईपीडी",
    ),
    "DOB": ("dob", "birth date", "birthdate", "birth dt", "janm tithi", "janm dinank", "जन्म तिथि"),
    "AGE": ("age", "umar", "umr", "age sex", "आयु", "उम्र"),
    "GENDER": ("gender", "sex", "ling", "लिंग"),
    "ADDRESS": ("address", "pata", "pata full", "street", "line", "address note", "res addr", "पता", "निवास", "निवास पता"),
    "VILLAGE": (
        "village", "village mohalla", "gram mohalla", "gaon", "gaon name", "gram", "city", "kasba", "rural unit",
        "ग्राम", "ग्राम मोहल्ला", "गाँव",
    ),
    "TEHSIL": ("tehsil", "tahsil", "taluka", "block", "तहसील"),
    "DISTRICT": ("district", "dist", "distt", "jila", "जिला"),
    "PIN": ("pin", "pin code", "pincode", "postal", "postal 6", "postalcode", "area pin", "zip", "पिन", "पिन कोड"),
    "DATE": (
        "date", "reg date", "registered on", "visit date", "admission date",
        "discharge date", "collected on", "reported on", "collected", "तारीख",
    ),
    "DOCTOR": ("doctor", "ref doctor", "referred by", "consultant", "author", "reviewed by"),
    "DIAGNOSIS": ("diagnosis", "icd", "condition", "procedure", "complaints", "history", "निदान"),
    "MEDICATION": ("drug", "rx", "medicine", "strength", "frequency", "duration", "package"),
    "LAB": ("test", "result", "value", "unit", "ref range", "range", "loinc", "vitals", "bp"),
    "NOTES": ("notes", "remarks", "comment", "comments"),
    "METADATA_PATH": ("source path", "export path", "exported by", "file path"),
}

_PHONE_SHAPE = re.compile(r"^(?:\+?91[\s\-]?|0091[\s\-]?|0)?[6-9]\d{4}[\s\-.]?\d{5}$")
_AADHAAR_SHAPE = re.compile(r"^(?:\d{4}[\s\-]?\d{4}[\s\-]?\d{4}|[X*]{4}[\s\-]?[X*]{4}[\s\-]?\d{4})$", re.I)
_ABHA_SHAPE = re.compile(r"^(?:\d{2}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}|[\w.]{3,}@(?:abdm|sbx))$", re.I)
_PAN_SHAPE = re.compile(r"^[A-Z]{5}\d{4}[A-Z]$")
_PIN_SHAPE = re.compile(r"^[1-8]\d{5}$")
_AGE_SHAPE = re.compile(r"^(?:\d{1,3}(?:\s*(?:yrs?|years?|वर्ष|y))?(?:\s*/\s*[MF])?)$", re.I)
_DATE_SLASH = re.compile(r"^(\d{1,2})[/-](\d{1,2})[/-]((?:19|20)\d{2})$")
_DATE_ISO = re.compile(r"^(?:19|20)\d{2}-\d{2}-\d{2}")
_EMAIL_SHAPE = re.compile(r"^[\w.+-]+@[\w-]+\.[\w.]+$")


def _norm_header(h: str) -> str:
    s = re.sub(r"[_\-/()\[\].#]+", " ", str(h).strip().lower())
    return " ".join(s.split())


def infer_role_from_header(header: str) -> tuple[str | None, float]:
    nh = _norm_header(header)
    if not nh:
        return None, 0.0
    words = set(nh.split())
    best_role: str | None = None
    best_score = 0.0
    for role, syns in _ROLE_SYNONYMS.items():
        for syn in syns:
            if nh == syn:
                return role, 100.0
            sc = fuzz.token_sort_ratio(nh, syn)
            if syn in words and len(syn) >= 3:
                sc = max(sc, 93.0 + min(5.0, len(syn) * 0.2))
            elif syn in nh and len(syn) >= 4:
                sc = max(sc, 90.0)
            if sc > best_score:
                best_score = sc
                best_role = role
    if best_score >= 82.0:
        return best_role, best_score
    return None, 0.0


def infer_role_from_values(values: list[str]) -> str | None:
    non_empty = [str(v).strip().translate(_DIGIT_MAP) for v in values if v is not None and str(v).strip()]
    if not non_empty:
        return None
    n = len(non_empty)

    def _rate(rx: re.Pattern[str]) -> float:
        return sum(1 for v in non_empty if rx.match(v)) / n

    if _rate(_EMAIL_SHAPE) >= 0.5:
        return "EMAIL"
    if _rate(_PAN_SHAPE) >= 0.5:
        return "PAN"
    if _rate(_ABHA_SHAPE) >= 0.5:
        return "ABHA"
    if _rate(_AADHAAR_SHAPE) >= 0.5:
        return "AADHAAR"
    if _rate(_PHONE_SHAPE) >= 0.55:
        return "PHONE"
    if _rate(_PIN_SHAPE) >= 0.6:
        return "PIN"
    if _rate(_DATE_ISO) >= 0.5 or _rate(_DATE_SLASH) >= 0.5:
        # Distinguish DOB columns from clinical visit/reg dates if median year <= 2019
        yrs = [int(m.group(1)) for v in non_empty if (m := re.search(r"\b((?:19|20)\d{2})\b", v))]
        if yrs and (sum(1 for y in yrs if y <= 2019) / len(yrs)) >= 0.5:
            return "DOB"
        return "DATE"
    if _rate(_AGE_SHAPE) >= 0.7:
        # Guard against generic integers > 115
        nums = [int(m.group(1)) for v in non_empty if (m := re.match(r"^(\d{1,3})\b", v))]
        if nums and all(0 <= x <= 115 for x in nums):
            return "AGE"
    return None


def infer_column_roles(headers: list[str], rows: list[list[str]]) -> dict[int, str]:
    roles: dict[int, str] = {}
    for idx, h in enumerate(headers):
        h_role, _ = infer_role_from_header(h)
        col_vals = [r[idx] for r in rows if idx < len(r)]
        v_role = infer_role_from_values(col_vals)
        # Value distribution wins when header is obscure/renamed, except DOB/DATE/LAB distinctions
        if h_role in ("LAB", "DIAGNOSIS", "MEDICATION", "DOCTOR"):
            roles[idx] = h_role
        elif v_role and (not h_role or h_role not in ("DOB", "DATE", "MRN", "PIN")):
            roles[idx] = v_role
        elif h_role:
            roles[idx] = h_role
        elif v_role:
            roles[idx] = v_role
        else:
            roles[idx] = "FREE_TEXT"
    return roles


def infer_date_order(values: list[str], header_hint: str = "") -> str:
    """Infer 'dmy' vs 'mdy' across a column or document's date tokens."""
    hl = header_hint.lower()
    if "mm/dd" in hl or "mm-dd" in hl:
        return "mdy"
    if "dd/mm" in hl or "dd-mm" in hl:
        return "dmy"
    first_gt_12 = 0
    second_gt_12 = 0
    for v in values:
        s = str(v).strip().translate(_DIGIT_MAP)
        for m in re.finditer(r"\b(\d{1,2})[./-](\d{1,2})[./-](?:19|20)\d{2}\b", s):
            a, b = int(m.group(1)), int(m.group(2))
            if a > 12 and b <= 12:
                first_gt_12 += 1
            elif b > 12 and a <= 12:
                second_gt_12 += 1
    if second_gt_12 > first_gt_12:
        return "mdy"
    return "dmy"

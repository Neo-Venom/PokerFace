"""Step A: Value-first structured field classifier, header/key sanitizer, and composite address scrubber."""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any

from raksha.detect.normalize import (
    DEVA_LETTER_CLASS,
    LATIN_LETTER_CLASS,
    LATIN_UPPER_CLASS,
    fold_latin_accents,
    is_deva_char,
    is_upper_or_deva_start,
)
from raksha.export.filenames import sanitize_base64_string

# Devanagari -> ASCII digit mapping (used locally before Step C offset-mapped normalization)
_DEVA_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
_DL = DEVA_LETTER_CLASS
_LU = LATIN_UPPER_CLASS
_LL = LATIN_LETTER_CLASS

# --------------------------------------------------------------------------- value matchers
_RE_PHONE = re.compile(r"^(?:\+?91[\s\-]?|0)?[6-9]\d{4}[\s\-]?\d{5}$|^(?:\+?91[\s\-]?|0)?[6-9]\d{2}[\s\-]?\d{3}[\s\-]?\d{4}$")
_RE_PHONE_MASKED = re.compile(r"^[6-9]\d[xX*]{4,6}\d{2,4}$")
_RE_AADHAAR = re.compile(r"^\d{4}[\s\-]?\d{4}[\s\-]?\d{4}$")
_RE_AADHAAR_MASKED = re.compile(r"^[xX*]{4}[\s\-]?[xX*]{4}[\s\-]?\d{4}$")
_RE_ABHA = re.compile(r"^\d{2}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}$")
_RE_ABHA_ADDR = re.compile(r"^[A-Za-z0-9._+-]+@(?:abdm|ndhm|sbx)$", re.I)
_RE_PAN = re.compile(r"^[A-Z]{5}\d{4}[A-Z]$")
_RE_EMAIL = re.compile(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$")
_RE_INSURANCE = re.compile(r"^(?:PMJAY[-/A-Z0-9]{4,}|POL/\d+/\d+|AB-PMJAY[-/A-Z0-9]+|CGHS[-/A-Z0-9]+|ESIS[-/A-Z0-9]+)$", re.I)
_RE_MRN = re.compile(
    r"^(?:"
    r"[A-Z]{2,6}/\d{2,4}/\d{3,}"
    r"|[A-Z]{2,6}-\d{4,}"
    r"|[A-Z]{2,6}\d{2,4}-\d{3,}"
    r"|[A-Z]{2,6}-[A-Z]{2,4}-\d{3,}"
    r"|[A-Z]{2,5}-\d{2,4}-\d{3,}"
    r"|[A-Z]{2,5}\d{5,}"
    r")$"
)
_RE_PIN = re.compile(r"^[1-8]\d{5}$")
_RE_DATE = re.compile(
    r"^(?:(\d{4})[-/](\d{1,2})[-/](\d{1,2})|(\d{1,2})[-/.](\d{1,2})[-/.](\d{4}))$"
)
_RE_REL_PREFIX = re.compile(r"^(?:[SDWC]/[Oo]|[SDWC]\s*/\s*[Oo])\s+(.+)$")
_RE_ICD_OR_CODE = re.compile(r"^[A-TV-Z]\d{2}(?:\.\d{1,2})?$")
_RE_CLINICAL_UNIT_VAL = re.compile(
    r"(?i)\b(?:\d+(?:\.\d+)?\s*(?:mg|mcg|µg|ug|ml|g/dl|mg/dl|mmol/l|miu/l|iu/l|/ul|/mm3|mmhg|bpm|°[fc])|1-0-1|0-0-1|1-1-1|1-0-0|bd|tid|qid)\b"
)
_RE_HOUSE_VAL = re.compile(
    r"(?i)^(?:h\.?\s*no\.?|house(?:\s+no\.?)?|makan(?:\s+no\.?)?|qtr\.?|quarter|flat(?:\s+no\.?)?|plot(?:\s+no\.?)?|door(?:\s+no\.?)?)[\s.\-#:/]*[A-Za-z0-9/-]*\d[A-Za-z0-9/-]*$"
)
_RE_LOCALITY_VAL = re.compile(
    rf"(?i)^(?:ward\s*(?:no\.?)?\s*\d+|sector\s*[-#]?\s*[A-Za-z0-9]+(?:\s+(?:colony|phase|extension|block))?|phase\s*[-#]?\s*\d+|mohalla\s+.+|[{_LU}][a-z0-9\u00DF-\u024F\s\-]{{2,35}}\s+(?:colony|nagar|vihar|enclave|puram|extension|basti|tola|layout))$"
)
_RE_LANDMARK_VAL = re.compile(r"(?i)^(?:near|opp\.?|opposite|behind|beside)\s+.+")
_SYNTH_CANARY_KEY_RE = re.compile(r"^[A-Z][a-z]{4,20}\d{1,2}[a-z]{0,4}$")

# Clinical/non-PII vocabulary that must NEVER make a column look "name-like"
_CLINICAL_OR_NON_NAME_WORDS = {
    "male", "female", "other", "m", "f", "approved", "pending", "settled", "rejected",
    "active", "final", "current", "completed", "finished", "routine", "medicine", "surgery",
    "pulmonology", "neurology", "cardiology", "pediatrics", "paediatrics", "gynaecology",
    "haemoglobin", "hemoglobin", "hba1c", "random blood sugar", "serum creatinine", "tsh",
    "platelet count", "total wbc count", "serum sodium", "tab", "cap", "inj", "inh", "sachet",
    "antibiotic", "essential", "chronic care", "amlodipine", "artesunate", "atorvastatin",
    "azithromycin", "ceftriaxone", "doxycycline", "glimepiride", "insulin glargine",
    "levothyroxine", "metformin", "ors", "pantoprazole", "paracetamol", "ramipril",
    "telmisartan", "typhoid fever", "dengue", "malaria", "anaemia", "hypertension",
    "diabetes", "asthma", "pneumonia", "appendicectomy", "caesarean section",
    "cataract surgery", "haemodialysis session", "inguinal hernia repair",
    "laparoscopic cholecystectomy", "billing query", "closed", "open",
    "पुरुष", "महिला", "अन्य",
}

_NON_PII_KEY_TOKENS = {
    "drug", "medicine", "medication", "strength", "dose", "dosage", "frequency", "duration",
    "form", "schedule", "category", "test", "loinc", "unit", "range", "ref_range",
    "reference_range", "result", "results", "value", "values", "diagnosis", "diagnoses",
    "complaints", "complaint", "history", "procedure", "package_code", "icd", "code",
    "department", "ward", "wards", "bed", "beds", "occupied", "cases", "case_count", "week",
    "status", "amount", "amount_inr", "doctor", "ref_doctor", "referred_by", "consultant",
    "practitioner", "author", "hospital", "facility", "system", "exported_by", "schema",
    "report_type", "lab", "resourcetype", "gender", "sex", "ling", "age", "umar",
    "age_sex", "district", "jila", "tehsil", "state", "country", "time", "visits", "rx",
    "vitals", "bp", "pulse", "temp", "temp_f", "spo2", "weight_kg", "trip", "from", "to",
    "bill_no", "sample_id", "visit_id", "intent", "clinicalstatus", "class",
}

_CLINICAL_DATE_KEYS = {
    "visit_date", "admission_date", "discharge_date", "collected", "collected_on",
    "reported_on", "registered_on", "reg_date", "registration_date", "generated_at",
    "exported_at", "generated", "date", "recordeddate", "authoredon", "performeddatetime",
    "lastupdated", "timestamp", "effectivedatetime", "start", "end",
}


def _norm_digits(s: str) -> str:
    return unicodedata.normalize("NFKC", s).translate(_DEVA_DIGITS).strip()


def _norm_key(k: str) -> str:
    k = unicodedata.normalize("NFKC", k).casefold().strip()
    k = re.sub(r"\([^)]*\)", "", k)  # strip "(MM/DD/YYYY)" etc.
    k = re.sub(rf"[^a-z0-9\u00DF-\u024F{_DL}]+", "_", k).strip("_")
    return k


def split_key_parts(k: str) -> tuple[str, ...]:
    """Split header/JSON key into lowercase morphological parts (handles camelCase, Unicode, delimiters)."""
    s = unicodedata.normalize("NFKC", k).strip()
    s = re.sub(r"\([^)]*\)", "", s)
    s = re.sub(r"([a-z0-9\u00DF-\u024F])([A-Z\u00C0-\u00DE])", r"\1_\2", s)
    s = fold_latin_accents(s).casefold()
    parts = [p for p in re.split(r"[^a-z0-9\u0900-\u097f]+", s) if p]
    return tuple(parts)


def _extract_year(s: str) -> int | None:
    m = _RE_DATE.match(_norm_digits(s))
    if not m:
        return None
    return int(m.group(1) or m.group(6))


_PROSE_OR_CLINICAL_STOPWORDS = _CLINICAL_OR_NON_NAME_WORDS | {
    "oxygen", "given", "en", "route", "for", "with", "and", "the", "by", "called",
    "from", "to", "fever", "pain", "days", "stable", "normal", "reviewed", "hospital",
    "district", "ward", "bed", "protocol", "report", "allergic", "sulfa", "drugs",
    "arrival", "sent", "emergency", "referred", "worker", "village", "visit",
    "next", "verify", "identity", "old", "card", "merged", "temp", "diet", "salt",
    "low", "diabetic", "advice", "morning", "evening", "condition", "discharged",
    "consent", "signed", "referral", "note", "sector", "phase", "colony", "quarter", "qtr",
    "flat", "plot", "unknown", "none", "null", "na",
}


def _looks_like_person_name(val: str, allow_synthetic_canary: bool = False) -> bool:
    s = unicodedata.normalize("NFC", val).strip()
    if not s or len(s) > 60:
        return False
    if "@" in s or "/" in s or "\\" in s:
        return False
    low = fold_latin_accents(s).casefold()
    if low in _PROSE_OR_CLINICAL_STOPWORDS:
        return False
    if low.startswith(("dr.", "dr ", "डॉ", "mr.", "mrs.", "ms.")):
        return False
    if _RE_CLINICAL_UNIT_VAL.search(s) or _RE_ICD_OR_CODE.match(s.upper()) or _RE_HOUSE_VAL.match(s) or _RE_LOCALITY_VAL.match(s):
        return False
    toks = [t for t in re.split(r"[\s.]+", s) if t]
    if not (1 <= len(toks) <= 4):
        return False
    has_digit = any(ch.isdigit() for ch in s)
    if has_digit:
        if not allow_synthetic_canary:
            return False
        alpha_toks = [t for t in toks if not any(ch.isdigit() for ch in t)]
        digit_toks = [t for t in toks if any(ch.isdigit() for ch in t)]
        if len(toks) >= 2 and not alpha_toks:
            return False
        for dt in digit_toks:
            folded = fold_latin_accents(dt)
            if not _SYNTH_CANARY_KEY_RE.match(folded):
                return False
            if re.sub(r"\d+", "", folded).casefold() in _PROSE_OR_CLINICAL_STOPWORDS:
                return False

    has_latin = False
    has_upper_latin = False
    for t in toks:
        t_clean = re.sub(r"\d+", "", t) if allow_synthetic_canary else t
        if not t_clean or not re.fullmatch(rf"[{_LL}{_DL}'\-]{{1,25}}", t_clean):
            return False
        if fold_latin_accents(t_clean).casefold() in _PROSE_OR_CLINICAL_STOPWORDS:
            return False
        if any(not is_deva_char(c) and c.isalpha() for c in t_clean):
            has_latin = True
            if unicodedata.normalize("NFC", t_clean)[0].isupper():
                has_upper_latin = True
    # Reject all-lowercase Latin prose like "oxygen given en route"
    if has_latin and not has_upper_latin:
        return False
    return True


def _looks_like_address_composite(val: str) -> bool:
    s = unicodedata.normalize("NFC", val)
    return bool(
        re.search(
            r"(?i)(?:h\.?\s*no\.?|house\s+\d|makan\s*no|मकान\s*नं|qtr[\s.\-#]*\d|quarter\s*\d|flat\s*(?:no\.?)?\s*\d|plot\s*(?:no\.?)?\s*\d|"
            r"ward\s+\d+|sector\s*[-#]?\s*\w+|mohalla|colony|मोहल्ला|"
            r"tehsil|तहसील|distt\.?|जिला|near\s+|opp\.|के\s+पास|के\s+पीछे|vill\.|gram\s+|ग्राम\s+|c/o\s+)",
            s,
        )
    )


# --------------------------------------------------------------------------- composite address scrubber
_ADDR_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("RELATIVE_NAME", re.compile(rf"(?i)(?<=C/O\s)([{_LU}{_DL}][{_LL}{_DL}.'\-]+(?:\s+[{_LU}{_DL}][{_LL}{_DL}.'\-]+){{0,3}})(?=,|$)")),
    ("HOUSE", re.compile(r"(?i)\b(?:H\.?\s*No\.?|House(?:\s+No\.?)?|Makan\s+No\.?|Qtr\.?|Quarter|Flat(?:\s+No\.?)?|Plot(?:\s+No\.?)?|Door(?:\s+No\.?)?)[\s.\-#:/]*[\dA-Za-z/-]*\d[\dA-Za-z/-]*")),
    ("HOUSE", re.compile(r"मकान\s*(?:नं\.?|सं\.?)\s*[\d०-९A-Za-z/-]+")),
    ("LOCALITY", re.compile(r"(?i)\bWard\s*(?:No\.?)?\s*\d+")),
    ("LOCALITY", re.compile(r"वार्ड\s*(?:नं\.?)?\s*[\d०-९]+")),
    ("LOCALITY", re.compile(r"(?i)\bSector\s*[-#]?\s*[A-Za-z0-9]+(?:\s+(?:Colony|Phase|Extension|Enclave|Block))?")),
    ("LOCALITY", re.compile(r"(?i)\bMohalla\s+[^,\n]+")),
    ("LOCALITY", re.compile(rf"\b[{_LU}][a-zA-Z0-9\u00DF-\u024F]*(?:\s+[{_LU}0-9][a-zA-Z0-9\u00DF-\u024F]*){{0,2}}\s+(?:Colony|Nagar|Vihar|Enclave|Puram|Extension|Basti|Tola|Layout)\b")),
    ("LOCALITY", re.compile(r"मोहल्ला\s+[^,\n]+")),
    ("LANDMARK", re.compile(r"(?i)\b(?:near|opp\.?|opposite|behind|beside)\s+[^,\n]+")),
    ("LANDMARK", re.compile(r"[^,\n]+\s+(?:के\s+पास|के\s+पीछे|के\s+सामने)")),
    ("VILLAGE", re.compile(r"(?i)\b(?:Vill\.?|Village|Gram)\s+[^,\n\-]+")),
    ("VILLAGE", re.compile(r"ग्राम\s+[^,\n\-]+")),
    ("VILLAGE", re.compile(rf"(?i)(?<=,\s)(?!(?:H\.?\s*No|House|Makan|Qtr|Quarter|Flat|Plot|Ward|Sector|Mohalla|near|opp|Tehsil|Distt|C/O)\b)([{_LU}{_DL}][^\n,]+?)(?=,\s*(?:Tehsil|तहसील)\b)")),
    ("PIN", re.compile(r"(?<!\d)[1-8]\d{5}(?!\d)")),
]


def scrub_composite_address(text: str, stats: dict[str, int] | None = None) -> str:
    """Scrub House, Locality, Landmark, Village, PIN, and C/O relative inside composite address text while preserving Tehsil/District."""
    out = unicodedata.normalize("NFC", text)
    for label, rx in _ADDR_PATTERNS:
        out, n = rx.subn(f"[{label}]", out)
        if stats is not None and n:
            stats[label] = stats.get(label, 0) + n
    return out


_FREE_TEXT_HEADERS = {
    "remarks", "remark", "notes", "note", "comment", "comments", "description", "summary", "text",
}


# --------------------------------------------------------------------------- header / key synonym mapping
# --------------------------------------------------------------------------- compositional header/key inference + schema engine
_PATIENT_SUBJ_PARTS = {
    "pt", "pat", "patient", "beneficiary", "client", "member", "subject", "person",
    "individual", "rogi", "mariz", "मरीज़", "रोगी", "लाभार्थी",
}
_REL_SUBJ_PARTS = {
    "rel", "relative", "guardian", "caretaker", "attendant", "nok", "kin", "father",
    "mother", "husband", "wife", "spouse", "son", "daughter", "parent", "witnessed",
    "pita", "pati", "mata", "अभिभावक", "पिता", "पति", "माता", "संरक्षक",
}
_CLINICIAN_PARTS = {
    "dr", "doc", "doctor", "consultant", "prescriber", "practitioner", "physician",
    "author", "clinician", "nurse", "surgeon",
}
_FACILITY_PARTS = {
    "hosp", "hospital", "clinic", "facility", "org", "organization", "centre", "center",
    "trust", "institute", "laboratory", "lab", "dept", "department", "system",
}
_NAME_PROP_PARTS = {
    "name", "nm", "naam", "full", "fullname", "first", "given", "fore", "fname",
    "last", "sur", "family", "sname", "lname", "middle", "local", "नाम",
}
_FIRST_NAME_PARTS = {"first", "given", "fore", "fname"}
_LAST_NAME_PARTS = {"last", "sur", "family", "sname", "lname"}
_ID_PROP_PARTS = {
    "id", "uid", "uhid", "mrn", "ref", "code", "num", "no", "reg", "ipd", "opd",
    "adm", "enroll", "enrol", "record", "panchikaran", "पंजीकरण",
}
_NON_PATIENT_ID_EXCLUDE = {
    "package", "icd", "loinc", "sex", "gender", "test", "drug", "bill", "sample",
    "visit", "encounter", "claim", "batch", "doc", "att", "org", "category",
}
_CLINICAL_ONLY_PARTS = {
    "drug", "medicine", "medication", "rx", "strength", "dose", "dosage", "freq",
    "frequency", "duration", "form", "schedule", "category", "test", "loinc", "unit",
    "range", "reference", "result", "results", "value", "values", "diagnosis",
    "diagnoses", "dx", "complaint", "complaints", "history", "procedure", "package",
    "icd", "symptom", "cases", "week", "status", "amount", "inr", "vitals", "bp",
    "pulse", "temp", "spo2", "weight", "trip", "from", "to", "sample", "bill",
    "intent", "clinicalstatus", "resourcetype", "class", "ward", "wards", "beds", "occupied",
}
_DATE_PROP_PARTS = {
    "dt", "date", "dinank", "tithi", "दिनांक", "तारीख", "collected", "reported",
    "registered", "admission", "discharge", "adm", "dis", "exported", "generated",
    "recorded", "authored", "performed", "effective", "lastupdated", "timestamp", "भर्ती", "छुट्टी",
}
_DOB_PROP_PARTS = {"dob", "birth", "bdate", "janm", "janam", "जन्म"}
_PHONE_PARTS = {
    "mobile", "mob", "phone", "ph", "phn", "tel", "cell", "contact", "contct",
    "whatsapp", "callback", "sampark", "दूरभाष", "संपर्क", "फ़ोन", "फोन", "मोबाइल",
}
_PII_STRUCT_LABELS = {
    "PATIENT_NAME", "RELATIVE_NAME", "NAME_FIRST", "NAME_LAST", "NAME_LOCAL",
    "MRN", "AADHAAR", "AADHAAR_MASKED", "ABHA_NUMBER", "ABHA_ADDRESS", "PAN",
    "INSURANCE_ID", "PHONE", "EMAIL", "DOB", "PIN", "HOUSE", "LOCALITY",
    "LANDMARK", "VILLAGE", "ADDRESS", "REDACTED_PII",
}


def classify_header_parts(header_raw: str, parent_key: str = "") -> str | None:
    """Infer column/key semantic category compositionally from header parts + parent key (no unseen-literal rules)."""
    k = _norm_key(header_raw)
    pk = _norm_key(parent_key)
    parts = split_key_parts(header_raw)
    pset = set(parts)
    pparts = set(split_key_parts(parent_key))
    if not parts:
        return None

    # Metadata filesystem/source paths (e.g. source_path, source_file, export_path)
    if pset & {"path", "filepath", "file_path", "export_path", "dirpath"} or (
        "file" in pset and (bool(pset & {"source", "export", "orig", "raw"}) or bool(pparts & {"meta"}))
    ):
        return "META_PATH"

    # Explicit facility / clinical sub-objects take precedence (except hospital_uhid / address)
    if (
        pk in ("facility", "hospital", "organization", "lab", "panels", "diagnosis", "diagnoses", "rx", "medications", "vitals", "results")
        or bool(pparts & (_CLINICAL_ONLY_PARTS | _FACILITY_PARTS))
    ) and not (pset & {"uhid", "mrn"}):
        return "NON_PII"
    if (pset & _CLINICIAN_PARTS) and not (pset & _PATIENT_SUBJ_PARTS):
        return "DOCTOR"
    if (
        k in _FREE_TEXT_HEADERS
        or ("note" in pset or "notes" in pset or "remarks" in pset or "comment" in pset)
    ) and not (pset & {"addr", "address", "pata", "niwas", "residence", "residential", "street", "निवास", "पता"}):
        return "FREE_TEXT"

    # Strong national/contact/demographic stems checked before clinical exclusions
    if any(w in k for w in ("aadhar", "aadhaar", "adhaar", "addhar", "uidai", "आधार")):
        return "AADHAAR"
    if ("abha" in k or "abdm" in k or "phr" in k or "health_id" in k or "healthid" in k) and ("addr" in pset or "addr" in k):
        return "ABHA_ADDRESS"
    if any(w in k for w in ("abha", "abdm", "ndhm", "health_id", "healthid", "आभा")):
        return "ABHA_NUMBER"
    if pset & {"email", "mail", "e_mail"} or "email" in k or "e_mail" in k:
        return "EMAIL"
    if "pan" in pset or "pancard" in k or k.startswith("pan_") or k.endswith("_pan"):
        return "PAN"
    if any(w in k for w in ("pmjay", "policy", "insurance", "ins_id", "bima")):
        return "INSURANCE_ID"
    if pset & _PHONE_PARTS:
        return "PHONE"

    # Birth dates before clinical dates
    if pset & _DOB_PROP_PARTS:
        return "DOB"

    # Age & Gender (age_at_visit, umar, sex_code, age_sex)
    if (pset & {"age", "umar", "आयु", "उम्र"}) and (pset & {"sex", "gender", "ling", "लिंग"}):
        return "AGE_SEX"
    if (pset & {"age", "umar", "आयु", "उम्र"}) and not (pset & {"package", "stage"}):
        return "AGE"
    if (pset & {"sex", "gender", "ling", "लिंग"}) and not (pset & {"relation"}):
        return "GENDER"

    # Geographical & Address parts
    if pset & {"pin", "pincode", "zip", "zipcode", "postal", "postalcode", "dak", "पिन"}:
        return "PIN"
    if pset & {"dist", "distt", "district", "jila", "जिला"}:
        return "DISTRICT"
    if pset & {"tehsil", "taluka", "तहसील"}:
        return "TEHSIL"
    if pset & {"landmark"}:
        return "LANDMARK"
    if pset & {"house", "makan", "hno", "qtr", "quarter", "flat", "apt", "apartment", "door", "plot", "dwelling", "मकान"}:
        return "HOUSE"
    if pset & {"locality", "mohalla", "colony", "sector", "basti", "tola", "neighborhood", "neighbourhood", "suburb", "मोहल्ला"}:
        return "LOCALITY"
    if pset & {"vill", "village", "gram", "gaon", "native", "ग्राम"}:
        return "VILLAGE"
    if pset & {"addr", "address", "pata", "niwas", "residence", "residential", "street", "निवास", "पता"} or (
        "line" in pset and "address" in pparts
    ):
        return "ADDRESS"
    if pset & {"city", "town"}:
        return "VILLAGE"

    # Patient IDs vs Clinical Codes
    if pset & {"uhid", "mrn", "panchikaran", "पंजीकरण"}:
        return "MRN"
    if (pset & _ID_PROP_PARTS) and not (pset & _NON_PATIENT_ID_EXCLUDE) and not (pset & _FACILITY_PARTS):
        if pset & _PATIENT_SUBJ_PARTS or k in ("reg_no", "regno", "adm_no", "ipd_no", "ipd"):
            return "MRN"

    # Clinical / facility tokens (avoid matching dx_text, rx_text, bed_count, cases, etc. as PII)
    if (pset & _FACILITY_PARTS) and not (pset & _PATIENT_SUBJ_PARTS):
        return "NON_PII"
    if (pset & _CLINICAL_ONLY_PARTS) and not (pset & {"name", "nm", "naam"} and pset & _PATIENT_SUBJ_PARTS):
        return "NON_PII"

    # Explicit Clinical / Encounter dates vs generic unknown date columns
    if k in _CLINICAL_DATE_KEYS or (
        pset & {
            "collected", "reported", "registered", "admission", "discharge", "adm", "dis",
            "visit", "encounter", "exported", "generated", "recorded", "authored", "performed",
            "effective", "lastupdated", "reg", "भर्ती", "छुट्टी",
        }
    ):
        return "CLINICAL_DATE"
    if pset & _DATE_PROP_PARTS:
        return "DATE"

    # Relatives & Caretakers (care_of_rel, father_name, अभिभावक, etc.)
    if (pset & _REL_SUBJ_PARTS) or ("care" in pset and ("of" in pset or "taker" in pset or "rel" in pset)):
        if pset & {"type", "relation"} and not (pset & _NAME_PROP_PARTS):
            return "REL_TYPE"
        return "RELATIVE_NAME"

    # Split name parts inside dict or CSV (given_nm, sur_nm, first_name, last_name, local)
    is_rel_parent = bool(pparts & _REL_SUBJ_PARTS or pparts & {"contact"})
    if (pset & _FIRST_NAME_PARTS) and (pset & {"name", "nm", "naam", "नाम"} or len(pset) == 1 or pparts & {"name", "patient"}):
        return "REL_NAME_FIRST" if is_rel_parent else "NAME_FIRST"
    if (pset & _LAST_NAME_PARTS) and (pset & {"name", "nm", "naam", "नाम"} or len(pset) == 1 or pparts & {"name", "patient"}):
        return "REL_NAME_LAST" if is_rel_parent else "NAME_LAST"
    if "local" in pset and (pparts & {"name", "patient", "pt", "naam"} or not pparts):
        return "NAME_LOCAL"

    # Full Patient Names (patient_name, pt_full, full_pt_name, रोगी_नाम, name)
    if (pset & _PATIENT_SUBJ_PARTS) and (pset & _NAME_PROP_PARTS or len(pset) == 1):
        return "PATIENT_NAME"
    if pset & {"name", "naam", "नाम"}:
        if is_rel_parent:
            return "RELATIVE_NAME"
        if not pparts or (pparts & (_PATIENT_SUBJ_PARTS | {"name", "raised", "meta", "roster", "patients", "records", "visits"})):
            return "PATIENT_NAME"
    if "full" in pset and (pset & _PATIENT_SUBJ_PARTS):
        return "PATIENT_NAME"

    return None


def _hint_from_header(header_raw: str, parent_key: str = "") -> str | None:
    """Legacy wrapper mapping compositional header classification to canonical PII/control labels."""
    res = classify_header_parts(header_raw, parent_key)
    if res in ("NAME_FIRST", "NAME_LAST", "NAME_LOCAL"):
        return "PATIENT_NAME"
    if res in ("REL_NAME_FIRST", "REL_NAME_LAST"):
        return "RELATIVE_NAME"
    if res in ("DOCTOR", "CLINICAL_NON_PII", "FREE_TEXT", "REL_TYPE", "CITY_DISTRICT", "DISTRICT", "TEHSIL", "AGE", "AGE_SEX", "GENDER"):
        return "NON_PII" if res in ("DOCTOR", "FREE_TEXT", "REL_TYPE", "CLINICAL_NON_PII", "DISTRICT", "TEHSIL", "AGE", "AGE_SEX", "GENDER", "CITY_DISTRICT") else res
    return res


@dataclass
class TableSchema:
    """Unified semantic schema inferred from table headers/parts, value patterns, and neighbor columns."""

    col_types: list[str | None]          # Redaction PII type per column (None if clinical/QI/safe)
    part_hints: list[str | None]         # Compositional classify_header_parts result per column
    is_patient_table: bool = False
    first_name_cols: list[int] = field(default_factory=list)
    last_name_cols: list[int] = field(default_factory=list)
    age_cols: list[int] = field(default_factory=list)
    sex_cols: list[int] = field(default_factory=list)
    age_sex_cols: list[int] = field(default_factory=list)
    district_cols: list[int] = field(default_factory=list)
    tehsil_cols: list[int] = field(default_factory=list)
    doctor_cols: list[int] = field(default_factory=list)
    notes_cols: list[int] = field(default_factory=list)
    rel_type_cols: list[int] = field(default_factory=list)
    date_cols: list[int] = field(default_factory=list)


@dataclass
class DictSchema:
    """Semantic schema inferred for a JSON dict object via key parts, values, and sibling context."""

    is_patient_obj: bool = False
    is_person_map_dict: bool = False
    field_types: dict[str, str | None] = field(default_factory=dict)
    part_hints: dict[str, str | None] = field(default_factory=dict)
    sub_dict_roles: dict[str, str] = field(default_factory=dict)  # e.g. "PERSON_NAME_OBJ" / "RELATIVE_NAME_OBJ"


def _is_clinical_column_values(non_empty: list[Any]) -> bool:
    """Return True when column values appear clinical, fixed status/enum, dates, or lab scalars."""
    if not non_empty:
        return True
    n = len(non_empty)
    strs = [_norm_digits(str(v)) for v in non_empty]
    uniq = {s.casefold() for s in strs}
    # Low-cardinality enums in multi-row tables (statuses, departments, M/F, booleans)
    if n >= 8 and len(uniq) <= 4:
        return True
    # Small integers (e.g., age, beds, occupied, counts) or floats (lab results, amounts)
    if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in non_empty):
        if any(isinstance(v, float) for v in non_empty):
            return True
        if all(0 <= int(v) <= 9999 for v in non_empty):
            return True
    clin_hits = 0
    doc_hits = 0
    date_hits = 0
    for s in strs:
        sl = fold_latin_accents(s).casefold()
        if sl in _PROSE_OR_CLINICAL_STOPWORDS or sl in _CLINICAL_OR_NON_NAME_WORDS:
            clin_hits += 1
        elif _RE_CLINICAL_UNIT_VAL.search(s) or _RE_ICD_OR_CODE.match(s.upper()):
            clin_hits += 1
        elif re.match(r"^(?:dr\.?|डॉ\.?)\s+", s, re.I):
            doc_hits += 1
        elif _RE_DATE.match(s) or re.match(r"^\d{1,3}\s*/\s*[MF]$", s, re.I):
            date_hits += 1
    if clin_hits >= max(1, int(0.3 * n)) or doc_hits >= max(1, int(0.3 * n)) or date_hits >= max(1, int(0.5 * n)):
        return True
    # Pure float-like or short numeric measurement columns ("12.4", "70-140")
    num_hits = sum(1 for s in strs if re.fullmatch(r"\d+(?:\.\d+)?(?:\s*[-–]\s*\d+(?:\.\d+)?)?", s))
    if num_hits >= max(1, int(0.6 * n)):
        return True
    return False


def infer_column_type(values: list[Any], header: str = "", parent_key: str = "") -> str | None:
    """Infer PII semantic category from VALUES FIRST, using compositional header/key parts as hints."""
    part_hint = classify_header_parts(header, parent_key)
    hint = _hint_from_header(header, parent_key)
    if hint in ("NON_PII", "CLINICAL_DATE", "META_PATH"):
        return hint if hint == "META_PATH" else None

    non_empty = [v for v in values if v is not None and str(v).strip() != ""]
    if not non_empty:
        return hint

    # Value-first tallies
    n = len(non_empty)
    counts: dict[str, int] = {}
    years: list[int] = []
    int_only = all(isinstance(v, int) and not isinstance(v, bool) for v in non_empty)

    for v in non_empty:
        s = _norm_digits(str(v))
        if _RE_EMAIL.match(s):
            k = "ABHA_ADDRESS" if _RE_ABHA_ADDR.match(s) else "EMAIL"
            counts[k] = counts.get(k, 0) + 1
        elif _RE_PAN.match(s.upper()):
            counts["PAN"] = counts.get("PAN", 0) + 1
        elif _RE_ABHA.match(s):
            counts["ABHA_NUMBER"] = counts.get("ABHA_NUMBER", 0) + 1
        elif _RE_AADHAAR_MASKED.match(s):
            counts["AADHAAR_MASKED"] = counts.get("AADHAAR_MASKED", 0) + 1
        elif _RE_AADHAAR.match(s):
            counts["AADHAAR"] = counts.get("AADHAAR", 0) + 1
        elif _RE_PHONE.match(s) or _RE_PHONE_MASKED.match(s):
            counts["PHONE"] = counts.get("PHONE", 0) + 1
        elif _RE_INSURANCE.match(s):
            counts["INSURANCE_ID"] = counts.get("INSURANCE_ID", 0) + 1
        elif _RE_MRN.match(s):
            counts["MRN"] = counts.get("MRN", 0) + 1
        elif _RE_PIN.match(s):
            counts["PIN"] = counts.get("PIN", 0) + 1
        elif _RE_DATE.match(s):
            yr = _extract_year(s)
            if yr is not None:
                years.append(yr)
            counts["DATE"] = counts.get("DATE", 0) + 1
        elif _RE_HOUSE_VAL.match(s):
            counts["HOUSE"] = counts.get("HOUSE", 0) + 1
        elif _RE_LOCALITY_VAL.match(s):
            counts["LOCALITY"] = counts.get("LOCALITY", 0) + 1
        elif _RE_LANDMARK_VAL.match(s):
            counts["LANDMARK"] = counts.get("LANDMARK", 0) + 1
        elif _looks_like_address_composite(s):
            counts["ADDRESS"] = counts.get("ADDRESS", 0) + 1
        elif _RE_REL_PREFIX.match(s):
            counts["RELATIVE_NAME"] = counts.get("RELATIVE_NAME", 0) + 1
        elif not int_only and _looks_like_person_name(s):
            counts["NAME_LIKE"] = counts.get("NAME_LIKE", 0) + 1

    # Strong value-based overrides (even if header is misspelled or completely unknown)
    for tag in ("EMAIL", "ABHA_ADDRESS", "PAN", "ABHA_NUMBER", "AADHAAR", "AADHAAR_MASKED", "PHONE", "INSURANCE_ID", "MRN"):
        if counts.get(tag, 0) >= max(1, int(0.5 * n)):
            return tag
    # Mixed Aadhaar + Masked Aadhaar column
    if counts.get("AADHAAR", 0) + counts.get("AADHAAR_MASKED", 0) >= max(1, int(0.5 * n)):
        return "AADHAAR"
    # PIN: 6 digits [1-8]\d{5}, only if majority match and header doesn't look like lab/amount
    if counts.get("PIN", 0) >= max(1, int(0.7 * n)) and (hint == "PIN" or _norm_key(header) not in ("result", "value", "amount", "count")):
        return "PIN"
    # Date: distinguish DOB vs clinical date by header OR birth-year spread
    if counts.get("DATE", 0) >= max(1, int(0.5 * n)):
        if hint == "DOB":
            return "DOB"
        if years and (min(years) < 2018 or (max(years) - min(years) >= 8)):
            return "DOB"
        return None
    # Composite address & explicit residential components
    if counts.get("ADDRESS", 0) >= max(1, int(0.4 * n)):
        return "ADDRESS"
    if counts.get("HOUSE", 0) >= max(1, int(0.5 * n)):
        return "HOUSE"
    if counts.get("LOCALITY", 0) >= max(1, int(0.5 * n)):
        return "LOCALITY"
    if counts.get("LANDMARK", 0) >= max(1, int(0.5 * n)):
        return "LANDMARK"
    # Relative prefix in column values
    if counts.get("RELATIVE_NAME", 0) >= max(1, int(0.4 * n)):
        return "RELATIVE_NAME"

    # Header-guided resolution for names / geography / ambiguous IDs
    if hint in ("PATIENT_NAME", "RELATIVE_NAME", "VILLAGE", "HOUSE", "LOCALITY", "LANDMARK", "PIN", "ADDRESS", "MRN", "INSURANCE_ID", "PHONE", "EMAIL", "PAN", "AADHAAR", "ABHA_NUMBER", "ABHA_ADDRESS", "DOB"):
        return hint

    # Skip value-guessing on columns whose header parts already signal clinical / geo / demographic roles
    if part_hint in ("DOCTOR", "CLINICAL_NON_PII", "FREE_TEXT", "REL_TYPE", "CITY_DISTRICT", "DISTRICT", "TEHSIL", "AGE", "AGE_SEX", "GENDER", "CLINICAL_DATE"):
        return None

    # Value-only name inference when header is unknown (require >=2 tokens & high uniqueness to avoid drug/test FPs)
    if counts.get("NAME_LIKE", 0) >= max(2, int(0.7 * n)) and not _is_clinical_column_values(non_empty):
        multi_tok = sum(1 for v in non_empty if len(str(v).split()) >= 2)
        if multi_tok >= max(1, int(0.5 * len(non_empty))):
            return "PATIENT_NAME"

    return None


def infer_table_schema(header: list[str], body: list[list[str]], parent_key: str = "") -> TableSchema:
    """Two-pass table schema inference combining header parts, value patterns, and neighboring columns."""
    n_cols = len(header)
    col_vals_all = [[r[c] if c < len(r) else "" for r in body] for c in range(n_cols)]
    part_hints = [classify_header_parts(h, parent_key) for h in header]
    col_types = [infer_column_type(col_vals_all[c], header=header[c], parent_key=parent_key) for c in range(n_cols)]

    schema = TableSchema(col_types=col_types, part_hints=part_hints)
    for c_idx, (ph, ct) in enumerate(zip(part_hints, col_types)):
        if ph in ("NAME_FIRST", "REL_NAME_FIRST"):
            schema.first_name_cols.append(c_idx)
        elif ph in ("NAME_LAST", "REL_NAME_LAST"):
            schema.last_name_cols.append(c_idx)
        elif ph == "AGE":
            schema.age_cols.append(c_idx)
        elif ph == "GENDER":
            schema.sex_cols.append(c_idx)
        elif ph == "AGE_SEX":
            schema.age_sex_cols.append(c_idx)
        elif ph in ("DISTRICT", "CITY_DISTRICT"):
            schema.district_cols.append(c_idx)
        elif ph == "TEHSIL":
            schema.tehsil_cols.append(c_idx)
        elif ph == "DOCTOR":
            schema.doctor_cols.append(c_idx)
        elif ph == "FREE_TEXT":
            schema.notes_cols.append(c_idx)
        elif ph == "REL_TYPE":
            schema.rel_type_cols.append(c_idx)
        elif ph in ("CLINICAL_DATE", "DATE") or ct == "DOB":
            schema.date_cols.append(c_idx)
        elif ct is None and any(v and _RE_DATE.match(_norm_digits(str(v))) for v in col_vals_all[c_idx][:10]):
            schema.date_cols.append(c_idx)

    # Neighbor check for AGE_SEX/AGE values even under unknown headers (e.g. "34/M", "45 वर्ष")
    for c_idx in range(n_cols):
        if c_idx in schema.age_cols or c_idx in schema.age_sex_cols or col_types[c_idx] is not None:
            continue
        non_empty = [str(v).strip() for v in col_vals_all[c_idx] if v is not None and str(v).strip()]
        if non_empty:
            as_hits = sum(1 for v in non_empty if re.fullmatch(r"\d{1,3}\s*/\s*[MF]", v, re.I))
            if as_hits >= max(1, int(0.6 * len(non_empty))):
                schema.age_sex_cols.append(c_idx)

    has_strong_pt_anchor = any(
        ct in ("PATIENT_NAME", "RELATIVE_NAME", "MRN", "AADHAAR", "ABHA_NUMBER", "ABHA_ADDRESS", "PHONE", "EMAIL", "PAN", "INSURANCE_ID", "DOB")
        for ct in col_types
    )
    has_geo_anchor = any(ct in ("PIN", "VILLAGE", "ADDRESS", "HOUSE", "LOCALITY") for ct in col_types) or bool(schema.district_cols)
    schema.is_patient_table = has_strong_pt_anchor or (bool(schema.age_cols or schema.age_sex_cols) and bool(schema.sex_cols or schema.age_sex_cols))

    if not schema.is_patient_table:
        return schema

    # Stage 2: Neighbor-contextual escalation for unfamiliar columns inside patient-bearing tables
    for c_idx in range(n_cols):
        if col_types[c_idx] is not None:
            continue
        ph = part_hints[c_idx]
        if ph in ("DOCTOR", "NON_PII", "CLINICAL_NON_PII", "FREE_TEXT", "REL_TYPE", "DISTRICT", "TEHSIL", "CITY_DISTRICT", "AGE", "AGE_SEX", "GENDER", "CLINICAL_DATE", "DATE"):
            continue
        non_empty = [v for v in col_vals_all[c_idx] if v is not None and str(v).strip() != ""]
        if not non_empty:
            continue
        if _is_clinical_column_values(non_empty):
            continue

        strs = [_norm_digits(str(v)) for v in non_empty]
        n = len(strs)
        uniq_ratio = len({s.casefold() for s in strs}) / max(1, n)

        # Neighbor geo escalation for unfamiliar residential columns next to PIN/District/Village
        if has_geo_anchor:
            h_cnt = sum(1 for s in strs if _RE_HOUSE_VAL.match(s))
            l_cnt = sum(1 for s in strs if _RE_LOCALITY_VAL.match(s))
            if h_cnt >= max(1, int(0.35 * n)):
                col_types[c_idx] = "HOUSE"
                continue
            if l_cnt >= max(1, int(0.35 * n)):
                col_types[c_idx] = "LOCALITY"
                continue

        # Likely person name next to patient IDs/phones/demographics even with single-token local names
        name_cnt = sum(1 for s in strs if _looks_like_person_name(s))
        if name_cnt >= max(1, int(0.6 * n)) and (uniq_ratio >= 0.4 or n <= 2):
            col_types[c_idx] = "RELATIVE_NAME" if ph == "RELATIVE_NAME" else "PATIENT_NAME"
            continue

        # Uncertain high-cardinality non-clinical identifier / string adjacent to strong patient anchors
        if uniq_ratio >= 0.5 and n >= 3:
            alnum_id_cnt = sum(
                1 for s in strs
                if 5 <= len(s) <= 28
                and any(c.isdigit() for c in s)
                and any(c.isalpha() for c in s)
                and not _RE_ICD_OR_CODE.match(s.upper())
                and not _RE_DATE.match(s)
                and not _RE_CLINICAL_UNIT_VAL.search(s)
            )
            if alnum_id_cnt >= int(0.6 * n):
                col_types[c_idx] = "MRN"
            elif has_geo_anchor and all(len(s.split()) >= 2 for s in strs):
                col_types[c_idx] = "ADDRESS"
            elif all(not s.isdigit() for s in strs) and not any(c in "".join(strs) for c in "+/%°"):
                col_types[c_idx] = "REDACTED_PII"

    return schema


def infer_dict_schema(obj: dict[str, Any], parent_key: str = "") -> DictSchema:
    """Inspect a JSON dict object via key parts, leaf value patterns, and sibling context."""
    schema = DictSchema()
    pk_parts = set(split_key_parts(parent_key))

    # First check person-indexed map dicts like {"by_beneficiary": {"Name": "MRN", ...}}
    if len(obj) >= 1 and (
        parent_key and (split_key_parts(parent_key)[0] == "by" or "map" in pk_parts or "index" in pk_parts)
    ):
        if not (pk_parts & _CLINICAL_ONLY_PARTS) and not (pk_parts & _FACILITY_PARTS):
            schema.is_person_map_dict = True

    scalar_keys: list[str] = []
    dict_keys: list[str] = []
    for k, v in obj.items():
        ks = str(k)
        ph = classify_header_parts(ks, parent_key)
        schema.part_hints[ks] = ph
        if isinstance(v, dict):
            dict_keys.append(ks)
            continue
        if isinstance(v, list):
            continue
        scalar_keys.append(ks)
        ct = _classify_scalar_by_value_and_key(v, ks, parent_key)
        schema.field_types[ks] = ct
        if ct in _PII_STRUCT_LABELS or ph in ("PATIENT_NAME", "MRN", "AGE", "AGE_SEX", "GENDER", "DOB", "DISTRICT"):
            schema.is_patient_obj = True

    for ks in dict_keys:
        v = obj[ks]
        child_parts = [classify_header_parts(str(ck), ks) for ck in v.keys()]
        has_name_parts = any(cp in ("NAME_FIRST", "NAME_LAST", "REL_NAME_FIRST", "REL_NAME_LAST") for cp in child_parts)
        is_non_clin_dict = _norm_key(ks) not in ("facility", "hospital", "organization", "lab", "address", "_export_meta", "meta", "diagnosis", "vitals", "rx", "panels")
        str_vals = [str(cv).strip() for cv in v.values() if isinstance(cv, str) and cv.strip()]
        all_name_vals = bool(str_vals) and len(str_vals) <= 4 and all(_looks_like_person_name(sv) for sv in str_vals) and not _is_clinical_column_values(str_vals)
        if is_non_clin_dict and (has_name_parts or ((split_key_parts(ks)[0] in ("name", "naam", "patient") or schema.is_patient_obj) and all_name_vals)):
            is_rel = any(cp in ("REL_NAME_FIRST", "REL_NAME_LAST", "RELATIVE_NAME") for cp in child_parts) or bool(set(split_key_parts(ks)) & _REL_SUBJ_PARTS)
            schema.sub_dict_roles[ks] = "RELATIVE_NAME_OBJ" if is_rel else "PERSON_NAME_OBJ"
            schema.is_patient_obj = True

    # Also detect by_map pattern from values if top-level dict values are mostly MRNs/phones/patient records
    if not schema.is_person_map_dict and len(obj) >= 2 and not schema.is_patient_obj:
        val_id_cnt = sum(
            1 for v in obj.values()
            if isinstance(v, str) and (_RE_MRN.match(_norm_digits(v)) or _RE_PHONE.match(_norm_digits(v)))
        )
        key_name_cnt = sum(1 for k in obj.keys() if _looks_like_person_name(str(k), allow_synthetic_canary=True))
        if val_id_cnt >= max(2, int(0.5 * len(obj))) and key_name_cnt >= 1:
            schema.is_person_map_dict = True

    # Neighbor escalation for remaining unclassified non-clinical scalar fields inside a patient dict
    if schema.is_patient_obj:
        for ks in scalar_keys:
            if schema.field_types.get(ks) is not None:
                continue
            ph = schema.part_hints.get(ks)
            if ph in ("DOCTOR", "NON_PII", "CLINICAL_NON_PII", "FREE_TEXT", "REL_TYPE", "DISTRICT", "TEHSIL", "CITY_DISTRICT", "AGE", "AGE_SEX", "GENDER", "CLINICAL_DATE"):
                continue
            v = obj[ks]
            if not isinstance(v, str) or not v.strip() or _is_clinical_column_values([v]):
                continue
            ns = _norm_digits(v)
            if _RE_HOUSE_VAL.match(ns):
                schema.field_types[ks] = "HOUSE"
            elif _RE_LOCALITY_VAL.match(ns):
                schema.field_types[ks] = "LOCALITY"
            elif _RE_LANDMARK_VAL.match(ns):
                schema.field_types[ks] = "LANDMARK"

    return schema


def is_non_patient_table_content(header: list[str], body: list[list[str]], table_schema: TableSchema | None = None) -> bool:
    """Content-based check: True when table contains NO patient PII and NO patient demographic/visit QIs."""
    ts = table_schema if table_schema is not None else infer_table_schema(header, body)
    if any(ct in _PII_STRUCT_LABELS for ct in ts.col_types if ct is not None):
        return False
    if ts.age_cols or ts.age_sex_cols or ts.sex_cols or ts.district_cols or ts.tehsil_cols or ts.date_cols:
        return False
    return True


def is_non_patient_json_content(data: Any) -> bool:
    """Content-based check: True when JSON contains NO patient PII, NO FHIR Patient/Encounter, and NO patient QIs."""
    if not isinstance(data, dict):
        return True
    if _is_fhir_bundle(data):
        for ent in data.get("entry", []):
            res = ent.get("resource", {}) if isinstance(ent, dict) else {}
            if res.get("resourceType") in ("Patient", "Encounter", "Condition", "Observation", "DocumentReference"):
                return False
        return True

    stack: list[tuple[Any, str]] = [(data, "")]
    while stack:
        node, pk = stack.pop()
        if isinstance(node, dict):
            ds = infer_dict_schema(node, parent_key=pk)
            if ds.is_patient_obj or ds.is_person_map_dict or ds.sub_dict_roles:
                return False
            for k, v in node.items():
                ks = str(k)
                ph = ds.part_hints.get(ks)
                if ph in ("PATIENT_NAME", "RELATIVE_NAME", "MRN", "AGE", "AGE_SEX", "GENDER", "DOB", "DISTRICT", "TEHSIL"):
                    return False
                if isinstance(v, (dict, list)):
                    stack.append((v, ks))
                elif isinstance(v, str) and len(v.strip()) >= 24 and _RE_MRN.search(v) is None:
                    # Inspect scalars for base64 text or embedded identifiers
                    dec = sanitize_base64_string(v, lambda s: "[CHECKED]" if s else s)
                    if dec != v:
                        return False
        elif isinstance(node, list):
            for it in node:
                if isinstance(it, (dict, list)):
                    stack.append((it, pk))
    return True


_GENERIC_PATH_SEGMENTS = {
    "exports", "export", "his", "data", "opd", "ipd", "registration", "lab", "billing",
    "discharge", "notes", "records", "reports", "patients", "attachments", "clean",
    "backup", "archive", "temp", "tmp", "files", "input", "output",
}


def _scrub_meta_path_value(path_str: str) -> str:
    """Scrub patient names/canaries/MRNs/phones from filesystem path segments."""
    tokens = re.split(r"([\\/]+)", path_str)
    out: list[str] = []
    last_non_sep = max((i for i, t in enumerate(tokens) if t and t[0] not in "\\/"), default=-1)
    for idx, seg in enumerate(tokens):
        if not seg or seg[0] in "\\/":
            out.append(seg)
            continue
        if re.fullmatch(r"[A-Za-z]:", seg):
            out.append(seg)
            continue
        stem, ext = seg, ""
        if idx == last_non_sep and "." in seg:
            dot = seg.rfind(".")
            if 1 <= len(seg) - dot <= 6:
                stem, ext = seg[:dot], seg[dot:]

        cleaned_stem = stem
        if stem.casefold() in _GENERIC_PATH_SEGMENTS:
            out.append(f"{stem}{ext}")
            continue
        # Scrub identifier cores inside path segments
        cleaned_stem = re.sub(
            r"(?:[A-Z]{2,6}/\d{2,4}/\d{3,}|[A-Z]{2,6}-\d{4,}|[A-Z]{2,6}\d{2,4}-\d{3,}|[A-Z]{2,6}-[A-Z]{2,4}-\d{3,}|[A-Z]{2,5}-\d{2,4}-\d{3,}|[A-Z]{2,5}\d{5,}|[A-Z]{2,6}-\d{2,4}-\d{4,}|[6-9]\d{9}|\d{5,})",
            "[MRN]",
            cleaned_stem,
        )
        rem_parts = [
            p for p in re.split(r"[_\-\s]+", cleaned_stem)
            if p and p != "[MRN]" and p.casefold() not in {"na", "n/a", "none", "null", "unknown", "unk"}
        ]
        non_gen_parts = [
            p for p in rem_parts
            if p.casefold() not in (_GENERIC_PATH_SEGMENTS | _CLINICAL_ONLY_PARTS | _FACILITY_PARTS)
            and not p.isdigit()
        ]
        is_name_next_to_mrn = (
            "[MRN]" in cleaned_stem
            and bool(non_gen_parts)
            and all(
                re.fullmatch(rf"[{_LU}][{_LL}'\-]{{1,25}}", p) or re.fullmatch(rf"[{_DL}]{{2,25}}", p)
                for p in non_gen_parts
            )
        )
        if (rem_parts and _looks_like_person_name(" ".join(rem_parts), allow_synthetic_canary=True)) or is_name_next_to_mrn:
            # Replace name portion while keeping [MRN] if also present
            if "[MRN]" in cleaned_stem:
                out.append(f"[PATIENT_NAME]_[MRN]{ext}")
            else:
                out.append(f"[PATIENT_NAME]{ext}")
        else:
            out.append(f"{cleaned_stem}{ext}")
    return "".join(out)


def redact_structured_value(val: Any, col_type: str, stats: dict[str, int] | None = None) -> Any:
    """Redact a structured cell/JSON leaf according to its inferred type."""
    if val is None:
        return val
    s = str(val)
    if not s.strip():
        return val

    if col_type == "ADDRESS":
        return scrub_composite_address(s, stats)

    if col_type == "META_PATH":
        out = _scrub_meta_path_value(s)
        if out != s and stats is not None:
            stats["META_PATH"] = stats.get("META_PATH", 0) + 1
        return out

    if col_type == "RELATIVE_NAME":
        m = _RE_REL_PREFIX.match(s.strip())
        if m:
            prefix = s.strip()[: m.start(1)]
            if stats is not None:
                stats["RELATIVE_NAME"] = stats.get("RELATIVE_NAME", 0) + 1
            return f"{prefix}[RELATIVE_NAME]"
        if stats is not None:
            stats["RELATIVE_NAME"] = stats.get("RELATIVE_NAME", 0) + 1
        return "[RELATIVE_NAME]"

    if col_type == "AADHAAR":
        ns = _norm_digits(s)
        tag = "AADHAAR_MASKED" if _RE_AADHAAR_MASKED.match(ns) else "AADHAAR"
        if stats is not None:
            stats[tag] = stats.get(tag, 0) + 1
        return f"[{tag}]"

    if stats is not None:
        stats[col_type] = stats.get(col_type, 0) + 1
    return f"[{col_type}]"


def _classify_scalar_by_value_and_key(val: Any, key: str, parent_key: str) -> str | None:
    """Inspect a single JSON scalar (including int/float!) by value first, then key parts."""
    hint = _hint_from_header(key, parent_key)
    if hint in ("NON_PII", "CLINICAL_DATE"):
        return None
    if hint == "META_PATH":
        return "META_PATH"

    if isinstance(val, bool) or val is None:
        return None

    # Handle numeric identifiers (e.g. Aadhaar as int under a misspelled key)
    if isinstance(val, int):
        if 100_000_000_000 <= val <= 999_999_999_999:
            return "AADHAAR"
        if 10_000_000_000_000 <= val <= 99_999_999_999_999:
            return "ABHA_NUMBER"
        if 6_000_000_000 <= val <= 9_999_999_999:
            return "PHONE"
        if hint == "PIN" and 100_000 <= val <= 899_999:
            return "PIN"
        return hint if hint in ("AADHAAR", "ABHA_NUMBER", "PHONE", "PIN", "MRN") else None

    if isinstance(val, float):
        return None

    if not isinstance(val, str) or not val.strip():
        return None

    ns = _norm_digits(val)
    # Catch filesystem path strings inside metadata/source keys even under unfamiliar key names
    if (
        (re.match(r"^[A-Za-z]:\\", ns) or ns.startswith(("/tmp/", "/var/", "/home/", "/mnt/", "/data/")))
        and ("meta" in parent_key.casefold() or set(split_key_parts(key)) & {"source", "file", "path", "dir", "export"})
    ):
        return "META_PATH"
    if _RE_ABHA_ADDR.match(ns):
        return "ABHA_ADDRESS"
    if _RE_EMAIL.match(ns):
        return "EMAIL"
    if _RE_PAN.match(ns.upper()):
        return "PAN"
    if _RE_ABHA.match(ns):
        return "ABHA_NUMBER"
    if _RE_AADHAAR_MASKED.match(ns):
        return "AADHAAR"
    if _RE_AADHAAR.match(ns):
        return "AADHAAR"
    if _RE_PHONE.match(ns) or _RE_PHONE_MASKED.match(ns):
        return "PHONE"
    if _RE_INSURANCE.match(ns):
        return "INSURANCE_ID"
    if _RE_MRN.match(ns):
        return "MRN"
    if _RE_PIN.match(ns) and hint in ("PIN", None) and _norm_key(key) not in ("value", "result", "amount", "beds", "occupied"):
        if hint == "PIN":
            return "PIN"
    if _RE_DATE.match(ns):
        if hint == "DOB":
            return "DOB"
        yr = _extract_year(ns)
        if yr is not None and yr < 2018 and _norm_key(key) not in _CLINICAL_DATE_KEYS:
            return "DOB"
        return None
    if _looks_like_address_composite(ns):
        return "ADDRESS"
    if _RE_HOUSE_VAL.match(ns):
        return "HOUSE"
    if _RE_LOCALITY_VAL.match(ns):
        return "LOCALITY"

    return hint


def sanitize_header_or_key(
    text: str,
    parent_key: str = "",
    stats: dict[str, int] | None = None,
    allow_person_key: bool = False,
) -> str:
    """Detect and redact PII inside a CSV header or JSON key."""
    s = str(text)
    if not s.strip():
        return s
    ns = _norm_digits(s)
    pk = _norm_key(parent_key)

    # Check direct key-level PII values (e.g. patient_index.json by_uhid / by_name)
    if _RE_MRN.match(ns):
        if stats is not None:
            stats["MRN"] = stats.get("MRN", 0) + 1
        return "[MRN]"
    if _RE_AADHAAR.match(ns) or _RE_AADHAAR_MASKED.match(ns):
        if stats is not None:
            stats["AADHAAR"] = stats.get("AADHAAR", 0) + 1
        return "[AADHAAR]"
    if _RE_ABHA.match(ns):
        if stats is not None:
            stats["ABHA_NUMBER"] = stats.get("ABHA_NUMBER", 0) + 1
        return "[ABHA_NUMBER]"
    if _RE_PHONE.match(ns) or _RE_PHONE_MASKED.match(ns):
        if stats is not None:
            stats["PHONE"] = stats.get("PHONE", 0) + 1
        return "[PHONE]"
    if _RE_PAN.match(ns.upper()):
        if stats is not None:
            stats["PAN"] = stats.get("PAN", 0) + 1
        return "[PAN]"
    if _RE_EMAIL.match(ns):
        if stats is not None:
            stats["EMAIL"] = stats.get("EMAIL", 0) + 1
        return "[EMAIL]"

    # Compound headers containing MRNs and/or patient names (e.g. "<MRN> <FirstName>")
    out = s
    out, n_mrn = re.subn(
        r"(?:[A-Z]{2,6}/\d{2,4}/\d{3,}|[A-Z]{2,6}-\d{4,}|[A-Z]{2,6}\d{2,4}-\d{3,}|[A-Z]{2,6}-[A-Z]{2,4}-\d{3,}|[A-Z]{2,5}-\d{2,4}-\d{3,}|[A-Z]{2,5}\d{5,}|[A-Z]{2,6}-\d{2,4}-\d{4,})",
        "[MRN]",
        out,
    )
    if n_mrn and stats is not None:
        stats["MRN"] = stats.get("MRN", 0) + n_mrn

    # If remaining key sits under a person-indexed map (by_name, by_beneficiary, ...) or looks like a person name
    rem = re.sub(r"\[[A-Z_]+\]", "", out).strip(" _-:")
    rem_cand = rem.replace("_", " ") if "_" in rem else rem
    if rem_cand and _looks_like_person_name(rem_cand, allow_synthetic_canary=(allow_person_key or pk in ("by_name", "patient_map", "names"))) and (
        allow_person_key
        or pk in ("by_name", "patient_map", "names")
        or n_mrn > 0
        or (len(rem_cand.split()) >= 2 and _norm_key(rem) not in _NON_PII_KEY_TOKENS)
    ):
        out = out.replace(rem, "[PATIENT_NAME]")
        if stats is not None:
            stats["PATIENT_NAME"] = stats.get("PATIENT_NAME", 0) + 1

    return out


def sanitize_headers(headers: list[str], stats: dict[str, int] | None = None) -> list[str]:
    """Sanitize CSV/TSV headers and resolve collisions deterministically."""
    seen: dict[str, int] = {}
    out: list[str] = []
    for h in headers:
        sh = sanitize_header_or_key(h, stats=stats)
        if sh in seen:
            seen[sh] += 1
            out.append(f"{sh}_{seen[sh]}")
        else:
            seen[sh] = 1
            out.append(sh)
    return out


def _put_unique_key(target: dict, key: str, val: Any) -> None:
    if key not in target:
        target[key] = val
        return
    idx = 2
    while f"{key}_{idx}" in target:
        idx += 1
    target[f"{key}_{idx}"] = val


# --------------------------------------------------------------------------- table & JSON processors
def process_table_rows(
    rows: list[list[str]],
    text_scrubber=None,
    stats: dict[str, int] | None = None,
) -> tuple[list[str], list[list[str]]]:
    """Infer per-column semantics via header parts, value patterns, and neighbor columns and redact PII."""
    header = rows[0]
    body = rows[1:]
    n_cols = len(header)
    schema = infer_table_schema(header, body)
    col_types = schema.col_types

    clean_header = sanitize_headers(header, stats=stats)
    clean_body: list[list[str]] = []
    for row in body:
        new_row: list[str] = []
        for c_idx, cell in enumerate(row):
            ctype = col_types[c_idx] if c_idx < n_cols else None
            if ctype is not None:
                cleaned = redact_structured_value(cell, ctype, stats=stats)
                new_row.append(str(cleaned))
            elif text_scrubber is not None:
                new_row.append(text_scrubber(cell))
            else:
                new_row.append(cell)
        clean_body.append(new_row)
    return clean_header, clean_body


def _is_fhir_bundle(data: Any) -> bool:
    return isinstance(data, dict) and data.get("resourceType") == "Bundle" and isinstance(data.get("entry"), list)


def _sanitize_fhir_bundle(bundle: dict, text_scrubber=None, stats: dict[str, int] | None = None) -> dict:
    """Sanitize FHIR R4 Bundle while preserving valid R4 types, syntax, and internal references."""
    out = dict(bundle)
    meta = out.get("meta")
    if isinstance(meta, dict):
        meta = dict(meta)
        src = meta.get("source")
        if isinstance(src, str) and src.startswith("#"):
            meta["source"] = "#redacted"
            if stats is not None:
                stats["MRN"] = stats.get("MRN", 0) + 1
        out["meta"] = meta

    new_entries = []
    for ent in out.get("entry", []):
        if not isinstance(ent, dict):
            new_entries.append(ent)
            continue
        ent_copy = dict(ent)
        res = ent_copy.get("resource")
        if not isinstance(res, dict):
            new_entries.append(ent_copy)
            continue
        res = dict(res)
        rt = res.get("resourceType")

        if rt == "Patient":
            if isinstance(res.get("text"), dict) and isinstance(res["text"].get("div"), str):
                div_txt = res["text"]["div"]
                # Scrub inner text between tags without breaking XHTML wrapper
                res["text"] = dict(res["text"])
                res["text"]["div"] = re.sub(
                    r">([^<]+)<",
                    lambda m: ">" + _scrub_fhir_narrative(m.group(1), stats) + "<",
                    div_txt,
                )
            if isinstance(res.get("identifier"), list):
                new_ids = []
                for item in res["identifier"]:
                    if isinstance(item, dict) and "value" in item:
                        it = dict(item)
                        sys_uri = str(it.get("system", "")).casefold()
                        val_str = str(it["value"])
                        if "healthid" in sys_uri or _RE_ABHA.match(_norm_digits(val_str)):
                            it["value"] = "[ABHA_NUMBER]"
                            if stats is not None:
                                stats["ABHA_NUMBER"] = stats.get("ABHA_NUMBER", 0) + 1
                        else:
                            it["value"] = "[MRN]"
                            if stats is not None:
                                stats["MRN"] = stats.get("MRN", 0) + 1
                        new_ids.append(it)
                    else:
                        new_ids.append(item)
                res["identifier"] = new_ids
            if isinstance(res.get("name"), list):
                new_names = []
                for nm in res["name"]:
                    if isinstance(nm, dict):
                        nm_c = dict(nm)
                        if "text" in nm_c and isinstance(nm_c["text"], str):
                            nm_c["text"] = "[PATIENT_NAME]"
                        if "family" in nm_c and isinstance(nm_c["family"], str):
                            nm_c["family"] = "[PATIENT_NAME]"
                        if "given" in nm_c and isinstance(nm_c["given"], list):
                            nm_c["given"] = ["[PATIENT_NAME]" for _ in nm_c["given"]]
                        if stats is not None:
                            stats["PATIENT_NAME"] = stats.get("PATIENT_NAME", 0) + 1
                        new_names.append(nm_c)
                    else:
                        new_names.append(nm)
                res["name"] = new_names
            if isinstance(res.get("telecom"), list):
                new_tel = []
                for tc in res["telecom"]:
                    if isinstance(tc, dict) and isinstance(tc.get("value"), str):
                        tc_c = dict(tc)
                        kind = "EMAIL" if tc_c.get("system") == "email" or "@" in tc_c["value"] else "PHONE"
                        tc_c["value"] = f"[{kind}]"
                        if stats is not None:
                            stats[kind] = stats.get(kind, 0) + 1
                        new_tel.append(tc_c)
                    else:
                        new_tel.append(tc)
                res["telecom"] = new_tel
            if isinstance(res.get("birthDate"), str) and len(res["birthDate"]) >= 4:
                # Preserve valid FHIR date (YYYY) without exposing full DOB
                res["birthDate"] = res["birthDate"][:4]
                if stats is not None:
                    stats["DOB"] = stats.get("DOB", 0) + 1
            if isinstance(res.get("address"), list):
                new_addrs = []
                for addr in res["address"]:
                    if isinstance(addr, dict):
                        ad = dict(addr)
                        if isinstance(ad.get("line"), list):
                            ad["line"] = [scrub_composite_address(str(x), stats) for x in ad["line"]]
                        if isinstance(ad.get("city"), str) and ad["city"].strip():
                            ad["city"] = "[VILLAGE]"
                            if stats is not None:
                                stats["VILLAGE"] = stats.get("VILLAGE", 0) + 1
                        if isinstance(ad.get("postalCode"), str) and ad["postalCode"].strip():
                            ad["postalCode"] = "[PIN]"
                            if stats is not None:
                                stats["PIN"] = stats.get("PIN", 0) + 1
                        new_addrs.append(ad)
                    else:
                        new_addrs.append(addr)
                res["address"] = new_addrs
            if isinstance(res.get("contact"), list):
                new_contacts = []
                for ct in res["contact"]:
                    if isinstance(ct, dict):
                        ct_c = dict(ct)
                        if isinstance(ct_c.get("name"), dict):
                            nm_c = dict(ct_c["name"])
                            if isinstance(nm_c.get("text"), str):
                                nm_c["text"] = "[RELATIVE_NAME]"
                            if isinstance(nm_c.get("family"), str):
                                nm_c["family"] = "[RELATIVE_NAME]"
                            if isinstance(nm_c.get("given"), list):
                                nm_c["given"] = ["[RELATIVE_NAME]" for _ in nm_c["given"]]
                            ct_c["name"] = nm_c
                            if stats is not None:
                                stats["RELATIVE_NAME"] = stats.get("RELATIVE_NAME", 0) + 1
                        new_contacts.append(ct_c)
                    else:
                        new_contacts.append(ct)
                res["contact"] = new_contacts
            if isinstance(res.get("extension"), list):
                new_exts = []
                for ext in res["extension"]:
                    if isinstance(ext, dict) and isinstance(ext.get("valueString"), str):
                        ex_c = dict(ext)
                        url = str(ex_c.get("url", "")).casefold()
                        if any(w in url for w in ("mother", "father", "spouse", "guardian", "relative", "parent", "kin")):
                            ex_c["valueString"] = "[RELATIVE_NAME]"
                            if stats is not None:
                                stats["RELATIVE_NAME"] = stats.get("RELATIVE_NAME", 0) + 1
                        elif any(w in url for w in ("name", "patient", "phone", "aadhaar", "abha", "address")):
                            ex_c["valueString"] = "[PATIENT_NAME]"
                            if stats is not None:
                                stats["PATIENT_NAME"] = stats.get("PATIENT_NAME", 0) + 1
                        new_exts.append(ex_c)
                    else:
                        new_exts.append(ext)
                res["extension"] = new_exts

        elif rt == "Composition":
            if isinstance(res.get("subject"), dict) and isinstance(res["subject"].get("display"), str):
                subj = dict(res["subject"])
                subj["display"] = "[PATIENT_NAME]"
                res["subject"] = subj
                if stats is not None:
                    stats["PATIENT_NAME"] = stats.get("PATIENT_NAME", 0) + 1
            # Composition.author holds Practitioner doctor display -> preserve!

        elif rt == "Encounter":
            if isinstance(res.get("identifier"), list):
                new_ids = []
                for item in res["identifier"]:
                    if isinstance(item, dict) and isinstance(item.get("value"), str):
                        it = dict(item)
                        it["value"] = "[MRN]"
                        if stats is not None:
                            stats["MRN"] = stats.get("MRN", 0) + 1
                        new_ids.append(it)
                    else:
                        new_ids.append(item)
                res["identifier"] = new_ids

        elif rt == "DocumentReference":
            if isinstance(res.get("content"), list):
                new_contents = []
                for item in res["content"]:
                    if isinstance(item, dict) and isinstance(item.get("attachment"), dict):
                        it = dict(item)
                        att = dict(it["attachment"])
                        if isinstance(att.get("data"), str):
                            att["data"] = sanitize_base64_string(att["data"], text_scrubber)
                        it["attachment"] = att
                        new_contents.append(it)
                    else:
                        new_contents.append(item)
                res["content"] = new_contents

        ent_copy["resource"] = res
        new_entries.append(ent_copy)
    out["entry"] = new_entries
    return out


def _scrub_fhir_narrative(text: str, stats: dict[str, int] | None = None) -> str:
    """Clean Patient.text.div narrative like 'Ravi Verma, 34 yrs, male, HOSP/27/001234'."""
    parts = [p.strip() for p in text.split(",")]
    cleaned = []
    for idx, part in enumerate(parts):
        ns = _norm_digits(part)
        if _RE_MRN.match(ns):
            cleaned.append("[MRN]")
            if stats is not None:
                stats["MRN"] = stats.get("MRN", 0) + 1
        elif idx == 0 and _looks_like_person_name(part):
            cleaned.append("[PATIENT_NAME]")
            if stats is not None:
                stats["PATIENT_NAME"] = stats.get("PATIENT_NAME", 0) + 1
        else:
            cleaned.append(part)
    return ", ".join(cleaned)


def sanitize_json_data(data: Any, text_scrubber=None, stats: dict[str, int] | None = None) -> Any:
    """Iteratively sanitize arbitrary JSON (supports deep nesting, dict neighbor context, and array-of-dict inference)."""
    if _is_fhir_bundle(data):
        return _sanitize_fhir_bundle(data, text_scrubber=text_scrubber, stats=stats)

    # Iterative stack-based traversal so 1,500+ deep JSON never triggers RecursionError
    root_wrapper: dict[str, Any] = {"root": data}
    # stack items: (parent_container, key_or_idx, value, parent_key, inferred_type_override)
    stack: list[tuple[Any, Any, Any, str, str | None]] = [(root_wrapper, "root", data, "", None)]

    while stack:
        parent, slot, val, parent_key, forced_type = stack.pop()

        if isinstance(val, dict):
            new_dict: dict[str, Any] = {}
            parent[slot] = new_dict

            pk_name = slot[0] if isinstance(slot, tuple) else ("" if slot == "root" else str(slot))
            d_schema = infer_dict_schema(val, parent_key=pk_name)
            items = list(val.items())
            for k, child in reversed(items):
                sk = sanitize_header_or_key(
                    k,
                    parent_key=pk_name,
                    stats=stats,
                    allow_person_key=d_schema.is_person_map_dict,
                )
                child_forced = forced_type
                if isinstance(child, dict) and k in d_schema.sub_dict_roles:
                    child_forced = "RELATIVE_NAME" if d_schema.sub_dict_roles[k] == "RELATIVE_NAME_OBJ" else "PATIENT_NAME"
                elif not isinstance(child, (dict, list)) and child_forced is None:
                    child_forced = d_schema.field_types.get(str(k))
                stack.append((new_dict, (k, sk), child, pk_name, child_forced))

        elif isinstance(val, list):
            new_list = [None] * len(val)
            if isinstance(parent, dict) and isinstance(slot, tuple):
                _orig_k, san_k = slot
                key_to_use = _unique_key_name(parent, san_k)
                parent[key_to_use] = new_list
            else:
                parent[slot] = new_list

            # If list of dicts, run unified table-level neighbor + value schema inference across items
            col_map: dict[str, str | None] = {}
            if val and all(isinstance(x, dict) for x in val):
                all_subkeys: list[str] = []
                seen_sk = set()
                for row in val:
                    for rk in row.keys():
                        if rk not in seen_sk:
                            seen_sk.add(rk)
                            all_subkeys.append(rk)
                pk = slot[0] if isinstance(slot, tuple) else str(slot)
                scalar_keys = [rk for rk in all_subkeys if not any(isinstance(row.get(rk), (dict, list)) for row in val)]
                if scalar_keys:
                    mat = [[row.get(rk, "") for rk in scalar_keys] for row in val]
                    ts = infer_table_schema(scalar_keys, mat, parent_key=pk)
                    for rk, ct in zip(scalar_keys, ts.col_types):
                        col_map[rk] = ct
                for rk in all_subkeys:
                    if rk not in col_map:
                        col_vals = [row.get(rk) for row in val if rk in row]
                        if col_vals and not any(isinstance(x, (dict, list)) for x in col_vals):
                            col_map[rk] = infer_column_type(col_vals, header=rk, parent_key=pk)

            pk = slot[0] if isinstance(slot, tuple) else str(slot)
            for i in range(len(val) - 1, -1, -1):
                item = val[i]
                if isinstance(item, dict):
                    new_item: dict[str, Any] = {}
                    new_list[i] = new_item
                    d_schema = infer_dict_schema(item, parent_key=pk)
                    for rk, rchild in reversed(list(item.items())):
                        r_sk = sanitize_header_or_key(
                            rk,
                            parent_key=pk,
                            stats=stats,
                            allow_person_key=d_schema.is_person_map_dict,
                        )
                        child_forced = forced_type
                        if isinstance(rchild, dict) and rk in d_schema.sub_dict_roles:
                            child_forced = "RELATIVE_NAME" if d_schema.sub_dict_roles[rk] == "RELATIVE_NAME_OBJ" else "PATIENT_NAME"
                        elif not isinstance(rchild, (dict, list)) and child_forced is None:
                            child_forced = col_map.get(rk) or d_schema.field_types.get(str(rk))
                        stack.append((new_item, (rk, r_sk), rchild, pk, child_forced))
                else:
                    stack.append((new_list, i, item, pk, None))

        else:
            # Scalar leaf
            orig_key = slot[0] if isinstance(slot, tuple) else str(slot)
            ctype = forced_type or _classify_scalar_by_value_and_key(val, orig_key, parent_key)
            if ctype is not None:
                cleaned = redact_structured_value(val, ctype, stats=stats)
            elif isinstance(val, str):
                # Decode & sanitize base64 text attachments (e.g. scanned_consent_form.json)
                b64_checked = sanitize_base64_string(val, text_scrubber)
                if b64_checked != val:
                    cleaned = b64_checked
                elif text_scrubber is not None:
                    cleaned = text_scrubber(val)
                else:
                    cleaned = val
            else:
                cleaned = val

            if isinstance(parent, dict) and isinstance(slot, tuple):
                _put_unique_key(parent, slot[1], cleaned)
            else:
                parent[slot] = cleaned

        # Fix dict child attachment when parent is dict and slot is (orig_k, san_k)
        if isinstance(val, dict) and isinstance(parent, dict) and isinstance(slot, tuple):
            # Move from tuple key into deterministic sanitized string key
            attached = parent.pop(slot)
            _put_unique_key(parent, slot[1], attached)

    return root_wrapper["root"]


def _unique_key_name(d: dict, key: str) -> str:
    if key not in d:
        return key
    idx = 2
    while f"{key}_{idx}" in d:
        idx += 1
    return f"{key}_{idx}"

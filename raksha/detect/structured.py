"""Step A: Value-first structured field classifier, header/key sanitizer, and composite address scrubber."""
from __future__ import annotations

import re
import unicodedata
from typing import Any

from raksha.export.filenames import sanitize_base64_string

# Devanagari -> ASCII digit mapping (used locally in Step A before Step C adds full offset-mapped normalization)
_DEVA_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")

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
    r"SNJ/\d{2}/\d{4,}"
    r"|JJH-\d{4,}"
    r"|UHID-\d{4,}"
    r"|DHH\d{4}-\d{4,}"
    r"|DHH-IPD-\d{3,}"
    r"|IPD-\d{4}-\d{3,}"
    r"|IP/\d{2}/\d{3,}"
    r"|ADM\d{4,}"
    r"|[A-Z]{2,5}[/-]\d{2,4}[/-]\d{4,}"
    r")$"
)
_RE_PIN = re.compile(r"^[1-8]\d{5}$")
_RE_DATE = re.compile(
    r"^(?:(\d{4})[-/](\d{1,2})[-/](\d{1,2})|(\d{1,2})[-/.](\d{1,2})[-/.](\d{4}))$"
)
_RE_REL_PREFIX = re.compile(r"^(?:[SDWC]/[Oo]|[SDWC]\s*/\s*[Oo])\s+(.+)$")

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
    "department", "ward", "wards", "bed", "beds", "occupied", "dengue_cases", "week",
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
    k = re.sub(r"[^a-z0-9\u0900-\u097f]+", "_", k).strip("_")
    return k


def _extract_year(s: str) -> int | None:
    m = _RE_DATE.match(_norm_digits(s))
    if not m:
        return None
    return int(m.group(1) or m.group(6))


_PROSE_OR_CLINICAL_STOPWORDS = _CLINICAL_OR_NON_NAME_WORDS | {
    "oxygen", "given", "en", "route", "for", "with", "and", "the", "by", "called",
    "from", "to", "fever", "pain", "days", "stable", "normal", "reviewed", "hospital",
    "district", "ward", "bed", "protocol", "report", "allergic", "sulfa", "drugs",
    "arrival", "sent", "emergency", "referred", "asha", "worker", "village", "visit",
    "next", "verify", "identity", "old", "card", "merged", "temp", "diet", "salt",
    "low", "diabetic", "advice", "morning", "evening", "condition", "discharged",
}
# Note: exclude "asha" from prose stopwords because "Asha" is a common Indian first name;
# "ASHA worker" is protected in context/clinical spans instead.
_PROSE_OR_CLINICAL_STOPWORDS.discard("asha")


def _looks_like_person_name(val: str) -> bool:
    s = unicodedata.normalize("NFKC", val).strip()
    if not s or len(s) > 60:
        return False
    if any(ch.isdigit() for ch in s) or "@" in s or "/" in s or "\\" in s:
        return False
    low = s.casefold()
    if low in _PROSE_OR_CLINICAL_STOPWORDS:
        return False
    if low.startswith(("dr.", "dr ", "डॉ", "mr.", "mrs.", "ms.")):
        return False
    toks = [t for t in re.split(r"[\s.]+", s) if t]
    if not (1 <= len(toks) <= 4):
        return False
    has_latin = False
    has_upper_latin = False
    for t in toks:
        if not re.fullmatch(r"[A-Za-z\u0900-\u097F'-]{1,25}", t):
            return False
        if t.casefold() in _PROSE_OR_CLINICAL_STOPWORDS:
            return False
        if re.search(r"[A-Za-z]", t):
            has_latin = True
            if t[0].isupper():
                has_upper_latin = True
    # Reject all-lowercase Latin prose like "oxygen given en route"
    if has_latin and not has_upper_latin:
        return False
    return True


def _looks_like_address_composite(val: str) -> bool:
    s = unicodedata.normalize("NFKC", val)
    return bool(
        re.search(
            r"(?i)(?:h\.?\s*no\.?|house\s+\d|makan\s*no|मकान\s*नं|ward\s+\d+|mohalla|मोहल्ला|"
            r"tehsil|तहसील|distt\.?|जिला|near\s+|opp\.|के\s+पास|के\s+पीछे|vill\.|gram\s+|ग्राम\s+|c/o\s+)",
            s,
        )
    )


# --------------------------------------------------------------------------- composite address scrubber
_ADDR_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("RELATIVE_NAME", re.compile(r"(?i)(?<=C/O\s)([A-Z\u0900-\u0963\u0970-\u097F][A-Za-z\u0900-\u0963\u0970-\u097F.\-']+(?:\s+[A-Z\u0900-\u0963\u0970-\u097F][A-Za-z\u0900-\u0963\u0970-\u097F.\-']+){0,3})(?=,|$)")),
    ("HOUSE", re.compile(r"(?i)\b(?:H\.?\s*No\.?|House(?:\s+No\.?)?|Makan\s+No\.?)\s*[\dA-Za-z/-]+")),
    ("HOUSE", re.compile(r"मकान\s*(?:नं\.?|सं\.?)\s*[\d०-९A-Za-z/-]+")),
    ("LOCALITY", re.compile(r"(?i)\bWard\s*(?:No\.?)?\s*\d+")),
    ("LOCALITY", re.compile(r"वार्ड\s*(?:नं\.?)?\s*[\d०-९]+")),
    ("LOCALITY", re.compile(r"(?i)\bMohalla\s+[^,\n]+")),
    ("LOCALITY", re.compile(r"मोहल्ला\s+[^,\n]+")),
    ("LANDMARK", re.compile(r"(?i)\b(?:near|opp\.?|opposite|behind|beside)\s+[^,\n]+")),
    ("LANDMARK", re.compile(r"[^,\n]+\s+(?:के\s+पास|के\s+पीछे|के\s+सामने)")),
    ("VILLAGE", re.compile(r"(?i)\b(?:Vill\.?|Village|Gram)\s+[^,\n\-]+")),
    ("VILLAGE", re.compile(r"ग्राम\s+[^,\n\-]+")),
    ("VILLAGE", re.compile(r"(?i)(?<=,\s)(?!(?:H\.?\s*No|House|Makan|Ward|Mohalla|near|opp|Tehsil|Distt|C/O)\b)([A-Z\u0900-\u0963\u0970-\u097F][^\n,]+?)(?=,\s*(?:Tehsil|तहसील)\b)")),
    ("PIN", re.compile(r"(?<!\d)[1-8]\d{5}(?!\d)")),
]


def scrub_composite_address(text: str, stats: dict[str, int] | None = None) -> str:
    """Scrub House, Locality, Landmark, Village, PIN, and C/O relative inside composite address text while preserving Tehsil/District."""
    out = text
    for label, rx in _ADDR_PATTERNS:
        out, n = rx.subn(f"[{label}]", out)
        if stats is not None and n:
            stats[label] = stats.get(label, 0) + n
    return out


_FREE_TEXT_HEADERS = {
    "remarks", "remark", "notes", "note", "comment", "comments", "description", "summary", "text",
}


# --------------------------------------------------------------------------- header / key synonym mapping
def _hint_from_header(header_raw: str, parent_key: str = "") -> str | None:
    k = _norm_key(header_raw)
    pk = _norm_key(parent_key)
    if not k:
        return None
    if k in _NON_PII_KEY_TOKENS or k in _FREE_TEXT_HEADERS:
        return "NON_PII"
    if k in _CLINICAL_DATE_KEYS:
        return "CLINICAL_DATE"
    if pk in ("facility", "hospital", "organization", "lab", "panels", "diagnosis", "rx", "vitals"):
        return "NON_PII"

    ktoks = set(k.split("_"))

    # Relative / Guardian (use token membership for short Hindi words like pita/pati!)
    if ktoks & {"relative", "guardian", "father", "husband", "mother", "spouse", "attendant", "pita", "pati", "kin", "witnessed"} or any(
        w in k for w in ("relative", "guardian", "father", "husband", "mother", "spouse", "attendant", "witnessed")
    ):
        return "RELATIVE_NAME"
    # Patient name
    if k in ("patient_name", "patientname", "pt_name", "pname", "naam", "firstname", "first_name", "lastname", "last_name", "fname", "lname", "mariz_ka_naam", "rogi_ka_naam", "patient"):
        return "PATIENT_NAME"
    if k in ("name", "local", "first", "last", "middle", "given", "family") and pk in ("", "patient", "patients", "name", "raised_by", "meta"):
        return "PATIENT_NAME"
    if k in ("name", "first", "last", "given", "family") and pk in ("guardian", "contact", "relative"):
        return "RELATIVE_NAME"
    # Identifiers
    if any(w in k for w in ("aadhar", "aadhaar", "adhaar", "addhar", "uidai")):
        return "AADHAAR"
    if "abha_addr" in k or "phr_addr" in k or "health_addr" in k:
        return "ABHA_ADDRESS"
    if any(w in k for w in ("abha", "health_id", "healthid", "ndhm", "abdm")):
        return "ABHA_NUMBER"
    if "pan" in ktoks or k.endswith("_pan") or k.startswith("pan_") or k == "pancard":
        return "PAN"
    if any(w in k for w in ("pmjay", "policy", "insurance", "ins_id", "bima")):
        return "INSURANCE_ID"
    if any(w in k for w in ("uhid", "mrn", "patient_ref", "patient_id", "patientid", "reg_no", "regno", "ipd", "adm_no", "panchikaran")):
        return "MRN"
    if (ktoks & {"mobile", "mob", "phone", "phn", "tel", "cell", "whatsapp", "callback", "sampark"}) or any(
        w in k for w in ("mobile", "phone", "whatsapp", "callback", "sampark")
    ):
        return "PHONE"
    if any(w in k for w in ("email", "e_mail")):
        return "EMAIL"
    # DOB
    if (ktoks & {"dob", "birth", "bdate", "janm", "janam", "जन्म"}) or any(w in k for w in ("dob", "birth", "bdate", "janm", "janam", "जन्म")):
        return "DOB"
    # Address components
    if ktoks & {"pin", "pincode", "zip", "zipcode", "postal", "postalcode"}:
        return "PIN"
    if (ktoks & {"house", "makan", "hno"}) or "h_no" in k or "makan_no" in k or "house_no" in k:
        return "HOUSE"
    if ktoks & {"locality", "mohalla", "colony", "basti", "tola"}:
        return "LOCALITY"
    if "landmark" in k:
        return "LANDMARK"
    if ktoks & {"village", "gaon", "gram"}:
        return "VILLAGE"
    if k == "city" and pk in ("", "address"):
        return "VILLAGE"
    if any(w in k for w in ("address", "pata", "street", "address_note")) or (k == "line" and pk == "address"):
        return "ADDRESS"
    if k in ("source_path", "filepath", "file_path", "export_path"):
        return "META_PATH"
    return None


def infer_column_type(values: list[Any], header: str = "", parent_key: str = "") -> str | None:
    """Infer PII semantic category from VALUES FIRST, using header/key synonyms as hints."""
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
    # Composite address
    if counts.get("ADDRESS", 0) >= max(1, int(0.4 * n)):
        return "ADDRESS"
    # Relative prefix in column values
    if counts.get("RELATIVE_NAME", 0) >= max(1, int(0.4 * n)):
        return "RELATIVE_NAME"

    # Header-guided resolution for names / geography / ambiguous IDs
    if hint in ("PATIENT_NAME", "RELATIVE_NAME", "VILLAGE", "HOUSE", "LOCALITY", "LANDMARK", "PIN", "ADDRESS", "MRN", "INSURANCE_ID", "PHONE", "EMAIL", "PAN", "AADHAAR", "ABHA_NUMBER", "ABHA_ADDRESS", "DOB"):
        return hint

    # Value-only name inference when header is unknown (require >=2 tokens on most values to avoid drug/test false positives)
    if counts.get("NAME_LIKE", 0) >= max(2, int(0.7 * n)):
        multi_tok = sum(1 for v in non_empty if len(str(v).split()) >= 2)
        if multi_tok >= max(1, int(0.5 * len(non_empty))):
            return "PATIENT_NAME"

    return None


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
        # Replace patient-specific folder segments like Kavita_Pandey_064371
        out = re.sub(r"([\\/])[A-Za-z]+_[A-Za-z]+_\d{4,}([\\/])", r"\1[PATIENT_NAME]_[MRN]\2", s)
        out = re.sub(r"([\\/])[A-Za-z]+_[A-Za-z]+([\\/])", r"\1[PATIENT_NAME]\2", out)
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
    """Inspect a single JSON scalar (including int/float!) by value first, then key context."""
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

    return hint


def sanitize_header_or_key(text: str, parent_key: str = "", stats: dict[str, int] | None = None) -> str:
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

    # Compound headers like "JJH-0077112 Zainab"
    out = s
    out, n_mrn = re.subn(
        r"(?:SNJ/\d{2}/\d{4,}|JJH-\d{4,}|UHID-\d{4,}|DHH\d{4}-\d{4,}|DHH-IPD-\d{3,}|IPD-\d{4}-\d{3,}|IP/\d{2}/\d{3,}|ADM\d{4,}|[A-Z]{2,5}[/-]\d{2,4}[/-]\d{4,})",
        "[MRN]",
        out,
    )
    if n_mrn and stats is not None:
        stats["MRN"] = stats.get("MRN", 0) + n_mrn

    # If the remaining key is under a name-indexed dict or holds a person-name-like string
    rem = re.sub(r"\[[A-Z_]+\]", "", out).strip(" _-:")
    if rem and _looks_like_person_name(rem) and (
        pk in ("by_name", "patient_map", "names")
        or n_mrn > 0
        or (len(rem.split()) >= 2 and _norm_key(rem) not in _NON_PII_KEY_TOKENS)
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
    """Infer per-column semantics from values + header hints and redact structured cells."""
    header = rows[0]
    body = rows[1:]
    n_cols = len(header)
    col_types: list[str | None] = []
    for c_idx in range(n_cols):
        col_vals = [r[c_idx] if c_idx < len(r) else "" for r in body]
        col_types.append(infer_column_type(col_vals, header=header[c_idx]))

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
    """Clean Patient.text.div narrative like 'Shabana Bano, 27 yrs, female, SNJ/26/064545'."""
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
    """Iteratively sanitize arbitrary JSON (supports deep nesting and array-of-dict column inference)."""
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
            if isinstance(parent, dict):
                # Re-attach new_dict; key already sanitized when pushed
                parent[slot] = new_dict
            else:
                parent[slot] = new_dict

            # Pre-collect keys in reverse order so pop() processes in original deterministic key order
            pk_name = slot[0] if isinstance(slot, tuple) else ("" if slot == "root" else str(slot))
            items = list(val.items())
            for k, child in reversed(items):
                sk = sanitize_header_or_key(k, parent_key=pk_name, stats=stats)
                stack.append((new_dict, (k, sk), child, pk_name, None))

        elif isinstance(val, list):
            new_list = [None] * len(val)
            if isinstance(parent, dict) and isinstance(slot, tuple):
                orig_k, san_k = slot
                _put_unique_key_dict_slot = san_k
                # Resolve collision at insertion time
                key_to_use = _unique_key_name(parent, san_k)
                parent[key_to_use] = new_list
            elif isinstance(parent, dict):
                parent[slot] = new_list
            else:
                parent[slot] = new_list

            # If list of dicts, infer column types across all items for each subkey
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
                for rk in all_subkeys:
                    col_vals = [row.get(rk) for row in val if rk in row]
                    col_map[rk] = infer_column_type(col_vals, header=rk, parent_key=pk)

            for i in range(len(val) - 1, -1, -1):
                item = val[i]
                pk = slot[0] if isinstance(slot, tuple) else str(slot)
                if isinstance(item, dict):
                    # Apply column map to scalar children via a small helper wrapper
                    new_item: dict[str, Any] = {}
                    new_list[i] = new_item
                    for rk, rchild in reversed(list(item.items())):
                        r_sk = sanitize_header_or_key(rk, parent_key=pk, stats=stats)
                        stack.append((new_item, (rk, r_sk), rchild, pk, col_map.get(rk)))
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

"""Dual-signal schema-agnostic column/key profiler & date-order inference.

Two independent signals decide what a column (or JSON key) holds:

1. the **header / key label** (bilingual synonyms, relation & doctor cues, fuzzy matching), and
2. the **values** (shape of phones / Aadhaar / ABHA / PAN / PIN / IDs / dates, relation prefixes, addresses,
   and, once the clinical lexicon is learned, name-like and village-like proper nouns).

The value signal is what keeps detection working on headers never seen before (`holder`, `abode`, `F7`).
"""
from __future__ import annotations

import re
from collections import Counter
from functools import lru_cache
from typing import Any

from rapidfuzz import fuzz

# All Indic-script, Arabic-Indic and fullwidth digits -> ASCII
_DIGIT_MAP = {z + i: 48 + i for z in (0x0966, 0x09E6, 0x0A66, 0x0AE6, 0x0B66, 0x0BE6, 0x0C66, 0x0CE6, 0x0D66, 0x0660, 0x06F0, 0xFF10)
              for i in range(10)}

_ROLE_SYNONYMS: dict[str, tuple[str, ...]] = {
    "PATIENT_NAME": (
        "patient name", "patient's name", "patients name", "naam", "name", "full name", "fullname", "nm",
        "pt name", "pt_name", "pt fullname", "mrz naam", "mrz nm", "first name", "last name", "given", "family",
        "firstname", "lastname", "first", "last", "surname", "forename", "subjectname", "patient", "pt",
        "pt first", "pt last", "beneficiary", "beneficiary name", "insured", "member", "member name", "client",
        "rogi", "rogi ka naam", "carerecipient", "name of patient", "नाम", "मरीज का नाम", "मरीज़ का नाम", "रोगी नाम", "रोगी का नाम",
    ),
    "RELATIVE_NAME": (
        "relative name", "relative", "guardian", "guardian full", "pita pati ka naam", "pita ka naam",
        "pati ka naam", "father name", "husband name", "caregiver", "caregiver name", "relation name",
        "attendant", "kin", "nok", "rel nm", "relnm", "sambandhi", "spouse",
        "पिता का नाम", "पति का नाम", "अभिभावक",
    ),
    "RELATION": ("relation", "relationship", "रिश्ता"),
    "PHONE": (
        "mobile", "mob", "mob no", "mobile no", "contact", "contact#", "contact num", "contact no", "phone",
        "callback", "callback no", "telecom", "sampark", "sampark num", "tel", "cell", "cell phone", "ph", "ph#",
        "dooraabhash", "whatsapp", "फोन", "फोन नंबर", "मोबाइल", "दूरभाष",
    ),
    "EMAIL": ("email", "e-mail", "mail"),
    "AADHAAR": (
        "aadhaar", "aadhar", "aadhaar no", "aadhar no", "aadhr num", "aadhaar sankhya", "uid", "uid 12", "uid num",
        "uidai", "national id", "natl id", "national id num", "govt id", "govtid a", "आधार", "आधार संख्या",
    ),
    "ABHA": ("abha", "abha no", "abha number", "abha address", "health id"),
    "PAN": ("pan", "payer pan", "pan no"),
    "INSURANCE_ID": ("pmjay", "pmjay id", "policy", "policy no", "insurance", "tpa", "scheme id"),
    "MRN": (
        "uhid", "reg no", "reg code", "patientid", "patient id", "patient ref", "mrno", "mr no",
        "mrn", "mrn link", "ipd", "ipd no", "opd no", "hospital uhid", "hosp id", "hosp ref", "hosp no",
        "pt code", "card id", "subject ref", "care id", "panjikaran", "panjikaran kramank",
        "bill no", "sample id", "पंजीकरण", "पंजीकरण सं", "आईपीडी",
    ),
    "DOB": ("dob", "birth date", "birthdate", "birth dt", "born", "dt of birth", "date of birth", "janm tithi",
            "janm din", "janm dinank", "जन्म तिथि"),
    "AGE": ("age", "umar", "umr", "age sex", "yrs", "age yr", "aayu", "आयु", "उम्र"),
    "GENDER": ("gender", "sex", "ling", "g", "लिंग"),
    "ADDRESS": ("address", "addr", "addr1", "pata", "pata full", "street", "line", "address note", "res addr",
                "abode", "nivas", "nivas sthan", "पता", "निवास", "निवास पता"),
    "VILLAGE": (
        "village", "vill", "village mohalla", "gram mohalla", "gaon", "gaon wadi", "gaon name", "gram", "city",
        "kasba", "rural unit", "locality", "mohalla", "ग्राम", "ग्राम मोहल्ला", "गाँव",
    ),
    "TEHSIL": ("tehsil", "tahsil", "taluka", "tal", "block", "तहसील"),
    "DISTRICT": ("district", "dist", "distt", "jila", "zila", "zilla", "जिला"),
    "PIN": ("pin", "pin code", "pincode", "postal", "postal 6", "postalcode", "postcode", "area pin", "zip",
            "पिन", "पिन कोड"),
    "DATE": (
        "date", "reg date", "dt reg", "registered", "registered on", "visit date", "admission date", "admitted",
        "discharge date", "discharged", "collected on", "reported on", "collected", "reported", "tithi", "तारीख",
    ),
    "DOCTOR": ("doctor", "ref doctor", "referred by", "ref by", "consultant", "author", "reviewed by", "referring"),
    "DIAGNOSIS": ("diagnosis", "icd", "condition", "procedure", "complaints", "history", "निदान"),
    "MEDICATION": ("drug", "rx", "medicine", "strength", "frequency", "duration", "package"),
    "LAB": ("test", "result", "value", "unit", "ref range", "range", "loinc", "vitals", "bp"),
    "AMOUNT": ("amount", "amount inr", "amt", "inr", "rs", "cost", "fee", "fees", "charges", "price", "total",
               "bill amount", "paid", "payable"),
    "NOTES": ("notes", "remarks", "comment", "comments"),
    "METADATA_PATH": ("source path", "export path", "exported by", "file path"),
}

# Header words that override the generic "name" cue
_REL_CUES = {"father", "husband", "mother", "wife", "spouse", "guardian", "kin", "nok", "relative", "attendant",
             "pita", "pati", "mata", "sambandhi", "caregiver", "पिता", "पति", "माता", "संबंधी", "अभिभावक"}
_DOC_CUES = {"doctor", "dr", "consultant", "physician", "referring", "referred", "surgeon"}
# Short / generic labels: only an exact header match counts ("Patient Age" is not a name column)
_EXACT_ONLY = {"patient", "pt", "given", "family", "first", "last", "member", "client", "insured", "name", "nm",
               "naam", "beneficiary", "rogi", "g", "tal", "ph", "cell", "born", "yrs", "line", "city", "locality",
               "registered", "admitted", "discharged", "reported", "collected", "kin", "spouse", "relative"}
_NON_PERSON_CUES = {"hospital", "facility", "drug", "medicine", "test", "ward", "department", "dept", "unit", "lab",
                    "organization", "organisation", "org", "clinic", "scheme", "package", "procedure", "diagnosis",
                    "village", "district", "tehsil", "state", "city", "file", "sheet", "report", "age", "sex"}
_AMOUNT_CUES = {"amount", "amt", "inr", "rs", "cost", "fee", "fees", "charges", "price", "total", "paid", "payable"}

_PHONE_DIGITS = re.compile(r"^(?:\+?91|0091|0)?[6-9]\d{9}$")
_PAN_SHAPE = re.compile(r"^[A-Z]{5}\d{4}[A-Z]$")
_PIN_SHAPE = re.compile(r"^[1-8]\d{5}$")
_AGE_SHAPE = re.compile(r"^(?:\d{1,3}(?:\s*(?:yrs?|years?|वर्ष|y))?(?:\s*/\s*[MF])?)$", re.I)
_DATE_SLASH = re.compile(r"^(\d{1,2})[./-](\d{1,2})[./-]((?:19|20)\d{2})$")
_DATE_ISO = re.compile(r"^(?:19|20)\d{2}-\d{2}-\d{2}")
_DATE_MON = re.compile(r"^\d{1,2}[- ](?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*[- ,]+(?:19|20)\d{2}$", re.I)
_EMAIL_SHAPE = re.compile(r"^[\w.+-]+@[\w-]+\.[\w.]+$")
_ABHA_ADDR = re.compile(r"^[\w.]{3,}@(?:abdm|sbx|ndhm)$", re.I)
_ID_SHAPE = re.compile(r"^[A-Z]{2,6}(?:[/_.|\-#][A-Z]{1,6}){0,3}(?:[/_.|\-#]?\d{2,10}){1,4}$")
_REL_PREFIX = re.compile(r"^\s*(?:S/o|D/o|W/o|C/o|son of|daughter of|wife of|husband of|पुत्र|पुत्री|पत्नी|पति|पिता)\b", re.I)
# "C/O fever and weakness" is a complaint, not "care of <relative>"
_COMPLAINT_WORDS = {
    "fever", "pain", "cough", "cold", "weakness", "vomiting", "nausea", "headache", "giddiness", "dizziness",
    "breathlessness", "swelling", "burning", "loose", "abdominal", "chest", "body", "joint", "back", "fatigue",
    "palpitations", "sob", "dyspnoea", "dyspnea", "diarrhoea", "diarrhea", "rash", "itching", "bleeding", "high",
    "low", "mild", "severe", "acute", "chronic", "sore", "decreased", "increased", "difficulty", "pus", "redness",
}


def _rel_value(v: str) -> bool:
    """A relative cell: a relation prefix followed by a short person name (not a complaint sentence)."""
    m = _REL_PREFIX.match(v)
    if not m:
        return False
    rest = re.sub(r"\([^)]*\)|\[[^\]]*\]", " ", v[m.end():]).strip(" :.-\t,")  # "C/O Ravi (PAN: X)" -> "Ravi"
    if not rest:
        return True        # a bare "S/o" / "W/o" relation column: relation type is a quasi-identifier
    if len(rest) > 40 or re.search(r"[0-9०-९;|%]", rest):
        return False
    toks = rest.split()
    if len(toks) > 5:
        return False
    return toks[0].lower().strip(".,") not in _COMPLAINT_WORDS


_ADDR_CUE = re.compile(
    r"(?i)\b(?:h\.?\s*no|house|flat|plot|street|road|lane|nagar|colony|near|opp|behind|ward|sector|mohalla|peth|chowk|"
    r"village|vill|post|dist)\b|मकान|गली|नगर|कॉलोनी|मोहल्ला|के पास|चौक|पेठ|वार्ड"
)
_VILLAGE_SUFFIX = re.compile(
    r"(?i)(?:pur|puram|gaon|gaun|wadi|wadi|palli|palle|khera|khurd|kalan|tola|dih|halli|ganj|garh|abad|nagar|"
    r"kheda|bari|para|mau|patti|pada|kuppam|valasa|पुर|गांव|गाँव|वाड़ी|पल्ली|खेड़ा|खुर्द|कलां|टोला|डीह|हल्ली|गंज|गढ़)$"
)
_HOSP_WORDS = {"hospital", "clinic", "trust", "centre", "center", "chikitsalaya", "sevashram", "diagnostics",
               "diagnostic", "lab", "labs", "laboratory", "pathology", "nursing", "college", "phc", "chc", "institute",
               "medical", "health", "care", "pharmacy", "dispensary", "sadan", "mission", "foundation", "society"}
_MED_COMMON = {
    "blood", "sugar", "serum", "count", "total", "random", "fasting", "test", "level", "profile", "panel", "function",
    "urine", "culture", "haemoglobin", "hemoglobin", "glucose", "creatinine", "platelet", "platelets", "sodium",
    "potassium", "lipid", "liver", "kidney", "thyroid", "vitamin", "calcium", "iron", "ferritin", "bilirubin",
    "albumin", "protein", "cholesterol", "triglycerides", "leucocyte", "leukocyte", "differential", "smear", "widal",
    "dengue", "malaria", "typhoid", "xray", "ecg", "echo", "mri", "scan", "ultrasound", "usg", "biopsy", "fever",
    "pain", "cough", "tablet", "tab", "inj", "syrup", "cap", "capsule", "drops", "ointment", "management", "surgery",
    "medicine", "ward", "unit", "opd", "ipd", "icu", "emergency", "casualty", "general", "male", "female", "yes",
    "no", "normal", "abnormal", "positive", "negative", "pending", "done", "stable", "discharged", "admitted",
    "referred", "lama", "expired", "cash", "insurance", "cghs", "esic", "self", "paid", "unpaid", "approved",
    "rejected", "claim", "package", "procedure", "morning", "evening", "night", "daily", "weekly", "monthly",
    "review", "follow", "routine", "urgent", "stat", "repair", "section", "acute", "chronic", "fracture",
    "obstetrics", "gynaecology", "paediatrics", "pediatrics", "orthopaedics", "cardiology", "nephrology",
    "oncology", "neurology", "dermatology", "psychiatry", "ent", "ophthalmology", "dental", "physiotherapy",
}
_MED_SUFFIX = re.compile(r"(?:pril|sartan|statin|azole|mycin|cillin|cycline|oxacin|dipine|formin|parin|prazole|olol|"
                         r"itis|osis|aemia|emia|algia|ectomy|otomy|plasty|scopy|graphy|tidine|vir|amide|amine)$")
_TITLE_PREFIX = re.compile(r"^(?:Mr|Mrs|Ms|Smt|Shri|Sri|Km|Kumari|Md|Mohd|Mohammad|Late)\.?\s+", re.I)
# Honorific suffixes in Hinglish name cells ("Pooja didi", "Asha ji"): part of a name cell, never a name alone
_NAME_HONORIFICS = {"ji", "jee", "didi", "behenji", "sahab", "saab", "sahib", "bhai", "bhaiya", "babu", "amma", "garu",
                    "anna", "akka", "ben", "bai", "chacha", "मौसी", "जी", "दीदी", "साहब"}


def _norm_header(h: str) -> str:
    s = re.sub(r"[_\-/()\[\].#:]+", " ", str(h).strip().lower())
    s = re.sub(r"(?<=[a-z])(?=[0-9])|(?<=[0-9])(?=[a-z])", " ", s)
    return " ".join(s.split())


def _split_camel(h: str) -> str:
    """'PtFirst' -> 'Pt First', 'DtOfBirth' -> 'Dt Of Birth' (keeps acronyms)."""
    return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", str(h))


def infer_role_from_header(header: str) -> tuple[str | None, float]:
    return _infer_role_from_header_cached(str(header))


@lru_cache(maxsize=16384)
def _infer_role_from_header_cached(header: str) -> tuple[str | None, float]:
    nh = _norm_header(_split_camel(header))
    if not nh:
        return None, 0.0
    words = set(nh.split())
    # Cue words decide before generic synonyms ("Mother's Name" is a relative, "Ref Doctor Name" a doctor)
    if words & _DOC_CUES:
        return "DOCTOR", 95.0
    if words & _REL_CUES or nh.startswith(("father", "husband", "mother")):
        return "RELATIVE_NAME", 95.0
    if words & _AMOUNT_CUES:
        return "AMOUNT", 92.0
    best_role: str | None = None
    best_score = 0.0
    non_person = bool(words & _NON_PERSON_CUES)
    for role, syns in _ROLE_SYNONYMS.items():
        for syn in syns:
            if nh == syn:
                return role, 100.0
            if syn in _EXACT_ONLY or (non_person and role in ("PATIENT_NAME", "RELATIVE_NAME")):
                continue
            sc = fuzz.token_sort_ratio(nh, syn) if len(syn) >= 4 else 0.0
            if len(syn) >= 3 and syn in words:
                sc = max(sc, 93.0 + min(5.0, len(syn) * 0.2))
            elif len(syn) >= 4 and re.search(rf"(?<![a-z0-9ऀ-ॿ]){re.escape(syn)}(?![a-z0-9ऀ-ॿ])", nh):
                sc = max(sc, 90.0)
            if sc > best_score:
                best_score = sc
                best_role = role
    if best_score >= 82.0:
        return best_role, best_score
    return None, 0.0


def _digits_only(v: str) -> str | None:
    """Digits of a purely numeric value (allowing +, spaces, dots, dashes, parentheses); None if letters."""
    if re.search(r"[A-Za-zऀ-॥॰-ॿ]", v):
        return None
    d = re.sub(r"\D", "", v)
    return d or None


def _phone_val(v: str) -> bool:
    d = _digits_only(v)
    if d is None:
        return False
    if len(d) == 10 and d[0] in "6789":
        return True
    if len(d) == 11 and d[0] == "0" and d[1] in "6789":
        return True
    if len(d) == 12 and d.startswith("91") and d[2] in "6789" and re.match(r"^\s*(?:\+|91[\s\-])", v):
        return True
    return len(d) == 14 and d.startswith("0091") and d[4] in "6789"


def _abha_val(v: str) -> bool:
    d = _digits_only(v)
    return d is not None and len(d) == 14 and d.startswith("91") and not v.lstrip().startswith(("+", "00"))


def infer_role_from_values(values: list[str]) -> str | None:
    non_empty = [str(v).strip().translate(_DIGIT_MAP) for v in values if v is not None and str(v).strip()]
    if not non_empty:
        return None
    n = len(non_empty)

    def _rate(pred) -> float:
        return sum(1 for v in non_empty if pred(v)) / n

    def _rx(rx: re.Pattern[str]):
        return lambda v: bool(rx.match(v))

    if _rate(_rx(_EMAIL_SHAPE)) >= 0.5:
        return "EMAIL"
    if _rate(_rx(_PAN_SHAPE)) >= 0.5:
        return "PAN"
    if _rate(lambda v: bool(_ABHA_ADDR.match(v)) or _abha_val(v)) >= 0.5:
        return "ABHA"
    if _rate(_phone_val) >= 0.55:
        return "PHONE"
    if _rate(lambda v: (d := _digits_only(v)) is not None and len(d) == 12 and not _phone_val(v)) >= 0.5 or _rate(
            lambda v: bool(re.match(r"^[X*x]{4}[\s\-.]?[X*x]{4}[\s\-.]?\d{4}$", v))) >= 0.5:
        return "AADHAAR"
    if _rate(_rel_value) >= 0.5:
        return "RELATIVE_NAME"
    if _rate(lambda v: bool(_ID_SHAPE.match(v)) and sum(c.isdigit() for c in v) >= 4) >= 0.6:
        return "MRN"
    if _rate(_rx(_PIN_SHAPE)) >= 0.8:
        return "PIN"
    if _rate(lambda v: bool(_DATE_ISO.match(v) or _DATE_SLASH.match(v) or _DATE_MON.match(v))) >= 0.5:
        yrs = [int(m.group(1)) for v in non_empty if (m := re.search(r"\b((?:19|20)\d{2})\b", v))]
        if yrs and (sum(1 for y in yrs if y <= 2015) / len(yrs)) >= 0.5:
            return "DOB"
        return "DATE"
    # Bare integers are NOT ages by shape alone (pulse, SpO2, weight look the same): need an age marker
    if _rate(_rx(_AGE_SHAPE)) >= 0.7 and _rate(lambda v: bool(re.search(r"(?i)\d\s*(?:yrs?|years?|y\b|वर्ष|/\s*[mf]\b)", v))) >= 0.5:
        nums = [int(m.group(1)) for v in non_empty if (m := re.match(r"^(\d{1,3})\b", v))]
        if nums and all(0 <= x <= 115 for x in nums):
            return "AGE"
    if _rate(lambda v: bool(_ADDR_CUE.search(v)) and ("," in v or bool(re.search(r"\d", v)))) >= 0.5:
        return "ADDRESS"
    return None


def infer_column_roles(headers: list[str], rows: list[list[str]]) -> dict[int, str]:
    roles: dict[int, str] = {}
    for idx, h in enumerate(headers):
        h_role, _ = infer_role_from_header(h)
        col_vals = [r[idx] for r in rows if idx < len(r)]
        v_role = infer_role_from_values(col_vals)
        if h_role in ("LAB", "DIAGNOSIS", "MEDICATION", "DOCTOR", "AMOUNT"):
            roles[idx] = h_role
        elif v_role and (not h_role or h_role not in ("DOB", "DATE", "MRN", "PIN", "AGE", "RELATIVE_NAME")):
            roles[idx] = v_role
        elif h_role:
            roles[idx] = h_role
        elif v_role:
            roles[idx] = v_role
        else:
            roles[idx] = "FREE_TEXT"
    return roles


# ----------------------------------------------------------------------------- guard-aware refinement
def name_like_value(v: str, guard: Any | None) -> bool:
    """A person-name-shaped cell: 1-4 capitalised alphabetic tokens (Latin or Devanagari), not clinical words."""
    v = str(v).strip()
    if not v or len(v) > 48 or re.search(r"[0-9०-९@/:;()]", v):
        return False
    if re.match(r"(?i)^(?:dr|doctor)\b", v):
        return False
    v2 = _TITLE_PREFIX.sub("", v).strip(" ,.")
    toks = [t for t in re.split(r"[\s,]+", v2) if t]
    if len(toks) > 1 and toks[-1].lower() in _NAME_HONORIFICS:
        toks = toks[:-1]
    if not 1 <= len(toks) <= 4:
        return False
    for t in toks:
        tt = t.strip(".'")
        if re.fullmatch(r"[A-Z]", tt):
            continue  # initial
        if re.fullmatch(r"[ऀ-ॿ]{2,25}", tt):
            pass
        elif re.fullmatch(r"[A-Z][a-z'\-]{1,24}", tt) or re.fullmatch(r"[A-Z]{4,24}", tt):
            pass
        else:
            return False
        low = tt.lower()
        if low in _MED_COMMON or low in _HOSP_WORDS or _MED_SUFFIX.search(low):
            return False
        if guard is not None and guard.is_clinical_word(low):
            return False
    return True


def refine_value_roles(values: list[str], guard: Any | None) -> str | None:
    """Second-stage value role for unresolved columns/keys: PATIENT_NAME / VILLAGE / TEHSIL / DOCTOR."""
    vals = [str(v).strip() for v in values if v is not None and str(v).strip()]
    if not vals:
        return None
    n = len(vals)
    if sum(1 for v in vals if re.match(r"(?i)^(?:dr\.?|डॉ\.?)\s", v)) / n >= 0.5:
        return "DOCTOR"
    if sum(1 for v in vals if _rel_value(v)) / n >= 0.5:
        return "RELATIVE_NAME"
    name_rate = sum(1 for v in vals if name_like_value(v, guard)) / n
    if name_rate < 0.7:
        return None
    distinct = len(set(v.lower() for v in vals)) / n
    vill_rate = sum(1 for v in vals if _VILLAGE_SUFFIX.search(v.split()[-1] if v.split() else v) or
                    _VILLAGE_SUFFIX.search(v.split()[0] if v.split() else v)) / n
    if vill_rate >= 0.4:
        return "VILLAGE"
    avg_toks = sum(len(v.split()) for v in vals) / n
    if n >= 6 and distinct < 0.3 and avg_toks < 1.5:
        return "TEHSIL"   # a few repeated single-word proper nouns: admin area -> generalise, never keep raw
    return "PATIENT_NAME"  # fail closed: anything person-shaped is treated as a name


def refine_entry_roles(entry: Any, guard: Any) -> None:
    """Re-profile columns / JSON keys that neither header nor shape could resolve (needs the clinical lexicon)."""
    if entry.file_type in ("csv", "tsv") and entry.parsed_data:
        rows = entry.parsed_data
        header, body = rows[0], rows[1:]
        for idx in range(len(header)):
            if entry.column_roles.get(idx, "FREE_TEXT") not in ("FREE_TEXT", "NOTES"):
                continue
            if entry.column_roles.get(idx) == "NOTES":
                continue
            role = refine_value_roles([r[idx] for r in body if idx < len(r)], guard)
            if role:
                entry.column_roles[idx] = role
        return
    if entry.file_type in ("json", "fhir") and entry.parsed_data is not None:
        by_key: dict[str, list[str]] = {}
        stack: list[Any] = [entry.parsed_data]
        while stack:
            node = stack.pop()
            if isinstance(node, dict):
                for k, v in node.items():
                    if isinstance(v, (dict, list)):
                        stack.append(v)
                    elif isinstance(v, str):
                        by_key.setdefault(str(k).lower(), []).append(v)
                    elif isinstance(v, (int, float)) and not isinstance(v, bool):
                        by_key.setdefault(str(k).lower(), []).append(str(v))
            elif isinstance(node, list):
                stack.extend(x for x in node if isinstance(x, (dict, list)))
        roles: dict[str, str] = {}
        for k, vals in by_key.items():
            h_role, _ = infer_role_from_header(k)
            if h_role:
                continue
            v_role = infer_role_from_values(vals) or refine_value_roles(vals, guard)
            if v_role:
                roles[k] = v_role
        entry.key_roles = roles


def resolve_key_role(entry: Any, key: str) -> str | None:
    """JSON key role: header synonyms first, then the per-file value profile."""
    role, _ = infer_role_from_header(key)
    if role:
        return role
    kr = getattr(entry, "key_roles", None) or {}
    return kr.get(str(key).lower())


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

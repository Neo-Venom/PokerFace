"""Step B: Multilingual identifier regex/context detector, corpus ID learner, and numeric context classifier."""
from __future__ import annotations

import re
from dataclasses import dataclass

from raksha.detect.normalize import NormalizedView, build_normalized_view

_DEVA_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    label: str
    text: str
    source: str = "identifier"


# --------------------------------------------------------------------------- numeric context classifier
_CLINICAL_UNIT_AFTER_RE = re.compile(
    r"^\s*(?:/\s*(?:uL|ul|mm3|min|hr|day|wk)|mg/dL|g/dL|mmol/L|mIU/L|IU/L|mg|mcg|ug|ml|mL|g\b|IU\b|%|mmHg|bpm|kg\b|°?[FC]\b|days?\b|wks?\b|weeks?\b|months?\b|yrs?\b|years?\b|puffs?\b|tabs?\b|sachet|वर्ष|दिन|माह|सप्ताह)",
    re.I,
)
_CLINICAL_TERM_BEFORE_RE = re.compile(
    r"(?:platelet(?:\s+count)?|wbc(?:\s+count)?|rbc|hb|haemoglobin|hemoglobin|hba1c|rbs|blood\s+sugar|"
    r"creatinine|sodium|potassium|tsh|loinc|bp|pulse|temp(?:erature)?|spo2|weight|"
    r"dose|dosage|ref(?:erence)?\s*range|range|amount|rs\.?|inr|₹|"
    r"tab\.?|cap\.?|inj\.?|syr\.?|inh\.?|sachet|paracetamol|dolo|metformin|atorvastatin|"
    r"azithromycin|ceftriaxone|amlodipine|telmisartan|ramipril|artesunate|pantoprazole)\s*[:=\-]?\s*$",
    re.I,
)
_PII_NUM_BEFORE_RE = re.compile(
    r"(?:mob(?:ile)?|ph(?:one)?|tel|call(?:\s+karo)?|contact|whatsapp|callback|मोबाइल|फ़ोन|फोन|संपर्क|"
    r"aadhaar|aadhar|adhaar|uidai|आधार|abha|आभा|uhid|mrn|ipd|reg\.?\s*no|पंजीकरण|आईपीडी|"
    r"pin(?:\s*code)?|zip|पिन)\s*[:.#=\-]?\s*$",
    re.I,
)
_DATE_CONTEXT_RE = re.compile(r"(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2})")


def is_clinical_numeric_context(text: str, start: int, end: int, kind: str) -> bool:
    """Return True if a numeric span sits inside a clinical/lab/dose/vital context (NOT an ID)."""
    before = text[max(0, start - 45) : start]
    after = text[end : min(len(text), end + 30)]

    # Explicit PII label before the number overrides ambiguity
    if _PII_NUM_BEFORE_RE.search(before):
        return False

    # Unit immediately following (e.g. "240124 /uL", "650 mg", "150000-410000")
    if _CLINICAL_UNIT_AFTER_RE.match(after):
        return True

    # Range like "150000-410000" or "4000-11000"
    if re.match(r"^\s*[-–]\s*\d+", after) or re.search(r"\d+\s*[-–]\s*$", before):
        if kind in ("PIN", "PHONE", "AADHAAR", "ABHA_NUMBER"):
            return True

    # Clinical test or drug right before the number
    if _CLINICAL_TERM_BEFORE_RE.search(before):
        return True

    # Frequency / vital slash patterns like 1-0-1, 150/90
    if re.match(r"^\s*/\s*\d{2,3}\b", after) or re.search(r"\b\d{2,3}\s*/\s*$", before):
        return True

    return False


# --------------------------------------------------------------------------- core identifier patterns
_PHONE_PATTERNS = [
    # +91 / 0 + 10 digits with spaces/dashes/dots + optional extension
    re.compile(r"(?<![\dA-Za-z])(?:\+?91[\s.\-]?|0)?[6-9]\d{4}[\s.\-]?\d{5}(?:\s*(?:ext\.?|extension|x)\s*\d{2,5})?(?!\d)", re.I),
    re.compile(r"(?<![\dA-Za-z])(?:\+?91[\s.\-]?|0)?[6-9]\d{2}[\s.\-]?\d{3}[\s.\-]?\d{4}(?:\s*(?:ext\.?|extension|x)\s*\d{2,5})?(?!\d)", re.I),
    re.compile(r"(?<![\dA-Za-z])(?:\+?91[\s.\-]?|0)?[6-9]\d[\s.\-]?\d{4}[\s.\-]?\d{4}(?:\s*(?:ext\.?|extension|x)\s*\d{2,5})?(?!\d)", re.I),
    # Masked phone like 98XXXXXX10
    re.compile(r"(?<![\dA-Za-z])[6-9]\d[xX*]{4,6}\d{2,4}(?![\dA-Za-z])"),
]
# Phone trigger for landlines or short/non-[6-9] local extensions when explicitly labeled
_LABELED_PHONE_RE = re.compile(
    r"(?i)\b(?:mob|mobile|phone|ph|tel|contact|whatsapp|callback|मोबाइल|फ़ोन|फोन)\s*[:.#\-]\s*(\+?[\d][\d\s.\-]{7,16}\d(?:\s*(?:ext\.?|x)\s*\d{2,5})?)"
)

_ABHA_NUM_RE = re.compile(r"(?<!\d)\d{2}[\s\-]\d{4}[\s\-]\d{4}[\s\-]\d{4}(?!\d)|(?<!\d)\d{14}(?!\d)")
_ABHA_ADDR_RE = re.compile(r"(?<![\w.+-])[A-Za-z0-9._+-]{2,}@(?:abdm|ndhm|sbx)\b", re.I)
_EMAIL_RE = re.compile(r"(?<![\w.+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_AADHAAR_MASKED_RE = re.compile(r"(?<!\w)(?:[xX*]{4}[\s\-]?[xX*]{4}[\s\-]?\d{4}|[xX*]{8}\d{4})(?!\w)")
_AADHAAR_FULL_RE = re.compile(r"(?<!\d)\d{4}[\s\-]?\d{4}[\s\-]?\d{4}(?!\d)")
_PAN_RE = re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b")
_INSURANCE_PATTERNS = [
    re.compile(r"\b(?:PMJAY|AB-?PMJAY|CGHS|ESIS|RSBY)[-/][A-Z0-9]{1,8}(?:[-/][A-Z0-9]{2,})+\b", re.I),
    re.compile(r"\bPOL/\d{2,4}/\d{4,}\b", re.I),
]
_MRN_PATTERNS = [
    re.compile(r"\b[A-Z]{2,6}(?:/[A-Z]{1,6})?/\d{2,4}/\d{3,}\b"),  # e.g. HOSP/26/001234, NVJ/OP/25/722967
    re.compile(r"\b[A-Z]{2,6}:[A-Z0-9]{2,8}:[A-Z0-9]{1,8}:\d{3,}\b"),  # e.g. KLS:UHID:25:18763
    re.compile(r"\b[A-Z]{2,6}_[A-Z0-9]{2,8}_\d{2,4}_\d{3,}\b"),       # e.g. PGC_REG_25_851282
    re.compile(r"\b[A-Z]{2,6}-[A-Z0-9]{1,5}-\d{3,}\b"),             # e.g. RGH-B25-639176, HOSP-IPD-00123
    re.compile(r"\b[A-Z]{2,6}-\d{4,}\b"),                           # e.g. HOSP-0012345, ZQ-901827
    re.compile(r"\b[A-Z]{2,6}\d{2,4}-\d{3,}\b"),                      # e.g. HOSP2026-001234, NVP27-110421
    re.compile(r"\b[A-Z]{2,5}\d{5,}\b"),                            # e.g. ADM000123
]
_LABELED_MRN_RE = re.compile(
    r"(?i)(?:\bUHID|\bMRN|\bReg\.?(?:\s*No\.?)?|\bIPD(?:\s*No\.?)?|\bID|पंजीकरण\s*(?:सं\.?|क्रमांक)|पंजीयन\s*(?:सं\.?|क्रमांक)|आईपीडी\s*(?:सं\.?|क्रमांक))"
    r"\s*(?:[:#]\s*|\s+(?=[A-Z]{2,}[-/:_0-9]))([A-Z0-9][A-Z0-9/:_\-]{4,28})"
)
_DOB_LINE_RE = re.compile(
    r"(?i)(?:\bDOB|\bDate\s+of\s+Birth|\bBirth\s+Date|जन्म\s*तिथि)\s*[:\-]\s*(\d{1,4}[./\-]\d{1,2}[./\-]\d{1,4})"
)
_PIN_CONTEXT_RE = re.compile(
    r"(?:-\s+|,\s+|\bPIN\s*[:\s\-]+|पिन\s*[:\s\-]+)([1-8]\d{5})(?!\d)",
    re.I,
)

# Exclusions for MRN regex (billing/encounter/order/clinical codes that are not patient UHID/MRN)
_MRN_PREFIX_EXCLUDE = ("BL/", "OPD/", "LOINC", "ICD", "LAB-", "ORD-")


def derive_id_patterns_from_values(mrn_values: list[str]) -> list[re.Pattern]:
    """Derive regex patterns dynamically from structured MRN/UHID columns seen in the current run."""
    skeletons: set[str] = set()
    for raw in mrn_values:
        s = raw.translate(_DEVA_DIGITS).strip()
        if not s or len(s) < 5 or len(s) > 30:
            continue
        # Convert concrete token into a generalized shape regex: alpha runs, digit runs, separators
        parts: list[str] = []
        i = 0
        while i < len(s):
            if s[i].isalpha() and s[i].isascii():
                j = i
                while j < len(s) and s[j].isalpha() and s[j].isascii():
                    j += 1
                ln = j - i
                parts.append(f"[A-Z]{{{max(2, ln-1)},{ln+2}}}" if s[i:j].isupper() else f"[A-Za-z]{{{ln}}}")
                i = j
            elif s[i].isdigit():
                j = i
                while j < len(s) and s[j].isdigit():
                    j += 1
                ln = j - i
                parts.append(f"\\d{{{max(2, ln-1)},{ln+2}}}" if ln >= 3 else f"\\d{{{ln}}}")
                i = j
            elif s[i] in "-/_":
                parts.append(re.escape(s[i]))
                i += 1
            else:
                parts = []
                break
        if parts and any("\\d" in p for p in parts) and any("[A-Z" in p for p in parts):
            skeletons.add(r"\b" + "".join(parts) + r"\b")
    return [re.compile(pat) for pat in sorted(skeletons)]


def detect_identifier_spans(
    text: str,
    extra_mrn_patterns: list[re.Pattern] | None = None,
    view: NormalizedView | None = None,
) -> list[Span]:
    """Find identifier spans in `text` via Step C normalized analysis view, mapped back to exact original offsets."""
    if not text:
        return []
    nv = view if view is not None else build_normalized_view(text)
    norm_text = nv.clean
    spans: list[Span] = []

    def add_span(ns: int, ne: int, label: str) -> None:
        os_, oe_ = nv.map_clean_span(ns, ne)
        if oe_ > os_:
            spans.append(Span(os_, oe_, label, text[os_:oe_]))

    # 1. ABHA address & Email
    for m in _ABHA_ADDR_RE.finditer(norm_text):
        add_span(m.start(), m.end(), "ABHA_ADDRESS")
    for m in _EMAIL_RE.finditer(norm_text):
        if not _ABHA_ADDR_RE.match(m.group(0)):
            add_span(m.start(), m.end(), "EMAIL")

    # 2. PAN
    for m in _PAN_RE.finditer(norm_text):
        add_span(m.start(), m.end(), "PAN")

    # 3. Insurance IDs
    for rx in _INSURANCE_PATTERNS:
        for m in rx.finditer(norm_text):
            add_span(m.start(), m.end(), "INSURANCE_ID")

    # 4. ABHA 14-digit (before 12-digit Aadhaar and 10-digit phone!)
    for m in _ABHA_NUM_RE.finditer(norm_text):
        if not is_clinical_numeric_context(norm_text, m.start(), m.end(), "ABHA_NUMBER"):
            add_span(m.start(), m.end(), "ABHA_NUMBER")

    # 5. Masked Aadhaar & Full Aadhaar (12-digit)
    for m in _AADHAAR_MASKED_RE.finditer(norm_text):
        add_span(m.start(), m.end(), "AADHAAR_MASKED")
    for m in _AADHAAR_FULL_RE.finditer(norm_text):
        os_, oe_ = nv.map_clean_span(m.start(), m.end())
        if any(s.start <= os_ and oe_ <= s.end for s in spans):
            continue
        if not is_clinical_numeric_context(norm_text, m.start(), m.end(), "AADHAAR"):
            add_span(m.start(), m.end(), "AADHAAR")

    # 6. Phones (+91, 0-prefix, spaces, dashes, Devanagari digits, extensions, masked)
    for rx in _PHONE_PATTERNS:
        for m in rx.finditer(norm_text):
            os_, oe_ = nv.map_clean_span(m.start(), m.end())
            if any(s.start <= os_ and oe_ <= s.end for s in spans):
                continue
            if not is_clinical_numeric_context(norm_text, m.start(), m.end(), "PHONE"):
                add_span(m.start(), m.end(), "PHONE")
    for m in _LABELED_PHONE_RE.finditer(norm_text):
        os_, oe_ = nv.map_clean_span(m.start(1), m.end(1))
        if not any(s.start <= os_ and oe_ <= s.end for s in spans):
            add_span(m.start(1), m.end(1), "PHONE")

    # 7. MRN / UHID / IPD (built-in + corpus-derived + labeled)
    for rx in _MRN_PATTERNS + (extra_mrn_patterns or []):
        for m in rx.finditer(norm_text):
            val = m.group(0)
            if val.startswith(_MRN_PREFIX_EXCLUDE):
                continue
            os_, oe_ = nv.map_clean_span(m.start(), m.end())
            if any(s.start <= os_ and oe_ <= s.end for s in spans):
                continue
            if any(irx.match(val) for irx in _INSURANCE_PATTERNS):
                continue
            add_span(m.start(), m.end(), "MRN")
    for m in _LABELED_MRN_RE.finditer(norm_text):
        os_, oe_ = nv.map_clean_span(m.start(1), m.end(1))
        if not any(s.start <= os_ and oe_ <= s.end for s in spans):
            add_span(m.start(1), m.end(1), "MRN")

    # 8. Labeled DOB & Address PIN
    for m in _DOB_LINE_RE.finditer(norm_text):
        add_span(m.start(1), m.end(1), "DOB")
    for m in _PIN_CONTEXT_RE.finditer(norm_text):
        s0, e0 = m.start(1), m.end(1)
        if not is_clinical_numeric_context(norm_text, s0, e0, "PIN"):
            add_span(s0, e0, "PIN")

    return resolve_overlaps(spans)


def resolve_overlaps(spans: list[Span]) -> list[Span]:
    """Resolve overlapping spans deterministically: longer/earlier wins."""
    if not spans:
        return []
    ordered = sorted(spans, key=lambda sp: (sp.start, -(sp.end - sp.start), sp.label))
    kept: list[Span] = []
    last_end = -1
    for sp in ordered:
        if sp.start >= last_end:
            kept.append(sp)
            last_end = sp.end
    return kept


def redact_spans(text: str, spans: list[Span], stats: dict[str, int] | None = None) -> str:
    """Replace non-overlapping spans in original text from right to left."""
    if not spans:
        return text
    ordered = sorted(spans, key=lambda sp: sp.start)
    parts: list[str] = []
    cur = 0
    for sp in ordered:
        if sp.start < cur:
            continue
        parts.append(text[cur : sp.start])
        parts.append(f"[{sp.label}]")
        if stats is not None:
            stats[sp.label] = stats.get(sp.label, 0) + 1
        cur = sp.end
    parts.append(text[cur:])
    return "".join(parts)

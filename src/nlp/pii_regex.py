"""High-precision Indian ID, phone, email, address, and PIN pattern detectors."""
from __future__ import annotations

import regex as re

from src.models import Span
from src.nlp.indic_norm import detect_script, shadow_normalize

D = r"[0-9]"
S = r"[\s\-.]?"

# Order reflects overlap arbitration priority
PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("ABHA_NUMBER", re.compile(rf"(?<!{D}){D}{{2}}-{D}{{4}}-{D}{{4}}-{D}{{4}}(?!{D})")),
    ("ABHA_ADDRESS", re.compile(r"(?i)\b[\w.]{3,}@(?:abdm|sbx|ndhm)\b")),
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+\.[A-Za-z]{2,}\b")),
    ("PAN", re.compile(r"\b[A-Z]{3}[PCHFATBLJG][A-Z]\d{4}[A-Z]\b")),
    ("AADHAAR_MASKED", re.compile(r"(?i)(?<![A-Z0-9])[X*]{4}[\s\-]?[X*]{4}[\s\-]?\d{4}(?!\d)")),
    ("AADHAAR", re.compile(rf"(?<!{D}){D}{{4}}{S}{D}{{4}}{S}{D}{{4}}(?!{D})")),
    ("PHONE", re.compile(rf"(?<!{D})(?:(?:\+|00)?91{S}|0)?[6-9](?:{S}{D}){{9}}(?!{D})")),
    # Generic hospital / insurance / sample / bill IDs (excl. dates & ICD10 & LOINC)
    ("INSURANCE_ID", re.compile(r"\b(?:PMJAY-[A-Z]{2}-\d{6,}|POL/\d{2}/\d{6,}|TPA[-/][A-Z0-9/-]{4,})\b")),
    (
        "MRN",
        re.compile(
            r"\b(?:SNJ|JJH|UHID|DHH\d{0,4}|ADM|IPD|OPD|IP|BL|MRN)[/_-]?(?:\d{2,4}[/_-]){0,3}\d{3,}\b"
            r"|\b[A-Z]{2,5}/\d{2,4}/\d{4,}\b"
            r"|\b[A-Z]{2,5}-\d{5,}\b"
        ),
    ),
]

_PIN_CONTEXT_RE = re.compile(
    r"(?i)(?:pin(?:\s*code)?|postal(?:\s*code)?|zip|distt?\.?[^,\n-]{2,25}|जिला[^,\n-]{2,25})[:\s\-–]*([1-8]\d{5})\b"
    r"|,\s*([1-8]\d{5})(?!\s*(?:/uL|/cumm|mg|g/dl))"
)

_HOUSE_RE = re.compile(
    r"(?i)(?:\b(?:H\.?\s*No\.?|House(?:\s+No\.?)?|Makan(?:\s+No\.?)?|Flat(?:\s+No\.?)?)\s*[:.]?\s*[A-Za-z0-9/\-]{1,12}|मकान\s*नं\.?\s*[0-9०-९A-Za-z/\-]{1,12})"
)
_LOCALITY_RE = re.compile(
    r"(?i)(?:\bMohalla\s+[A-Z\u0900-\u097f][^\n,;.]{2,35}"
    r"|\bWard\s+\d+(?=\s*(?:,|\n|$|\bmohalla\b|\btehsil\b|\bdist|\bgram\b|\bvill|\bpin\b))"
    r"|मोहल्ला\s+[^\n,;.]{2,35})"
)
_LANDMARK_EN_RE = re.compile(r"(?i)\b(?:near|opp\.?|opposite|behind)\s+[A-Z][A-Za-z\s]{2,35}(?=,|\n|$)")
_LANDMARK_HI_RE = re.compile(r"[^\n,;.]{2,35}\s+के\s+(?:पास|पीछे|सामने)")
_VILLAGE_RE = re.compile(
    r"(?i)(?:\b(?:Vill\.?|Village|Gram)\s*[:\-]?\s*([A-Z\u0900-\u097f][A-Za-z\u0900-\u097f\s]{2,30})(?=,|\n|$)"
    r"|ग्राम\s*[:\-]?\s*([^\n,;.]{2,30}))"
)
_DOB_LINE_RE = re.compile(
    r"(?i)(?:\b(?:DOB|Birth\s*Date|Date\s+of\s+Birth|Janm\s*Tithi|जन्म\s*तिथि)\s*[:=]?\s*)(\d{1,2}[./-]\d{1,2}[./-]\d{4}|\d{4}-\d{2}-\d{2})"
)
_LABELLED_MRN_RE = re.compile(
    r"(?i)(?:\b(?:UHID|Reg(?:\.?|istration)?\s*No\.?|IPD\s*No\.?|OPD\s*No\.?|Patient\s*ID|MRN)|पंजीकरण\s*सं\.?|आईपीडी\s*सं\.?)\s*[:#\-]?\s*([A-Z0-9][A-Z0-9/\-]{4,22})"
)
_PATH_TRAIL_ID_RE = re.compile(r"(?<=[_/\\-])(\d{6,})(?=\.[A-Za-z]{2,5}\b)")
_UUID_RE = re.compile(r"(?i)\b(?:urn:uuid:)?[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b")


def extract_regex_spans(text: str) -> list[Span]:
    """Detect structured IDs, phones, emails, PINs, houses, landmarks, DOBs on shadow text."""
    shadow = shadow_normalize(text)
    uuid_spans = [(m.start(), m.end()) for m in _UUID_RE.finditer(shadow)]

    def _in_uuid(s: int, e: int) -> bool:
        return any(us < e and ue > s for us, ue in uuid_spans)

    spans: list[Span] = []

    for kind, rx in PATTERNS:
        for m in rx.finditer(shadow):
            if _in_uuid(m.start(), m.end()):
                continue
            raw_surf = text[m.start() : m.end()]
            spans.append(Span(m.start(), m.end(), kind, raw_surf, script=detect_script(raw_surf)))

    for m in _LABELLED_MRN_RE.finditer(shadow):
        s, e = m.start(1), m.end(1)
        if _in_uuid(s, e):
            continue
        spans.append(Span(s, e, "MRN", text[s:e], script=detect_script(text[s:e])))

    for m in _PATH_TRAIL_ID_RE.finditer(shadow):
        s, e = m.start(1), m.end(1)
        if _in_uuid(s, e):
            continue
        spans.append(Span(s, e, "MRN", text[s:e], script=detect_script(text[s:e])))

    for m in _DOB_LINE_RE.finditer(shadow):
        s, e = m.start(1), m.end(1)
        if _in_uuid(s, e):
            continue
        spans.append(Span(s, e, "DOB", text[s:e], script=detect_script(text[s:e])))

    for m in _PIN_CONTEXT_RE.finditer(shadow):
        grp = 1 if m.group(1) else 2
        s, e = m.start(grp), m.end(grp)
        if _in_uuid(s, e):
            continue
        spans.append(Span(s, e, "PIN", text[s:e], script=detect_script(text[s:e])))

    for kind, rx in (
        ("HOUSE", _HOUSE_RE),
        ("LOCALITY", _LOCALITY_RE),
        ("LANDMARK", _LANDMARK_EN_RE),
    ):
        for m in rx.finditer(shadow):
            if _in_uuid(m.start(), m.end()):
                continue
            raw_surf = text[m.start() : m.end()]
            spans.append(Span(m.start(), m.end(), kind, raw_surf, script=detect_script(raw_surf)))

    if "के" in shadow:
        for m in _LANDMARK_HI_RE.finditer(shadow):
            if _in_uuid(m.start(), m.end()):
                continue
            raw_surf = text[m.start() : m.end()]
            spans.append(Span(m.start(), m.end(), "LANDMARK", raw_surf, script=detect_script(raw_surf)))

    for m in _VILLAGE_RE.finditer(shadow):
        grp = 1 if m.group(1) else 2
        s, e = m.start(grp), m.end(grp)
        if _in_uuid(s, e):
            continue
        spans.append(Span(s, e, "VILLAGE", text[s:e].strip(), script=detect_script(text[s:e])))

    return spans

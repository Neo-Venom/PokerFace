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
    ("AADHAAR", re.compile(rf"(?<![\d+]){D}{{4}}{S}{D}{{4}}{S}{D}{{4}}(?!{D})")),
    # Indian mobiles in any grouping: +91 / 0091 / 0 prefixes, spaces, dots, dashes, (xxx) parentheses
    ("PHONE", re.compile(rf"(?<![\d/])(?:(?:\+|00)\s?91[\s\-.]?|\b91[\s\-]|0)?\(?[6-9](?:[\s\-.)]{{0,2}}{D}){{9}}(?![\d/])")),
    # Generic insurance / scheme IDs (PMJAY-style, policy numbers, TPA references)
    ("INSURANCE_ID", re.compile(r"\b(?:(?:AB)?PMJAY[-/][A-Z0-9/-]{6,}|POL/\d{2}/\d{6,}|TPA[-/][A-Z0-9/-]{4,}|[A-Z]{2,5}-POL-\d{5,})\b")),
    # Generic hospital / IPD / registration IDs: an uppercase prefix + separators + >=4 digits.
    # Shape-based on purpose (no hospital-code list) so unseen hospitals' formats are caught.
    (
        "MRN",
        re.compile(
            r"\b(?:UHID|MRN|IPD|OPD|ADM|IP|REG)[/_.\-]?(?:\d{2,4}[/_.\-]){0,3}\d{3,}\b"
            r"|\b[A-Z]{2,8}(?:[/_.|\-][A-Z]{1,12}){0,3}(?:[/_.|\-]?\d{2,8}){1,4}(?:[/_.|\-][A-Z]{2,12})?\b"
        ),
    ),
]

_ICD_RE = re.compile(r"^[A-Z]\d{2}(?:\.\d{1,4})?$")
_NOT_ID_PREFIX = re.compile(r"^(?:ICD|LOINC|SNOMED|COVID|SARS|HBA|SPO|BP|HR|RR|PR|FBS|PPBS|RBS|TSH|WBC|RBC|PKG|ATC|NABH|ISO|PIN|PH)\b", re.I)


def _is_generic_id(tok: str) -> bool:
    """Generic ID token: >=4 digits overall, not an ICD/LOINC/lab/package code."""
    if _ICD_RE.match(tok) or _NOT_ID_PREFIX.match(tok):
        return False
    digits = sum(ch.isdigit() for ch in tok)
    if 4 <= digits <= 20:
        return True
    # audit / reference tokens such as ABC-XYZ-013-REF: >=4 segments, a 3-digit segment
    segs = [s for s in re.split(r"[/_.|\-]", tok) if s]
    return digits == 3 and len(segs) >= 4 and any(s.isdigit() and len(s) == 3 for s in segs)

_PIN_CONTEXT_RE = re.compile(
    r"(?i)(?:\bpin(?:\s*code)?|\bpost(?:al)?(?:\s*code)?|\bzip|\bdistt?\.?[^,\n\-]{2,25}|\bdistrict[^,\n\-]{2,25}|\bzill?a[^,\n\-]{2,25}"
    r"|[जज़]़?िला[^,\n\-]{2,25}|पिन(?:\s*कोड)?)[:\s\-–]*([1-8]\d{5})\b"
    r"|,\s*([1-8]\d{5})(?!\s*(?:/uL|/cumm|/mm3|mg|g/dl|iu|units?))"
)
# A 6-digit number closing an address-like line ("... Vill. X, Dist. Y - 415673")
_PIN_LINE_END_RE = re.compile(r"[-–]\s*([1-8]\d{5})\s*$", re.M)
_ADDR_CUE_RE = re.compile(r"(?i)\b(?:h\.?\s*no|house|vill|village|gram|tal\.|tehsil|taluka|dist|district|zill?a|road|nagar|colony|near|opp)\b|मकान|ग्राम|तहसील|ज़िला|जिला|पता")
# Locality: 1-3 proper-noun words + a locality head noun (Latin or Devanagari)
_LOCALITY_SUFFIX_RE = re.compile(
    r"\b(?:[A-Z][a-z]{1,20}\s+){1,3}(?:Colony|Nagar|Peth|Chowk|Tola|Basti|Mohalla|Bazar|Bazaar|Gali|Marg|Layout|Camp(?:\s+Lane\s+\d+)?|Para|Wadi|Puram|Extension|Enclave|Lane\s+\d+|Road|Rd\.?|Street|St\.?|Galli|Cross|Main)(?![a-z])"
    r"|(?:[ऀ-ॿ]{2,20}\s+){1,3}(?:कॉलोनी|नगर|पेठ|चौक|टोला|बस्ती|गली|मार्ग|बाज़ार|बाजार|लेन\s*\d*|कैंप(?:\s+लेन\s*\d+)?)"
)

_HOUSE_RE = re.compile(
    r"(?i)(?:\b(?:H\.?\s*No\.?|House(?:\s+No\.?)?|Makan(?:\s+No\.?)?|Flat(?:\s+No\.?)?)\s*[:.]?\s*[A-Za-z0-9/\-]{1,12}|मकान\s*नं\.?\s*[0-9०-९A-Za-z/\-]{1,12})"
)
_LOCALITY_RE = re.compile(
    r"(?i)(?:\bMohalla\s+[A-Z\u0900-\u097f][^\n,;.]{2,35}"
    r"|\bWard\s+\d+(?=\s*(?:,|\n|$|\bmohalla\b|\btehsil\b|\bdist|\bgram\b|\bvill|\bpin\b))"
    r"|\u0935\u093e\u0930\u094d\u0921\s*[0-9\u0966-\u096f]+(?=\s*(?:,|\n|$))"
    r"|\u092e\u094b\u0939\u0932\u094d\u0932\u093e\s+[^\n,;.]{2,35})"
)
_LANDMARK_EN_RE = re.compile(
    r"\b(?i:near|opp\.?|opposite|behind|next\s+to|beside)\s+[A-Z][A-Za-z.'&]*(?:[ \t]+[A-Z][A-Za-z.'&]*){0,4}(?=[ \t]*(?:,|\n|;|$))"
)
_LANDMARK_HI_RE = re.compile(r"[^\n,;.]{2,35}\s+के\s+(?:पास|पीछे|सामने)")
_VILLAGE_RE = re.compile(
    r"(?:\b(?i:Vill\.|Vill(?=\s)|Village(?=[\s:\-])|Gram(?=[\s:\-])|Gaon(?=[\s:\-])|resident\s+of|native\s+of|r/o)\s*[:\-]?\s*"
    r"([A-Z\u0900-\u097f][A-Za-z\u0900-\u097f]{1,20}(?:[ \t]+[A-Z\u0900-\u097f][A-Za-z\u0900-\u097f]{1,20}){0,2})(?=[,.;)\n]|\s+(?:-|\u2013|\d)|$)"
    r"|\u0917\u094d\u0930\u093e\u092e(?![\u0900-\u097f])\s*[:\-]?\s*([^\n,;.]{2,30}))"
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
            if kind == "MRN" and not _is_generic_id(raw_surf):
                continue
            if kind == "PHONE" and sum(ch.isdigit() for ch in shadow[m.start() : m.end()]) not in (10, 11, 12, 13, 14):
                continue
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

    for m in _PIN_LINE_END_RE.finditer(shadow):
        line_start = shadow.rfind("\n", 0, m.start()) + 1
        if _ADDR_CUE_RE.search(shadow[line_start : m.start()]):
            s, e = m.start(1), m.end(1)
            spans.append(Span(s, e, "PIN", text[s:e], script=detect_script(text[s:e])))

    for kind, rx in (
        ("HOUSE", _HOUSE_RE),
        ("LOCALITY", _LOCALITY_RE),
        ("LOCALITY", _LOCALITY_SUFFIX_RE),
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

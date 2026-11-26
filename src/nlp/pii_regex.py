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

# ------------------------------------------------------------------ other government / financial identifiers
_STATE = r"(?:AN|AP|AR|AS|BR|CG|CH|DD|DL|DN|GA|GJ|HP|HR|JH|JK|KA|KL|LA|LD|MH|ML|MN|MP|MZ|NL|OD|OR|PB|PY|RJ|SK|TN|TR|TS|UK|UA|UP|WB)"
_UPI_HANDLE = (r"(?:ok[a-z]{2,12}|wa[a-z]{3,10}|[a-z]{0,12}bank|[a-z]{0,10}pay|ybl|ibl|axl|apl|upi|paytm|sbi|icici|axis|hdfc"
               r"|kotak|pnb|boi|cnrb|kmbl|fbl|mbk|ikwik|jio|airtel|freecharge|federal|indus|rbl|yesbank|idfc|dbs|citi|hsbc"
               r"|ippb|postbank|unionbankofindia|barodampay|timecosmos|jupiteraxis|abfspay|axisb|pingpay|sliceaxis)")
_UPI_HANDLE_RE = re.compile(r"(?i)" + _UPI_HANDLE)
_NOT_UPI_HANDLE = frozenset({"abdm", "sbx", "ndhm"})


def _is_upi(vpa: str) -> bool:
    local, _, handle = vpa.partition("@")
    if handle.lower() in _NOT_UPI_HANDLE:
        return False
    return bool(_UPI_HANDLE_RE.fullmatch(handle)) or any(c.isdigit() or c in "._" for c in local)
OTHER_ID_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    # Driving licence: state + RTO + year of issue + 7 digits ("MH12 20110012345", "DL-0420110012345")
    ("GOV_ID", re.compile(rf"(?<![A-Za-z0-9]){_STATE}[ \-]?\d{{2}}[ \-]?(?:19|20)\d{{2}}[ \-]?\d{{7}}(?![0-9])")),
    # Vehicle registration ("KA-01-AB-4521", "MH 14 AB 7342", BH series "22 BH 1234 AA")
    ("GOV_ID", re.compile(rf"(?<![A-Za-z0-9]){_STATE}[ \-.]?\d{{1,2}}[ \-.]?[A-Z]{{1,3}}[ \-.]?\d{{3,4}}(?![A-Za-z0-9])"
                          r"|(?<![A-Za-z0-9])\d{2}[ \-]?BH[ \-]?\d{4}[ \-]?[A-Z]{1,2}(?![A-Za-z0-9])")),
    # Passport: one letter + 7 digits
    ("GOV_ID", re.compile(r"(?<![A-Za-z0-9])[A-Z][ ]?[1-9]\d{6}(?![A-Za-z0-9])")),
    # UPI VPA ("ravi.k@okaxis", "9876543210@ybl"): a known PSP handle, or a local part that carries a digit / dot
    # (bounded local part keeps it linear; the digit/dot condition for unknown handles is checked in code)
    ("UPI", re.compile(r"(?<![\w.+\-])[A-Za-z0-9][A-Za-z0-9._\-]{1,48}@[A-Za-z]{2,20}(?![\w@]|\.[A-Za-z])")),
    # Masked phone numbers ("98XXXXXX21", "+91-98****4321")
    ("PHONE_MASKED", re.compile(r"(?<![\w*])(?:\+?91[ \-]?)?[6-9][0-9]{1,5}[Xx*•#]{3,7}[0-9]{2,5}(?![\w*])")),
]
# Labelled identifiers whose value has no fixed shape (bank account, ration card, voter ID, licence, passport ...)
_LABELLED_GOV_ID_RE = re.compile(
    r"(?:\b(?i:bank[ \t]*(?:a/?c|acc(?:oun)?t)|a/c|acc(?:oun)?t|ration[ \t]*card|voter[ \t]*(?:id|card)|epic|passport"
    r"|driving[ \t]*licen[cs]e|licen[cs]e|employee[ \t]*id|emp[ \t]*id|staff[ \t]*id|member[ \t]*id|claim|policy)"
    r"(?:[ \t]*(?i:no\.?|number|num|#|id))?|\bDL[ \t]*(?i:no\.?|number|#)|खाता[ \t]*(?:संख्या|सं\.?)?)"
    r"[ \t]*[:#\-=]?[ \t]*([A-Z0-9](?:[A-Z0-9/\-]|[ ](?=[A-Z0-9/\-]*[0-9])){3,24}[A-Z0-9])"
)
# Bare long digit runs (bank accounts, card / scheme numbers) that no other pattern claimed
_LONG_DIGITS_RE = re.compile(r"(?<![0-9A-Za-z.])[0-9]{11,18}(?![0-9A-Za-z]|\.[0-9])")
_CODE_CTX_RE = re.compile(r"(?i)(?:snomed|sct|loinc|code|coding|system|isbn|gtin|ean|hsn|barcode|batch|lot|timestamp|epoch|ts)\W{0,4}$")
_COMPACT_TS_RE = re.compile(r"^(?:19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])(?:[01]\d|2[0-3])[0-5]\d(?:[0-5]\d)?(?:\d{1,4})?$")


def _is_timestamp_digits(d: str) -> bool:
    return len(d) in (12, 14, 15, 16, 17, 18) and bool(_COMPACT_TS_RE.match(d))


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

    for kind, rx in OTHER_ID_PATTERNS:
        for m in rx.finditer(shadow):
            if _in_uuid(m.start(), m.end()):
                continue
            raw_surf = text[m.start() : m.end()]
            if kind == "PHONE_MASKED" and len(re.sub(r"^\+?91[ \-]?", "", re.sub(r"[ \-]", "", shadow[m.start() : m.end()]))) != 10:
                continue
            if kind == "UPI" and not _is_upi(shadow[m.start() : m.end()]):
                continue
            out_kind = "GOV_ID" if kind == "UPI" else kind
            spans.append(Span(m.start(), m.end(), out_kind, raw_surf, script=detect_script(raw_surf)))

    for m in _LABELLED_GOV_ID_RE.finditer(shadow):
        s, e = m.start(1), m.end(1)
        if _in_uuid(s, e) or sum(ch.isdigit() for ch in m.group(1)) < 4:
            continue
        spans.append(Span(s, e, "GOV_ID", text[s:e], script=detect_script(text[s:e])))

    for m in _LONG_DIGITS_RE.finditer(shadow):
        s, e = m.start(), m.end()
        if _in_uuid(s, e) or _is_timestamp_digits(m.group(0)) or _CODE_CTX_RE.search(shadow[max(0, s - 24) : s]):
            continue
        spans.append(Span(s, e, "GOV_ID", text[s:e], script=detect_script(text[s:e])))

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

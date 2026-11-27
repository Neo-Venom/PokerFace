"""Multilingual EN/HI/Hinglish context NER, canary catcher, and Harvest-Then-Sweep engine."""
from __future__ import annotations

from bisect import bisect_left
import json
import regex as re
from typing import Any

from data.gazetteers import (
    COMMON_INDIAN_DEVA_NAMES,
    COMMON_INDIAN_NAMES,
    HINGLISH_STOPWORDS,
    HONORIFICS_AFTER,
    HONORIFICS_BEFORE,
    KNOWN_GEO_DISTRICTS_STATES,
)
from data.clinical_vocab import CLINICAL_ABBREVIATIONS, NON_NAME_TITLE_WORDS, ROLE_WORDS
from src.models import FileEntry, Span
from src.nlp.indic_norm import GENERIC_WEAK_TOKENS, deobfuscate_unicode, detect_script, name_tokens, normalize_name_token, romanize_token, shadow_normalize, unquote_percent_text
from src.nlp.medical_guard import MedicalGuard
from src.nlp.pii_regex import extract_regex_spans

_NORM_INDIAN_NAMES = {normalize_name_token(n) for n in COMMON_INDIAN_NAMES if len(normalize_name_token(n)) >= 3}

# ------------------------------------------------------------------ Context patterns
_LABEL_NAME_RE = re.compile(
    r"(?i)(?:(?:\b(?:Patient(?:'s)?[ \t]+Name|Pt\.?[ \t]*(?:Full[ \t]+)?Name|Name|Naam|Patient|Pt|Beneficiary|Insured(?:[ \t]+Person)?|Member|Rogi|Client)|रोगी|मरीज़[ \t]+का[ \t]+नाम|मरीज[ \t]+का[ \t]+नाम|रोगी[ \t]+का[ \t]+नाम|नाम)[ \t]*[:\-][ \t]*"
    r"|\bReport[ \t]+of[ \t]+)"
    r"([^\n,;|\[\]]{2,55})"
)
_LABEL_CUT_RE = re.compile(
    r"(?i)[ \t]+(?:age(?:/sex)?|sex|gender|umar|reg(?:\.?|[ \t]*no)|uhid|ipd|opd|date|dob|mob(?:ile)?|phone|\d{1,3}[ \t]*(?:/[ \t]*[MF]|yrs?|years?|y\b|वर्ष)|आयु|उम्र|पंजीकरण)\b"
)

_REL_PAT_RE = re.compile(
    r"(?i)(?:\b(?:S/o|D/o|W/o|son[ \t]+of|daughter[ \t]+of|wife[ \t]+of)[ \t]*[:\-]?[ \t]*"
    r"|\b(?:attendant|guardian|caregiver|care[ \t]*taker|informant|bystander|escort|visitor|nominee|caller|sender"
    r"|receiver|recipient|applicant|claimant|witness|next[ \t]+of[ \t]+kin|emergency[ \t]+contact|contact[ \t]+person)"
    r"(?:[ \t]+name)?[ \t]*[:\-][ \t]*"
    r"|पिता[ \t]+का[ \t]+नाम[ \t]*[:\-]?[ \t]*|पति[ \t]+का[ \t]+नाम[ \t]*[:\-]?[ \t]*|पुत्र[ \t]+|पुत्री[ \t]+|पत्नी[ \t]+)"
    r"([A-Z\u0900-\u097f][A-Za-z\u0900-\u097f.'\u2019\- \t]{1,45}?)(?=,|\n|;|\)|\(|\.?[ \t]*(?:mob|phone|age|h\.?no|house|makan|ward|mohalla|near|vill|gram|tehsil|dist|informed|\d)|$)"
)

_BEFORE_REL_RE = re.compile(
    r"([A-Z\u0900-\u097f][A-Za-z\u0900-\u097f.]{1,25}(?:[ \t]+[A-Z\u0900-\u097f][A-Za-z\u0900-\u097f.]{1,25}){0,2})[ \t]*,[ \t]*(?:S/o|D/o|W/o)\b",
    re.I,
)

_CO_REL_RE = re.compile(
    r"(?i)\bC/O[ \t]*[:\-]?[ \t]*([A-Z\u0900-\u097f][A-Za-z\u0900-\u097f. \t]{1,40}?)(?=,|\n|;|\)|\.|\b(?:H\.?No|House|Makan|Ward|Mohalla|near|Vill|Gram)\b|$)"
)

_HONORIFIC_AFTER_RE = re.compile(
    r"(?i)\b([A-Za-z\u0900-\u097f]{2,25}(?:[ \t]+[A-Za-z\u0900-\u097f]{2,25}){0,1})[ \t]+"
    r"(?:ji|jee|sahab|saab|sahib|saheb|didi|behenji|bahenji|bhai|bhaiya|babu|amma|ammi|papa|chacha|chachu|chachi|garu|anna"
    r"|akka|uncle|aunty|auntie|aunti|mausi|masi|mami|nani|dadi|bua|tai|tau|kaka|kaki|appa|thambi|chechi|chetta|chettan"
    r"|ettan|boudi|mashi|bhau|bhabhi|bhabi|begum|miyan|mian"
    r"|जी|साहब|साहेब|दीदी|बहनजी|भैया|भाई|बाबू|अम्मा|चाचा|चाची|मौसी|मामी|दादी|नानी|आंटी|अंकल|भाभी)(?![\wऀ-ॿ])"
)

# Mixed-script names ("लखन Lepcha", "Rajesh कुमार") are tried first, then Latin title-case names.
_MIXED_TOKS = r"[ऀ-ॿ]{2,20}[ \t]+[A-Z][a-zA-Z'\-]{1,24}|[A-Z][a-zA-Z'\-]{1,24}[ \t]+[ऀ-ॿ]{2,20}(?![ऀ-ॿ])"
_TITLE_TOKS = (r"(?:" + _MIXED_TOKS + r"|(?:(?:Md|Mohd|Mohammad)\.?[ \t]+)?[A-Z][a-zA-Z'\-]{1,24}"
               r"(?:[ \t]+(?:[A-Z][a-zA-Z'\-]{1,24}|[A-Z]\.))?(?:[ \t]+[A-Z][a-zA-Z'\-]{1,24}){0,1})")
_DEVA_TOKS = r"[\u0900-\u097f]{2,20}(?:[ \t]+[\u0900-\u097f]{2,20}){0,2}"

_HONORIFIC_BEFORE_RE = re.compile(
    r"\b(?i:Shri|Sri|Smt\.?|Km\.?|Kumari|Mr\.?|Mrs\.?|Ms\.?|Master|Pt\.?|Patient|Late|Baby[ \t]+of|B/o|Mareez|Mariz|Marij|Rogi)[ \t]+"
    r"((?:[A-Z]\.[ \t]*){0,3}" + _TITLE_TOKS + r")"
    r"|(?:श्री|श्रीमती|कुमारी|स्व\.)[ \t]*(" + _DEVA_TOKS + r")"
)

# Name directly followed by an age/sex marker: "Thiruvenkadam (62M)", "Kezhokhoto Sema (aged 44)", "Ravi, 34 yrs"
_AGE_ANCHOR_NAME_RE = re.compile(
    r"(" + _TITLE_TOKS + r")[ \t]*(?:\([ \t]*(?:aged?[ \t]*)?\d{1,3}[ \t]*(?:Y|y|yrs?|years?)?[ \t]*/?[ \t]*(?:[MFmf]\b)?[ \t]*\)"
    r"|,[ \t]*\d{1,3}[ \t]*(?:yrs?|years?|Y)\b|,[ \t]*(?:a[ \t]+)?\d{1,3}-year-old"
    r"|[ \t]+\d{1,3}[ \t]*(?:Y|y|yrs?|years?)?[ \t]*/[ \t]*[MFmf]\b|,?[ \t]+[MF][ \t]*/[ \t]*\d{1,3}\b)"
)
# "RAMACHANDRAN K, 58/M", "SUBRAMANIAM, Kalyani  F/62", "MANJUNATH GOWDA M/45"
_CAPS_AGE_NAME_RE = re.compile(
    r"(?<![A-Za-z0-9])([A-Z]{3,20}(?:[ \t]+[A-Z]{1,20}\.?){0,2}(?:,[ \t]*[A-Z][a-z]{2,20})?)[ \t]*,?[ \t]*"
    r"(?:\(?\d{1,3}[ \t]*(?:Y|YRS?|yrs?)?[ \t]*/[ \t]*[MFmf]\b|\(?[MF][ \t]*/[ \t]*\d{1,3}\b)"
)
# Hinglish frames around a bare name: "Anbarasi se baat hui", "Sreeja ki mummy", "Jeevanandham aur unki wife",
# "Pt ka naam Kannan hai", "this is Arokiaraj from Bed 9"
_HINGLISH_FRAME_RE = re.compile(
    r"\b([A-Z][a-z]{2,20}(?:[ \t]+[A-Z][a-z]{2,20})?)[ \t]+(?:(?i:ji|jee)[ \t]+)?"
    r"(?i:se[ \t]+(?:baat|mil|pooch|puch)|ne[ \t]+(?:bola|boli|bataya|batayi|kaha|kahi|call|phone|fone|sign|pay|form|consent)"
    r"|aur[ \t]+(?:unki|unke|unka|uski|uske|uska)|ki[ \t]+(?:mummy|mother|maa|mata|wife|beti|behen|bahen|bahu|saas|nani|dadi)"
    r"|ke[ \t]+(?:papa|father|pita|pati|husband|bete|beta|bhai|bhaiya|sasur|chacha|mama|nana|dada|ghar|gharwale)"
    r"|ka[ \t]+(?:beta|bhai|pati|ghar|phone|fone|number|mobile))\b"
    r"|\b(?i:naam|name[ \t]+is|named|this[ \t]+is|myself|my[ \t]+name[ \t]+is)[ \t]+([A-Z][a-z]{2,20}(?:[ \t]+[A-Z][a-z]{2,20})?)"
)
_DEVA_FRAME_RE = re.compile(
    r"(?<![\u0900-\u097f])([\u0900-\u097f]{3,20})[ \t]+(?:जी[ \t]+)?"
    r"(?:ने[ \t]+(?:बताया|कहा|बोला|बोली|फोन|फ़ोन)|से[ \t]+बात|की[ \t]+(?:माँ|मां|मम्मी|पत्नी|बेटी|बहन)|के[ \t]+(?:पिता|पापा|पति|बेटे|भाई))"
)
_DEVA_NON_NAME = {
    "डॉक्टर", "डॉ", "डाक्टर", "मरीज", "मरीज़", "रोगी", "नर्स", "सिस्टर", "स्टाफ", "परिजन", "परिवार", "पति", "पत्नी", "बेटे",
    "बेटा", "बेटी", "माँ", "मां", "पिता", "भाई", "बहन", "उन्होंने", "उसने", "इन्होंने", "मैंने", "हमने", "सबने", "किसी", "कोई",
    "घरवालों", "लोगों", "आशा", "एएनएम", "डॉक्टरों", "नर्सों", "उन", "उस", "इस", "जिस", "सभी", "दोनों", "पड़ोसी", "मालिक",
    "ड्राइवर", "गार्ड", "कर्मचारी", "अस्पताल", "विभाग", "रिसेप्शन", "काउंटर", "हेल्पडेस्क",
}

# Relation noun + name: "brought by son Arivazhagan", "unke pati Zothanmawia ne", "Attendant Venkatesh Murmu"
# (the trailing \b keeps "pati" from firing inside "IN-PATIENT" and "son" inside "SONOGRAPHY")
_REL_NOUN_RE = re.compile(
    r"\b(?i:son|daughter|husband|wife|brother|sister|mother|father|nephew|niece|grandson|granddaughter|uncle|aunt"
    r"|beta|beti|bete|pati|patni|bhai|behen|bahen|bhabhi|devar|sasur|saas|chacha|mama|bahu|damad|jija"
    r"|attendant|relative|guardian|caretaker|nok|kin)\b"
    r"(?:[ \t]+(?i:of|named|is))?[ \t]*[:\-]?[ \t]*(" + _TITLE_TOKS + r")"
    r"|(?:पिता|पति|पत्नी|पुत्र|पुत्री|बेटा|बेटी|बेटे|बहू|दामाद|पोता|पोती|भतीजा|भांजा|माता|भाई|बहन|भाभी|देवर|ससुर|सास"
    r"|अभिभावक|परिजन)(?![ऀ-ॿ])[ \t]*[:\-]?[ \t]*(" + _DEVA_TOKS + r")"
)

# "Bed 7: Kezhokhoto Sema", "Bed 12 pe Ravi Kumar"
_BED_NAME_RE = re.compile(r"\b(?i:bed)(?:[ \t]*(?i:no)\.?)?[ \t]*\d{1,3}[ \t]*(?:[:\-–]|(?i:pe|par|par\b))[ \t]*(" + _TITLE_TOKS + r")")

# "a 97-year-old man from Hirakhera Chhota" -> village (an age number must precede: "Yr of Birth" is a label)
_FROM_PLACE_RE = re.compile(
    r"(?:\d{1,3}[ \t-]*(?i:year[- ]old|yrs?|years?)\b|(?i:aged)[ \t]+\d{1,3}(?:[ \t]*(?i:yrs?|years?))?)[ \t]+"
    r"(?i:(?:man|woman|male|female|boy|girl|patient|lady|gentleman)[ \t]+)?(?i:from|of)[ \t]+"
    r"([A-Z][a-zA-Z]{2,24}(?:[ \t]+[A-Z][a-zA-Z]{2,24}){0,2})"
)

_DEVA_STOP = {"ने", "को", "का", "की", "के", "से", "है", "हैं", "और", "या", "में", "पर", "जी", "ने", "था", "थी", "थे", "बताया", "आए", "आई"}

_POSTPOS_NAME_RE = re.compile(
    r"([A-Z\u0900-\u097f][a-zA-Z\u0900-\u097f]{2,25}(?:[ \t]+[A-Z\u0900-\u097f][a-zA-Z\u0900-\u097f]{2,25})?)[ \t]+"
    r"(?i:ka|ki|ke|ko|को|का|की|के)[ \t]+(?i:\d+[ \t]+(?:दिन|days)|BP|pulse|temp|condition|dawai|bukhar|sugar|report|phone|बुखार)\b"
)

_PERSON_VERB_RE = re.compile(
    r"(?:\b(?i:called[ \t]+by|witnessed[ \t]+by|raised[ \t]+by|accompanied[ \t]+by|informed[ \t]+by|brought[ \t]+by|sample[ \t]+received[ \t]+from"
    r"|signed[ \t]+by|consent(?:ed)?[ \t]+by|given[ \t]+by|issued[ \t]+to|handed[ \t]+over[ \t]+to|picked[ \t]+up|dropped[ \t]+off|received[ \t]+from"
    r"|visited[ \t]+by|enquiry[ \t]+(?:by|from)|complaint[ \t]+(?:by|from)|spoke[ \t]+(?:to|with)|refund[ \t]+to|registered[ \t]+by"
    r"|relieved[ \t]+by|escorted[ \t]+by|collected[ \t]+by|received[ \t]+by|paid[ \t]+by|deposited[ \t]+by|submitted[ \t]+by"
    r"|requested[ \t]+by|pass[ \t]+issued[ \t]+to|security[ \t]+guard|ward[ \t]*boy|watchman|driver|sweeper)|'s[ \t]+attendant)"
    r"[ \t]*[:\-]?[ \t]*"
    r"(" + _TITLE_TOKS + r")"
)

_POSSESSIVE_RE = re.compile(r"\b([A-Z][a-z]{2,24})'s\b")

_WA_SENDER_RE = re.compile(
    r"^\[?\d{1,2}[./-]\d{1,2}[./-]\d{2,4},\s*\d{1,2}:\d{2}(?:\s*[ap]m)?\]?\s*(?:-|\])?\s*([^:\n]{2,45}):",
    re.M | re.I,
)

_PRIORITY = {
    "ABHA_NUMBER": 100,
    "ABHA_ADDRESS": 99,
    "EMAIL": 98,
    "PAN": 97,
    "AADHAAR_MASKED": 96,
    "AADHAAR": 95,
    "PHONE": 94,
    "PHONE_MASKED": 93,
    "INSURANCE_ID": 93,
    "MRN": 92,
    "GOV_ID": 91,
    "DOB": 90,
    "PIN": 85,
    "HOUSE": 80,
    "LOCALITY": 79,
    "LANDMARK": 78,
    "VILLAGE": 77,
    "RELATIVE_NAME": 70,
    "PATIENT_NAME": 65,
    "ADDRESS": 60,
}


# Words that are labels / facility words, never part of a person's name
LABEL_STOP = {
    "age", "sex", "gender", "name", "naam", "patient", "pt", "tab", "tablet", "inj", "injection", "cap", "capsule",
    "syp", "syrup", "dr", "ward", "bed", "dist", "district", "vill", "village", "tal", "taluka", "tehsil", "ph",
    "mob", "mobile", "phone", "uhid", "ipd", "opd", "reg", "no", "date", "dob", "kin", "address", "add", "hno",
    "house", "near", "opp", "pin", "ref", "rx", "dx", "diagnosis", "visit", "admitted", "discharged", "status",
    "remarks", "notes", "relation", "contact", "email", "aadhaar", "abha", "id", "yrs", "years", "parchi",
    "summary", "report", "forwarded", "attachment", "consent", "witness", "nursing", "shift", "note", "notes",
    "unit", "round", "department", "dept", "section", "floor", "room", "opd", "icu", "casualty", "emergency",
    "medicine", "surgery", "obstetrics", "gynaecology", "paediatrics", "orthopaedics", "general", "block",
    "full", "fullname", "first", "last", "given", "family", "surname", "middle", "names", "code", "num", "number",
    # address head nouns and admin words: labels and places, never a person's name
    "mohalla", "nagar", "colony", "gali", "tola", "basti", "chowk", "peth", "wadi", "para", "gram", "gaon", "town",
    "city", "locality", "area", "sector", "road", "marg", "lane", "street", "landmark", "post", "state", "zila", "jila",
    "pincode", "postal", "residence", "abode", "hamlet", "subdiv", "panchayat",
    # document / form words ("IN-PATIENT DISCHARGE RECORD", "Patient Registration Form")
    "discharge", "admission", "record", "records", "slip", "form", "sheet", "register", "registration", "details",
    "information", "info", "copy", "file", "card", "list", "chart", "log", "entry",
}
FACILITY_WORDS = {
    "hospital", "hospitals", "trust", "clinic", "chikitsalaya", "sevashram", "mission", "diagnostics", "diagnostic",
    "pathology", "lab", "labs", "laboratory", "nursing", "college", "centre", "center", "institute", "foundation",
    "dispensary", "phc", "chc", "sadan", "ashram", "pharmacy", "medicals", "अस्पताल", "चिकित्सालय",
}


# "ASHA", "ANM", "AWW" written in capitals are roles; "Asha" in title case stays a possible given name
_CAPS_ROLE_ACRONYMS = frozenset(CLINICAL_ABBREVIATIONS | {"asha"})


def _is_hospital_fragment(joined_low: str, guard: MedicalGuard) -> bool:
    if any(w in FACILITY_WORDS for w in joined_low.split()):
        return True
    return any(h and (joined_low in h or h in joined_low) for h in guard.hospital_names)


def _is_valid_name_candidate(raw: str, guard: MedicalGuard) -> bool:
    toks = [t.strip(".,:;()[]'\"") for t in raw.split() if t.strip(".,:;()[]'\"")]
    if not toks:
        return False
    if _is_hospital_fragment(" ".join(toks).lower(), guard):
        return False
    # Reject if every token is a clinical/stopword/geo term
    meaningful = []
    for t in toks:
        lt = t.lower()
        if lt in HONORIFICS_BEFORE or lt in HONORIFICS_AFTER:
            continue
        if lt in HINGLISH_STOPWORDS or guard.is_clinical_word(t):
            continue
        if lt in KNOWN_GEO_DISTRICTS_STATES:
            continue
        if lt in ROLE_WORDS or lt in NON_NAME_TITLE_WORDS or (t.isupper() and t.lower() in _CAPS_ROLE_ACRONYMS):
            continue
        if len(t) >= 2:
            meaningful.append(t)
    if not meaningful:
        return False
    # Reject hospital names
    joined = " ".join(toks).lower()
    if joined in KNOWN_GEO_DISTRICTS_STATES:
        return False
    if any(h in joined for h in guard.hospital_names if h):
        return False
    return True


def extract_context_name_spans(text: str, guard: MedicalGuard, anchor_spans: list[Span]) -> list[Span]:
    """Extract EN/HI/Hinglish person & relative names via labels, relations, honorifics, colocation."""
    spans: list[Span] = []

    def _add(start: int, end: int, kind: str) -> None:
        raw = text[start:end].strip(" .,;:\t")
        if not raw or len(raw) < 2:
            return
        # Trim leading/trailing whitespace accurately
        lpad = len(text[start:end]) - len(text[start:end].lstrip(" .,;:\t"))
        rpad = len(text[start:end]) - len(text[start:end].rstrip(" .,;:\t"))
        s, e = start + lpad, end - rpad
        if s >= e:
            return
        # Trim label / facility words off both ends ("Ishani    Age" -> "Ishani", "Tab Metformin" -> rejected)
        while True:
            frag = text[s:e]
            parts = list(re.finditer(r"\S+", frag))
            if not parts:
                return
            first_w = parts[0].group(0).strip(".,:;()[]'\"").lower()
            last_w = parts[-1].group(0).strip(".,:;()[]'\"").lower()
            if first_w in LABEL_STOP:
                s = s + (parts[1].start() if len(parts) > 1 else len(frag))
                continue
            if last_w in LABEL_STOP and len(parts) > 1:
                e = s + parts[-2].end()
                continue
            break
        if s >= e:
            return
        frag = text[s:e]
        # "Brahmaputra Care Trust Hospital": a run right before a facility word is a hospital name
        nxt = re.match(r"[ \t]+([A-Za-z\u0900-\u097f]+)", text[e : e + 40])
        if nxt and nxt.group(1).lower() in FACILITY_WORDS:
            return
        if _is_valid_name_candidate(frag, guard):
            spans.append(Span(s, e, kind, frag, script=detect_script(frag)))

    # 1. Labelled names ("Patient Name:", "नाम:", "Report of")
    for m in _LABEL_NAME_RE.finditer(text):
        seg = m.group(1)
        cut = _LABEL_CUT_RE.search(seg)
        end_off = m.start(1) + (cut.start() if cut else len(seg))
        _add(m.start(1), end_off, "PATIENT_NAME")

    # 2. Relations (S/o, D/o, W/o, Attendant, पिता का नाम, etc.)
    for m in _REL_PAT_RE.finditer(text):
        _add(m.start(1), m.end(1), "RELATIVE_NAME")

    for m in _BEFORE_REL_RE.finditer(text):
        _add(m.start(1), m.end(1), "PATIENT_NAME")

    # 3. C/O relative vs complaint
    for m in _CO_REL_RE.finditer(text):
        rhs = m.group(1).strip()
        rhs_toks = [w.lower() for w in re.findall(r"[A-Za-z\u0900-\u097f]{2,}", rhs)]
        if rhs_toks and not any(guard.is_clinical_word(w) for w in rhs_toks[:2]):
            _add(m.start(1), m.end(1), "RELATIVE_NAME")

    # 4. Honorifics & Hinglish postpositions ("Sukhwinder ji", "Sunita ka BP", "मोहम्मद शाहिद को 9 दिन...")
    for m in _HONORIFIC_AFTER_RE.finditer(text):
        cand = m.group(1)
        # Filter leading stopwords like "Report of Asha" -> keep Asha
        parts = cand.split()
        while parts and (parts[0].lower() in HINGLISH_STOPWORDS or guard.is_clinical_word(parts[0])
                         or (parts[0].isupper() and parts[0].lower() in _CAPS_ROLE_ACRONYMS)):
            offset = len(parts[0]) + 1
            parts.pop(0)
            cand = cand[offset:]
        if parts:
            start_pos = m.start(1) + (len(m.group(1)) - len(cand))
            _add(start_pos, m.end(1), "PATIENT_NAME")

    for m in _HONORIFIC_BEFORE_RE.finditer(text):
        g = 1 if m.group(1) else 2
        _add(m.start(g), m.end(g), "PATIENT_NAME")

    for m in _AGE_ANCHOR_NAME_RE.finditer(text):
        _add(m.start(1), m.end(1), "PATIENT_NAME")

    for m in _CAPS_AGE_NAME_RE.finditer(text):
        _add(m.start(1), m.end(1), "PATIENT_NAME")

    for m in _HINGLISH_FRAME_RE.finditer(text):
        g = 1 if m.group(1) else 2
        st, en = m.start(g), m.end(g)
        toks = text[st:en].split()
        # a sentence-initial function word is not part of the name ("Kal Sukanya se baat hui")
        while toks and (toks[0].lower() in HINGLISH_STOPWORDS or toks[0].lower() in ROLE_WORDS):
            st = text.find(toks[1], st) if len(toks) > 1 else en
            toks = toks[1:]
        if toks:
            _add(st, en, "PATIENT_NAME")

    if "ने" in text or "से" in text or "की" in text or "के" in text:
        for m in _DEVA_FRAME_RE.finditer(text):
            w = m.group(1)
            if w not in _DEVA_NON_NAME and w not in _DEVA_STOP and w not in HINGLISH_STOPWORDS and not guard.is_clinical_word(w):
                _add(m.start(1), m.end(1), "PATIENT_NAME")

    for m in _BED_NAME_RE.finditer(text):
        _add(m.start(1), m.end(1), "PATIENT_NAME")

    for m in _REL_NOUN_RE.finditer(text):
        g = 1 if m.group(1) else 2
        st, en = m.start(g), m.end(g)
        if g == 2:  # trim trailing Hindi function words ("रमेश ने बताया")
            toks = text[st:en].split()
            while toks and toks[-1] in _DEVA_STOP:
                toks.pop()
            if not toks:
                continue
            en = st + len(" ".join(toks)) if text[st:en].startswith(" ".join(toks)) else en
        _add(st, en, "RELATIVE_NAME")

    for m in _FROM_PLACE_RE.finditer(text):
        frag = m.group(1)
        if not guard.is_clinical_word(frag.split()[0]):
            spans.append(Span(m.start(1), m.end(1), "VILLAGE", frag, script=detect_script(frag)))

    for m in _POSTPOS_NAME_RE.finditer(text):
        cand = m.group(1)
        parts = cand.split()
        while parts and (parts[0].lower() in HINGLISH_STOPWORDS or guard.is_clinical_word(parts[0])):
            parts.pop(0)
        if parts:
            joined = " ".join(parts)
            pos = text.find(joined, m.start(1), m.end(1))
            if pos >= 0:
                _add(pos, pos + len(joined), "PATIENT_NAME")

    # 5. Person verbs & possessive ('s) -> catches canaries & "Harish's attendant"
    for m in _PERSON_VERB_RE.finditer(text):
        _add(m.start(1), m.end(1), "PATIENT_NAME")

    for m in _POSSESSIVE_RE.finditer(text):
        word = m.group(1)
        if not guard.is_clinical_word(word) and not guard.is_clinical_word(word + "'s"):
            if word.lower() not in HINGLISH_STOPWORDS and word.lower() not in KNOWN_GEO_DISTRICTS_STATES:
                _add(m.start(1), m.end(1), "PATIENT_NAME")

    # 6. WhatsApp forward sender header
    for m in _WA_SENDER_RE.finditer(text):
        sender = m.group(1).strip()
        if not re.match(r"(?i)^(?:dr\.?|doctor|डॉ\.?)\b", sender):
            _add(m.start(1), m.end(1), "RELATIVE_NAME")

    # 7. Colocation Canary Catcher: Title-Case 2-3 word spans near phones/IDs/MRNs/attendant/callback
    anchor_positions = [sp.start for sp in anchor_spans if sp.kind in ("PHONE", "AADHAAR", "ABHA_NUMBER", "PAN", "EMAIL", "MRN")]
    for km in re.finditer(r"(?i)\b(?:mob(?:ile)?|callback|attendant|witnessed|called)\b", text):
        anchor_positions.append(km.start())
    if anchor_positions:
        for m in re.finditer(r"\b([A-Z][a-z]{2,18}\s+[A-Z][a-z]{2,18}(?:\s+[A-Z][a-z]{2,18})?)\b", text):
            if any(abs(m.start() - ap) <= 90 for ap in anchor_positions):
                if any(guard.is_clinical_word(w) or w.lower() in LABEL_STOP for w in m.group(1).split()):
                    continue
                _add(m.start(1), m.end(1), "PATIENT_NAME")

    # 8. Two-token Pan-India gazetteer match (Latin & Devanagari)
    for m in re.finditer(r"\b([A-Za-z]{3,18})\s+([A-Za-z]{3,18})\b", text):
        w1, w2 = m.group(1).lower(), m.group(2).lower()
        if w1 in COMMON_INDIAN_NAMES and w2 in COMMON_INDIAN_NAMES:
            if not (guard.is_clinical_word(w1) or guard.is_clinical_word(w2)):
                _add(m.start(), m.end(), "PATIENT_NAME")

    for m in re.finditer(r"([\u0900-\u097f]{2,18})\s+([\u0900-\u097f]{2,18})", text):
        w1, w2 = m.group(1), m.group(2)
        if w1 in COMMON_INDIAN_DEVA_NAMES and w2 in COMMON_INDIAN_DEVA_NAMES:
            _add(m.start(), m.end(), "PATIENT_NAME")
        elif (
            w1 not in HINGLISH_STOPWORDS
            and w2 not in HINGLISH_STOPWORDS
            and not guard.is_clinical_word(w1)
            and not guard.is_clinical_word(w2)
        ):
            n1, n2 = normalize_name_token(w1), normalize_name_token(w2)
            if len(n1) >= 3 and len(n2) >= 3 and n1 in _NORM_INDIAN_NAMES and n2 in _NORM_INDIAN_NAMES:
                _add(m.start(), m.end(), "PATIENT_NAME")

    return spans


class PiiHarvester:
    """Pass-1 harvested registry of exact names, tokens, digit cores, and address strings."""

    def __init__(self, guard: MedicalGuard) -> None:
        self.guard = guard
        self.full_names: set[str] = set()
        self.name_tokens: set[str] = set()
        self.digit_cores: set[str] = set()
        self.address_phrases: set[str] = set()
        self.exact_ids: set[str] = set()
        self.masked_aadhaar_last4: set[str] = set()
        self._sweep_rx: re.Pattern[str] | None = None
        self._digit_rx: re.Pattern[str] | None = None
        # Exact pre-filters (same results, far fewer regex scans): every sweep match starts with a whole word that is
        # the first word of some phrase, and every digit-core match is a whole maximal digit run.
        self._sweep_first: dict[str, list[tuple[str, int]]] | None = None
        self._sweep_odd: dict[str, list[tuple[str, int]]] = {}
        self._sweep_odd_rx: re.Pattern[str] | None = None
        self._digit_set: frozenset[str] = frozenset()

    def add_name(self, name: str) -> None:
        clean = " ".join(name.strip(" .,;:\t()[]'\"").split())
        if len(clean) < 3:
            return
        if _is_hospital_fragment(clean.lower(), self.guard):
            return
        toks = [t.strip(".,:;()[]'\"") for t in clean.split()]
        # Skip if doctor prefix
        if toks and toks[0].lower().rstrip(".") in ("dr", "doctor", "डॉ"):
            return
        meaningful = [
            t for t in toks
            if len(t) >= 3
            and t.lower() not in HINGLISH_STOPWORDS
            and t.lower() not in HONORIFICS_BEFORE
            and t.lower() not in HONORIFICS_AFTER
            and not self.guard.is_clinical_word(t)
            and t.lower() not in KNOWN_GEO_DISTRICTS_STATES
            and t.lower().strip(".") not in LABEL_STOP
        ]
        if len(meaningful) >= 2:
            self.full_names.add(clean)
            # Also add 2-token variant without honorifics
            self.full_names.add(" ".join(meaningful))
        for t in meaningful:
            tl = t.lower()
            if len(t) >= 4 and tl not in GENERIC_WEAK_TOKENS:
                self.name_tokens.add(t)

    def add_id(self, id_str: str) -> None:
        s = id_str.strip()
        if len(s) < 4:
            return
        self.exact_ids.add(s)
        digits = re.sub(r"\D", "", shadow_normalize(s))
        # Extract meaningful digit cores (>= 5 digits, skipping 4-digit year 2024/2025/2026)
        if len(digits) >= 5:
            self.digit_cores.add(digits)
            if len(digits) >= 6:
                self.digit_cores.add(digits[-6:])
        # Trailing numeric component after slash/dash (e.g. ABC2026-084999 -> 084999, XYZ/26/064545 -> 064545)
        for grp in re.findall(r"(\d{5,})", shadow_normalize(s)):
            self.digit_cores.add(grp)

    def add_address_piece(self, val: str) -> None:
        clean = " ".join(val.strip(" .,;:\t").split())
        if len(clean) >= 4 and clean.lower() not in KNOWN_GEO_DISTRICTS_STATES:
            if not self.guard.is_clinical_word(clean):
                self.address_phrases.add(clean)

    def compile(self) -> None:
        phrases: set[str] = set()
        for item in self.full_names | self.exact_ids | self.address_phrases | self.name_tokens:
            norm = shadow_normalize(item).strip()
            if len(norm) >= 4:
                phrases.add(norm)
        if phrases:
            # Longest first so multi-word names match before single tokens
            ordered = sorted(phrases, key=lambda s: (-len(s), s))
            # Phrases indexed by their first word, in alternation order: the same leftmost, first-alternative match as
            # the big regex below, but each word start only tries the phrases that can begin there.
            # Phrases that start outside the word class ("+91 ...") are indexed by their first character instead.
            by_first: dict[str, list[tuple[str, int]]] = {}
            odd: dict[str, list[tuple[str, int]]] = {}
            for ph in ordered:
                fm = _SWEEP_WORD_RX.match(ph)
                if fm is not None:
                    by_first.setdefault(fm.group(0).casefold(), []).append((ph.casefold(), len(ph)))
                elif len(ph[0].casefold()) == 1:
                    odd.setdefault(ph[0].casefold(), []).append((ph.casefold(), len(ph)))
                else:  # exotic first character: keep the plain regex path
                    by_first, odd = {}, {}
                    break
            self._sweep_first = by_first if (by_first or odd) else None
            self._sweep_odd = odd
            self._sweep_odd_rx = (
                re.compile("[" + "".join(re.escape(c) for c in sorted(odd)) + "]", re.I) if odd else None
            )
            escaped = [re.escape(p) for p in ordered]
            self._sweep_rx = re.compile(
                r"(?i)(?<![A-Za-z0-9\u0900-\u097f])(?:" + "|".join(escaped) + r")(?![A-Za-z0-9\u0900-\u097f])"
            )
        if self.digit_cores:
            # Match bare digit cores >= 5 digits (e.g. inside paths _084999 or filenames)
            d_ord = sorted((d for d in self.digit_cores if len(d) >= 5), key=lambda s: (-len(s), s))
            self._digit_set = frozenset(d_ord)
            if d_ord:
                self._digit_rx = re.compile(r"(?<!\d)(?:" + "|".join(re.escape(d) for d in d_ord) + r")(?!\d)")

    def sweep_spans(self, text: str) -> list[Span]:
        spans: list[Span] = []
        shadow = shadow_normalize(text)
        if self._sweep_rx is not None:
            if self._sweep_first is None:
                hits = [(m.start(), m.end()) for m in self._sweep_rx.finditer(shadow)]
            else:
                hits, pos, n = [], 0, len(shadow)
                # Candidate starts in text order: word starts (normal phrases) and odd first characters (preceded by
                # a non-word character, as the regex look-behind requires). A position belongs to one group only.
                starts = [(w.start(), self._sweep_first.get(w.group(0).casefold(), ())) for w in _SWEEP_WORD_RX.finditer(shadow)]
                if self._sweep_odd_rx is not None:
                    for c in self._sweep_odd_rx.finditer(shadow):
                        p0 = c.start()
                        if p0 == 0 or _SWEEP_WORD_RX.match(shadow[p0 - 1]) is None:
                            starts.append((p0, self._sweep_odd.get(c.group(0).casefold(), ())))
                    starts.sort(key=lambda x: x[0])
                for st, cands in starts:
                    if st < pos:
                        continue
                    for phf, ln in cands:
                        end = st + ln
                        if end <= n and shadow[st:end].casefold() == phf and _SWEEP_WORD_RX.match(shadow, end) is None:
                            hits.append((st, end))
                            pos = end
                            break
            for s, e in hits:
                raw = text[s:e]
                kind = "MRN" if raw in self.exact_ids else ("VILLAGE" if raw in self.address_phrases else "PATIENT_NAME")
                spans.append(Span(s, e, kind, raw, script=detect_script(raw)))
        if self._digit_rx is not None:
            for m in _DIGIT_RUN_RX.finditer(shadow):
                if m.group(0) in self._digit_set:
                    raw = text[m.start() : m.end()]
                    spans.append(Span(m.start(), m.end(), "MRN", raw, script=detect_script(raw)))
        return spans


_SWEEP_WORD_RX = re.compile(r"(?i)[A-Za-z0-9\u0900-\u097f]+")
_DIGIT_RUN_RX = re.compile(r"\d+")


def split_safe_chunks(text: str, max_len: int = 32768) -> list[str]:
    """Split huge single-line notes on sentence/whitespace boundaries to avoid quadratic scans."""
    if len(text) <= max_len:
        return [text]
    chunks: list[str] = []
    pos = 0
    n = len(text)
    while pos < n:
        if n - pos <= max_len:
            chunks.append(text[pos:])
            break
        hi = pos + max_len
        cut = text.rfind("\n", pos + max_len // 2, hi)
        if cut < 0:
            cut = text.rfind(". ", pos + max_len // 2, hi)
            if cut >= 0:
                cut += 2
        if cut < 0:
            cut = text.rfind(" ", pos + max_len // 2, hi)
            if cut >= 0:
                cut += 1
        if cut <= pos:
            cut = hi
        chunks.append(text[pos:cut])
        pos = cut
    return chunks


def _has_prot_overlap(prot: list[tuple[int, int, str]], s: int, e: int, tag: str, strict_inside: bool = False) -> bool:
    if not prot:
        return False
    idx = bisect_left(prot, (e, -1, ""))
    lo = max(0, idx - 64)
    for k in range(idx - 1, lo - 1, -1):
        ps, pe, ptag = prot[k]
        if ptag != tag:
            continue
        if strict_inside:
            if ps <= s and e <= pe:
                return True
        elif ps < e and pe > s:
            return True
    if idx > 64:
        for ps, pe, ptag in prot[: idx - 64]:
            if ptag != tag or pe <= s:
                continue
            if ps >= e:
                break
            if strict_inside:
                if ps <= s and e <= pe:
                    return True
            else:
                return True
    return False


def _resolve_chunk(
    text: str,
    guard: MedicalGuard,
    harvester: PiiHarvester | None = None,
    psn: str | None = None,
    replace_primary_id_with_psn: bool = False,
) -> tuple[str, list[Span]]:
    prot = guard.protected_spans(text)
    rx_spans = extract_regex_spans(text)
    ctx_spans = extract_context_name_spans(text, guard, rx_spans)
    sweep_spans = harvester.sweep_spans(text) if harvester is not None else []

    candidates = rx_spans + ctx_spans + sweep_spans
    filtered: list[Span] = []
    for sp in candidates:
        strict_veto = sp.kind in ("PATIENT_NAME", "RELATIVE_NAME", "PIN", "HOUSE", "LOCALITY", "LANDMARK", "VILLAGE")
        if guard.is_protected_span(sp.start, sp.end, prot, strict=strict_veto):
            if strict_veto:
                continue
            in_doc = _has_prot_overlap(prot, sp.start, sp.end, "DOCTOR", strict_inside=False)
            inside_med = _has_prot_overlap(prot, sp.start, sp.end, "MEDICAL", strict_inside=True)
            if sp.kind == "PHONE" and (in_doc or inside_med):
                continue
            if sp.kind != "PHONE" and (inside_med or (in_doc and sp.kind == "MRN")):
                continue
        if sp.kind == "PATIENT_NAME" and guard.is_clinical_word(sp.text):
            continue
        filtered.append(sp)

    # Sort by priority desc, length desc, start asc to select non-overlapping winners (O(K log K) via bisect)
    filtered.sort(key=lambda sp: (-_PRIORITY.get(sp.kind, 50), -(sp.end - sp.start), sp.start))
    chosen: list[Span] = []
    starts: list[int] = []
    ends: list[int] = []
    for sp in filtered:
        pos = bisect_left(starts, sp.start)
        if pos > 0 and ends[pos - 1] > sp.start:
            continue
        if pos < len(starts) and starts[pos] < sp.end:
            continue
        chosen.append(sp)
        starts.insert(pos, sp.start)
        ends.insert(pos, sp.end)

    chosen.sort(key=lambda sp: sp.start)
    if not chosen:
        return text, []

    parts: list[str] = []
    cursor = 0
    used_psn = False
    for sp in chosen:
        parts.append(text[cursor : sp.start])
        if replace_primary_id_with_psn and psn and sp.kind == "MRN" and not used_psn:
            repl = psn
            used_psn = True
        elif sp.kind in ("PATIENT_NAME",):
            repl = "[NAME]"
        elif sp.kind == "RELATIVE_NAME":
            repl = "[RELATIVE]"
        elif sp.kind in ("PHONE", "PHONE_MASKED"):
            repl = "[PHONE]"
        elif sp.kind == "EMAIL":
            repl = "[EMAIL]"
        elif sp.kind == "PIN":
            repl = sp.text[:3] + "***" if len(sp.text) == 6 else "[PIN]"
        elif sp.kind == "DOB":
            repl = "[DOB]"
        elif sp.kind in ("HOUSE", "LOCALITY", "LANDMARK", "VILLAGE", "ADDRESS"):
            repl = "[ADDRESS]"
        else:
            repl = "[ID]"
        parts.append(repl)
        cursor = sp.end
    parts.append(text[cursor:])
    return "".join(parts), chosen


def resolve_and_scrub(
    text: str,
    guard: MedicalGuard,
    harvester: PiiHarvester | None = None,
    psn: str | None = None,
    replace_primary_id_with_psn: bool = False,
) -> tuple[str, list[Span]]:
    """Run all detection layers, subtract protected medical/doctor spans, resolve overlaps, and rewrite."""
    if not text or not text.strip():
        return text, []
    text = unquote_percent_text(deobfuscate_unicode(text))
    if len(text) <= 49152:
        cl, sps = _resolve_chunk(text, guard, harvester, psn, replace_primary_id_with_psn)
        return _pin_after_address(cl), sps
    out_parts: list[str] = []
    all_spans: list[Span] = []
    offset = 0
    psn_left = replace_primary_id_with_psn
    for ch in split_safe_chunks(text, max_len=32768):
        cl, sps = _resolve_chunk(ch, guard, harvester, psn, psn_left)
        if psn_left and psn and psn in cl:
            psn_left = False
        out_parts.append(cl)
        for sp in sps:
            all_spans.append(Span(sp.start + offset, sp.end + offset, sp.kind, sp.text, script=sp.script))
        offset += len(ch)
    return _pin_after_address("".join(out_parts)), all_spans


# A 6-digit number directly after a removed village/locality ("Rampur 247451" -> "[ADDRESS] 247451") is its PIN.
_PIN_AFTER_ADDR_RE = re.compile(r"(\[ADDRESS\][ \t,\-–]*(?:(?i:pin(?:\s*code)?)[\s:\-]*)?)[1-8][0-9]{5}(?![0-9])")


def _pin_after_address(cleaned: str) -> str:
    return _PIN_AFTER_ADDR_RE.sub(r"\1[PIN]", cleaned) if "[ADDRESS]" in cleaned else cleaned

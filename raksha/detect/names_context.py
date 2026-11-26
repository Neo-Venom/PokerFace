"""Step D: Context-aware name, relative, and address detection in multilingual free text."""
from __future__ import annotations

import re

from raksha.detect.identifiers import Span, resolve_overlaps
from raksha.detect.normalize import (
    DEVA_LETTER_CLASS,
    LATIN_LETTER_CLASS,
    LATIN_UPPER_CLASS,
    NormalizedView,
    build_normalized_view,
    fold_latin_accents,
    is_deva_char,
    is_upper_or_deva_start,
)

# Unicode-aware Devanagari and Latin (including composed NFC accented letters; NFD is NFC-normalized first)
_DL = DEVA_LETTER_CLASS
_LU = LATIN_UPPER_CLASS
_LL = LATIN_LETTER_CLASS
_NL = rf"{_LU}{_DL}"
_AL = rf"{_LL}{_DL}'\-"

# Single name token: single-letter initial (only when followed by another token), Md./Mohd., or word without trailing sentence dot
_TOK = rf"(?:[{_NL}]\.(?=[ \t]+[{_NL}])|(?:Md|Mohd|Mr|Mrs|Ms)\.|[{_NL}][{_AL}]{{0,24}})"
_SYNTH_TOK = rf"(?:[{_LU}][a-z\u00DF-\u024F\u1E00-\u1EFF]{{4,20}}\d{{1,2}}[a-z]{{0,4}}|{_TOK})"
_NAME_SEQ = rf"{_TOK}(?:[ \t]+{_TOK}){{0,3}}"
_SLOT_SEQ = rf"{_SYNTH_TOK}(?:[ \t]+{_SYNTH_TOK}){{0,3}}"
_SHORT_SEQ = rf"{_TOK}(?:[ \t]+{_TOK}){{0,2}}"

# Protected doctor patterns (never redacted; strictly capitalized/Devanagari tokens after Dr./डॉ)
_DOCTOR_SPAN_RE = re.compile(
    rf"(?:\bDr\.?|डॉ\.?|डॉक्टर)[ \t]+(?:(?:Dr\.?|डॉ\.?)[ \t]+)?[{_NL}][{_AL}.]{{0,24}}(?:[ \t]+[{_NL}][{_AL}]{{1,24}}){{0,3}}"
    rf"|(?:\bConsultant|\bPrescriber|\bRef(?:erred)?[ \t]+by)[ \t]*[:\-][ \t]*(?:Dr\.?[ \t]*)?[{_NL}][{_AL}.]{{0,24}}(?:[ \t]+[{_NL}][{_AL}]{{1,24}}){{0,3}}"
)

# Protected disease eponyms and ASHA role (never redacted)
_EPONYM_RE = re.compile(
    r"(?i)\b(?:[A-Z][a-z]+(?:'s|s')?\s+(?:disease|syndrome|palsy|lymphoma|sign|phenomenon|triad|chorea|tetrad|ulcer|oesophagus|esophagus))"
    r"|\b(?:Parkinson(?:'s)?|Addison(?:'s)?|Bell(?:'s)?|Crohn(?:'s)?|Hodgkin(?:'s)?|Alzheimer(?:'s)?|Cushing(?:'s)?|Graves(?:'|s)?|Hashimoto(?:'s)?|Guillain[- ]Barre|Kawasaki|Marfan|Meniere(?:'s)?|Raynaud(?:'s)?|Sjogren(?:'s)?|Wilson(?:'s)?)\b"
    r"|\bASHA[ \t]+workers?\b"
)

_STOP_NAME_WORDS = {
    "patient", "pt", "mariz", "rogi", "attendant", "doctor", "dr", "ward", "bed",
    "hospital", "clinic", "department", "medicine", "surgery", "pulmonology", "neurology",
    "diagnosis", "complaints", "history", "course", "advice", "review", "medications",
    "normal", "stable", "condition", "fever", "bukhar", "pain", "cough", "cold",
    "vomiting", "weakness", "diarrhoea", "diabetes", "hypertension", "typhoid",
    "malaria", "dengue", "anaemia", "anemia", "pneumonia", "tuberculosis", "asthma",
    "tab", "cap", "inj", "inh", "syr", "sachet", "dolo", "paracetamol", "low", "high",
    "next", "morning", "evening", "day", "days", "week", "weeks", "month", "months",
    "opd", "ipd", "uhid", "mrn", "dob", "age", "sex", "mob", "mobile", "phone",
    "visited", "discharged", "informed", "treated", "conservatively", "referred",
    "sector", "phase", "colony", "quarter", "qtr", "flat", "plot", "consent", "signed",
    "referral", "note", "stage", "grade", "type", "group", "code", "level", "vitamin", "covid", "sars",
    "father", "mother", "wife", "husband", "brother", "sister", "son", "daughter", "family", "relative", "nurse", "staff", "ambulance", "police",
    "मरीज़", "रोगी", "डॉक्टर", "वार्ड", "निदान", "बुखार", "दर्द", "दवाई", "सारांश",
    "अस्पताल", "तारीख", "आयु", "उम्र", "ग्राम", "पता", "मोबाइल", "छुट्टी", "भर्ती", "है",
    "नोट", "रेफरल", "संदर्भ", "लाभार्थी",
}

_CLINICAL_COMPLAINT_WORDS = {
    "fever", "bukhar", "बुखार", "pain", "dard", "दर्द", "cough", "khansi", "खांसी",
    "cold", "जुकाम", "vomiting", "ulti", "उल्टी", "diarrhoea", "diarrhea", "dast",
    "loose", "weakness", "कमजोरी", "breathlessness", "dyspnoea", "headache", "सिरदर्द",
    "swelling", "rash", "bleeding", "chills", "giddiness", "vertigo", "chest", "abdomen",
    "pet", "पेट", "throat", "burning", "micturition", "stool", "constipation",
    "since", "for", "from", "with", "days", "day", "wks", "weeks", "months", "yrs", "din", "दिन",
}

_HONORIFIC_SUFFIXES = {
    "ji", "sahab", "saheb", "saab", "didi", "behenji", "bahenji", "bhai", "bhaiya",
    "bhaisahab", "chacha", "chachu", "mama", "mami", "mausi", "bhabhi", "amma", "baba",
    "dada", "dadi", "nana", "nani", "जी", "साहब", "दीदी", "बहनजी", "भाई", "भैया",
}

# --------------------------------------------------------------------------- name & address triggers
_PATIENT_LABEL_RE = re.compile(
    rf"(?<!डॉ\.)(?<!Dr\.)(?i:"
    rf"(?:Patient(?:'s)?|Pt\.?|Beneficiary|Client|Subject|Participant|Referral|Mareez|Mareej|Mariz|Rogi|लाभार्थी|रेफरल|संदर्भ|मरीज़|रोगी)"
    rf"(?:[ \t]+(?:ka[ \t]+naam|naam|का[ \t]+नाम|नाम|नोट|विवरण|Name|Note|Details))?"
    rf"|Name|नाम"
    rf")[ \t]*[:\-][ \t]*"
    rf"({_SLOT_SEQ}|[{_NL}][{_AL}]+,[ \t]*[{_NL}][{_AL}]+)"
    rf"(?=[ \t]{{2,}}|[ \t]+(?i:Age|Sex|Gender|DOB|UHID|MRN|Reg|IPD|S/o|D/o|W/o|आयु|उम्र|लिंग|पंजीकरण|पंजीयन|पिता|पति|मोबाइल)\b|[ \t]+(?![{_NL}])|[ \t]*[,;\.\(\)\|\n\r]|$)"
)
_PATIENT_UNDELIM_RE = re.compile(
    rf"(?<!डॉ\.)(?<!Dr\.)\b(?:Patient|Pt\.?)[ \t]+({_SLOT_SEQ})"
    rf"(?=[ \t]*[\(\n\r,;]|[ \t]+(?:ko|ne|ki|ka|se|who|visited|came|is|was|has|had|presented|signed|discharged)\b|[ \t]+(?![{_NL}])|$)"
)

# Relative triggers (delimited "Attendant: <Name>", "पिता: <Name>", and undelimited "Attendant <Name> (...)")
_REL_LABEL_RE = re.compile(
    rf"(?i:(?:[SDW][ \t]*/[ \t]*[Oo])"
    rf"|(?:पिता(?:/पति)?(?:[ \t]+(?:का|की)[ \t]+नाम)?|पति(?:[ \t]+(?:का|की)[ \t]+नाम)?|माता(?:[ \t]+(?:का|की)[ \t]+नाम)?|पुत्र|पुत्री|पत्नी|भाई|बहन|अभिभावक|संरक्षक)[ \t]*[:\-]"
    rf"|(?:पिता(?:/पति)?|पति|माता)[ \t]+(?:का|की)[ \t]+नाम[ \t]*[:\-]?"
    rf"|(?:अभिभावक|संरक्षक)[ \t]*[:\-]?"
    rf"|(?:Attendant|Caretaker|Guardian|Next[ \t]+of[ \t]+Kin|Father|Mother|Husband|Wife|Spouse)(?:[ \t]*\([^)]*\))?[ \t]*[:\-]"
    rf"|\bRef(?:erred[ \t]+by)?[ \t]*:[ \t]*(?!(?i:range|dr\.?|डॉ)\b))"
    rf"[ \t]*({_SLOT_SEQ})"
    rf"(?=[ \t]*[,.;|\(\)\n\r]|[ \t]+(?i:discharged|informed|signed|ne|ko|ke|ki|se|Mob|Mobile|Phone|फ़ोन|फोन|मोबाइल|को|से|ने)\b|[ \t]+(?![{_NL}])|$)"
)
_REL_UNDELIM_RE = re.compile(
    rf"\b(?i:Attendant|Caretaker|Guardian|Accompanied[ \t]+by)[ \t]+(?!(?i:physician|doctor|nurse|staff|dr\.?|डॉ))({_SLOT_SEQ})"
    rf"(?=[ \t]*[,.;\(\)\n\r]|[ \t]+(?i:ne|ko|ke|ki|se|signed|discharged|informed|Mob|Mobile|Phone|को|से|ने)\b|[ \t]+(?![{_NL}])|$)"
)

# Attendant inline S/o D/o W/o inside parentheses or sentences
_REL_INLINE_RE = re.compile(
    rf"\b([SDW]/[Oo])[ \t]+({_NAME_SEQ})"
)

# Prefix honorifics (Mr/Mrs/Ms/Smt/Shri/श्री/श्रीमती) and "Pt. <Capitalized Name>"
_PREFIX_HONORIFIC_RE = re.compile(
    rf"(?<!\w)(?:Mr\.?|Mrs\.?|Ms\.?|Miss|Smt\.?|Shri|Shree|Sri|श्री|श्रीमती|सुश्री|(?-i:Pt\.?))[ \t]+({_SLOT_SEQ})"
)

# Hinglish honorific/relationship suffixes ("Rohan ji", "Kiran sahab", "Jayesh babu", "Priya Bhaiya", "Sohan Mama")
_SUFFIX_HONORIFIC_RE = re.compile(
    rf"(?<![{_LL}{_DL}])({_SHORT_SEQ})[ \t]+"
    rf"(?i:ji|sahab|saheb|saab|didi|behenji|bahenji|bhai|bhaiya|bhaisahab|chacha|chachu|mama|mami|mausi|bhabhi|amma|baba|babu|ben|garu|anna|akka|tai|sahiba|जी|साहब|दीदी|बहनजी|भाई|भैया|बाबू|बेन|अन्ना|ताई|साहिबा)"
    rf"(?=[ \t]|[.,:;!?\)\n\r]|$)"
)

# WhatsApp sender: "[11/09/26, 8:13 pm] Ajay Bhaiya: <attachment..."
_WHATSAPP_SENDER_RE = re.compile(
    r"\[\d{1,2}/\d{1,2}/\d{2,4},[ \t]*\d{1,2}:\d{2}(?:[ \t]*(?:am|pm))?\][ \t]*([^:\n\r]{2,45}?)[ \t]*:",
    re.I,
)

# Report of <Name> <Age>
_REPORT_OF_RE = re.compile(
    rf"\bReport[ \t]+of[ \t]+({_NAME_SEQ})(?=[ \t]+\d{{1,3}}[ \t]*[Y/]|[,.\n]|$)",
    re.I,
)

# Discharge note / sentence-initial "<Name>, W/o ..." or "OPD note for <Name>."
_LEADING_REL_SUBJECT_RE = re.compile(
    rf"(?:^|\n)[ \t]*({_SHORT_SEQ})[ \t]*,[ \t]*(?=[SDWC]/[Oo]\b)"
)
_FOR_PATIENT_RE = re.compile(
    rf"\b(?:note[ \t]+for|prepared[ \t]+for|called[ \t]+by|witnessed[ \t]+by|from)[ \t]+({_SHORT_SEQ})(?=['.,(\s\n]|$)",
    re.I,
)
_CONSENT_PERSON_RE = re.compile(
    rf"\b(?i:consent[ \t]+(?:of|by|from|signed[ \t]+by)|signed[ \t]+by|on[ \t]+behalf[ \t]+of)[ \t]+({_SLOT_SEQ})(?=['.,;\(\)\n]|[ \t]+(?![{_NL}])|$)"
)
_POSSESSIVE_NAME_RE = re.compile(
    rf"(?:^|[ \t])({_SHORT_SEQ})'s[ \t]+(?:attendant|family|father|mother|wife|husband|son|daughter|report|samples?)\b"
)

# Additional multilingual narrative / clinical-note templates (kept strictly scoped so _SLOT_SEQ stays case-sensitive)
_SELF_CONSENT_RE = re.compile(
    rf"(?<!\w)I[ \t]*,[ \t]*({_SLOT_SEQ})[ \t]*,(?=[ \t]+(?i:consent|agree|hereby|authorize|authorise|acknowledge|declare|understand|confirm|son|daughter|wife|aged?|r/o|resident|bearing)\b)"
)
_ACTION_SUBJECT_RE = re.compile(
    rf"(?:^|[\n.!?][ \t]*)({_SLOT_SEQ})(?=[ \t]+(?i:wants|needs|requested|requests|is[ \t]+asking|asked|called|reported|complained|returned|refused|agreed|consented)[ \t]+(?:a[ \t]+|the[ \t]+|for[ \t]+|to[ \t]+|about[ \t]+)?(?i:refund|copy|reports?|bills?|discharge|admission|summary|records?|be\b|get\b))"
)
_BROUGHT_BY_RE = re.compile(
    rf"\b(?i:brought[ \t]+by|accompanied[ \t]+by|attended[ \t]+by)[ \t]+(?:(?i:her|his|their|patient'?s|pt'?s)[ \t]+)?(?:(?i:father|mother|husband|wife|spouse|son|daughter|brother|sister|uncle|aunt|cousin|nephew|niece|relative|attendant|guardian|friend|neighbour|neighbor|bhai|bhaiya|didi|pita|mata|pati|patni|beta|beti)[ \t]+)?({_SLOT_SEQ})(?=[ \t]*[,.;\(\)\n\r]|[ \t]+(?i:on|at|ne|ko|ke|ki|se|who|for|with|and|to)\b|[ \t]+(?![{_NL}])|$)"
)
_POSS_REL_NAME_RE = re.compile(
    rf"(?i:\b(?:unke|unki|unka|uske|uski|uska|inhe|inke|mere|meri|his|her|their)[ \t]+(?:father|mother|husband|wife|son|daughter|brother|sister|pita|mata|pati|patni|beta|beti|bhai|behen|bahen|mama|chacha|guardian|attendant|पिता|माता|पति|पत्नी|भाई|बहन|बेटा|बेटी))[ \t]+({_SLOT_SEQ})(?=[ \t]*[,.;\(\)\n\r]|[ \t]+(?i:ne|ko|ka|ki|ke|se|bataya|kaha|bola|boli|signed|called|ने|को)\b|[ \t]+(?![{_NL}])|$)"
)
_BED_PATIENT_RE = re.compile(
    rf"(?i:\b(?:bed(?:[ \t]*no\.?)?(?:[ \t]*\d{{1,4}})?|ward[ \t]*\d{{1,3}})[ \t]+(?:pe|par|per|on|[:\-]))[ \t]+({_SLOT_SEQ})(?=[ \t]*[\(\n\r,;]|[ \t]+(?:\d|(?i:ko|ne|ka|ki|hai|is))\b|$)"
)
_CALL_NAME_RE = re.compile(
    rf"\b(?i:call|contact|ring|speak[ \t]+to|inform)[ \t]+({_SLOT_SEQ})(?=[ \t]+(?i:on|at|par|pe)\b[ \t]+[\+\d(])"
)
_INLINE_AGESEX_NAME_RE = re.compile(
    rf"(?:^|\n|[|•\-–][ \t]*)[ \t]*({_SLOT_SEQ})(?:[ \t]{{2,}}|[ \t]*\([ \t]*)\d{{1,3}}[ \t]*(?i:Y|Yrs?|yr)?[ \t]*/[ \t]*(?i:[MF]|Male|Female|पुरुष|महिला)\b"
)
_HTML_CELL_NAME_RE = re.compile(
    rf"(?i:<(?:td|th)\b[^>]*>)[ \t]*({_TOK}[ \t]+{_TOK}(?:[ \t]+{_TOK}){{0,2}})[ \t]*(?i:</(?:td|th)>)"
)

# Hinglish / Hindi postpositions indicating a person subject
_HINGLISH_POSTPOS_RE = re.compile(
    rf"(?:^|[\n.\-–।][ \t]*)({_SHORT_SEQ})[ \t]+(?:ka[ \t]+(?:BP|pulse|temp|sugar|condition|report)|ki[ \t]+(?:condition|tabiyat|report)|को[ \t]+\d+[ \t]*दिन[ \t]+से|को[ \t]+बुखार|का[ \t]+बीपी)"
)
_TIMED_NOTE_NAME_RE = re.compile(
    rf"(?:^|\n)[ \t]*\d{{1,2}}:\d{{2}}[ \t]*[-–][ \t]*({_NAME_SEQ})(?=[ \t]*(?:[-–][ \t]*K/C/O|\(|,|\n|$))"
)

# Initials pattern like "K. R. Menon", "A. Nair", "Md. Tariq", "J.-P. Élodie"
_INITIALS_NAME_RE = re.compile(
    rf"(?<!\w)(?:[{_LU}]\.[ \t]*){{1,2}}[{_LU}][a-z\u00DF-\u024F\u1E00-\u1EFF\-]{{2,20}}(?!\w)"
)

# C/O ambiguity: scan candidate "C/O <phrase>" and resolve multi-token context
_CARE_OF_CAND_RE = re.compile(r"(?i)(?<![KH]/)C[ \t]*/[ \t]*O[ \t]*:?[ \t]*([^\n\r;]+)")

# Free-text address & village lines (LANDMARK matched before LOCALITY so 'near Neem Peepal Crossing' stays LANDMARK)
_ADDRESS_LINE_RE = re.compile(
    r"(?i)(?:Address(?:\s*\([^)]*\))?|पता|निवासी(?:\s*ग्राम)?)[ \t]*[:\-][ \t]*([^\n\r]+)"
    r"|\b(?:resident[ \t]+of|r/o)[ \t]+(?=(?:H\.?\s*No|House|Makan|Qtr|Quarter|Flat|Plot|Ward|Sector|Mohalla|near|opp|मकान|वार्ड|मोहल्ला)\b)([^\n\r]+)"
)
_VILLAGE_LINE_RE = re.compile(
    r"(?i)(?:(?:\bVill(?:age)?\.?|\bGram|निवासी[ \t]*ग्राम|ग्राम)[ \t]*[:\-]|(?<![0-9०-९][\s.])(?:\b(?:Gaon|Gram)|(?<![\w0-9०-९])ग्राम)[ \t]+)[ \t]*([^\n\r,;]+)"
    r"|\b(?i:resident[ \t]+of|native[ \t]+of|निवासी)[ \t]+(?!(?i:H\.?\s*No|House|Makan|Qtr|Quarter|Flat|Plot|Ward|Sector|Mohalla|near|opp|the|a|an|this|india)\b)([^\n\r,;.()]{2,40})"
)

# Address sub-components (English, residential quarters/colonies/bazaars/mandis, and Devanagari)
_FREE_ADDR_COMPONENTS: list[tuple[str, re.Pattern]] = [
    ("HOUSE", re.compile(r"(?i)\b(?:H\.?\s*No\.?|House(?:\s+No\.?)?|Makan\s+No\.?|Qtr\.?|Quarter|Flat(?:\s+No\.?)?|Plot(?:\s+No\.?)?|Door(?:\s+No\.?)?)[\s.\-#:/]*[\dA-Za-z/-]*\d[\dA-Za-z/-]*")),
    ("HOUSE", re.compile(r"मकान\s*(?:नं\.?|सं\.?)\s*[\dA-Za-z/-]+")),
    ("LANDMARK", re.compile(r"(?i)\b(?:near|opp\.?|opposite|behind|beside)\s+[^,\n;]+")),
    ("LANDMARK", re.compile(r"[^,\n;]+?\s+(?:के\s+पास|के\s+पीछे|के\s+सामने)")),
    ("LOCALITY", re.compile(r"(?i)\bWard\s*(?:No\.?)?\s*\d+")),
    ("LOCALITY", re.compile(r"वार्ड\s*(?:नं\.?)?\s*\d+")),
    ("LOCALITY", re.compile(r"(?i)\bSector\s*[-#]?\s*[A-Za-z0-9]+(?:\s+(?:Colony|Phase|Extension|Enclave|Block))?")),
    ("LOCALITY", re.compile(r"(?i)\bMohalla\s+[^,\n;]+")),
    ("LOCALITY", re.compile(rf"\b[{_LU}][{_LL}0-9]*(?:\s+[{_LU}0-9][{_LL}0-9]*){{0,2}}\s+(?:Colony|Nagar|Vihar|Enclave|Puram|Extension|Basti|Tola|Layout|Chowk|Bazaar|Bazar|Mandi|Ganj|Bagh|Sarai|Chauraha|Pura|Marg|Lane|Market|Phase|Block|Peth|Para|Palli|Galli|Chawl|Camp|Naka)(?:\s+\d{{1,4}})?\b")),
    ("LOCALITY", re.compile(r"मोहल्ला\s+[^,\n;]+")),
    ("LOCALITY", re.compile(rf"[{_DL}]+(?:\s+[{_DL}]+){{0,2}}\s+(?:मोहल्ला|टोला|बस्ती|कॉलोनी|नगर|विहार|बाज़ार|बाजार|मंडी|चौक|गली|गंज|बाग़|बाग|मार्ग|पेठ|पारा|चावल|कैंप|लेन|नाका)(?:\s+[\d०-९]{{1,4}})?(?=\s*[,;\.\-\n\r]|$)")),
    ("VILLAGE", re.compile(r"(?i)\b(?:Vill\.?|Village|Gram)\s+[^,\n;\-]+")),
    ("VILLAGE", re.compile(r"(?<![0-9०-९][\s.])(?<![\w0-9०-९])ग्राम\s+[^,\n;\-]+")),
    # Bare village name right before ", Tehsil ..." or ", तहसील ..."
    ("VILLAGE", re.compile(rf"(?<=,\s)(?!(?:H\.?\s*No|House|Makan|Qtr|Quarter|Flat|Plot|Ward|Sector|Mohalla|near|opp|Tehsil|Distt|C/O)\b)([{_NL}][^\n,;]+?)(?=,\s*(?:Tehsil|तहसील)\b)", re.I)),
    ("PIN", re.compile(r"(?<!\d)[1-8]\d{5}(?!\d)")),
]

_SYNTHETIC_CANARY_TOK_RE = re.compile(r"^[A-Z][a-z]{4,20}\d{1,2}[a-z]{0,4}$")


def _is_valid_name_candidate(cand: str, allow_synthetic_canary: bool = False) -> bool:
    s = cand.strip(" .,:;-'\"")
    if not s or len(s) < 2 or len(s) > 60:
        return False
    toks = [t.strip(".,'\"") for t in s.split() if t.strip(".,'\"")]
    if not toks or len(toks) > 4:
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
            if not _SYNTHETIC_CANARY_TOK_RE.match(folded):
                return False
            base = re.sub(r"\d+", "", folded).casefold()
            if base in _STOP_NAME_WORDS or base in _CLINICAL_COMPLAINT_WORDS:
                return False
    non_stop = [
        t for t in toks
        if re.sub(r"\d+", "", fold_latin_accents(t)).casefold() not in _STOP_NAME_WORDS
        and t.casefold() not in _HONORIFIC_SUFFIXES
    ]
    if not non_stop:
        return False
    if any(t.casefold() in _CLINICAL_COMPLAINT_WORDS for t in toks):
        return False
    return any(is_upper_or_deva_start(t) for t in toks)


def _resolve_care_of(phrase: str, before_ctx: str) -> str | None:
    """Disambiguate 'C/O <phrase>' using multi-token context. Return RELATIVE_NAME slice or None (complaint)."""
    first_seg = re.split(r"[,;\n]", phrase)[0].strip()
    if not first_seg:
        return None
    low_words = {w.strip(".,:").casefold() for w in first_seg.split()}
    if low_words & _CLINICAL_COMPLAINT_WORDS:
        return None
    if re.search(r"(?i)\b(?:for|since|x)\s+\d+", phrase):
        return None
    addr_hint = bool(re.search(r"(?i)(?:address|attendant|पता|guardian)", before_ctx[-40:]))
    right_addr_hint = bool(re.search(r"(?i),\s*(?:H\.?\s*No|Makan|House|Qtr|Flat|Ward|Sector|Mohalla|मकान|वार्ड|मोहल्ला)", phrase))
    toks = first_seg.split()
    if 1 <= len(toks) <= 3 and _is_valid_name_candidate(first_seg):
        if addr_hint or right_addr_hint or all(is_upper_or_deva_start(t) for t in toks):
            return first_seg
    return None


def find_protected_clinical_and_doctor_spans(text: str, view: NormalizedView | None = None) -> list[Span]:
    """Identify doctor names, disease eponyms, and ASHA worker references that MUST be preserved."""
    nv = view if view is not None else build_normalized_view(text)
    norm_text = nv.clean
    prot: list[Span] = []
    for rx in (_DOCTOR_SPAN_RE, _EPONYM_RE):
        for m in rx.finditer(norm_text):
            os_, oe_ = nv.map_clean_span(m.start(), m.end())
            if oe_ > os_:
                prot.append(Span(os_, oe_, "PROTECTED", text[os_:oe_], source="protected"))
    return resolve_overlaps(prot)


def detect_context_name_and_address_spans(
    text: str,
    view: NormalizedView | None = None,
    protected_spans: list[Span] | None = None,
) -> list[Span]:
    """Detect free-text patient names, relative names, and address PII via context triggers."""
    if not text:
        return []
    nv = view if view is not None else build_normalized_view(text)
    norm_text = nv.clean
    prot = protected_spans if protected_spans is not None else find_protected_clinical_and_doctor_spans(text, view=nv)
    spans: list[Span] = []

    def overlaps_protected(os_: int, oe_: int) -> bool:
        return any(not (oe_ <= p.start or os_ >= p.end) for p in prot)

    def add_orig_span(ns: int, ne: int, label: str, src: str = "context") -> None:
        os_, oe_ = nv.map_clean_span(ns, ne)
        if oe_ <= os_:
            return
        if overlaps_protected(os_, oe_):
            return
        spans.append(Span(os_, oe_, label, text[os_:oe_], source=src))

    # 1. Explicit Patient Name labels & undelimited Patient/Pt. <Name> (+ parenthesized bilingual/honorific alias)
    def _maybe_paren_alias(match_end: int, lbl: str) -> None:
        tail = norm_text[match_end:]
        m_p = re.match(r"[ \t]*\(([^{}\(\)\n\r]{2,55})\)", tail)
        if m_p:
            inner = m_p.group(1).strip()
            if (
                not re.search(r"(?i)^(?:[SDWC]/[Oo]|Dr\.?|MD|MS|MBBS|Age|MRN|UHID|ID|tel|mob|ph|\d)", inner)
                and _is_valid_name_candidate(inner, allow_synthetic_canary=True)
            ):
                p_start = match_end + m_p.start(1)
                add_orig_span(p_start, p_start + len(m_p.group(1)), lbl)

    for m in _PATIENT_LABEL_RE.finditer(norm_text):
        cand = m.group(1).strip()
        if _is_valid_name_candidate(cand, allow_synthetic_canary=True):
            add_orig_span(m.start(1), m.start(1) + len(cand), "PATIENT_NAME")
            _maybe_paren_alias(m.end(1), "PATIENT_NAME")
    for m in _PATIENT_UNDELIM_RE.finditer(norm_text):
        cand = m.group(1).strip()
        if _is_valid_name_candidate(cand, allow_synthetic_canary=True):
            add_orig_span(m.start(1), m.start(1) + len(cand), "PATIENT_NAME")
            _maybe_paren_alias(m.end(1), "PATIENT_NAME")

    # 2. Explicit Relative labels (delimited, undelimited Attendant/Caretaker, and S/o inline)
    for m in _REL_LABEL_RE.finditer(norm_text):
        cand = m.group(1).strip()
        if _is_valid_name_candidate(cand, allow_synthetic_canary=True):
            add_orig_span(m.start(1), m.start(1) + len(cand), "RELATIVE_NAME")
            _maybe_paren_alias(m.end(1), "RELATIVE_NAME")
    for m in _REL_UNDELIM_RE.finditer(norm_text):
        cand = m.group(1).strip()
        if _is_valid_name_candidate(cand, allow_synthetic_canary=True):
            add_orig_span(m.start(1), m.start(1) + len(cand), "RELATIVE_NAME")
    for m in _REL_INLINE_RE.finditer(norm_text):
        cand = m.group(2).strip()
        if _is_valid_name_candidate(cand):
            add_orig_span(m.start(2), m.start(2) + len(cand), "RELATIVE_NAME")

    # 3. C/O multi-token disambiguation
    for m in _CARE_OF_CAND_RE.finditer(norm_text):
        before_ctx = norm_text[max(0, m.start() - 50) : m.start()]
        person = _resolve_care_of(m.group(1), before_ctx)
        if person:
            idx = m.start(1) + m.group(1).find(person)
            add_orig_span(idx, idx + len(person), "RELATIVE_NAME")

    # 4. Prefix honorifics
    for m in _PREFIX_HONORIFIC_RE.finditer(norm_text):
        cand = m.group(1).strip()
        if _is_valid_name_candidate(cand, allow_synthetic_canary=True):
            add_orig_span(m.start(1), m.start(1) + len(cand), "PATIENT_NAME")

    # 5. Hinglish honorific suffixes
    for m in _SUFFIX_HONORIFIC_RE.finditer(norm_text):
        cand = m.group(1).strip()
        if _is_valid_name_candidate(cand):
            label = "RELATIVE_NAME" if any(w in m.group(0).casefold() for w in ("bhaiya", "chacha", "chachu", "mama", "mami", "mausi")) else "PATIENT_NAME"
            add_orig_span(m.start(), m.end(), label)

    # 6. WhatsApp sender names (skip generic hospital roles like Duty Sister / Staff Nurse)
    for m in _WHATSAPP_SENDER_RE.finditer(norm_text):
        cand = m.group(1).strip()
        if cand.casefold() in {"duty sister", "ward boy", "staff nurse", "duty doctor", "front desk", "reception", "admin"}:
            continue
        if _is_valid_name_candidate(cand):
            add_orig_span(m.start(1), m.start(1) + len(cand), "RELATIVE_NAME")

    # 7. Consent signatures, Report of <Name>, Leading "<Name>, W/o", and narrative clinical note triggers
    for rx in (_CONSENT_PERSON_RE, _SELF_CONSENT_RE):
        for m in rx.finditer(norm_text):
            cand = m.group(1).strip()
            if _is_valid_name_candidate(cand, allow_synthetic_canary=True):
                add_orig_span(m.start(1), m.start(1) + len(cand), "PATIENT_NAME")
    for rx, lbl in (
        (_REPORT_OF_RE, "PATIENT_NAME"),
        (_LEADING_REL_SUBJECT_RE, "PATIENT_NAME"),
        (_FOR_PATIENT_RE, "PATIENT_NAME"),
        (_POSSESSIVE_NAME_RE, "PATIENT_NAME"),
        (_HINGLISH_POSTPOS_RE, "PATIENT_NAME"),
        (_TIMED_NOTE_NAME_RE, "PATIENT_NAME"),
        (_ACTION_SUBJECT_RE, "PATIENT_NAME"),
        (_BED_PATIENT_RE, "PATIENT_NAME"),
        (_CALL_NAME_RE, "PATIENT_NAME"),
        (_INLINE_AGESEX_NAME_RE, "PATIENT_NAME"),
        (_HTML_CELL_NAME_RE, "PATIENT_NAME"),
        (_BROUGHT_BY_RE, "RELATIVE_NAME"),
        (_POSS_REL_NAME_RE, "RELATIVE_NAME"),
    ):
        for m in rx.finditer(norm_text):
            cand = m.group(1).strip()
            if _is_valid_name_candidate(cand, allow_synthetic_canary=True):
                out_lbl = "RELATIVE_NAME" if "(attendant" in norm_text[m.end(1) : m.end(1) + 20].casefold() else lbl
                add_orig_span(m.start(1), m.start(1) + len(cand), out_lbl)

    # 8. Initials
    for m in _INITIALS_NAME_RE.finditer(norm_text):
        cand = m.group(0).strip()
        if _is_valid_name_candidate(cand):
            add_orig_span(m.start(), m.end(), "PATIENT_NAME")

    # 9. Free-text Address & Village lines
    for m in _VILLAGE_LINE_RE.finditer(norm_text):
        g_idx = 1 if m.group(1) is not None else 2
        cand = (m.group(g_idx) or "").strip()
        if cand:
            add_orig_span(m.start(g_idx), m.start(g_idx) + len(cand), "VILLAGE")

    for m in _ADDRESS_LINE_RE.finditer(norm_text):
        g_idx = 1 if m.group(1) is not None else 2
        line_start = m.start(g_idx)
        line_txt = m.group(g_idx) or ""
        for comp_lbl, comp_rx in _FREE_ADDR_COMPONENTS:
            for cm in comp_rx.finditer(line_txt):
                s_idx = line_start + cm.start(1) if cm.lastindex else line_start + cm.start()
                e_idx = line_start + cm.end(1) if cm.lastindex else line_start + cm.end()
                add_orig_span(s_idx, e_idx, comp_lbl)

    return resolve_overlaps(spans)

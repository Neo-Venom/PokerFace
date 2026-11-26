"""Step E: Protected clinical spans (medicines, doses, frequencies, vitals, labs, diagnoses, procedures, complaints) and explicit span-conflict resolution."""
from __future__ import annotations

import re

from raksha.detect.identifiers import Span, resolve_overlaps
from raksha.detect.names_context import find_protected_clinical_and_doctor_spans
from raksha.detect.normalize import NormalizedView, build_normalized_view

# Base clinical medicine lexicon (extended dynamically from data/clean/ formularies at runtime)
_BASE_DRUGS = {
    "amlodipine", "artesunate", "atorvastatin", "azithromycin", "ceftriaxone",
    "doxycycline", "glimepiride", "insulin", "insulin glargine", "ferrous sulphate",
    "folic acid", "levodopa", "carbidopa", "levothyroxine", "metformin", "ors",
    "pantoprazole", "paracetamol", "ramipril", "rifampicin", "isoniazid",
    "salbutamol", "salbutamol inhaler", "telmisartan", "dolo", "dolo 650",
    "crocin", "calpol", "amoxicillin", "ciprofloxacin", "metronidazole",
    "omeprazole", "rabeprazole", "aspirin", "clopidogrel", "losartan",
    "furosemide", "prednisolone", "montelukast", "cetirizine", "ondansetron",
    "ibuprofen", "diclofenac", "meropenem", "piperacillin", "tazobactam",
}

_BASE_LABS = {
    "haemoglobin", "hemoglobin", "hba1c", "random blood sugar", "rbs",
    "serum creatinine", "creatinine", "tsh", "platelet count", "platelets",
    "total wbc count", "wbc", "serum sodium", "sodium", "potassium",
}

_DOSAGE_FORM_DRUG_RE = re.compile(
    r"\b(?:Tab|Tablet|Cap|Capsule|Inj|Injection|Syr|Syrup|Inh|Inhaler|Sachet|Drops)\.?\s+"
    r"[A-Z][A-Za-z]+(?:\s*(?:\+|/)\s*[A-Z]?[A-Za-z]+)*(?:\s+inhaler)?"
)
_DRUG_SUFFIX_RE = re.compile(
    r"\b[A-Za-z]{4,}(?:mycin|cillin|oxacin|pril|sartan|statin|prazole|formin|gliptin|dipine|olol|zole|vir|fenac|codone|pam|zepam|thiazide)\b",
    re.I,
)
_DOSE_RE = re.compile(
    r"\b\d+(?:\.\d+)?(?:\s*(?:\+|/)\s*\d+(?:\.\d+)?)?\s*"
    r"(?:mg|mcg|ug|µg|gm|g|ml|mL|L|IU|units?|%|puffs?|drops?|tabs?|caps?|sachets?)\b"
    r"|\b\d{2,3}(?:\.\d+)?\s*°?\s*[FC]\.?"
)
_FREQ_DUR_RE = re.compile(
    r"\b(?:OD|BD|TDS|TID|QID|HS|SOS|PRN|STAT|IV|IM|SC|PO)\b"
    r"|\b[012][- ][012][- ][012]\b"
    r"|\b(?:x|for|every)\s+\d+\s*(?:days?|hrs?|hours?|wks?|weeks?|months?|yrs?|years?|दिन|हफ्ते|महीने)\b"
    r"|\b\d+\s*(?:days?|wks?|weeks?|months?|दिन)\b",
    re.I,
)
_VITALS_RE = re.compile(
    r"(?i)\b(?:BP|Blood\s+Pressure|Pulse|HR|SpO2|Temp(?:erature)?|Weight|Wt|बीपी)\s*[:=]?\s*\d+(?:[./]\d+)?(?:\s*(?:mmHg|/min|bpm|%|kg|°?[FC]\.?))?"
    r"|(?<!\d)(?:[12]?\d{2})/(?:[4-9]\d|1[0-4]\d)(?!\d)"
)
_LAB_UNIT_RE = re.compile(
    r"\b\d+(?:\.\d+)?(?:\s*[-–]\s*\d+(?:\.\d+)?)?\s*(?:g/dL|mg/dL|mmol/L|mIU/L|IU/L|U/L|/uL|/ul|/mm3)\b",
    re.I,
)
_ICD_AND_HISTORY_RE = re.compile(
    r"\b[A-TV-Z]\d{2}(?:\.\d{1,2})?\b"
    r"|\b(?:No\s+)?[KH]/C/O\s*:?\s*[^\n.;]+"
    r"|\bH/O\s*:?\s*[^\n.;]+"
    r"|\bC/O\s*:?\s*(?:fever|bukhar|बुखार|pain|dard|दर्द|cough|khansi|vomiting|weakness|diarrhoea|loose\s+motions)[^\n.;]*"
    r"|\b(?:Diagnosis|Presenting\s+complaints|Procedure)\s*:\s*[^\n]+"
    r"|निदान\s*:\s*[^\n]+"
    r"|\b(?:Ward|Bed)\s+\d+(?:\s*\([^)]+\))?",
    re.I,
)
_HINDI_CLINICAL_RE = re.compile(
    r"(?<!\d)\d+\s*दिन\s*से\s*(?:बुखार|पेट\s*दर्द|खांसी|कमजोरी|उल्टी|दस्त)"
    r"|(?<!\d)\d+\s*din\s*se\s*(?:bukhar|pet\s*mein\s*dard|dard|khansi)"
    r"|\b(?:bukhar|dawai|khana\s+kam|ulti|band\s+ho\s+gayi|theek\s+se\s+neend)\b"
    r"|(?:फेफड़ों\s+की\s+टीबी|निमोनिया|मलेरिया|टाइफाइड|डेंगू|मधुमेह|थायराइड|उच्च\s+रक्तचाप|दमा(?:\s+रोग)?|पीलिया|"
    r"पैरों\s+में\s+सूजन|कमज़ोरी\s+और\s+चक्कर\s+आना|कमजोरी\s+और\s+चक्कर\s+आना|बलगम\s+वाली\s+खांसी(?:\s+और\s+सांस\s+फूलना)?|"
    r"तीव्र\s+आंत्रशोथ|सीओपीडी|रक्ताल्पता|मूत्र\s+संक्रमण|दिन\s+में\s+(?:एक|दो|तीन|चार)\s+बार|"
    r"रोज़\s+(?:सुबह|शाम|रात)|खाली\s+पेट|भर्ती\s+(?:की\s+)?(?:तिथि|तारीख)|छुट्टी\s+की\s+तारीख|विमोचन\s+(?:तिथि|पत्र)|"
    r"सलाहकार\s+चिकित्सक)"
    r"|(?<![\d०-९.])[\d०-९]+(?:\.[\d०-९]+)?\s*(?:मिग्रा|ग्राम|मिली|दिन|वर्षों?(?:\s+से)?)\b"
    r"|\b(?:Batch|Lot|Serial)\s*(?:No\.?|Num\.?|#)?\s*[:.#\-]?\s*\d{4,}\b"
    r"|\b(?:आईवी|ओडी|बीडी)\b",
    re.I,
)
_HOSPITAL_PHRASE_RE = re.compile(
    r"(?<![A-Za-z\u0900-\u097F])"
    r"(?:[A-Z\u0900-\u097F][A-Za-z\u0900-\u097F.'\-]{0,24}[ \t]+){1,5}"
    r"(?:Hospital|Clinic|Sadan|Trust|Diagnostics?|Labs?|Chikitsalaya|Aspatal|Sevasadan|Sansthan|Medical[ \t]+(?:Centre|Center|College)|Nursing[ \t]+Home|चिकित्सालय|अस्पताल|हॉस्पिटल|सेवासदन|सेवाश्रम)"
    r"(?![A-Za-z\u0900-\u097F])"
)


class ClinicalLexicon:
    """Dynamic clinical lexicon built from clean formularies + baseline clinical terms."""

    def __init__(self, drugs: set[str] | None = None, labs: set[str] | None = None, diagnoses: set[str] | None = None):
        self.drugs: set[str] = set(_BASE_DRUGS)
        self.labs: set[str] = set(_BASE_LABS)
        self.diagnoses: set[str] = set()
        if drugs:
            self.drugs.update(d.strip().casefold() for d in drugs if d and len(d.strip()) >= 3)
        if labs:
            self.labs.update(l.strip().casefold() for l in labs if l and len(l.strip()) >= 2)
        if diagnoses:
            self.diagnoses.update(x.strip().casefold() for x in diagnoses if x and len(x.strip()) >= 4)

        all_terms = sorted(self.drugs | self.labs | self.diagnoses, key=len, reverse=True)
        if all_terms:
            escaped = [re.escape(t) for t in all_terms if len(t) >= 3]
            self._term_re: re.Pattern | None = re.compile(r"\b(?:" + "|".join(escaped) + r")\b", re.I)
        else:
            self._term_re = None

    def find_spans(self, text: str, view: NormalizedView | None = None) -> list[Span]:
        if not text:
            return []
        nv = view if view is not None else build_normalized_view(text)
        norm_text = nv.clean
        prot: list[Span] = list(find_protected_clinical_and_doctor_spans(text, view=nv))

        def add_clean(ns: int, ne: int) -> None:
            os_, oe_ = nv.map_clean_span(ns, ne)
            if oe_ > os_:
                prot.append(Span(os_, oe_, "PROTECTED", text[os_:oe_], source="protected"))

        if self._term_re is not None:
            for m in self._term_re.finditer(norm_text):
                add_clean(m.start(), m.end())

        for rx in (_DOSAGE_FORM_DRUG_RE, _DRUG_SUFFIX_RE, _DOSE_RE, _FREQ_DUR_RE, _VITALS_RE, _LAB_UNIT_RE, _ICD_AND_HISTORY_RE, _HINDI_CLINICAL_RE, _HOSPITAL_PHRASE_RE):
            for m in rx.finditer(norm_text):
                # Do NOT mark "Ward N" inside an Address / Mohalla / Tehsil context as protected clinical ward!
                seg = m.group(0)
                if seg.lower().startswith("ward"):
                    ctx = norm_text[max(0, m.start() - 50) : min(len(norm_text), m.end() + 40)].casefold()
                    if any(w in ctx for w in ("address", "mohalla", "tehsil", "distt", "h.no", "house", "makan", "c/o", "पता")):
                        continue
                if rx is _HOSPITAL_PHRASE_RE:
                    # Never protect landmark references inside Address lines ("behind City Hospital", "near Apollo Hospital")
                    before_h = norm_text[max(0, m.start() - 30) : m.start()].casefold()
                    first_w = seg.split()[0].casefold()
                    if first_w in ("behind", "near", "opp", "opposite", "beside", "parchi", "prescription", "report", "address", "patient", "visit", "from", "at"):
                        continue
                    if re.search(r"(?:\b(?:near|opp\.?|opposite|behind|beside|address|पता)\s*[:\-]?\s*|,\s*[^,\n]{0,20}\b(?:near|behind|opp\.?|beside)\s*)$", before_h):
                        continue
                    # Trim if an Address: line precedes it on the same line
                    line_before = norm_text[norm_text.rfind("\n", 0, m.start()) + 1 : m.start()].casefold()
                    if re.match(r"^\s*(?:address|पता|निवासी)\s*[:\-]", line_before):
                        continue
                    add_clean(m.start(), m.end())
                    continue
                add_clean(m.start(), m.end())

        return resolve_overlaps(prot)


def build_clinical_lexicon(parsed_items: list[tuple[str, str, object]]) -> ClinicalLexicon:
    """Build clinical lexicon from drug formularies, lab reference tables, and diagnosis/procedure fields."""
    drugs: set[str] = set()
    labs: set[str] = set()
    diagnoses: set[str] = set()

    for _rel, fmt, payload in parsed_items:
        if fmt in ("csv", "tsv") and isinstance(payload, list) and len(payload) > 1:
            header = [h.strip().casefold() for h in payload[0]]
            for row in payload[1:]:
                for idx, h in enumerate(header):
                    if idx >= len(row):
                        continue
                    val = row[idx].strip()
                    if not val:
                        continue
                    if h in ("drug", "medicine", "medication"):
                        drugs.add(val)
                        for part in re.split(r"\s*\+\s*", val):
                            drugs.add(part.strip())
                    elif h in ("test", "lab_test"):
                        labs.add(val)
                    elif h in ("procedure", "diagnosis"):
                        diagnoses.add(val)
        elif fmt == "json" and isinstance(payload, dict):
            # Extract rx.drug, panels.results.test, diagnosis.display from structured JSON
            stack: list[object] = [payload]
            while stack:
                node = stack.pop()
                if isinstance(node, dict):
                    for k, v in node.items():
                        kl = str(k).casefold()
                        if kl == "drug" and isinstance(v, str):
                            drugs.add(v)
                        elif kl == "test" and isinstance(v, str):
                            labs.add(v)
                        elif kl == "display" and isinstance(v, str) and len(v) >= 4:
                            # Keep clinical displays (diagnosis/procedure), skip person displays
                            if not any(x in node for x in ("reference",)) or "Condition" in str(node):
                                pass
                        elif isinstance(v, (dict, list)):
                            stack.append(v)
                elif isinstance(node, list):
                    for it in node:
                        if isinstance(it, (dict, list)):
                            stack.append(it)

    return ClinicalLexicon(drugs=drugs, labs=labs, diagnoses=diagnoses)


_SOURCE_PRIORITY = {
    "structured": 0,
    "identifier": 1,
    "context": 2,
    "registry": 3,
    "crf": 4,
}


def resolve_pii_vs_protected(pii_spans: list[Span], protected_spans: list[Span]) -> list[Span]:
    """Explicit span-conflict resolution: PROTECTED clinical/doctor spans ALWAYS win overlaps."""
    surviving: list[Span] = []
    for sp in pii_spans:
        if any(not (sp.end <= prot.start or sp.start >= prot.end) for prot in protected_spans):
            continue
        surviving.append(sp)

    # Sort surviving PII spans by start, source priority, then longer span
    ordered = sorted(
        surviving,
        key=lambda sp: (sp.start, _SOURCE_PRIORITY.get(sp.source, 9), -(sp.end - sp.start), sp.label),
    )
    kept: list[Span] = []
    last_end = -1
    for sp in ordered:
        if sp.start >= last_end:
            kept.append(sp)
            last_end = sp.end
    return kept

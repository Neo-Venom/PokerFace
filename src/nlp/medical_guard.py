"""Layer 0 Protected Clinical Shield & Runtime Dynamic Medical Lexicon."""
from __future__ import annotations

from bisect import bisect_left, bisect_right
import csv
import io
import json
import re
from typing import Any

from data.clinical_vocab import (
    CLINICAL_ABBREVIATIONS,
    DRUG_NAMES,
    EPONYM_HEAD_NOUNS,
    EPONYM_STRONG_HEADS,
    NAMELIKE_EPONYMS,
    STRONG_EPONYMS,
    has_clinical_suffix,
)
from data.gazetteers import CLINICAL_SYMPTOMS_AND_TERMS, COMMON_INDIAN_NAMES, HINGLISH_STOPWORDS, MEDICAL_EPONYMS
from src.models import FileEntry

# Eponyms that are also given names / surnames / English words ("Wilson Varghese", "lie down") only count as
# clinical inside an eponym frame; every other eponym root is clinical on its own.
_ALL_EPONYM_ROOTS = {e.rstrip("'").removesuffix("'s") for e in MEDICAL_EPONYMS} | set(STRONG_EPONYMS)
_BARE_EPONYMS = frozenset(e for e in (set(MEDICAL_EPONYMS) | set(STRONG_EPONYMS))
                          if e.rstrip("'").removesuffix("'s") not in NAMELIKE_EPONYMS)
_EPONYM_RX = re.compile(
    r"(?i)\b(?:" + "|".join(sorted((re.escape(e) for e in STRONG_EPONYMS), key=lambda x: (-len(x), x))) + r")"
    r"(?:['\u2019]s|s['\u2019])?(?:[ \t]+(?:" + EPONYM_HEAD_NOUNS + r"))?\b"
    r"|(?i:\b(?:" + "|".join(sorted(NAMELIKE_EPONYMS)) + r"))(?:['\u2019]s|s['\u2019])?[ \t]+(?i:" + EPONYM_HEAD_NOUNS + r")\b"
    # hyphenated multi-person eponyms: "Guillain-Barre syndrome", "Cheyne-Stokes respiration"
    r"|\b[A-Z][a-z]+(?:-[A-Z][a-z]+)+(?:['\u2019]s)?[ \t]+(?i:" + EPONYM_HEAD_NOUNS + r")\b"
)
# "Fournier's gangrene", "Ludwig's angina": an unknown capitalised possessive in front of a disease-type head
_POSSESSIVE_EPONYM_RX = re.compile(r"\b([A-Z][a-z]{2,20})(?:['\u2019]s|s['\u2019])[ \t]+(?i:" + EPONYM_STRONG_HEADS + r")\b")
# Government health schemes are not people ("Ayushman Bharat", "Janani Suraksha Yojana")
_SCHEME_RX = re.compile(
    r"\bAyushman[ \t]+Bharat\b|\b(?:[A-Z][a-z]+[ \t]+){1,4}(?:Yojana|Yojna|Abhiyan|Abhiyaan)\b|\bPM-?JAY\b"
)
_NAMELIKE_LOWER = frozenset(NAMELIKE_EPONYMS)

_SAFE_SPAN_PATTERNS = [
    # Doctor spans (stopping before bracket/comma/EOL or clinical verbs like 'reviewed', 'took')
    re.compile(
        r"(?i)(?:\b(?:dr\.?|doctor|consultant|attending(?:\s+physician)?|physician)|डॉ\.?)\s+"
        r"(?:dr\.?\s+)?"
        r"[A-Z\u0900-\u097f][a-zA-Z\u0900-\u097f.]+"
        r"(?:\s+(?!reviewed\b|took\b|is\b|was\b|ordered\b|prescribed\b|advised\b)[A-Z\u0900-\u097f][a-zA-Z\u0900-\u097f.]+){0,2}"
        r"(?:\s*\([^)\n]{0,80}\b(?:MBBS|MD|MS|DM|DNB|MCh|FRCS|Reg\.?\s*No\.?)\b[^)\n]{0,40}\))?"
    ),
    # Medical credentials, ASHA role, and vaccine Lot / Batch numbers
    re.compile(r"\b(?:MBBS|MD|MS|DM|DNB|MCh|FRCS)\b(?:\s+[A-Z][a-zA-Z]+)?"),
    re.compile(r"(?i)\b(?:Lot|Batch)\s*(?:No\.?|Number|#)?\s*[:#\-]?\s*[A-Z0-9-]{4,16}\b"),
    re.compile(r"(?i)\bASHA\s+(?:worker|workers|karyakarta|sahayika|supervisor)\b"),
    re.compile(r"\b(?:by|from|via|with)\s+ASHA\b"),
    # Hospital Ward / Bed phrases (distinct from residential Ward 20 in addresses)
    re.compile(r"(?i)\bWard\s+\d+(?:\s*/\s*Bed\s*\d+)?\s*\([A-Za-z &]+\)"),
    re.compile(r"(?i)\bWard(?:/Bed)?\s*[:=]\s*Ward\s+\d+[^,\n]*"),
    # Clinical K/C/O, H/O (stop at sentence/clause boundary so Attendant/relative isn't swallowed!)
    re.compile(r"(?i)\b(?:K/C/O|H/O)\s*[:\-]?\s*[^\n.;]+(?=(?:\s*(?:Attendant|S/o|W/o|D/o))|[.;\n]|$)"),
    # Doses, frequencies, vitals, labs, ICD10, LOINC, ranges
    re.compile(r"\b[A-TV-Z]\d{2}(?:\.\d{1,2})?\b"),
    re.compile(r"(?<![\d-])\d{1,5}-\d(?![\d-])"),
    re.compile(r"(?i)\b\d+(?:\.\d+)?\s?(?:mg/dl|g/dl|miu/l|mmol/l|/ul|/cumm|mmhg|/min|mg|mcg|gm|g|ml|iu|units?|tabs?|caps?|%)\b"),
    re.compile(r"(?i)\b(?:BP|Pulse|PR|HR|RR|Temp|SpO2|HbA1c|Hb|RBS|FBS|PPBS|TSH|WBC)\s*[:=]?\s*\d+(?:\.\d+)?(?:/\d+(?:\.\d+)?)?(?:\s*(?:mmhg|/min|%|f|c|g/dl|mg/dl))?"),
    re.compile(r"(?<![\d-])[0-2½]-[0-2½]-[0-2½](?![\d-])"),
    re.compile(r"(?i)\b(?:OD|BD|TDS|QID|HS|SOS|PRN|STAT)\s*(?:x\s*\d+\s*(?:d|days?|wks?|weeks?|months?))?\b"),
    re.compile(r"(?<![\d-])(?!(?:0|91)?[6-9]\d{2,5}-\d{4,7}(?!\d|\.\d))\d{1,6}(?:\.\d+)?-\d{1,6}(?:\.\d+)?(?![\d-])"),
    re.compile(r"(?i)\bx\s*\d+\s*(?:d|days?|wks?|weeks?|months?|yrs?|years?)\b"),
]


class MedicalGuard:
    """Maintains runtime-harvested clinical vocabulary and protects medical spans."""

    def __init__(self) -> None:
        self.clinical_words: set[str] = (
            set(CLINICAL_SYMPTOMS_AND_TERMS) | set(_BARE_EPONYMS) | set(DRUG_NAMES) | set(CLINICAL_ABBREVIATIONS)
        )
        self.hospital_names: set[str] = set()
        self.doctor_full_names: set[str] = set()
        self.doctor_tokens: set[str] = set()
        self._doc_rx: re.Pattern[str] | None = None
        self._doc_rx_n = -1

    def learn_from_entries(self, entries: list[FileEntry]) -> None:
        """Dynamically harvest clean vocabulary, drugs, ICDs, LOINCs, and doctors from input."""
        for entry in entries:
            if entry.status != "sanitized":
                continue
            if entry.is_clean_doc and entry.text_decoded:
                # Mask out doctor lines/mentions first so doctor names (e.g. Ramesh) do not pollute clinical_words!
                clean_masked = _SAFE_SPAN_PATTERNS[0].sub(" ", entry.text_decoded)
                for tok in re.findall(r"[A-Za-z\u0900-\u097f][A-Za-z\u0900-\u097f'\-]{2,}", clean_masked):
                    if tok == "ASHA":
                        continue
                    low = tok.lower()
                    if low not in HINGLISH_STOPWORDS:
                        self.clinical_words.add(low)

            if entry.file_type in ("csv", "tsv") and entry.parsed_data:
                rows = entry.parsed_data
                roles = entry.column_roles
                for r in rows[1:]:
                    for idx, role in roles.items():
                        if idx >= len(r):
                            continue
                        val = str(r[idx]).strip()
                        if not val:
                            continue
                        if role in ("MEDICATION", "DIAGNOSIS", "LAB"):
                            for w in re.findall(r"[A-Za-z\u0900-\u097f]{3,}", val):
                                self.clinical_words.add(w.lower())
                        elif role == "DOCTOR":
                            self._add_doctor(val)

            elif entry.file_type == "fhir" and isinstance(entry.parsed_data, dict):
                for item in entry.parsed_data.get("entry") or []:
                    res = (item or {}).get("resource") or {}
                    rt = res.get("resourceType", "")
                    if rt == "Practitioner":
                        for nm in res.get("name") or []:
                            if isinstance(nm, dict) and isinstance(nm.get("text"), str):
                                self._add_doctor(nm["text"])
                    elif rt == "Organization" and isinstance(res.get("name"), str):
                        self.hospital_names.add(res["name"].lower())
                    elif rt in ("Condition", "Observation", "MedicationRequest", "Procedure"):
                        txt = json.dumps(res, ensure_ascii=False)
                        for w in re.findall(r"[A-Za-z\u0900-\u097f]{4,}", txt):
                            if w.lower() not in {"resource", "resourcetype", "display", "system", "subject", "reference"}:
                                self.clinical_words.add(w.lower())

            elif entry.file_type == "json" and entry.parsed_data is not None:
                self._scan_json_clinical(entry.parsed_data)

            if entry.text_decoded:
                for m in _SAFE_SPAN_PATTERNS[0].finditer(entry.text_decoded):
                    self._add_doctor(m.group(0))

    def _add_doctor(self, raw: str) -> None:
        cleaned = re.sub(r"(?i)^(?:dr\.?|doctor|consultant|attending|physician|डॉ\.?)\s+", "", raw).strip()
        cleaned = re.sub(r"\s*\(.*$", "", cleaned).strip(" .,")
        if cleaned:
            self.doctor_full_names.add(cleaned.lower())
            for tok in re.findall(r"[A-Za-z\u0900-\u097f]{3,}", cleaned):
                self.doctor_tokens.add(tok.lower())

    def _scan_json_clinical(self, obj: Any) -> None:
        stack = [obj]
        while stack:
            node = stack.pop()
            if isinstance(node, dict):
                for k, v in node.items():
                    kl = str(k).lower()
                    if isinstance(v, str):
                        if kl in ("doctor", "ref_doctor", "referred_by"):
                            self._add_doctor(v)
                        elif kl == "hospital":
                            self.hospital_names.add(v.lower())
                        elif kl in ("drug", "display", "test", "procedure", "complaints", "department"):
                            for w in re.findall(r"[A-Za-z\u0900-\u097f]{3,}", v):
                                self.clinical_words.add(w.lower())
                    elif isinstance(v, (dict, list)):
                        stack.append(v)
            elif isinstance(node, list):
                for v in node:
                    if isinstance(v, (dict, list)):
                        stack.append(v)

    def protected_spans(self, text: str) -> list[tuple[int, int, str]]:
        spans: list[tuple[int, int, str]] = []
        for idx, rx in enumerate(_SAFE_SPAN_PATTERNS):
            tag = "DOCTOR" if idx == 0 else "MEDICAL"
            for m in rx.finditer(text):
                spans.append((m.start(), m.end(), tag))
        for m in _EPONYM_RX.finditer(text):
            spans.append((m.start(), m.end(), "MEDICAL"))
        for m in _POSSESSIVE_EPONYM_RX.finditer(text):
            if m.group(1).lower() not in COMMON_INDIAN_NAMES:
                spans.append((m.start(), m.end(), "MEDICAL"))
        if "Yoj" in text or "Ayushman" in text or "JAY" in text or "Abhiya" in text:
            for m in _SCHEME_RX.finditer(text):
                spans.append((m.start(), m.end(), "MEDICAL"))
        names = [n for n in self.doctor_full_names if len(n.split()) >= 2 and len(n) >= 6]
        if names and self._doc_rx_n != len(names):
            ordered = sorted(names, key=lambda x: (-len(x), x))
            self._doc_rx = re.compile(r"(?i)(?<![\w])(?:" + "|".join(re.escape(n) for n in ordered) + r")(?![\w])")
            self._doc_rx_n = len(names)
        if self._doc_rx is not None:
            for m in self._doc_rx.finditer(text):
                spans.append((m.start(), m.end(), "DOCTOR"))
        # Also protect C/O when followed by clinical symptoms/terms
        for m in re.finditer(r"(?i)\bC/O\s*[:\-]?\s*([A-Za-z\u0900-\u097f][A-Za-z\u0900-\u097f\s\-]{1,40})", text):
            rhs_words = [w.lower() for w in re.findall(r"[A-Za-z\u0900-\u097f]{2,}", m.group(1)[:30])]
            if any(w in self.clinical_words for w in rhs_words[:3]):
                spans.append((m.start(), m.end(), "MEDICAL"))
        spans.sort(key=lambda sp: (sp[0], -sp[1]))
        return spans

    def is_protected_span(self, start: int, end: int, protected: list[tuple[int, int, str]], strict: bool = False) -> bool:
        if not protected:
            return False
        # Fast binary search window around sorted protected spans
        idx = bisect_left(protected, (end, -1, ""))
        lo = max(0, idx - 64)
        for k in range(idx - 1, lo - 1, -1):
            ps, pe, tag = protected[k]
            if start < pe and end > ps:
                if strict or tag == "DOCTOR":
                    return True
                if start >= ps and end <= pe:
                    return True
                return True
        if idx > 64:
            for ps, pe, tag in protected[: idx - 64]:
                if pe <= start:
                    continue
                if ps >= end:
                    break
                return True
        return False

    def is_clinical_word(self, word: str) -> bool:
        lw = word.strip(".,:;()[]'\"").lower()
        if lw.endswith(("'s", "’s")) and lw[:-2] in _ALL_EPONYM_ROOTS:
            return True
        if lw in _NAMELIKE_LOWER:
            return False  # "Wilson", "Turner", "Bell": a person unless an eponym frame says otherwise
        if lw in self.clinical_words or lw in HINGLISH_STOPWORDS or lw in _BARE_EPONYMS:
            return True
        return has_clinical_suffix(lw)

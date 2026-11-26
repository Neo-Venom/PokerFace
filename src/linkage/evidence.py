"""Unified structured & free-text identity evidence extractor for record linkage."""
from __future__ import annotations

import datetime as dt
import regex as re
from dataclasses import dataclass, field
from typing import Any

from data.gazetteers import KNOWN_GEO_DISTRICTS_STATES
from src.ingest.profiler import infer_role_from_header, resolve_key_role
from src.models import FileEntry, PatientRecord
from src.nlp.indic_norm import deobfuscate_unicode, name_tokens, normalize_name_token, phonetic_skeleton, shadow_normalize
from src.nlp.medical_guard import MedicalGuard
from src.nlp.pii_ner import extract_context_name_spans, split_safe_chunks
from src.nlp.pii_regex import extract_regex_spans

_AGE_NUM_RE = re.compile(r"(?i)(?:age|आयु|उम्र)?\s*[:\-]?\s*(\d{1,3})\s*(?:yrs?|years?|वर्ष|y)?(?:\s*/\s*([MF]))?\b")
# Structured sex/gender cells ("F", "Male", "महिला").
_GENDER_F_RE = re.compile(r"(?i)\b(?:f|female|महिला)\b|/\s*f\b|\b(?:w/o|d/o|didi|behenji|amma)\b")
_GENDER_M_RE = re.compile(r"(?i)\b(?:m|male|पुरुष)\b|/\s*m\b|\b(?:s/o|sahab|saab|bhaiya|papa|babu)\b")
# Free text: only cues anchored to the record's subject. Honorifics ("Ramesh bhaiya", "sahab", "didi") usually
# address an attendant or relative, and a bare "f"/"m" is too often "f/u" or "5 m", so neither sets the gender.
_TXT_GENDER_RE = re.compile(
    r"(?i)(?:\b(?:sex|gender)\b|लिंग)[ \t]*[:=\-/]?[ \t]*(?P<g1>male|female|m|f|पुरुष|महिला|स्त्री)(?![a-z])"
    r"|\b\d{1,3}[ \t]*(?:y(?:rs?|ears?)?\.?|वर्ष)?[ \t]*/[ \t]*(?P<g2>m|f|male|female|पुरुष|महिला)(?![a-z])"
    r"|\b\d{1,3}[ \t-]*(?:y(?:ea)?rs?|year)[ \t-]*old[ \t]+(?P<g3>man|woman|male|female|boy|girl|gentleman|lady)\b"
    r"|(?P<g4>\b(?:w/o|d/o|s/o)\b|(?<![ऀ-ॿ])(?:पत्नी|पुत्री|पुत्र)(?![ऀ-ॿ]))"
)
_TXT_F = {"f", "female", "महिला", "स्त्री", "woman", "girl", "lady", "w/o", "d/o", "पत्नी", "पुत्री"}
# Age cells written as floats/scientific notation/overflow ("1e309", "99999999999") are corrupt, not ages
_BAD_AGE_CELL_RE = re.compile(r"\d\s*[eE][+-]?\d|\d{4,}|^\s*-")
# Per-visit numbers (admission / IPD / OPD / bill / claim / Rx / lab sample): equal values link two records, but
# two different ones are normal for one patient, so they never count as conflicting registration numbers.
_EPISODE_ID_TOKENS = {
    "ip", "ipd", "op", "opd", "adm", "admn", "admission", "visit", "vis", "enc", "encounter", "bill", "bl",
    "inv", "invoice", "clm", "claim", "rx", "ref", "lab", "smp", "sample", "acc", "ord", "order", "epi",
    "episode", "tkt", "ticket", "req", "dis", "dsc", "er", "cas", "emg", "casualty", "trx", "txn", "rcpt",
}
_EPISODE_HDR_RE = re.compile(r"(?i)ipd|opd|admission|adm[\s_.-]*no|visit|episode|encounter|bill|claim|invoice|receipt|sample|order")
# Devanagari relation words as whole words only ("ब्रह्मपुत्र" is a river/hospital name, not "son of")
_REL_TYPE_RE = re.compile(r"(?i)\b(S/o|D/o|W/o)\b|(?<![ऀ-ॿ])(पुत्री|पुत्र|पत्नी|पिता|पति)(?![ऀ-ॿ])")
_DATE_RE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b|\b(\d{1,2})[./-](\d{1,2})[./-]((?:19|20)\d{2})\b")
_FNAME_SPLIT_RE = re.compile(r"[_\s]+")
_NON_PII_JSON_PARENTS = {"facility", "organization", "lab", "hospital", "doctor", "practitioner", "test", "rx", "diagnosis", "vitals", "panels", "results"}


@dataclass(slots=True)
class RecordEvidence:
    uid: str
    entry: FileEntry
    record: PatientRecord
    is_multi_patient_summary: bool = False
    uhids: set[str] = field(default_factory=set)
    uhid_cores: set[str] = field(default_factory=set)
    aadhaars: set[str] = field(default_factory=set)
    aadhaar_last4: set[str] = field(default_factory=set)
    abhas: set[str] = field(default_factory=set)
    pans: set[str] = field(default_factory=set)
    ins_ids: set[str] = field(default_factory=set)
    emails: set[str] = field(default_factory=set)
    phones: set[str] = field(default_factory=set)
    full_names: list[tuple[str, ...]] = field(default_factory=list)
    given_norms: set[str] = field(default_factory=set)
    surname_norms: set[str] = field(default_factory=set)
    given_skels: set[str] = field(default_factory=set)
    surname_skels: set[str] = field(default_factory=set)
    rel_names: list[tuple[str, ...]] = field(default_factory=list)
    rel_skels: set[str] = field(default_factory=set)
    rel_type: str | None = None  # S/o | D/o | W/o | None
    gender: str | None = None
    dobs: set[dt.date] = field(default_factory=set)
    ages: list[int] = field(default_factory=list)
    birth_years: set[int] = field(default_factory=set)
    clinical_dates: list[dt.date] = field(default_factory=list)
    villages: set[str] = field(default_factory=set)
    districts: set[str] = field(default_factory=set)
    pins: set[str] = field(default_factory=set)
    hospitals: set[str] = field(default_factory=set)
    diagnoses: set[str] = field(default_factory=set)
    episode_ids: set[str] = field(default_factory=set)   # subset of uhids that are per-visit numbers
    translit_tokens: set[str] = field(default_factory=set)  # name tokens romanised from Devanagari


def is_episode_id(raw: str) -> bool:
    """'KRM/IP/2025/7743', 'ADM-25-3397', 'VZC-IP-10462', 'OPD/26/001' are visit numbers, not registrations."""
    return any(p.lower() in _EPISODE_ID_TOKENS for p in re.findall(r"[A-Za-z]+", shadow_normalize(raw)))


def _norm_id_str(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", shadow_normalize(s).lower())


def _digit_core(s: str) -> str:
    d = re.sub(r"\D", "", shadow_normalize(s))
    return d[-6:] if len(d) >= 6 else d


def _parse_date_token(tok: str, dmy: bool = True) -> dt.date | None:
    s = shadow_normalize(tok).strip()
    m = _DATE_RE.search(s)
    if not m:
        return None
    try:
        if m.group(1):
            return dt.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        a, b, y = int(m.group(4)), int(m.group(5)), int(m.group(6))
        day, mon = (a, b) if dmy else (b, a)
        if not (1 <= mon <= 12):
            day, mon = mon, day
        return dt.date(y, mon, day)
    except ValueError:
        return None


_REL_MARKER_RE = re.compile(r"(?i)(?<![a-z])(?:s|d|w|c|h)[ \t]*/[ \t]*o(?![a-z])\.?[ \t]*[:\-]?")
_SURNAME_FIRST_RE = re.compile(r"^\s*([^,;]+?)\s*,\s*([^,;]+?)\s*$")
_HAS_DEVA_RE = re.compile(r"[ऀ-ॿ]")


def _reorder_surname_first(raw: str) -> str:
    """'Iyer, Rajesh' / 'SOREN, Govri' -> 'Rajesh Iyer' (registry exports often write the surname first)."""
    m = _SURNAME_FIRST_RE.match(raw)
    if not m or _REL_MARKER_RE.search(raw):
        return raw
    left, right = m.group(1), m.group(2)
    if len(name_tokens(left)) == 1 and 1 <= len(name_tokens(right)) <= 3:
        return f"{right} {left}"
    return raw


def _add_name_evidence(ev: RecordEvidence, raw: str, is_relative: bool = False) -> None:
    raw = _reorder_surname_first(_REL_MARKER_RE.sub(" ", raw) if is_relative else raw)
    toks = name_tokens(raw)
    if not toks or all(len(t) == 1 for t in toks) and (is_relative or len(toks) > 1):
        return
    if _HAS_DEVA_RE.search(raw):
        for part in re.split(r"[\s_/,.-]+", raw):
            if _HAS_DEVA_RE.search(part):
                ev.translit_tokens.update(name_tokens(part))
    if is_relative:
        ev.rel_names.append(tuple(toks))
        for t in toks:
            sk = phonetic_skeleton(t)
            if sk:
                ev.rel_skels.add(sk)
    else:
        ev.full_names.append(tuple(toks))
        ev.given_norms.add(toks[0])
        gsk = phonetic_skeleton(toks[0])
        if gsk:
            ev.given_skels.add(gsk)
        if len(toks) > 1:
            ev.surname_norms.add(toks[-1])
            ssk = phonetic_skeleton(toks[-1])
            if ssk:
                ev.surname_skels.add(ssk)


_ICD10_RE = re.compile(r"^[A-TV-Z]\d{2}(?:\.\d{1,3})?$", re.I)
_ADDR_LINE_RE = re.compile(r"(?i)^\s*(?:address|res\.?\s*addr(?:ess)?|pata|permanent\s+address|पता|निवास)\s*[:\-]")
_HIST_BEFORE_RE = re.compile(r"(?i)(?:\b(?:h/o|k/c/o|c/o|since|for|past|last|duration|history|approx|x)\b|से|पहले)\s*[:\-]?\s*$")
_HIST_AFTER_RE = re.compile(r"(?i)^\s*(?:back|ago|later|earlier|prior|duration|se|से|पहले|बाद|तक)\b")
_LOCATION_WORD_RE = re.compile(r"(?i)\b(?:school|schul|tehsil|tahsil|block|ward|mohalla|colony|nagar|marg|road|lane|street|hospital|clinic|near|khurd|kalan|gram|village)\b")


def _add_id_evidence(ev: RecordEvidence, kind: str, val: str, episode: bool = False) -> None:
    s = shadow_normalize(val).strip()
    if not s:
        return
    digits = re.sub(r"\D", "", s)
    if kind == "PHONE":
        if len(digits) >= 10:
            ev.phones.add(digits[-10:])
    elif kind == "AADHAAR":
        if len(digits) == 12:
            ev.aadhaars.add(digits)
            ev.aadhaar_last4.add(digits[-4:])
    elif kind == "AADHAAR_MASKED":
        if len(digits) >= 4:
            ev.aadhaar_last4.add(digits[-4:])
    elif kind in ("ABHA", "ABHA_NUMBER"):
        if len(digits) >= 12:
            ev.abhas.add(digits)
    elif kind == "PAN":
        ev.pans.add(s.upper())
    elif kind == "EMAIL":
        ev.emails.add(s.lower())
    elif kind == "INSURANCE_ID":
        ev.ins_ids.add(_norm_id_str(s))
    elif kind == "MRN":
        if _ICD10_RE.match(s):
            return
        nid = _norm_id_str(s)
        if len(nid) >= 4 and len(digits) >= 3:
            ev.uhids.add(nid)
            if episode or is_episode_id(s):
                ev.episode_ids.add(nid)
            dc = _digit_core(s)
            if len(dc) >= 5:
                ev.uhid_cores.add(dc)


def _detect_gender(text: str) -> str | None:
    """Gender from a structured sex/gender cell."""
    f = bool(_GENDER_F_RE.search(text))
    m = bool(_GENDER_M_RE.search(text))
    if f and not m:
        return "F"
    if m and not f:
        return "M"
    return None


def _detect_gender_text(text: str) -> str | None:
    """Gender from free text, using only cues that describe the record's subject (labels, '45 Y / F',
    'a 45-year-old woman', 'W/o' / 'D/o' / 'S/o'). Contradictory cues give None."""
    found: set[str] = set()
    for m in _TXT_GENDER_RE.finditer(text):
        g = (m.group("g1") or m.group("g2") or m.group("g3") or m.group("g4") or "").lower()
        found.add("F" if g in _TXT_F else "M")
        if len(found) > 1:
            return None
    return next(iter(found)) if found else None


def _age_cell_ok(val: str) -> bool:
    return not _BAD_AGE_CELL_RE.search(shadow_normalize(str(val)))


_EVIDENCE_PRIO = {
    "ABHA_NUMBER": 100,
    "EMAIL": 98,
    "PAN": 97,
    "AADHAAR_MASKED": 96,
    "AADHAAR": 95,
    "PHONE": 94,
    "INSURANCE_ID": 93,
    "MRN": 92,
    "DOB": 90,
    "PIN": 85,
    "HOUSE": 80,
    "LOCALITY": 79,
    "LANDMARK": 78,
    "VILLAGE": 77,
    "RELATIVE_NAME": 70,
    "PATIENT_NAME": 65,
}


def _resolve_evidence_overlaps(spans: list[Any]) -> list[Any]:
    ordered = sorted(
        spans,
        key=lambda sp: (-_EVIDENCE_PRIO.get(sp.kind, 50), -(sp.end - sp.start), sp.start),
    )
    kept: list[Any] = []
    for sp in ordered:
        if any(not (sp.end <= ks.start or sp.start >= ks.end) for ks in kept):
            continue
        kept.append(sp)
    return sorted(kept, key=lambda sp: (0 if sp.kind != "PATIENT_NAME" else 1, sp.start))


def _extract_from_text_blob(ev: RecordEvidence, text: str, guard: MedicalGuard, dmy: bool, allow_bare_names: bool = True) -> None:
    text = deobfuscate_unicode(text)
    if len(text) > 49152:
        for ch in split_safe_chunks(text, max_len=32768):
            _extract_from_text_blob(ev, ch, guard, dmy, allow_bare_names=allow_bare_names)
        return
    rx = extract_regex_spans(text)
    ctx = extract_context_name_spans(text, guard, rx)
    prot = guard.protected_spans(text)
    addr_spans = [
        (sp.start, sp.end)
        for sp in rx
        if sp.kind in ("HOUSE", "LOCALITY", "LANDMARK", "VILLAGE", "PIN", "DOB")
    ]
    resolved = _resolve_evidence_overlaps(rx + ctx)
    rel_toks_seen: set[tuple[str, ...]] = set(ev.rel_names)
    for sp in resolved:
        if sp.kind in ("PATIENT_NAME", "RELATIVE_NAME"):
            if guard.is_protected_span(sp.start, sp.end, prot, strict=True):
                continue
            if sp.kind == "PATIENT_NAME":
                if not allow_bare_names:
                    continue
                if any(as_ < sp.end and ae > sp.start for as_, ae in addr_spans):
                    continue
                if _LOCATION_WORD_RE.search(sp.text):
                    continue
                line_start = text.rfind("\n", 0, sp.start) + 1
                line_prefix = text[line_start:sp.start]
                if _ADDR_LINE_RE.match(line_prefix):
                    continue
                nt = tuple(name_tokens(sp.text))
                if nt and nt in rel_toks_seen:
                    continue
            _add_name_evidence(ev, sp.text, is_relative=(sp.kind == "RELATIVE_NAME"))
            if sp.kind == "RELATIVE_NAME":
                rel_toks_seen.update(ev.rel_names)
        elif sp.kind in ("PHONE", "AADHAAR", "AADHAAR_MASKED", "ABHA_NUMBER", "PAN", "EMAIL", "INSURANCE_ID", "MRN"):
            if sp.kind == "PHONE" and any(
                (tag == "DOCTOR" and ps < sp.end and pe > sp.start) or (tag == "MEDICAL" and ps <= sp.start and sp.end <= pe)
                for ps, pe, tag in prot
            ):
                continue
            if sp.kind == "MRN" and any(tag == "DOCTOR" and ps < sp.end and pe > sp.start for ps, pe, tag in prot):
                continue
            _add_id_evidence(ev, sp.kind, sp.text)
        elif sp.kind == "DOB":
            d = _parse_date_token(sp.text, dmy=dmy)
            if d:
                ev.dobs.add(d)
                ev.birth_years.add(d.year)
        elif sp.kind == "PIN":
            digits = re.sub(r"\D", "", shadow_normalize(sp.text))
            if len(digits) == 6:
                ev.pins.add(digits)
        elif sp.kind == "VILLAGE":
            vn = normalize_name_token(sp.text)
            if vn:
                ev.villages.add(vn)

    rm = _REL_TYPE_RE.search(text)
    if rm and not ev.rel_type:
        tok = (rm.group(1) or rm.group(2) or "").lower()
        if tok in ("s/o", "पुत्र", "पिता"):
            ev.rel_type = "S/o"
        elif tok in ("d/o", "पुत्री"):
            ev.rel_type = "D/o"
        elif tok in ("w/o", "पत्नी", "पति"):
            ev.rel_type = "W/o"

    if not ev.gender:
        ev.gender = _detect_gender_text(text)

    norm_txt = shadow_normalize(text)
    norm_prot = guard.protected_spans(norm_txt)
    for m in re.finditer(r"(?i)\b(?:age|आयु|उम्र)[ \t]*[:\-]?[ \t]*(\d{1,3})(?![\dx])[ \t]*(?:yrs?|years?|वर्ष|y)?(?:[ \t]*/[ \t]*([MF])\b)?|\b(\d{1,3})[ \t]*(?:yrs?|years?|वर्ष)(?:[ \t]*/[ \t]*([MF])\b)?|\b(\d{1,3})[ \t]*(?:y[ \t]*)?/[ \t]*([MF])\b", norm_txt):
        # Skip disease durations in clinical history (e.g. "H/O TB 12 yrs back", "x 3 yrs")
        if any(ps < m.end() and pe > m.start() and tag == "MEDICAL" for ps, pe, tag in norm_prot):
            continue
        is_explicit_age = bool(m.group(1) or m.group(4) or m.group(5))
        before = norm_txt[max(0, m.start() - 30):m.start()]
        after = norm_txt[m.end():min(len(norm_txt), m.end() + 20)]
        if _HIST_AFTER_RE.match(after):
            continue
        if not is_explicit_age and _HIST_BEFORE_RE.search(before):
            continue
        a_str = m.group(1) or m.group(3) or m.group(5)
        g_str = m.group(2) or m.group(4) or m.group(6)
        if a_str and 0 <= int(a_str) <= 115:
            ev.ages.append(int(a_str))
        if g_str and not ev.gender:
            ev.gender = g_str.upper()

    for m in _DATE_RE.finditer(shadow_normalize(text)):
        d = _parse_date_token(m.group(0), dmy=dmy)
        if d and 2020 <= d.year <= 2030 and d not in ev.dobs:
            ev.clinical_dates.append(d)

    for g in KNOWN_GEO_DISTRICTS_STATES:
        if len(g) >= 5 and re.search(rf"\b{re.escape(g)}\b", text.lower()):
            ev.districts.add(g.replace(" ", ""))


def _fhir_human_name(nm: Any) -> str:
    """FHIR HumanName: .text, else given[] + family."""
    if isinstance(nm, list):
        nm = next((x for x in nm if isinstance(x, dict)), None)
    if not isinstance(nm, dict):
        return ""
    if isinstance(nm.get("text"), str) and nm["text"].strip():
        return nm["text"]
    given = nm.get("given")
    parts = [g for g in (given if isinstance(given, list) else [given]) if isinstance(g, str)]
    if isinstance(nm.get("family"), str):
        parts.append(nm["family"])
    return " ".join(p.strip() for p in parts if p.strip())


def extract_all_evidence(entries: list[FileEntry], guard: MedicalGuard) -> list[RecordEvidence]:
    """Build RecordEvidence objects for every non-clean patient record."""
    evs: list[RecordEvidence] = []
    for entry in entries:
        if entry.status != "sanitized" or entry.is_clean_doc:
            continue

        # Multi-patient index/vitals tables (or generic non-patient readme) shouldn't be assigned one patient ID!
        is_summary_file = bool(entry.multi_patient)

        if entry.file_type in ("csv", "tsv") and entry.parsed_data:
            rows = entry.parsed_data
            roles = entry.column_roles
            # Multi-patient tables (patients in headers / JSON keys, not rows) are left unlinked
            if is_summary_file:
                for rec in entry.records:
                    evs.append(RecordEvidence(rec.record_uid, entry, rec, is_multi_patient_summary=True))
                continue

            for rec, row in zip(entry.records, rows[1:]):
                ev = RecordEvidence(rec.record_uid, entry, rec)
                first_parts: list[str] = []
                last_parts: list[str] = []
                for idx, cell in enumerate(row):
                    val = str(cell).strip()
                    if not val:
                        continue
                    role = roles.get(idx, "FREE_TEXT")
                    hdr = str(rows[0][idx]).lower() if idx < len(rows[0]) else ""
                    if role == "PATIENT_NAME":
                        hdr_n = re.sub(r"[^a-z]", "", hdr)
                        if any(x in hdr_n for x in ("first", "given", "fore")):
                            first_parts.append(val)
                        elif any(x in hdr_n for x in ("last", "family", "surname")):
                            last_parts.append(val)
                        else:
                            _add_name_evidence(ev, val, is_relative=False)
                    elif role == "RELATIVE_NAME":
                        rm = _REL_TYPE_RE.search(val)
                        if rm:
                            ev.rel_type = (rm.group(1) or "").capitalize() or ev.rel_type
                        elif re.match(r"(?i)^\s*c/o\b", val) and not ev.rel_type:
                            ev.rel_type = "C/O"          # structured care-of column: the guardian
                        cleaned_rel = re.sub(r"(?i)^\s*(?:son|daughter|wife|husband|care)\s+of\s+", "", _REL_MARKER_RE.sub(" ", val)).strip()
                        cleaned_rel = re.sub(r"\([^)]*\)|\[[^\]]*\]", " ", cleaned_rel)          # "(PAN: ...)"
                        cleaned_rel = re.sub(r"(?i)^(?:late|lt\.?|स्व\.?)\s+", "", cleaned_rel.strip()).strip()
                        _add_name_evidence(ev, cleaned_rel, is_relative=True)
                    elif role == "RELATION":
                        if val.lower().startswith(("s", "d", "w")):
                            ev.rel_type = val[:3].capitalize()
                            if not ev.gender:
                                ev.gender = "M" if val.lower().startswith("s") else "F"
                    elif role in ("PHONE", "AADHAAR", "ABHA", "PAN", "INSURANCE_ID", "MRN", "EMAIL"):
                        # Skip bill numbers / sample IDs as patient UHIDs unless UH/REG/PAT
                        if role == "MRN" and ("bill" in hdr or "sample" in hdr):
                            continue
                        if role == "AADHAAR" and "x" in val.lower():
                            _add_id_evidence(ev, "AADHAAR_MASKED", val)
                        else:
                            _add_id_evidence(ev, role, val, episode=(role == "MRN" and bool(_EPISODE_HDR_RE.search(hdr))))
                    elif role == "GENDER":
                        ev.gender = _detect_gender(val)
                    elif role == "DOB":
                        d = _parse_date_token(val, dmy=(entry.date_order_hint == "dmy"))
                        if d:
                            ev.dobs.add(d)
                            ev.birth_years.add(d.year)
                    elif role == "AGE":
                        if _age_cell_ok(val):
                            _extract_from_text_blob(ev, f"Age: {val}", guard, entry.date_order_hint == "dmy")
                    elif role == "VILLAGE":
                        vn = normalize_name_token(val)
                        if vn:
                            ev.villages.add(vn)
                    elif role == "DISTRICT":
                        dn = normalize_name_token(val)
                        if dn:
                            ev.districts.add(dn)
                    elif role == "PIN":
                        digits = re.sub(r"\D", "", shadow_normalize(val))
                        if len(digits) == 6:
                            ev.pins.add(digits)
                    elif role == "DATE":
                        d = _parse_date_token(val, dmy=(entry.date_order_hint == "dmy"))
                        if d:
                            ev.clinical_dates.append(d)
                    elif role == "DIAGNOSIS":
                        ev.diagnoses.add(val.lower())
                    elif role == "ADDRESS":
                        _extract_from_text_blob(ev, val, guard, entry.date_order_hint == "dmy", allow_bare_names=False)
                    elif role in ("NOTES", "FREE_TEXT"):
                        _extract_from_text_blob(ev, val, guard, entry.date_order_hint == "dmy")
                if first_parts or last_parts:
                    _add_name_evidence(ev, " ".join(first_parts + last_parts), is_relative=False)
                _finalize_birth_years(ev)
                evs.append(ev)
            continue

        if entry.file_type == "fhir" and isinstance(entry.parsed_data, dict):
            ev = RecordEvidence(entry.records[0].record_uid, entry, entry.records[0])
            bundle = entry.parsed_data
            src_meta = ((bundle.get("meta") or {}).get("source") or "").lstrip("#")
            if src_meta:
                _add_id_evidence(ev, "MRN", src_meta)
            for item in bundle.get("entry") or []:
                res = (item or {}).get("resource") or {}
                rt = res.get("resourceType", "")
                if rt == "Patient":
                    for nm in res.get("name") or []:
                        nm_txt = _fhir_human_name(nm)
                        if nm_txt:
                            _add_name_evidence(ev, nm_txt, is_relative=False)
                    for ident in res.get("identifier") or []:
                        if isinstance(ident, dict) and isinstance(ident.get("value"), str):
                            sys_u = str(ident.get("system", "")).lower()
                            if "healthid" in sys_u or "abha" in sys_u or "abdm" in sys_u or "ndhm" in sys_u:
                                _add_id_evidence(ev, "ABHA", ident["value"])
                            elif "uidai" in sys_u or "aadha" in sys_u:
                                _add_id_evidence(ev, "AADHAAR", ident["value"])
                            else:
                                _add_id_evidence(ev, "MRN", ident["value"])
                    for tel in res.get("telecom") or []:
                        if isinstance(tel, dict) and isinstance(tel.get("value"), str):
                            _add_id_evidence(ev, "PHONE", tel["value"])
                    if isinstance(res.get("gender"), str):
                        ev.gender = _detect_gender(res["gender"])
                    if isinstance(res.get("birthDate"), str):
                        d = _parse_date_token(res["birthDate"], dmy=True)
                        if d:
                            ev.dobs.add(d)
                            ev.birth_years.add(d.year)
                    for addr in res.get("address") or []:
                        if isinstance(addr, dict):
                            if isinstance(addr.get("city"), str):
                                ev.villages.add(normalize_name_token(addr["city"]))
                            if isinstance(addr.get("district"), str):
                                ev.districts.add(normalize_name_token(addr["district"]))
                            if isinstance(addr.get("postalCode"), str):
                                ev.pins.add(addr["postalCode"][:6])
                    for ct in res.get("contact") or []:
                        if isinstance(ct, dict):
                            nm = _fhir_human_name(ct.get("name"))
                            if nm:
                                _add_name_evidence(ev, nm, is_relative=True)
                elif rt == "Encounter":
                    per = res.get("period") or {}
                    for pk in ("start", "end"):
                        if isinstance(per.get(pk), str):
                            d = _parse_date_token(per[pk], dmy=True)
                            if d:
                                ev.clinical_dates.append(d)
                elif rt == "Condition":
                    cd = (res.get("code") or {}).get("text")
                    if isinstance(cd, str):
                        ev.diagnoses.add(cd.lower())
            _finalize_birth_years(ev)
            evs.append(ev)
            continue

        if entry.file_type == "json" and entry.parsed_data is not None:
            if is_summary_file:
                for rec in entry.records:
                    evs.append(RecordEvidence(rec.record_uid, entry, rec, is_multi_patient_summary=True))
                continue
            # Also collect file-level metadata clues (like source_path) if shared across visits
            file_meta_uhids: set[str] = set()
            file_meta_names: list[str] = []
            if isinstance(entry.parsed_data, dict):
                for mk, meta in entry.parsed_data.items():
                    if not isinstance(meta, dict) or isinstance(meta, list):
                        continue
                    for k2, sp_txt in meta.items():
                        if not isinstance(sp_txt, str) or not any(x in str(k2).lower() for x in ("source", "path", "file")):
                            continue
                        for m in re.finditer(r"([A-Za-z]+_[A-Za-z]+)_([A-Za-z0-9.\-]*\d{5,})", sp_txt):
                            file_meta_names.append(m.group(1).replace("_", " "))
                            file_meta_uhids.add(_digit_core(m.group(2)))

            for rec in entry.records:
                ev = RecordEvidence(rec.record_uid, entry, rec)
                node = rec.raw_payload
                _extract_json_node_evidence(ev, node, guard, entry.date_order_hint == "dmy", entry)
                # Also check filename phone/MRN
                _extract_filename_clues(ev, entry.rel_input_path)
                if len(entry.records) == 1:
                    for uh in file_meta_uhids:
                        ev.uhid_cores.add(uh)
                    for nm in file_meta_names:
                        _add_name_evidence(ev, nm, is_relative=False)
                _finalize_birth_years(ev)
                evs.append(ev)
            continue

        # Text / XML documents
        for rec in entry.records:
            if is_summary_file:
                evs.append(RecordEvidence(rec.record_uid, entry, rec, is_multi_patient_summary=True))
                continue
            ev = RecordEvidence(rec.record_uid, entry, rec)
            txt = entry.text_decoded or ""
            _extract_from_text_blob(ev, txt, guard, entry.date_order_hint == "dmy")
            _extract_filename_clues(ev, entry.rel_input_path)
            _finalize_birth_years(ev)
            evs.append(ev)

    _resolve_filename_clues(evs)
    return evs


_PENDING_FNAME_IDS: dict[int, list[tuple[str, str]]] = {}


def _extract_filename_clues(ev: RecordEvidence, rel_path: str) -> None:
    """IDs / phones embedded in a file name. Kept pending: they only count if the same ID is in some content."""
    stem = rel_path.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    found: list[tuple[str, str]] = []
    for part in [stem] + _FNAME_SPLIT_RE.split(stem):
        for sp in extract_regex_spans(part.upper()):
            if sp.kind in ("MRN", "PHONE"):
                found.append((sp.kind, sp.text))
    if found:
        _PENDING_FNAME_IDS[id(ev)] = found


def _resolve_filename_clues(evs: list[RecordEvidence]) -> None:
    """Accept a filename ID only when the same ID (or phone) occurs in record content somewhere."""
    content_ids = set().union(*(e.uhids for e in evs)) if evs else set()
    content_phones = set().union(*(e.phones for e in evs)) if evs else set()
    for ev in evs:
        for kind, raw in _PENDING_FNAME_IDS.pop(id(ev), []):
            if kind == "PHONE":
                d = re.sub(r"\D", "", shadow_normalize(raw))[-10:]
                if d in content_phones:
                    _add_id_evidence(ev, "PHONE", raw)
            elif _norm_id_str(raw) in content_ids:
                _add_id_evidence(ev, "MRN", raw)
    _PENDING_FNAME_IDS.clear()


_REL_PARENT_KEYS = {"guardian", "contact", "relative", "kin", "nok", "attendant", "spouse", "father", "husband", "mother"}


def _extract_json_node_evidence(ev: RecordEvidence, node: Any, guard: MedicalGuard, dmy: bool, entry: Any = None) -> None:
    stack: list[tuple[str, Any]] = [("", node)]
    first_parts: list[str] = []
    last_parts: list[str] = []
    while stack:
        pk, cur = stack.pop()
        if isinstance(cur, dict):
            for k, v in cur.items():
                kl = str(k).lower()
                if isinstance(v, (dict, list)):
                    stack.append((kl, v))
                elif isinstance(v, (int, float)) and not isinstance(v, bool) and pk not in _NON_PII_JSON_PARENTS:
                    iv = int(v) if isinstance(v, int) or float(v).is_integer() else None
                    if iv is not None:
                        if 91_000_000_000_000 <= iv <= 91_999_999_999_999:
                            _add_id_evidence(ev, "ABHA", str(iv))
                            continue
                        if 100_000_000_000 <= iv <= 999_999_999_999:
                            _add_id_evidence(ev, "AADHAAR", str(iv))
                            continue
                        if 6_000_000_000 <= iv <= 9_999_999_999:
                            _add_id_evidence(ev, "PHONE", str(iv))
                            continue
                if not isinstance(v, (dict, list)) and v is not None:
                    sv = str(v).strip()
                    if not sv:
                        continue
                    if kl in ("display", "code", "complaints", "test", "loinc", "range", "unit", "drug", "strength", "frequency", "duration"):
                        if kl in ("display", "code", "complaints"):
                            ev.diagnoses.add(sv.lower())
                        continue
                    role = resolve_key_role(entry, str(k)) if entry is not None else infer_role_from_header(str(k))[0]
                    pk_rel = pk in _REL_PARENT_KEYS or infer_role_from_header(pk)[0] == "RELATIVE_NAME"
                    if (kl in ("first", "last", "given", "family", "forename", "surname", "local", "name", "full_name") or role == "PATIENT_NAME") and pk not in _NON_PII_JSON_PARENTS:
                        if pk_rel:
                            _add_name_evidence(ev, sv, is_relative=True)
                        elif kl in ("first", "given", "forename"):
                            first_parts.append(sv)
                        elif kl in ("last", "family", "surname"):
                            last_parts.append(sv)
                        else:
                            _add_name_evidence(ev, sv, is_relative=False)
                    elif role == "PATIENT_NAME" and pk not in _NON_PII_JSON_PARENTS:
                        _add_name_evidence(ev, sv, is_relative=False)
                    elif role == "RELATIVE_NAME":
                        _add_name_evidence(ev, re.sub(r"(?i)^(?:S/o|D/o|W/o|C/O)\s+", "", sv), is_relative=True)
                    elif kl in ("relation",):
                        ev.rel_type = sv[:3].capitalize()
                        if not ev.gender and sv.lower().startswith(("d", "w")):
                            ev.gender = "F"
                        elif not ev.gender and sv.lower().startswith("s"):
                            ev.gender = "M"
                    elif pk not in _NON_PII_JSON_PARENTS and (role in ("PHONE", "AADHAAR", "ABHA", "PAN", "INSURANCE_ID", "MRN", "EMAIL") or kl in ("aadhar_no", "callback", "patient_ref", "hospital_uhid", "uhid")):
                        if kl in ("visit_id", "id") and "opd/" in sv.lower():
                            continue
                        eff_role = "AADHAAR" if kl == "aadhar_no" else ("PHONE" if kl == "callback" else ("MRN" if kl in ("patient_ref", "hospital_uhid", "uhid") else (role or "MRN")))
                        _add_id_evidence(ev, eff_role, sv, episode=(eff_role == "MRN" and bool(_EPISODE_HDR_RE.search(kl))))
                    elif role == "GENDER" or kl in ("sex", "gender"):
                        ev.gender = _detect_gender(sv)
                    elif role == "DOB" or kl == "dob":
                        d = _parse_date_token(sv, dmy=dmy)
                        if d:
                            ev.dobs.add(d)
                            ev.birth_years.add(d.year)
                    elif role == "AGE" or kl in ("age", "age_sex"):
                        if _age_cell_ok(sv):
                            _extract_from_text_blob(ev, f"Age: {sv}", guard, dmy)
                    elif role == "VILLAGE" or (kl in ("village", "city") and pk not in _NON_PII_JSON_PARENTS):
                        ev.villages.add(normalize_name_token(sv))
                    elif role == "DISTRICT":
                        ev.districts.add(normalize_name_token(sv))
                    elif role == "PIN" or kl in ("pin", "postalcode"):
                        digits = re.sub(r"\D", "", shadow_normalize(sv))
                        if len(digits) == 6:
                            ev.pins.add(digits)
                    elif kl in ("visit_date", "collected", "registered_on", "date"):
                        d = _parse_date_token(sv, dmy=dmy)
                        if d:
                            ev.clinical_dates.append(d)
                    elif kl in ("notes", "remarks"):
                        _extract_from_text_blob(ev, sv, guard, dmy)
                    elif kl in ("address_note", "line"):
                        _extract_from_text_blob(ev, sv, guard, dmy, allow_bare_names=False)
        elif isinstance(cur, list):
            for x in cur:
                if isinstance(x, (dict, list)):
                    stack.append((pk, x))
                elif isinstance(x, str) and pk == "line":
                    _extract_from_text_blob(ev, x, guard, dmy, allow_bare_names=False)
    if first_parts or last_parts:
        _add_name_evidence(ev, " ".join(first_parts + last_parts), is_relative=False)


def _finalize_birth_years(ev: RecordEvidence) -> None:
    if ev.abhas:
        # Guard against 12-digit suffixes of 91-XXXX-XXXX-XXXX ABHA being misclassified as Aadhaar
        false_aads = {a for a in ev.aadhaars if any(ab.startswith("91") and ab.endswith(a) for ab in ev.abhas)}
        if false_aads:
            ev.aadhaars -= false_aads
            ev.aadhaar_last4 -= {a[-4:] for a in false_aads}
    if not ev.birth_years and ev.ages:
        ref_year = ev.clinical_dates[0].year if ev.clinical_dates else 2026
        for a in ev.ages:
            ev.birth_years.add(ref_year - a)

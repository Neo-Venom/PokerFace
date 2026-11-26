"""M2 Step 1: Per-record identity evidence extraction (structured + free text, multi-script)."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any

from raksha.detect.normalize import build_normalized_view, deva_word_to_latin, phonetic_key
from raksha.detect.structured import _is_fhir_bundle, infer_column_type

_DEVA_RE = re.compile(r"[\u0900-\u0963\u0970-\u097F]")
_DL = r"\u0900-\u0963\u0970-\u097F"

_RE_PHONE_10 = re.compile(r"(?<!\d)(?:\+?91[\s.\-]?|0)?([6-9]\d{4}[\s.\-]?\d{5}|[6-9]\d{2}[\s.\-]?\d{3}[\s.\-]?\d{4})(?!\d)")
_RE_AADHAAR_12 = re.compile(r"(?<!\d)(\d{4}[\s\-]?\d{4}[\s\-]?\d{4})(?!\d)")
_RE_AADHAAR_MASK = re.compile(r"(?<!\w)[xX*]{4}[\s\-]?[xX*]{4}[\s\-]?(\d{4})(?!\w)")
_RE_ABHA_14 = re.compile(r"(?<!\d)(\d{2}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4})(?!\d)")
_RE_ABHA_ADDR = re.compile(r"(?<![\w.+-])([A-Za-z0-9._+-]{2,}@(?:abdm|ndhm|sbx))\b", re.I)
_RE_EMAIL = re.compile(r"(?<![\w.+-])([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})\b")
_RE_PAN = re.compile(r"\b([A-Z]{5}\d{4}[A-Z])\b")
_RE_INS = re.compile(r"\b((?:PMJAY|AB-?PMJAY|CGHS|ESIS|RSBY)[-/][A-Z0-9]{1,8}(?:[-/][A-Z0-9]{2,})+|\bPOL/\d{2,4}/\d{4,})\b", re.I)
_RE_MRN_FREE = re.compile(
    r"\b("
    r"[A-Z]{2,6}/\d{2,4}/\d{3,}"
    r"|[A-Z]{2,6}-\d{4,}"
    r"|[A-Z]{2,6}\d{4}-\d{3,}"
    r"|[A-Z]{2,6}-[A-Z]{2,4}-\d{3,}"
    r"|[A-Z]{2,5}-\d{2,4}-\d{3,}"
    r"|[A-Z]{2,5}\d{5,}"
    r")\b"
)
_RE_MRN_LABELED = re.compile(
    r"(?i)(?:\bUHID|\bMRN|\bReg\.?\s*No\.?|\bIPD\s*No\.?|पंजीकरण\s*सं\.?|आईपीडी\s*सं\.?)\s*[:#]\s*([A-Z0-9][A-Z0-9/\-]{3,24})"
)

# Hospital-prefixed UHIDs that uniquely carry their own hospital scope across external labs/billing
_PREFIXED_MRN_RE = re.compile(r"^(?:[A-Z]{2,6}/\d{2,4}/\d{3,}|[A-Z]{2,6}-\d{4,}|[A-Z]{2,6}\d{4}-\d{3,})$")
# Generic non-scoped prefixes (IPD-2026-00432, IP/26/00130, ADM000162, UHID-010610 etc.)
_GENERIC_PREFIX_WORDS = {"UHID", "MRN", "IPD", "IP", "ADM", "PAT", "REG", "OPD"}

_NON_PATIENT_FILES = {"clean/"}


@dataclass(frozen=True)
class NameForm:
    raw: str
    norm_tokens: tuple[str, ...]
    phonetic_tokens: tuple[str, ...]
    is_fragment_first_only: bool = False


@dataclass(frozen=True)
class RelativeClue:
    relation: str          # "S/o", "D/o", "W/o", "C/O", ""
    norm_tokens: tuple[str, ...]
    phonetic_tokens: tuple[str, ...]


@dataclass
class RecordEvidence:
    """Extracted identity evidence for one exportable record unit (rel, loc)."""

    rel: str
    loc: str
    content_hash: str
    hospital: str
    is_patient_record: bool = True
    names: list[NameForm] = field(default_factory=list)
    relatives: list[RelativeClue] = field(default_factory=list)
    sex: str | None = None
    dob: str | None = None              # canonical YYYY-MM-DD
    age: int | None = None
    record_year: int | None = None
    villages: set[str] = field(default_factory=set)          # normalized Latin-folded village
    village_phonetics: set[str] = field(default_factory=set)
    pins: set[str] = field(default_factory=set)
    phones: set[str] = field(default_factory=set)
    aadhaars: set[str] = field(default_factory=set)
    aadhaar_last4: set[str] = field(default_factory=set)
    abhas: set[str] = field(default_factory=set)
    pans: set[str] = field(default_factory=set)
    emails: set[str] = field(default_factory=set)
    insurances: set[str] = field(default_factory=set)
    # (scope, id_core) where scope is hospital key or unique hospital prefix
    scoped_mrns: set[tuple[str, str]] = field(default_factory=set)
    # Cross-hospital prefix-bearing IDs (when prefix itself identifies the hospital, not generic "UHID"/"IPD"/"ADM")
    global_mrns: set[str] = field(default_factory=set)
    record_date: str | None = None
    has_strong_id: bool = False
    is_fragmentary: bool = False


def _canon_hospital(raw: str) -> str:
    s = re.sub(r"[^a-z0-9]+", " ", unicodedata.normalize("NFKC", raw).casefold()).strip()
    if not s:
        return ""
    # Strip trailing city/generic words for matching across "Hospital Name, City" vs "Hospital Name"
    toks = [t for t in s.split() if t not in ("multispeciality", "charitable", "hospital", "clinic", "trust", "seva", "sadan", "centre", "center", "district")]
    return "_".join(toks[:3]) if toks else s.replace(" ", "_")


def _infer_hospital_from_rel(rel: str, acronym_map: dict[str, str] | None = None) -> str:
    stem = rel.rsplit("/", 1)[-1].casefold()
    m = re.match(r"^(?:rx_|[a-z]+_ds_)?([a-z]+?)_(?:registration|patients|claims|opd|ds|results|legacy)", stem)
    if m:
        tok = m.group(1)
        if acronym_map and tok in acronym_map:
            return acronym_map[tok]
        return _canon_hospital(tok)
    return ""


def _build_hospital_acronym_map(parsed_items: list[tuple[str, str, Any]]) -> dict[str, str]:
    """Build hospital stem/acronym -> canonical hospital key purely from corpus headers/metadata."""
    raw_names: set[str] = set()
    for _rel, fmt, payload in parsed_items:
        if fmt == "json" and isinstance(payload, dict):
            if isinstance(payload.get("_export_meta"), dict) and isinstance(payload["_export_meta"].get("hospital"), str):
                raw_names.add(payload["_export_meta"]["hospital"])
            if isinstance(payload.get("facility"), dict) and isinstance(payload["facility"].get("name"), str):
                raw_names.add(payload["facility"]["name"])
            if _is_fhir_bundle(payload):
                for ent in payload.get("entry", []):
                    res = ent.get("resource", {}) if isinstance(ent, dict) else {}
                    if res.get("resourceType") == "Organization" and isinstance(res.get("name"), str):
                        raw_names.add(res["name"])
        elif fmt == "txt" and isinstance(payload, str) and payload.strip():
            first_line = payload.splitlines()[0].strip()
            if any(w in first_line.casefold() for w in ("hospital", "sadan", "clinic", "trust", "institute", "medical")):
                raw_names.add(re.split(r"\s*[-–,]\s*(?:OPD|ओपीडी|Diagnostic)", first_line)[0])

    acr: dict[str, str] = {}
    for rn in sorted(raw_names):
        words = [w for w in re.findall(r"[a-z]+", rn.casefold()) if w]
        canon = _canon_hospital(rn)
        if len(words) >= 2 and canon:
            initials = "".join(w[0] for w in words)
            if len(initials) >= 2:
                acr.setdefault(initials, canon)
            # Also handle initial + last word pattern (e.g., first letters + last token: "dh" + "hardoi")
            if len(words) >= 3:
                acr.setdefault("".join(w[0] for w in words[:-1]) + words[-1], canon)
            acr.setdefault(words[-1], canon)
    return acr


def _fold_text_tokens(raw: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    s = unicodedata.normalize("NFC", raw)
    s = re.sub(r"[^\w\s.\u0900-\u097F'-]", " ", s)
    toks: list[str] = []
    ptoks: list[str] = []
    for part in s.split():
        p = part.strip(".,'\"-")
        if not p:
            continue
        if _DEVA_RE.search(p):
            lat = deva_word_to_latin(p)
        else:
            lat = p.casefold()
        lat = re.sub(r"[^a-z0-9]", "", lat)
        if not lat:
            continue
        toks.append(lat)
        pk = phonetic_key(p)
        ptoks.append(pk.replace(" ", "") if pk else lat)
    return tuple(toks), tuple(ptoks)


def make_name_form(raw: str, is_fragment: bool = False) -> NameForm | None:
    s = re.sub(r"(?i)^\s*(?:mr\.?|mrs\.?|ms\.?|smt\.?|shri|श्री|श्रीमती|pt\.?)\s+", "", raw.strip())
    s = re.sub(r"(?i)\s+(?:ji|sahab|saheb|saab|didi|behenji|bahenji|bhai|bhaiya|amma|जी|साहब|दीदी)\s*$", "", s).strip()
    if not s or any(ch.isdigit() for ch in s):
        return None
    # Handle "Last, First"
    if "," in s and len(s.split(",")) == 2:
        p1, p2 = [x.strip() for x in s.split(",")]
        if p1 and p2 and len(p1.split()) == 1 and len(p2.split()) <= 2:
            s = f"{p2} {p1}"
    ntoks, ptoks = _fold_text_tokens(s)
    if not ntoks:
        return None
    return NameForm(
        raw=s,
        norm_tokens=ntoks,
        phonetic_tokens=ptoks,
        is_fragment_first_only=is_fragment or len(ntoks) == 1,
    )


def _norm_sex(raw: str | None) -> str | None:
    if not raw:
        return None
    s = unicodedata.normalize("NFKC", str(raw)).strip().casefold()
    if s in ("m", "male", "पुरुष") or s.endswith("/m"):
        return "M"
    if s in ("f", "female", "महिला") or s.endswith("/f"):
        return "F"
    m = re.search(r"/\s*([mf])\b", s)
    return m.group(1).upper() if m else None


def _parse_date_str(raw: str | None, month_first_hint: bool = False) -> str | None:
    if not raw:
        return None
    nv = build_normalized_view(str(raw).strip())
    s = nv.clean.strip()
    # ISO YYYY-MM-DD (optionally with time)
    m_iso = re.match(r"^(\d{4})[-/](\d{1,2})[-/](\d{1,2})(?:T|\s|$)", s)
    if m_iso:
        y, mo, d = int(m_iso.group(1)), int(m_iso.group(2)), int(m_iso.group(3))
        if 1900 <= y <= 2100 and 1 <= mo <= 12 and 1 <= d <= 31:
            return f"{y:04d}-{mo:02d}-{d:02d}"
        return None
    m_dmy = re.match(r"^(\d{1,2})[-/.](\d{1,2})[-/.](\d{4})(?:\s|$)", s)
    if m_dmy:
        p1, p2, y = int(m_dmy.group(1)), int(m_dmy.group(2)), int(m_dmy.group(3))
        if not (1900 <= y <= 2100):
            return None
        if p1 > 12 and 1 <= p2 <= 12:
            d, mo = p1, p2
        elif p2 > 12 and 1 <= p1 <= 12:
            mo, d = p1, p2
        elif month_first_hint:
            mo, d = p1, p2
        else:
            d, mo = p1, p2
        if 1 <= mo <= 12 and 1 <= d <= 31:
            try:
                dt.date(y, mo, d)
                return f"{y:04d}-{mo:02d}-{d:02d}"
            except ValueError:
                return None
    return None


def _infer_column_month_first(vals: list[Any], header: str) -> bool:
    hl = header.casefold()
    if "mm/dd" in hl or "mm-dd" in hl:
        return True
    if "dd/mm" in hl or "dd-mm" in hl:
        return False
    p2_gt12 = 0
    p1_gt12 = 0
    for v in vals:
        if not v:
            continue
        m = re.match(r"^\s*(\d{1,2})[-/.](\d{1,2})[-/.](\d{4})", str(v))
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            if b > 12:
                p2_gt12 += 1
            if a > 12:
                p1_gt12 += 1
    return p2_gt12 > p1_gt12


def _add_mrn(ev: RecordEvidence, raw_mrn: str) -> None:
    s = unicodedata.normalize("NFKC", raw_mrn).strip()
    if not s or len(s) < 4:
        return
    if s.startswith(("BL/", "OPD/")):
        return
    canon = s.upper()
    ev.scoped_mrns.add((ev.hospital or "unknown", canon))
    # Extract alphabetic prefix to decide if MRN is hospital-self-scoped
    m_pre = re.match(r"^([A-Z]{2,6})(?:[-/]|\d{4}-)", canon)
    if m_pre and m_pre.group(1) not in _GENERIC_PREFIX_WORDS:
        ev.global_mrns.add(canon)
    ev.has_strong_id = True


def _add_relative(ev: RecordEvidence, relation: str, raw_name: str) -> None:
    s = raw_name.strip(" .,:;-'\"")
    if not s or any(ch.isdigit() for ch in s):
        return
    m_pre = re.match(r"(?i)^([SDWC]\s*/\s*[Oo])\s+(.+)$", s)
    if m_pre:
        relation = m_pre.group(1).replace(" ", "").upper()[:1] + "/o"
        s = m_pre.group(2).strip()
    ntoks, ptoks = _fold_text_tokens(s)
    if ntoks:
        ev.relatives.append(RelativeClue(relation=relation, norm_tokens=ntoks, phonetic_tokens=ptoks))


def _add_village(ev: RecordEvidence, raw_vill: str) -> None:
    s = re.sub(r"(?i)^(?:vill\.?|village|gram|ग्राम)\s*[:\-]?\s*", "", raw_vill.strip(" .,-"))
    if not s or any(ch.isdigit() for ch in s):
        return
    ntoks, _ = _fold_text_tokens(s)
    if ntoks:
        ev.villages.add(" ".join(ntoks))
        pk = phonetic_key(s)
        if pk:
            ev.village_phonetics.add(pk)


def _extract_address_clues(ev: RecordEvidence, addr: str) -> None:
    nv = build_normalized_view(addr)
    c = nv.clean
    m_co = re.search(r"(?i)\bC/O\s+([A-Z" + _DL + r"][A-Za-z" + _DL + r".'\-]+(?:\s+[A-Z" + _DL + r"][A-Za-z" + _DL + r".'\-]+){0,3})(?=,|$)", c)
    if m_co:
        _add_relative(ev, "C/O", m_co.group(1))
    for m_v in re.finditer(r"(?i)(?:\bVill\.?|\bVillage|\bGram|ग्राम)\s+([^,\n\-]+)", c):
        _add_village(ev, m_v.group(1))
    m_tehsil_vill = re.search(r"(?i)(?<=,\s)(?!(?:H\.?\s*No|House|Makan|Ward|Mohalla|near|opp|Tehsil|Distt|C/O)\b)([A-Z" + _DL + r"][^\n,]+?)(?=,\s*(?:Tehsil|तहसील)\b)", c)
    if m_tehsil_vill:
        _add_village(ev, m_tehsil_vill.group(1))
    m_pin = re.search(r"(?<!\d)([1-8]\d{5})(?!\d)", c)
    if m_pin:
        ev.pins.add(m_pin.group(1))


def _extract_free_text_clues(ev: RecordEvidence, text: str, month_first_hint: bool = False) -> None:
    nv = build_normalized_view(text)
    c = nv.clean

    # Hospital header on first line
    if not ev.hospital:
        first_line = c.splitlines()[0].strip() if c.splitlines() else ""
        if any(w in first_line.casefold() for w in ("hospital", "sadan", "clinic", "pathology", "diagnostics", "labs", "trust", "institute")):
            hosp_part = re.split(r"\s*[-–,]\s*(?:OPD|ओपीडी|Diagnostic)", first_line)[0]
            ev.hospital = _canon_hospital(hosp_part)

    # Strong IDs
    for m in _RE_ABHA_ADDR.finditer(c):
        ev.abhas.add(m.group(1).casefold())
        ev.has_strong_id = True
    for m in _RE_EMAIL.finditer(c):
        em = m.group(1).casefold()
        if not em.endswith(("@abdm", "@ndhm", "@sbx")):
            ev.emails.add(em)
            ev.has_strong_id = True
    for m in _RE_PAN.finditer(c):
        ev.pans.add(m.group(1).upper())
        ev.has_strong_id = True
    abha_spans: list[tuple[int, int]] = []
    for m in _RE_ABHA_14.finditer(c):
        digits = re.sub(r"\D", "", m.group(1))
        if len(digits) == 14:
            ev.abhas.add(digits)
            ev.has_strong_id = True
            abha_spans.append((m.start(), m.end()))
    for m in _RE_AADHAAR_12.finditer(c):
        # Skip trailing/overlapping substrings of 14-digit ABHA IDs or compound dashed tokens
        if any(not (m.end() <= a or m.start() >= b) for a, b in abha_spans):
            continue
        before = c[max(0, m.start() - 2) : m.start()]
        after = c[m.end() : m.end() + 15]
        if re.search(r"\d[- ]$", before) or re.match(r"^\s*(?:/uL|mg|IU|-)|^[- ]\d", after, re.I):
            continue
        digits = re.sub(r"\D", "", m.group(1))
        if len(digits) == 12 and digits not in ev.abhas:
            ev.aadhaars.add(digits)
            ev.aadhaar_last4.add(digits[-4:])
            ev.has_strong_id = True
    for m in _RE_AADHAAR_MASK.finditer(c):
        ev.aadhaar_last4.add(m.group(1))
    for m in _RE_PHONE_10.finditer(c):
        after = c[m.end() : m.end() + 15]
        if re.match(r"^\s*(?:/uL|mg|IU|-)", after, re.I):
            continue
        digits = re.sub(r"\D", "", m.group(1))[-10:]
        if len(digits) == 10:
            ev.phones.add(digits)
    for m in _RE_INS.finditer(c):
        ev.insurances.add(m.group(1).upper())
        ev.has_strong_id = True
    for rx in (_RE_MRN_FREE, _RE_MRN_LABELED):
        for m in rx.finditer(c):
            _add_mrn(ev, m.group(1))

    # Patient name labels
    for m in re.finditer(
        r"(?i)(?:Patient(?:'s)?\s+Name|Pt\.?\s+Name|Beneficiary|Client\s+Name|Name|मरीज़\s+का\s+नाम|रोगी\s+का\s+नाम|नाम)\s*[:\-]\s*([^\n\r,;]+?)(?=\s{2,}|\s+(?:Age|Sex|Gender|DOB|UHID|MRN|Reg|IPD|आयु|उम्र|लिंग|पंजीकरण)\b|\n|$)",
        c,
    ):
        nf = make_name_form(m.group(1), is_fragment=False)
        if nf:
            ev.names.append(nf)

    # Fragment / inline patient names
    for m in re.finditer(r"(?i)\bReport\s+of\s+([A-Z" + _DL + r"][A-Za-z" + _DL + r".'\-]+(?:\s+(?!\d)[A-Za-z" + _DL + r".'\-]+){0,2})(?=\s+\d{1,3}|\s|$)", c):
        nf = make_name_form(m.group(1), is_fragment=True)
        if nf:
            ev.names.append(nf)
    for m in re.finditer(r"(?:^|\n)\s*([A-Z" + _DL + r"][A-Za-z" + _DL + r".'\-]+(?:\s+[A-Z" + _DL + r"][A-Za-z" + _DL + r".'\-]+){0,2})\s*,\s*(?=[SDWC]/[Oo]\b)", c):
        nf = make_name_form(m.group(1), is_fragment=True)
        if nf:
            ev.names.append(nf)
    for m in re.finditer(r"(?:^|\n)\s*\d{1,2}:\d{2}\s*[-–]\s*(?!Dr\.|डॉ|Patient\b|Next\b)([A-Z" + _DL + r"][A-Za-z" + _DL + r".'\-]+(?:\s+[A-Z" + _DL + r"][A-Za-z" + _DL + r".'\-]+){0,3})(?=\s*[-–]\s*K/C/O)", c):
        nf = make_name_form(m.group(1), is_fragment=False)
        if nf:
            ev.names.append(nf)
    for m in re.finditer(r"([A-Z" + _DL + r"][A-Za-z" + _DL + r".'\-]+(?:\s+[A-Z" + _DL + r"][A-Za-z" + _DL + r".'\-]+){0,2})\s+(?:ji|sahab|saheb|saab|didi|behenji|bahenji|bhai|bhaiya|amma|जी|साहब|दीदी)\s+(?:ka|ki|ke)\b", c):
        nf = make_name_form(m.group(1), is_fragment=True)
        if nf:
            ev.names.append(nf)
    for m in re.finditer(r"([" + _DL + r"]{2,20}(?:\s+[" + _DL + r"]{2,20}){0,2})\s+को\s+\d+\s*दिन\s*से", c):
        tok0 = m.group(1).strip()
        if tok0 not in ("मरीज़", "रोगी"):
            nf = make_name_form(tok0, is_fragment=False)
            if nf:
                ev.names.append(nf)

    # Relatives
    for m in re.finditer(r"(?i)\b([SDW]/[Oo])\s+([A-Z" + _DL + r"][A-Za-z" + _DL + r".'\-]+(?:\s+[A-Z" + _DL + r"][A-Za-z" + _DL + r".'\-]+){0,3})(?=[,.;\)\n\r]|\s+(?:discharged|informed|Mob|Phone)\b|$)", c):
        _add_relative(ev, m.group(1), m.group(2))
    for m in re.finditer(r"(?i)(?:पिता\s+का\s+नाम|पति\s+का\s+नाम|माता\s+का\s+नाम|अभिभावक|Attendant|Caretaker)\s*[:\-]\s*([^\n\r,;]+)", c):
        _add_relative(ev, "", m.group(1))

    # Age / Sex / DOB
    m_agesex = re.search(r"(?i)(?:Age(?:/Sex)?|आयु)\s*[:\-]\s*(\d{1,3})\s*(?:Yrs?|वर्ष|Y)?\s*(?:/\s*([MF]|पुरुष|महिला))?", c)
    if m_agesex:
        ev.age = int(m_agesex.group(1))
        if m_agesex.group(2):
            ev.sex = _norm_sex(m_agesex.group(2))
    else:
        m_wa = re.search(r"\b(\d{1,3})\s*Y\s*/\s*([MF])\b", c, re.I)
        if m_wa:
            ev.age = int(m_wa.group(1))
            ev.sex = _norm_sex(m_wa.group(2))
    if not ev.sex:
        if any(r.relation == "S/o" for r in ev.relatives):
            ev.sex = "M"
        elif any(r.relation in ("D/o", "W/o") for r in ev.relatives):
            ev.sex = "F"

    m_dob = re.search(r"(?i)(?:\bDOB|\bDate\s+of\s+Birth|जन्म\s*तिथि)\s*[:\-]\s*(\d{1,4}[./\-]\d{1,2}[./\-]\d{1,4})", c)
    if m_dob:
        parsed_dob = _parse_date_str(m_dob.group(1), month_first_hint=month_first_hint)
        if parsed_dob:
            ev.dob = parsed_dob

    # Address / Village
    for m in re.finditer(r"(?i)(?:Address(?:\s*\([^)]*\))?|पता)\s*[:\-]\s*([^\n\r]+)", c):
        _extract_address_clues(ev, m.group(1))
    for m in re.finditer(r"(?i)(?:\bVill(?:age)?\.?|\bGram|ग्राम)\s*[:\-]\s*([^\n\r,;]+)", c):
        _add_village(ev, m.group(1))

    # Record date (for age-drift tolerance)
    m_dt = re.search(r"(?i)(?:Date(?:\s+of\s+Admission)?|तारीख|भर्ती\s+की\s+तारीख)\s*[:\-]\s*(\d{1,4}[./\-]\d{1,2}[./\-]\d{1,4})", c)
    if m_dt:
        rd = _parse_date_str(m_dt.group(1), month_first_hint=month_first_hint)
        if rd:
            ev.record_date = rd
            ev.record_year = int(rd[:4])
    if ev.record_year is None:
        m_yr = re.search(r"\b(202[0-9])\b", c)
        if m_yr:
            ev.record_year = int(m_yr.group(1))


def _record_hash(obj: Any) -> str:
    raw = json.dumps(obj, ensure_ascii=False, sort_keys=True) if not isinstance(obj, str) else obj
    return hashlib.sha256(raw.encode("utf-8", errors="replace")).hexdigest()


def _is_non_patient_path(rel: str) -> bool:
    low = rel.casefold()
    return any(tok in low for tok in _NON_PATIENT_FILES)


def get_json_record_units(data: Any) -> list[tuple[str, Any]]:
    """Determine record units (loc, sub_object) matching SPEC.md and records.jsonl conventions."""
    if not isinstance(data, dict) or _is_fhir_bundle(data):
        return [("doc", data)]
    # Collections representing multiple independent patient/visit/lab rows (exclude 'panels' inside single lab reports)
    multi_keys = [
        k for k, v in data.items()
        if k != "panels" and isinstance(v, list) and v and all(isinstance(x, dict) for x in v)
    ]
    if not multi_keys:
        return [("doc", data)]
    units: list[tuple[str, Any]] = []
    for k in multi_keys:
        pk = k.replace("~", "~0").replace("/", "~1")
        for i, item in enumerate(data[k]):
            units.append((f"/{pk}/{i}", item))
    return units


def extract_all_evidences(parsed_items: list[tuple[str, str, Any]]) -> list[RecordEvidence]:
    """Extract RecordEvidence objects across all parsed input files in deterministic order."""
    from raksha.privacy.dates import infer_month_first

    evidences: list[RecordEvidence] = []
    acronym_map = _build_hospital_acronym_map(parsed_items)

    for rel, fmt, payload in parsed_items:
        if _is_non_patient_path(rel):
            continue
        base_hosp = _infer_hospital_from_rel(rel, acronym_map=acronym_map)

        if fmt in ("csv", "tsv") and isinstance(payload, list) and len(payload) > 1:
            header, body = payload[0], payload[1:]
            col_types = [
                infer_column_type([r[i] if i < len(r) else "" for r in body], header=h)
                for i, h in enumerate(header)
            ]
            # If table has no patient/MRN/ID/phone columns at all, treat rows as non-patient
            has_patient_cols = any(ct in ("PATIENT_NAME", "MRN", "AADHAAR", "ABHA_NUMBER", "PHONE") for ct in col_types)
            if not has_patient_cols:
                continue

            date_month_first: dict[int, bool] = {}
            for c_idx, ct in enumerate(col_types):
                if ct == "DOB" or "date" in header[c_idx].casefold():
                    date_month_first[c_idx] = _infer_column_month_first([r[c_idx] if c_idx < len(r) else "" for r in body], header[c_idx])

            fn_idx = next((i for i, h in enumerate(header) if h.strip().casefold() in ("firstname", "first_name", "fname")), None)
            ln_idx = next((i for i, h in enumerate(header) if h.strip().casefold() in ("lastname", "last_name", "lname")), None)
            rel_type_idx = next((i for i, h in enumerate(header) if h.strip().casefold() in ("relation", "rel_type")), None)

            for r_idx, row in enumerate(body):
                ev = RecordEvidence(
                    rel=rel,
                    loc=f"row:{r_idx}",
                    content_hash=_record_hash(row),
                    hospital=base_hosp,
                )
                if fn_idx is not None and ln_idx is not None:
                    fn = row[fn_idx].strip() if fn_idx < len(row) else ""
                    ln = row[ln_idx].strip() if ln_idx < len(row) else ""
                    if f"{fn} {ln}".strip():
                        nf = make_name_form(f"{fn} {ln}".strip())
                        if nf:
                            ev.names.append(nf)

                for c_idx, h in enumerate(header):
                    if c_idx >= len(row):
                        continue
                    val = row[c_idx].strip()
                    if not val:
                        continue
                    hl = h.casefold().strip()
                    ct = col_types[c_idx]

                    if ct == "PATIENT_NAME" and c_idx not in (fn_idx, ln_idx):
                        nf = make_name_form(val)
                        if nf:
                            ev.names.append(nf)
                    elif ct == "RELATIVE_NAME":
                        rel_prefix = row[rel_type_idx].strip() if (rel_type_idx is not None and rel_type_idx < len(row)) else ""
                        _add_relative(ev, rel_prefix, val)
                    elif ct == "MRN":
                        _add_mrn(ev, val)
                    elif ct == "PHONE":
                        digits = re.sub(r"\D", "", val)[-10:]
                        if len(digits) == 10:
                            ev.phones.add(digits)
                    elif ct == "AADHAAR":
                        nv = build_normalized_view(val).clean
                        m_mask = _RE_AADHAAR_MASK.search(nv)
                        if m_mask:
                            ev.aadhaar_last4.add(m_mask.group(1))
                        else:
                            digits = re.sub(r"\D", "", nv)
                            if len(digits) == 12:
                                ev.aadhaars.add(digits)
                                ev.aadhaar_last4.add(digits[-4:])
                                ev.has_strong_id = True
                    elif ct == "ABHA_NUMBER":
                        digits = re.sub(r"\D", "", val)
                        if len(digits) == 14:
                            ev.abhas.add(digits)
                            ev.has_strong_id = True
                    elif ct == "ABHA_ADDRESS":
                        ev.abhas.add(val.casefold())
                        ev.has_strong_id = True
                    elif ct == "PAN":
                        ev.pans.add(val.upper())
                        ev.has_strong_id = True
                    elif ct == "EMAIL":
                        ev.emails.add(val.casefold())
                        ev.has_strong_id = True
                    elif ct == "INSURANCE_ID":
                        ev.insurances.add(val.upper())
                        ev.has_strong_id = True
                    elif ct == "DOB":
                        d = _parse_date_str(val, month_first_hint=date_month_first.get(c_idx, False))
                        if d:
                            ev.dob = d
                    elif ct == "VILLAGE":
                        _add_village(ev, val)
                    elif ct == "PIN":
                        m_p = re.search(r"([1-8]\d{5})", val)
                        if m_p:
                            ev.pins.add(m_p.group(1))
                    elif ct == "ADDRESS":
                        _extract_address_clues(ev, val)
                    elif hl in ("gender", "sex", "ling"):
                        ev.sex = _norm_sex(val) or ev.sex
                    elif hl in ("age", "umar", "umar_yrs", "age_val"):
                        m_a = re.search(r"(\d{1,3})", val)
                        if m_a:
                            ev.age = int(m_a.group(1))
                    elif hl == "age_sex":
                        m_as = re.match(r"(\d{1,3})\s*/\s*([MF])", val, re.I)
                        if m_as:
                            ev.age = int(m_as.group(1))
                            ev.sex = _norm_sex(m_as.group(2))
                    elif any(w in hl for w in ("date", "collected", "admission", "registered")):
                        rd = _parse_date_str(val, month_first_hint=date_month_first.get(c_idx, False))
                        if rd and not ev.record_date:
                            ev.record_date = rd
                            ev.record_year = int(rd[:4])
                    elif hl in ("notes", "remarks", "comment"):
                        _extract_free_text_clues(ev, val)

                if ev.sex is None:
                    if any(r.relation == "S/o" for r in ev.relatives):
                        ev.sex = "M"
                    elif any(r.relation in ("D/o", "W/o") for r in ev.relatives):
                        ev.sex = "F"
                if ev.names or ev.has_strong_id:
                    evidences.append(ev)

        elif fmt == "json" and isinstance(payload, dict):
            file_hosp = base_hosp
            if isinstance(payload.get("_export_meta"), dict) and isinstance(payload["_export_meta"].get("hospital"), str):
                file_hosp = _canon_hospital(payload["_export_meta"]["hospital"])
            elif isinstance(payload.get("facility"), dict) and isinstance(payload["facility"].get("name"), str):
                file_hosp = _canon_hospital(payload["facility"]["name"])
            elif isinstance(payload.get("lab"), str):
                file_hosp = file_hosp or _canon_hospital(payload["lab"])

            if _is_fhir_bundle(payload):
                ev = RecordEvidence(rel=rel, loc="doc", content_hash=_record_hash(payload), hospital=file_hosp)
                for ent in payload.get("entry", []):
                    res = ent.get("resource", {}) if isinstance(ent, dict) else {}
                    rt = res.get("resourceType")
                    if rt == "Organization" and isinstance(res.get("name"), str):
                        ev.hospital = _canon_hospital(res["name"])
                    elif rt == "Patient":
                        for nm in res.get("name", []):
                            if isinstance(nm, dict) and isinstance(nm.get("text"), str):
                                nf = make_name_form(nm["text"])
                                if nf:
                                    ev.names.append(nf)
                        ev.sex = _norm_sex(res.get("gender")) or ev.sex
                        ev.dob = _parse_date_str(res.get("birthDate")) or ev.dob
                        for ident in res.get("identifier", []):
                            if isinstance(ident, dict) and isinstance(ident.get("value"), str):
                                val = ident["value"]
                                if "healthid" in str(ident.get("system", "")).casefold() or re.fullmatch(r"\d{2}-\d{4}-\d{4}-\d{4}", val):
                                    ev.abhas.add(re.sub(r"\D", "", val))
                                    ev.has_strong_id = True
                                else:
                                    _add_mrn(ev, val)
                        for tc in res.get("telecom", []):
                            if isinstance(tc, dict) and isinstance(tc.get("value"), str):
                                v = tc["value"]
                                if "@" in v:
                                    ev.emails.add(v.casefold())
                                    ev.has_strong_id = True
                                else:
                                    d10 = re.sub(r"\D", "", v)[-10:]
                                    if len(d10) == 10:
                                        ev.phones.add(d10)
                        for addr in res.get("address", []):
                            if isinstance(addr, dict):
                                for ln in addr.get("line", []):
                                    _extract_address_clues(ev, str(ln))
                                if isinstance(addr.get("city"), str):
                                    _add_village(ev, addr["city"])
                                if isinstance(addr.get("postalCode"), str):
                                    ev.pins.add(addr["postalCode"].strip())
                        for ct in res.get("contact", []):
                            if isinstance(ct, dict) and isinstance(ct.get("name"), dict) and isinstance(ct["name"].get("text"), str):
                                rel_txt = ""
                                for ritem in ct.get("relationship", []):
                                    if isinstance(ritem, dict) and isinstance(ritem.get("text"), str):
                                        rt_low = ritem["text"].casefold()
                                        rel_txt = "W/o" if "husband" in rt_low else ("S/o" if "father" in rt_low else "")
                                _add_relative(ev, rel_txt, ct["name"]["text"])
                    elif rt == "Encounter":
                        for ident in res.get("identifier", []):
                            if isinstance(ident, dict) and isinstance(ident.get("value"), str):
                                _add_mrn(ev, ident["value"])
                if ev.names or ev.has_strong_id:
                    evidences.append(ev)
            else:
                units = get_json_record_units(payload)
                for loc, sub in units:
                    if not isinstance(sub, dict):
                        continue
                    ev = RecordEvidence(rel=rel, loc=loc, content_hash=_record_hash(sub), hospital=file_hosp)
                    _extract_dict_evidence(ev, sub)
                    if ev.names or ev.has_strong_id:
                        evidences.append(ev)

        elif fmt == "txt" and isinstance(payload, str):
            ev = RecordEvidence(rel=rel, loc="doc", content_hash=_record_hash(payload), hospital=base_hosp)
            month_first = infer_month_first([payload], header_or_rel_hint=rel)
            _extract_free_text_clues(ev, payload, month_first_hint=month_first)
            if ev.names:
                ev.is_fragmentary = all(n.is_fragment_first_only for n in ev.names) and not ev.has_strong_id
            if ev.names or ev.has_strong_id:
                evidences.append(ev)

    return evidences


def _extract_dict_evidence(ev: RecordEvidence, obj: dict) -> None:
    """Iteratively extract evidence from a JSON record dict."""
    stack: list[tuple[dict, str]] = [(obj, "")]
    while stack:
        node, pk = stack.pop()
        if ("first" in node and "last" in node) or (pk == "name" and "first" in node):
            fn = str(node.get("first") or "").strip()
            ln = str(node.get("last") or "").strip()
            loc_nm = str(node.get("local") or "").strip()
            if f"{fn} {ln}".strip():
                nf = make_name_form(f"{fn} {ln}".strip())
                if nf:
                    ev.names.append(nf)
            if loc_nm:
                nf = make_name_form(loc_nm)
                if nf:
                    ev.names.append(nf)

        for k, v in node.items():
            kl = str(k).casefold()
            if isinstance(v, dict):
                stack.append((v, kl))
                continue
            if isinstance(v, list):
                for it in v:
                    if isinstance(it, dict):
                        stack.append((it, kl))
                continue
            if v is None:
                continue

            if kl in ("doctor", "referred_by", "ref_doctor", "consultant", "doc_consultant", "reviewed_by"):
                continue
            if isinstance(v, int) and not isinstance(v, bool):
                if 100_000_000_000 <= v <= 999_999_999_999:
                    s12 = str(v)
                    ev.aadhaars.add(s12)
                    ev.aadhaar_last4.add(s12[-4:])
                    ev.has_strong_id = True
                elif 10_000_000_000_000 <= v <= 99_999_999_999_999:
                    ev.abhas.add(str(v))
                    ev.has_strong_id = True
                elif 6_000_000_000 <= v <= 9_999_999_999:
                    ev.phones.add(str(v))
                elif kl in ("age", "umar"):
                    ev.age = v
                continue

            if not isinstance(v, str) or not v.strip():
                continue
            vs = v.strip()

            if kl in ("uhid", "patient_ref", "hospital_uhid", "mrn", "pt_uid", "record_code"):
                _add_mrn(ev, vs)
            elif kl in ("patient_name", "beneficiary_full_name", "mariz_naam", "patient") or (kl == "name" and pk in ("", "patient", "raised_by")):
                nf = make_name_form(vs)
                if nf:
                    ev.names.append(nf)
            elif kl == "name" and pk == "guardian":
                rel_t = str(node.get("relation") or "")
                _add_relative(ev, rel_t, vs)
            elif kl in ("mobile", "phone", "callback", "mob_num", "contact_phone"):
                d10 = re.sub(r"\D", "", vs)[-10:]
                if len(d10) == 10:
                    ev.phones.add(d10)
            elif kl in ("email", "mail_id"):
                ev.emails.add(vs.casefold())
                ev.has_strong_id = True
            elif kl in ("dob", "birth_dt", "janm_dinank"):
                d = _parse_date_str(vs)
                if d:
                    ev.dob = d
            elif kl in ("gender", "sex", "ling"):
                ev.sex = _norm_sex(vs) or ev.sex
            elif kl == "age_sex":
                m_as = re.match(r"(\d{1,3})\s*/\s*([MF])", vs, re.I)
                if m_as:
                    ev.age = int(m_as.group(1))
                    ev.sex = _norm_sex(m_as.group(2))
            elif kl in ("village", "city") and pk in ("", "address"):
                _add_village(ev, vs)
            elif kl in ("pin", "pin_zip", "dak_code"):
                m_p = re.search(r"([1-8]\d{5})", vs)
                if m_p:
                    ev.pins.add(m_p.group(1))
            elif kl in ("line", "address_note") or "address" in kl:
                _extract_address_clues(ev, vs)
            elif "aadhar" in kl or "aadhaar" in kl or "adhaar" in kl:
                digits = re.sub(r"\D", "", vs)
                if len(digits) == 12:
                    ev.aadhaars.add(digits)
                    ev.aadhaar_last4.add(digits[-4:])
                    ev.has_strong_id = True
            elif "abha_addr" in kl or "abdm_addr" in kl:
                ev.abhas.add(vs.casefold())
                ev.has_strong_id = True
            elif "abha" in kl or "abdm" in kl:
                digits = re.sub(r"\D", "", vs)
                if len(digits) == 14:
                    ev.abhas.add(digits)
                    ev.has_strong_id = True
            elif "pan" in kl and re.fullmatch(r"[A-Z]{5}\d{4}[A-Z]", vs.upper()):
                ev.pans.add(vs.upper())
                ev.has_strong_id = True
            elif kl in ("visit_date", "collected", "date", "registered_on") and not ev.record_date:
                rd = _parse_date_str(vs)
                if rd:
                    ev.record_date = rd
                    ev.record_year = int(rd[:4])
            elif kl in ("notes", "note", "remarks"):
                _extract_free_text_clues(ev, vs)

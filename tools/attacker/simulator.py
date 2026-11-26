#!/usr/bin/env python3
"""Offline local attacker simulator (A1-A7 per ATTACKS.md). Dev-only, never imported by main.py."""
from __future__ import annotations

import argparse
import base64
import binascii
import csv
import datetime as dt
import hashlib
import io
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from raksha.detect.normalize import phonetic_key

_DEVA_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
_B64_RE = re.compile(r"[A-Za-z0-9+/]{24,}={0,2}")
_PSN_RE = re.compile(r"\bPSN-[0-9A-HJKMNP-TV-Z]{16}\b")

_RE_PHONE = re.compile(r"(?<!\d)(?:\+?91[\s.\-]?|0)?([6-9]\d{4}[\s.\-]?\d{5}|[6-9]\d{2}[\s.\-]?\d{3}[\s.\-]?\d{4})(?!\d)")
_RE_PHONE_MASK = re.compile(r"(?<!\d)([6-9]\d)[xX*]{4,6}(\d{2,4})(?!\d)")
_RE_EMAIL = re.compile(r"(?<![\w.+-])([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})(?!\w)")
_RE_LAST4 = re.compile(r"(?i)[xX*]{4}[\s\-]?[xX*]{4}[\s\-]?(\d{4})\b")
_RE_AADHAAR = re.compile(r"(?<!\d)\d{4}[\s\-]?\d{4}[\s\-]?\d{4}(?!\d)")
_RE_PIN = re.compile(r"(?<!\d)([1-8]\d{5})(?!\d)")
_RE_PIN_MASK = re.compile(r"(?<!\d)([1-8]\d{2})\*{3}(?!\d)")
_RE_DATE_ISO = re.compile(r"\b(19\d{2}|20\d{2})-(\d{2})-(\d{2})(?:T\d{2}:\d{2}[^\s\"']*)?\b")
_RE_DATE_DMY = re.compile(r"(?<!\d)(\d{1,2})[./\-](\d{1,2})[./\-](19\d{2}|20\d{2})(?!\d)")
_RE_DATE_SHORT = re.compile(r"(?<!\d)(\d{1,2})/(\d{1,2})/(\d{2})(?!\d)")
_RE_HOUSE = re.compile(r"(?i)(?:\bH\.?\s*No\.?|\bHouse(?:\s*No\.?)?|\bमकान\s*नं\.?)\s*[:\-]?\s*(\d{1,4})\b")

# Strict age patterns matching ATTACKS.md self-checks (skip 'x 12 yrs', '70-140', '8 yrs back')
_RE_AGE_BAND = re.compile(
    r"(?i)(?:\b(?:age(?:/sex)?|आयु)\s*[:\-]\s*(\d{1,3})\s*-\s*(\d{1,3})(?:\s*(?:Yrs?|वर्ष|Y))?(?:\s*/\s*([MF]|पुरुष|महिला))?)"
    r"|\b(\d{1,3})\s*-\s*(\d{1,3})\s*(?:Yrs?|वर्ष)(?:\s*/\s*([MF]|पुरुष|महिला))?"
    r"|\b(\d{1,3})\s*-\s*(\d{1,3})\s*/\s*([MF])\b"
)
_RE_AGE_PLUS = re.compile(r"(?i)(?:(?:\bAge(?:/Sex)?|\bआयु)\s*[:\-]\s*)?\b(80|85|90)\+\s*(?:Yrs?|वर्ष|Y)?(?:\s*/\s*([MF]))?")
_RE_AGE_LABELED = re.compile(r"(?i)(?:\bAge(?:/Sex)?|\bआयु)\s*[:\-]\s*(\d{1,3})(?:\s*(?:Yrs?|वर्ष|Y))?(?:\s*/\s*([MF]|पुरुष|महिला))?")
_RE_AGE_COMPACT = re.compile(r"(?i)(?<!-)\b(\d{1,3})\s*(?:Yrs?\s*/\s*([MF])|/\s*([MF])\b|Y\s*/\s*([MF])\b|वर्ष(?:\s*/\s*(पुरुष|महिला))?)")
_RE_AGE_TEXT_YRS = re.compile(r"(?i)(?<!x\s)(?<!back\s)(?<!-)\b(\d{1,3})\s+yrs?\b(?!\s*(?:back|ago))")


def _norm(s: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", s).translate(_DEVA_DIGITS).casefold().split())


def _toks(s: str) -> list[str]:
    pk = phonetic_key(s)
    return pk.split() if pk else re.findall(r"[a-z0-9]+", _norm(s))


def _norm_gender(s: str | None) -> str | None:
    if not s:
        return None
    low = _norm(s)
    if low in ("m", "male", "पुरुष") or "s/o" in low:
        return "M"
    if low in ("f", "female", "महिला") or "w/o" in low or "d/o" in low:
        return "F"
    return None


def _parse_dates(text: str, month_first_hint: bool = False) -> list[dt.date]:
    out: list[dt.date] = []
    for m in _RE_DATE_ISO.finditer(text):
        try:
            out.append(dt.date(int(m.group(1)), int(m.group(2)), int(m.group(3))))
        except ValueError:
            pass
    for m in _RE_DATE_DMY.finditer(text):
        a, b, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        d_first = (y, b, a) if (b <= 12 and a <= 31) else None
        m_first = (y, a, b) if (a <= 12 and b <= 31) else None
        order = (m_first, d_first) if month_first_hint else (d_first, m_first)
        for cand in order:
            if cand:
                try:
                    out.append(dt.date(*cand))
                    break
                except ValueError:
                    continue
    for m in _RE_DATE_SHORT.finditer(text):
        d, mo, y2 = int(m.group(1)), int(m.group(2)), int(m.group(3))
        try:
            out.append(dt.date(2000 + y2, mo, d))
        except ValueError:
            pass
    return out


def _read_text_with_b64(p: Path) -> str:
    raw = p.read_bytes()[: 50 * 1024 * 1024]
    for enc in ("utf-8-sig", "utf-16", "cp1252", "latin1"):
        try:
            t = raw.decode(enc)
            if enc == "cp1252" and "\x00" in t:
                continue
            break
        except UnicodeError:
            continue
    else:
        t = raw.decode("utf-8", errors="replace")

    extra = []
    for m in _B64_RE.finditer(t):
        tok = m.group(0)
        try:
            dec = base64.b64decode(tok + "=" * (-len(tok) % 4)).decode("utf-8", errors="ignore")
            if dec:
                extra.append(dec)
        except (binascii.Error, ValueError):
            pass
    return (t + "\n" + "\n".join(extra)) if extra else t


@dataclass
class RegistryPerson:
    rid: str
    name: str
    name_tokens: tuple[str, ...]
    given_tok: str
    surname_tok: str
    rel_tokens: tuple[str, ...]
    gender: str
    age: int
    house_no: str
    village_key: str
    tehsil_key: str
    district_key: str
    pin: str
    pin_prefix: str
    phone: str
    email: str
    aadhaar_last4: str


@dataclass
class AttackedEvidence:
    rel: str
    loc: str
    true_patient: str | None
    psn: str | None
    raw_text: str
    ages: list[tuple[int, int]] = field(default_factory=list)
    birth_years: set[int] = field(default_factory=set)
    gender: str | None = None
    districts: set[str] = field(default_factory=set)
    tehsils: set[str] = field(default_factory=set)
    villages: set[str] = field(default_factory=set)
    pins: set[str] = field(default_factory=set)
    pin_prefixes: set[str] = field(default_factory=set)
    phones: set[str] = field(default_factory=set)
    phone_masks: set[tuple[str, str]] = field(default_factory=set)
    emails: set[str] = field(default_factory=set)
    last4s: set[str] = field(default_factory=set)
    house_nos: set[str] = field(default_factory=set)
    hospitals: set[str] = field(default_factory=set)
    conditions: set[str] = field(default_factory=set)
    dates: list[dt.date] = field(default_factory=list)
    stay_days: set[int] = field(default_factory=set)
    refs: set[str] = field(default_factory=set)
    tokens: set[str] = field(default_factory=set)


class Gazetteer:
    """Builds transliterated district/tehsil/village/hospital/condition indices from public attacker registry."""

    def __init__(self, registry_rows: list[RegistryPerson]):
        self.districts: dict[str, set[str]] = {}
        self.tehsils: dict[str, set[str]] = {}
        self.villages: dict[str, set[str]] = {}
        for r in registry_rows:
            if r.district_key:
                self.districts.setdefault(r.district_key, set()).add(" ".join(_toks(r.district_key)))
            if r.tehsil_key:
                self.tehsils.setdefault(r.tehsil_key, set()).add(" ".join(_toks(r.tehsil_key)))
            if r.village_key:
                self.villages.setdefault(r.village_key, set()).add(" ".join(_toks(r.village_key)))

        self.district_phon = {ph: dk for dk, phs in self.districts.items() for ph in phs if len(ph) >= 3}
        self.tehsil_phon = {ph: tk for tk, phs in self.tehsils.items() for ph in phs if len(ph) >= 3}
        self.village_phon = {ph: vk for vk, phs in self.villages.items() for ph in phs if len(ph) >= 3}


def load_registry(path: Path) -> list[RegistryPerson]:
    people: list[RegistryPerson] = []
    with path.open("r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            ntoks = tuple(_toks(row.get("name", "")))
            rtoks = tuple(_toks(row.get("relative_name", "")))
            pin = re.sub(r"\D", "", row.get("pin", ""))
            people.append(
                RegistryPerson(
                    rid=row["rid"],
                    name=row.get("name", ""),
                    name_tokens=ntoks,
                    given_tok=ntoks[0] if ntoks else "",
                    surname_tok=ntoks[-1] if ntoks else "",
                    rel_tokens=rtoks,
                    gender=_norm_gender(row.get("gender")) or "",
                    age=int(row["age"]),
                    house_no=str(row.get("house_no", "")).strip(),
                    village_key=_norm(row.get("village", "")),
                    tehsil_key=_norm(row.get("tehsil", "")),
                    district_key=_norm(row.get("district", "")),
                    pin=pin,
                    pin_prefix=pin[:3] if len(pin) == 6 else "",
                    phone=re.sub(r"\D", "", row.get("phone", ""))[-10:],
                    email=_norm(row.get("email", "")),
                    aadhaar_last4=re.sub(r"\D", "", row.get("aadhaar_last4", "")),
                )
            )
    return people


def map_truth_patients_to_registry(truth: dict, input_dir: Path, registry: list[RegistryPerson]) -> dict[str, RegistryPerson]:
    """Match each truth patient PTxxxx to its true row in registry.csv via identity attributes."""
    by_name: dict[tuple[str, ...], list[RegistryPerson]] = defaultdict(list)
    for r in registry:
        by_name[r.name_tokens].append(r)

    # Gather per-PT hints from truth entities + fields
    pt_names: dict[str, set[tuple[str, ...]]] = defaultdict(set)
    pt_rel: dict[str, set[tuple[str, ...]]] = defaultdict(set)
    pt_ages: dict[str, list[int]] = defaultdict(list)
    pt_villages: dict[str, set[str]] = defaultdict(set)
    pt_phones: dict[str, set[str]] = defaultdict(set)

    for rel, finfo in truth["files"].items():
        for _loc, u in finfo["units"].items():
            pt = u.get("patient")
            if not pt:
                continue
            for e in u.get("entities", []):
                if e.get("type") == "PATIENT_NAME":
                    t = tuple(_toks(e.get("surface", "")))
                    if len(t) >= 2:
                        pt_names[pt].add(t)
                elif e.get("type") == "RELATIVE_NAME":
                    t = tuple(_toks(e.get("surface", "")))
                    if len(t) >= 1:
                        pt_rel[pt].add(t)
                elif e.get("type") == "VILLAGE":
                    pt_villages[pt].add(" ".join(_toks(e.get("surface", ""))))
                elif e.get("type") == "PHONE":
                    d = re.sub(r"\D", "", _norm(e.get("surface", "")))[-10:]
                    if len(d) == 10:
                        pt_phones[pt].add(d)
            for fld in u.get("fields", []):
                if fld.get("kind") == "age" and isinstance(fld.get("true"), int):
                    pt_ages[pt].append(fld["true"])

    pt_to_reg: dict[str, RegistryPerson] = {}
    for pt, nt_set in pt_names.items():
        cands: list[tuple[int, RegistryPerson]] = []
        for nt in nt_set:
            for r in by_name.get(nt, []):
                score = 10
                if pt_phones[pt] and r.phone in pt_phones[pt]:
                    score += 30
                if pt_rel[pt] and any(set(rt) & set(r.rel_tokens) for rt in pt_rel[pt]):
                    score += 20
                if pt_villages[pt] and " ".join(_toks(r.village_key)) in pt_villages[pt]:
                    score += 15
                if pt_ages[pt] and any(abs(a - r.age) <= 2 for a in pt_ages[pt]):
                    score += 10
                cands.append((score, r))
        if cands:
            cands.sort(key=lambda x: -x[0])
            pt_to_reg[pt] = cands[0][1]
    return pt_to_reg


def _extract_ages_and_gender(text: str, ev: AttackedEvidence) -> None:
    # Remove known doctor lines before scanning text tokens
    n = _norm(text)

    # First extract explicit age ranges ("50-54 Yrs/F", "age: 50-54", "90+")
    for m in _RE_AGE_BAND.finditer(text):
        lo_s = m.group(1) or m.group(4) or m.group(7)
        hi_s = m.group(2) or m.group(5) or m.group(8)
        g_s = m.group(3) or m.group(6) or m.group(9)
        lo, hi = int(lo_s), int(hi_s)
        span_ctx = text[max(0, m.start() - 18) : m.end() + 18].casefold()
        if any(w in span_ctx for w in ("range", "ref", "bp", "pulse", "hemoglobin", "haemoglobin", "platelet", "wbc", "creatinine", "rbs", "glucose", "sodium", "tsh")):
            continue
        if 0 <= lo <= hi <= 115 and 1 <= (hi - lo) <= 60:
            ev.ages.append((lo, hi))
            g = _norm_gender(g_s)
            if g and not ev.gender:
                ev.gender = g

    for m in _RE_AGE_PLUS.finditer(text):
        lo = int(m.group(1))
        ev.ages.append((lo, 110))
        g = _norm_gender(m.group(2))
        if g and not ev.gender:
            ev.gender = g

    if not ev.ages:
        for m in _RE_AGE_LABELED.finditer(text):
            a = int(m.group(1))
            if 0 <= a <= 110:
                ev.ages.append((a, a))
            g = _norm_gender(m.group(2))
            if g and not ev.gender:
                ev.gender = g
        for m in _RE_AGE_COMPACT.finditer(text):
            a = int(m.group(1))
            if 0 <= a <= 110:
                ev.ages.append((a, a))
            g = _norm_gender(m.group(2) or m.group(3) or m.group(4) or m.group(5))
            if g and not ev.gender:
                ev.gender = g
        for m in _RE_AGE_TEXT_YRS.finditer(text):
            before = text[max(0, m.start() - 12) : m.start()].casefold()
            if re.search(r"\b(?:back|ago|since|for|x|history)\s*$", before):
                continue
            a = int(m.group(1))
            if 0 <= a <= 110:
                ev.ages.append((a, a))

    if not ev.gender:
        if re.search(r"(?i)\b(?:sex|gender|लिंग)\s*[:\-]\s*(?:female|f|महिला)\b|\b(?:w/o|d/o|behenji|didi|श्रीमती)\b|\bमहिला\b", n):
            ev.gender = "F"
        elif re.search(r"(?i)\b(?:sex|gender|लिंग)\s*[:\-]\s*(?:male|m|पुरुष)\b|\b(?:s/o|sahab|bhaiya)\b|\bपुरुष\b", n):
            ev.gender = "M"


def _extract_geo_and_ids(text: str, gaz: Gazetteer, ev: AttackedEvidence, month_first: bool = False) -> None:
    nt = _norm(text)
    ph_str = " " + " ".join(_toks(text)) + " "

    for dk_ph, dk in gaz.district_phon.items():
        if f" {dk_ph} " in ph_str:
            ev.districts.add(dk)
    for tk_ph, tk in gaz.tehsil_phon.items():
        if f" {tk_ph} " in ph_str:
            ev.tehsils.add(tk)
    for vk_ph, vk in gaz.village_phon.items():
        if f" {vk_ph} " in ph_str:
            ev.villages.add(vk)

    for m in _RE_PIN.finditer(nt):
        ev.pins.add(m.group(1))
        ev.pin_prefixes.add(m.group(1)[:3])
    for m in _RE_PIN_MASK.finditer(nt):
        ev.pin_prefixes.add(m.group(1))

    # Strip doctor names from token bag so Dr. Ramesh doesn't vote for Ramesh patient
    stripped = re.sub(r"(?im)(?:dr\.?|डॉ\.?|consultant|prescriber|referred\s*by|author)\s*[:\-]?\s*[^\n,;)]+", " ", nt)
    ev.tokens.update(_toks(stripped))

    for m in _RE_PHONE.finditer(nt):
        d = re.sub(r"\D", "", m.group(1))[-10:]
        if len(d) == 10:
            ev.phones.add(d)
    for m in _RE_PHONE_MASK.finditer(nt):
        ev.phone_masks.add((m.group(1), m.group(2)))
    for m in _RE_EMAIL.finditer(nt):
        em = m.group(1)
        if not em.endswith((".example", "@abdm", "@ndhm", "@sbx")):
            ev.emails.add(em)
    for m in _RE_LAST4.finditer(nt):
        ev.last4s.add(m.group(1))
    for m in _RE_AADHAAR.finditer(nt):
        d = re.sub(r"\D", "", m.group(0))
        if len(d) == 12:
            ev.last4s.add(d[-4:])
    for m in _RE_HOUSE.finditer(nt):
        ev.house_nos.add(m.group(1))

    # Hospital extraction
    low = nt.casefold()
    for h_key, h_toks in (
        ("arogya seva sadan", ("arogya seva sadan", "arogya")),
        ("jeevan jyoti charitable hospital", ("jeevan jyoti",)),
        ("sanjeevani multispeciality hospital", ("sanjeevani",)),
        ("district hospital hardoi", ("district hospital hardoi", "dhhardoi")),
    ):
        if any(ht in low for ht in h_toks):
            ev.hospitals.add(h_key)

    ev.dates.extend(_parse_dates(nt, month_first_hint=month_first))


def _extract_conditions(text: str, obj: dict | None, ev: AttackedEvidence) -> None:
    if isinstance(obj, dict):
        # Structured JSON diagnosis / procedure / FHIR Condition
        stack = [obj]
        while stack:
            node = stack.pop()
            for k, v in node.items():
                kl = str(k).casefold()
                if kl == "diagnosis" and isinstance(v, list):
                    for d in v:
                        if isinstance(d, dict) and isinstance(d.get("display"), str):
                            ev.conditions.add(_norm(d["display"]))
                elif kl in ("procedure", "diagnosis") and isinstance(v, str) and v.strip():
                    ev.conditions.add(_norm(v))
                elif node.get("resourceType") == "Condition" and kl == "code" and isinstance(v, dict):
                    if isinstance(v.get("text"), str):
                        ev.conditions.add(_norm(v["text"]))
                elif isinstance(v, dict):
                    stack.append(v)
                elif isinstance(v, list):
                    for it in v:
                        if isinstance(it, dict):
                            stack.append(it)
    for m in re.finditer(r"(?im)(?:Diagnosis|निदान|Dx|Procedure)\s*[:\-]\s*([^\n]+)", text):
        for seg in re.split(r"[;|]", m.group(1)):
            s = re.sub(r"\([^)]*\)", "", seg).strip()
            s = re.sub(r"\b[A-Z]\d{2}(?:\.\d+)?\b", "", s).strip()
            if len(s) >= 4:
                ev.conditions.add(_norm(s))


def _score_candidates(ev: AttackedEvidence, registry: list[RegistryPerson]) -> list[tuple[float, RegistryPerson]]:
    """ATTACKS.md A2/A3/A4 filter-skip cascade + candidate weighting."""
    cands = list(registry)

    def _apply_filter(pred) -> None:
        nonlocal cands
        sub = [r for r in cands if pred(r)]
        if sub:
            cands = sub

    if ev.phones:
        _apply_filter(lambda r: r.phone in ev.phones)
    if ev.emails:
        _apply_filter(lambda r: r.email in ev.emails)
    if ev.last4s:
        _apply_filter(lambda r: r.aadhaar_last4 in ev.last4s)
    if ev.phone_masks:
        _apply_filter(lambda r: any(r.phone.startswith(p) and r.phone.endswith(s) for p, s in ev.phone_masks))
    if ev.districts:
        _apply_filter(lambda r: r.district_key in ev.districts)
    if ev.gender:
        _apply_filter(lambda r: r.gender == ev.gender)
    if ev.ages:
        _apply_filter(lambda r: any((lo - 2) <= r.age <= (hi + 2) for lo, hi in ev.ages))
    if ev.birth_years:
        _apply_filter(lambda r: any(abs((2026 - by) - r.age) <= 2 for by in ev.birth_years))
    if ev.tehsils:
        _apply_filter(lambda r: r.tehsil_key in ev.tehsils)
    if ev.villages:
        _apply_filter(lambda r: r.village_key in ev.villages)
    if ev.pins:
        _apply_filter(lambda r: r.pin in ev.pins)
    elif ev.pin_prefixes:
        _apply_filter(lambda r: r.pin_prefix in ev.pin_prefixes)

    scored: list[tuple[float, RegistryPerson]] = []
    for r in cands:
        w = 1.0
        if len(r.name_tokens) >= 2 and all(t in ev.tokens for t in r.name_tokens):
            w += 60.0
        elif r.given_tok and r.given_tok in ev.tokens:
            w += 8.0
        elif r.surname_tok and r.surname_tok in ev.tokens:
            w += 3.0
        if r.rel_tokens and all(t in ev.tokens for t in r.rel_tokens[:2]):
            w += 8.0
        if r.house_no and r.house_no in ev.house_nos:
            w += 2.0
        scored.append((w, r))
    return scored


def _extract_release_records(
    root_dir: Path,
    truth: dict,
    registry: list[RegistryPerson],
    *,
    is_raw: bool,
    records_jsonl: Path | None = None,
    report_json: Path | None = None,
) -> list[AttackedEvidence]:
    gaz = Gazetteer(registry)
    renamed: dict[str, str] = {}
    if report_json and report_json.is_file():
        rpt = json.loads(report_json.read_text(encoding="utf-8"))
        renamed = {str(e.get("input_sha256")): str(e.get("output")) for e in rpt.get("files") or []}

    rec_psns: dict[tuple[str, str], str] = {}
    if records_jsonl and records_jsonl.is_file():
        for line in records_jsonl.read_text(encoding="utf-8").splitlines():
            if line.strip():
                obj = json.loads(line)
                rel_s = str(obj["file"]).replace("sanitized/", "", 1)
                if obj.get("patient"):
                    rec_psns[(rel_s, str(obj["loc"]))] = str(obj["patient"])

    evs: list[AttackedEvidence] = []
    # Cache parsed file views per relative path
    for rel, finfo in sorted(truth["files"].items()):
        out_rel = rel if is_raw else str(renamed.get(finfo["sha256"], rel)).replace("sanitized/", "", 1)
        p = root_dir / out_rel
        if not p.is_file():
            continue
        txt_full = _read_text_with_b64(p)
        month_first = "arogya_legacy" in rel or "rx_arogya" in rel

        csv_rows: list[list[str]] | None = None
        json_obj: object | None = None
        if p.suffix in (".csv", ".tsv"):
            try:
                csv_rows = list(csv.reader(io.StringIO(txt_full, newline=""), delimiter="\t" if p.suffix == ".tsv" else ","))
            except csv.Error:
                csv_rows = None
        elif p.suffix == ".json":
            try:
                json_obj = json.loads(txt_full)
            except ValueError:
                json_obj = None

        for loc, u in finfo["units"].items():
            if loc.startswith("@"):
                continue
            pt = u.get("patient")
            rec_txt = ""
            rec_obj: dict | None = None
            if loc.startswith("row:") and csv_rows and len(csv_rows) > 1:
                ridx = int(loc.split(":")[1]) + 1
                if ridx < len(csv_rows):
                    hdr = csv_rows[0]
                    row = csv_rows[ridx]
                    rec_txt = "\n".join(f"{hdr[i] if i < len(hdr) else ''}: {row[i]}" for i in range(len(row)) if row[i])
            elif loc.startswith("/") and isinstance(json_obj, dict):
                parts = [x.replace("~1", "/").replace("~0", "~") for x in loc[1:].split("/")]
                cur: object = json_obj
                for part in parts:
                    if isinstance(cur, dict) and part in cur:
                        cur = cur[part]
                    elif isinstance(cur, list) and part.isdigit() and int(part) < len(cur):
                        cur = cur[int(part)]
                    else:
                        cur = None
                        break
                if isinstance(cur, dict):
                    rec_obj = cur
                    rec_txt = json.dumps(cur, ensure_ascii=False)
                else:
                    rec_txt = txt_full
            else:
                rec_txt = txt_full
                if isinstance(json_obj, dict):
                    rec_obj = json_obj

            m_psn = _PSN_RE.search(rec_txt)
            # Attacker ONLY sees PSNs physically present in exported content!
            psn = m_psn.group(0) if m_psn else None
            # For records.jsonl sanity, verify non-null claim matches
            if not psn and not is_raw:
                claimed = rec_psns.get((out_rel, loc))
                if claimed and claimed in rec_txt:
                    psn = claimed

            aev = AttackedEvidence(rel=out_rel, loc=loc, true_patient=pt, psn=psn, raw_text=rec_txt)

            # Extract structured JSON age/gender/DOB/refs/intervals directly if dict or FHIR
            if isinstance(rec_obj, dict):
                if rec_obj.get("resourceType") == "Bundle":
                    for ent in rec_obj.get("entry", []):
                        res = ent.get("resource", {}) if isinstance(ent, dict) else {}
                        rt = res.get("resourceType")
                        if rt == "Patient":
                            if isinstance(res.get("birthDate"), str) and res["birthDate"][:4].isdigit():
                                aev.birth_years.add(int(res["birthDate"][:4]))
                            if isinstance(res.get("gender"), str):
                                aev.gender = _norm_gender(res["gender"]) or aev.gender
                            for it in res.get("identifier", []):
                                if isinstance(it, dict) and isinstance(it.get("value"), str) and "[MRN]" not in it["value"] and not _PSN_RE.match(it["value"]):
                                    aev.refs.add(it["value"])
                        elif rt == "Encounter":
                            per = res.get("period", {})
                            if isinstance(per, dict):
                                ds = _parse_dates(str(per.get("start", "")))
                                de = _parse_dates(str(per.get("end", "")))
                                if ds and de:
                                    aev.stay_days.add(abs((de[0] - ds[0]).days))
                else:
                    # Generic dict leaves
                    stack = [rec_obj]
                    while stack:
                        nd = stack.pop()
                        for k, v in nd.items():
                            kl = str(k).casefold()
                            if isinstance(v, dict):
                                stack.append(v)
                            elif isinstance(v, list):
                                for x in v:
                                    if isinstance(x, dict):
                                        stack.append(x)
                            elif v is not None:
                                sv = str(v).strip()
                                if kl in ("age", "umar", "age_band") and sv:
                                    mb = re.fullmatch(r"(\d{1,3})\s*-\s*(\d{1,3})", sv)
                                    mp = re.fullmatch(r"(\d{1,3})\+", sv)
                                    if mb:
                                        aev.ages.append((int(mb.group(1)), int(mb.group(2))))
                                    elif mp:
                                        aev.ages.append((int(mp.group(1)), 110))
                                    elif sv.isdigit() and 0 <= int(sv) <= 110:
                                        aev.ages.append((int(sv), int(sv)))
                                elif kl in ("dob", "janm_tithi") and len(sv) >= 4 and sv[:4].isdigit() and "[DOB]" not in sv:
                                    aev.birth_years.add(int(sv[:4]))
                                elif kl in ("gender", "sex", "ling") and sv:
                                    aev.gender = _norm_gender(sv) or aev.gender
                                elif kl in ("uhid", "patient_ref", "hospital_uhid") and sv and not sv.startswith("[") and not _PSN_RE.match(sv):
                                    aev.refs.add(sv)
                    if "admission_date" in rec_obj and "discharge_date" in rec_obj:
                        da = _parse_dates(str(rec_obj["admission_date"]), month_first)
                        dd = _parse_dates(str(rec_obj["discharge_date"]), month_first)
                        if da and dd:
                            aev.stay_days.add(abs((dd[0] - da[0]).days))
            elif csv_rows and loc.startswith("row:"):
                ridx = int(loc.split(":")[1]) + 1
                if ridx < len(csv_rows):
                    hdr = [h.casefold().strip() for h in csv_rows[0]]
                    row = csv_rows[ridx]
                    adm_d = dis_d = None
                    for ci, col_h in enumerate(hdr):
                        if ci >= len(row):
                            continue
                        val = row[ci].strip()
                        if not val or val.startswith("["):
                            continue
                        if col_h in ("age", "umar"):
                            mb = re.fullmatch(r"(\d{1,3})\s*-\s*(\d{1,3})", val)
                            mp = re.fullmatch(r"(\d{1,3})\+", val)
                            if mb:
                                aev.ages.append((int(mb.group(1)), int(mb.group(2))))
                            elif mp:
                                aev.ages.append((int(mp.group(1)), 110))
                            elif val.isdigit():
                                aev.ages.append((int(val), int(val)))
                        elif col_h in ("dob", "dob (mm/dd/yyyy)", "janm tithi"):
                            ds = _parse_dates(val, month_first)
                            if ds:
                                aev.birth_years.add(ds[0].year)
                        elif col_h == "admission_date":
                            ds = _parse_dates(val, month_first)
                            adm_d = ds[0] if ds else None
                        elif col_h == "discharge_date":
                            ds = _parse_dates(val, month_first)
                            dis_d = ds[0] if ds else None
                        elif col_h in ("uhid", "patientid", "reg no") and not _PSN_RE.match(val):
                            aev.refs.add(val)
                    if adm_d and dis_d:
                        aev.stay_days.add(abs((dis_d - adm_d).days))

            # Text admission / discharge LoS check
            m_adm = re.search(r"(?i)(?:Date\s+of\s+Admission|भर्ती\s+की\s+तारीख)\s*[:\-]\s*([^\n]+)", rec_txt)
            m_dis = re.search(r"(?i)(?:Date\s+of\s+Discharge|छुट्टी\s+की\s+तारीख)\s*[:\-]\s*([^\n]+)", rec_txt)
            if m_adm and m_dis:
                da = _parse_dates(m_adm.group(1), month_first)
                dd = _parse_dates(m_dis.group(1), month_first)
                if da and dd:
                    aev.stay_days.add(abs((dd[0] - da[0]).days))

            _extract_ages_and_gender(rec_txt, aev)
            _extract_geo_and_ids(rec_txt, gaz, aev, month_first=month_first)
            _extract_conditions(rec_txt, rec_obj, aev)
            evs.append(aev)
    return evs


def _merge_profile_evs(grp: list[AttackedEvidence]) -> AttackedEvidence:
    base = AttackedEvidence(
        rel=grp[0].rel,
        loc=",".join(e.loc for e in grp[:3]),
        true_patient=grp[0].true_patient,
        psn=grp[0].psn,
        raw_text="\n".join(e.raw_text for e in grp),
    )
    for e in grp:
        base.ages.extend(e.ages)
        base.birth_years.update(e.birth_years)
        if not base.gender and e.gender:
            base.gender = e.gender
        base.districts.update(e.districts)
        base.tehsils.update(e.tehsils)
        base.villages.update(e.villages)
        base.pins.update(e.pins)
        base.pin_prefixes.update(e.pin_prefixes)
        base.phones.update(e.phones)
        base.phone_masks.update(e.phone_masks)
        base.emails.update(e.emails)
        base.last4s.update(e.last4s)
        base.house_nos.update(e.house_nos)
        base.hospitals.update(e.hospitals)
        base.conditions.update(e.conditions)
        base.dates.extend(e.dates)
        base.stay_days.update(e.stay_days)
        base.tokens.update(e.tokens)
    return base


def _eval_single_release(
    root_dir: Path,
    truth: dict,
    registry: list[RegistryPerson],
    pt_to_reg: dict[str, RegistryPerson],
    facts: list[dict],
    *,
    is_raw: bool,
    records_jsonl: Path | None = None,
    report_json: Path | None = None,
) -> dict:
    evs = _extract_release_records(
        root_dir,
        truth,
        registry,
        is_raw=is_raw,
        records_jsonl=records_jsonl,
        report_json=report_json,
    )

    # A1: direct exposure per patient record (excluding preserve-probes / doctor mentions / clean/)
    exposed_pts: set[str] = set()
    all_pts = {u.get("patient") for f in truth["files"].values() for u in f["units"].values() if u.get("patient")}
    allowed_preserves: Counter[str] = Counter()
    for rel, f in truth["files"].items():
        for _loc, u in f["units"].items():
            for pv in u.get("preserve") or []:
                for tok in _norm(pv.get("text", "")).split():
                    if len(tok) >= 4:
                        allowed_preserves[tok] += 1

    # Exclude clean/ and doctor-prefixed lines when checking direct leaks
    all_out_chunks = []
    out_text_by_rel: dict[str, str] = {}
    if root_dir.is_dir():
        for p in sorted(root_dir.rglob("*")):
            if not p.is_file():
                continue
            r_posix = p.relative_to(root_dir).as_posix()
            if r_posix.startswith("clean/"):
                continue
            cleaned_t = re.sub(r"(?im)(?:dr\.?|डॉ\.?|consultant|prescriber|referred\s*by)\s*[:\-]?\s*[^\n,;)]+", " ", _read_text_with_b64(p))
            nt = _norm(cleaned_t + " " + p.name)
            out_text_by_rel[r_posix] = nt
            all_out_chunks.append(nt)
    all_file_text = "\n".join(all_out_chunks)

    for rel, f in truth["files"].items():
        for _loc, u in f["units"].items():
            pt = u.get("patient")
            if not pt:
                continue
            for ent in u.get("entities", []):
                et = ent.get("type")
                if et not in ("PATIENT_NAME", "PHONE", "AADHAAR", "ABHA_NUMBER", "EMAIL", "PAN", "MRN"):
                    continue
                if et == "PATIENT_NAME" and ent.get("full") is False:
                    continue
                surf = _norm(ent.get("surface", ""))
                if len(surf) >= 5 and surf in all_file_text:
                    exposed_pts.add(pt)
    a1_risk = len(exposed_pts) / max(1, len(all_pts))

    # A2/A3: record linkage probability mass on true rid
    rec_probs: dict[tuple[str, str], float] = {}
    rec_by_key: dict[tuple[str, str], AttackedEvidence] = {}
    for ev in evs:
        if not ev.true_patient or ev.true_patient not in pt_to_reg:
            continue
        true_rid = pt_to_reg[ev.true_patient].rid
        scored = _score_candidates(ev, registry)
        tot = sum(w for w, _ in scored) or 1.0
        p_true = sum(w for w, r in scored if r.rid == true_rid) / tot
        rec_probs[(ev.rel, ev.loc)] = p_true
        rec_by_key[(ev.rel, ev.loc)] = ev
    a3_mean = sum(rec_probs.values()) / max(1, len(rec_probs))

    # A4: profile aggregation (via PSN present in content, or unredacted shared ref)
    profiles: dict[str, list[AttackedEvidence]] = defaultdict(list)
    for ev in evs:
        if ev.psn:
            profiles[f"psn:{ev.psn}"].append(ev)
        elif ev.refs:
            profiles[f"ref:{sorted(ev.refs)[0]}"].append(ev)
        else:
            profiles[f"solo:{ev.rel}#{ev.loc}"].append(ev)

    prof_merged: dict[str, AttackedEvidence] = {}
    prof_probs: dict[str, dict[str, float]] = {}
    pt_max_prob: dict[str, float] = defaultdict(float)
    at_risk_details: list[dict] = []

    for pk, grp in profiles.items():
        m = _merge_profile_evs(grp)
        prof_merged[pk] = m
        scored = _score_candidates(m, registry)
        tot = sum(w for w, _ in scored) or 1.0
        probs_map = {r.rid: w / tot for w, r in scored}
        prof_probs[pk] = probs_map
        top_rid, top_p = max(probs_map.items(), key=lambda x: x[1]) if probs_map else ("", 0.0)

        for ev in grp:
            if not ev.true_patient or ev.true_patient not in pt_to_reg:
                continue
            true_rid = pt_to_reg[ev.true_patient].rid
            comb_p = max(rec_probs.get((ev.rel, ev.loc), 0.0), probs_map.get(true_rid, 0.0))
            if comb_p > pt_max_prob[ev.true_patient]:
                pt_max_prob[ev.true_patient] = comb_p

        if pk.startswith("psn:") and top_p >= 0.35:
            reasons = []
            if m.districts:
                reasons.append(f"district={sorted(m.districts)}")
            if m.tehsils:
                reasons.append(f"tehsil={sorted(m.tehsils)}")
            if m.ages:
                reasons.append(f"ages={m.ages[:3]}")
            if m.conditions:
                reasons.append(f"conds={sorted(m.conditions)[:2]}")
            at_risk_details.append(
                {
                    "profile": pk,
                    "top_prob": round(top_p, 3),
                    "candidates": len(probs_map),
                    "true_pts": sorted({e.true_patient for e in grp if e.true_patient}),
                    "reasons": reasons,
                }
            )

    a4_reid = sum(1 for pt in all_pts if pt_max_prob.get(pt, 0.0) >= 0.5) / max(1, len(all_pts))

    # A5: Targeted attack (global date-shift recovery + condition rarity + LoS + hospital)
    # 1. Global shift recovery from confidence >= 0.9 profiles
    reg_to_pt = {r.rid: pt for pt, r in pt_to_reg.items()}
    facts_by_rid = {ft["rid"]: ft for ft in facts}
    offset_votes: Counter[int] = Counter()
    for pk, m in prof_merged.items():
        probs_map = prof_probs[pk]
        if not probs_map:
            continue
        top_rid, top_p = max(probs_map.items(), key=lambda x: x[1])
        if top_p >= 0.9 and top_rid in facts_by_rid:
            known_ds = [dt.date.fromisoformat(d) for d in facts_by_rid[top_rid].get("visit_dates", [])]
            for kd in known_ds:
                for od in m.dates:
                    diff = (od - kd).days
                    if abs(diff) <= 400:
                        offset_votes[diff] += 1

    recovered_global_shift = 0
    if offset_votes:
        best_off, best_cnt = offset_votes.most_common(1)[0]
        if best_cnt >= 2 and best_off != 0:
            recovered_global_shift = best_off

    cond_profile_counts = Counter()
    for m in prof_merged.values():
        for c in m.conditions:
            cond_profile_counts[c] += 1

    a5_probs: list[float] = []
    decoy_fp = 0
    decoy_cnt = 0
    for ft in facts:
        rid = ft["rid"]
        is_target_patient = rid in reg_to_pt
        true_pt = reg_to_pt.get(rid)
        known_dates = [dt.date.fromisoformat(d) for d in ft.get("visit_dates", [])]
        known_hosp = _norm(ft.get("hospital", ""))
        known_cond = _norm(ft.get("condition_hint", ""))
        known_stay = ft.get("stay_days")

        c_weights: dict[str, float] = {}
        for pk, m in prof_merged.items():
            # Un-shift dates if global shift recovered
            adj_dates = [d - dt.timedelta(days=recovered_global_shift) for d in m.dates]
            fact_score = 0.0
            if known_dates and any(kd == od for kd in known_dates for od in adj_dates):
                fact_score += 5.0
            if known_cond and any(known_cond in c or c in known_cond for c in m.conditions):
                rarity = min((cond_profile_counts.get(c, 99) for c in m.conditions if known_cond in c or c in known_cond), default=99)
                fact_score += 8.0 if rarity <= 3 else 2.0
            if known_stay is not None and m.stay_days and any(abs(sd - int(known_stay)) <= 1 for sd in m.stay_days):
                fact_score += 3.0
            if fact_score <= 0.0:
                continue
            # Check consistency with registry rid
            p_rid = prof_probs[pk].get(rid, 0.0)
            if p_rid <= 0.0:
                continue
            if known_hosp and any(h in known_hosp or known_hosp in h for h in m.hospitals):
                fact_score *= 2.0
            c_weights[pk] = fact_score * p_rid

        tot_w = sum(c_weights.values())
        if is_target_patient:
            if tot_w > 0:
                true_w = sum(w for pk, w in c_weights.items() if any(e.true_patient == true_pt for e in profiles[pk]))
                a5_probs.append(true_w / tot_w)
            else:
                a5_probs.append(0.0)
        else:
            decoy_cnt += 1
            if c_weights and max(c_weights.values()) >= 0.5:
                decoy_fp += 1

    a5_mean = sum(a5_probs) / max(1, len(a5_probs))

    # A6: Pseudonym inversion dictionary attack (skip on raw)
    a6_inv = 0.0
    if not is_raw:
        pt_secrets_dict: dict[str, set[str]] = defaultdict(set)
        for rel, finfo in truth["files"].items():
            for _loc, u in finfo["units"].items():
                pt = u.get("patient")
                if not pt:
                    continue
                for ent in u.get("entities", []):
                    if ent.get("type") in ("PATIENT_NAME", "PHONE", "AADHAAR", "ABHA_NUMBER", "MRN"):
                        s = str(ent.get("surface", "")).strip()
                        if s:
                            pt_secrets_dict[pt].add(s)
                            pt_secrets_dict[pt].add(re.sub(r"\D", "", s))
        psn_pts: dict[str, set[str]] = defaultdict(set)
        for ev in evs:
            if ev.psn and ev.true_patient:
                psn_pts[ev.true_patient].add(ev.psn)
        inv_hits = 0
        for pt, psns in psn_pts.items():
            cands = set()
            for raw_s in pt_secrets_dict.get(pt, set()):
                if not raw_s:
                    continue
                for v in (raw_s, raw_s.lower(), raw_s.upper(), raw_s[::-1]):
                    vb = v.encode("utf-8", errors="ignore")
                    cands.add(hashlib.md5(vb).hexdigest())
                    cands.add(hashlib.sha1(vb).hexdigest())
                    cands.add(hashlib.sha256(vb).hexdigest())
                    cands.add(hashlib.sha512(vb).hexdigest())
                    cands.add(base64.b64encode(vb).decode("ascii"))
            digits_set = {re.sub(r"\D", "", s) for s in pt_secrets_dict.get(pt, set()) if len(re.sub(r"\D", "", s)) >= 5}
            hit = False
            for psn in psns:
                core = psn.replace("PSN-", "")
                if any(h.lower().startswith(core.lower()) for h in cands if len(h) >= 8):
                    hit = True
                if any(d in core for d in digits_set):
                    hit = True
            if hit:
                inv_hits += 1
        a6_inv = inv_hits / max(1, len(psn_pts))

    at_risk_details.sort(key=lambda x: (-x["top_prob"], x["candidates"]))
    return {
        "A1_direct_exposure": a1_risk,
        "A3_record_linkage": a3_mean,
        "A4_profile_reid": a4_reid,
        "A5_targeted_known_facts": a5_mean,
        "A5_recovered_global_shift": recovered_global_shift,
        "A5_decoy_fp": f"{decoy_fp}/{decoy_cnt}",
        "A6_pseudonym_inversion": a6_inv,
        "at_risk_profiles": at_risk_details,
        "profile_dates_by_pt": {
            ev.true_patient: {d.isoformat() for d in ev.dates}
            for ev in evs
            if ev.true_patient
        },
        "psns_by_pt": {
            ev.true_patient: ev.psn
            for ev in evs
            if ev.true_patient and ev.psn
        },
    }


def compute_cross_release_linkability(res_a: dict, res_b: dict) -> float:
    """A7: Max of stable PSNs or identical shifted dates across two different secrets."""
    pa, pb = res_a["psns_by_pt"], res_b["psns_by_pt"]
    common_pts = set(pa) & set(pb)
    psn_same = sum(1 for p in common_pts if pa[p] == pb[p]) / max(1, len(common_pts))

    da, db = res_a["profile_dates_by_pt"], res_b["profile_dates_by_pt"]
    common_d_pts = set(da) & set(db)
    date_same = 0
    date_tot = 0
    for p in common_d_pts:
        if da[p] and db[p]:
            date_tot += len(da[p])
            date_same += len(da[p] & db[p])
    date_overlap = date_same / max(1, date_tot)
    return max(psn_same, date_overlap)


def main() -> int:
    ap = argparse.ArgumentParser(description="Offline M3 Attacker Simulator (A1-A7)")
    ap.add_argument("--input", required=True, type=Path, help="dev/input directory")
    ap.add_argument("--output", required=True, type=Path, help="sanitized output root (contains sanitized/, report.json, records.jsonl)")
    ap.add_argument("--output2", type=Path, default=None, help="optional second run output (different secret) for A7")
    ap.add_argument("--truth", type=Path, default=None)
    ap.add_argument("--registry", type=Path, default=None)
    ap.add_argument("--facts", type=Path, default=None)
    ap.add_argument("--top", type=int, default=15)
    args = ap.parse_args()

    base_dev = args.input.parent
    truth_p = args.truth or (base_dev / "truth" / "truth.json")
    reg_p = args.registry or (base_dev / "attacker" / "registry.csv")
    facts_p = args.facts or (base_dev / "attacker" / "known_facts.jsonl")

    truth = json.loads(truth_p.read_text(encoding="utf-8"))
    registry = load_registry(reg_p)
    facts = [json.loads(l) for l in facts_p.read_text(encoding="utf-8").splitlines() if l.strip()]
    pt_to_reg = map_truth_patients_to_registry(truth, args.input, registry)

    raw_res = _eval_single_release(args.input, truth, registry, pt_to_reg, facts, is_raw=True)
    out_res = _eval_single_release(
        args.output / "sanitized",
        truth,
        registry,
        pt_to_reg,
        facts,
        is_raw=False,
        records_jsonl=args.output / "records.jsonl",
        report_json=args.output / "report.json",
    )

    a7 = None
    if args.output2 and args.output2.is_dir():
        out2_res = _eval_single_release(
            args.output2 / "sanitized",
            truth,
            registry,
            pt_to_reg,
            facts,
            is_raw=False,
            records_jsonl=args.output2 / "records.jsonl",
            report_json=args.output2 / "report.json",
        )
        a7 = compute_cross_release_linkability(out_res, out2_res)

    print("=" * 72)
    print("LOCAL ATTACKER ARENA REPORT (A1-A7)")
    print("=" * 72)
    for k in ("A1_direct_exposure", "A3_record_linkage", "A4_profile_reid", "A5_targeted_known_facts"):
        rv = raw_res[k]
        ov = out_res[k]
        rel = ov / rv if rv > 0 else 0.0
        print(f"  {k:<26}: raw={rv:6.3f}  out={ov:6.3f}  ratio(vs raw)={rel:6.3f}")
    print(f"  {'A5_recovered_global_shift':<26}: {out_res['A5_recovered_global_shift']} days (decoy FP: {out_res['A5_decoy_fp']})")
    print(f"  {'A6_pseudonym_inversion':<26}: {out_res['A6_pseudonym_inversion']:6.3f}")
    if a7 is not None:
        print(f"  {'A7_cross_release_link':<26}: {a7:6.3f}")

    # Weighted ATTACKS/SCORING privacy score (A1..A7)
    w_sum = 0.0
    r_sum = 0.0
    comps = [
        (0.20, out_res["A1_direct_exposure"]),
        (0.20, out_res["A3_record_linkage"] / raw_res["A3_record_linkage"] if raw_res["A3_record_linkage"] else 0.0),
        (0.20, out_res["A4_profile_reid"] / raw_res["A4_profile_reid"] if raw_res["A4_profile_reid"] else 0.0),
        (0.20, out_res["A5_targeted_known_facts"] / raw_res["A5_targeted_known_facts"] if raw_res["A5_targeted_known_facts"] else 0.0),
        (0.10, out_res["A6_pseudonym_inversion"]),
    ]
    if a7 is not None:
        comps.append((0.10, a7))
    for w, val in comps:
        w_sum += w
        r_sum += w * min(1.0, max(0.0, val))
    print(f"\n  Overall Weighted Attack Risk : {r_sum / w_sum:.4f} (Resistance: {1.0 - r_sum / w_sum:.4f})")

    if out_res["at_risk_profiles"] and args.top > 0:
        print(f"\nTop {min(args.top, len(out_res['at_risk_profiles']))} At-Risk Profiles (prob >= 0.35):")
        for item in out_res["at_risk_profiles"][: args.top]:
            print(f"  {item['profile']} p={item['top_prob']} cands={item['candidates']} pts={item['true_pts']} -> {item['reasons']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

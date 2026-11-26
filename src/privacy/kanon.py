"""Patient-pooled k-anonymity optimizer: adaptive age bands, geo roll-up, and inline scrubbing."""
from __future__ import annotations

import datetime as dt
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

import regex as re

from src.config import Policy
from src.ingest.profiler import infer_role_from_header, resolve_key_role
from src.linkage.evidence import RecordEvidence
from src.models import FileEntry
from src.nlp.indic_norm import normalize_name_token, shadow_normalize
from src.nlp.medical_guard import MedicalGuard
from src.security.keys import KeyRing

_TEHSIL_INLINE_RE = re.compile(
    r"(?i)(?:,?\s*\b(?:Tehsil|Tahsil|Taluka|Tal\.|Sub[\s-]?Dist(?:rict)?)\s*[:.\-]?\s*(\[NAME\]|[A-Z\u0900-\u097f][^\n,;.|]{1,28})"
    r"|तहसील\s*[:.\-]?\s*(\[NAME\]|[^\n,;.|]{1,28}))"
)
_AGE_REWRITE_RE = re.compile(
    r"(?i)(\b(?:age(?:/sex)?|आयु|उम्र)\s*[:\-]?\s*)(\d{1,3})([ \t]*(?:yrs?|years?|वर्ष|y)?(?:[ \t]*/[ \t]*[MF]\b)?)"
    r"|\b(\d{1,3})([ \t]*(?:yrs?|years?|वर्ष)(?:[ \t]*/[ \t]*[MF]\b)?)"
    r"|\b(\d{1,3})([ \t]*(?:yrs?|years?|y)?[ \t]*/[ \t]*[MF]\b)"
)
_HIST_BEFORE_RE = re.compile(r"(?i)(?:\b(?:h/o|k/c/o|c/o|since|for|past|last|duration|history|approx|x)\b|से|पहले)\s*[:\-]?\s*$")
_HIST_AFTER_RE = re.compile(r"(?i)^\s*(?:back|ago|later|earlier|prior|duration|se|से|पहले|बाद|तक)\b")


def format_age_band(age_min: int, age_max: int, width: int = 5, top_code: int = 85) -> str:
    """Compute a truthful age band covering [age_min, age_max] without over-widening straddlers."""
    lo_c = max(0, min(age_min, age_max))
    hi_c = max(0, max(age_min, age_max))
    if lo_c >= top_code:
        return f"{top_code}+"
    if hi_c >= top_code and lo_c >= top_code - width:
        return f"{top_code - width}+"
    lo_b = (lo_c // width) * width
    hi_b = ((hi_c // width) * width) + (width - 1)
    # If [lo_c, hi_c] crosses a width boundary (e.g., 39-40 at width=10 or 20), pick a tighter 5-yr-aligned band
    if (hi_b - lo_b + 1) > width:
        min_span = ((hi_c - lo_c) // 5 + 1) * 5
        w_eff = max(width, min_span)
        lo_5 = max(0, (lo_c // 5) * 5)
        hi_5 = ((hi_c // 5) * 5) + 4
        if (hi_5 - lo_5 + 1) <= w_eff:
            best_lo = min(lo_5, max(0, hi_5 - w_eff + 1))
            mid_shift = max(0, (w_eff - (hi_5 - lo_5 + 1)) // 2 // 5 * 5)
            lo_b = max(0, lo_5 - mid_shift)
            hi_b = lo_b + w_eff - 1
            if hi_b < hi_5:
                lo_b = best_lo
                hi_b = lo_b + w_eff - 1
    if hi_b >= top_code:
        return f"{lo_b}+"
    return f"{lo_b}-{hi_b}"


@dataclass
class PrivacyPlan:
    rec_cluster: dict[str, str] = field(default_factory=dict)
    rec_shift_days: dict[str, int] = field(default_factory=dict)
    rec_anchor_year: dict[str, int] = field(default_factory=dict)
    file_shift_days: dict[str, int] = field(default_factory=dict)
    rec_age_band: dict[str, str] = field(default_factory=dict)
    rec_birth_cohort: dict[str, str] = field(default_factory=dict)
    rec_suppress_tehsil: set[str] = field(default_factory=set)
    rec_pin_level: dict[str, str] = field(default_factory=dict)  # "pin3" | "pin2" | "suppress"
    harvested_tehsils: set[str] = field(default_factory=set)
    tehsil_rx: re.Pattern[str] | None = None
    records_generalized: int = 0
    records_suppressed: int = 0


def _collect_tehsil_values(entries: list[FileEntry], rec_tehsils: dict[str, set[str]]) -> set[str]:
    tehsils: set[str] = set()
    for entry in entries:
        if entry.status != "sanitized" or entry.is_clean_doc:
            continue
        if entry.file_type in ("csv", "tsv") and entry.parsed_data:
            roles = entry.column_roles
            for rec, row in zip(entry.records, entry.parsed_data[1:]):
                for idx, val in enumerate(row):
                    s = str(val).strip()
                    if not s:
                        continue
                    role = roles.get(idx, "FREE_TEXT")
                    if role == "TEHSIL":
                        nt = normalize_name_token(s)
                        if nt:
                            rec_tehsils[rec.record_uid].add(nt)
                        tehsils.add(s)
                    elif role in ("ADDRESS", "NOTES", "FREE_TEXT"):
                        for m in _TEHSIL_INLINE_RE.finditer(s):
                            tok = (m.group(1) or m.group(2) or "").strip()
                            if tok:
                                tehsils.add(tok)
                                rec_tehsils[rec.record_uid].add(normalize_name_token(tok))
        elif entry.file_type in ("json", "fhir") and entry.parsed_data is not None:
            for rec in entry.records:
                stack = [entry.parsed_data if rec.loc == "doc" else rec.raw_payload]
                while stack:
                    cur = stack.pop()
                    if isinstance(cur, dict):
                        for k, v in cur.items():
                            if isinstance(v, (dict, list)):
                                stack.append(v)
                            elif isinstance(v, str) and v.strip():
                                role = resolve_key_role(entry, str(k))
                                if role == "TEHSIL" or "tehsil" in str(k).lower():
                                    tehsils.add(v.strip())
                                    rec_tehsils[rec.record_uid].add(normalize_name_token(v))
                                elif str(k).lower() in ("line", "address", "address_note", "notes", "remarks"):
                                    for m in _TEHSIL_INLINE_RE.finditer(v):
                                        tok = (m.group(1) or m.group(2) or "").strip()
                                        if tok:
                                            tehsils.add(tok)
                                            rec_tehsils[rec.record_uid].add(normalize_name_token(tok))
                    elif isinstance(cur, list):
                        for item in cur:
                            if isinstance(item, (dict, list)):
                                stack.append(item)
                            elif isinstance(item, str):
                                for m in _TEHSIL_INLINE_RE.finditer(item):
                                    tok = (m.group(1) or m.group(2) or "").strip()
                                    if tok:
                                        tehsils.add(tok)
                                        rec_tehsils[rec.record_uid].add(normalize_name_token(tok))
        elif entry.text_decoded and entry.records:
            rec = entry.records[0]
            for m in _TEHSIL_INLINE_RE.finditer(entry.text_decoded):
                tok = (m.group(1) or m.group(2) or "").strip()
                if tok:
                    tehsils.add(tok)
                    rec_tehsils[rec.record_uid].add(normalize_name_token(tok))
    return tehsils


def build_privacy_plan(
    entries: list[FileEntry],
    evs: list[RecordEvidence],
    keyring: KeyRing,
    policy: Policy,
) -> PrivacyPlan:
    """Compute patient-pooled k-anonymity plan and deterministic per-patient date shifts."""
    plan = PrivacyPlan()
    rec_tehsils: dict[str, set[str]] = defaultdict(set)
    plan.harvested_tehsils = _collect_tehsil_values(entries, rec_tehsils)

    # Compile harvested tehsil regex so free-text references are scrubbed consistently
    clean_tehs = sorted({t.strip() for t in plan.harvested_tehsils if len(t.strip()) >= 4}, key=len, reverse=True)
    if clean_tehs:
        plan.tehsil_rx = re.compile(r"(?i)(?<!\w)(?:" + "|".join(re.escape(t) for t in clean_tehs) + r")(?!\w)")

    # Pool patient quasi-identifiers across linked clusters
    clusters: dict[str, list[RecordEvidence]] = defaultdict(list)
    for entry in entries:
        if entry.status == "sanitized":
            plan.file_shift_days[entry.rel_input_path] = keyring.date_shift_days(
                f"file::{entry.input_sha256}", policy.date_shift_min_days, policy.date_shift_max_days
            )

    for ev in evs:
        ckey = ev.record.cluster_id or f"unlinked::{ev.entry.input_sha256}::{ev.record.loc}"
        clusters[ckey].append(ev)
        plan.rec_cluster[ev.uid] = ckey

    # Assign shifts & collect per-cluster age ranges and geo sets
    c_age_bounds: dict[str, tuple[int, int] | None] = {}
    c_gender: dict[str, str] = {}
    c_dist: dict[str, str] = {}
    c_teh: dict[str, str] = {}
    c_pin3: dict[str, str] = {}
    c_anchor: dict[str, int] = {}

    # Collect DOBs across all records in each file so shifted clinical dates never collide with any DOB in that file
    file_dobs: dict[str, set[dt.date]] = defaultdict(set)
    for ev in evs:
        if ev.dobs:
            file_dobs[ev.entry.rel_input_path].update(ev.dobs)

    for ckey, grp in clusters.items():
        all_dobs = {d for e in grp for d in e.dobs}
        for e in grp:
            all_dobs.update(file_dobs.get(e.entry.rel_input_path, set()))
        all_dates = [d for e in grp for d in e.clinical_dates]
        shift = keyring.date_shift_days(ckey, policy.date_shift_min_days, policy.date_shift_max_days)
        if all_dobs and all_dates:
            disambig = 0
            while any((cd + dt.timedelta(days=shift)) in all_dobs for cd in all_dates) and disambig < 64:
                disambig += 1
                shift = keyring.date_shift_days(f"{ckey}#nodob{disambig}", policy.date_shift_min_days, policy.date_shift_max_days)
        anchor = sorted(all_dates)[len(all_dates) // 2].year if all_dates else 2026
        c_anchor[ckey] = anchor

        ages: list[int] = [a for e in grp for a in e.ages]
        for e in grp:
            for d in e.dobs:
                ages.append(max(0, min(115, anchor - d.year)))
        c_age_bounds[ckey] = (min(ages), max(ages)) if ages else None

        genders = [e.gender for e in grp if e.gender]
        c_gender[ckey] = Counter(genders).most_common(1)[0][0] if genders else "?"
        dists = sorted({d for e in grp for d in e.districts if d})
        c_dist[ckey] = dists[0] if dists else "?"
        tehs = sorted({t for e in grp for t in rec_tehsils.get(e.uid, set()) if t})
        c_teh[ckey] = tehs[0] if tehs else ""
        pins = sorted({p[:3] for e in grp for p in e.pins if len(p) == 6})
        c_pin3[ckey] = pins[0] if pins else ""

        for e in grp:
            plan.rec_shift_days[e.uid] = shift
            plan.rec_anchor_year[e.uid] = anchor

    # Adaptive k-anonymity on age_band: widen 5 -> 10 -> 15 -> 20 yrs per (gender, district) if eq_class < k
    k_target = max(2, policy.min_group_size)
    c_band: dict[str, str] = {}
    for ckey, bounds in c_age_bounds.items():
        if bounds is None:
            continue
        c_band[ckey] = format_age_band(bounds[0], bounds[1], width=policy.age_band_years, top_code=policy.age_top_code)

    for width in (10, 15, 20):
        eq_counts = Counter((c_gender[c], c_dist[c], c_band[c]) for c in c_band)
        changed = False
        for ckey, band in list(c_band.items()):
            if eq_counts[(c_gender[ckey], c_dist[ckey], band)] < k_target:
                bounds = c_age_bounds[ckey]
                assert bounds is not None
                new_band = format_age_band(bounds[0], bounds[1], width=width, top_code=min(80, policy.age_top_code))
                if new_band != band:
                    c_band[ckey] = new_band
                    changed = True
        if not changed:
            break

    # Safety step for extreme young/elderly outliers still in small buckets (< 3 in district)
    eq_counts = Counter((c_gender[c], c_dist[c], c_band[c]) for c in c_band)
    for ckey, band in list(c_band.items()):
        if eq_counts[(c_gender[ckey], c_dist[ckey], band)] < 3:
            bounds = c_age_bounds[ckey]
            assert bounds is not None
            if bounds[1] <= 29:
                c_band[ckey] = "0-29"
            elif bounds[0] >= 65:
                c_band[ckey] = "65+"

    # Adaptive geo roll-up: check (gender, district, tehsil) and (gender, district, pin3) group sizes
    teh_eq = Counter((c_gender[c], c_dist[c], c_teh[c]) for c in clusters if c_teh[c])
    pin_eq = Counter((c_gender[c], c_dist[c], c_pin3[c]) for c in clusters if c_pin3[c])
    geo_k = max(2, policy.location_min_group)

    for ckey, grp in clusters.items():
        band = c_band.get(ckey, "")
        bounds = c_age_bounds.get(ckey)
        cohort = ""
        if bounds:
            anchor = c_anchor[ckey]
            # Derive truthful 5-to-20 year birth cohort matching the generalized band
            m_top = re.match(r"^(\d+)\+$", band)
            if m_top:
                cohort = f"<={anchor - int(m_top.group(1))}"
            elif "-" in band:
                b_lo, b_hi = band.split("-", 1)
                cohort = f"{anchor - int(b_hi)}-{anchor - int(b_lo)}"
            else:
                cohort = f"{anchor - bounds[1] - 4}-{anchor - bounds[0] + 4}"

        suppress_teh = not c_teh[ckey] or teh_eq[(c_gender[ckey], c_dist[ckey], c_teh[ckey])] < policy.min_group_size
        pin3 = c_pin3[ckey]
        pin_mode = "suppress" if (not pin3 or pin_eq[(c_gender[ckey], c_dist[ckey], pin3)] < geo_k) else "pin3"

        for e in grp:
            if band:
                plan.rec_age_band[e.uid] = band
            if cohort:
                plan.rec_birth_cohort[e.uid] = cohort
            if suppress_teh:
                plan.rec_suppress_tehsil.add(e.uid)
            plan.rec_pin_level[e.uid] = pin_mode

    plan.records_generalized = sum(
        1 for ev in evs
        if ev.uid in plan.rec_age_band or ev.uid in plan.rec_suppress_tehsil or plan.rec_pin_level.get(ev.uid) in ("pin3", "suppress")
    )
    plan.records_suppressed = 0  # quasi-identifier fields are generalised/blanked; whole records are never dropped
    return plan


_AGE_NULLS = {"", "nan", "null", "none", "na", "n/a", "nil", "unknown", "-", "--", "?"}
# "8 months", "3 days", "45 Y 3 M", "2 wks": an age written with units (never hex like 0x7fffffff)
_AGE_UNIT = re.compile(r"(?i)^\d{1,3}\s*(?:months?|mons?|mths?|mo|m|days?|d|weeks?|wks?|w|yrs?|years?|y)\b"
                       r"|(?:महीने|माह|दिन|सप्ताह|वर्ष)")


def generalize_age_value(raw: str, age_band: str | None) -> str:
    """Replace exact numeric age inside structured Age/age_sex cells with its truthful band.

    Fails closed: a value that cannot be turned into a truthful band (negative, > 120, integer-overflow
    probes, free text) becomes [AGE] instead of passing through raw."""
    s = str(raw).strip()
    norm = shadow_normalize(s)
    if norm.lower() in _AGE_NULLS:
        return s
    m = re.match(r"^(\d{1,3})([ \t]*(?:y?[ \t]*/[ \t]*[MF]|yrs?|years?|वर्ष|y)?)$", norm, re.I)
    if m and int(m.group(1)) <= 120:
        return f"{age_band}{m.group(2) or ''}" if age_band else "[AGE]"
    if age_band and not norm.startswith("-") and not m and _AGE_UNIT.search(norm) \
            and re.search(r"\d", norm) and not re.search(r"\d{4,}", norm):
        return age_band  # "8 months", "45 Y 3 M": the record's truthful band
    return "[AGE]"


def scrub_tehsils_and_ages_in_text(
    text: str,
    age_band: str | None,
    suppress_tehsil: bool,
    tehsil_rx: re.Pattern[str] | None,
    guard: MedicalGuard,
) -> str:
    """Scrub tehsils and generalize inline exact ages while preserving clinical values/history."""
    if not text:
        return text
    out = text
    if suppress_tehsil:
        out = _TEHSIL_INLINE_RE.sub(lambda m: ", [ADDRESS]" if m.group(0).lstrip().startswith(",") else "[ADDRESS]", out)
        if tehsil_rx is not None:
            prot_t = guard.protected_spans(out)
            pieces: list[str] = []
            pos = 0
            for m in tehsil_rx.finditer(out):
                if any(ps < m.end() and pe > m.start() for ps, pe, _ in prot_t):
                    continue
                pieces.append(out[pos:m.start()])
                pieces.append("[ADDRESS]")
                pos = m.end()
            pieces.append(out[pos:])
            out = "".join(pieces)

    if not age_band:
        return out

    norm = shadow_normalize(out)
    prot = guard.protected_spans(norm)
    parts: list[str] = []
    pos = 0
    for m in _AGE_REWRITE_RE.finditer(norm):
        ms, me = m.start(), m.end()
        if any(ps < me and pe > ms and tag == "MEDICAL" for ps, pe, tag in prot):
            continue
        is_explicit_age = bool(m.group(1) or m.group(6))
        before = norm[max(0, ms - 30):ms]
        after = norm[me:min(len(norm), me + 20)]
        if _HIST_AFTER_RE.match(after):
            continue
        if not is_explicit_age and _HIST_BEFORE_RE.search(before):
            continue
        parts.append(out[pos:ms])
        if m.group(1):
            parts.append(f"{out[m.start(1):m.end(1)]}{age_band}{out[m.start(3):m.end(3)]}")
        elif m.group(4):
            parts.append(f"{age_band}{out[m.start(5):m.end(5)]}")
        else:
            parts.append(f"{age_band}{out[m.start(7):m.end(7)]}")
        pos = me
    parts.append(out[pos:])
    return "".join(parts)

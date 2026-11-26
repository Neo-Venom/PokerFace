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
from src.nlp.indic_norm import normalize_name_token, phonetic_skeleton, shadow_normalize
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


# Coarse life-stage partitions used once 20-year bands still leave a patient in a group smaller than k.
# Minors keep a <=20-year band ("0-17") at every level; open-ended "N+" bands follow the "85+" convention.
_LIFE_STAGES: tuple[tuple[tuple[int, int | None], ...], ...] = (
    ((0, 17), (18, 59), (60, None)),
    ((0, 17), (18, None)),
)
_OPEN_HI = 200
_CLOSED_TOP = 120  # upper bound written for adult-wide bands (ages are capped at 115 everywhere)


def _stage_band(lo: int, hi: int, stages: tuple[tuple[int, int | None], ...]) -> str:
    def _stage_of(age: int) -> tuple[int, int | None]:
        for s_lo, s_hi in stages:
            if s_hi is None or age <= s_hi:
                if age >= s_lo:
                    return s_lo, s_hi
        return stages[-1]

    a = _stage_of(lo)[0]
    b = _stage_of(hi)[1]
    if b is None:
        # "60+" reads as an elderly band everywhere; an adult-wide open band is written closed ("18-120") because
        # age parsers read "Age: 18+" as the exact age 18, which would single out 18-20 year olds.
        return f"{a}+" if a >= 60 else f"{a}-{_CLOSED_TOP}"
    return f"{a}-{b}"


def age_band_ladder(age_min: int, age_max: int, base_width: int = 5, top_code: int = 85) -> list[str]:
    """Truthful bands for [age_min, age_max] from most to least specific (duplicates removed).

    5y -> 10y -> 20y (format_age_band, straddler aware) -> life stage (0-17 / 18-59 / 60+) -> (0-17 / 18+).
    Every band contains every age in [age_min, age_max]."""
    lo, hi = max(0, min(age_min, age_max)), max(0, max(age_min, age_max))
    bands = [
        format_age_band(lo, hi, width=base_width, top_code=top_code),
        format_age_band(lo, hi, width=10, top_code=min(80, top_code)),
        format_age_band(lo, hi, width=20, top_code=min(80, top_code)),
    ]
    bands.extend(_stage_band(lo, hi, st) for st in _LIFE_STAGES)
    out: list[str] = []
    for b in bands:
        if b not in out:
            out.append(b)
    return out


def band_bounds(band: str) -> tuple[int, int]:
    """(lo, hi) of an age band string ("45-49", "85+"); open bands end at a sentinel age."""
    m = re.match(r"^(\d+)\+$", band)
    if m:
        return int(m.group(1)), _OPEN_HI
    m = re.match(r"^(\d+)-(\d+)$", band)
    if m:
        return int(m.group(1)), int(m.group(2))
    return 0, _OPEN_HI


_ICD_RE = re.compile(r"^[A-Z]\d{2}(?:\.\d{1,4})?$", re.I)
# "x 5 days", "7 din se", "since 2 weeks", "3 saal": a free-text duration, not part of a condition name
_DURATION_RE = re.compile(
    r"(?i)\d+\s*(?:days?|din|dino|d|weeks?|wks?|hafte?|months?|mahine?|mo|yrs?|years?|saal|sal|hrs?|hours?|ghante?)\b"
    r"|\bx\s*\d|\bsince\b|\bse\b"
)


def condition_keys(diagnoses: set[str] | list[str]) -> set[str]:
    """Diagnosis strings that name a condition (ICD-10 codes or condition names), not complaint durations.

    Complaints such as "c/o fever x 5 days" or "7 din se bukhar" carry free-text durations that make every
    string look unique; they are not sensitive attributes and would make every patient look rare."""
    out: set[str] = set()
    for d in diagnoses:
        s = " ".join(str(d).strip().lower().split())
        if not s or len(s) > 90:
            continue
        if _ICD_RE.match(s):
            out.add(s.upper())
            continue
        if _DURATION_RE.search(s) or re.match(r"(?:c/o|h/o|k/c/o|complain)", s):
            continue
        if len(s) >= 4 and re.search(r"[a-zऀ-ॿ]{3}", s):
            out.add(s)
    return out


@dataclass
class _ClusterQI:
    key: str
    gender: str                      # "M" | "F" | "?"
    bounds: tuple[int, int] | None   # pooled (min, max) true age
    district: str                    # canonical district key ("" = unknown)
    pin3: str
    conditions: frozenset[str]


class _AnonIndex:
    """Counts distinct patient clusters consistent with a generalised quasi-identifier tuple.

    A cluster is consistent with (gender, district, [pin3,] band) when its gender is compatible (unknown gender is
    compatible with both), it lives in that district (an unknown district is matched against every district) and
    its pooled true age range overlaps the band. Counting uses per-group sorted bounds and bisection: O(log n)."""

    def __init__(self, qis: list[_ClusterQI]) -> None:
        from bisect import bisect_left, bisect_right

        self._bl, self._br = bisect_left, bisect_right
        groups: dict[tuple[str, str, str], list[tuple[int, int]]] = defaultdict(list)
        members: dict[tuple[str, str], list[_ClusterQI]] = defaultdict(list)
        for q in qis:
            if q.bounds is None:
                continue
            for dkey in ({q.district, "*"} if q.district else {"*"}):
                groups[(dkey, "", q.gender)].append(q.bounds)
                if q.pin3:
                    groups[(dkey, q.pin3, q.gender)].append(q.bounds)
                members[(dkey, q.gender)].append(q)
        self._los = {k: sorted(b[0] for b in v) for k, v in groups.items()}
        self._his = {k: sorted(b[1] for b in v) for k, v in groups.items()}
        self._members = {k: sorted(v, key=lambda q: (q.bounds or (0, 0), q.key)) for k, v in members.items()}

    @staticmethod
    def _genders(g: str) -> tuple[str, ...]:
        return (g, "?") if g in ("M", "F") else ("M", "F", "?")

    def count(self, q: _ClusterQI, band: str, use_pin3: bool = False) -> int:
        blo, bhi = band_bounds(band)
        dkey = q.district or "*"
        pkey = q.pin3 if use_pin3 else ""
        n = 0
        for g in self._genders(q.gender):
            k = (dkey, pkey, g)
            los = self._los.get(k)
            if not los:
                continue
            n += self._br(los, bhi) - self._bl(self._his[k], blo)
        return n

    def distinct_conditions(self, q: _ClusterQI, band: str, primary: dict[str, str], need: int) -> int:
        """Distinct primary conditions among consistent clusters (stops early once `need` is reached)."""
        blo, bhi = band_bounds(band)
        dkey = q.district or "*"
        seen: set[str] = set()
        for g in self._genders(q.gender):
            for m in self._members.get((dkey, g), []):
                assert m.bounds is not None
                if m.bounds[0] > bhi:
                    break
                if m.bounds[1] >= blo and m.key in primary:
                    seen.add(primary[m.key])
                    if len(seen) >= need:
                        return len(seen)
        return len(seen)


# Country / state / region words are not districts (a record saying "Uttar Pradesh, India" has no district vote)
_REGION_WORDS = frozenset(re.sub(r"[^a-z]", "", w) for w in (
    "india", "bharat", "pradesh", "uttar", "state", "uttar pradesh", "up", "bihar", "madhya pradesh", "mp",
    "rajasthan", "maharashtra", "delhi", "new delhi", "haryana", "punjab", "uttarakhand", "himachal pradesh",
    "jharkhand", "chhattisgarh", "odisha", "orissa", "west bengal", "bengal", "gujarat", "karnataka", "tamil nadu",
    "kerala", "andhra pradesh", "telangana", "assam", "jammu and kashmir", "goa", "tripura", "manipur", "meghalaya",
    "mizoram", "nagaland", "sikkim", "arunachal pradesh",
))


def _is_region(token: str) -> bool:
    t = re.sub(r"[^a-z]", "", token.lower())
    return t in _REGION_WORDS or token.strip() in ("उत्तर प्रदेश", "उत्तरप्रदेश", "भारत")


def _canon_district(token: str) -> str:
    return phonetic_skeleton(token) or token


def plan_quasi_identifiers(qis: list[_ClusterQI], policy: Policy) -> tuple[dict[str, str], dict[str, bool], dict[str, int]]:
    """Choose, per patient cluster, the most specific age band whose equivalence class is k-anonymous.

    Returns (band per cluster, keep-PIN-prefix per cluster, ladder level per cluster).

    * k-anonymity is measured over DISTINCT patient clusters (never rows) that are consistent with the released
      (gender, district, band) - see _AnonIndex - so a patient with 40 visits still counts once.
    * Patients carrying a rare condition (seen in fewer than rare_dx_min_patients clusters) are the targets of the
      known-facts attack: the condition finds their profile, so only the registry anonymity of the remaining
      quasi-identifiers protects them; they need 2k.
    * l-diversity: an equivalence class with diagnoses must show at least l distinct primary conditions, otherwise
      membership alone discloses the condition (homogeneity attack).
    * The 3-digit PIN prefix is released only when the class stays k-anonymous with it.
    """
    k = max(2, int(policy.min_group_size))
    l_div = max(1, int(policy.l_diversity))
    rare_n = max(1, int(policy.rare_dx_min_patients))
    cond_freq: Counter[str] = Counter()
    for q in qis:
        cond_freq.update(q.conditions)
    primary: dict[str, str] = {}
    for q in qis:
        if q.conditions:
            primary[q.key] = min(q.conditions, key=lambda c: (cond_freq[c], c))
    index = _AnonIndex(qis)
    bands: dict[str, str] = {}
    keep_pin: dict[str, bool] = {}
    levels: dict[str, int] = {}
    for q in sorted(qis, key=lambda x: x.key):
        if q.bounds is None:
            continue
        rare = bool(q.conditions) and cond_freq[primary[q.key]] < rare_n
        need = 2 * k if rare else k
        ladder = age_band_ladder(q.bounds[0], q.bounds[1], base_width=policy.age_band_years, top_code=policy.age_top_code)
        chosen, lvl = ladder[-1], len(ladder) - 1
        for i, band in enumerate(ladder):
            if index.count(q, band) < need:
                continue
            if l_div > 1 and q.key in primary and index.distinct_conditions(q, band, primary, l_div) < l_div:
                continue
            chosen, lvl = band, i
            break
        bands[q.key] = chosen
        levels[q.key] = lvl
        keep_pin[q.key] = bool(q.pin3) and index.count(q, chosen, use_pin3=True) >= need
    return bands, keep_pin, levels


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
    # The record's own structured district value: a village/city field is generalised to it (truthful, and the
    # record already releases it) instead of being blanked.
    rec_district: dict[str, str] = field(default_factory=dict)
    harvested_tehsils: set[str] = field(default_factory=set)
    tehsil_rx: re.Pattern[str] | None = None
    records_generalized: int = 0
    records_suppressed: int = 0


_DISTRICT_KEYS = ("district", "distt", "dist", "jila", "zila", "zilla")


def _note_district(rec_district: dict[str, str], uid: str, value: str) -> None:
    """Remember a record's structured district; a record naming two different districts keeps none."""
    v = " ".join(value.split())
    if not v or len(v) > 40 or not re.search(r"[^\W\d_]{3}", v) or re.search(r"\d", v):
        return
    prev = rec_district.get(uid)
    if prev is None:
        rec_district[uid] = v
    elif prev and prev.casefold() != v.casefold():
        rec_district[uid] = ""


def _collect_tehsil_values(
    entries: list[FileEntry], rec_tehsils: dict[str, set[str]], rec_district: dict[str, str] | None = None
) -> set[str]:
    tehsils: set[str] = set()
    if rec_district is None:
        rec_district = {}
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
                    if role == "DISTRICT":
                        _note_district(rec_district, rec.record_uid, s)
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
                                if role == "DISTRICT" or str(k).lower() in _DISTRICT_KEYS:
                                    _note_district(rec_district, rec.record_uid, v)
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
    rec_district: dict[str, str] = {}
    plan.harvested_tehsils = _collect_tehsil_values(entries, rec_tehsils, rec_district)
    plan.rec_district = {uid: d for uid, d in sorted(rec_district.items()) if d}

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
    c_conditions: dict[str, frozenset[str]] = {}

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
        # Majority district over the cluster's records (a hospital's own town mentioned once must not win)
        dist_votes = Counter(_canon_district(d) for e in grp for d in e.districts if d and not _is_region(d))
        c_dist[ckey] = min(dist_votes, key=lambda d: (-dist_votes[d], d)) if dist_votes else ""
        tehs = sorted({t for e in grp for t in rec_tehsils.get(e.uid, set()) if t})
        c_teh[ckey] = tehs[0] if tehs else ""
        pin_votes = Counter(p[:3] for e in grp for p in e.pins if len(p) == 6)
        c_pin3[ckey] = min(pin_votes, key=lambda p: (-pin_votes[p], p)) if pin_votes else ""
        c_conditions[ckey] = frozenset(condition_keys({d for e in grp for d in e.diagnoses}))

        for e in grp:
            plan.rec_shift_days[e.uid] = shift
            plan.rec_anchor_year[e.uid] = anchor

    # Joint k-anonymity / l-diversity over distinct patient clusters (see plan_quasi_identifiers)
    qis = [
        _ClusterQI(ckey, c_gender[ckey] if c_gender[ckey] in ("M", "F") else "?", c_age_bounds[ckey],
                   c_dist[ckey], c_pin3[ckey], c_conditions[ckey])
        for ckey in sorted(clusters)
    ]
    c_band, c_keep_pin, _levels = plan_quasi_identifiers(qis, policy)

    # Tehsil is finer than district: release it only when (gender, district, tehsil) holds >= k patients
    teh_eq = Counter((c_gender[c], c_dist[c], c_teh[c]) for c in clusters if c_teh[c])

    for ckey, grp in clusters.items():
        band = c_band.get(ckey, "")
        bounds = c_age_bounds.get(ckey)
        cohort = ""
        if bounds and band:
            anchor = c_anchor[ckey]
            # Truthful birth cohort matching the generalised band
            b_lo, b_hi = band_bounds(band)
            cohort = f"<={anchor - b_lo}" if b_hi >= _OPEN_HI else f"{anchor - b_hi}-{anchor - b_lo}"

        suppress_teh = not c_teh[ckey] or teh_eq[(c_gender[ckey], c_dist[ckey], c_teh[ckey])] < policy.min_group_size
        pin_mode = "pin3" if c_keep_pin.get(ckey) else "suppress"

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

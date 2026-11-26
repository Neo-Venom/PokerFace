"""Two-stage cross-script Fellegi-Sunter scorer + family phone & hard contradiction vetoes."""
from __future__ import annotations

import datetime as dt
import math
import re
from collections import Counter, defaultdict
from typing import Iterable

from rapidfuzz.distance import JaroWinkler

from src.linkage.evidence import RecordEvidence
from src.nlp.indic_norm import GENERIC_WEAK_TOKENS, phonetic_skeleton

MATCH = 0.88    # token similarity that makes two name parts the same
COMPAT = 0.85   # transliteration variant ("gres" ~ "grace", "naiar" ~ "nair"): never a contradiction

# Name tokens romanised from Devanagari in the current corpus. Machine romanisation keeps schwas and vowel
# lengths ("कुलकर्णी" -> "kulakarni", "अय्यर" -> "aiar"), so such tokens get a looser consonant comparison.
_TRANSLIT: set[str] = set()


def set_translit_tokens(tokens: Iterable[str]) -> None:
    _TRANSLIT.clear()
    _TRANSLIT.update(tokens)


def _fold_key(t: str) -> str:
    """Consonant key that ignores vowels, semivowels and aspiration ('iyer'/'aiar' -> '_r', 'grace'/'gres' -> 'grs')."""
    s = re.sub(r"c(?=[eiy])", "s", t).replace("ck", "k").replace("c", "k")
    head, tail = s[:1], s[1:]
    if head in "aeiou":
        head = "_"
    tail = re.sub(r"([kgjtdpbs])h", r"\1", tail)
    tail = re.sub(r"[aeiouyvwh]", "", tail)
    return re.sub(r"(.)\1+", r"\1", head + tail)


def _tok_sim(a: str, b: str) -> float:
    if a == b:
        return 1.0
    # Initial match: "r" matches "ramesh" / "m" matches "mohammad"
    if (len(a) == 1 and b.startswith(a)) or (len(b) == 1 and a.startswith(b)):
        return 0.88
    if len(a) == 1 or len(b) == 1:
        return 0.0
    jw = JaroWinkler.normalized_similarity(a, b)
    same_skel = phonetic_skeleton(a) == phonetic_skeleton(b)
    if jw >= 0.80 and same_skel:
        return max(jw, 0.90)
    if MATCH <= jw < 0.95 and not same_skel:
        jw = 0.84   # one consonant apart ("rajesh" / "ramesh"): different names, not a spelling variant
    if len(a) >= 3 and len(b) >= 3 and jw < MATCH:
        fk = _fold_key(a)
        if fk == _fold_key(b) and len(fk.lstrip("_")) >= 1 and (a in _TRANSLIT or b in _TRANSLIT or jw >= 0.80):
            return max(min(jw, 0.87), COMPAT)
    return jw


def _align(sa: tuple[str, ...], sb: tuple[str, ...]) -> tuple[list[float], int]:
    """Greedy token alignment: (similarities of matched tokens, number of transliteration-compatible tokens)."""
    used: set[int] = set()
    sims: list[float] = []
    ncompat = 0
    for wa in sa:
        best_j, best_s = -1, 0.0
        for j, wb in enumerate(sb):
            if j in used:
                continue
            s = _tok_sim(wa, wb)
            if s > best_s:
                best_s, best_j = s, j
        if best_j < 0:
            continue
        if best_s >= MATCH:
            used.add(best_j)
            sims.append(best_s)
        elif best_s >= COMPAT:
            used.add(best_j)
            ncompat += 1
    return sims, ncompat


def _name_pair_match(names_a: list[tuple[str, ...]], names_b: list[tuple[str, ...]]) -> tuple[float, int, bool]:
    """Return (best_similarity, matched_token_count, explicit_full_conflict)."""
    if not names_a or not names_b:
        return 0.0, 0, False
    best_key = (0, 0.0)
    best_sim = 0.0
    best_matched = 0
    aligned = False   # some multi-token pair aligned completely (exact + transliteration variants)
    for ta in names_a:
        for tb in names_b:
            if not ta or not tb:
                continue
            sa, sb = (ta, tb) if len(ta) <= len(tb) else (tb, ta)
            sims, ncompat = _align(sa, sb)
            matched = len(sims)
            if len(sa) >= 2 and matched + ncompat == len(sa) and matched >= 1:
                aligned = True
                # "Grace Kulkarni" ~ "ग्रेस कुलकर्णी": a transliteration variant counts as a (weaker) match
                avg = (sum(sims) + MATCH * ncompat) / len(sa)
                key = (2, avg)
                if key > best_key:
                    best_key, best_sim, best_matched = key, avg, len(sa)
            elif matched >= 1 and (len(sa) == 1 or len(sb) == 1):
                # One side only has single token (e.g. "Shahid" vs "Mohammad Shahid", or "Asha" vs "Asha Rajput")
                avg = sum(sims) / matched * 0.92
                key = (1, avg)
                if key > best_key:
                    best_key, best_sim, best_matched = key, avg, matched
    # Check explicit conflict when both sides have >=2 tokens and none/few match
    conflict = False
    multi_a = any(len(t) >= 2 for t in names_a)
    multi_b = any(len(t) >= 2 for t in names_b)
    if aligned or not (multi_a and multi_b):
        return best_sim, best_matched, False
    if best_matched == 0:
        conflict = True
    elif best_matched == 1:
        # E.g. "Sunita Verma" vs "Sunita Patel" -> 1 shared first name, different surnames!
        s_a = {t for fn in names_a if len(fn) >= 2 for t in fn[1:]}
        s_b = {t for fn in names_b if len(fn) >= 2 for t in fn[1:]}
        if s_a and s_b and not any(_tok_sim(x, y) >= COMPAT for x in s_a for y in s_b):
            conflict = True
    return best_sim, best_matched, conflict


def _given_tokens(ev: RecordEvidence) -> set[str]:
    return {fn[0] for fn in ev.full_names if len(fn) >= 2 and len(fn[0]) > 1}


def _givens_compatible(ga: set[str], gb: set[str]) -> bool:
    return any(_tok_sim(x, y) >= COMPAT for x in ga for y in gb)


def dob_compatible(d1: dt.date, d2: dt.date) -> bool:
    """Same date, or day and month swapped (DD/MM vs MM/DD export) in the same year."""
    return d1 == d2 or (d1.year == d2.year and d1.day == d2.month and d1.month == d2.day)


def dobs_compatible(da: Iterable[dt.date], db: Iterable[dt.date]) -> bool:
    db = list(db)
    return any(dob_compatible(x, y) for x in da for y in db)


def _registry_ids(ev: RecordEvidence) -> set[str]:
    return ev.uhids - ev.episode_ids


def strong_id_groups(evs: list[RecordEvidence]) -> list[int]:
    """Group records that share a single-person identifier (Aadhaar, ABHA, PAN, email, insurance ID,
    hospital or visit number: equal numbers link even when they are per-visit) and do not contradict each other
    on gender or DOB. Returns a group id per record."""
    parent = list(range(len(evs)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    first: dict[tuple[str, str], int] = {}
    for i, ev in enumerate(evs):
        if ev.is_multi_patient_summary:
            continue
        keys = sorted(
            [("A", x) for x in ev.aadhaars] + [("B", x) for x in ev.abhas] + [("P", x) for x in ev.pans]
            + [("E", x) for x in ev.emails] + [("I", x) for x in ev.ins_ids] + [("U", x) for x in ev.uhids]
        )
        for k in keys:
            j = first.setdefault(k, i)
            if j == i:
                continue
            o = evs[j]
            if (ev.gender and o.gender and ev.gender != o.gender) or (ev.dobs and o.dobs and not dobs_compatible(ev.dobs, o.dobs)):
                continue
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[max(ri, rj)] = min(ri, rj)
    return [find(i) for i in range(len(evs))]


def detect_shared_family_phones(evs: list[RecordEvidence], groups: list[int] | None = None) -> set[str]:
    """Identify phones shared by distinct individuals (attendant/family phones).

    Records are first grouped by single-person identifiers, so one patient written two ways ("Grace Kulkarni"
    and "ग्रेस कुलकर्णी", or a nickname next to the full name under one UHID) is not mistaken for two people."""
    if groups is None:
        groups = strong_id_groups(evs)
    by_ph: dict[str, dict[int, list[RecordEvidence]]] = defaultdict(lambda: defaultdict(list))
    for i, ev in enumerate(evs):
        if ev.is_multi_patient_summary:
            continue
        for ph in ev.phones:
            by_ph[ph][groups[i]].append(ev)
    shared: set[str] = set()
    for ph in sorted(by_ph):
        grp = by_ph[ph]
        if len(grp) < 2:
            continue
        profiles = []
        for g in sorted(grp):
            members = grp[g]
            genders = {ev.gender for ev in members if ev.gender}
            years = sorted(y for ev in members for y in ev.birth_years)
            med = years[len(years) // 2] if years else None
            givens = set().union(*(_given_tokens(ev) for ev in members))
            profiles.append((genders, med, givens))
        gset = {g for p in profiles if len(p[0]) == 1 for g in p[0]}
        meds = [p[1] for p in profiles if p[1] is not None]
        if len(gset) > 1 or (len(meds) >= 2 and max(meds) - min(meds) > 5):
            shared.add(ph)
            continue
        # Distinct people on the same phone: their GIVEN names differ (spouses, parent/child, twins).
        named = [p[2] for p in profiles if p[2]]
        if any(not _givens_compatible(named[x], named[y]) for x in range(len(named)) for y in range(x + 1, len(named))):
            shared.add(ph)
    return shared


def compute_token_idf(evs: list[RecordEvidence]) -> dict[str, float]:
    counts: Counter[str] = Counter()
    n = max(1, len(evs))
    for ev in evs:
        seen = {t for fn in ev.full_names for t in fn}
        for t in seen:
            counts[t] += 1
    return {t: min(6.5, max(1.0, math.log2(n / max(1, c)))) for t, c in counts.items()}


def _id_prefix(uid: str) -> str:
    m = re.match(r"[a-z]+", uid)
    return m.group(0) if m else ""


def conflicting_system_ids(a: RecordEvidence, b: RecordEvidence) -> bool:
    """Different registration numbers from the same hospital system, with nothing strong in common."""
    ra, rb = _registry_ids(a), _registry_ids(b)
    if not ra or not rb or (a.uhids & b.uhids) or (a.uhid_cores & b.uhid_cores):
        return False
    pa = {_id_prefix(u) for u in ra} - {""}
    pb = {_id_prefix(u) for u in rb} - {""}
    if not (pa & pb):
        return False
    strong = (a.aadhaars & b.aadhaars) or (a.abhas & b.abhas) or (a.pans & b.pans) or (a.emails & b.emails) \
        or (a.ins_ids & b.ins_ids) or (bool(a.dobs) and bool(b.dobs) and dobs_compatible(a.dobs, b.dobs))
    return not strong


def strong_shared(a: RecordEvidence, b: RecordEvidence, shared_phones: set[str]) -> bool:
    """A shared identifier that belongs to one person (family phones excluded)."""
    if (a.aadhaars & b.aadhaars) or (a.abhas & b.abhas) or (a.pans & b.pans) or (a.emails & b.emails):
        return True
    if (a.ins_ids & b.ins_ids) or (a.uhids & b.uhids):
        return True
    return any(ph not in shared_phones for ph in (a.phones & b.phones))


_GUARDIAN_TYPES = {"S/o", "D/o", "W/o", "C/O"}


def rel_conflict(names_a: list[tuple[str, ...]], names_b: list[tuple[str, ...]]) -> bool:
    """Relatives of the same role that are clearly different people.

    Relatives are usually written as a given name alone ("W/o Salim" vs "W/o Dinesh"), so unlike patient
    names a single-token mismatch is a conflict too. Initials ("R. K.") are undecidable, never a conflict."""
    _, matched, conf = _name_pair_match(names_a, names_b)
    if conf:
        return True
    if matched or not names_a or not names_b:
        return False
    toks = {t for n in names_a for t in n} | {t for n in names_b for t in n}
    return bool(toks) and all(len(t) >= 3 for t in toks)


def rel_types_comparable(ta: str | None, tb: str | None) -> bool:
    """Same relation type, or a structured care-of (the guardian) against S/o, D/o or W/o."""
    if not ta or not tb or ta == "*" or tb == "*":
        return False
    return ta == tb or ("C/O" in (ta, tb) and ta in _GUARDIAN_TYPES and tb in _GUARDIAN_TYPES)


def independent_strong_ids(a: RecordEvidence, b: RecordEvidence, shared_phones: set[str]) -> int:
    """How many independent single-person identifier types agree (family phones excluded)."""
    n = sum(bool(x) for x in (
        a.aadhaars & b.aadhaars, a.abhas & b.abhas, a.pans & b.pans, a.emails & b.emails,
        a.ins_ids & b.ins_ids, a.uhids & b.uhids,
    ))
    if any(ph not in shared_phones for ph in (a.phones & b.phones)):
        n += 1
    return n


def has_hard_veto(a: RecordEvidence, b: RecordEvidence, soft: bool = True) -> bool:
    """Hard vetoes always apply. With soft=True also the error-prone ones (birth year, name or relative
    spelling), which score_pair waives when two independent single-person identifiers agree."""
    if a.gender and b.gender and a.gender != b.gender:
        return True
    if conflicting_system_ids(a, b):
        return True
    if a.dobs and b.dobs and not dobs_compatible(a.dobs, b.dobs):
        return True
    if a.aadhaars and b.aadhaars and not (a.aadhaars & b.aadhaars):
        return True
    if a.aadhaar_last4 and b.aadhaar_last4 and not (a.aadhaar_last4 & b.aadhaar_last4):
        return True
    if a.abhas and b.abhas and not (a.abhas & b.abhas):
        return True
    if a.pans and b.pans and not (a.pans & b.pans):
        return True
    if not soft:
        return False
    if a.birth_years and b.birth_years:
        min_diff = min(abs(ya - yb) for ya in a.birth_years for yb in b.birth_years)
        if min_diff > 4:
            return True
    if a.rel_names and b.rel_names and rel_types_comparable(a.rel_type, b.rel_type):
        if rel_conflict(a.rel_names, b.rel_names):
            return True
    _, _, name_conf = _name_pair_match(a.full_names, b.full_names)
    if name_conf:
        return True
    return False


def score_pair(
    a: RecordEvidence,
    b: RecordEvidence,
    shared_phones: set[str],
    idf: dict[str, float],
) -> float:
    """Return log-likelihood score (or -999.0 if vetoed)."""
    if a.is_multi_patient_summary or b.is_multi_patient_summary:
        return -999.0
    if has_hard_veto(a, b):
        # a corrupted age cell or a transliteration variant must not split records that agree on
        # two independent single-person identifiers (e.g. exact UHID + personal phone)
        if has_hard_veto(a, b, soft=False) or independent_strong_ids(a, b, shared_phones) < 2:
            return -999.0

    score = 0.0

    # 1. National / hospital strong IDs
    if a.aadhaars & b.aadhaars:
        score += 14.0
    if a.abhas & b.abhas:
        score += 14.0
    if a.pans & b.pans:
        score += 12.0
    if a.emails & b.emails:
        score += 12.0
    if a.ins_ids & b.ins_ids:
        score += 10.0
    if (a.uhids & b.uhids) or (a.uhid_cores & b.uhid_cores):
        score += 11.5
    if a.aadhaar_last4 & b.aadhaar_last4:
        score += 4.0

    # 2. Phone (with shared family phone downgrade!)
    shared_ph = a.phones & b.phones
    if shared_ph:
        if all(ph in shared_phones for ph in shared_ph):
            score += 3.5
        else:
            score += 9.5

    # 3. Name similarity weighted by empirical rarity (u-probability)
    name_sim, matched_toks, _ = _name_pair_match(a.full_names, b.full_names)
    if matched_toks >= 2 and name_sim >= 0.88:
        common_toks = {t for fn in a.full_names for t in fn} & {t for fn in b.full_names for t in fn}
        rarity_boost = sum(idf.get(t, 3.0) for t in common_toks if t not in GENERIC_WEAK_TOKENS)
        score += max(5.5, min(11.0, rarity_boost))
    elif matched_toks == 1 and name_sim >= 0.80:
        # Single-token match ("Asha" vs "Asha Rajput", "Shahid" vs "Mohammad Shahid")
        # Not enough alone -> requires corroborating clue (relative, phone, DOB, village+age)
        single_toks = {t for fn in a.full_names for t in fn} & {t for fn in b.full_names for t in fn}
        best_idf = max((idf.get(t, 2.5) for t in single_toks if t not in GENERIC_WEAK_TOKENS), default=1.5)
        score += min(4.2, 1.5 + 0.5 * best_idf)

    # 4. Relative name
    if a.rel_names and b.rel_names:
        rel_sim, rel_cnt, _ = _name_pair_match(a.rel_names, b.rel_names)
        if rel_cnt >= 1 and rel_sim >= 0.88:
            score += 6.2 if rel_cnt >= 2 else 4.8

    # 5. DOB / Birth Year / Gender / Geo / Clinical Continuity
    if a.dobs & b.dobs:
        score += 8.5
    elif a.dobs and b.dobs and dobs_compatible(a.dobs, b.dobs):
        score += 3.0     # day/month swapped between exports
    elif a.birth_years and b.birth_years:
        min_diff = min(abs(ya - yb) for ya in a.birth_years for yb in b.birth_years)
        if min_diff <= 1:
            score += 2.2
        elif min_diff <= 3:
            score += 0.8

    if a.gender and b.gender and a.gender == b.gender:
        score += 1.0

    if a.villages & b.villages:
        score += 4.5
    if a.pins & b.pins:
        score += 2.5
    elif a.districts & b.districts:
        score += 1.2

    if a.diagnoses & b.diagnoses:
        score += 1.5

    return score

"""Two-stage cross-script Fellegi-Sunter scorer + family phone & hard contradiction vetoes."""
from __future__ import annotations

import math
from collections import Counter, defaultdict

from rapidfuzz.distance import JaroWinkler

from src.linkage.evidence import RecordEvidence
from src.nlp.indic_norm import GENERIC_WEAK_TOKENS, phonetic_skeleton


def _tok_sim(a: str, b: str) -> float:
    if a == b:
        return 1.0
    # Initial match: "r" matches "ramesh" / "m" matches "mohammad"
    if (len(a) == 1 and b.startswith(a)) or (len(b) == 1 and a.startswith(b)):
        return 0.88
    jw = JaroWinkler.normalized_similarity(a, b)
    if jw >= 0.80 and phonetic_skeleton(a) == phonetic_skeleton(b):
        return max(jw, 0.90)
    return jw


def _name_pair_match(names_a: list[tuple[str, ...]], names_b: list[tuple[str, ...]]) -> tuple[float, int, bool]:
    """Return (best_similarity, matched_token_count, explicit_full_conflict)."""
    if not names_a or not names_b:
        return 0.0, 0, False
    best_sim = 0.0
    best_matched = 0
    for ta in names_a:
        for tb in names_b:
            if not ta or not tb:
                continue
            sa, sb = (ta, tb) if len(ta) <= len(tb) else (tb, ta)
            used = set()
            sims = []
            for wa in sa:
                best_j, best_s = -1, 0.0
                for j, wb in enumerate(sb):
                    if j in used:
                        continue
                    s = _tok_sim(wa, wb)
                    if s > best_s:
                        best_s, best_j = s, j
                if best_s >= 0.88 and best_j >= 0:
                    used.add(best_j)
                    sims.append(best_s)
            matched = len(sims)
            if matched == len(sa) and len(sa) >= 2:
                avg = sum(sims) / matched
                if avg > best_sim:
                    best_sim = avg
                    best_matched = matched
            elif matched >= 1 and (len(sa) == 1 or len(sb) == 1):
                # One side only has single token (e.g. "Shahid" vs "Mohammad Shahid", or "Asha" vs "Asha Rajput")
                avg = sum(sims) / matched * 0.92
                if avg > best_sim:
                    best_sim = avg
                    best_matched = matched
    # Check explicit conflict when both sides have >=2 tokens and none/few match
    conflict = False
    if best_matched == 0 and any(len(t) >= 2 for t in names_a) and any(len(t) >= 2 for t in names_b):
        conflict = True
    elif best_matched == 1 and any(len(t) >= 2 for t in names_a) and any(len(t) >= 2 for t in names_b):
        # E.g. "Sunita Verma" vs "Sunita Patel" -> 1 shared first name, different surnames!
        # Unless one surname is generic weak token missing on the other, flag conflict
        s_a = {t for fn in names_a if len(fn) >= 2 for t in fn[1:]}
        s_b = {t for fn in names_b if len(fn) >= 2 for t in fn[1:]}
        if s_a and s_b and not any(_tok_sim(x, y) >= 0.88 for x in s_a for y in s_b):
            conflict = True
    return best_sim, best_matched, conflict


def detect_shared_family_phones(evs: list[RecordEvidence]) -> set[str]:
    """Identify phones shared by distinct individuals (attendant/family phones)."""
    by_ph: dict[str, list[RecordEvidence]] = defaultdict(list)
    for ev in evs:
        for ph in ev.phones:
            by_ph[ph].append(ev)
    shared: set[str] = set()
    for ph, group in by_ph.items():
        genders = {ev.gender for ev in group if ev.gender}
        if len(genders) > 1:
            shared.add(ph)
            continue
        by_years = sorted({y for ev in group for y in ev.birth_years})
        if len(by_years) >= 2 and (by_years[-1] - by_years[0]) > 5:
            shared.add(ph)
            continue
        # Distinct full names on same phone
        multi_names = [ev.full_names for ev in group if any(len(fn) >= 2 for fn in ev.full_names)]
        if len(multi_names) >= 2:
            base = multi_names[0]
            for other in multi_names[1:]:
                _, _, conf = _name_pair_match(base, other)
                if conf:
                    shared.add(ph)
                    break
    return shared


def compute_token_idf(evs: list[RecordEvidence]) -> dict[str, float]:
    counts: Counter[str] = Counter()
    n = max(1, len(evs))
    for ev in evs:
        seen = {t for fn in ev.full_names for t in fn}
        for t in seen:
            counts[t] += 1
    return {t: min(6.5, max(1.0, math.log2(n / max(1, c)))) for t, c in counts.items()}


def has_hard_veto(a: RecordEvidence, b: RecordEvidence) -> bool:
    if a.gender and b.gender and a.gender != b.gender:
        return True
    if a.dobs and b.dobs and not (a.dobs & b.dobs):
        return True
    if a.aadhaars and b.aadhaars and not (a.aadhaars & b.aadhaars):
        return True
    if a.aadhaar_last4 and b.aadhaar_last4 and not (a.aadhaar_last4 & b.aadhaar_last4):
        return True
    if a.abhas and b.abhas and not (a.abhas & b.abhas):
        return True
    if a.pans and b.pans and not (a.pans & b.pans):
        return True
    if a.birth_years and b.birth_years:
        min_diff = min(abs(ya - yb) for ya in a.birth_years for yb in b.birth_years)
        if min_diff > 4:
            return True
    if a.rel_names and b.rel_names and a.rel_type and b.rel_type and a.rel_type == b.rel_type:
        _, _, rel_conf = _name_pair_match(a.rel_names, b.rel_names)
        if rel_conf:
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

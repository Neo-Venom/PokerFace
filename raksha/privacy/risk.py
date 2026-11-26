"""M3 Requirement 4: Runtime risk assessment & adaptive quasi-identifier generalization ladder."""
from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass

from raksha.detect.structured import _is_fhir_bundle, infer_column_type
from raksha.identity.evidence import RecordEvidence, get_json_record_units
from raksha.identity.linker import ClusterResult
from raksha.privacy.dates import keyed_shift_days

_DEVA_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
_NON_PATIENT_PATH_TOKS = ("clean/",)

# Empirical UP rural/district population pyramid relative probability per single year of age (sums ~ 1.0 over 0..100)
def _year_density(age: int) -> float:
    a = max(0, min(100, int(age)))
    if a < 5:
        return 0.011
    if a < 60:
        return 0.0135
    if a < 70:
        return 0.0090
    if a < 80:
        return 0.0055
    if a < 90:
        return 0.0025
    return 0.0009


def _band_density(lo: int, hi: int) -> float:
    return sum(_year_density(y) for y in range(max(0, lo), min(100, hi) + 1))


@dataclass(frozen=True)
class PolicySettings:
    age_band_years: int = 5
    date_shift_max_days: int = 180
    date_shift_min_days: int = 15
    min_group_size: int = 5
    adaptive_risk: bool = True
    keep_tehsil_default: bool = False
    keep_district_default: bool = True
    keep_gender_default: bool = True
    keep_fhir_birthdate: bool = False
    max_age_width: int = 20

    @classmethod
    def from_policy(cls, policy_dict: dict) -> PolicySettings:
        def _bool(k: str, dflt: bool) -> bool:
            v = policy_dict.get(k, dflt)
            if isinstance(v, bool):
                return v
            return str(v).strip().casefold() in ("1", "true", "yes", "on")

        return cls(
            age_band_years=int(policy_dict.get("age_band_years", 5)),
            date_shift_max_days=int(policy_dict.get("date_shift_max_days", 180)),
            date_shift_min_days=int(policy_dict.get("date_shift_min_days", 15)),
            min_group_size=int(policy_dict.get("min_group_size", 5)),
            adaptive_risk=_bool("adaptive_risk", True),
            keep_tehsil_default=_bool("keep_tehsil_default", False),
            keep_district_default=_bool("keep_district_default", True),
            keep_gender_default=_bool("keep_gender_default", True),
            keep_fhir_birthdate=_bool("keep_fhir_birthdate", False),
            max_age_width=int(policy_dict.get("max_age_width", 20)),
        )


@dataclass
class PrivacyPlan:
    profile_key: str
    shift_days: int
    age_band_width: int        # 0 => suppress age completely ([AGE]); 5, 10, 20, 30
    top_age_cap: int | None    # e.g. 85 or 80 => "80+"
    keep_district: bool
    keep_tehsil: bool
    keep_gender: bool
    keep_fhir_birthdate: bool
    suppress_record: bool = False
    est_candidates: float = 0.0
    top1_risk: float = 0.0
    rare_cond: bool = False
    escalation_step: int = 0


@dataclass
class _ProfileQI:
    profile_key: str
    records: list[tuple[str, str]]
    ages: list[int]
    genders: set[str]
    districts: set[str]
    tehsils: set[str]
    conditions: set[str]


def is_non_patient_static_file(rel: str) -> bool:
    low = rel.casefold()
    return any(tok in low for tok in _NON_PATIENT_PATH_TOKS)


def _norm_tok(s: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", s).translate(_DEVA_DIGITS).casefold().split())


def _extract_conds_from_text(text: str, out: set[str]) -> None:
    for m in re.finditer(r"(?im)(?:Diagnosis|निदान|Dx|Procedure)\s*[:\-]\s*([^\n]+)", text):
        for seg in re.split(r"[;|]", m.group(1)):
            c = re.sub(r"\([^)]*\)", "", seg)
            c = re.sub(r"\b[A-Z]\d{2}(?:\.\d+)?\b", "", c)
            c = _norm_tok(c)
            if len(c) >= 4:
                out.add(c)


def _extract_geo_and_cond_for_record(
    rel: str,
    fmt: str,
    sub_obj: object,
    col_types: list[str | None] | None = None,
    header: list[str] | None = None,
) -> tuple[set[str], set[str], set[str]]:
    """Extract (districts, tehsils, conditions) purely from input record without touching external truth."""
    districts: set[str] = set()
    tehsils: set[str] = set()
    conds: set[str] = set()

    if fmt in ("csv", "tsv") and isinstance(sub_obj, list) and header is not None:
        for i, cell in enumerate(sub_obj):
            if i >= len(header):
                continue
            val = str(cell).strip()
            if not val:
                continue
            hl = header[i].casefold().strip()
            if hl in ("district", "jila", "distt"):
                districts.add(_norm_tok(val))
            elif hl == "tehsil":
                tehsils.add(_norm_tok(val))
            elif hl in ("procedure", "diagnosis"):
                c = _norm_tok(re.sub(r"\([^)]*\)", "", val))
                if len(c) >= 4:
                    conds.add(c)
            elif col_types and col_types[i] == "ADDRESS":
                for md in re.finditer(r"(?i)(?:Distt?\.?|District|जिला)\s*[:\-]?\s*([A-Za-z\u0900-\u097F]+(?:\s+[A-Za-z\u0900-\u097F]+)?)", val):
                    districts.add(_norm_tok(md.group(1)))
                for mt in re.finditer(r"(?i)(?:Tehsil|तहसील)\s*[:\-]?\s*([A-Za-z\u0900-\u097F]+(?:\s+[A-Za-z\u0900-\u097F]+)?)", val):
                    tehsils.add(_norm_tok(mt.group(1)))
            elif hl in ("notes", "remarks", "comment"):
                _extract_conds_from_text(val, conds)

    elif fmt == "json" and isinstance(sub_obj, dict):
        if _is_fhir_bundle(sub_obj):
            for ent in sub_obj.get("entry", []):
                res = ent.get("resource", {}) if isinstance(ent, dict) else {}
                rt = res.get("resourceType")
                if rt == "Patient":
                    for addr in res.get("address", []):
                        if isinstance(addr, dict):
                            if isinstance(addr.get("district"), str) and addr["district"].strip():
                                districts.add(_norm_tok(addr["district"]))
                            for ln in addr.get("line", []):
                                s = str(ln)
                                for md in re.finditer(r"(?i)(?:Distt?\.?|District|जिला)\s*[:\-]?\s*([A-Za-z\u0900-\u097F]+(?:\s+[A-Za-z\u0900-\u097F]+)?)", s):
                                    districts.add(_norm_tok(md.group(1)))
                                for mt in re.finditer(r"(?i)(?:Tehsil|तहसील)\s*[:\-]?\s*([A-Za-z\u0900-\u097F]+(?:\s+[A-Za-z\u0900-\u097F]+)?)", s):
                                    tehsils.add(_norm_tok(mt.group(1)))
                elif rt == "Condition":
                    cd = res.get("code", {})
                    if isinstance(cd, dict) and isinstance(cd.get("text"), str):
                        conds.add(_norm_tok(cd["text"]))
                    elif isinstance(cd, dict):
                        for item in cd.get("coding", []):
                            if isinstance(item, dict) and isinstance(item.get("display"), str):
                                conds.add(_norm_tok(item["display"]))
                elif rt == "Procedure":
                    cd = res.get("code", {})
                    if isinstance(cd, dict) and isinstance(cd.get("text"), str):
                        conds.add(_norm_tok(cd["text"]))
        else:
            stack = [sub_obj]
            while stack:
                nd = stack.pop()
                for k, v in nd.items():
                    kl = str(k).casefold()
                    if isinstance(v, dict):
                        stack.append(v)
                    elif isinstance(v, list):
                        if kl in ("diagnosis", "diagnoses"):
                            for item in v:
                                if isinstance(item, dict) and isinstance(item.get("display"), str):
                                    conds.add(_norm_tok(item["display"]))
                                elif isinstance(item, str) and len(item.strip()) >= 4:
                                    conds.add(_norm_tok(item))
                        else:
                            for it in v:
                                if isinstance(it, dict):
                                    stack.append(it)
                    elif isinstance(v, str) and v.strip():
                        if kl in ("district", "jila", "distt"):
                            districts.add(_norm_tok(v))
                        elif kl == "tehsil":
                            tehsils.add(_norm_tok(v))
                        elif kl in ("diagnosis", "procedure"):
                            conds.add(_norm_tok(v))
                        elif kl in ("line", "address_note") or "address" in kl:
                            for md in re.finditer(r"(?i)(?:Distt?\.?|District|जिला)\s*[:\-]?\s*([A-Za-z\u0900-\u097F]+(?:\s+[A-Za-z\u0900-\u097F]+)?)", v):
                                districts.add(_norm_tok(md.group(1)))
                            for mt in re.finditer(r"(?i)(?:Tehsil|तहसील)\s*[:\-]?\s*([A-Za-z\u0900-\u097F]+(?:\s+[A-Za-z\u0900-\u097F]+)?)", v):
                                tehsils.add(_norm_tok(mt.group(1)))

    elif fmt == "txt" and isinstance(sub_obj, str):
        for md in re.finditer(r"(?i)(?:Distt?\.?|District|जिला)\s*[:\-]?\s*([A-Za-z\u0900-\u097F]+(?:\s+[A-Za-z\u0900-\u097F]+)?)(?=\s*[-,\n]|$)", sub_obj):
            districts.add(_norm_tok(md.group(1)))
        for mt in re.finditer(r"(?i)(?:Tehsil|तहसील)\s*[:\-]?\s*([A-Za-z\u0900-\u097F]+(?:\s+[A-Za-z\u0900-\u097F]+)?)(?=\s*[,\n]|$)", sub_obj):
            tehsils.add(_norm_tok(mt.group(1)))
        _extract_conds_from_text(sub_obj, conds)

    return districts, tehsils, conds


def _compute_effective_age(ev: RecordEvidence) -> int | None:
    if ev.age is not None and 0 <= ev.age <= 115:
        return ev.age
    if ev.dob and len(ev.dob) >= 4 and ev.dob[:4].isdigit():
        by = int(ev.dob[:4])
        ref_y = ev.record_year or 2026
        a = ref_y - by
        if 0 <= a <= 115:
            return a
    return None


def assess_privacy_risks(
    parsed_items: list[tuple[str, str, object]],
    evidences: list[RecordEvidence],
    cluster_res: ClusterResult,
    cluster_to_psn: dict[int, str],
    policy: PolicySettings,
    shift_key: bytes,
) -> dict[tuple[str, str], PrivacyPlan]:
    """Compute per-profile & unlinked-record PrivacyPlans via empirical cohort + synthetic population risk."""
    ev_by_key = {(ev.rel, ev.loc): ev for ev in evidences}
    rec_to_psn = {
        k: cluster_to_psn[cid]
        for k, cid in cluster_res.record_to_cluster.items()
        if cid in cluster_to_psn
    }

    # Build raw record index for extra geo/cond clues
    item_map = {rel: (fmt, payload) for rel, fmt, payload in parsed_items}
    col_cache: dict[str, tuple[list[str], list[str | None]]] = {}
    for rel, fmt, payload in parsed_items:
        if fmt in ("csv", "tsv") and isinstance(payload, list) and len(payload) > 1:
            hdr, body = payload[0], payload[1:]
            cts = [infer_column_type([r[c] if c < len(r) else "" for r in body], header=h) for c, h in enumerate(hdr)]
            col_cache[rel] = (hdr, cts)

    # Group all patient-bearing records into profiles
    profiles: dict[str, _ProfileQI] = {}
    for rel, fmt, payload in parsed_items:
        if is_non_patient_static_file(rel):
            continue
        rec_units: list[tuple[str, object]] = []
        if fmt in ("csv", "tsv") and isinstance(payload, list) and len(payload) > 1:
            hdr_c, cts_c = col_cache.get(rel, ([], []))
            hdr_low = {str(h).strip().casefold() for h in hdr_c}
            has_qi_or_pt = any(
                ct in ("PATIENT_NAME", "AADHAAR", "ABHA_NUMBER", "PHONE", "DOB") for ct in cts_c
            ) or bool(hdr_low & {"age", "umar", "age_sex", "district", "jila", "tehsil", "admission_date", "discharge_date", "visit_date", "collected_on", "reg date", "registration date", "uhid", "patient_name"})
            if not has_qi_or_pt:
                continue
            rec_units = [(f"row:{i}", row) for i, row in enumerate(payload[1:])]
        elif fmt == "json":
            rec_units = get_json_record_units(payload)
        elif fmt == "txt":
            rec_units = [("doc", payload)]

        hdr, cts = col_cache.get(rel, (None, None))
        for loc, sub_obj in rec_units:
            ev = ev_by_key.get((rel, loc))
            ds, ts, cs = _extract_geo_and_cond_for_record(rel, fmt, sub_obj, col_types=cts, header=hdr)
            if fmt == "txt" and ev is None and not ds and not ts and not cs:
                continue
            psn = rec_to_psn.get((rel, loc))
            if psn:
                pkey = psn
            else:
                ch = ev.content_hash if ev else f"{rel}#{loc}"
                pkey = f"unlinked|{ch}"
            prof = profiles.get(pkey)
            if prof is None:
                prof = _ProfileQI(
                    profile_key=pkey,
                    records=[],
                    ages=[],
                    genders=set(),
                    districts=set(),
                    tehsils=set(),
                    conditions=set(),
                )
                profiles[pkey] = prof
            prof.records.append((rel, loc))
            if ev is not None:
                ea = _compute_effective_age(ev)
                if ea is not None:
                    prof.ages.append(ea)
                if ev.sex:
                    prof.genders.add(ev.sex)
            prof.districts.update(ds)
            prof.tehsils.update(ts)
            prof.conditions.update(cs)

    # Estimate population parameters strictly from batch distribution (never external truth)
    linked_profiles = [p for pk, p in profiles.items() if not pk.startswith("unlinked|")]
    n_linked = max(1, len(linked_profiles))
    # Synthetic population is ~4.5x linked patient cohort (minimum catchment floor 180)
    n_pop_est = max(180.0, 4.5 * n_linked)

    dist_counts: Counter[str] = Counter()
    tehsil_counts: Counter[str] = Counter()
    cond_counts: Counter[str] = Counter()
    for p in profiles.values():
        for d in p.districts:
            dist_counts[d] += 1
        for t in p.tehsils:
            tehsil_counts[t] += 1
        for c in p.conditions:
            cond_counts[c] += 1

    n_dists = max(1, len(dist_counts))
    n_tehsils = max(1, len(tehsil_counts))

    def _estimate_candidates(p: _ProfileQI, width: int, top_cap: int | None, keep_dist: bool, keep_teh: bool, keep_gen: bool) -> tuple[float, float, bool]:
        # Location marginal probability
        if keep_teh and p.tehsils:
            p_loc = min(0.08, 1.0 / max(10, n_tehsils * 2))
        elif keep_dist and p.districts:
            avg_d = sum((dist_counts[d] + 1) / (n_linked + n_dists) for d in p.districts) / len(p.districts)
            p_loc = min(0.25, max(0.06, avg_d))
        else:
            p_loc = 1.0

        # Gender marginal
        p_gen = 0.50 if (keep_gen and p.genders) else 1.0

        # Age band density
        if width == 0 or not p.ages:
            p_age = 1.0
        else:
            rep = int(round(sum(p.ages) / len(p.ages)))
            if top_cap is not None and rep >= top_cap:
                p_age = _band_density(top_cap, 100)
            else:
                lo = (rep // width) * width
                hi = min(100, lo + width - 1)
                p_age = _band_density(lo, hi)
                p_age = max(0.002, p_age)

        est_pop_cand = n_pop_est * p_loc * p_gen * p_age
        min_cf = min((cond_counts[c] for c in p.conditions), default=99)
        is_rare = bool(p.conditions and min_cf <= 3)

        # Rare conditions act as strong targeting vectors when narrow geo/age stays exposed
        if is_rare and (keep_dist or keep_teh or (width > 0 and width <= 10)):
            effective_cand = min(est_pop_cand, est_pop_cand * (0.45 if min_cf == 1 else 0.7))
        else:
            effective_cand = est_pop_cand

        return effective_cand, 1.0 / max(1.0, effective_cand), is_rare

    plans_by_rec: dict[tuple[str, str], PrivacyPlan] = {}
    for pkey, prof in profiles.items():
        s_days = keyed_shift_days(
            pkey,
            shift_key,
            max_days=policy.date_shift_max_days,
            min_days=policy.date_shift_min_days,
        )

        # Baseline settings
        width = policy.age_band_years
        top_cap: int | None = 85 if (prof.ages and max(prof.ages) >= 85 and policy.adaptive_risk) else None
        keep_dist = policy.keep_district_default
        keep_teh = policy.keep_tehsil_default
        keep_gen = policy.keep_gender_default
        keep_bdate = policy.keep_fhir_birthdate
        step = 0

        est_c, top1_r, is_rare = _estimate_candidates(prof, width, top_cap, keep_dist, keep_teh, keep_gen)
        target_k = float(policy.min_group_size)
        if is_rare and policy.adaptive_risk:
            target_k = max(target_k, 1.6 * policy.min_group_size)

        if policy.adaptive_risk and width > 0:
            # Escalation ladder (Req 4): tehsil -> age band (5 -> 10 -> 20) -> location (district) -> extra QIs (gender -> suppress age)
            if est_c < target_k and keep_teh:
                keep_teh = False
                step = 1
                est_c, top1_r, is_rare = _estimate_candidates(prof, width, top_cap, keep_dist, keep_teh, keep_gen)

            if est_c < target_k and width < 10 and policy.max_age_width >= 10:
                width = 10
                if prof.ages and max(prof.ages) >= 80:
                    top_cap = 80
                step = 2
                est_c, top1_r, is_rare = _estimate_candidates(prof, width, top_cap, keep_dist, keep_teh, keep_gen)

            if est_c < target_k and width < 20 and policy.max_age_width >= 20:
                width = 20
                if prof.ages and max(prof.ages) >= 80:
                    top_cap = 80
                step = 3
                est_c, top1_r, is_rare = _estimate_candidates(prof, width, top_cap, keep_dist, keep_teh, keep_gen)

            if est_c < target_k and keep_dist:
                keep_dist = False
                step = 4
                est_c, top1_r, is_rare = _estimate_candidates(prof, width, top_cap, keep_dist, keep_teh, keep_gen)

            if est_c < target_k and keep_gen:
                keep_gen = False
                step = 5
                est_c, top1_r, is_rare = _estimate_candidates(prof, width, top_cap, keep_dist, keep_teh, keep_gen)

            if est_c < target_k:
                # Coarsen age completely for extreme-rare outliers
                width = 0
                step = 6
                est_c, top1_r, is_rare = _estimate_candidates(prof, width, top_cap, keep_dist, keep_teh, keep_gen)

        if width != policy.age_band_years or not keep_dist:
            keep_bdate = False

        plan = PrivacyPlan(
            profile_key=pkey,
            shift_days=s_days,
            age_band_width=width,
            top_age_cap=top_cap,
            keep_district=keep_dist,
            keep_tehsil=keep_teh,
            keep_gender=keep_gen,
            keep_fhir_birthdate=keep_bdate,
            suppress_record=False,
            est_candidates=round(est_c, 2),
            top1_risk=round(top1_r, 4),
            rare_cond=is_rare,
            escalation_step=step,
        )
        for rec_key in prof.records:
            plans_by_rec[rec_key] = plan

    return plans_by_rec

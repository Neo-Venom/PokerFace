"""M2 Steps 3-6: Multi-pass blocking, Fellegi-Sunter scoring, and order-independent constrained clustering."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from raksha.identity.evidence import RecordEvidence
from raksha.identity.name_compare import compare_name_lists, compare_relatives

CORE_MERGE_THRESHOLD = 7.5
FRAGMENT_THRESHOLD = 5.5
FRAGMENT_MARGIN = 2.5


@dataclass
class ClusterResult:
    # Map (rel, loc) -> cluster_id (int) for confidently linked / identifiable records; missing/null for ambiguous
    record_to_cluster: dict[tuple[str, str], int]
    clusters: dict[int, list[RecordEvidence]]


def _has_hard_id_conflict(a: RecordEvidence, b: RecordEvidence) -> bool:
    """Exclusive government/personal IDs (Aadhaar, ABHA, PAN) cannot conflict for the same person."""
    if a.aadhaars and b.aadhaars and a.aadhaars.isdisjoint(b.aadhaars):
        return True
    a14 = {x for x in a.abhas if x.isdigit()}
    b14 = {x for x in b.abhas if x.isdigit()}
    if a14 and b14 and a14.isdisjoint(b14):
        return True
    if a.pans and b.pans and a.pans.isdisjoint(b.pans):
        return True
    if a.aadhaar_last4 and b.aadhaar_last4 and a.aadhaar_last4.isdisjoint(b.aadhaar_last4):
        return True
    return False


def _age_interval_at_year(ev: RecordEvidence, ref_year: int) -> tuple[int, int] | None:
    if ev.age is not None:
        if ev.record_year is not None and ev.record_year != ref_year:
            adj = ev.age + (ref_year - ev.record_year)
            return (adj - 1, adj + 1)
        return (ev.age, ev.age)
    if ev.dob and len(ev.dob) >= 4 and ev.dob[:4].isdigit():
        by = int(ev.dob[:4])
        # Age in ref_year is (ref_year - by - 1) before birthday or (ref_year - by) on/after birthday
        return (ref_year - by - 1, ref_year - by)
    return None


def _age_diff(a: RecordEvidence, b: RecordEvidence) -> int | None:
    if a.age is None and b.age is None:
        return None
    ref_year = b.record_year if (b.age is not None and b.record_year is not None) else (a.record_year or b.record_year)
    if ref_year is None:
        if a.age is not None and b.age is not None:
            return abs(a.age - b.age)
        return None
    ia = _age_interval_at_year(a, ref_year)
    ib = _age_interval_at_year(b, ref_year)
    if ia is None or ib is None:
        return None
    if ia[1] < ib[0]:
        return ib[0] - ia[1]
    if ib[1] < ia[0]:
        return ia[0] - ib[1]
    return 0


def score_pair(a: RecordEvidence, b: RecordEvidence) -> tuple[float, bool]:
    """Compute Fellegi-Sunter match weight and hard-conflict flag between two records."""
    if _has_hard_id_conflict(a, b):
        return -50.0, True

    w = 0.0
    strong_id_match = False

    # 1. Strong exclusive IDs
    if a.aadhaars and b.aadhaars and not a.aadhaars.isdisjoint(b.aadhaars):
        w += 14.0
        strong_id_match = True
    if a.abhas and b.abhas and not a.abhas.isdisjoint(b.abhas):
        w += 14.0
        strong_id_match = True
    if a.pans and b.pans and not a.pans.isdisjoint(b.pans):
        w += 12.0
        strong_id_match = True
    if a.emails and b.emails and not a.emails.isdisjoint(b.emails):
        w += 11.0
        strong_id_match = True
    if a.insurances and b.insurances and not a.insurances.isdisjoint(b.insurances):
        w += 10.0
        strong_id_match = True
    if a.aadhaar_last4 and b.aadhaar_last4 and not a.aadhaar_last4.isdisjoint(b.aadhaar_last4):
        w += 2.5

    # 2. Hospital MRN / UHID (scoped per hospital OR hospital-prefixed global UHID)
    same_scoped_mrn = not a.scoped_mrns.isdisjoint(b.scoped_mrns)
    same_global_mrn = not a.global_mrns.isdisjoint(b.global_mrns)
    a_mrn_vals = {m for _, m in a.scoped_mrns}
    b_mrn_vals = {m for _, m in b.scoped_mrns}
    same_generic_mrn_val = bool(a_mrn_vals and b_mrn_vals and not a_mrn_vals.isdisjoint(b_mrn_vals))

    # 3. Phone (WEAK: shared by relatives!)
    if a.phones and b.phones:
        if not a.phones.isdisjoint(b.phones):
            w += 3.0
        else:
            w -= 1.0

    # 4. Name comparison across separate channels
    ncmp = compare_name_lists(a.names, b.names)
    lvl = ncmp.best_level
    if lvl == "exact":
        w += 5.5
    elif lvl == "fuzzy":
        w += 4.2
    elif lvl == "phonetic":
        w += 3.5
    elif lvl == "given_only":
        w += 2.0
    elif lvl == "disagree":
        w -= 10.0

    if same_scoped_mrn or same_global_mrn:
        w += 12.0
        strong_id_match = True
    elif same_generic_mrn_val and lvl in ("exact", "fuzzy", "phonetic"):
        # Generic UHID-xxxx shared with matching full name (e.g., registration <-> external lab)
        w += 3.5
        strong_id_match = True
    elif a.hospital and b.hospital and a.hospital == b.hospital and a.scoped_mrns and b.scoped_mrns and not same_generic_mrn_val:
        # Different ID at same hospital can happen on re-registration (mild negative, not veto)
        w -= 1.5

    # 5. Relative name & relation
    rel_cmp = compare_relatives(a.relatives, b.relatives)
    if rel_cmp == "exact":
        w += 4.5
    elif rel_cmp == "first":
        w += 3.0
    elif rel_cmp == "disagree":
        w -= 5.5

    # 6. DOB & Age consistency
    yr_diff: int | None = None
    if a.dob and b.dob:
        if a.dob == b.dob:
            w += 5.5
        else:
            w -= 7.5
        if len(a.dob) >= 4 and len(b.dob) >= 4 and a.dob[:4].isdigit() and b.dob[:4].isdigit():
            yr_diff = abs(int(a.dob[:4]) - int(b.dob[:4]))
    else:
        ad = _age_diff(a, b)
        if ad is not None:
            yr_diff = ad
            if ad <= 2:
                w += 2.0
            elif ad >= 5:
                w -= 6.5

    # 7. Sex consistency
    if a.sex and b.sex:
        if a.sex == b.sex:
            w += 1.0
        else:
            w -= 7.0

    # 8. Village & PIN
    vill_match = bool(
        (a.villages and b.villages and not a.villages.isdisjoint(b.villages))
        or (a.village_phonetics and b.village_phonetics and not a.village_phonetics.isdisjoint(b.village_phonetics))
    )
    pin_match = bool(a.pins and b.pins and not a.pins.isdisjoint(b.pins))
    if vill_match:
        w += 2.5
    elif a.villages and b.villages:
        w -= 3.5
    if pin_match:
        w += 1.5
    elif a.pins and b.pins and not vill_match:
        w -= 1.5

    # Guard: strong exclusive government IDs override minor name spelling mismatch,
    # whereas non-government MRN/email alone cannot merge records whose full multi-token names disagree.
    has_exclusive_govt_match = bool(
        (a.aadhaars and b.aadhaars and not a.aadhaars.isdisjoint(b.aadhaars))
        or (a.abhas and b.abhas and not a.abhas.isdisjoint(b.abhas))
        or (a.pans and b.pans and not a.pans.isdisjoint(b.pans))
    )
    both_full_names = any(len(n.norm_tokens) >= 2 and not n.is_fragment_first_only for n in a.names) and any(
        len(n.norm_tokens) >= 2 and not n.is_fragment_first_only for n in b.names
    )
    if strong_id_match and lvl == "disagree" and not has_exclusive_govt_match and both_full_names:
        w -= 5.0

    # Generational (father/son, mother/daughter) and twin/sibling anti-merge guards when no exclusive ID matches:
    if not has_exclusive_govt_match and not same_scoped_mrn and not same_global_mrn:
        if yr_diff is not None and yr_diff >= 10:
            return -50.0, True
        if lvl == "disagree" and both_full_names:
            return -50.0, True
        if lvl == "weak" and both_full_names:
            from rapidfuzz import fuzz as _fuzz
            from raksha.identity.name_compare import _vowel_fold

            any_first_ok = False
            for na in a.names:
                for nb in b.names:
                    if not na.norm_tokens or not nb.norm_tokens:
                        continue
                    t1, t2 = na.norm_tokens[0], nb.norm_tokens[0]
                    p1, p2 = na.phonetic_tokens[0], nb.phonetic_tokens[0]
                    if (
                        t1 == t2
                        or p1 == p2
                        or _vowel_fold(p1) == _vowel_fold(p2)
                        or _fuzz.ratio(t1, t2) >= 85.0
                        or (p1[0] == p2[0] and _fuzz.ratio(p1, p2) >= 80.0)
                    ):
                        any_first_ok = True
                        break
                if any_first_ok:
                    break
            if not any_first_ok:
                return -50.0, True

    return w, False


def _build_candidate_pairs(evidences: list[RecordEvidence]) -> set[tuple[int, int]]:
    """Pass 1: Multi-pass blocking (union of candidate sets across 5 passes)."""
    blocks: dict[str, list[int]] = defaultdict(list)

    for idx, ev in enumerate(evidences):
        # Block 1: Strong IDs (Aadhaar, ABHA, PAN, Email, Insurance, Scoped/Global MRN)
        for val in ev.aadhaars:
            blocks[f"aad:{val}"].append(idx)
        for val in ev.abhas:
            blocks[f"abh:{val}"].append(idx)
        for val in ev.pans:
            blocks[f"pan:{val}"].append(idx)
        for val in ev.emails:
            blocks[f"eml:{val}"].append(idx)
        for val in ev.insurances:
            blocks[f"ins:{val}"].append(idx)
        for hosp, mrn in ev.scoped_mrns:
            blocks[f"smrn:{hosp}:{mrn}"].append(idx)
        for mrn in ev.global_mrns:
            blocks[f"gmrn:{mrn}"].append(idx)

        # Block 2: Transliterated full name phonetic key (and sorted tokens)
        from raksha.identity.name_compare import _vowel_fold

        for nf in ev.names:
            if len(nf.phonetic_tokens) >= 2:
                v_toks = tuple(_vowel_fold(p) for p in nf.phonetic_tokens)
                blocks[f"fn:{'_'.join(v_toks)}"].append(idx)
                blocks[f"fns:{'_'.join(sorted(v_toks))}"].append(idx)
                # Also first-letter + vowel-folded surname for initials/cross-script borrowings
                blocks[f"init:{v_toks[0][0]}_{v_toks[-1]}"].append(idx)

            first_p = nf.phonetic_tokens[0]
            if first_p in ("md", "mohd") and len(nf.phonetic_tokens) >= 2:
                first_p = nf.phonetic_tokens[1]

            # Block 3: Name first token + Relative first token
            for rel in ev.relatives:
                if rel.phonetic_tokens:
                    blocks[f"nrel:{first_p}_{rel.phonetic_tokens[0]}"].append(idx)

            # Block 4: Name first token + approx age bucket (overlapping bins for +-2 drift)
            if ev.age is not None:
                for b in ((ev.age - 2) // 4, ev.age // 4, (ev.age + 2) // 4):
                    blocks[f"nage:{first_p}_{b}"].append(idx)

            # Block 5: Phone + Name first token
            for ph in ev.phones:
                blocks[f"phn:{ph}_{first_p}"].append(idx)

        # Also block on phone alone ONLY as candidate generator (Fellegi-Sunter filters relatives!)
        for ph in ev.phones:
            blocks[f"ph:{ph}"].append(idx)

    pairs: set[tuple[int, int]] = set()
    for bkey, members in blocks.items():
        is_strong_blk = bkey.startswith(("aad:", "abh:", "pan:", "eml:", "ins:", "smrn:", "gmrn:"))
        if len(members) < 2:
            continue
        if is_strong_blk:
            if len(members) <= 250:
                target_members = members
                for i_pos in range(len(target_members)):
                    u = target_members[i_pos]
                    for j_pos in range(i_pos + 1, len(target_members)):
                        v = target_members[j_pos]
                        if u != v:
                            pairs.add((min(u, v), max(u, v)))
            elif len(members) <= 2000:
                # Star-hub connection from first 4 members so large repeated strong-ID histories still link
                for i_pos in range(min(4, len(members))):
                    u = members[i_pos]
                    for j_pos in range(i_pos + 1, len(members)):
                        v = members[j_pos]
                        if u != v:
                            pairs.add((min(u, v), max(u, v)))
            continue

        if len(members) <= 50:
            target_members = members
        else:
            # Compress by strong-ID/MRN signature so high-visit-count patients (>50 rows) still bridge across datasets
            seen_sigs: set[tuple] = set()
            target_members = []
            for idx_m in members:
                ev_m = evidences[idx_m]
                sig = (
                    tuple(sorted(ev_m.aadhaars)),
                    tuple(sorted(ev_m.abhas)),
                    tuple(sorted(ev_m.pans)),
                    tuple(sorted(ev_m.emails)),
                    tuple(sorted(ev_m.insurances)),
                    tuple(sorted(ev_m.global_mrns)),
                    tuple(sorted(ev_m.scoped_mrns)),
                )
                if any(sig):
                    if sig in seen_sigs:
                        continue
                    seen_sigs.add(sig)
                target_members.append(idx_m)
            if len(target_members) > 80:
                continue

        for i_pos in range(len(target_members)):
            u = target_members[i_pos]
            for j_pos in range(i_pos + 1, len(target_members)):
                v = target_members[j_pos]
                if u != v:
                    pairs.add((min(u, v), max(u, v)))
    return pairs


def resolve_clusters(evidences: list[RecordEvidence]) -> ClusterResult:
    """Cluster records deterministically (order-independent) with strict anti-false-merge constraints."""
    if not evidences:
        return ClusterResult(record_to_cluster={}, clusters={})

    # Canonicalize record ordering internally by content_hash so algorithm behavior never depends on file/row input order
    order_idx = sorted(range(len(evidences)), key=lambda i: (evidences[i].content_hash, evidences[i].rel, evidences[i].loc))
    ordered_evs = [evidences[i] for i in order_idx]
    n = len(ordered_evs)

    # Separate non-fragmentary core records vs fragmentary notes (fragmentary attach in Stage B)
    is_frag = [ev.is_fragmentary for ev in ordered_evs]
    candidate_pairs = _build_candidate_pairs(ordered_evs)

    pair_cache: dict[tuple[int, int], tuple[float, bool]] = {}
    scored_edges: list[tuple[float, str, str, int, int]] = []

    for i, j in candidate_pairs:
        if is_frag[i] or is_frag[j]:
            continue
        sc, conflict = score_pair(ordered_evs[i], ordered_evs[j])
        pair_cache[(i, j)] = (sc, conflict)
        if not conflict and sc >= CORE_MERGE_THRESHOLD:
            hi, hj = ordered_evs[i].content_hash, ordered_evs[j].content_hash
            lo_h, hi_h = (hi, hj) if hi <= hj else (hj, hi)
            scored_edges.append((sc, lo_h, hi_h, i, j))

    # Deterministic descending score sort with content-hash tie-breakers (order-independent!)
    scored_edges.sort(key=lambda x: (-x[0], x[1], x[2]))

    # Union-find + explicit member lists for all-pairs cluster validation
    parent = list(range(n))
    members: dict[int, list[int]] = {i: [i] for i in range(n)}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def can_merge_clusters(ra: int, rb: int) -> bool:
        ma, mb = members[ra], members[rb]
        total_cross = 0.0
        count_cross = 0
        min_cross = 999.0
        max_cross = -999.0
        for u in ma:
            for v in mb:
                key = (min(u, v), max(u, v))
                if key not in pair_cache:
                    pair_cache[key] = score_pair(ordered_evs[u], ordered_evs[v])
                sc_uv, conf_uv = pair_cache[key]
                if conf_uv or sc_uv < -2.0:
                    return False
                min_cross = min(min_cross, sc_uv)
                max_cross = max(max_cross, sc_uv)
                total_cross += sc_uv
                count_cross += 1
        mean_cross = total_cross / max(1, count_cross)
        # Allow bridge merge when ZERO member pairs disagree on any non-missing field; else require strong ID bridge
        if min_cross >= 0.0 and max_cross >= CORE_MERGE_THRESHOLD:
            return True
        return max_cross >= 11.0 or mean_cross >= 5.0

    for _sc, _h1, _h2, i, j in scored_edges:
        ra, rb = find(i), find(j)
        if ra == rb:
            continue
        if can_merge_clusters(ra, rb):
            # Keep deterministic root by min content_hash
            if ordered_evs[ra].content_hash > ordered_evs[rb].content_hash:
                ra, rb = rb, ra
            parent[rb] = ra
            members[ra].extend(members[rb])
            del members[rb]

    # Build core cluster profiles (only clusters with non-fragmentary records that have identity evidence)
    core_roots = sorted(
        [r for r in members if not is_frag[r]],
        key=lambda r: ordered_evs[r].content_hash,
    )

    # Stage B: Attach fragmentary records ONLY if exactly one core cluster scores >= threshold with clear margin
    for f_idx in range(n):
        if not is_frag[f_idx]:
            continue
        frag_ev = ordered_evs[f_idx]
        cluster_scores: list[tuple[float, int]] = []
        for r in core_roots:
            # Evaluate fragmentary record against EVERY member in cluster `r`
            best_m = -999.0
            conflict_any = False
            for m_idx in members[r]:
                sc_fm, conf_fm = score_pair(frag_ev, ordered_evs[m_idx])
                if conf_fm or sc_fm < -2.0:
                    conflict_any = True
                    break
                if sc_fm > best_m:
                    best_m = sc_fm
            if not conflict_any and best_m >= FRAGMENT_THRESHOLD:
                cluster_scores.append((best_m, r))

        cluster_scores.sort(key=lambda x: (-x[0], ordered_evs[x[1]].content_hash))
        if len(cluster_scores) == 1 and cluster_scores[0][0] >= FRAGMENT_THRESHOLD:
            target_root = cluster_scores[0][1]
            members[target_root].append(f_idx)
        elif len(cluster_scores) >= 2 and (cluster_scores[0][0] - cluster_scores[1][0]) >= FRAGMENT_MARGIN:
            target_root = cluster_scores[0][1]
            members[target_root].append(f_idx)
        # Else ambiguous -> stays unlinked (not added to record_to_cluster -> patient = null)

    # Emit final deterministic cluster IDs ONLY for clusters that are identifiable
    # (multi-record clusters OR singletons with strong ID / full name+second clue)
    record_to_cluster: dict[tuple[str, str], int] = {}
    final_clusters: dict[int, list[RecordEvidence]] = {}
    cid = 0
    for r in sorted(members.keys(), key=lambda x: ordered_evs[x].content_hash):
        c_evs = [ordered_evs[idx] for idx in members[r]]
        # Unlinked single fragmentary note -> leave patient=null
        if len(c_evs) == 1 and c_evs[0].is_fragmentary:
            continue
        # Singleton without any strong ID and without >=2 name tokens -> leave patient=null
        if len(c_evs) == 1 and not c_evs[0].has_strong_id and not any(len(nf.norm_tokens) >= 2 for nf in c_evs[0].names):
            continue
        cid += 1
        c_evs.sort(key=lambda ev: (ev.content_hash, ev.rel, ev.loc))
        final_clusters[cid] = c_evs
        for ev in c_evs:
            record_to_cluster[(ev.rel, ev.loc)] = cid

    return ClusterResult(record_to_cluster=record_to_cluster, clusters=final_clusters)

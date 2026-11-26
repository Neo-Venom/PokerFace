"""Constrained descending-score clustering preventing transitive over-merging."""
from __future__ import annotations

import dataclasses
from collections import defaultdict

from src.config import Policy
from src.linkage.blocking import generate_candidate_pairs
from src.linkage.evidence import RecordEvidence
from src.linkage.scorer import (
    _name_pair_match, compute_token_idf, detect_shared_family_phones, dobs_compatible, independent_strong_ids,
    rel_conflict, rel_types_comparable, score_pair, set_translit_tokens, strong_id_groups, strong_shared,
)
from src.nlp.indic_norm import phonetic_skeleton
from src.security.keys import KeyRing


class ConstrainedClusters:
    """Union-find that aggregates cluster evidence and refuses any contradicting merge."""

    def __init__(self, evs: list[RecordEvidence]) -> None:
        self.parent = list(range(len(evs)))
        self.members: dict[int, list[int]] = {i: [i] for i in range(len(evs))}
        self.genders: dict[int, set[str]] = {i: ({evs[i].gender} if evs[i].gender else set()) for i in range(len(evs))}
        self.dobs: dict[int, set] = {i: set(evs[i].dobs) for i in range(len(evs))}
        self.aadhaars: dict[int, set[str]] = {i: set(evs[i].aadhaars) for i in range(len(evs))}
        self.last4: dict[int, set[str]] = {i: set(evs[i].aadhaar_last4) for i in range(len(evs))}
        self.abhas: dict[int, set[str]] = {i: set(evs[i].abhas) for i in range(len(evs))}
        self.pans: dict[int, set[str]] = {i: set(evs[i].pans) for i in range(len(evs))}
        self.byears: dict[int, set[int]] = {i: set(evs[i].birth_years) for i in range(len(evs))}
        self.names: dict[int, list[tuple[str, ...]]] = {i: list(evs[i].full_names) for i in range(len(evs))}
        self.rels_by_type: dict[int, dict[str, list[tuple[str, ...]]]] = {}
        for i, ev in enumerate(evs):
            d: dict[str, list[tuple[str, ...]]] = {}
            if ev.rel_names:
                d[ev.rel_type or "*"] = list(ev.rel_names)
            self.rels_by_type[i] = d

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def _can_merge(self, ra: int, rb: int, strong: bool = False, two_ids: bool = False) -> bool:
        if self.genders[ra] and self.genders[rb] and not (self.genders[ra] & self.genders[rb]):
            return False
        if self.dobs[ra] and self.dobs[rb] and not dobs_compatible(self.dobs[ra], self.dobs[rb]):
            return False
        if self.aadhaars[ra] and self.aadhaars[rb] and not (self.aadhaars[ra] & self.aadhaars[rb]):
            return False
        if self.last4[ra] and self.last4[rb] and not (self.last4[ra] & self.last4[rb]):
            return False
        if self.abhas[ra] and self.abhas[rb] and not (self.abhas[ra] & self.abhas[rb]):
            return False
        if self.pans[ra] and self.pans[rb] and not (self.pans[ra] & self.pans[rb]):
            return False
        if self.byears[ra] and self.byears[rb] and not two_ids:   # a corrupted age cell must not split one UHID
            all_y = self.byears[ra] | self.byears[rb]
            if max(all_y) - min(all_y) > 4:
                return False
        if strong:
            # A shared single-person identifier outranks spelling/transliteration differences in names
            return True
        _, _, name_conf = _name_pair_match(self.names[ra], self.names[rb])
        if name_conf:
            return False
        for rtype, rlist_a in self.rels_by_type[ra].items():
            for rtype_b, rlist_b in self.rels_by_type[rb].items():
                if rel_types_comparable(rtype, rtype_b) and rel_conflict(rlist_a, rlist_b):
                    return False
        return True

    def merge(self, i: int, j: int, strong: bool = False, two_ids: bool = False) -> bool:
        ra, rb = self.find(i), self.find(j)
        if ra == rb:
            return False
        if ra > rb:
            ra, rb = rb, ra
        if not self._can_merge(ra, rb, strong, two_ids):
            return False
        self.parent[rb] = ra
        self.members[ra].extend(self.members.pop(rb))
        self.genders[ra].update(self.genders.pop(rb))
        self.dobs[ra].update(self.dobs.pop(rb))
        self.aadhaars[ra].update(self.aadhaars.pop(rb))
        self.last4[ra].update(self.last4.pop(rb))
        self.abhas[ra].update(self.abhas.pop(rb))
        self.pans[ra].update(self.pans.pop(rb))
        self.byears[ra].update(self.byears.pop(rb))
        self.names[ra].extend(self.names.pop(rb))
        for rt, rl in self.rels_by_type.pop(rb).items():
            self.rels_by_type[ra].setdefault(rt, []).extend(rl)
        return True


_SIG_FIELDS = [f.name for f in dataclasses.fields(RecordEvidence) if f.name not in ("uid", "entry", "record")]


def _evidence_signature(ev: RecordEvidence) -> tuple:
    """Hashable digest of everything score_pair / strong_shared read (bulk exports repeat identical evidence)."""
    out = []
    for name in _SIG_FIELDS:
        v = getattr(ev, name)
        if isinstance(v, (set, frozenset)):
            out.append(frozenset(v))
        elif isinstance(v, list):
            out.append(tuple(v))
        else:
            out.append(v)
    return tuple(out)


AMBIGUITY_MARGIN = 1.5


def _ambiguous_records(evs: list[RecordEvidence], uf: ConstrainedClusters,
                       scored_edges: list[tuple[float, int, int]], shared_phones: set[str]) -> set[int]:
    """Records with no single-person ID of their own that fit two clusters (almost) equally well.

    Example: father and son share a name, a phone and an address; a note naming "Ramesh Sharma" with that
    phone fits both, and the two clusters may never merge. Linking it to either is a coin flip that
    precision (F0.5) punishes, so it is linked to neither (patient: null)."""
    best: dict[int, dict[int, float]] = defaultdict(dict)
    for sc, i, j in scored_edges:
        for a, b in ((i, j), (j, i)):
            rb = uf.find(b)
            if sc > best[a].get(rb, float("-inf")):
                best[a][rb] = sc
    out: set[int] = set()
    for idx, per_root in best.items():
        ev = evs[idx]
        own = uf.find(idx)
        own_best = per_root.get(own)
        others = sorted((s for r, s in per_root.items() if r != own), reverse=True)
        if own_best is None or not others or others[0] < own_best - AMBIGUITY_MARGIN:
            continue
        # A record whose own identifier links nowhere (an OPD number of another hospital) is as undecided
        # as one without identifiers; one that shares a single-person ID with its cluster is anchored.
        if any(m != idx and (strong_shared(ev, evs[m], shared_phones) or ev.uhid_cores & evs[m].uhid_cores)
               for m in uf.members[own]):
            continue
        out.add(idx)
    return out


def _has_identifying_anchor(ev: RecordEvidence) -> bool:
    return bool(
        ev.uhids or ev.uhid_cores or ev.aadhaars or ev.abhas or ev.pans
        or ev.emails or ev.ins_ids or ev.phones or ev.full_names
    )


# A weak record (no identifier, only a given name) attaches to a cluster only when one cluster clearly wins
FRAGMENT_MARGIN = 2.5
# ... and when its given name + clue (relative's name or DOB) would rarely coincide by chance in this corpus:
# expected coincidences = clusters sharing the given name * clusters sharing the clue / all clusters.
FRAGMENT_MAX_COINCIDENCE = 0.25


def is_weak_identity(ev: RecordEvidence) -> bool:
    """No identifier of any kind and no multi-part name: a note that only says "Meera, W/o Rajan"."""
    if ev.is_multi_patient_summary:
        return False
    if ev.uhids or ev.uhid_cores or ev.aadhaars or ev.abhas or ev.pans or ev.emails or ev.ins_ids or ev.phones:
        return False
    return not any(sum(1 for t in fn if len(t) > 1) >= 2 for fn in ev.full_names)


_POOL_SETS = ("given_norms", "surname_norms", "given_skels", "surname_skels", "rel_skels", "dobs", "birth_years",
              "villages", "districts", "pins", "hospitals", "diagnoses")


def _pool_evidence(members: list[RecordEvidence]) -> RecordEvidence:
    """Union of the soft evidence (names, relatives, DOB, place) of one cluster, for scoring a weak record.
    Identifiers are left out: a weak record has none, so they could only add vetoes, never support."""
    first = members[0]
    p = RecordEvidence(f"pool:{first.uid}", first.entry, first.record)
    for m in members:
        for f in _POOL_SETS:
            getattr(p, f).update(getattr(m, f))
        p.full_names.extend(n for n in m.full_names if n not in p.full_names)
        p.rel_names.extend(n for n in m.rel_names if n not in p.rel_names)
    genders = {m.gender for m in members if m.gender}
    p.gender = next(iter(genders)) if len(genders) == 1 else None
    rtypes = {m.rel_type for m in members if m.rel_type and m.rel_names}
    p.rel_type = next(iter(rtypes)) if len(rtypes) == 1 else None
    return p


def _attach_weak_records(
    evs: list[RecordEvidence],
    uf: ConstrainedClusters,
    weak: list[bool],
    cand_pairs: list[tuple[int, int]],
    shared_phones: set[str],
    idf: dict[str, float],
    thresh: float,
) -> set[int]:
    """Stage B: link each weak record to the single core cluster it fits, or to none (patient: null).

    A first name plus a relative's first name identifies someone only when that combination is rare in the
    corpus: "Asha, W/o Kamal" among 60 patients with varied names is one person, while "Meera ji, D/o Mahesh"
    in a register where six patients are called Meera and a dozen relatives Mahesh is a guess."""
    partners: dict[int, set[int]] = defaultdict(set)
    for i, j in cand_pairs:
        if weak[i] != weak[j]:
            w, c = (i, j) if weak[i] else (j, i)
            partners[w].add(c)
    if not partners:
        return set()
    given_roots: dict[str, set[int]] = defaultdict(set)
    rel_roots: dict[str, set[int]] = defaultdict(set)
    dob_roots: dict[object, set[int]] = defaultdict(set)
    roots: set[int] = set()
    for j, ev in enumerate(evs):
        if weak[j] or ev.is_multi_patient_summary or not ev.full_names:
            continue
        r = uf.find(j)
        roots.add(r)
        for fn in ev.full_names:
            for t in fn[:1]:
                if len(t) > 1:
                    given_roots[phonetic_skeleton(t)].add(r)
        for rn in ev.rel_names:
            for t in rn:
                if len(t) > 1:
                    rel_roots[phonetic_skeleton(t)].add(r)
        for d in ev.dobs:
            dob_roots[d].add(r)
    n_roots = max(1, len(roots))
    attached: set[int] = set()
    pooled: dict[int, RecordEvidence] = {}
    for w in sorted(partners, key=lambda i: evs[i].uid):
        wev = evs[w]
        best: dict[int, float] = {}
        for c in sorted(partners[w]):
            r = uf.find(c)
            sc = score_pair(wev, evs[c], shared_phones, idf)
            if r not in best:
                # The clues may sit in different records of one patient ("Asha" in an OPD row, "W/o Kamal" in
                # the registration): score against the cluster's pooled evidence as well.
                if r not in pooled:
                    pooled[r] = _pool_evidence([evs[m] for m in sorted(uf.members[r])])
                sc = max(sc, score_pair(wev, pooled[r], shared_phones, idf))
            best[r] = max(best.get(r, sc), sc)
        own = uf.find(w)
        valid = sorted(((sc, r) for r, sc in best.items() if sc > 0 and uf._can_merge(own, r)),
                       key=lambda x: (-x[0], x[1]))
        if not valid or valid[0][0] < thresh or (len(valid) >= 2 and valid[0][0] - valid[1][0] < FRAGMENT_MARGIN):
            continue
        root = valid[0][1]
        givens = {fn[0] for fn in wev.full_names if fn and len(fn[0]) > 1}
        n_given = max((len(given_roots.get(phonetic_skeleton(g), ())) for g in givens), default=n_roots)
        clue_counts: list[int] = []
        cl_rel = {phonetic_skeleton(t) for rl in uf.rels_by_type[root].values() for rn in rl for t in rn if len(t) > 1}
        for rn in wev.rel_names:
            for t in rn:
                sk = phonetic_skeleton(t)
                if len(t) > 1 and sk in cl_rel:
                    clue_counts.append(len(rel_roots.get(sk, ())))
        for d in wev.dobs & uf.dobs[root]:
            clue_counts.append(len(dob_roots.get(d, ())))
        if not clue_counts:
            continue
        if max(1, n_given) * max(1, min(clue_counts)) / n_roots > FRAGMENT_MAX_COINCIDENCE:
            continue
        member = next(i for i in uf.members[root])
        if uf.merge(w, member):
            attached.add(w)
    return attached


def link_and_assign_pseudonyms(
    evs: list[RecordEvidence],
    keyring: KeyRing,
    policy: Policy,
) -> dict[str, list[RecordEvidence]]:
    """Cluster records conservatively and issue deterministic HMAC pseudonyms."""
    set_translit_tokens(sorted({t for ev in evs for t in ev.translit_tokens}))
    shared_phones = detect_shared_family_phones(evs, strong_id_groups(evs))
    idf = compute_token_idf(evs)
    cand_pairs = generate_candidate_pairs(evs)
    weak = [is_weak_identity(ev) for ev in evs]

    scored_edges: list[tuple[float, int, int]] = []
    thresh = policy.linkage_merge_threshold
    sigs = [_evidence_signature(ev) for ev in evs]
    memo: dict[tuple[tuple, tuple], float] = {}
    for i, j in cand_pairs:
        if weak[i] or weak[j]:
            continue
        key = (sigs[i], sigs[j])
        sc = memo.get(key)
        if sc is None:
            sc = memo[key] = score_pair(evs[i], evs[j], shared_phones, idf)
        if sc >= thresh:
            scored_edges.append((sc, i, j))

    # Deterministic descending merge (tie-broken by record_uid)
    scored_edges.sort(key=lambda e: (-e[0], evs[e[1]].uid, evs[e[2]].uid))
    uf = ConstrainedClusters(evs)
    flag_memo: dict[tuple[tuple, tuple], tuple[bool, bool]] = {}
    for _sc, i, j in scored_edges:
        key = (sigs[i], sigs[j])
        flags = flag_memo.get(key)
        if flags is None:
            flags = flag_memo[key] = (strong_shared(evs[i], evs[j], shared_phones),
                                      independent_strong_ids(evs[i], evs[j], shared_phones) >= 2)
        uf.merge(i, j, strong=flags[0], two_ids=flags[1])

    ambiguous = _ambiguous_records(evs, uf, scored_edges, shared_phones)
    attached = _attach_weak_records(evs, uf, weak, cand_pairs, shared_phones, idf, thresh)

    # Group clusters deterministically by sorted member UIDs
    groups: dict[int, list[RecordEvidence]] = defaultdict(list)
    for idx, ev in enumerate(evs):
        if ev.is_multi_patient_summary or not _has_identifying_anchor(ev) or idx in ambiguous \
                or (weak[idx] and idx not in attached):
            ev.record.cluster_id = None
            ev.record.pseudonym = None
            continue
        root = uf.find(idx)
        groups[root].append(ev)

    # Ambiguity check: if a singleton fragmented record (only 1-word first name, no IDs/phone)
    # has >=2 plausible clusters matching it equally -> abstain (patient = null)
    clusters_by_psn: dict[str, list[RecordEvidence]] = {}
    used_psns: set[str] = set()

    ordered_groups = sorted(groups.values(), key=lambda g: min(e.uid for e in g))
    for grp in ordered_groups:
        # Canonical deterministic cluster key: smallest (input_sha256, loc) in the cluster
        canon_anchor = min(f"{e.entry.input_sha256}::{e.record.loc}" for e in grp)
        disambig = 0
        psn = keyring.pseudonym(canon_anchor, length=policy.psn_length, disambiguator=disambig)
        while psn in used_psns:
            disambig += 1
            psn = keyring.pseudonym(canon_anchor, length=policy.psn_length, disambiguator=disambig)
        used_psns.add(psn)
        for ev in grp:
            ev.record.cluster_id = canon_anchor
            ev.record.pseudonym = psn
        clusters_by_psn[psn] = grp

    return clusters_by_psn

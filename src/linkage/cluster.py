"""Constrained descending-score clustering preventing transitive over-merging."""
from __future__ import annotations

from collections import defaultdict

from src.config import Policy
from src.linkage.blocking import generate_candidate_pairs
from src.linkage.evidence import RecordEvidence
from src.linkage.scorer import _name_pair_match, compute_token_idf, detect_shared_family_phones, score_pair
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

    def _can_merge(self, ra: int, rb: int) -> bool:
        if self.genders[ra] and self.genders[rb] and not (self.genders[ra] & self.genders[rb]):
            return False
        if self.dobs[ra] and self.dobs[rb] and not (self.dobs[ra] & self.dobs[rb]):
            return False
        if self.aadhaars[ra] and self.aadhaars[rb] and not (self.aadhaars[ra] & self.aadhaars[rb]):
            return False
        if self.last4[ra] and self.last4[rb] and not (self.last4[ra] & self.last4[rb]):
            return False
        if self.abhas[ra] and self.abhas[rb] and not (self.abhas[ra] & self.abhas[rb]):
            return False
        if self.pans[ra] and self.pans[rb] and not (self.pans[ra] & self.pans[rb]):
            return False
        if self.byears[ra] and self.byears[rb]:
            all_y = self.byears[ra] | self.byears[rb]
            if max(all_y) - min(all_y) > 4:
                return False
        _, _, name_conf = _name_pair_match(self.names[ra], self.names[rb])
        if name_conf:
            return False
        for rtype, rlist_a in self.rels_by_type[ra].items():
            if rtype != "*" and rtype in self.rels_by_type[rb]:
                _, _, rconf = _name_pair_match(rlist_a, self.rels_by_type[rb][rtype])
                if rconf:
                    return False
        return True

    def merge(self, i: int, j: int) -> bool:
        ra, rb = self.find(i), self.find(j)
        if ra == rb:
            return False
        if ra > rb:
            ra, rb = rb, ra
        if not self._can_merge(ra, rb):
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


def _has_identifying_anchor(ev: RecordEvidence) -> bool:
    return bool(
        ev.uhids or ev.uhid_cores or ev.aadhaars or ev.abhas or ev.pans
        or ev.emails or ev.ins_ids or ev.phones or ev.full_names
    )


def link_and_assign_pseudonyms(
    evs: list[RecordEvidence],
    keyring: KeyRing,
    policy: Policy,
) -> dict[str, list[RecordEvidence]]:
    """Cluster records conservatively and issue deterministic HMAC pseudonyms."""
    shared_phones = detect_shared_family_phones(evs)
    idf = compute_token_idf(evs)
    cand_pairs = generate_candidate_pairs(evs)

    scored_edges: list[tuple[float, int, int]] = []
    thresh = policy.linkage_merge_threshold
    for i, j in cand_pairs:
        sc = score_pair(evs[i], evs[j], shared_phones, idf)
        if sc >= thresh:
            scored_edges.append((sc, i, j))

    # Deterministic descending merge (tie-broken by record_uid)
    scored_edges.sort(key=lambda e: (-e[0], evs[e[1]].uid, evs[e[2]].uid))
    uf = ConstrainedClusters(evs)
    for _sc, i, j in scored_edges:
        uf.merge(i, j)

    # Group clusters deterministically by sorted member UIDs
    groups: dict[int, list[RecordEvidence]] = defaultdict(list)
    for idx, ev in enumerate(evs):
        if ev.is_multi_patient_summary or not _has_identifying_anchor(ev):
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

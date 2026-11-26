"""Multi-key blocking for O(N)-scalable candidate pair generation."""
from __future__ import annotations

from collections import defaultdict

from src.linkage.evidence import RecordEvidence


def generate_candidate_pairs(evs: list[RecordEvidence]) -> list[tuple[int, int]]:
    """Build deterministic sorted list of candidate record index pairs."""
    blocks: dict[tuple[str, ...], list[int]] = defaultdict(list)

    for idx, ev in enumerate(evs):
        if ev.is_multi_patient_summary:
            continue
        for u in ev.uhids:
            blocks[("UHID", u)].append(idx)
        for uc in ev.uhid_cores:
            blocks[("UCORE", uc)].append(idx)
        for a in ev.aadhaars:
            blocks[("AAD", a)].append(idx)
        for ab in ev.abhas:
            blocks[("ABHA", ab)].append(idx)
        for pn in ev.pans:
            blocks[("PAN", pn)].append(idx)
        for em in ev.emails:
            blocks[("EM", em)].append(idx)
        for ins in ev.ins_ids:
            blocks[("INS", ins)].append(idx)
        for ph in ev.phones:
            blocks[("PH", ph)].append(idx)

        g_tag = ev.gender or "?"
        for gsk in ev.given_skels:
            blocks[("GSK_SEX", gsk, g_tag)].append(idx)
            blocks[("GSK_ONLY", gsk)].append(idx)
            for ssk in ev.surname_skels:
                blocks[("FULL_SKEL", gsk, ssk)].append(idx)
            for by in ev.birth_years:
                blocks[("GSK_BYEAR", gsk, str(by))].append(idx)
                blocks[("GSK_BYEAR", gsk, str(by + 1))].append(idx)
                blocks[("GSK_BYEAR", gsk, str(by - 1))].append(idx)
            for v in ev.villages:
                blocks[("GSK_VILL", gsk, v)].append(idx)
            for rsk in ev.rel_skels:
                blocks[("GSK_RSK", gsk, rsk)].append(idx)
            for dist in ev.districts:
                blocks[("GSK_DIST", gsk, dist)].append(idx)
        for ssk in ev.surname_skels:
            blocks[("SSK_SEX", ssk, g_tag)].append(idx)
            for dist in ev.districts:
                blocks[("SSK_DIST", ssk, dist)].append(idx)

    pairs: set[tuple[int, int]] = set()
    for bkey, idxs in blocks.items():
        # Guard against overly huge generic buckets (e.g. generic surname/token)
        if len(idxs) > 120 and bkey[0] in ("GSK_ONLY", "SSK_SEX"):
            continue
        for i_pos in range(len(idxs)):
            a = idxs[i_pos]
            for j_pos in range(i_pos + 1, len(idxs)):
                b = idxs[j_pos]
                pairs.add((a, b) if a < b else (b, a))
    return sorted(pairs)

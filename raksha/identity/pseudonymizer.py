"""M2 Step 7: Deterministic >=80-bit HMAC-SHA256 cluster pseudonyms with explicit collision avoidance."""
from __future__ import annotations

import hashlib
import hmac
import re
from typing import Any

from raksha.identity.evidence import RecordEvidence

_ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"  # Crockford base32
_PREFERRED_MRN_KEY_RE = re.compile(r"(?i)(?:uhid|mrn|pt_uid|patient_id|patient_ref|reg_no|hosp|record_code)")
_NON_PATIENT_ID_KEY_RE = re.compile(r"(?i)(?:encounter|visit|claim|bill|ipd)")
_PREFERRED_NAME_KEY_RE = re.compile(r"(?i)(?:patient|pt_name|full_name|^name$)")
_FALLBACK_ID_TAGS = ("[ABHA_NUMBER]", "[ABHA_ADDRESS]", "[AADHAAR]", "[AADHAAR_MASKED]", "[PHONE]")
_TEXT_TAG_PRIORITY = ("[MRN]", "[PATIENT_NAME]", *_FALLBACK_ID_TAGS)


def build_cluster_fingerprint(cluster_evs: list[RecordEvidence]) -> str:
    """Build a deterministic, order-independent cluster fingerprint from strongest stable evidence (never raw name alone)."""
    aadhaars = sorted({x for ev in cluster_evs for x in ev.aadhaars})
    abhas = sorted({x for ev in cluster_evs for x in ev.abhas})
    pans = sorted({x for ev in cluster_evs for x in ev.pans})
    mrns = sorted({f"{h}:{m}" for ev in cluster_evs for h, m in ev.scoped_mrns})
    phones = sorted({x for ev in cluster_evs for x in ev.phones})
    dobs = sorted({ev.dob for ev in cluster_evs if ev.dob})
    sexes = sorted({ev.sex for ev in cluster_evs if ev.sex})
    villages = sorted({v for ev in cluster_evs for v in ev.village_phonetics})
    rels = sorted({"_".join(r.phonetic_tokens) for ev in cluster_evs for r in ev.relatives if r.phonetic_tokens})
    names = sorted({"_".join(nf.phonetic_tokens) for ev in cluster_evs for nf in ev.names if nf.phonetic_tokens})
    hashes = sorted({ev.content_hash for ev in cluster_evs})

    # Prefer immutable government/hospital IDs + demographic anchors; fallback includes content hashes when no IDs exist
    parts = []
    if aadhaars:
        parts.append("aad:" + ",".join(aadhaars))
    if abhas:
        parts.append("abh:" + ",".join(abhas))
    if pans:
        parts.append("pan:" + ",".join(pans))
    if mrns:
        parts.append("mrn:" + ",".join(mrns))
    if dobs:
        parts.append("dob:" + ",".join(dobs))
    if sexes:
        parts.append("sex:" + ",".join(sexes))
    if villages:
        parts.append("vil:" + ",".join(villages))
    if rels:
        parts.append("rel:" + ",".join(rels))
    if phones:
        parts.append("ph:" + ",".join(phones))
    if names:
        parts.append("nm:" + ",".join(names))
    if not (aadhaars or abhas or pans or mrns):
        # Never rely on plaintext name alone for clusters lacking strong IDs
        parts.append("h:" + hashes[0][:16])
    return "|".join(parts)


def issue_cluster_pseudonyms(
    clusters: dict[int, list[RecordEvidence]],
    psn_key: bytes,
) -> dict[int, str]:
    """Issue unique >=80-bit Crockford base32 HMAC pseudonyms per cluster with explicit collision detection."""
    cluster_to_psn: dict[int, str] = {}
    used_psns: dict[str, str] = {}  # psn -> fingerprint

    # Sort deterministically by fingerprint (never file order or cluster int ID!)
    fp_items = [
        (build_cluster_fingerprint(evs), cid)
        for cid, evs in clusters.items()
    ]
    fp_items.sort(key=lambda x: (x[0], x[1]))

    for fp, cid in fp_items:
        nonce = 0
        while True:
            material = fp if nonce == 0 else f"{fp}#c{nonce}"
            digest = hmac.new(psn_key, material.encode("utf-8"), hashlib.sha256).digest()
            # 10 bytes = 80 bits -> 16 Crockford base32 chars
            n = int.from_bytes(digest[:10], "big")
            code = "".join(_ALPHABET[(n >> (5 * i)) & 31] for i in range(15, -1, -1))
            psn = f"PSN-{code}"
            if psn not in used_psns or used_psns[psn] == fp:
                used_psns[psn] = fp
                cluster_to_psn[cid] = psn
                break
            nonce += 1

    return cluster_to_psn


def _slot_priority(key_str: str, val_str: str) -> tuple[int, str] | None:
    """Rank sanitized placeholder slots so patient MRN/name cells take priority over auxiliary fields."""
    if not val_str:
        return None
    kl = key_str.casefold()
    is_pref_mrn = bool(_PREFERRED_MRN_KEY_RE.search(kl))
    is_non_pt = bool(_NON_PATIENT_ID_KEY_RE.search(kl))
    is_pref_name = bool(_PREFERRED_NAME_KEY_RE.search(kl))

    if val_str == "[MRN]":
        if is_pref_mrn and not is_non_pt:
            return (0, "[MRN]")
        return (1 if not is_non_pt else 2, "[MRN]")
    if val_str == "[PATIENT_NAME]":
        return (3 if is_pref_name else 4, "[PATIENT_NAME]")
    if "[MRN]" in val_str:
        return (5 if not is_non_pt else 6, "[MRN]")
    if "[PATIENT_NAME]" in val_str:
        return (7 if is_pref_name else 8, "[PATIENT_NAME]")
    for tag in _FALLBACK_ID_TAGS:
        if val_str == tag:
            return (9, tag)
    for tag in _FALLBACK_ID_TAGS:
        if tag in val_str:
            return (10, tag)
    return None


def inject_psn_tabular_row(clean_header: list[str], clean_row: list[str], psn: str) -> list[str]:
    """Insert `psn` into the best matching redacted identity cell of a sanitized CSV/TSV row."""
    row = list(clean_row)
    best: tuple[int, int, str] | None = None
    for idx, cell in enumerate(row):
        hdr = clean_header[idx] if idx < len(clean_header) else ""
        prio = _slot_priority(hdr, cell)
        if prio is not None:
            cand = (prio[0], idx, prio[1])
            if best is None or cand < best:
                best = cand
                if cand[0] == 0:
                    break
    if best is not None:
        _rank, idx, tag = best
        row[idx] = psn if row[idx] == tag else row[idx].replace(tag, psn, 1)
    return row


def inject_psn_json_obj(cleaned_obj: Any, psn: str) -> Any:
    """Insert `psn` into a sanitized JSON record or FHIR R4 Bundle without breaking its schema."""
    if isinstance(cleaned_obj, dict) and cleaned_obj.get("resourceType") == "Bundle" and isinstance(cleaned_obj.get("entry"), list):
        for ent in cleaned_obj["entry"]:
            if not isinstance(ent, dict):
                continue
            res = ent.get("resource")
            if not isinstance(res, dict) or res.get("resourceType") != "Patient":
                continue
            idents = res.get("identifier")
            if isinstance(idents, list):
                for tag_want in ("[MRN]", "[ABHA_NUMBER]", "[ABHA_ADDRESS]", "[AADHAAR]", "[PHONE]"):
                    for it in idents:
                        if isinstance(it, dict) and it.get("value") == tag_want:
                            it["value"] = psn
                            return cleaned_obj
            names = res.get("name")
            if isinstance(names, list):
                for nm in names:
                    if isinstance(nm, dict):
                        if nm.get("text") == "[PATIENT_NAME]":
                            nm["text"] = psn
                            return cleaned_obj
                        if nm.get("family") == "[PATIENT_NAME]":
                            nm["family"] = psn
                            return cleaned_obj
            div = res.get("text", {}).get("div") if isinstance(res.get("text"), dict) else None
            if isinstance(div, str):
                for tag in ("[MRN]", "[PATIENT_NAME]"):
                    if tag in div:
                        res["text"]["div"] = div.replace(tag, psn, 1)
                        return cleaned_obj
        return cleaned_obj

    if not isinstance(cleaned_obj, (dict, list)):
        return cleaned_obj

    # Iterative traversal so deeply nested JSON (1,500+ levels) cannot hit RecursionError
    best: tuple[int, int, Any, Any, str] | None = None
    seq = 0
    stack: list[tuple[Any, Any, str]] = []
    if isinstance(cleaned_obj, dict):
        for k, v in reversed(list(cleaned_obj.items())):
            if k != "_export_meta":
                stack.append((cleaned_obj, k, str(k)))
    else:
        for i in range(len(cleaned_obj) - 1, -1, -1):
            stack.append((cleaned_obj, i, ""))

    while stack:
        container, slot, key_str = stack.pop()
        val = container[slot]
        if isinstance(val, dict):
            for k, _v in reversed(list(val.items())):
                if k != "_export_meta":
                    stack.append((val, k, str(k)))
        elif isinstance(val, list):
            for i in range(len(val) - 1, -1, -1):
                stack.append((val, i, key_str))
        elif isinstance(val, str):
            prio = _slot_priority(key_str, val)
            if prio is not None:
                seq += 1
                if best is None or prio[0] < best[0]:
                    best = (prio[0], seq, container, slot, prio[1])
                    if prio[0] == 0:
                        break

    if best is not None:
        _rank, _seq, container, slot, tag = best
        old = container[slot]
        container[slot] = psn if old == tag else old.replace(tag, psn, 1)
    return cleaned_obj


def inject_psn_text(cleaned_text: str, psn: str) -> str:
    """Insert `psn` into the primary redacted patient slot of a sanitized free-text document."""
    for tag in _TEXT_TAG_PRIORITY:
        if tag in cleaned_text:
            return cleaned_text.replace(tag, psn, 1)
    return cleaned_text

"""Main Two-Pass + Global Optimization pipeline orchestrator for Project RAKSHA."""
from __future__ import annotations

import base64
import json
import os
import regex as re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

from src.config import Policy, load_policy
from src.ingest.csv_handler import parse_csv_entry, serialize_csv_entry
from src.ingest.fhir_handler import sanitize_fhir_bundle
from src.ingest.hidden import scrub_hidden
from src.ingest.json_handler import parse_json_entry, serialize_json_entry, transform_json_iterative
from src.ingest.profiler import infer_role_from_header, refine_entry_roles, resolve_key_role
from src.ingest.router import maybe_rename_rel_path, scan_input_tree
from src.ingest.text_handler import parse_text_entry, rewrite_text_preserving_lines
from src.ingest.xml_handler import inject_psn_xml, parse_xml_entry, sanitize_xml_text
from src.linkage.cluster import link_and_assign_pseudonyms
from src.linkage.evidence import extract_all_evidence
from src.models import FileEntry
from src.nlp.date_shifter import shift_compact_hl7_date, shift_date_value, shift_dates_in_text
from src.nlp.indic_norm import deobfuscate_unicode, name_tokens, shadow_normalize, unquote_percent_text
from src.nlp.medical_guard import MedicalGuard
from src.nlp.pii_ner import PiiHarvester, extract_context_name_spans, resolve_and_scrub, split_safe_chunks
from src.nlp.pii_regex import extract_regex_spans
from src.privacy.audit import run_egress_audit
from src.privacy.kanon import PrivacyPlan, build_privacy_plan, generalize_age_value, scrub_tehsils_and_ages_in_text
from src.security.keys import KeyRing
from src.security.safelog import install_pii_free_guards
from src.security.vault import EncryptedVault

_B64_RE = re.compile(r"^[A-Za-z0-9+/]{24,}={0,2}$")
_REL_PREFIX_RE = re.compile(r"(?i)^(?:S/o|D/o|W/o|C/O|son of|daughter of|wife of)\s+")
_PIN_6_RE = re.compile(r"\b([1-8]\d{5})\b")
_PIN3_MASKED_RE = re.compile(r"(?<!\d)([1-8]\d{2})\*{3}(?!\d)")
_ENCOUNTER_KEY_WORDS = ("visit", "encounter", "claim", "bill", "sample", "accession", "invoice", "receipt", "token",
                        "slip", "order", "ticket", "trip", "batch", "lot")
_JOINED_WORDS_RE = re.compile(r"(?<=[^\W\d_])[_.]+(?=[^\W\d_])")
# JSON objects that hold a relative / companion of the patient (their names are RELATIVE_NAME, never the patient)
_REL_PARENT_KEYS = frozenset({
    "guardian", "relative", "relatives", "kin", "nextofkin", "spouse", "father", "mother", "husband", "wife",
    "attendant", "attendants", "relatedperson", "parent", "parents", "son", "daughter", "brother", "sister",
    "emergencycontact", "informant", "nominee", "caretaker", "caregiver", "escort", "bystander",
})


def _is_relative_path(path_lower: str) -> bool:
    return any(re.sub(r"[^a-z]", "", seg) in _REL_PARENT_KEYS for seg in path_lower.split("/")[:-1])


_NON_PII_JSON_PARENTS = {"facility", "organization", "lab", "hospital", "doctor", "practitioner", "test", "rx", "diagnosis", "vitals", "panels", "results"}


def peak_rss_mb() -> float:
    try:
        import resource

        self_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        child_rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        total = self_rss + child_rss
        return round(total / 1024.0 if sys.platform != "darwin" else total / 1048576.0, 2)
    except Exception:
        if sys.platform == "win32":
            try:
                import ctypes
                import ctypes.wintypes

                class _PMC(ctypes.Structure):
                    _fields_ = [
                        ("cb", ctypes.wintypes.DWORD),
                        ("PageFaultCount", ctypes.wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t),
                        ("PeakPagefileUsage", ctypes.c_size_t),
                    ]

                pmc = _PMC()
                pmc.cb = ctypes.sizeof(_PMC)
                k32 = ctypes.windll.kernel32
                k32.GetCurrentProcess.restype = ctypes.c_void_p
                k32.K32GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.wintypes.DWORD]
                k32.K32GetProcessMemoryInfo.restype = ctypes.wintypes.BOOL
                proc = k32.GetCurrentProcess()
                if k32.K32GetProcessMemoryInfo(proc, ctypes.byref(pmc), pmc.cb):
                    return round(pmc.PeakWorkingSetSize / (1024.0 * 1024.0), 2)
            except Exception:
                pass
        return 0.0


def _fail_closed(entry: FileEntry, reason: str) -> None:
    """A stage that cannot process one file quarantines that file (never a half-processed copy); the run goes on."""
    entry.status = "quarantined"
    entry.rel_output_path = None
    entry.reason = reason


def _each_entry(entries: list[FileEntry], fn: Any, reason: str) -> None:
    for entry in entries:
        if entry.status != "sanitized":
            continue
        try:
            fn(entry)
        except Exception:
            _fail_closed(entry, reason)


def _parse_entry(entry: FileEntry, policy: Policy) -> None:
    if entry.status != "sanitized" or entry.text_decoded is None:
        return
    if entry.file_type in ("csv", "tsv"):
        parse_csv_entry(entry)
    elif entry.file_type == "json":
        parse_json_entry(entry, policy)
    elif entry.file_type == "xml":
        parse_xml_entry(entry)
    else:
        parse_text_entry(entry)


def _harvest_text_spans(text: str, guard: MedicalGuard, harvester: PiiHarvester) -> None:
    if not text or not text.strip():
        return
    work = unquote_percent_text(deobfuscate_unicode(text))
    for chunk in split_safe_chunks(work, max_len=32768):
        rx = extract_regex_spans(chunk)
        ctx = extract_context_name_spans(chunk, guard, rx)
        prot = guard.protected_spans(chunk)
        for sp in rx + ctx:
            if sp.kind in ("PATIENT_NAME", "RELATIVE_NAME"):
                if not guard.is_protected_span(sp.start, sp.end, prot, strict=True):
                    harvester.add_name(sp.text)
            elif sp.kind in ("MRN", "PHONE", "AADHAAR", "ABHA_NUMBER", "PAN", "INSURANCE_ID"):
                in_doc_or_med = any(
                    (tag == "DOCTOR" and ps < sp.end and pe > sp.start) or (tag == "MEDICAL" and ps <= sp.start and sp.end <= pe)
                    for ps, pe, tag in prot
                )
                if not in_doc_or_med:
                    harvester.add_id(sp.text)
            elif sp.kind in ("VILLAGE", "LOCALITY", "LANDMARK", "HOUSE"):
                harvester.add_address_piece(sp.text)


def _harvest_entry(entry: FileEntry, guard: MedicalGuard, harvester: PiiHarvester) -> None:
    if entry.status != "sanitized" or entry.is_clean_doc:
        return

    if entry.file_type in ("csv", "tsv") and entry.parsed_data:
        rows = entry.parsed_data
        roles = entry.column_roles
        for h in rows[0]:
            _harvest_text_spans(str(h), guard, harvester)
        for row in rows[1:]:
            for idx, cell in enumerate(row):
                val = str(cell).strip()
                if not val:
                    continue
                role = roles.get(idx, "FREE_TEXT")
                if role == "PATIENT_NAME":
                    harvester.add_name(val)
                elif role == "RELATIVE_NAME":
                    cleaned_rel = _REL_PREFIX_RE.sub("", val).strip()
                    harvester.add_name(cleaned_rel)
                elif role in ("PHONE", "AADHAAR", "ABHA", "PAN", "INSURANCE_ID", "MRN"):
                    harvester.add_id(val)
                elif role == "VILLAGE":
                    harvester.add_address_piece(val)
                elif role in ("ADDRESS", "NOTES", "FREE_TEXT"):
                    _harvest_text_spans(val, guard, harvester)
        return

    if entry.file_type in ("json", "fhir") and entry.parsed_data is not None:
        stack: list[tuple[str, Any]] = [("", entry.parsed_data)]
        while stack:
            parent_key, node = stack.pop()
            if isinstance(node, dict):
                for k, v in node.items():
                    ks = str(k)
                    _harvest_text_spans(ks, guard, harvester)
                    kl = ks.lower()
                    if isinstance(v, (dict, list)):
                        stack.append((kl, v))
                    elif isinstance(v, (int, float)) and not isinstance(v, bool) and parent_key not in _NON_PII_JSON_PARENTS:
                        iv = int(v) if isinstance(v, int) or float(v).is_integer() else None
                        if iv is not None and (
                            100_000_000_000 <= iv <= 999_999_999_999
                            or 6_000_000_000 <= iv <= 9_999_999_999
                            or 91_000_000_000_000 <= iv <= 91_999_999_999_999
                        ):
                            harvester.add_id(str(iv))
                    elif v is not None:
                        sv = str(v).strip()
                        if not sv:
                            continue
                        if kl == "data" or _B64_RE.match(sv):
                            try:
                                dec = base64.b64decode(sv + "=" * (-len(sv) % 4)).decode("utf-8", errors="ignore")
                                _harvest_text_spans(dec, guard, harvester)
                            except Exception:
                                pass
                        role = resolve_key_role(entry, ks)
                        if kl in ("first", "last", "local", "name", "full_name") and parent_key not in _NON_PII_JSON_PARENTS:
                            harvester.add_name(sv)
                        elif role == "PATIENT_NAME" and parent_key not in _NON_PII_JSON_PARENTS:
                            harvester.add_name(sv)
                        elif role == "RELATIVE_NAME":
                            harvester.add_name(_REL_PREFIX_RE.sub("", sv))
                        elif role in ("PHONE", "AADHAAR", "ABHA", "PAN", "INSURANCE_ID", "MRN") or kl in ("aadhar_no", "callback", "patient_ref", "hospital_uhid", "uhid", "source"):
                            harvester.add_id(sv.lstrip("#"))
                        elif role == "VILLAGE" or kl in ("village", "city"):
                            if parent_key not in ("organization", "facility", "hospital"):
                                harvester.add_address_piece(sv)
                        elif role == "ADDRESS" or kl in ("line", "address", "address_note", "source_path", "remarks", "notes", "div"):
                            _harvest_text_spans(sv, guard, harvester)
                        elif role == "TEHSIL":
                            pass
                        elif role is None and len(sv) >= 8 and " " in sv:
                            _harvest_text_spans(sv, guard, harvester)
            elif isinstance(node, list):
                for item in node:
                    if isinstance(item, (dict, list)):
                        stack.append((parent_key, item))
                    elif isinstance(item, str) and parent_key == "line":
                        _harvest_text_spans(item, guard, harvester)
        return

    if entry.text_decoded:
        _harvest_text_spans(entry.text_decoded, guard, harvester)


_ADDR_WORD_RE = re.compile(r"[A-Z][A-Za-z'\-]{1,30}|[\u0900-\u097f]{2,30}")
_PLACEHOLDER_RE = re.compile(r"\[(?:ADDRESS|NAME|RELATIVE|PHONE|ID|PIN|EMAIL|DOB)\]")


def _fail_closed_address(text: str) -> str:
    """Inside an address field, every remaining proper noun is location PII unless it is a public district/state."""
    from data.gazetteers import KNOWN_GEO_DISTRICTS_STATES

    placeholders = [(m.start(), m.end()) for m in _PLACEHOLDER_RE.finditer(text)]

    def _repl(m: "re.Match[str]") -> str:
        if any(ps <= m.start() < pe for ps, pe in placeholders):
            return m.group(0)
        if m.group(0).lower() in KNOWN_GEO_DISTRICTS_STATES:
            return m.group(0)
        return "[ADDRESS]"

    out = _ADDR_WORD_RE.sub(_repl, text)
    return re.sub(r"\[ADDRESS\](?:[ \t]+\[ADDRESS\])+", "[ADDRESS]", out)


def _pii_label_count(labels: list[str], guard: MedicalGuard, harvester: PiiHarvester, cap: int = 2) -> int:
    n = 0
    for lab in labels:
        s = str(lab)
        if len(s) < 4:
            continue
        spans = [sp for sp in extract_regex_spans(s) if sp.kind in ("MRN", "PHONE", "AADHAAR", "ABHA_NUMBER", "PAN", "EMAIL")]
        # a single swept token ("Full" in "Guardian_Full") is not evidence; a multi-word name or an ID is
        names = [sp for sp in harvester.sweep_spans(s) if sp.kind != "PATIENT_NAME" or len(sp.text.split()) >= 2]
        if spans or names:
            n += 1
            if n >= cap:
                break
    return n


def _looks_multi_patient(entry: FileEntry, guard: MedicalGuard, harvester: PiiHarvester) -> bool:
    """A table whose column headers or JSON keys carry patient identifiers belongs to several patients."""
    if entry.file_type in ("csv", "tsv") and entry.parsed_data:
        return _pii_label_count([str(h) for h in entry.parsed_data[0]], guard, harvester) >= 2
    if entry.file_type == "json" and entry.parsed_data is not None:
        keys: list[str] = []
        stack: list[Any] = [entry.parsed_data]
        while stack and len(keys) < 4000:
            node = stack.pop()
            if isinstance(node, dict):
                keys.extend(str(k) for k in node.keys())
                stack.extend(v for v in node.values() if isinstance(v, (dict, list)))
            elif isinstance(node, list):
                stack.extend(v for v in node if isinstance(v, (dict, list)))
        return _pii_label_count(keys, guard, harvester) >= 2
    return False


def _rewrite_entry(
    entry: FileEntry,
    guard: MedicalGuard,
    harvester: PiiHarvester,
    entity_counts: Counter[str],
    plan: PrivacyPlan,
    psn_by_id: dict[str, set[str]] | None = None,
    psn_by_name: dict[str, set[str]] | None = None,
) -> str:
    if entry.is_clean_doc:
        if entry.file_type in ("csv", "tsv"):
            return serialize_csv_entry(entry, scrub_header_fn=lambda h: h, rewrite_row_fn=lambda _i, r: r)
        if entry.file_type in ("json", "fhir"):
            return serialize_json_entry(entry, entry.parsed_data)
        return str(entry.parsed_data)

    dmy = entry.date_order_hint == "dmy"
    file_shift = plan.file_shift_days.get(entry.rel_input_path, 47)

    def _apply_pin_policy(txt: str, pin_mode: str, is_address: bool = False) -> str:
        if is_address:
            if pin_mode == "suppress":
                txt = _PIN_6_RE.sub("[PIN]", txt)
            else:
                txt = _PIN_6_RE.sub(lambda m: m.group(1)[:3] + "***", txt)
        if pin_mode == "suppress":
            txt = _PIN3_MASKED_RE.sub("[PIN]", txt)
        return txt

    def _scrub_str(
        text: str,
        psn: str | None = None,
        inject_psn: bool = False,
        rec_uid: str | None = None,
        is_address: bool = False,
    ) -> str:
        # base64 / gzip / hex payloads (whole value or embedded) are decoded, scrubbed and re-encoded
        return scrub_hidden(text, lambda t: _scrub_plain_str(t, psn, inject_psn, rec_uid, is_address))

    def _scrub_plain_str(
        text: str,
        psn: str | None = None,
        inject_psn: bool = False,
        rec_uid: str | None = None,
        is_address: bool = False,
    ) -> str:
        work = unquote_percent_text(deobfuscate_unicode(text))
        cleaned, spans = resolve_and_scrub(
            work, guard, harvester, psn=psn, replace_primary_id_with_psn=inject_psn
        )
        for sp in spans:
            entity_counts[sp.kind] += 1
        uid = rec_uid or (entry.records[0].record_uid if entry.records else "")
        delta = plan.rec_shift_days.get(uid, file_shift)
        anchor = plan.rec_anchor_year.get(uid, 2026)
        age_band = plan.rec_age_band.get(uid)
        sup_teh = (uid in plan.rec_suppress_tehsil) if uid else True
        pin_mode = plan.rec_pin_level.get(uid, "suppress")
        if len(cleaned) <= 49152:
            cleaned = scrub_tehsils_and_ages_in_text(cleaned, age_band, sup_teh, plan.tehsil_rx, guard)
            cleaned = shift_dates_in_text(cleaned, delta_days=delta, dmy=dmy, anchor_year=anchor, guard=guard)
            cleaned = _apply_pin_policy(cleaned, pin_mode, is_address=is_address)
        else:
            out_chunks: list[str] = []
            for ch in split_safe_chunks(cleaned, max_len=32768):
                ch = scrub_tehsils_and_ages_in_text(ch, age_band, sup_teh, plan.tehsil_rx, guard)
                ch = shift_dates_in_text(ch, delta_days=delta, dmy=dmy, anchor_year=anchor, guard=guard)
                ch = _apply_pin_policy(ch, pin_mode, is_address=is_address)
                out_chunks.append(ch)
            cleaned = "".join(out_chunks)
        return cleaned

    if entry.file_type in ("csv", "tsv"):
        roles = entry.column_roles
        headers = [str(h).lower() for h in (entry.parsed_data[0] if entry.parsed_data else [])]
        # Pick primary ID column for PSN injection (prefer patient UHID/Reg/PatientID over bill_no/sample_id)
        psn_col_idx: int | None = None
        for idx, role in sorted(roles.items()):
            hdr = headers[idx] if idx < len(headers) else ""
            if role == "MRN" and not any(x in hdr for x in _ENCOUNTER_KEY_WORDS):
                psn_col_idx = idx
                break
        if psn_col_idx is None:
            for idx, role in sorted(roles.items()):
                if role == "PATIENT_NAME":
                    psn_col_idx = idx
                    break
        text_psn_col_idx: int | None = None
        if psn_col_idx is None:
            for want in ("NOTES", "ADDRESS", "FREE_TEXT"):
                for idx, role in sorted(roles.items()):
                    if role == want:
                        text_psn_col_idx = idx
                        break
                if text_psn_col_idx is not None:
                    break

        def _scrub_header(h: str) -> str:
            cleaned, spans = resolve_and_scrub(h, guard, harvester)
            id_kinds = ("MRN", "PHONE", "AADHAAR", "ABHA_NUMBER", "PAN", "EMAIL", "INSURANCE_ID")
            if spans and not any(sp.kind in id_kinds for sp in spans):
                # A header is a label: a one-word name hit inside a compound label ("Village/Mohalla") is not a
                # person. Keep it unless the whole header is that name or the hit is a multi-word name.
                whole = re.sub(r"[\W_]+", " ", h).strip().lower()
                if all(len(sp.text.split()) < 2 and sp.text.strip().lower() != whole for sp in spans):
                    return h
            for sp in spans:
                entity_counts[sp.kind] += 1
            if any(sp.kind in ("MRN", "PHONE", "AADHAAR", "ABHA_NUMBER") for sp in spans):
                # Strip adjacent single-word personal names embedded in pivoted patient column headers
                safe_hdr_words = {"time", "ward", "bed", "unit", "patient", "record", "date", "vitals", "sheet", "note", "notes", "ref", "code", "num", "id", "name", "total", "count", "index", "room"}
                toks = cleaned.split()
                new_toks = []
                for tok in toks:
                    if re.match(r"^[A-Z][a-z]{2,18}$", tok) and tok.lower() not in safe_hdr_words and not guard.is_clinical_word(tok):
                        entity_counts["PATIENT_NAME"] += 1
                        new_toks.append("[NAME]")
                    else:
                        new_toks.append(tok)
                cleaned = " ".join(new_toks)
            return cleaned

        def _rewrite_row(row_idx: int, row: list[str]) -> list[str]:
            rec = entry.records[row_idx] if row_idx < len(entry.records) else None
            row_psn = rec.pseudonym if rec else None
            uid = rec.record_uid if rec else ""
            delta = plan.rec_shift_days.get(uid, file_shift)
            anchor = plan.rec_anchor_year.get(uid, 2026)
            age_band = plan.rec_age_band.get(uid)
            birth_cohort = plan.rec_birth_cohort.get(uid)
            sup_teh = (uid in plan.rec_suppress_tehsil) if uid else True
            pin_mode = plan.rec_pin_level.get(uid, "suppress")
            psn_placed = psn_col_idx is not None
            out: list[str] = []
            for idx, cell in enumerate(row):
                val = str(cell)
                if not val.strip():
                    out.append(val)
                    continue
                role = roles.get(idx, "FREE_TEXT")
                if row_psn and idx == psn_col_idx:
                    entity_counts[role if role in ("MRN", "PATIENT_NAME") else "MRN"] += 1
                    out.append(row_psn)
                elif row_psn and not psn_placed and idx == text_psn_col_idx:
                    cleaned_val = _scrub_str(val, psn=row_psn, inject_psn=True, rec_uid=uid)
                    if row_psn not in cleaned_val:
                        cleaned_val = f"[{row_psn}] {cleaned_val}"
                    psn_placed = True
                    out.append(cleaned_val)
                elif role == "PATIENT_NAME":
                    entity_counts["PATIENT_NAME"] += 1
                    out.append("[NAME]")
                elif role == "RELATIVE_NAME":
                    entity_counts["RELATIVE_NAME"] += 1
                    out.append("[RELATIVE]")
                elif role == "PHONE":
                    entity_counts["PHONE"] += 1
                    out.append("[PHONE]")
                elif role == "EMAIL":
                    entity_counts["EMAIL"] += 1
                    out.append("[EMAIL]")
                elif role in ("AADHAAR", "ABHA", "PAN", "INSURANCE_ID", "MRN"):
                    entity_counts[role] += 1
                    out.append("[ID]")
                elif role == "DOB":
                    entity_counts["DOB"] += 1
                    out.append(birth_cohort or "[DOB]")
                elif role == "AGE":
                    out.append(generalize_age_value(val, age_band))
                elif role == "DATE":
                    out.append(shift_date_value(val, delta_days=delta, dmy=dmy, anchor_year=anchor))
                elif role == "PIN":
                    entity_counts["PIN"] += 1
                    digits = re.sub(r"\D", "", shadow_normalize(val))
                    if pin_mode == "suppress" or len(digits) != 6:
                        out.append("[PIN]")
                    else:
                        out.append(digits[:3] + "***")
                elif role == "VILLAGE":
                    entity_counts["VILLAGE"] += 1
                    out.append("[ADDRESS]")
                elif role == "TEHSIL":
                    out.append("[ADDRESS]" if sup_teh else val)
                elif role == "DOCTOR":
                    out.append(val)
                elif role == "ADDRESS":
                    out.append(_fail_closed_address(_scrub_str(val, rec_uid=uid, is_address=True)))
                else:
                    out.append(_scrub_str(val, rec_uid=uid))
            return out

        return serialize_csv_entry(entry, scrub_header_fn=_scrub_header, rewrite_row_fn=_rewrite_row)

    if entry.file_type == "fhir":
        rec = entry.records[0] if entry.records else None
        psn = rec.pseudonym if rec else None
        uid = rec.record_uid if rec else ""
        delta = plan.rec_shift_days.get(uid, file_shift)
        anchor = plan.rec_anchor_year.get(uid, 2026)
        pin_mode = plan.rec_pin_level.get(uid, "suppress")

        def _fhir_geo(addr: dict[str, Any]) -> None:
            if "line" in addr:
                addr["line"] = []
            if "city" in addr:
                addr.pop("city", None)
            if isinstance(addr.get("postalCode"), str) and addr["postalCode"].strip():
                digits = re.sub(r"\D", "", shadow_normalize(addr["postalCode"]))
                addr["postalCode"] = "[PIN]" if (pin_mode == "suppress" or len(digits) != 6) else (digits[:3] + "***")

        cleaned = sanitize_fhir_bundle(
            entry.parsed_data,
            psn=psn,
            scrub_fn=lambda s: _scrub_str(s, rec_uid=uid),
            shift_date_fn=lambda d: shift_date_value(d, delta_days=delta, dmy=True, anchor_year=anchor),
            geo_fn=_fhir_geo,
        )
        return json.dumps(cleaned, ensure_ascii=False, indent=2) + "\n"

    if entry.file_type == "json":
        rec_by_ptr = {rec.loc: rec for rec in entry.records}
        doc_rec = rec_by_ptr.get("doc")
        is_summary_json = (not rec_by_ptr) or (doc_rec is not None and doc_rec.pseudonym is None and len(rec_by_ptr) == 1)
        psn_placed_for_ptr: set[str] = set()

        def _lookup_summary_psn(raw: str, allow_name: bool = False) -> str | None:
            if not is_summary_json or not psn_by_id:
                return None
            norm_k = re.sub(r"[^a-z0-9]", "", shadow_normalize(raw).lower())
            if len(norm_k) >= 5:
                hits = psn_by_id.get(norm_k)
                if hits and len(hits) == 1:
                    return next(iter(hits))
            if allow_name and psn_by_name:
                nn = " ".join(name_tokens(raw))
                if nn:
                    hits = psn_by_name.get(nn)
                    if hits and len(hits) == 1:
                        return next(iter(hits))
            return None

        def _rec_for_path(path: str) -> tuple[str, Any]:
            if doc_rec is not None:
                return "doc", doc_rec
            parts = path.split("/")
            for i in range(len(parts), 1, -1):  # longest record pointer that covers this path
                ptr = "/".join(parts[:i])
                r_obj = rec_by_ptr.get(ptr)
                if r_obj is not None:
                    return ptr, r_obj
            return "", None

        def _scrub_key(k: str) -> str:
            mapped = _lookup_summary_psn(k, allow_name=True)
            if mapped:
                entity_counts["MRN"] += 1
                return mapped
            cleaned, spans = resolve_and_scrub(k, guard, harvester)
            spaced = _JOINED_WORDS_RE.sub(" ", k)
            if spaced != k and " " not in k and harvester.sweep_spans(spaced):
                # "patient_Mohammad_Arif_meta": the name is only seen whole once the '_' joints are spaces
                cleaned, spans = resolve_and_scrub(spaced, guard, harvester)
                cleaned = cleaned.replace(" ", "_")
            for sp in spans:
                entity_counts[sp.kind] += 1
            return cleaned

        def _scrub_val(path: str, val: Any, key_name: str) -> Any:
            kl = key_name.lower()
            pl = path.lower()
            ptr_key, r_obj = _rec_for_path(path)
            r_psn = r_obj.pseudonym if r_obj else None
            r_uid = r_obj.record_uid if r_obj else ""
            r_delta = plan.rec_shift_days.get(r_uid, file_shift)
            r_anchor = plan.rec_anchor_year.get(r_uid, 2026)
            r_age_band = plan.rec_age_band.get(r_uid)
            r_cohort = plan.rec_birth_cohort.get(r_uid)
            r_sup_teh = (r_uid in plan.rec_suppress_tehsil) if r_uid else True
            r_pin_mode = plan.rec_pin_level.get(r_uid, "suppress")

            if val is None or isinstance(val, bool):
                return val
            if isinstance(val, (int, float)):
                role = resolve_key_role(entry, key_name)
                if role in ("AADHAAR", "ABHA", "PHONE", "MRN", "INSURANCE_ID") or kl in ("aadhar_no", "aadhaar", "mobile", "phone"):
                    entity_counts[role or "AADHAAR"] += 1
                    return 0
                if not any(x in pl for x in ("/panels/", "/results/", "/vitals/", "/rx/", "/diagnosis/")):
                    iv = int(val) if isinstance(val, int) or float(val).is_integer() else None
                    if iv is not None:
                        if 91_000_000_000_000 <= iv <= 91_999_999_999_999:
                            entity_counts["ABHA"] += 1
                            return 0
                        if 100_000_000_000 <= iv <= 999_999_999_999:
                            entity_counts["AADHAAR"] += 1
                            return 0
                        if 6_000_000_000 <= iv <= 9_999_999_999:
                            entity_counts["PHONE"] += 1
                            return 0
                if role == "PIN" or kl in ("pin", "postalcode", "zip"):
                    entity_counts["PIN"] += 1
                    return 0
                if (role == "AGE" or kl == "age") and not any(x in pl for x in ("/panels/", "/results/", "/vitals/", "/rx/", "/diagnosis/")) and r_age_band:
                    return r_age_band
                return val
            if not isinstance(val, str) or not val.strip():
                return val
            if kl in ("doctor", "ref_doctor", "referred_by", "hospital", "test", "unit", "range", "loinc", "drug", "strength", "frequency", "duration", "display", "code"):
                return val
            if "/facility/" in pl or "/organization/" in pl or kl == "lab":
                return val

            role = resolve_key_role(entry, key_name)
            is_encounter_key = any(x in kl for x in _ENCOUNTER_KEY_WORDS)
            if r_psn and ptr_key not in psn_placed_for_ptr and (kl in ("patient_ref", "hospital_uhid", "uhid", "reg_code", "hosp_ref", "subject_ref") or (role == "MRN" and not is_encounter_key)):
                entity_counts["MRN"] += 1
                psn_placed_for_ptr.add(ptr_key)
                return r_psn
            if kl in ("first", "last", "local", "name", "full_name") and not any(x in pl for x in ("/facility/", "/lab", "/panels/", "/diagnosis/")):
                if _is_relative_path(pl):
                    entity_counts["RELATIVE_NAME"] += 1
                    return "[RELATIVE]"
                if any(x in pl for x in ("/raised_by/", "/caller/", "/reporter/", "/submitted_by/", "/requested_by/", "/contact_person/")):
                    entity_counts["PATIENT_NAME"] += 1
                    if r_psn and ptr_key not in psn_placed_for_ptr and kl in ("name", "full_name", "first"):
                        psn_placed_for_ptr.add(ptr_key)
                        return r_psn
                    return "[NAME]"
                if "/patient/" in pl or "/patients/" in pl:
                    entity_counts["PATIENT_NAME"] += 1
                    if r_psn and ptr_key not in psn_placed_for_ptr and kl in ("name", "first"):
                        psn_placed_for_ptr.add(ptr_key)
                        return r_psn
                    return "[NAME]"
            if role == "PATIENT_NAME" and "/panels/" not in pl:
                entity_counts["PATIENT_NAME"] += 1
                if r_psn and ptr_key not in psn_placed_for_ptr:
                    psn_placed_for_ptr.add(ptr_key)
                    return r_psn
                return "[NAME]"
            if role == "RELATIVE_NAME":
                entity_counts["RELATIVE_NAME"] += 1
                return "[RELATIVE]"
            if role == "PHONE" or kl in ("callback", "mobile", "phone"):
                entity_counts["PHONE"] += 1
                return "[PHONE]"
            if role == "EMAIL":
                entity_counts["EMAIL"] += 1
                return "[EMAIL]"
            if role == "DOB" or kl == "dob":
                entity_counts["DOB"] += 1
                return r_cohort or "[DOB]"
            if role == "AGE" or kl in ("age", "age_sex"):
                return generalize_age_value(val, r_age_band)
            if role == "DATE" or kl in ("visit_date", "collected", "reported_on", "registered_on", "admission_date", "discharge_date", "generated_at", "date", "timestamp"):
                return shift_date_value(val, delta_days=r_delta, dmy=dmy, anchor_year=r_anchor)
            if role in ("AADHAAR", "ABHA", "PAN", "INSURANCE_ID", "MRN") or kl in ("aadhar_no", "uhid", "patient_ref", "hospital_uhid"):
                mapped_id = _lookup_summary_psn(val)
                entity_counts[role or "MRN"] += 1
                return mapped_id or "[ID]"
            if role == "PIN" or kl in ("pin", "postalcode", "zip"):
                entity_counts["PIN"] += 1
                digits = re.sub(r"\D", "", shadow_normalize(val))
                if r_pin_mode == "suppress" or len(digits) != 6:
                    return "[PIN]"
                return digits[:3] + "***"
            if role == "VILLAGE" or (kl in ("village", "city") and "/facility/" not in pl):
                entity_counts["VILLAGE"] += 1
                return "[ADDRESS]"
            if role == "TEHSIL" or "tehsil" in kl:
                return "[ADDRESS]" if r_sup_teh else val
            mapped_val = _lookup_summary_psn(val, allow_name=True)
            if mapped_val:
                entity_counts["MRN"] += 1
                return mapped_val
            if role == "ADDRESS" and "/facility/" not in pl and "/organization/" not in pl:
                return _fail_closed_address(_scrub_str(val, rec_uid=r_uid, is_address=True))
            return _scrub_str(val, rec_uid=r_uid)

        cleaned = transform_json_iterative(entry.parsed_data, scrub_key_fn=_scrub_key, scrub_val_fn=_scrub_val)
        return serialize_json_entry(entry, cleaned)

    if entry.file_type == "xml":
        rec = entry.records[0] if entry.records else None
        psn = rec.pseudonym if rec else None
        uid = rec.record_uid if rec else ""
        delta = plan.rec_shift_days.get(uid, file_shift)

        def _scrub_xml(s: str) -> str:
            if s.isdigit() and len(s) in (8, 14) and s.startswith(("19", "20")):
                shifted = shift_compact_hl7_date(s, delta)
                if shifted != s:
                    return shifted
            return _scrub_str(s, psn=psn, inject_psn=bool(psn), rec_uid=uid)

        out_xml = sanitize_xml_text(entry.parsed_data, scrub_fn=_scrub_xml)
        if psn and psn not in out_xml:
            out_xml = inject_psn_xml(out_xml, psn)
        return out_xml

    # Text / WhatsApp preserving exact line counts and ensuring PSN appears inside linked doc
    rec = entry.records[0] if entry.records else None
    psn = rec.pseudonym if rec else None
    uid = rec.record_uid if rec else ""
    psn_state = {"placed": False}

    def _scrub_doc_line(line: str) -> str:
        cleaned = _scrub_str(line, psn=psn, inject_psn=(bool(psn) and not psn_state["placed"]), rec_uid=uid)
        if psn and not psn_state["placed"] and psn in cleaned:
            psn_state["placed"] = True
        return cleaned

    rendered = rewrite_text_preserving_lines(entry.parsed_data, scrub_line_fn=_scrub_doc_line)
    # If text note had no MRN line (e.g. "Asha, W/o Kamal..."), replace first [NAME] with PSN
    if psn and not psn_state["placed"]:
        if "[NAME]" in rendered:
            rendered = rendered.replace("[NAME]", psn, 1)
        elif "[PHONE]" in rendered:
            rendered = rendered.replace("[PHONE]", psn, 1)
    return rendered


def run_pipeline(input_dir: Path, output_dir: Path, policy: Policy | None = None) -> int:
    install_pii_free_guards()
    t0 = time.monotonic()
    pol = policy or load_policy()
    secret_for_audit = os.environ.get("RAKSHA_SECRET", "")
    keyring = KeyRing.from_env(scrub_env=True)

    san_dir = output_dir / "sanitized"
    san_dir.mkdir(parents=True, exist_ok=True)

    scanned = scan_input_tree(input_dir)
    entries: list[FileEntry] = [entry for _path, _rel, entry in scanned]
    _each_entry(entries, lambda e: _parse_entry(e, pol), "parse_error")

    # Pass 1: Learn runtime clinical lexicon & harvest PII registry
    guard = MedicalGuard()
    _each_entry(entries, lambda e: guard.learn_from_entries([e]), "parse_error")
    _each_entry([e for e in entries if not e.is_clean_doc], lambda e: refine_entry_roles(e, guard), "parse_error")

    harvester = PiiHarvester(guard)
    _each_entry(entries, lambda e: _harvest_entry(e, guard, harvester), "parse_error")
    harvester.compile()

    def _set_multi(e: FileEntry) -> None:
        e.multi_patient = _looks_multi_patient(e, guard, harvester)
    _each_entry([e for e in entries if not e.is_clean_doc], _set_multi, "parse_error")

    # Global Cross-Script Record Linkage (Phase 4)
    try:
        evs = extract_all_evidence(entries, guard)
    except Exception:  # find and quarantine the file(s) the extractor cannot read, then link the rest
        _each_entry(entries, lambda e: extract_all_evidence([e], guard), "parse_error")
        evs = extract_all_evidence(entries, guard)
    try:
        link_and_assign_pseudonyms(evs, keyring, pol)
    except Exception:  # fail safe: an unlinked record is never a false merge
        for ev in evs:
            ev.record.pseudonym = None

    psn_by_id: dict[str, set[str]] = {}
    psn_by_name: dict[str, set[str]] = {}
    for ev in evs:
        psn = ev.record.pseudonym
        if not psn or ev.is_multi_patient_summary:
            continue
        for tok in ev.uhids | ev.abhas | ev.pans | ev.ins_ids | ev.phones | ev.aadhaars:
            nk = re.sub(r"[^a-z0-9]", "", shadow_normalize(tok).lower())
            if len(nk) >= 5:
                psn_by_id.setdefault(nk, set()).add(psn)
        for nt in ev.full_names:
            nn = " ".join(nt)
            if nn:
                psn_by_name.setdefault(nn, set()).add(psn)

    # Adversarial Privacy Layer (Phase 5: per-patient date shifts + k-anonymity plan)
    try:
        privacy_plan = build_privacy_plan(entries, evs, keyring, pol)
    except Exception:  # fail closed: no per-record bands -> structured ages become [AGE], tehsils/PINs suppressed
        privacy_plan = PrivacyPlan()
    raw_path_by_sha: dict[str, Path] = {e.input_sha256: p for p, _r, e in scanned}
    vault = EncryptedVault(keyring.k_vault) if pol.encrypt_quarantine_archive else None

    # Rename PII-bearing filenames deterministically using harvested PII + regex
    used_out_paths: set[str] = set()
    for entry in entries:
        if entry.status != "sanitized" or not entry.rel_output_path:
            continue

        def _fname_has_pii(stem_text: str) -> bool:
            _, spans = resolve_and_scrub(stem_text, guard, harvester)
            return bool(spans)

        try:
            new_rel, renamed = maybe_rename_rel_path(
                entry.rel_input_path,
                entry.input_sha256,
                keyring,
                has_pii_fn=_fname_has_pii,
            )
        except Exception:  # cannot vet the name: never re-emit it
            new_rel, renamed = maybe_rename_rel_path(entry.rel_input_path, entry.input_sha256, keyring, lambda _s: True)
        if new_rel in used_out_paths:
            p = Path(new_rel)
            new_rel = (p.parent / f"{p.stem}_{keyring.file_tag(entry.input_sha256, 4)}{p.suffix}").as_posix()
        used_out_paths.add(new_rel)
        entry.rel_output_path = new_rel
        entry.has_pii_filename = renamed
        for rec in entry.records:
            rec.rel_output_path = new_rel

    files_report: list[dict[str, Any]] = []
    records_out: list[dict[str, Any]] = []
    entity_counts: Counter[str] = Counter()

    for entry in entries:
        if entry.status == "sanitized" and entry.rel_output_path:
            out_rel = entry.rel_output_path
            dest = san_dir / out_rel
            try:
                file_counts: Counter[str] = Counter()
                serialized = _rewrite_entry(entry, guard, harvester, file_counts, privacy_plan, psn_by_id=psn_by_id, psn_by_name=psn_by_name)
                payload = serialized.encode("utf-8")
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(payload)
                entity_counts.update(file_counts)
            except Exception:
                dest.unlink(missing_ok=True)
                _fail_closed(entry, "rewrite_error")
        if entry.status == "sanitized" and entry.rel_output_path:
            out_rel = entry.rel_output_path
            files_report.append(
                {
                    "input_sha256": entry.input_sha256,
                    "output": f"sanitized/{out_rel}",
                    "status": "sanitized",
                }
            )
            for rec in entry.records:
                records_out.append(
                    {
                        "file": f"sanitized/{out_rel}",
                        "loc": rec.loc,
                        "patient": rec.pseudonym,
                    }
                )
        else:
            files_report.append(
                {
                    "input_sha256": entry.input_sha256,
                    "output": None,
                    "status": entry.status,
                    "reason": entry.reason or "quarantined",
                }
            )
            if vault is not None and entry.input_sha256 in raw_path_by_sha:
                vault.write_quarantined_blob(
                    output_dir / "quarantine" / f"{entry.rel_input_path}.enc",
                    raw_path_by_sha[entry.input_sha256].read_bytes(),
                    entry.input_sha256,
                )

    audit_res = run_egress_audit(
        entries=entries,
        san_dir=san_dir,
        files_report=files_report,
        records_out=records_out,
        guard=guard,
        harvester=harvester,
        secret_str=secret_for_audit,
    )
    if vault is not None:
        entry_by_sha = {e.input_sha256: e for e in entries}
        for fr in files_report:
            if fr.get("status") == "quarantined":
                sha = str(fr.get("input_sha256", ""))
                ent = entry_by_sha.get(sha)
                if ent is not None and sha in raw_path_by_sha:
                    q_dest = output_dir / "quarantine" / f"{ent.rel_input_path}.enc"
                    if not q_dest.exists():
                        vault.write_quarantined_blob(q_dest, raw_path_by_sha[sha].read_bytes(), sha)
    secret_for_audit = ""

    records_out.sort(key=lambda r: (r["file"], r["loc"]))
    with open(output_dir / "records.jsonl", "w", encoding="utf-8", newline="\n") as fh:
        for row in records_out:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    n_san = sum(1 for f in files_report if f["status"] == "sanitized")
    n_quar = sum(1 for f in files_report if f["status"] == "quarantined")
    report = {
        "schema": "raksha-report/1",
        "files": files_report,
        "counts": {
            "files_total": len(files_report),
            "files_sanitized": n_san,
            "files_quarantined": n_quar,
            "records_processed": len(records_out),
            "identifiers_detected": sum(entity_counts.values()),
            "records_linked": sum(1 for r in records_out if r["patient"] is not None),
            "records_generalized": privacy_plan.records_generalized,
            "records_suppressed": privacy_plan.records_suppressed,
            "records_quarantined": audit_res.quarantined_files,
        },
        "entities": dict(sorted(entity_counts.items())),
        "timing": {"seconds": round(time.monotonic() - t0, 2)},
        "resources": {"peak_rss_mb": peak_rss_mb()},
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    keyring.clear()
    print(f"files {len(files_report)} sanitized {n_san} quarantined {n_quar} records {len(records_out)}")
    return 0

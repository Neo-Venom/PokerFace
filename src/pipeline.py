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
from src.ingest.json_handler import parse_json_entry, transform_json_iterative
from src.ingest.profiler import infer_role_from_header
from src.ingest.router import maybe_rename_rel_path, scan_input_tree
from src.ingest.text_handler import parse_text_entry, rewrite_text_preserving_lines
from src.ingest.xml_handler import parse_xml_entry, sanitize_xml_text
from src.linkage.cluster import link_and_assign_pseudonyms
from src.linkage.evidence import extract_all_evidence
from src.models import FileEntry
from src.nlp.date_shifter import shift_compact_hl7_date, shift_date_value, shift_dates_in_text
from src.nlp.indic_norm import shadow_normalize
from src.nlp.medical_guard import MedicalGuard
from src.nlp.pii_ner import PiiHarvester, extract_context_name_spans, resolve_and_scrub
from src.nlp.pii_regex import extract_regex_spans
from src.privacy.audit import run_egress_audit
from src.privacy.kanon import PrivacyPlan, build_privacy_plan, generalize_age_value, scrub_tehsils_and_ages_in_text
from src.security.keys import KeyRing
from src.security.safelog import install_pii_free_guards

_B64_RE = re.compile(r"^[A-Za-z0-9+/]{24,}={0,2}$")
_REL_PREFIX_RE = re.compile(r"(?i)^(?:S/o|D/o|W/o|C/O|son of|daughter of|wife of)\s+")
_PIN_6_RE = re.compile(r"\b([1-8]\d{5})\b")
_PIN3_MASKED_RE = re.compile(r"(?<!\d)([1-8]\d{2})\*{3}(?!\d)")
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
    rx = extract_regex_spans(text)
    ctx = extract_context_name_spans(text, guard, rx)
    prot = guard.protected_spans(text)
    for sp in rx + ctx:
        if sp.kind in ("PATIENT_NAME", "RELATIVE_NAME"):
            if not guard.is_protected_span(sp.start, sp.end, prot, strict=True):
                harvester.add_name(sp.text)
        elif sp.kind in ("MRN", "PHONE", "AADHAAR", "ABHA_NUMBER", "PAN", "INSURANCE_ID"):
            if not any(ps <= sp.start and sp.end <= pe and tag == "MEDICAL" for ps, pe, tag in prot):
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
                        role, _ = infer_role_from_header(ks)
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
                        elif kl in ("line", "address", "address_note", "source_path", "remarks", "notes", "div"):
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


def _rewrite_entry(
    entry: FileEntry,
    guard: MedicalGuard,
    harvester: PiiHarvester,
    entity_counts: Counter[str],
    plan: PrivacyPlan,
) -> str:
    if entry.is_clean_doc:
        if entry.file_type in ("csv", "tsv"):
            return serialize_csv_entry(entry, scrub_header_fn=lambda h: h, rewrite_row_fn=lambda _i, r: r)
        if entry.file_type in ("json", "fhir"):
            return json.dumps(entry.parsed_data, ensure_ascii=False, indent=2) + "\n"
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
        cleaned, spans = resolve_and_scrub(
            text, guard, harvester, psn=psn, replace_primary_id_with_psn=inject_psn
        )
        for sp in spans:
            entity_counts[sp.kind] += 1
        uid = rec_uid or (entry.records[0].record_uid if entry.records else "")
        delta = plan.rec_shift_days.get(uid, file_shift)
        anchor = plan.rec_anchor_year.get(uid, 2026)
        age_band = plan.rec_age_band.get(uid)
        sup_teh = (uid in plan.rec_suppress_tehsil) if uid else True
        pin_mode = plan.rec_pin_level.get(uid, "suppress")
        cleaned = scrub_tehsils_and_ages_in_text(cleaned, age_band, sup_teh, plan.tehsil_rx, guard)
        cleaned = shift_dates_in_text(cleaned, delta_days=delta, dmy=dmy, anchor_year=anchor, guard=guard)
        cleaned = _apply_pin_policy(cleaned, pin_mode, is_address=is_address)
        return cleaned

    if entry.file_type in ("csv", "tsv"):
        roles = entry.column_roles
        headers = [str(h).lower() for h in (entry.parsed_data[0] if entry.parsed_data else [])]
        # Pick primary ID column for PSN injection (prefer patient UHID/Reg/PatientID over bill_no/sample_id)
        psn_col_idx: int | None = None
        for idx, role in roles.items():
            hdr = headers[idx] if idx < len(headers) else ""
            if role == "MRN" and not any(x in hdr for x in ("bill", "sample")):
                psn_col_idx = idx
                break
        if psn_col_idx is None:
            for idx, role in roles.items():
                if role == "PATIENT_NAME":
                    psn_col_idx = idx
                    break
        text_psn_col_idx: int | None = None
        if psn_col_idx is None:
            for idx, role in roles.items():
                if role in ("NOTES", "FREE_TEXT"):
                    text_psn_col_idx = idx

        def _scrub_header(h: str) -> str:
            cleaned, spans = resolve_and_scrub(h, guard, harvester)
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
                    out.append(_scrub_str(val, rec_uid=uid, is_address=True))
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
        psn_placed_for_ptr: set[str] = set()

        def _rec_for_path(path: str) -> tuple[str, Any]:
            if doc_rec is not None:
                return "doc", doc_rec
            for ptr, r_obj in rec_by_ptr.items():
                if path == ptr or path.startswith(ptr + "/"):
                    return ptr, r_obj
            return "", None

        def _scrub_key(k: str) -> str:
            cleaned, spans = resolve_and_scrub(k, guard, harvester)
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
                role, _ = infer_role_from_header(key_name)
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

            role, _ = infer_role_from_header(key_name)
            if r_psn and ptr_key not in psn_placed_for_ptr and (kl in ("patient_ref", "hospital_uhid", "uhid", "reg_code", "hosp_ref", "subject_ref") or role == "MRN"):
                entity_counts["MRN"] += 1
                psn_placed_for_ptr.add(ptr_key)
                return r_psn
            if kl in ("first", "last", "local", "name", "full_name") and not any(x in pl for x in ("/facility/", "/lab", "/panels/", "/diagnosis/")):
                if "/guardian/" in pl:
                    entity_counts["RELATIVE_NAME"] += 1
                    return "[RELATIVE]"
                if "/raised_by/" in pl:
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
                entity_counts[role or "MRN"] += 1
                return "[ID]"
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
            return _scrub_str(val, rec_uid=r_uid)

        cleaned = transform_json_iterative(entry.parsed_data, scrub_key_fn=_scrub_key, scrub_val_fn=_scrub_val)
        return json.dumps(cleaned, ensure_ascii=False, indent=2) + "\n"

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

        return sanitize_xml_text(entry.parsed_data, scrub_fn=_scrub_xml)

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
    entries: list[FileEntry] = []
    for _path, _rel, entry in scanned:
        _parse_entry(entry, pol)
        entries.append(entry)

    # Pass 1: Learn runtime clinical lexicon & harvest PII registry
    guard = MedicalGuard()
    guard.learn_from_entries(entries)

    harvester = PiiHarvester(guard)
    for entry in entries:
        _harvest_entry(entry, guard, harvester)
    harvester.compile()

    # Global Cross-Script Record Linkage (Phase 4)
    evs = extract_all_evidence(entries, guard)
    link_and_assign_pseudonyms(evs, keyring, pol)

    # Adversarial Privacy Layer (Phase 5: per-patient date shifts + k-anonymity plan)
    privacy_plan = build_privacy_plan(entries, evs, keyring, pol)

    # Rename PII-bearing filenames deterministically using harvested PII + regex
    used_out_paths: set[str] = set()
    for entry in entries:
        if entry.status != "sanitized" or not entry.rel_output_path:
            continue

        def _fname_has_pii(stem_text: str) -> bool:
            _, spans = resolve_and_scrub(stem_text, guard, harvester)
            return bool(spans)

        new_rel, renamed = maybe_rename_rel_path(
            entry.rel_input_path,
            entry.input_sha256,
            keyring,
            has_pii_fn=_fname_has_pii,
        )
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
            dest.parent.mkdir(parents=True, exist_ok=True)
            serialized = _rewrite_entry(entry, guard, harvester, entity_counts, privacy_plan)
            dest.write_text(serialized, encoding="utf-8", newline="")
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

    audit_res = run_egress_audit(
        entries=entries,
        san_dir=san_dir,
        files_report=files_report,
        records_out=records_out,
        guard=guard,
        harvester=harvester,
        secret_str=secret_for_audit,
    )
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

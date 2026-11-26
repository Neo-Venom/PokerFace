"""Pre-emission global egress audit gate: decodes Base64, sweeps residual PII, verifies secret hygiene."""
from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import regex as re

from src.models import FileEntry
from src.nlp.indic_norm import norm_match_text, shadow_normalize
from src.nlp.medical_guard import MedicalGuard
from src.nlp.pii_ner import PiiHarvester, resolve_and_scrub
from src.nlp.pii_regex import extract_regex_spans

_B64_RE = re.compile(r"[A-Za-z0-9+/]{24,}={0,2}")
_CRITICAL_RX_KINDS = {"ABHA_NUMBER", "ABHA_ADDRESS", "EMAIL", "PAN", "AADHAAR_MASKED", "INSURANCE_ID"}
_UUID_RE = re.compile(r"(?i)\b(?:urn:uuid:)?[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b")
_FHIR_STRUCT_KEYS = {"fullUrl", "id", "reference", "resourceType", "system", "code", "url", "versionId", "profile"}


@dataclass
class AuditSummary:
    rescrubbed_files: int = 0
    quarantined_files: int = 0
    secret_clean: bool = True


def _mask_uuids(text: str) -> str:
    """Replace UUIDs with equal-length non-digit placeholders so audit regexes never split hex segments."""
    return _UUID_RE.sub(lambda m: "u" * len(m.group(0)), text)


def _decode_b64_snippets(text: str) -> str:
    extra: list[str] = []
    for m in _B64_RE.finditer(text):
        tok = m.group(0)
        try:
            raw = base64.b64decode(tok + "=" * (-len(tok) % 4))
            dec = raw.decode("utf-8", errors="ignore")
            if dec and not dec.startswith("[ATTACHMENT_REDACTED]"):
                extra.append(dec)
        except Exception:
            pass
    return text if not extra else (text + "\n" + "\n".join(extra))


def _has_residual_pii(
    text: str,
    guard: MedicalGuard,
    harvester: PiiHarvester,
    secret_str: str,
) -> bool:
    if secret_str and len(secret_str) >= 4 and secret_str in text:
        return True
    combined = _mask_uuids(_decode_b64_snippets(text))
    prot = guard.protected_spans(combined)
    for sp in extract_regex_spans(combined):
        if sp.kind in _CRITICAL_RX_KINDS and not guard.is_protected_span(sp.start, sp.end, prot, strict=True):
            return True
    for sp in harvester.sweep_spans(combined):
        if not guard.is_protected_span(sp.start, sp.end, prot, strict=True):
            return True
    # Direct exact match against harvested phone/Aadhaar/ABHA >= 10 digits
    for m in re.finditer(r"(?<!\d)(\d{10,14})(?!\d)", shadow_normalize(combined)):
        if m.group(1) in harvester.digit_cores:
            if not guard.is_protected_span(m.start(), m.end(), prot, strict=True):
                return True
    return False


def _scrub_json_tree_fallback(
    obj: Any,
    guard: MedicalGuard,
    harvester: PiiHarvester,
    key_name: str = "",
    preserve_struct: bool = False,
) -> Any:
    if isinstance(obj, dict):
        is_struct = preserve_struct or obj.get("resourceType") in (
            "Bundle",
            "Patient",
            "Encounter",
            "Observation",
            "Condition",
            "Procedure",
            "MedicationRequest",
            "DiagnosticReport",
            "DocumentReference",
        )
        return {
            (str(k) if (is_struct and str(k) in _FHIR_STRUCT_KEYS) else resolve_and_scrub(str(k), guard, harvester)[0]): _scrub_json_tree_fallback(
                v, guard, harvester, key_name=str(k), preserve_struct=is_struct
            )
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [_scrub_json_tree_fallback(x, guard, harvester, key_name=key_name, preserve_struct=preserve_struct) for x in obj]
    if isinstance(obj, str):
        st = obj.strip()
        if _UUID_RE.fullmatch(st) or (preserve_struct and key_name in _FHIR_STRUCT_KEYS):
            return obj
        if _B64_RE.match(st):
            try:
                raw = base64.b64decode(st + "=" * (-len(st) % 4)).decode("utf-8", errors="strict")
                cleaned, _ = resolve_and_scrub(raw, guard, harvester)
                return base64.b64encode(cleaned.encode("utf-8")).decode("ascii")
            except Exception:
                return obj
        return resolve_and_scrub(obj, guard, harvester)[0]
    return obj


def run_egress_audit(
    entries: list[FileEntry],
    san_dir: Path,
    files_report: list[dict[str, Any]],
    records_out: list[dict[str, Any]],
    guard: MedicalGuard,
    harvester: PiiHarvester,
    secret_str: str = "",
) -> AuditSummary:
    """Scan all sanitized outputs and decoded Base64; re-scrub or quarantine fail-closed on residual PII."""
    summary = AuditSummary()
    entry_by_sha = {e.input_sha256: e for e in entries}
    quarantined_rel_paths: set[str] = set()

    for f_rec in files_report:
        if f_rec.get("status") != "sanitized" or not f_rec.get("output"):
            continue
        out_str = str(f_rec["output"])
        rel_out = out_str.removeprefix("sanitized/")
        dest = san_dir / rel_out
        entry = entry_by_sha.get(str(f_rec.get("input_sha256", "")))
        if not dest.is_file():
            f_rec["status"] = "quarantined"
            f_rec["output"] = None
            f_rec["reason"] = "missing_output"
            quarantined_rel_paths.add(out_str)
            summary.quarantined_files += 1
            continue

        if secret_str and len(secret_str) >= 4 and secret_str in rel_out:
            summary.secret_clean = False

        if entry is not None and entry.is_clean_doc:
            # Only verify no leaked secret in clean docs
            raw_clean = dest.read_text(encoding="utf-8", errors="replace")
            if secret_str and len(secret_str) >= 4 and secret_str in raw_clean:
                summary.secret_clean = False
                dest.unlink(missing_ok=True)
                f_rec["status"] = "quarantined"
                f_rec["output"] = None
                f_rec["reason"] = "egress_secret_leak"
                quarantined_rel_paths.add(out_str)
                summary.quarantined_files += 1
            continue

        txt = dest.read_text(encoding="utf-8", errors="replace")
        # Check residual PII in filename or file content (+ decoded Base64)
        fname_dirty = _has_residual_pii(rel_out, guard, harvester, secret_str)
        body_dirty = _has_residual_pii(txt, guard, harvester, secret_str)

        if body_dirty:
            try:
                if entry and entry.file_type in ("json", "fhir"):
                    data = json.loads(txt)
                    is_fhir = entry.file_type == "fhir" or (isinstance(data, dict) and data.get("resourceType") == "Bundle")
                    rescrubbed = _scrub_json_tree_fallback(data, guard, harvester, preserve_struct=is_fhir)
                    new_txt = json.dumps(rescrubbed, ensure_ascii=False, indent=2) + "\n"
                else:
                    lines = txt.splitlines(keepends=True)
                    new_txt = "".join(resolve_and_scrub(ln, guard, harvester)[0] for ln in lines)
                dest.write_text(new_txt, encoding="utf-8", newline="")
                summary.rescrubbed_files += 1
                txt = new_txt
                body_dirty = _has_residual_pii(txt, guard, harvester, secret_str)
            except Exception:
                body_dirty = True

        # Fail closed only on leaked secret, malformed JSON, or persistent critical ID after re-scrubbing
        json_broken = False
        if entry and entry.file_type in ("json", "fhir"):
            try:
                json.loads(txt)
            except ValueError:
                json_broken = True

        if fname_dirty or json_broken or (secret_str and len(secret_str) >= 4 and secret_str in txt):
            dest.unlink(missing_ok=True)
            f_rec["status"] = "quarantined"
            f_rec["output"] = None
            f_rec["reason"] = "egress_audit_fail"
            quarantined_rel_paths.add(out_str)
            summary.quarantined_files += 1
            continue

        if body_dirty:
            prot = guard.protected_spans(_decode_b64_snippets(txt))
            critical_still = any(
                sp.kind in _CRITICAL_RX_KINDS and not guard.is_protected_span(sp.start, sp.end, prot, strict=True)
                for sp in extract_regex_spans(_decode_b64_snippets(txt))
            )
            if critical_still:
                dest.unlink(missing_ok=True)
                f_rec["status"] = "quarantined"
                f_rec["output"] = None
                f_rec["reason"] = "egress_critical_leak"
                quarantined_rel_paths.add(out_str)
                summary.quarantined_files += 1

    if quarantined_rel_paths:
        records_out[:] = [r for r in records_out if r.get("file") not in quarantined_rel_paths]

    # Verify filename hashes / report schema safety against secret leakage
    if secret_str and len(secret_str) >= 4:
        for r in records_out:
            if secret_str in str(r):
                summary.secret_clean = False
    return summary

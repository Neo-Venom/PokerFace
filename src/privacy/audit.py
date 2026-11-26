"""Pre-emission global egress audit gate: decodes Base64, sweeps residual PII, verifies secret hygiene."""
from __future__ import annotations

import base64
import gzip
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import regex as re

from src.ingest.hidden import binary_text_runs, decoded_views, scrub_hidden
from src.ingest.json_handler import is_json_lines, serialize_json_entry, transform_json_iterative
from src.models import FileEntry
from src.nlp.indic_norm import deobfuscate_unicode, norm_match_text, shadow_normalize, unquote_percent_text
from src.nlp.medical_guard import MedicalGuard
from src.nlp.pii_ner import PiiHarvester, resolve_and_scrub, split_safe_chunks
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


def _decode_b64_snippets(text: str, depth: int = 0) -> str:
    """`text` plus every hidden reading of it: %XX, HTML entities, \\uXXXX, '_'-joined words, BiDi runs, and
    base64 / gzip+base64 / hex payloads (nested), with the readable runs of binary blobs."""
    extra: list[str] = []
    if depth == 0:
        unq = unquote_percent_text(text)
        if unq != text:
            extra.append(unq)
        extra.extend(decoded_views(text))
    for m in _B64_RE.finditer(text):
        tok = m.group(0)
        try:
            raw = base64.b64decode(tok + "=" * (-len(tok) % 4))
            if raw.startswith(b"\x1f\x8b"):
                raw = gzip.decompress(raw)
            try:
                dec = raw.decode("utf-8", errors="strict")
            except UnicodeDecodeError:
                dec = binary_text_runs(raw)
            if dec and not dec.startswith("[ATTACHMENT_REDACTED]"):
                extra.append(dec)
                if depth < 2 and _B64_RE.search(dec):
                    extra.append(_decode_b64_snippets(dec, depth + 1))
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
    combined = deobfuscate_unicode(_mask_uuids(_decode_b64_snippets(text)))
    for chunk in split_safe_chunks(combined, max_len=32768):
        prot = guard.protected_spans(chunk)
        for sp in extract_regex_spans(chunk):
            if sp.kind in _CRITICAL_RX_KINDS and not guard.is_protected_span(sp.start, sp.end, prot, strict=True):
                return True
        for sp in harvester.sweep_spans(chunk):
            if not guard.is_protected_span(sp.start, sp.end, prot, strict=True):
                return True
        for m in re.finditer(r"(?<!\d)(\d{10,14})(?!\d)", shadow_normalize(chunk)):
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
    """Second-chance scrub of a whole JSON tree (iterative: safe at any nesting depth). FHIR structure keys and
    their values, and UUIDs, are kept so the Bundle stays valid."""

    def _plain(s: str) -> str:
        return resolve_and_scrub(unquote_percent_text(s), guard, harvester)[0]

    def _key(k: str) -> str:
        return k if (preserve_struct and k in _FHIR_STRUCT_KEYS) else resolve_and_scrub(k, guard, harvester)[0]

    def _val(_path: str, v: Any, key: str) -> Any:
        if not isinstance(v, str):
            return v
        if _UUID_RE.fullmatch(v.strip()) or (preserve_struct and key in _FHIR_STRUCT_KEYS):
            return v
        return scrub_hidden(v, _plain)

    return transform_json_iterative(obj, scrub_key_fn=_key, scrub_val_fn=_val)


def _parse_json_output(txt: str, jsonl: bool) -> Any:
    """Re-parse an exported JSON / JSON Lines file (raises ValueError when it is not valid)."""
    def _no_const(name: str) -> Any:
        raise ValueError(name)

    if not jsonl:
        return json.loads(txt, parse_constant=_no_const)
    return [json.loads(ln, parse_constant=_no_const) for ln in txt.splitlines() if ln.strip()]


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
                _parse_json_output(txt, is_json_lines(entry))
            except (ValueError, RecursionError):
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

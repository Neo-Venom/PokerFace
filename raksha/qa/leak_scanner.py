"""Step H: Independent post-export leakage scanner (zero shared state with detection engine).

Rescans exported files, filenames, base64 payloads, report.json, and records.jsonl
for known identifier surfaces, high-confidence aliases, and raw PII patterns.
On a hit, independently re-evaluates clinical/doctor context and either re-scrubs
safely or quarantines the file if unsafe.
"""
from __future__ import annotations

import base64
import binascii
import csv
import io
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

_ZERO_WIDTH_SET = {"\u200b", "\u200c", "\u200d", "\ufeff", "\u00ad"}
_B64_SCAN_RE = re.compile(r"[A-Za-z0-9+/]{24,}={0,2}")

# Independent raw identifier checks
_QA_RAW_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("ABHA_NUMBER", re.compile(r"(?<!\d)\d{2}[\s\-]\d{4}[\s\-]\d{4}[\s\-]\d{4}(?!\d)")),
    ("AADHAAR", re.compile(r"(?<!\d)\d{4}[\s\-]\d{4}[\s\-]\d{4}(?!\d)")),
    ("AADHAAR_MASKED", re.compile(r"(?<!\w)[xX*]{4}[\s\-]?[xX*]{4}[\s\-]?\d{4}(?!\w)")),
    ("PHONE", re.compile(r"(?<!\d)(?:\+?91[\s.\-]?|0)?[6-9]\d{4}[\s.\-]?\d{5}(?!\d)")),
    ("PAN", re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b")),
    ("EMAIL", re.compile(r"(?<![\w.+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
]

_QA_DOCTOR_BEFORE_RE = re.compile(
    r"(?i)(?:\bdr\.?|doctor|डॉ\.?|डॉक्टर|consultant|prescriber|ref(?:erred)?\s*by)\s*[:\-]?\s*(?:dr\.?\s*)?$"
)
_QA_CLINICAL_AFTER_RE = re.compile(
    r"(?i)^(?:'s\s+(?:disease|syndrome|palsy|lymphoma|sign)|\s+(?:disease|syndrome|palsy|workers?)|\s*(?:/uL|mg/dL|g/dL|mmol/L|mIU/L|mg\b|mcg\b|ml\b|IU\b|%|mmHg|/min|bpm|days?\b|wks?\b|[-–]\s*\d+))"
)
_QA_CLINICAL_BEFORE_RE = re.compile(
    r"(?i)(?:platelet(?:\s+count)?|wbc(?:\s+count)?|range|ref(?:erence)?\s*range|bp|pulse|temp|dose|tab\.?|cap\.?|inj\.?|dolo|paracetamol|tehsil|distt\.?|district|hospital|तहसील|जिला)\s*[:=\-]?\s*$"
)


@dataclass
class ScanHit:
    start: int
    end: int
    label: str
    surface: str


@dataclass
class AuditSummary:
    files_scanned: int = 0
    hits_found: int = 0
    hits_ignored_context: int = 0
    files_rescrubbed: int = 0
    files_quarantined_qa: int = 0


def _qa_norm_with_map(text: str) -> tuple[str, list[tuple[int, int]]]:
    """Independent NFKC + zero-width strip + Devanagari-digit normalization with exact offset map."""
    clusters: list[tuple[str, int, int]] = []
    for idx, ch in enumerate(text):
        if ch in _ZERO_WIDTH_SET:
            continue
        if clusters and unicodedata.combining(ch) > 0:
            prev, ps, _ = clusters[-1]
            clusters[-1] = (prev + ch, ps, idx + 1)
        else:
            clusters.append((ch, idx, idx + 1))

    out_chars: list[str] = []
    out_map: list[tuple[int, int]] = []
    for seg, os_, oe_ in clusters:
        nseg = unicodedata.normalize("NFKC", seg)
        for c in nseg:
            code = ord(c)
            if 0x0966 <= code <= 0x096F:
                c = chr(ord("0") + (code - 0x0966))
            out_chars.append(c)
            out_map.append((os_, oe_))
    return "".join(out_chars), out_map


def _is_safe_context(norm_text: str, ns: int, ne: int, surface: str, label: str, doctor_surfaces: set[str]) -> bool:
    """Independently re-evaluate surrounding context on a hit."""
    before = norm_text[max(0, ns - 40) : ns]
    after = norm_text[ne : min(len(norm_text), ne + 35)]

    # 1. Doctor protection: explicit Dr./Consultant prefix or matches known doctor name inside doctor context
    if _QA_DOCTOR_BEFORE_RE.search(before):
        return True
    if surface.casefold() in doctor_surfaces and re.search(r"(?i)\b(?:dr\.?|डॉ|consultant)\b", before[-25:]):
        return True

    # 2. Disease eponyms, ASHA worker, or immediate clinical units/ranges
    if _QA_CLINICAL_AFTER_RE.match(after):
        return True
    if label in ("PHONE", "AADHAAR", "ABHA_NUMBER", "VILLAGE") and _QA_CLINICAL_BEFORE_RE.search(before):
        return True

    # 3. Never flag inside an already-redacted placeholder tag like [PATIENT_NAME]
    if before.endswith("[") and after.startswith("]"):
        return True

    return False


class _CompiledSurfaceMatcher:
    """Precompiles all known PII surfaces into a single alternation regex once per QA scan."""

    def __init__(self, known_surfaces: dict[str, str]):
        self.surface_to_label: dict[str, str] = {}
        ordered_norms: list[str] = []
        for surf, lbl in sorted(known_surfaces.items(), key=lambda kv: (-len(kv[0]), kv[0])):
            s_norm, _ = _qa_norm_with_map(surf)
            s_low = s_norm.casefold().strip()
            if len(s_low) < 4 or s_low in self.surface_to_label:
                continue
            self.surface_to_label[s_low] = lbl
            ordered_norms.append(re.escape(s_low))

        if ordered_norms:
            # Sort longest normalized surface first so longest match wins in regex alternation
            ordered_norms.sort(key=len, reverse=True)
            self.pattern: re.Pattern | None = re.compile(
                rf"(?<![A-Za-z0-9\u0900-\u097F])(?:{'|'.join(ordered_norms)})(?![A-Za-z0-9\u0900-\u097F])"
            )
        else:
            self.pattern = None


def _find_hits(
    text: str,
    known_surfaces: dict[str, str] | _CompiledSurfaceMatcher,
    doctor_surfaces: set[str],
) -> tuple[list[ScanHit], int]:
    """Scan `text` independently for raw identifier regexes and known PII surfaces."""
    matcher = known_surfaces if isinstance(known_surfaces, _CompiledSurfaceMatcher) else _CompiledSurfaceMatcher(known_surfaces)
    norm_text, offset_map = _qa_norm_with_map(text)
    norm_low = norm_text.casefold()
    raw_hits: list[ScanHit] = []
    ignored = 0

    def map_span(ns: int, ne: int) -> tuple[int, int]:
        if ns >= ne or not offset_map:
            return (0, 0)
        s = max(0, min(ns, len(offset_map) - 1))
        e = max(s + 1, min(ne, len(offset_map)))
        return (offset_map[s][0], offset_map[e - 1][1])

    # 1. Raw identifier regexes
    for lbl, rx in _QA_RAW_PATTERNS:
        for m in rx.finditer(norm_text):
            if _is_safe_context(norm_text, m.start(), m.end(), m.group(0), lbl, doctor_surfaces):
                ignored += 1
                continue
            os_, oe_ = map_span(m.start(), m.end())
            if oe_ > os_:
                raw_hits.append(ScanHit(os_, oe_, lbl, text[os_:oe_]))

    # 2. Precompiled single-pass known surface matcher
    if matcher.pattern is not None:
        for m in matcher.pattern.finditer(norm_low):
            s_low = m.group(0)
            lbl = matcher.surface_to_label.get(s_low, "PATIENT_NAME")
            if _is_safe_context(norm_text, m.start(), m.end(), s_low, lbl, doctor_surfaces):
                ignored += 1
                continue
            os_, oe_ = map_span(m.start(), m.end())
            if oe_ > os_:
                raw_hits.append(ScanHit(os_, oe_, lbl, text[os_:oe_]))

    # Resolve overlaps: earlier, longer wins
    raw_hits.sort(key=lambda h: (h.start, -(h.end - h.start)))
    kept: list[ScanHit] = []
    last_end = -1
    for h in raw_hits:
        if h.start >= last_end:
            kept.append(h)
            last_end = h.end
    return kept, ignored


def _apply_rescrub(text: str, hits: list[ScanHit]) -> str:
    if not hits:
        return text
    buf: list[str] = []
    cur = 0
    for h in hits:
        if h.start < cur:
            continue
        buf.append(text[cur : h.start])
        buf.append(f"[{h.label}]")
        cur = h.end
    buf.append(text[cur:])
    return "".join(buf)


def _rescrub_string_and_b64(
    val: str,
    known_surfaces: dict[str, str],
    doctor_surfaces: set[str],
    summary: AuditSummary,
) -> tuple[str, bool]:
    """Scan a scalar string (and any embedded base64 UTF-8 payload), returning (cleaned, changed)."""
    changed = False
    s = val.strip()
    if len(s) >= 24 and _B64_SCAN_RE.fullmatch(s):
        try:
            padded = s + "=" * (-len(s) % 4)
            raw = base64.b64decode(padded, validate=True)
            if not raw.startswith((b"%PDF", b"\xff\xd8\xff", b"\x89PNG", b"PK\x03\x04")):
                dec = raw.decode("utf-8", errors="strict")
                if dec and sum(1 for c in dec if c.isprintable() or c in "\r\n\t") / len(dec) >= 0.85:
                    hits, ign = _find_hits(dec, known_surfaces, doctor_surfaces)
                    summary.hits_found += len(hits)
                    summary.hits_ignored_context += ign
                    if hits:
                        dec_clean = _apply_rescrub(dec, hits)
                        return base64.b64encode(dec_clean.encode("utf-8")).decode("ascii"), True
        except (binascii.Error, UnicodeDecodeError, ValueError):
            pass

    hits, ign = _find_hits(val, known_surfaces, doctor_surfaces)
    summary.hits_found += len(hits)
    summary.hits_ignored_context += ign
    if hits:
        return _apply_rescrub(val, hits), True
    return val, changed


def _rescrub_json_iterative(
    data: object,
    known_surfaces: dict[str, str],
    doctor_surfaces: set[str],
    summary: AuditSummary,
) -> tuple[object, bool]:
    """Iteratively scan and re-scrub JSON string leaves without recursion limit risk."""
    changed = False
    root = {"r": data}
    stack: list[tuple[object, object, object]] = [(root, "r", data)]
    while stack:
        parent, slot, val = stack.pop()
        if isinstance(val, dict):
            for k, v in val.items():
                stack.append((val, k, v))
        elif isinstance(val, list):
            for i, v in enumerate(val):
                stack.append((val, i, v))
        elif isinstance(val, str):
            new_s, ch = _rescrub_string_and_b64(val, known_surfaces, doctor_surfaces, summary)
            if ch:
                parent[slot] = new_s  # type: ignore[index]
                changed = True
    return root["r"], changed


def _quarantine_file(
    san_file: Path,
    san_rel: str,
    report: dict,
    records_path: Path,
    reason: str = "residual_pii_quarantined",
) -> None:
    """Safely quarantine a file that failed post-export QA: remove from sanitized/ and update report.json & records.jsonl."""
    if san_file.is_file():
        san_file.unlink()
    target_out = f"sanitized/{san_rel}"
    for ent in report.get("files") or []:
        if ent.get("output") == target_out:
            ent["output"] = None
            ent["status"] = "quarantined"
            ent["reason"] = reason
            break

    if records_path.is_file():
        kept_lines: list[str] = []
        for line in records_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                if rec.get("file") != target_out:
                    kept_lines.append(line)
            except ValueError:
                continue
        records_path.write_text("\n".join(kept_lines) + ("\n" if kept_lines else ""), encoding="utf-8")


def scan_and_remediate(
    output_dir: Path,
    known_surfaces: dict[str, str],
    doctor_surfaces: set[str],
) -> AuditSummary:
    """Independently rescan exported outputs and re-scrub or quarantine on residual PII hits."""
    summary = AuditSummary()
    matcher = _CompiledSurfaceMatcher(known_surfaces)
    san_dir = output_dir / "sanitized"
    report_path = output_dir / "report.json"
    records_path = output_dir / "records.jsonl"

    if not san_dir.is_dir() or not report_path.is_file():
        return summary

    report = json.loads(report_path.read_text(encoding="utf-8"))
    exported_files = sorted((p for p in san_dir.rglob("*") if p.is_file()), key=lambda p: p.relative_to(san_dir).as_posix())

    for path in exported_files:
        rel = path.relative_to(san_dir).as_posix()
        summary.files_scanned += 1

        # 1. Check filename / path itself for any leak; quarantine if filename leaks
        fn_hits, fn_ign = _find_hits(rel.replace("_", " "), matcher, doctor_surfaces)
        summary.hits_found += len(fn_hits)
        summary.hits_ignored_context += fn_ign
        if fn_hits:
            _quarantine_file(path, rel, report, records_path, reason="filename_leak_quarantined")
            summary.files_quarantined_qa += 1
            continue

        # 2. Scan file content & attempt re-scrub if hit found
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            _quarantine_file(path, rel, report, records_path, reason="invalid_output_encoding")
            summary.files_quarantined_qa += 1
            continue

        ext = path.suffix.lower()
        file_changed = False
        re_scrubbed_text = text

        is_json_file = ext == ".json"
        if not is_json_file and ext in (".csv", ".tsv") and text.lstrip().startswith(("{", "[")):
            try:
                json.loads(text)
                is_json_file = True
            except (ValueError, RecursionError):
                is_json_file = False

        if is_json_file:
            try:
                data = json.loads(text)
                cleaned_data, file_changed = _rescrub_json_iterative(data, matcher, doctor_surfaces, summary)
                if file_changed:
                    re_scrubbed_text = json.dumps(cleaned_data, ensure_ascii=False, indent=2) + "\n"
            except (ValueError, RecursionError):
                _quarantine_file(path, rel, report, records_path, reason="malformed_output_json")
                summary.files_quarantined_qa += 1
                continue
        elif ext in (".csv", ".tsv"):
            delim = "\t" if ext == ".tsv" else ","
            try:
                rows = list(csv.reader(io.StringIO(text, newline=""), delimiter=delim))
                new_rows: list[list[str]] = []
                for row in rows:
                    new_row: list[str] = []
                    for cell in row:
                        nc, ch = _rescrub_string_and_b64(cell, matcher, doctor_surfaces, summary)
                        if ch:
                            file_changed = True
                        new_row.append(nc)
                    new_rows.append(new_row)
                if file_changed:
                    buf = io.StringIO()
                    csv.writer(buf, delimiter=delim, lineterminator="\n").writerows(new_rows)
                    re_scrubbed_text = buf.getvalue()
            except csv.Error:
                _quarantine_file(path, rel, report, records_path, reason="malformed_output_csv")
                summary.files_quarantined_qa += 1
                continue
        else:
            re_scrubbed_text, file_changed = _rescrub_string_and_b64(text, matcher, doctor_surfaces, summary)

        if file_changed:
            path.write_text(re_scrubbed_text, encoding="utf-8", newline="")
            summary.files_rescrubbed += 1

            # Second verification pass: if ANY hit still survives (e.g. embedded base64/unscrubbed artifact), quarantine
            check_text = re_scrubbed_text
            if ext == ".json":
                # Decode any remaining base64 strings for verification
                extras = []
                for m in _B64_SCAN_RE.finditer(re_scrubbed_text):
                    try:
                        extras.append(base64.b64decode(m.group(0) + "=" * (-len(m.group(0)) % 4)).decode("utf-8", errors="ignore"))
                    except (binascii.Error, ValueError):
                        pass
                if extras:
                    check_text = re_scrubbed_text + "\n" + "\n".join(extras)
            rem_hits, _ = _find_hits(check_text, matcher, doctor_surfaces)
            if rem_hits:
                _quarantine_file(path, rel, report, records_path, reason="residual_pii_quarantined")
                summary.files_quarantined_qa += 1

    # Recompute honest report counts if QA quarantined any files
    if summary.files_quarantined_qa > 0:
        files_list = report.get("files") or []
        n_san = sum(1 for e in files_list if e.get("status") == "sanitized")
        rec_count = 0
        linked_count = 0
        gen_count = 0
        sup_count = 0
        if records_path.is_file():
            for line in records_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rec_count += 1
                    try:
                        obj = json.loads(line)
                        if obj.get("patient") is not None:
                            linked_count += 1
                        if obj.get("status") == "generalized":
                            gen_count += 1
                        elif obj.get("status") == "suppressed":
                            sup_count += 1
                    except ValueError:
                        pass
        counts = report.setdefault("counts", {})
        counts["files_total"] = len(files_list)
        counts["files_sanitized"] = n_san
        counts["files_quarantined"] = len(files_list) - n_san
        counts["records_processed"] = rec_count
        counts["records_linked"] = linked_count
        counts["records_generalized"] = gen_count
        counts["records_suppressed"] = sup_count

    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return summary

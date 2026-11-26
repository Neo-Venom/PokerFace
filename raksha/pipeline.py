"""Core deterministic sanitization pipeline orchestration."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import sys
import time
from pathlib import Path

from raksha.config import derive_subkey, load_policy, require_secret
from raksha.detect.crf_model import CRFPredictor
from raksha.detect.identifiers import (
    derive_id_patterns_from_values,
    detect_identifier_spans,
    redact_spans,
)
from raksha.detect.names_context import (
    detect_context_name_and_address_spans,
)
from raksha.detect.normalize import build_normalized_view
from raksha.detect.protected import build_clinical_lexicon, resolve_pii_vs_protected
from raksha.detect.registry import build_alias_registry
from raksha.detect.structured import (
    infer_column_type,
    process_table_rows,
    sanitize_header_or_key,
    sanitize_json_data,
)
from raksha.export.filenames import sanitize_relative_path
from raksha.identity.evidence import extract_all_evidences, get_json_record_units
from raksha.identity.linker import resolve_clusters
from raksha.identity.pseudonymizer import (
    inject_psn_json_obj,
    inject_psn_tabular_row,
    inject_psn_text,
    issue_cluster_pseudonyms,
)
from raksha.ingest.loader import SUPPORTED_EXTS, decode_bytes, load_json_safe, load_tabular, sniff_format
from raksha.qa.leak_scanner import scan_and_remediate


def peak_rss_mb() -> float:
    try:
        import resource

        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return round(r / 1024 if sys.platform != "darwin" else r / 1048576, 1)
    except Exception:
        return 0.0


def _collect_mrn_examples(parsed_items: list[tuple[str, str, object]]) -> list[str]:
    """Pre-scan structured tables/JSONs to learn hospital-specific MRN/UHID formats from the data."""
    mrns: list[str] = []
    for _rel, fmt, payload in parsed_items:
        if fmt in ("csv", "tsv") and isinstance(payload, list) and len(payload) > 1:
            header, body = payload[0], payload[1:]
            for c_idx, h in enumerate(header):
                vals = [r[c_idx] if c_idx < len(r) else "" for r in body]
                if infer_column_type(vals, header=h) == "MRN":
                    mrns.extend(str(v).strip() for v in vals if str(v).strip())
        elif fmt == "json" and isinstance(payload, dict):
            # Iterative walk of list-of-dict records for MRN columns
            stack = [payload]
            while stack:
                node = stack.pop()
                if isinstance(node, dict):
                    for k, v in node.items():
                        if isinstance(v, list) and v and all(isinstance(x, dict) for x in v):
                            for rk in v[0].keys():
                                vals = [row.get(rk) for row in v if rk in row]
                                if infer_column_type(vals, header=rk, parent_key=k) == "MRN":
                                    mrns.extend(str(x).strip() for x in vals if x is not None and str(x).strip())
                        elif isinstance(v, (dict, list)):
                            stack.append(v)
                elif isinstance(node, list):
                    for item in node:
                        if isinstance(item, (dict, list)):
                            stack.append(item)
    return mrns


def _process_tabular(
    rows: list[list[str]],
    delimiter: str,
    rel: str,
    out_rel: str,
    records: list[dict],
    record_to_psn: dict[tuple[str, str], str] | None = None,
    text_scrubber=None,
    stats: dict[str, int] | None = None,
) -> str:
    clean_header, clean_body = process_table_rows(rows, text_scrubber=text_scrubber, stats=stats)
    buf = io.StringIO()
    writer = csv.writer(buf, delimiter=delimiter, lineterminator="\n")
    writer.writerow(clean_header)
    for i, row in enumerate(clean_body):
        loc = f"row:{i}"
        psn = record_to_psn.get((rel, loc)) if record_to_psn is not None else None
        if psn is not None:
            row = inject_psn_tabular_row(clean_header, row, psn)
        writer.writerow(row)
        records.append({"file": f"sanitized/{out_rel}", "loc": loc, "patient": psn})
    return buf.getvalue()


def _process_json(
    data: object,
    rel: str,
    out_rel: str,
    records: list[dict],
    record_to_psn: dict[tuple[str, str], str] | None = None,
    text_scrubber=None,
    stats: dict[str, int] | None = None,
) -> str:
    cleaned = sanitize_json_data(data, text_scrubber=text_scrubber, stats=stats)
    units = get_json_record_units(data)
    if len(units) == 1 and units[0][0] == "doc":
        psn = record_to_psn.get((rel, "doc")) if record_to_psn is not None else None
        if psn is not None:
            cleaned = inject_psn_json_obj(cleaned, psn)
        records.append({"file": f"sanitized/{out_rel}", "loc": "doc", "patient": psn})
    else:
        key_to_cleaned: dict[str, str] = {}
        if isinstance(data, dict) and isinstance(cleaned, dict):
            for orig_k in data.keys():
                if orig_k in cleaned:
                    key_to_cleaned[orig_k] = orig_k
                else:
                    sk = sanitize_header_or_key(orig_k)
                    if sk in cleaned:
                        key_to_cleaned[orig_k] = sk

        for loc, _sub in units:
            psn = record_to_psn.get((rel, loc)) if record_to_psn is not None else None
            if psn is not None and isinstance(cleaned, dict) and loc.startswith("/"):
                parts = loc[1:].split("/", 1)
                if len(parts) == 2:
                    raw_pk, idx_s = parts
                    orig_k = raw_pk.replace("~1", "/").replace("~0", "~")
                    ck = key_to_cleaned.get(orig_k, orig_k)
                    if isinstance(cleaned.get(ck), list):
                        try:
                            idx = int(idx_s)
                            if 0 <= idx < len(cleaned[ck]):
                                cleaned[ck][idx] = inject_psn_json_obj(cleaned[ck][idx], psn)
                        except ValueError:
                            pass
            records.append({"file": f"sanitized/{out_rel}", "loc": loc, "patient": psn})

    return json.dumps(cleaned, ensure_ascii=False, indent=2) + "\n"


def _process_text(
    text: str,
    rel: str,
    out_rel: str,
    records: list[dict],
    record_to_psn: dict[tuple[str, str], str] | None = None,
    text_scrubber=None,
) -> str:
    psn = record_to_psn.get((rel, "doc")) if record_to_psn is not None else None
    records.append({"file": f"sanitized/{out_rel}", "loc": "doc", "patient": psn})
    cleaned = text_scrubber(text) if text_scrubber is not None else text
    if psn is not None:
        cleaned = inject_psn_text(cleaned, psn)
    return cleaned


def run_pipeline(input_dir: Path, output_dir: Path) -> int:
    """Run deterministic two-pass de-identification pipeline from input_dir to output_dir."""
    t0 = time.monotonic()
    try:
        secret = require_secret()
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    _policy = load_policy()
    _psn_key = derive_subkey(secret, b"pseudonym")
    _shift_key = derive_subkey(secret, b"date_shift")

    san_dir = output_dir / "sanitized"
    san_dir.mkdir(parents=True, exist_ok=True)
    sys.setrecursionlimit(5000)

    input_paths = sorted(
        (p for p in input_dir.rglob("*") if p.is_file()),
        key=lambda p: p.relative_to(input_dir).as_posix(),
    )

    # Pass 1: ingest & parse safely (fail-safe quarantine on error)
    ingested: list[dict] = []
    parsed_items: list[tuple[str, str, object]] = []
    for path in input_paths:
        rel = path.relative_to(input_dir).as_posix()
        raw = path.read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        try:
            if path.suffix.lower() not in SUPPORTED_EXTS:
                raise ValueError("unsupported_format")
            text = decode_bytes(raw)
            fmt = sniff_format(path, text)
            if fmt in ("csv", "tsv"):
                payload = load_tabular(text, "\t" if fmt == "tsv" else ",")
            elif fmt == "json":
                payload = load_json_safe(text)
            else:
                payload = text
            ingested.append({"rel": rel, "sha": sha, "fmt": fmt, "payload": payload, "error": None})
            parsed_items.append((rel, fmt, payload))
        except (ValueError, RecursionError, UnicodeError, csv.Error) as exc:
            reason = str(exc) if isinstance(exc, ValueError) and str(exc).isidentifier() else type(exc).__name__
            ingested.append({"rel": rel, "sha": sha, "fmt": None, "payload": None, "error": reason})

    # Derive hospital-specific MRN/UHID regexes, clinical lexicon, and run-level alias registry
    extra_mrn_patterns: list[re.Pattern] = derive_id_patterns_from_values(
        _collect_mrn_examples(parsed_items)
    )
    clinical_lexicon = build_clinical_lexicon(parsed_items)
    alias_registry = build_alias_registry(parsed_items)
    crf_predictor = CRFPredictor()

    # Link first, redact second: resolve patient clusters & issue >=80-bit HMAC pseudonyms
    evidences = extract_all_evidences(parsed_items)
    cluster_res = resolve_clusters(evidences)
    cluster_to_psn = issue_cluster_pseudonyms(cluster_res.clusters, _psn_key)
    record_to_psn: dict[tuple[str, str], str] = {
        rec_key: cluster_to_psn[cid]
        for rec_key, cid in cluster_res.record_to_cluster.items()
        if cid in cluster_to_psn
    }

    entity_stats: dict[str, int] = {}

    def scrub_free_text(txt: str) -> str:
        nv = build_normalized_view(txt)
        prot = clinical_lexicon.find_spans(txt, view=nv)
        id_spans = detect_identifier_spans(txt, extra_mrn_patterns=extra_mrn_patterns, view=nv)
        ctx_spans = detect_context_name_and_address_spans(txt, view=nv, protected_spans=prot)
        reg_spans = alias_registry.propose_spans(txt, view=nv)
        crf_spans = crf_predictor.predict_spans(txt) if crf_predictor.available else []
        merged = resolve_pii_vs_protected(id_spans + ctx_spans + reg_spans + crf_spans, prot)
        return redact_spans(txt, merged, stats=entity_stats)

    # Pass 2: sanitize & write outputs
    files_report: list[dict] = []
    records: list[dict] = []
    used_out_paths: set[str] = set()
    for item in ingested:
        rel = item["rel"]
        entry: dict = {"input_sha256": item["sha"]}
        if item["error"] is not None:
            entry.update(output=None, status="quarantined", reason=item["error"])
            files_report.append(entry)
            continue

        fmt = item["fmt"]
        payload = item["payload"]
        try:
            out_rel = sanitize_relative_path(
                rel,
                input_sha256=item["sha"],
                registry=alias_registry,
                extra_mrn_patterns=extra_mrn_patterns,
                used_paths=used_out_paths,
                stats=entity_stats,
            )
            if fmt in ("csv", "tsv"):
                delim = "\t" if fmt == "tsv" else ","
                result = _process_tabular(
                    payload,
                    delim,
                    rel,
                    out_rel,
                    records,
                    record_to_psn=record_to_psn,
                    text_scrubber=scrub_free_text,
                    stats=entity_stats,
                )
            elif fmt == "json":
                result = _process_json(
                    payload,
                    rel,
                    out_rel,
                    records,
                    record_to_psn=record_to_psn,
                    text_scrubber=scrub_free_text,
                    stats=entity_stats,
                )
            else:
                result = _process_text(
                    payload,
                    rel,
                    out_rel,
                    records,
                    record_to_psn=record_to_psn,
                    text_scrubber=scrub_free_text,
                )

            dest = san_dir / out_rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(result, encoding="utf-8", newline="")
            entry.update(output=f"sanitized/{out_rel}", status="sanitized")
        except (ValueError, RecursionError, UnicodeError, csv.Error) as exc:
            reason = str(exc) if isinstance(exc, ValueError) and str(exc).isidentifier() else type(exc).__name__
            entry.update(output=None, status="quarantined", reason=reason)
        files_report.append(entry)

    exported = {e["output"] for e in files_report if e.get("output")}
    records = [r for r in records if r["file"] in exported]

    with open(output_dir / "records.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")

    n_san = sum(e["status"] == "sanitized" for e in files_report)
    total_detected = sum(entity_stats.values())
    report = {
        "schema": "raksha-report/1",
        "files": files_report,
        "counts": {
            "files_total": len(files_report),
            "files_sanitized": n_san,
            "files_quarantined": len(files_report) - n_san,
            "records_processed": len(records),
            "identifiers_detected": total_detected,
            "records_linked": sum(r["patient"] is not None for r in records),
            "records_generalized": 0,
            "records_suppressed": 0,
            "records_quarantined": 0,
        },
        "entities": dict(sorted(entity_stats.items())),
        "timing": {"seconds": round(time.monotonic() - t0, 2)},
        "resources": {"peak_rss_mb": peak_rss_mb()},
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    # Step H: Independent post-export leakage scan (builds plain surface snapshot with no shared state)
    qa_surfaces: dict[str, str] = {}
    for surf, (lbl, _role) in alias_registry.full_phrases.items():
        if len(surf) >= 5:
            qa_surfaces[surf] = lbl
    for core, lbl in alias_registry.id_cores.items():
        if len(core) >= 6:
            qa_surfaces[core] = lbl
    qa_doctors = set(alias_registry.doctor_full_names)
    scan_and_remediate(output_dir, known_surfaces=qa_surfaces, doctor_surfaces=qa_doctors)

    # Re-read final report counts after QA remediation for honest stdout counts
    final_report = json.loads((output_dir / "report.json").read_text(encoding="utf-8"))
    fc = final_report["counts"]
    print(f"files {fc['files_total']} sanitized {fc['files_sanitized']} quarantined {fc['files_quarantined']} records {fc['records_processed']}")
    return 0

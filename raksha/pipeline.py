"""Core deterministic sanitization pipeline orchestration."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import sys
import time
from pathlib import Path

from raksha.config import derive_subkey, load_policy, require_secret
from raksha.detect.structured import process_table_rows, sanitize_json_data
from raksha.ingest.loader import decode_bytes, load_json_safe, load_tabular, sniff_format


def peak_rss_mb() -> float:
    try:
        import resource

        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return round(r / 1024 if sys.platform != "darwin" else r / 1048576, 1)
    except Exception:
        return 0.0


def _process_tabular(
    text: str,
    delimiter: str,
    rel: str,
    records: list[dict],
    text_scrubber=None,
    stats: dict[str, int] | None = None,
) -> str:
    rows = load_tabular(text, delimiter)
    clean_header, clean_body = process_table_rows(rows, text_scrubber=text_scrubber, stats=stats)
    buf = io.StringIO()
    writer = csv.writer(buf, delimiter=delimiter, lineterminator="\n")
    writer.writerow(clean_header)
    for i, row in enumerate(clean_body):
        writer.writerow(row)
        records.append({"file": f"sanitized/{rel}", "loc": f"row:{i}", "patient": None})
    return buf.getvalue()


def _process_json(
    text: str,
    rel: str,
    records: list[dict],
    text_scrubber=None,
    stats: dict[str, int] | None = None,
) -> str:
    data = load_json_safe(text)
    is_doc = (
        not isinstance(data, dict)
        or data.get("resourceType") == "Bundle"
        or not any(
            isinstance(v, list) and v and all(isinstance(x, dict) for x in v)
            for v in data.values()
        )
    )
    if is_doc:
        records.append({"file": f"sanitized/{rel}", "loc": "doc", "patient": None})
    else:
        for k, v in data.items():
            if isinstance(v, list) and v and all(isinstance(x, dict) for x in v):
                pointer_key = k.replace("~", "~0").replace("/", "~1")
                for i in range(len(v)):
                    records.append(
                        {"file": f"sanitized/{rel}", "loc": f"/{pointer_key}/{i}", "patient": None}
                    )

    cleaned = sanitize_json_data(data, text_scrubber=text_scrubber, stats=stats)
    return json.dumps(cleaned, ensure_ascii=False, indent=2) + "\n"


def _process_text(
    text: str,
    rel: str,
    records: list[dict],
    text_scrubber=None,
) -> str:
    records.append({"file": f"sanitized/{rel}", "loc": "doc", "patient": None})
    return text_scrubber(text) if text_scrubber is not None else text


def run_pipeline(input_dir: Path, output_dir: Path, text_scrubber=None) -> int:
    """Run deterministic de-identification pipeline from input_dir to output_dir."""
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

    files_report: list[dict] = []
    records: list[dict] = []
    entity_stats: dict[str, int] = {}

    input_paths = sorted((p for p in input_dir.rglob("*") if p.is_file()), key=lambda p: p.relative_to(input_dir).as_posix())

    for path in input_paths:
        rel = path.relative_to(input_dir).as_posix()
        raw = path.read_bytes()
        entry: dict = {"input_sha256": hashlib.sha256(raw).hexdigest()}
        try:
            text = decode_bytes(raw)
            fmt = sniff_format(path, text)
            if fmt in ("csv", "tsv"):
                delim = "\t" if fmt == "tsv" else ","
                result = _process_tabular(text, delim, rel, records, text_scrubber=text_scrubber, stats=entity_stats)
            elif fmt == "json":
                result = _process_json(text, rel, records, text_scrubber=text_scrubber, stats=entity_stats)
            else:
                result = _process_text(text, rel, records, text_scrubber=text_scrubber)

            dest = san_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(result, encoding="utf-8", newline="")
            entry.update(output=f"sanitized/{rel}", status="sanitized")
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
    print(f"files {len(files_report)} sanitized {n_san} quarantined {len(files_report) - n_san} records {len(records)}")
    return 0

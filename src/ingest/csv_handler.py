"""CSV/TSV parser, ragged/shifted row handler, header scrubber, and writer."""
from __future__ import annotations

import csv
import io
from typing import Callable

from src.ingest.profiler import infer_column_roles, infer_date_order
from src.models import FileEntry, PatientRecord


def detect_delimiter(text: str, default: str = ",") -> str:
    first = text.splitlines()[0] if text.splitlines() else ""
    counts = {d: first.count(d) for d in ("\t", ",", ";", "|")}
    best = max(counts.items(), key=lambda kv: kv[1])
    return best[0] if best[1] > 0 else default


def _realign_extra_cells(row: list[str], target_len: int, roles: dict[int, str]) -> list[str]:
    """Repair unquoted comma splits inside address/notes columns without dropping cells."""
    if len(row) <= target_len:
        return row + [""] * (target_len - len(row))
    # Prefer merging extra tokens into ADDRESS or NOTES column if known
    merge_idx = None
    for idx in range(target_len):
        if roles.get(idx) in ("ADDRESS", "NOTES"):
            merge_idx = idx
            break
    if merge_idx is None:
        # Keep exact row if no clean anchor, OR merge overflow into middle free-text slot
        return row
    extra = len(row) - target_len
    left = row[:merge_idx]
    merged = ", ".join(row[merge_idx : merge_idx + extra + 1])
    right = row[merge_idx + extra + 1 :]
    return left + [merged] + right


def parse_csv_entry(entry: FileEntry) -> None:
    text = entry.text_decoded or ""
    delim = "\t" if entry.file_type == "tsv" else detect_delimiter(text, ",")
    entry.delimiter = delim
    rows = list(csv.reader(io.StringIO(text, newline=""), delimiter=delim))
    if not rows:
        entry.status = "quarantined"
        entry.rel_output_path = None
        entry.reason = "empty_file"
        return

    header = rows[0]
    body = rows[1:]
    roles = infer_column_roles(header, body)
    # Repair any ragged rows that exceeded header width due to unquoted commas
    if any(len(r) > len(header) for r in body):
        body = [_realign_extra_cells(r, len(header), roles) for r in body]
    # Infer column-level date order per date/DOB column
    date_vals: list[str] = []
    hint_str = " ".join(h for idx, h in enumerate(header) if roles.get(idx) in ("DATE", "DOB"))
    for idx, role in roles.items():
        if role in ("DATE", "DOB"):
            date_vals.extend(r[idx] for r in body if idx < len(r))
    entry.date_order_hint = infer_date_order(date_vals, hint_str)
    entry.column_roles = roles
    entry.parsed_data = [header] + body

    rel_in = entry.rel_input_path
    rel_out = entry.rel_output_path or rel_in
    for idx, row in enumerate(body):
        loc = f"row:{idx}"
        entry.records.append(
            PatientRecord(
                record_uid=f"{rel_in}::{loc}",
                rel_input_path=rel_in,
                rel_output_path=rel_out,
                file_type=entry.file_type,
                loc=loc,
                raw_payload=row,
            )
        )


def serialize_csv_entry(
    entry: FileEntry,
    scrub_header_fn: Callable[[str], str],
    rewrite_row_fn: Callable[[int, list[str]], list[str]],
) -> str:
    rows = entry.parsed_data or []
    if not rows:
        return ""
    header = [scrub_header_fn(h) for h in rows[0]]
    buf = io.StringIO()
    writer = csv.writer(buf, delimiter=entry.delimiter or ",", lineterminator="\n")
    writer.writerow(header)
    for idx, row in enumerate(rows[1:]):
        writer.writerow(rewrite_row_fn(idx, row))
    return buf.getvalue()

"""Line-preserving plain-text and WhatsApp forward parser/writer."""
from __future__ import annotations

import re
from typing import Callable

from src.ingest.profiler import infer_date_order
from src.models import FileEntry, PatientRecord

_WA_HEADER_RE = re.compile(
    r"^\[?(\d{1,2}[./-]\d{1,2}[./-]\d{2,4}),\s*\d{1,2}:\d{2}(?:\s*[ap]m)?\]?\s*(?:-|\])?\s*([^:\n]{1,60}):",
    re.I,
)


def parse_text_entry(entry: FileEntry) -> None:
    text = entry.text_decoded or ""
    entry.date_order_hint = infer_date_order(text.splitlines())
    entry.parsed_data = text
    rel_in = entry.rel_input_path
    rel_out = entry.rel_output_path or rel_in
    entry.records.append(
        PatientRecord(
            record_uid=f"{rel_in}::doc",
            rel_input_path=rel_in,
            rel_output_path=rel_out,
            file_type="text",
            loc="doc",
            raw_payload=text,
        )
    )


def rewrite_text_preserving_lines(text: str, scrub_line_fn: Callable[[str], str]) -> str:
    """Process line-by-line so output keeps exact newline structure and line counts."""
    lines = text.splitlines(keepends=True)
    out: list[str] = []
    for ln in lines:
        ending = ""
        if ln.endswith("\r\n"):
            body, ending = ln[:-2], "\r\n"
        elif ln.endswith("\n"):
            body, ending = ln[:-1], "\n"
        else:
            body = ln
        out.append(scrub_line_fn(body) + ending)
    return "".join(out)

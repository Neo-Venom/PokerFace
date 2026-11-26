"""Safe XML parser & scrubber using defusedxml (blocks XXE/entity-expansion)."""
from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Callable

import defusedxml.ElementTree as DET

from src.models import FileEntry, PatientRecord

_PII_ATTRS = {"extension", "value", "name", "patient", "phone", "mobile", "id", "root"}


def parse_xml_entry(entry: FileEntry) -> None:
    text = entry.text_decoded or ""
    try:
        DET.fromstring(text.encode("utf-8"))
    except Exception as exc:
        entry.status = "quarantined"
        entry.rel_output_path = None
        entry.reason = type(exc).__name__
        return
    entry.parsed_data = text
    rel_in = entry.rel_input_path
    rel_out = entry.rel_output_path or rel_in
    entry.records.append(
        PatientRecord(
            record_uid=f"{rel_in}::doc",
            rel_input_path=rel_in,
            rel_output_path=rel_out,
            file_type="xml",
            loc="doc",
            raw_payload=text,
        )
    )


def sanitize_xml_text(text: str, scrub_fn: Callable[[str], str]) -> str:
    root = DET.fromstring(text.encode("utf-8"))
    stack = [root]
    while stack:
        elem = stack.pop()
        if elem.text and elem.text.strip():
            elem.text = scrub_fn(elem.text)
        if elem.tail and elem.tail.strip():
            elem.tail = scrub_fn(elem.tail)
        for ak, av in list(elem.attrib.items()):
            if ak.lower() != "xmlns" and av.strip():
                # Preserve OID roots like 2.16.356.aadhaar, scrub PII attributes
                if ak.lower() != "root":
                    elem.attrib[ak] = scrub_fn(av)
        for child in elem:
            stack.append(child)
    return ET.tostring(root, encoding="unicode")

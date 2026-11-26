"""Safe file decoding, format validation, and quarantine-on-error ingestion."""
from __future__ import annotations

import csv
import io
import json
import sys
from pathlib import Path

SUPPORTED_EXTS = {".csv", ".tsv", ".json", ".txt"}

# Bytes valid in legacy Windows-1252 medical text exports (smart quotes, dashes, degree, micro, nbsp, etc.).
# Random high bytes like 0xff/0xfe/0xc3 indicate binary corruption or mojibake, not legitimate cp1252.
_CP1252_SAFE_EXTRA = {
    0x85, 0x91, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97,
    0xA0, 0xA7, 0xA9, 0xAE, 0xB0, 0xB1, 0xB5, 0xB7,
    0xBC, 0xBD, 0xBE, 0xD7, 0xF7,
}


def decode_bytes(raw: bytes) -> str:
    """Decode raw bytes to UTF-8 text or raise ValueError('invalid_encoding')."""
    if not raw.strip():
        raise ValueError("empty_file")

    if raw.startswith(b"\xef\xbb\xbf"):
        try:
            text = raw[3:].decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise ValueError("invalid_encoding") from exc
        return text

    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        try:
            text = raw.decode("utf-16", errors="strict")
        except UnicodeDecodeError as exc:
            raise ValueError("invalid_encoding") from exc
        if "\x00" in text:
            raise ValueError("invalid_encoding")
        return text

    # Reject NUL bytes in 8-bit text
    if b"\x00" in raw:
        raise ValueError("invalid_encoding")

    try:
        return raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        pass

    # Strict cp1252 check: only allow well-formed cp1252 punctuation/clinical symbols
    for b in raw:
        if b < 0x80:
            if b < 0x20 and b not in (0x09, 0x0A, 0x0D):
                raise ValueError("invalid_encoding")
        elif b not in _CP1252_SAFE_EXTRA:
            raise ValueError("invalid_encoding")

    try:
        return raw.decode("cp1252", errors="strict")
    except UnicodeDecodeError as exc:
        raise ValueError("invalid_encoding") from exc


def load_tabular(text: str, delimiter: str) -> list[list[str]]:
    rows = list(csv.reader(io.StringIO(text, newline=""), delimiter=delimiter))
    if not rows:
        raise ValueError("empty_file")
    return rows


def load_json_safe(text: str):
    """Parse JSON safely; supports legitimate deep structures without crashing."""
    old_limit = sys.getrecursionlimit()
    try:
        if old_limit < 5000:
            sys.setrecursionlimit(5000)
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("malformed_json") from exc
    except RecursionError as exc:
        raise ValueError("excessive_nesting") from exc


def sniff_format(path: Path, text: str) -> str:
    """Return canonical format ('csv', 'tsv', 'json', 'txt') or raise ValueError."""
    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTS:
        raise ValueError("unsupported_format")
    if ext in (".csv", ".tsv"):
        stripped = text.lstrip()
        # Catch JSON payloads hidden behind a .csv/.tsv extension
        if stripped.startswith(("{", "[")):
            try:
                load_json_safe(text)
                return "json"
            except ValueError:
                pass
        return "tsv" if ext == ".tsv" else "csv"
    if ext == ".json":
        return "json"
    return "txt"

"""Sorted input inventory, raw SHA-256 hashing, byte sniffing, strict decoding, and safe filenames."""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Callable

from src.models import FileEntry
from src.security.keys import KeyRing

_TEXT_EXTS = {".csv", ".tsv", ".json", ".jsonl", ".ndjson", ".txt", ".md", ".log", ".xml"}
_MAGIC_QUARANTINE = (
    (b"%PDF", "unsupported_pdf"),
    (b"\xff\xd8\xff", "unsupported_image"),
    (b"\x89PNG\r\n\x1a\n", "unsupported_image"),
    (b"GIF87a", "unsupported_image"),
    (b"GIF89a", "unsupported_image"),
    (b"PK\x03\x04", "unsupported_archive"),
    (b"SQLite format 3", "unsupported_binary"),
)


def decode_strict(raw: bytes) -> tuple[str | None, str | None]:
    """Strictly decode raw bytes without lossy replacement. Returns (text, error_reason)."""
    if not raw.strip():
        return None, "empty_file"

    for magic, reason in _MAGIC_QUARANTINE:
        if raw.startswith(magic):
            return None, reason

    if raw.startswith(b"\xef\xbb\xbf"):
        try:
            text = raw[3:].decode("utf-8", errors="strict")
            if "\x00" in text:
                return None, "corrupt_binary_bytes"
            return text, None
        except UnicodeDecodeError:
            return None, "undecodable_text"

    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        try:
            text = raw.decode("utf-16", errors="strict")
            if "\x00" in text:
                return None, "corrupt_binary_bytes"
            return text, None
        except UnicodeDecodeError:
            return None, "undecodable_text"

    # Unbommed UTF-16 heuristic
    nul_count = raw.count(b"\x00")
    if nul_count > 0:
        if len(raw) % 2 == 0 and nul_count >= max(2, len(raw) // 8):
            even_nuls = raw[0::2].count(0)
            odd_nuls = raw[1::2].count(0)
            enc = "utf-16-le" if odd_nuls > even_nuls else "utf-16-be"
            try:
                text = raw.decode(enc, errors="strict")
                if "\x00" not in text:
                    return text, None
            except UnicodeDecodeError:
                pass
        return None, "corrupt_binary_bytes"

    # Reject UTF-16 BOM markers appearing mid-stream inside corrupt text
    if b"\xff\xfe" in raw or b"\xfe\xff" in raw:
        return None, "corrupt_binary_bytes"

    try:
        return raw.decode("utf-8", errors="strict"), None
    except UnicodeDecodeError:
        pass

    try:
        text = raw.decode("cp1252", errors="strict")
        # Guard against high density of non-ASCII control/mojibake bytes
        ctrl = sum(1 for ch in text if ord(ch) < 32 and ch not in "\r\n\t")
        if ctrl > 0:
            return None, "corrupt_binary_bytes"
        return text, None
    except UnicodeDecodeError:
        return None, "undecodable_text"


def sniff_format(ext: str, text: str) -> str:
    """Determine true logical format from content first, falling back to extension."""
    stripped = text.lstrip()
    if stripped.startswith(("{", "[")):
        # .json / .jsonl / .ndjson are always JSON (invalid JSON gets quarantined cleanly).
        if ext in (".json", ".jsonl", ".ndjson"):
            return "json"
        # Non-JSON extension (e.g. mislabeled .csv vs WhatsApp '[14/02/26...]' .txt):
        # classify as JSON only if it actually parses as JSON.
        try:
            import json
            json.loads(stripped)
            return "json"
        except Exception:
            pass
    if stripped.startswith("<?xml") or stripped.startswith("<ClinicalDocument"):
        return "xml"
    if ext == ".tsv":
        return "tsv"
    if ext == ".csv":
        first_line = stripped.splitlines()[0] if stripped.splitlines() else ""
        if first_line.count("\t") > first_line.count(","):
            return "tsv"
        return "csv"
    if ext in (".txt", ".md", ".log"):
        return "text"
    if ext == ".json":
        return "json"
    if ext == ".xml":
        return "xml"
    return "unsupported"


def scan_input_tree(inp: Path) -> list[tuple[Path, str, FileEntry]]:
    """Walk input deterministically in sorted order; hash raw bytes."""
    paths = sorted((p for p in inp.rglob("*")), key=lambda p: p.relative_to(inp).as_posix())
    items: list[tuple[Path, str, FileEntry]] = []
    for path in paths:
        if path.is_dir():
            continue
        rel = path.relative_to(inp).as_posix()
        if path.is_symlink() or not path.is_file():
            entry = FileEntry(
                input_sha256="0" * 64,
                rel_input_path=rel,
                rel_output_path=None,
                status="quarantined",
                reason="unsupported_special_file",
            )
            items.append((path, rel, entry))
            continue

        raw = path.read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        ext = path.suffix.lower()
        if ext and ext not in _TEXT_EXTS:
            # Verify magic/unsupported binary before anything else
            _, err = decode_strict(raw)
            reason = err or "unsupported_format"
            entry = FileEntry(
                input_sha256=sha,
                rel_input_path=rel,
                rel_output_path=None,
                status="quarantined",
                reason=reason,
            )
            items.append((path, rel, entry))
            continue

        text, err = decode_strict(raw)
        if text is None:
            entry = FileEntry(
                input_sha256=sha,
                rel_input_path=rel,
                rel_output_path=None,
                status="quarantined",
                reason=err or "undecodable_text",
            )
            items.append((path, rel, entry))
            continue

        ftype = sniff_format(ext, text)
        if ftype == "unsupported":
            entry = FileEntry(
                input_sha256=sha,
                rel_input_path=rel,
                rel_output_path=None,
                status="quarantined",
                reason="unsupported_format",
            )
            items.append((path, rel, entry))
            continue

        entry = FileEntry(
            input_sha256=sha,
            rel_input_path=rel,
            rel_output_path=rel,
            status="sanitized",
            file_type=ftype,
            text_decoded=text,
            is_clean_doc=(rel.startswith("clean/") or "/clean/" in rel),
        )
        items.append((path, rel, entry))
    return items


_SAFE_FNAME_WORDS = {
    "discharge", "summary", "ds", "report", "nursing", "note", "notes", "whatsapp",
    "forward", "opd", "ipd", "claims", "registration", "legacy", "export", "patients",
    "bundle", "ward", "visitor", "log", "consent", "form", "ambulance", "helpdesk",
    "tickets", "patient", "index", "vitals", "sheet", "bulk", "lab", "results",
    "dengue", "management", "protocol", "drug", "formulary", "reference", "ranges",
    "census", "rx", "readme", "hl7", "cda", "mislabeled", "city", "lifeline",
    "pathkind", "sanjeevani", "jeevanjyoti", "arogya", "dhhardoi", "dhh", "doc",
    "guidelines", "template", "summary", "record", "records", "hospital",
}
_FNAME_ID_RE = re.compile(
    r"(?:(?<!\d)[6-9]\d{9}(?!\d)"
    r"|(?<!\d)\d{12,14}(?!\d)"
    r"|\b[A-Z]{5}\d{4}[A-Z]\b"
    r"|(?i:(?:uhid|ipd|mrn|adm|ip)[-_]?(?:\d{2,4}[-_])?\d{4,})"
    r"|\b[A-Z]{2,5}-\d{2,4}-\d{4,}\b"
    r"|\b[A-Z]{2,5}-\d{6,}\b"
    r"|\b[A-Z]{2,5}\d{4}-\d{5,}\b)"
)


def maybe_rename_rel_path(
    rel: str,
    input_sha256: str,
    keyring: KeyRing,
    has_pii_fn: Callable[[str], bool],
) -> tuple[str, bool]:
    """Deterministically sanitize PII-bearing filenames while keeping folder & extension."""
    p = Path(rel)
    stem = p.stem
    ext = p.suffix
    parent = p.parent.as_posix()

    has_id = bool(_FNAME_ID_RE.search(stem))
    has_text_pii = has_pii_fn(re.sub(r"[_]+", " ", stem))
    if has_id or has_text_pii:
        prefix = "doc"
        low = stem.lower()
        for cand in ("discharge", "report", "nursing", "rx", "opd", "lab"):
            if cand in low:
                prefix = cand
                break
        safe_name = f"{prefix}_{keyring.file_tag(input_sha256, 8)}{ext}"
        new_rel = f"{parent}/{safe_name}" if parent and parent != "." else safe_name
        return new_rel, True
    return rel, False

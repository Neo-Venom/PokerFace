"""Iterative JSON parser, RFC 6901 pointer extractor, key scrubber, and type-preserving writer."""
from __future__ import annotations

import json
import re
from typing import Any, Callable

from src.config import Policy
from src.ingest.hidden import fix_surrogates_tree, scrub_hidden, scrub_payload_value
from src.ingest.profiler import infer_date_order
from src.models import FileEntry, PatientRecord

# Strings JSON keeps out of the bracket count, and the brackets themselves (pre-parse depth guard).
_JSON_STR_RE = re.compile(r'"[^"\\]*(?:\\.[^"\\]*)*"', re.S)
_JSON_BRACKET_RE = re.compile(r"[\[\]{}]")
_LONE_SURROGATE_ESC_RE = re.compile(r"\\u[dD][89a-fA-F][0-9a-fA-F]{2}")
# Key words whose string value is file content: a binary base64 blob there is replaced, never shipped raw.
_ATTACH_KEY_WORDS = frozenset({"data", "photo", "image", "img", "scan", "blob", "attachment", "attachments", "attach",
                               "signature", "pdf", "base64", "b64", "content", "thumbnail", "binary", "jpeg", "png"})
JSON_LINES_EXTS = (".jsonl", ".ndjson")


def escape_pointer_token(token: str) -> str:
    return token.replace("~", "~0").replace("/", "~1")


def json_text_depth_exceeds(text: str, limit: int) -> bool:
    """Streaming nesting check on the raw text, before json.loads: a 1,500-deep bomb is rejected the same way on
    every Python version (no recursion-limit games, no deep tree for later stages to walk)."""
    depth = 0
    for m in _JSON_BRACKET_RE.finditer(_JSON_STR_RE.sub("", text)):
        if m.group(0) in "[{":
            depth += 1
            if depth > limit:
                return True
        else:
            depth -= 1
    return False


def deep_copy_json(data: Any) -> Any:
    """Iterative deep copy of a parsed JSON tree (copy.deepcopy recurses ~2 frames per level)."""
    if not isinstance(data, (dict, list)):
        return data
    root: Any = {} if isinstance(data, dict) else []
    stack: list[tuple[Any, Any]] = [(data, root)]
    while stack:
        src, dst = stack.pop()
        items = src.items() if isinstance(src, dict) else enumerate(src)
        for k, v in items:
            if isinstance(v, (dict, list)):
                child: Any = {} if isinstance(v, dict) else []
                stack.append((v, child))
            else:
                child = v
            if isinstance(dst, dict):
                dst[k] = child
            else:
                dst.append(child)
    return root


def is_json_lines(entry: FileEntry) -> bool:
    """JSON Lines entries carry "\\n" as their record delimiter (FileEntry.delimiter is otherwise CSV-only)."""
    return entry.delimiter == "\n"


def _parse_json_lines(text: str) -> list[Any] | None:
    """One JSON value per non-blank line, or None if any line is not JSON."""
    out: list[Any] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            out.append(json.loads(line, parse_constant=_reject_constant))
        except ValueError:
            return None
    return out if out else None


def serialize_json_entry(entry: FileEntry, data: Any) -> str:
    """JSON stays pretty-printed JSON; JSON Lines stays one compact value per line with the input's blank lines
    and line endings."""
    if not is_json_lines(entry) or not isinstance(data, list):
        return json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    text = entry.text_decoded or ""
    nl = "\r\n" if "\r\n" in text else "\n"
    out: list[str] = []
    it = iter(data)
    for line in text.splitlines():
        if line.strip():
            out.append(json.dumps(next(it, None), ensure_ascii=False))
        else:
            out.append(line)
    out.extend(json.dumps(v, ensure_ascii=False) for v in it)
    return nl.join(out) + nl


def json_depth_exceeds(obj: Any, limit: int) -> bool:
    stack: list[tuple[Any, int]] = [(obj, 1)]
    while stack:
        node, depth = stack.pop()
        if depth > limit:
            return True
        if isinstance(node, dict):
            for v in node.values():
                if isinstance(v, (dict, list)):
                    stack.append((v, depth + 1))
        elif isinstance(node, list):
            for v in node:
                if isinstance(v, (dict, list)):
                    stack.append((v, depth + 1))
    return False


def extract_json_record_locs(data: Any) -> list[tuple[str, dict[str, Any]]]:
    """Iteratively extract deepest patient-bearing dict-array objects."""
    found: list[tuple[str, dict[str, Any]]] = []
    if isinstance(data, list):
        if data and all(isinstance(x, dict) for x in data):
            for idx, item in enumerate(data):
                found.append((f"/{idx}", item))
        return found
    if not isinstance(data, dict):
        return found

    # Single-patient structured docs (e.g. lab/report_*.json with root "patient" + "panels")
    # must stay whole ("doc") unless an explicit outer multi-patient container holds patient records.
    _MULTI_CONTAINER_KEYS = {"visits", "patients", "tickets", "records", "rows", "entries", "encounters", "admissions", "list"}
    has_outer_multi = any(
        isinstance(data.get(k), list) and data[k] and all(isinstance(x, dict) for x in data[k])
        for k in _MULTI_CONTAINER_KEYS
    )
    if not has_outer_multi and (
        isinstance(data.get("patient"), dict)
        or any(k in data for k in ("patient_ref", "aadhaar", "aadhar_no", "uhid", "hospital_uhid", "abha", "dob", "birthDate"))
    ):
        return found

    stack: list[tuple[str, Any]] = [("", data)]
    while stack:
        prefix, node = stack.pop()
        if isinstance(node, dict):
            for k in reversed(list(node.keys())):
                v = node[k]
                ptr = f"{prefix}/{escape_pointer_token(str(k))}"
                if isinstance(v, list) and v and all(isinstance(x, dict) for x in v):
                    # Check if nested further with dict-arrays inside each item
                    has_nested_arrays = any(
                        isinstance(iv, list) and iv and all(isinstance(x, dict) for x in iv)
                        for item in v
                        for iv in item.values()
                    )
                    if not has_nested_arrays:
                        for idx, item in enumerate(v):
                            found.append((f"{ptr}/{idx}", item))
                    else:
                        for idx, item in enumerate(v):
                            found.append((f"{ptr}/{idx}", item))
                elif isinstance(v, dict):
                    stack.append((ptr, v))
    return found


def _collect_date_strings(obj: Any) -> list[str]:
    vals: list[str] = []
    stack = [obj]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            for k, v in node.items():
                if isinstance(v, str) and ("date" in str(k).lower() or "dob" in str(k).lower()):
                    vals.append(v)
                elif isinstance(v, (dict, list)):
                    stack.append(v)
        elif isinstance(node, list):
            for v in node:
                if isinstance(v, (dict, list)):
                    stack.append(v)
    return vals


def _reject_constant(name: str):
    """NaN / Infinity are not JSON (RFC 8259): such a file cannot be re-emitted as valid JSON."""
    raise ValueError(f"invalid JSON constant {name}")


def _quarantine(entry: FileEntry, reason: str) -> None:
    entry.status = "quarantined"
    entry.rel_output_path = None
    entry.reason = reason


def parse_json_entry(entry: FileEntry, policy: Policy) -> None:
    """Strict JSON / JSON Lines parse. Quarantined (never re-emitted): truncated or invalid JSON, NaN/Infinity
    (not JSON, would ship an unparseable file), nesting deeper than policy.max_json_depth (depth bombs).
    Duplicate keys follow the JSON parser (last value wins): an earlier duplicate never reaches the output, so it
    cannot leak, and the file's records are kept instead of losing the whole file to quarantine."""
    text = entry.text_decoded or ""
    if json_text_depth_exceeds(text, policy.max_json_depth):
        _quarantine(entry, "too_deep")
        return
    jsonl = entry.rel_input_path.lower().endswith(JSON_LINES_EXTS)
    data: Any = None
    try:
        if jsonl:
            data = _parse_json_lines(text)
            if data is None:
                raise ValueError("invalid JSON line")
        else:
            data = json.loads(text, parse_constant=_reject_constant)
    except RecursionError:
        _quarantine(entry, "too_deep")
        return
    except ValueError as exc:
        # A .json file holding JSON Lines ("Extra data" after the first value) is still JSON Lines.
        data = None if jsonl or "Extra data" not in str(exc) else _parse_json_lines(text)
        if data is None or len(data) < 2:
            _quarantine(entry, type(exc).__name__)
            return
        jsonl = True
    if jsonl:
        entry.delimiter = "\n"
    if _LONE_SURROGATE_ESC_RE.search(text):
        data = fix_surrogates_tree(data)

    if json_depth_exceeds(data, policy.max_json_depth):
        _quarantine(entry, "too_deep")
        return

    entry.parsed_data = data
    entry.date_order_hint = infer_date_order(_collect_date_strings(data))
    rel_in = entry.rel_input_path
    rel_out = entry.rel_output_path or rel_in

    if isinstance(data, dict) and data.get("resourceType") == "Bundle":
        entry.file_type = "fhir"
        entry.records.append(
            PatientRecord(
                record_uid=f"{rel_in}::doc",
                rel_input_path=rel_in,
                rel_output_path=rel_out,
                file_type="fhir",
                loc="doc",
                raw_payload=data,
            )
        )
        return

    locs = extract_json_record_locs(data)
    if not locs:
        entry.records.append(
            PatientRecord(
                record_uid=f"{rel_in}::doc",
                rel_input_path=rel_in,
                rel_output_path=rel_out,
                file_type="json",
                loc="doc",
                raw_payload=data,
            )
        )
    else:
        for ptr, node in locs:
            entry.records.append(
                PatientRecord(
                    record_uid=f"{rel_in}::{ptr}",
                    rel_input_path=rel_in,
                    rel_output_path=rel_out,
                    file_type="json",
                    loc=ptr,
                    raw_payload=node,
                )
            )


def is_attachment_key(key: str) -> bool:
    """Key words that say "this string is file content" (photo, scan, pdf, base64 ...)."""
    parts = re.findall(r"[a-z0-9]+", re.sub(r"([a-z])([A-Z])", r" ", key).lower())
    return any(p in _ATTACH_KEY_WORDS for p in parts)


def sanitize_base64_value(val: str, scrub_text_fn: Callable[[str], str], attachment: bool = False) -> str:
    """Scrub one string value, hidden payloads included (see src.ingest.hidden.scrub_hidden)."""
    return scrub_hidden(val, scrub_text_fn, attachment=attachment)


def transform_json_iterative(
    data: Any,
    scrub_key_fn: Callable[[str], str],
    scrub_val_fn: Callable[[str, Any, str], Any],
) -> Any:
    """Iteratively rewrite keys and values while preserving structure, order, and types."""

    def _transform_leaf(path: str, key_name: str, val: Any) -> Any:
        if isinstance(val, str) and len(val) >= 32 and is_attachment_key(key_name):
            # File content under an attachment key: a binary blob becomes a placeholder, text is scrubbed inside.
            out = scrub_payload_value(val, lambda s: sanitize_base64_value(s, lambda t: scrub_val_fn(path, t, key_name)), attachment=True)
            if out is not None:
                return out
        return scrub_val_fn(path, val, key_name)

    if isinstance(data, dict):
        root: Any = {}
    elif isinstance(data, list):
        root = []
    else:
        return _transform_leaf("", "", data)

    # Stack items: (src_node, dst_container, path_prefix)
    stack: list[tuple[Any, Any, str]] = [(data, root, "")]
    while stack:
        src, dst, prefix = stack.pop()
        if isinstance(src, dict):
            seen_keys: set[str] = set()
            for raw_k, v in src.items():
                k_str = str(raw_k)
                new_k = scrub_key_fn(k_str)
                # Ensure no two scrubbed keys collide
                if new_k in seen_keys:
                    idx = 2
                    while f"{new_k}_{idx}" in seen_keys:
                        idx += 1
                    new_k = f"{new_k}_{idx}"
                seen_keys.add(new_k)
                child_path = f"{prefix}/{escape_pointer_token(k_str)}"
                if isinstance(v, dict):
                    child: Any = {}
                    dst[new_k] = child
                    stack.append((v, child, child_path))
                elif isinstance(v, list):
                    child = []
                    dst[new_k] = child
                    stack.append((v, child, child_path))
                else:
                    dst[new_k] = _transform_leaf(child_path, k_str, v)
        elif isinstance(src, list):
            for idx, v in enumerate(src):
                child_path = f"{prefix}/{idx}"
                if isinstance(v, dict):
                    child = {}
                    dst.append(child)
                    stack.append((v, child, child_path))
                elif isinstance(v, list):
                    child = []
                    dst.append(child)
                    stack.append((v, child, child_path))
                else:
                    dst.append(_transform_leaf(child_path, "", v))
    return root

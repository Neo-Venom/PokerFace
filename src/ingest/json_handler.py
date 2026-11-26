"""Iterative JSON parser, RFC 6901 pointer extractor, key scrubber, and type-preserving writer."""
from __future__ import annotations

import base64
import json
import re
import sys
from typing import Any, Callable

from src.config import Policy
from src.ingest.profiler import infer_date_order
from src.models import FileEntry, PatientRecord

_B64_RE = re.compile(r"^[A-Za-z0-9+/]{24,}={0,2}$")


def escape_pointer_token(token: str) -> str:
    return token.replace("~", "~0").replace("/", "~1")


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


def parse_json_entry(entry: FileEntry, policy: Policy) -> None:
    text = entry.text_decoded or ""
    prev = sys.getrecursionlimit()
    try:
        sys.setrecursionlimit(max(prev, 5000))
        data = json.loads(text)
    except RecursionError:
        entry.status = "quarantined"
        entry.rel_output_path = None
        entry.reason = "too_deep"
        return
    except ValueError as exc:
        entry.status = "quarantined"
        entry.rel_output_path = None
        entry.reason = type(exc).__name__
        return
    finally:
        sys.setrecursionlimit(prev)

    if json_depth_exceeds(data, policy.max_json_depth):
        entry.status = "quarantined"
        entry.rel_output_path = None
        entry.reason = "too_deep"
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


def sanitize_base64_value(val: str, scrub_text_fn: Callable[[str], str]) -> str:
    if not _B64_RE.match(val.strip()):
        return scrub_text_fn(val)
    tok = val.strip()
    try:
        raw = base64.b64decode(tok + "=" * (-len(tok) % 4))
        decoded = raw.decode("utf-8", errors="strict")
        cleaned = scrub_text_fn(decoded)
        return base64.b64encode(cleaned.encode("utf-8")).decode("ascii")
    except Exception:
        # Non-text binary base64: replace with placeholder base64
        return base64.b64encode(b"[ATTACHMENT_REDACTED]").decode("ascii")


def transform_json_iterative(
    data: Any,
    scrub_key_fn: Callable[[str], str],
    scrub_val_fn: Callable[[str, Any, str], Any],
) -> Any:
    """Iteratively rewrite keys and values while preserving structure, order, and types."""

    def _transform_leaf(path: str, key_name: str, val: Any) -> Any:
        if isinstance(val, str):
            kl = key_name.lower()
            if kl == "data" or "base64" in kl or _B64_RE.match(val.strip()):
                return sanitize_base64_value(val, lambda s: scrub_val_fn(path, s, key_name))
            return scrub_val_fn(path, val, key_name)
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

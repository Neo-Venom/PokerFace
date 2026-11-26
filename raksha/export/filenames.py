"""Sanitize filenames, directory names, and base64-encoded text attachments."""
from __future__ import annotations

import base64
import binascii
import hashlib
import re
from pathlib import Path
from typing import TYPE_CHECKING

from raksha.detect.identifiers import Span, detect_identifier_spans, resolve_overlaps

if TYPE_CHECKING:
    from raksha.detect.registry import AliasRegistry

_B64_CAND_RE = re.compile(r"^[A-Za-z0-9+/]{24,}={0,2}$")
_DOC_STEM_NON_NAMES = {
    "report", "forward", "whatsapp", "nursing", "discharge", "summary", "protocol",
    "census", "claims", "results", "patients", "bundle", "export", "visits", "admissions",
    "hinglish", "hindi", "english", "note", "notes", "slip", "cbc", "lft", "rft", "kft",
    "opd", "ipd", "rx", "dx", "lab", "scan", "scanned", "copy", "audit", "feed", "bulk",
    "shifted", "legacy", "unicode", "cp1252", "regional", "diagnostics", "keyed", "patient",
}
_FILENAME_ID_RE = re.compile(
    r"(?i)(?<![A-Za-z0-9])"
    r"(?!(?:report|forward|whatsapp|nursing|discharge|summary|protocol|census|claims|results|patients|bundle|export|visits)[-_])"
    r"("
    r"[A-Za-z]{2,6}[-_][A-Za-z0-9]{1,5}[-_]\d{3,}"
    r"|[A-Za-z]{2,6}[-_][A-Za-z0-9]{2,8}[-_]\d{2,4}[-_]\d{3,}"
    r"|[A-Za-z]{2,6}[-_]\d{2,4}[-_]\d{3,}"
    r"|[A-Za-z]{2,6}\d{2,4}[-_]\d{3,}"
    r"|[A-Za-z]{2,6}[-_]\d{4,}"
    r"|[A-Za-z]{2,4}\d{5,}"
    r"|(?:[6-9]\d{9})"
    r")(?![A-Za-z0-9])"
)


def sanitize_relative_path(
    rel: str,
    input_sha256: str,
    registry: AliasRegistry,
    extra_mrn_patterns: list[re.Pattern] | None = None,
    used_paths: set[str] | None = None,
    stats: dict[str, int] | None = None,
) -> str:
    """Scrub PII from directory names and filename stem (preserving extension and uniqueness)."""
    p = Path(rel)
    parts = list(p.parts)
    changed = False
    new_parts: list[str] = []

    for idx, part in enumerate(parts):
        is_last = idx == len(parts) - 1
        stem = Path(part).stem if is_last else part
        ext = Path(part).suffix if is_last else ""

        # Collect spans from identifier regexes (Unicode NFC/NFD safe) + registry
        space_stem = stem.replace("_", " ")
        spans: list[Span] = []
        spans.extend(detect_identifier_spans(space_stem, extra_mrn_patterns=extra_mrn_patterns))
        for m in _FILENAME_ID_RE.finditer(stem):
            lbl = "PHONE" if re.fullmatch(r"[6-9]\d{9}", m.group(1)) else "MRN"
            spans.append(Span(m.start(1), m.end(1), lbl, m.group(1), source="identifier"))
        spans.extend(registry.propose_for_filename(stem))
        # When an ID/Phone anchor sits inside the filename (e.g. 7990055667_Celandine_Frostbourne_cbc.json),
        # also catch adjacent multi-word Title-Case person names not matched cleanly by doc/clinical words.
        if any(sp.label in ("PHONE", "MRN", "AADHAAR", "ABHA_NUMBER") for sp in spans):
            from raksha.detect.structured import _looks_like_person_name

            tokens_fn = [(m.group(0), m.start(), m.end()) for m in re.finditer(r"[^\s_\-/\\]+", stem)]
            for i in range(len(tokens_fn)):
                for win in (3, 2):
                    if i + win > len(tokens_fn):
                        continue
                    sub = tokens_fn[i : i + win]
                    words_sub = [t[0] for t in sub]
                    if any(w.casefold() in _DOC_STEM_NON_NAMES or any(c.isdigit() for c in w) for w in words_sub):
                        continue
                    cand_s = " ".join(words_sub)
                    if all(w and w[0].isupper() for w in words_sub) and _looks_like_person_name(cand_s):
                        spans.append(Span(sub[0][1], sub[-1][2], "PATIENT_NAME", stem[sub[0][1] : sub[-1][2]], source="context"))
        merged = resolve_overlaps(spans)

        if merged:
            changed = True
            buf: list[str] = []
            cur = 0
            for sp in merged:
                if sp.start < cur:
                    continue
                buf.append(stem[cur : sp.start])
                buf.append(sp.label)
                if stats is not None:
                    stats[sp.label] = stats.get(sp.label, 0) + 1
                cur = sp.end
            buf.append(stem[cur:])
            clean_stem = re.sub(r"_{2,}", "_", "".join(buf)).strip("_") or "redacted"
            if is_last:
                # Append deterministic short digest to prevent collisions across renamed files
                clean_stem = f"{clean_stem}_{input_sha256[:8]}"
            new_parts.append(f"{clean_stem}{ext}")
        else:
            new_parts.append(part)

    out_rel = "/".join(new_parts)
    if used_paths is not None:
        while out_rel in used_paths:
            # Deterministic collision resolution
            h = hashlib.sha256(out_rel.encode("utf-8")).hexdigest()[:6]
            op = Path(out_rel)
            out_rel = op.with_name(f"{op.stem}_{h}{op.suffix}").as_posix()
        used_paths.add(out_rel)
    return out_rel


def sanitize_base64_string(val: str, text_scrubber) -> str:
    """If `val` is base64-encoded UTF-8 text, decode, scrub PII, and re-encode; leave binary/non-b64 unchanged or scrub."""
    s = val.strip()
    if len(s) < 24 or not _B64_CAND_RE.match(s):
        return val
    try:
        padded = s + "=" * (-len(s) % 4)
        raw = base64.b64decode(padded, validate=True)
    except (binascii.Error, ValueError):
        return val

    # Binary attachments (PDF, JPEG, PNG, ZIP) -> do not corrupt or pass raw PII text; check magic
    if raw.startswith((b"%PDF", b"\xff\xd8\xff", b"\x89PNG", b"PK\x03\x04")):
        return val

    try:
        decoded = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        return val

    # Ensure decoded text is mostly printable text (not random binary bytes)
    if not decoded or sum(1 for ch in decoded if ch.isprintable() or ch in "\r\n\t") / len(decoded) < 0.85:
        return val

    scrubbed = text_scrubber(decoded) if text_scrubber is not None else decoded
    return base64.b64encode(scrubbed.encode("utf-8")).decode("ascii")

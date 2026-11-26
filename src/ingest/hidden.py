"""Hidden channels: base64 / gzip+base64 / hex / %XX / \\uXXXX / HTML-entity payloads inside values, keys and names.

A token counts as an encoded payload only when it decodes to printable UTF-8 text. Random IDs, hashes and API
tokens that merely use the base64 or hex alphabet are left exactly as they are (they used to be overwritten with
an "[ATTACHMENT_REDACTED]" blob). A payload is decoded, scrubbed with the caller's scrubber, and re-encoded in the
same codec (gzip with mtime=0, so output stays byte-deterministic); a payload the scrubber leaves unchanged keeps
its original bytes.
"""
from __future__ import annotations

import base64
import binascii
import gzip
import html
import re
import zlib
from typing import Any, Callable

ATTACHMENT_PLACEHOLDER = base64.b64encode(b"[ATTACHMENT_REDACTED]").decode("ascii")

_B64_RUN_RE = re.compile(r"[A-Za-z0-9+/_-]{16,}={0,2}")
_B64_STRICT_RE = re.compile(r"^(?:[A-Za-z0-9+/]+|[A-Za-z0-9_-]+)={0,2}$")
_HEX_RUN_RE = re.compile(r"(?<![0-9A-Fa-f])(?:[0-9A-Fa-f]{2}){8,}(?![0-9A-Fa-f])")
_UESC_RE = re.compile(r"(?:\\[uU][0-9A-Fa-f]{4})+")
_MAX_DECODED = 8 * 1024 * 1024
_MAX_DEPTH = 4


def _printable_text(raw: bytes) -> str | None:
    """UTF-8 text that a human would read (>=90% printable, has letters or digits) or None."""
    try:
        s = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        return None
    if not s.strip():
        return None
    good = sum(1 for ch in s if ch.isprintable() or ch in "\r\n\t")
    if good < 0.9 * len(s) or not any(ch.isalnum() for ch in s):
        return None
    return s


def _wordy_or_id(s: str) -> bool:
    """A digits-only hex run ("4532756279624064") decodes to printable junk by chance: accept it only when the
    text reads as words or is itself an ID number."""
    core = s.replace(" ", "").replace("-", "")
    if core.isdigit() and len(core) >= 6:
        return True
    return sum(1 for ch in s if ch.isalpha() or ch.isspace()) >= 0.7 * len(s)


def _gunzip(raw: bytes) -> bytes | None:
    try:
        d = zlib.decompressobj(16 + zlib.MAX_WBITS)
        out = d.decompress(raw, _MAX_DECODED)
        return out if d.eof else None
    except zlib.error:
        return None


def b64_bytes(tok: str) -> bytes | None:
    """Strict base64 / base64url decode of a whole token (no whitespace, canonical length)."""
    t = tok.strip()
    if len(t) < 8 or not _B64_STRICT_RE.match(t):
        return None
    body = t.rstrip("=")
    if len(body) % 4 == 1:
        return None
    try:
        if "-" in body or "_" in body:
            return base64.urlsafe_b64decode(body + "=" * (-len(body) % 4))
        return base64.b64decode(body + "=" * (-len(body) % 4), validate=True)
    except (binascii.Error, ValueError):
        return None


def decode_payload(tok: str) -> tuple[str, str] | None:
    """(decoded_text, codec) when `tok` is a base64 / gzip+base64 / hex payload of readable text, else None."""
    t = tok.strip()
    if len(t) >= 16 and len(t) % 2 == 0 and _HEX_RUN_RE.fullmatch(t):
        try:
            raw = bytes.fromhex(t)
        except ValueError:
            raw = b""
        txt = _printable_text(raw)
        if txt is not None and (not t.isdigit() or _wordy_or_id(txt)):
            return txt, ("hex" if t.lower() == t else "HEX")
    raw = b64_bytes(t)
    if raw is None:
        return None
    url = "-" in t or "_" in t
    pad = t.endswith("=")
    if raw[:2] == b"\x1f\x8b":
        inner = _gunzip(raw)
        txt = _printable_text(inner) if inner is not None else None
        return (txt, "gz" + ("u" if url else "") + ("" if pad or len(t) % 4 == 0 else "n")) if txt is not None else None
    txt = _printable_text(raw)
    if txt is None:
        return None
    return txt, "b64" + ("u" if url else "") + ("" if pad or len(t) % 4 == 0 else "n")


def encode_payload(text: str, codec: str) -> str:
    data = text.encode("utf-8")
    if codec in ("hex", "HEX"):
        out = data.hex()
        return out.upper() if codec == "HEX" else out
    if codec.startswith("gz"):
        data = gzip.compress(data, mtime=0)
    enc = base64.urlsafe_b64encode(data) if "u" in codec[2:] else base64.b64encode(data)
    out = enc.decode("ascii")
    return out.rstrip("=") if codec.endswith("n") else out


def binary_text_runs(raw: bytes) -> str:
    """Readable ASCII runs inside binary bytes (what a leak scanner sees after decode(errors='ignore'))."""
    return " ".join(m.group(0).decode("ascii") for m in re.finditer(rb"[\x20-\x7e]{4,}", raw[:_MAX_DECODED]))


def scrub_payload_value(val: str, scrub_fn: Callable[[str], str], attachment: bool = False) -> str | None:
    """Whole-value payload handling. Returns the rewritten value, or None when `val` is not an encoded payload
    (the caller then scrubs it as plain text). `attachment=True` (key/MIME says file content): an undecodable or
    binary base64 blob is replaced by a valid base64 placeholder instead of being shipped raw."""
    dec = decode_payload(val)
    if dec is not None:
        txt, codec = dec
        cleaned = scrub_fn(txt)
        return val if cleaned == txt else encode_payload(cleaned, codec)
    raw = b64_bytes(val)
    if raw is None:
        return None
    if attachment or len(val.strip()) >= 200:  # binary file content (scan, photo, PDF): never shipped raw
        return ATTACHMENT_PLACEHOLDER
    if len(val.strip()) >= 64:
        runs = binary_text_runs(_gunzip(raw) or raw if raw[:2] == b"\x1f\x8b" else raw)
        if runs and scrub_fn(runs) != runs:  # a binary document with readable PII inside (uncompressed PDF text)
            return ATTACHMENT_PLACEHOLDER
    return None


def has_payload_run(text: str) -> bool:
    """Cheap pre-check: does `text` hold a run long enough to be a base64 / hex payload?"""
    return bool(_B64_RUN_RE.search(text) or _HEX_RUN_RE.search(text))


def _scrub_embedded(text: str, scrub_fn: Callable[[str], str]) -> str:
    """Decode, scrub and re-encode base64 / hex payload tokens embedded in free text ("Ref: UmFtZXNo...")."""

    def _one(tok: str) -> str:
        out = scrub_payload_value(tok, scrub_fn)
        return tok if out is None else out

    def _run(m: "re.Match[str]") -> str:
        tok = m.group(0)
        out = scrub_payload_value(tok, scrub_fn)
        if out is not None:
            return out
        if "/" in tok:  # "/view/UmFtZXNo..." : try each path segment on its own
            return "/".join(_one(seg) if len(seg) >= 16 else seg for seg in tok.split("/"))
        return tok

    out = _HEX_RUN_RE.sub(lambda m: _one(m.group(0)), text)
    return _B64_RUN_RE.sub(_run, out)


def scrub_hidden(text: str, scrub_plain: Callable[[str], str], attachment: bool = False, depth: int = 0) -> str:
    """Full scrub of one string value. Hidden payloads (the whole value, or tokens embedded in it) are decoded,
    scrubbed recursively (payload inside payload, up to _MAX_DEPTH) and re-encoded in their own codec; literal
    \\uXXXX escapes are revealed; everything else goes to `scrub_plain`."""
    if text and depth < _MAX_DEPTH and len(text) <= _MAX_DECODED and (attachment or has_payload_run(text)):
        def inner(t: str) -> str:
            return scrub_hidden(t, scrub_plain, False, depth + 1)

        if len(text.strip()) >= 8:
            out = scrub_payload_value(text, inner, attachment)
            if out is not None:
                return out
        text = _scrub_embedded(text, inner)
    return scrub_plain(reveal_escapes(text))


def scrub_embedded_payloads(text: str, scrub_fn: Callable[[str], str]) -> str:
    """Payload tokens inside `text` are decoded, scrubbed with `scrub_fn` and re-encoded; the rest is untouched."""
    if not text or len(text) > _MAX_DECODED or not has_payload_run(text):
        return text
    return _scrub_embedded(text, scrub_fn)


def _decode_uesc(m: "re.Match[str]") -> str:
    """\\u0052\\u0061 -> 'Ra'; surrogate pairs are joined, lone surrogates (U+D800..U+DFFF) stay escaped text."""
    units = [int(m.group(0)[i + 2 : i + 6], 16) for i in range(0, len(m.group(0)), 6)]
    out: list[str] = []
    i = 0
    while i < len(units):
        u = units[i]
        if 0xD800 <= u <= 0xDBFF and i + 1 < len(units) and 0xDC00 <= units[i + 1] <= 0xDFFF:
            out.append(chr(0x10000 + ((u - 0xD800) << 10) + (units[i + 1] - 0xDC00)))
            i += 2
            continue
        if 0xD800 <= u <= 0xDFFF or u < 0x20:
            out.append(m.group(0)[i * 6 : i * 6 + 6])
        else:
            out.append(chr(u))
        i += 1
    return "".join(out)


def reveal_escapes(text: str) -> str:
    """Turn literal \\uXXXX escapes into the characters they spell (a scanner decodes them anyway)."""
    if "\\u" not in text and "\\U" not in text:
        return text
    return _UESC_RE.sub(_decode_uesc, text)


_LONE_SURROGATE_RE = re.compile("[\ud800-\udfff]")


def fix_surrogates(s: str) -> str:
    """Lone UTF-16 surrogates (from JSON "\\ud83d" escapes) cannot be written as UTF-8: replace them."""
    return _LONE_SURROGATE_RE.sub("�", s) if _LONE_SURROGATE_RE.search(s) else s


def fix_surrogates_tree(data: Any) -> Any:
    """Iteratively replace lone surrogates in every key and string value of a parsed JSON tree."""
    def fix_leaf(v: Any) -> Any:
        return fix_surrogates(v) if isinstance(v, str) else v

    if not isinstance(data, (dict, list)):
        return fix_leaf(data)
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
                child = fix_leaf(v)
            if isinstance(dst, dict):
                dst[fix_surrogates(k) if isinstance(k, str) else k] = child
            else:
                dst.append(child)
    return root


def decoded_views(text: str, depth: int = 0) -> list[str]:
    """Every hidden reading of `text` a leak scanner would produce: %XX, HTML entities, \\uXXXX, base64/hex
    payloads (recursively), '_'/'.'-joined words split apart, BiDi-reversed runs."""
    from urllib.parse import unquote_plus

    views: list[str] = []
    if "%" in text:
        v = unquote_plus(text)
        if v != text:
            views.append(v)
    if "&" in text and ";" in text:
        v = html.unescape(text)
        if v != text:
            views.append(v)
    v = reveal_escapes(text)
    if v != text:
        views.append(v)
    if "_" in text or "." in text:
        v = re.sub(r"(?<=[^\W\d_])[_.]+(?=[^\W\d_])", " ", text)
        if v != text:
            views.append(v)
    if "‮" in text:
        views.append(text[::-1])
    if depth < _MAX_DEPTH and len(text) <= _MAX_DECODED:
        for rx in (_HEX_RUN_RE, _B64_RUN_RE):
            for m in rx.finditer(text):
                for tok in [m.group(0)] + [s for s in m.group(0).split("/") if len(s) >= 16]:
                    dec = decode_payload(tok)
                    if dec is not None:
                        views.append(dec[0])
                        views.extend(decoded_views(dec[0], depth + 1))
    return views

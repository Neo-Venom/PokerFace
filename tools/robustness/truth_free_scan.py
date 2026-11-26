"""Truth-free survivor scanner for ANY input/output folder pair (requires no truth.json).

Extracts obvious structured PII values (names, phones, IDs, emails, DOBs, villages, PINs)
directly from input CSV/TSV/JSON files and greps the output directory (sanitized files,
filenames, decoded base64, report.json, records.jsonl, and logs/) for any survivors.
"""
from __future__ import annotations

import argparse
import base64
import binascii
import csv
import io
import json
import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from raksha.detect.structured import infer_column_type
from raksha.ingest.loader import decode_bytes, load_json_safe, sniff_format

_B64_RE = re.compile(r"[A-Za-z0-9+/]{24,}={0,2}")
_DOCTOR_CTX_RE = re.compile(r"(?i)(?:\bdr\.?|doctor|डॉ\.?|consultant|prescriber|ref(?:erred)?\s*by)\s*[:\-]?\s*$")


def _norm(s: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", s).casefold().split())


def _extract_input_candidates(inp_dir: Path) -> tuple[dict[str, str], set[str]]:
    """Collect obvious structured PII surfaces and doctor names from input without reading truth."""
    surfaces: dict[str, str] = {}
    doctors: set[str] = set()

    for p in sorted(inp_dir.rglob("*")):
        if not p.is_file():
            continue
        try:
            txt = decode_bytes(p.read_bytes())
            fmt = sniff_format(p, txt)
        except Exception:
            continue

        if fmt in ("csv", "tsv"):
            delim = "\t" if fmt == "tsv" else ","
            try:
                rows = list(csv.reader(io.StringIO(txt, newline=""), delimiter=delim))
            except csv.Error:
                continue
            if len(rows) < 2:
                continue
            hdr, body = rows[0], rows[1:]
            for c_idx, h in enumerate(hdr):
                vals = [r[c_idx] if c_idx < len(r) else "" for r in body]
                hl = h.casefold()
                if any(w in hl for w in ("doctor", "consultant", "ref_doc")):
                    for v in vals:
                        if v.strip():
                            doctors.add(_norm(re.sub(r"(?i)^dr\.?\s*", "", v)))
                    continue
                ctype = infer_column_type(vals, header=h)
                if ctype in ("PATIENT_NAME", "RELATIVE_NAME", "PHONE", "MRN", "AADHAAR", "ABHA_NUMBER", "ABHA_ADDRESS", "PAN", "EMAIL", "INSURANCE_ID", "VILLAGE", "DOB", "PIN"):
                    for v in vals:
                        vs = v.strip()
                        if len(vs) >= 4:
                            surfaces[vs] = ctype
        elif fmt == "json":
            try:
                data = load_json_safe(txt)
            except ValueError:
                continue
            stack = [(data, "")]
            while stack:
                node, pk = stack.pop()
                if isinstance(node, dict):
                    for k, v in node.items():
                        kl = str(k).casefold()
                        if isinstance(v, str) and v.strip():
                            if kl in ("doctor", "referred_by", "ref_doctor", "consultant", "doc_consultant"):
                                doctors.add(_norm(re.sub(r"(?i)^dr\.?\s*", "", v)))
                            elif any(w in kl for w in ("patient_name", "beneficiary_full_name", "mariz_naam", "uhid", "pt_uid", "aadhaar", "aadhar", "adhaar", "abha", "pan", "mobile", "phone", "sampark", "email")):
                                if len(v.strip()) >= 4:
                                    surfaces[v.strip()] = kl.upper()
                        elif isinstance(v, int) and (100_000_000_000 <= v <= 999_999_999_999 or 6_000_000_000 <= v <= 9_999_999_999):
                            surfaces[str(v)] = "NUMERIC_ID"
                        elif isinstance(v, (dict, list)):
                            stack.append((v, kl))
                elif isinstance(node, list):
                    for it in node:
                        if isinstance(it, (dict, list)):
                            stack.append((it, pk))
    return surfaces, doctors


def _decode_all_b64(txt: str) -> str:
    extras = []
    for m in _B64_RE.finditer(txt):
        tok = m.group(0)
        try:
            extras.append(base64.b64decode(tok + "=" * (-len(tok) % 4)).decode("utf-8", errors="ignore"))
        except (binascii.Error, ValueError):
            pass
    return txt + "\n" + "\n".join(extras) if extras else txt


def scan_truth_free(inp_dir: Path, out_dir: Path) -> list[tuple[str, str, str, str]]:
    """Return list of (rel_file, channel, label, surface) survivors across sanitized/, filenames, report, records, logs."""
    surfaces, doctors = _extract_input_candidates(inp_dir)
    survivors: list[tuple[str, str, str, str]] = []

    ordered = sorted(surfaces.items(), key=lambda kv: (-len(kv[0]), kv[0]))
    for p in sorted(out_dir.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(out_dir).as_posix()
        # Check filename itself (skip quarantine/ which is explicitly non-released)
        if rel.startswith("quarantine/"):
            continue

        rel_norm = _norm(rel.replace("_", " "))
        for surf, lbl in ordered:
            sn = _norm(surf)
            if len(sn) >= 5 and re.search(rf"(?<![a-z0-9]){re.escape(sn)}(?![a-z0-9])", rel_norm):
                survivors.append((rel, "filename", lbl, surf))

        raw_txt = p.read_text(encoding="utf-8", errors="replace")
        full_norm = _norm(_decode_all_b64(raw_txt))
        for surf, lbl in ordered:
            sn = _norm(surf)
            if len(sn) < 4:
                continue
            for m in re.finditer(rf"(?<![a-z0-9\u0900-\u097F]){re.escape(sn)}(?![a-z0-9\u0900-\u097F])", full_norm):
                before = full_norm[max(0, m.start() - 30) : m.start()]
                if sn in doctors or _DOCTOR_CTX_RE.search(before):
                    continue
                survivors.append((rel, "content", lbl, surf))
                break

    return survivors


def main() -> int:
    ap = argparse.ArgumentParser(description="Truth-free survivor scanner across any input/output")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    survivors = scan_truth_free(Path(a.input), Path(a.output))
    print(f"Truth-free scan survivors: {len(survivors)}")
    for rel, chan, lbl, surf in survivors[:40]:
        print(f"  {rel} | {chan} | {lbl} | {surf}")
    return 1 if survivors else 0


if __name__ == "__main__":
    raise SystemExit(main())

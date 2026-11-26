"""Step F: Run-level alias registry with cross-script transliterated variants and role metadata."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from raksha.detect.identifiers import Span, resolve_overlaps
from raksha.detect.normalize import (
    NormalizedView,
    build_normalized_view,
    deva_word_to_latin,
    latin_word_to_deva,
    phonetic_key,
)
from raksha.detect.structured import _is_fhir_bundle, infer_column_type

# General pan-Indian common surnames/titles/locality suffixes that must NOT match standalone as single tokens
_GENERIC_SINGLE_TOKENS = {
    "devi", "kumar", "kumari", "singh", "kaur", "sharma", "verma", "gupta", "patel",
    "yadav", "khan", "das", "rao", "nath", "prasad", "lal", "ram", "ji", "md", "mohd",
    "kalan", "khurd", "pur", "ganj", "garh", "nagar", "bazar",
}

_DEVA_RE = re.compile(r"[\u0900-\u0963\u0970-\u097F]")
_WORD_SPAN_RE = re.compile(r"[A-Za-z\u0900-\u0963\u0970-\u097F][A-Za-z\u0900-\u0963\u0970-\u097F'\-\.]*")


@dataclass
class AliasEntry:
    canonical: str
    label: str  # PATIENT_NAME, RELATIVE_NAME, PHONE, MRN, VILLAGE
    role: str   # patient, relative, doctor
    surface_variants: set[str] = field(default_factory=set)
    phonetic_full: str = ""
    token_count: int = 1


class AliasRegistry:
    """Collects high-confidence structured entities across the run and proposes candidates in text/filenames/base64."""

    def __init__(self) -> None:
        self.doctor_full_names: set[str] = set()
        self.doctor_phonetic: set[str] = set()
        self.full_phrases: dict[str, tuple[str, str]] = {}       # lowered surface -> (label, role)
        self.phonetic_phrases: dict[str, tuple[str, str, int]] = {}  # phonetic_key -> (label, role, tok_len)
        self.first_names: dict[str, tuple[str, str]] = {}        # lowercase/phonetic single given name -> (label, role)
        self.id_cores: dict[str, str] = {}                       # normalized alnum core -> label (MRN / PHONE / AADHAAR)
        self.villages: set[str] = set()
        self._id_core_re: re.Pattern | None = None
        self._id_core_re_dirty: bool = False

    def add_doctor(self, raw_name: str) -> None:
        s = re.sub(r"(?i)^\s*(?:dr\.?|डॉ\.?|डॉक्टर)\s*", "", raw_name).strip(" .,")
        s = re.sub(r"\([^)]*\)", "", s).strip()
        if not s or len(s) < 3:
            return
        self.doctor_full_names.add(s.casefold())
        pk = phonetic_key(s)
        if pk:
            self.doctor_phonetic.add(pk)

    def add_person_name(self, raw_name: str, role: str = "patient") -> None:
        s = raw_name.strip(" .,:;-'\"")
        s = re.sub(r"(?i)^(?:[SDWC]/[Oo]|Mr\.?|Mrs\.?|Ms\.?|Smt\.?|Shri|श्री|श्रीमती)\s+", "", s).strip()
        if not s or len(s) < 3 or any(ch.isdigit() for ch in s):
            return
        label = "RELATIVE_NAME" if role == "relative" else "PATIENT_NAME"
        # Skip if identical to a known doctor full name
        pk = phonetic_key(s)
        if s.casefold() in self.doctor_full_names or (pk and pk in self.doctor_phonetic):
            return

        variants = self._expand_name_variants(s)
        for v in variants:
            # Prefer patient over relative if both appear, or keep first seen
            if v not in self.full_phrases:
                self.full_phrases[v] = (label, role)

        toks = [t for t in re.split(r"[\s_]+", s) if t]
        if pk and len(toks) >= 2:
            self.phonetic_phrases.setdefault(pk, (label, role, len(toks)))
            # Also reversed surname-first / given-last order
            rev_pk = phonetic_key(" ".join(reversed(toks)))
            if rev_pk:
                self.phonetic_phrases.setdefault(rev_pk, (label, role, len(toks)))

        # Register distinctive first token (given name) for single-token proposal in free text
        if toks:
            first = toks[0].strip(".").casefold()
            if len(first) >= 4 and first not in _GENERIC_SINGLE_TOKENS:
                self.first_names.setdefault(first, (label, role))
                fpk = phonetic_key(first)
                if fpk and len(fpk) >= 3:
                    self.first_names.setdefault(fpk, (label, role))

    def _expand_name_variants(self, s: str) -> set[str]:
        out: set[str] = {s.casefold()}
        if _DEVA_RE.search(s):
            lat = " ".join(deva_word_to_latin(w) for w in s.split())
            if lat.strip():
                out.add(lat.strip().casefold())
        else:
            deva = " ".join(latin_word_to_deva(w) for w in s.split())
            if deva.strip():
                out.add(deva.strip().casefold())
            toks = s.split()
            if len(toks) >= 2:
                out.add(" ".join(reversed(toks)).casefold())
                if toks[0].casefold().rstrip(".") in ("mohammad", "mohammed", "muhammad", "mohd", "md"):
                    rest = " ".join(toks[1:]).casefold()
                    for prefix in ("mohammad", "mohammed", "mohd.", "mohd", "md.", "md"):
                        out.add(f"{prefix} {rest}")
        return {x for x in out if len(x) >= 3}

    def add_identifier(self, raw_val: str, label: str) -> None:
        s = raw_val.strip()
        if not s:
            return
        if label == "PHONE":
            digits = re.sub(r"\D", "", s)
            if len(digits) >= 10:
                self.id_cores[digits[-10:]] = "PHONE"
                self._id_core_re_dirty = True
        elif label == "MRN":
            if len(s) >= 5:
                self.id_cores[s.casefold()] = "MRN"
                # Also strip separators for fuzzy/filename matching
                compact = re.sub(r"[^a-z0-9]", "", s.casefold())
                if len(compact) >= 6:
                    self.id_cores[compact] = "MRN"
                self._id_core_re_dirty = True

    def add_village(self, raw_val: str) -> None:
        s = raw_val.strip(" .,-")
        if len(s) >= 4 and not any(ch.isdigit() for ch in s):
            self.villages.add(s.casefold())
            if _DEVA_RE.search(s):
                lat = " ".join(deva_word_to_latin(w) for w in s.split())
                if len(lat.strip()) >= 4:
                    self.villages.add(lat.strip().casefold())

    def propose_spans(
        self,
        text: str,
        view: NormalizedView | None = None,
        allow_single_first_name: bool = True,
    ) -> list[Span]:
        """Propose PII candidate spans in free text/filenames/base64 using registry variants."""
        if not text:
            return []
        nv = view if view is not None else build_normalized_view(text)
        clean = nv.clean
        spans: list[Span] = []

        # 1. Tokenize clean text into word spans for phrase & phonetic matching
        words = [(m.group(0), m.start(), m.end()) for m in _WORD_SPAN_RE.finditer(clean)]
        n_w = len(words)

        # Multi-word exact/transliterated & phonetic matching (window sizes 3, 2)
        for win in (3, 2):
            for i in range(0, max(0, n_w - win + 1)):
                w_slice = words[i : i + win]
                # Ensure tokens are adjacent on the same line (no newline gap)
                between = clean[w_slice[0][1] : w_slice[-1][2]]
                if "\n" in between:
                    continue
                phrase = " ".join(w[0] for w in w_slice)
                phrase_cf = phrase.casefold()
                if phrase_cf in self.doctor_full_names:
                    continue
                matched_label: str | None = None
                if phrase_cf in self.full_phrases:
                    matched_label = self.full_phrases[phrase_cf][0]
                else:
                    pk = phonetic_key(phrase)
                    if pk and pk in self.doctor_phonetic:
                        continue
                    if pk and pk in self.phonetic_phrases and self.phonetic_phrases[pk][2] == win:
                        matched_label = self.phonetic_phrases[pk][0]
                    elif win == 2 and phrase_cf in self.villages:
                        matched_label = "VILLAGE"

                if matched_label:
                    os_, oe_ = nv.map_clean_span(w_slice[0][1], w_slice[-1][2])
                    if oe_ > os_:
                        spans.append(Span(os_, oe_, matched_label, text[os_:oe_], source="registry"))

        # 2. Single-token matching (MRNs, distinctive single-word villages, or distinctive given names)
        for w_tok, ws, we in words:
            w_cf = w_tok.strip(".").casefold()
            if len(w_cf) < 4:
                continue
            if w_cf in self.villages and w_cf not in _GENERIC_SINGLE_TOKENS:
                before = clean[max(0, ws - 25) : ws].casefold()
                if not re.search(r"(?:tehsil|distt|district|तहसील|जिला)\s*[:.]?\s*$", before):
                    os_, oe_ = nv.map_clean_span(ws, we)
                    if oe_ > os_:
                        spans.append(Span(os_, oe_, "VILLAGE", text[os_:oe_], source="registry"))
                continue

            if allow_single_first_name and w_cf not in _GENERIC_SINGLE_TOKENS:
                hit = self.first_names.get(w_cf) or self.first_names.get(phonetic_key(w_cf))
                if hit:
                    before = clean[max(0, ws - 15) : ws].casefold()
                    # Never propose if immediately preceded by Dr./डॉ.
                    if re.search(r"(?:\bdr\.?|डॉ\.?|consultant)\s*$", before):
                        continue
                    os_, oe_ = nv.map_clean_span(ws, we)
                    if oe_ > os_:
                        spans.append(Span(os_, oe_, hit[0], text[os_:oe_], source="registry"))

        # 3. Registered MRN / Phone cores (precompiled once per registry; catches lowercase filenames like adm000161)
        if self._id_core_re_dirty:
            cores = sorted((c for c in self.id_cores if len(c) >= 5), key=lambda x: (-len(x), x))
            if cores:
                self._id_core_re = re.compile(
                    rf"(?<![A-Za-z0-9])(?:{'|'.join(re.escape(c) for c in cores)})(?![A-Za-z0-9])",
                    re.I,
                )
            else:
                self._id_core_re = None
            self._id_core_re_dirty = False

        if self._id_core_re is not None:
            for m in self._id_core_re.finditer(clean):
                lbl = self.id_cores.get(m.group(0).casefold(), "MRN")
                os_, oe_ = nv.map_clean_span(m.start(), m.end())
                if oe_ > os_:
                    spans.append(Span(os_, oe_, lbl, text[os_:oe_], source="registry"))

        return resolve_overlaps(spans)

    def propose_for_filename(self, text: str) -> list[Span]:
        """Match registry names/IDs/phones against filename tokens (underscores treated as spaces)."""
        # Replace '_' with ' ' (1:1 length) so multi-word names in filenames match cleanly
        norm_fn = text.replace("_", " ")
        nv = build_normalized_view(norm_fn)
        return self.propose_spans(norm_fn, view=nv, allow_single_first_name=True)


def build_alias_registry(parsed_items: list[tuple[str, str, object]]) -> AliasRegistry:
    """Populate run-level AliasRegistry from high-confidence structured fields across all ingested files."""
    reg = AliasRegistry()

    for _rel, fmt, payload in parsed_items:
        if fmt in ("csv", "tsv") and isinstance(payload, list) and len(payload) > 1:
            header, body = payload[0], payload[1:]
            col_types = [
                infer_column_type([r[i] if i < len(r) else "" for r in body], header=h)
                for i, h in enumerate(header)
            ]
            # Handle split FirstName + LastName columns together
            fn_idx = next((i for i, h in enumerate(header) if h.strip().casefold() in ("firstname", "first_name", "fname")), None)
            ln_idx = next((i for i, h in enumerate(header) if h.strip().casefold() in ("lastname", "last_name", "lname")), None)
            doc_cols = [i for i, h in enumerate(header) if any(w in h.casefold() for w in ("doctor", "ref_doctor", "consultant"))]

            for row in body:
                for i in doc_cols:
                    if i < len(row) and row[i].strip():
                        reg.add_doctor(row[i])
                if fn_idx is not None and ln_idx is not None:
                    fn = row[fn_idx].strip() if fn_idx < len(row) else ""
                    ln = row[ln_idx].strip() if ln_idx < len(row) else ""
                    full = f"{fn} {ln}".strip()
                    if full:
                        reg.add_person_name(full, role="patient")
                for i, ctype in enumerate(col_types):
                    if i >= len(row):
                        continue
                    val = row[i].strip()
                    if not val:
                        continue
                    if ctype == "PATIENT_NAME":
                        reg.add_person_name(val, role="patient")
                    elif ctype == "RELATIVE_NAME":
                        reg.add_person_name(val, role="relative")
                    elif ctype in ("PHONE", "MRN"):
                        reg.add_identifier(val, ctype)
                    elif ctype == "VILLAGE":
                        reg.add_village(val)

        elif fmt == "json" and isinstance(payload, dict):
            if _is_fhir_bundle(payload):
                for ent in payload.get("entry", []):
                    res = ent.get("resource", {}) if isinstance(ent, dict) else {}
                    rt = res.get("resourceType")
                    if rt == "Practitioner":
                        for nm in res.get("name", []):
                            if isinstance(nm, dict) and isinstance(nm.get("text"), str):
                                reg.add_doctor(nm["text"])
                    elif rt == "Composition":
                        for au in res.get("author", []):
                            if isinstance(au, dict) and isinstance(au.get("display"), str):
                                reg.add_doctor(au["display"])
                    elif rt == "Patient":
                        for nm in res.get("name", []):
                            if isinstance(nm, dict) and isinstance(nm.get("text"), str):
                                reg.add_person_name(nm["text"], role="patient")
                        for ct in res.get("contact", []):
                            if isinstance(ct, dict) and isinstance(ct.get("name"), dict) and isinstance(ct["name"].get("text"), str):
                                reg.add_person_name(ct["name"]["text"], role="relative")
                        for ext in res.get("extension", []):
                            if isinstance(ext, dict) and isinstance(ext.get("valueString"), str):
                                if any(w in str(ext.get("url", "")).casefold() for w in ("mother", "father", "relative", "guardian")):
                                    reg.add_person_name(ext["valueString"], role="relative")
                        for ident in res.get("identifier", []):
                            if isinstance(ident, dict) and isinstance(ident.get("value"), str):
                                reg.add_identifier(ident["value"], "MRN")
                        for tc in res.get("telecom", []):
                            if isinstance(tc, dict) and isinstance(tc.get("value"), str):
                                reg.add_identifier(tc["value"], "PHONE")
                        for addr in res.get("address", []):
                            if isinstance(addr, dict) and isinstance(addr.get("city"), str):
                                reg.add_village(addr["city"])
                    elif rt == "Encounter":
                        for ident in res.get("identifier", []):
                            if isinstance(ident, dict) and isinstance(ident.get("value"), str):
                                reg.add_identifier(ident["value"], "MRN")
            else:
                # Iterative extraction from structured JSON dicts
                stack: list[tuple[object, str]] = [(payload, "")]
                while stack:
                    node, pk = stack.pop()
                    if isinstance(node, dict):
                        # Compound name object {first, last, local}
                        if pk == "name" or ("first" in node and "last" in node):
                            first = str(node.get("first") or "").strip()
                            last = str(node.get("last") or "").strip()
                            loc = str(node.get("local") or "").strip()
                            if f"{first} {last}".strip():
                                reg.add_person_name(f"{first} {last}".strip(), role="patient")
                            if loc:
                                reg.add_person_name(loc, role="patient")
                        for k, v in node.items():
                            kl = str(k).casefold()
                            if isinstance(v, str) and v.strip():
                                if kl in ("doctor", "referred_by", "ref_doctor", "consultant"):
                                    reg.add_doctor(v)
                                elif kl in ("patient_name", "naam") or (kl == "name" and pk in ("patient", "raised_by")):
                                    reg.add_person_name(v, role="patient")
                                elif kl == "name" and pk in ("guardian", "relative"):
                                    reg.add_person_name(v, role="relative")
                                elif kl in ("uhid", "patient_ref", "hospital_uhid", "mrn"):
                                    reg.add_identifier(v, "MRN")
                                elif kl in ("phone", "mobile", "callback"):
                                    reg.add_identifier(v, "PHONE")
                                elif kl == "village":
                                    reg.add_village(v)
                            elif isinstance(v, (dict, list)):
                                stack.append((v, kl))
                    elif isinstance(node, list):
                        for it in node:
                            if isinstance(it, (dict, list)):
                                stack.append((it, pk))

    return reg

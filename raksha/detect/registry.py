"""Step F: Run-level alias registry with cross-script transliterated variants and role metadata."""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from raksha.detect.identifiers import Span, resolve_overlaps
from raksha.detect.normalize import (
    DEVA_LETTER_CLASS,
    LATIN_LETTER_CLASS,
    NormalizedView,
    build_normalized_view,
    deva_word_to_latin,
    fold_latin_accents,
    latin_word_to_deva,
    phonetic_key,
)
from raksha.detect.structured import (
    _is_fhir_bundle,
    infer_column_type,
    infer_dict_schema,
    infer_table_schema,
    is_non_patient_table_content,
)

# General pan-Indian common surnames/titles/locality suffixes that must NOT match standalone as single tokens
_GENERIC_SINGLE_TOKENS = {
    "devi", "kumar", "kumari", "singh", "kaur", "sharma", "verma", "gupta", "patel",
    "yadav", "khan", "das", "rao", "nath", "prasad", "lal", "ram", "ji", "md", "mohd",
    "kalan", "khurd", "pur", "ganj", "garh", "nagar", "bazar",
}

_DEVA_RE = re.compile(rf"[{DEVA_LETTER_CLASS}]")
_WORD_SPAN_RE = re.compile(rf"[{LATIN_LETTER_CLASS}{DEVA_LETTER_CLASS}][{LATIN_LETTER_CLASS}{DEVA_LETTER_CLASS}'\-\.]*")


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
        s = unicodedata.normalize("NFC", raw_name).strip(" .,:;-'\"")
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

        # Register distinctive first token (given name + accent-folded variant) for free text/filename matching
        if toks:
            first = toks[0].strip(".").casefold()
            first_ascii = fold_latin_accents(toks[0]).strip(".").casefold()
            for cand_f in {first, first_ascii}:
                if len(cand_f) >= 4 and cand_f not in _GENERIC_SINGLE_TOKENS:
                    self.first_names.setdefault(cand_f, (label, role))
            fpk = phonetic_key(first)
            if fpk and len(fpk) >= 3:
                self.first_names.setdefault(fpk, (label, role))

    def _expand_name_variants(self, s: str) -> set[str]:
        s_nfc = unicodedata.normalize("NFC", s)
        out: set[str] = {s_nfc.casefold(), fold_latin_accents(s_nfc).casefold()}
        if _DEVA_RE.search(s_nfc):
            lat = " ".join(deva_word_to_latin(w) for w in s_nfc.split())
            if lat.strip():
                out.add(lat.strip().casefold())
        else:
            ascii_s = fold_latin_accents(s_nfc)
            deva = " ".join(latin_word_to_deva(w) for w in ascii_s.split())
            if deva.strip():
                out.add(deva.strip().casefold())
            toks = s_nfc.split()
            atoks = ascii_s.split()
            if len(toks) >= 2:
                out.add(" ".join(reversed(toks)).casefold())
                out.add(" ".join(reversed(atoks)).casefold())
                if atoks[0].casefold().rstrip(".") in ("mohammad", "mohammed", "muhammad", "mohd", "md"):
                    rest = " ".join(atoks[1:]).casefold()
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


def build_alias_registry(
    parsed_items: list[tuple[str, str, object]],
    clinical_lexicon=None,
) -> AliasRegistry:
    """Populate run-level AliasRegistry via unified compositional schema inference across all files."""
    from raksha.detect.names_context import detect_context_name_and_address_spans
    from raksha.detect.protected import resolve_pii_vs_protected

    reg = AliasRegistry()

    for _rel, fmt, payload in parsed_items:
        if fmt in ("csv", "tsv") and isinstance(payload, list) and len(payload) > 1:
            header, body = payload[0], payload[1:]
            ts = infer_table_schema(header, body)
            if is_non_patient_table_content(header, body, ts):
                continue
            col_types = ts.col_types
            fn_idx = ts.first_name_cols[0] if ts.first_name_cols else None
            ln_idx = ts.last_name_cols[0] if ts.last_name_cols else None
            doc_cols = ts.doctor_cols

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
                    if ctype == "PATIENT_NAME" and i not in (fn_idx, ln_idx):
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
                # Iterative extraction from structured JSON dicts using infer_dict_schema
                stack: list[tuple[object, str]] = [(payload, "")]
                while stack:
                    node, pk = stack.pop()
                    if isinstance(node, dict):
                        ds = infer_dict_schema(node, parent_key=pk)
                        # Handle nested person/relative name objects ({given_nm, sur_nm, local} or {first, last})
                        for sub_k, role_tag in ds.sub_dict_roles.items():
                            sub_node = node.get(sub_k)
                            if isinstance(sub_node, dict):
                                sub_ds = infer_dict_schema(sub_node, parent_key=sub_k)
                                role = "relative" if role_tag == "RELATIVE_NAME_OBJ" else "patient"
                                f_vals = [
                                    str(sub_node[ck]).strip()
                                    for ck, ph in sub_ds.part_hints.items()
                                    if ph in ("NAME_FIRST", "REL_NAME_FIRST") and isinstance(sub_node.get(ck), str)
                                ]
                                l_vals = [
                                    str(sub_node[ck]).strip()
                                    for ck, ph in sub_ds.part_hints.items()
                                    if ph in ("NAME_LAST", "REL_NAME_LAST") and isinstance(sub_node.get(ck), str)
                                ]
                                loc_vals = [
                                    str(sub_node[ck]).strip()
                                    for ck, ph in sub_ds.part_hints.items()
                                    if ph == "NAME_LOCAL" and isinstance(sub_node.get(ck), str)
                                ]
                                full = " ".join(x for x in f_vals + l_vals if x).strip()
                                if full:
                                    reg.add_person_name(full, role=role)
                                for lv in loc_vals:
                                    if lv:
                                        reg.add_person_name(lv, role=role)
                                if not full and not loc_vals:
                                    raw_parts = [str(sv).strip() for sv in sub_node.values() if isinstance(sv, str) and sv.strip()]
                                    if raw_parts:
                                        reg.add_person_name(" ".join(raw_parts), role=role)

                        for k, v in node.items():
                            kl = str(k).casefold()
                            if isinstance(v, (dict, list)):
                                if k not in ds.sub_dict_roles:
                                    stack.append((v, kl))
                                continue
                            if v is None:
                                continue
                            vs = str(v).strip()
                            if not vs:
                                continue
                            ph = ds.part_hints.get(str(k))
                            ft = ds.field_types.get(str(k))
                            if ph == "DOCTOR":
                                reg.add_doctor(vs)
                            elif ft == "PATIENT_NAME":
                                reg.add_person_name(vs, role="patient")
                            elif ft == "RELATIVE_NAME":
                                reg.add_person_name(vs, role="relative")
                            elif ft in ("MRN", "PHONE"):
                                reg.add_identifier(vs, ft)
                            elif ft == "VILLAGE":
                                reg.add_village(vs)
                            elif len(vs) >= 24 and not isinstance(v, (dict, list)):
                                from raksha.export.filenames import sanitize_base64_string

                                def _reg_from_b64(dec_txt: str) -> str:
                                    nv_b = build_normalized_view(dec_txt)
                                    prot_b = clinical_lexicon.find_spans(dec_txt, view=nv_b) if clinical_lexicon is not None else []
                                    for sp in resolve_pii_vs_protected(
                                        detect_context_name_and_address_spans(dec_txt, view=nv_b, protected_spans=prot_b),
                                        prot_b,
                                    ):
                                        if sp.label in ("PATIENT_NAME", "RELATIVE_NAME"):
                                            toks_b = [t for t in sp.text.strip().split() if len(t) >= 2]
                                            if len(toks_b) >= 2:
                                                reg.add_person_name(sp.text, role="relative" if sp.label == "RELATIVE_NAME" else "patient")
                                    return dec_txt

                                sanitize_base64_string(vs, _reg_from_b64)
                    elif isinstance(node, list):
                        for it in node:
                            if isinstance(it, (dict, list)):
                                stack.append((it, pk))

        elif fmt == "txt" and isinstance(payload, str):
            for m_dr in re.finditer(rf"(?i)\b(?:dr\.?|डॉ\.?|consultant)\s+([{LATIN_LETTER_CLASS}{DEVA_LETTER_CLASS}][{LATIN_LETTER_CLASS}{DEVA_LETTER_CLASS}.'\-]+(?:\s+[{LATIN_LETTER_CLASS}{DEVA_LETTER_CLASS}.'\-]+){{1,2}})", payload):
                reg.add_doctor(m_dr.group(1))
            nv = build_normalized_view(payload)
            prot = clinical_lexicon.find_spans(payload, view=nv) if clinical_lexicon is not None else []
            ctx_spans = resolve_pii_vs_protected(
                detect_context_name_and_address_spans(payload, view=nv, protected_spans=prot),
                prot,
            )
            for sp in ctx_spans:
                if sp.label in ("PATIENT_NAME", "RELATIVE_NAME"):
                    toks = [t for t in sp.text.strip().split() if len(t) >= 2]
                    if len(toks) >= 2:
                        reg.add_person_name(sp.text, role="relative" if sp.label == "RELATIVE_NAME" else "patient")

    return reg

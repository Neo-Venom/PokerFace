"""Unicode, zero-width, homoglyph, Devanagari numeral, and cross-script Indic canonicalization."""
from __future__ import annotations

import re
import unicodedata

from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

from data.gazetteers import HONORIFICS_AFTER, HONORIFICS_BEFORE

_DEVA = re.compile(r"[\u0900-\u097f]")
_VIRAMA = "\u094d"
_MATRAS = set(chr(c) for c in range(0x093E, 0x094D)) | {"\u0962", "\u0963"}
_ZW_RE = re.compile(r"[\u200b-\u200f\ufeff\u2060]")
_DEVA_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
_HOMOGLYPHS = str.maketrans({
    "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M", "Н": "H", "О": "O", "Р": "P", "С": "C", "Т": "T", "Х": "X",
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "х": "x", "у": "y", "і": "i",
    "Α": "A", "Β": "B", "Ε": "E", "Ζ": "Z", "Η": "H", "Ι": "I", "Κ": "K", "Μ": "M", "Ν": "N", "Ο": "O", "Ρ": "P", "Τ": "T", "Χ": "X",
})

_NAME_ALIASES = {
    "md": "mohammad",
    "mohd": "mohammad",
    "mhd": "mohammad",
    "muhd": "mohammad",
    "mohammed": "mohammad",
    "muhammad": "mohammad",
    "mohamad": "mohammad",
    "kr": "kumar",
    "pd": "prasad",
    "sk": "shaikh",
}

_ALL_HONORIFICS = HONORIFICS_BEFORE | HONORIFICS_AFTER
GENERIC_WEAK_TOKENS = {
    "devi", "kumar", "kumari", "khatoon", "begum", "bano", "singh", "kaur", "mohamad", "shekh",
}


def shadow_normalize(text: str) -> str:
    """Normalize text for regex/detection while keeping 1:1 length where possible."""
    s = _ZW_RE.sub(" ", text)
    s = s.translate(_DEVA_DIGITS).translate(_HOMOGLYPHS)
    return s


def norm_match_text(text: str) -> str:
    """NFKC + casefold + Devanagari digits -> ASCII + collapsed whitespace for exact matching."""
    s = unicodedata.normalize("NFKC", text)
    s = _ZW_RE.sub("", s).translate(_DEVA_DIGITS).translate(_HOMOGLYPHS)
    s = "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))
    return " ".join(s.casefold().split())


_NUKTA_MAP = {
    "\u0915\u093c": "\u0915",  # क़ -> क
    "\u0916\u093c": "\u0916",  # ख़ -> ख
    "\u0917\u093c": "\u0917",  # ग़ -> ग
    "\u091c\u093c": "\u091c",  # ज़ -> ज
    "\u0921\u093c": "\u0921",  # ड़ -> ड
    "\u0922\u093c": "\u0922",  # ढ़ -> ढ
    "\u092b\u093c": "\u092b",  # फ़ -> फ
}
_NULL_NAME_TOKENS = {
    "na", "n/a", "nil", "none", "unknown", "null", "prescription", "discharge", "address", "adres",
}


def romanize_token(token: str) -> str:
    if not _DEVA.search(token):
        return token
    t = token
    for src, dst in _NUKTA_MAP.items():
        t = t.replace(src, dst)
    t = t.replace("\u093c", "")
    bare_end = len(t) > 0 and t[-1] not in _MATRAS and t[-1] != _VIRAMA and "\u0915" <= t[-1] <= "\u0939"
    s = transliterate(t, sanscript.DEVANAGARI, sanscript.ITRANS)
    for mark in ("M", "~N", ".N", "~n", ".n"):
        s = s.replace(mark, "n")
    s = s.lower()
    if bare_end and s.endswith("a") and len(s) > 2:
        s = s[:-1]
    return s


def normalize_name_token(token: str) -> str:
    raw = unicodedata.normalize("NFKC", token).strip(".,:;()[]'\"`~")
    if not raw:
        return ""
    rom = romanize_token(raw).lower()
    rom = "".join(c for c in unicodedata.normalize("NFKD", rom) if not unicodedata.combining(c))
    clean = re.sub(r"[^a-z]", "", rom)
    if not clean:
        return ""
    clean = _NAME_ALIASES.get(clean, clean)
    for src, dst in (
        ("aa", "a"),
        ("ee", "i"),
        ("ii", "i"),
        ("oo", "u"),
        ("uu", "u"),
        ("ai", "e"),
        ("ui", "i"),
        ("ou", "au"),
        ("w", "v"),
        ("ph", "f"),
        ("z", "j"),
        ("q", "k"),
        ("x", "ks"),
        ("y", "i"),
    ):
        clean = clean.replace(src, dst)
    # Collapse consecutive identical vowels/consonants
    clean = re.sub(r"(.)\1+", r"\1", clean)
    return clean


def phonetic_skeleton(token: str) -> str:
    """Coarse consonant skeleton for fast blocking only (never use alone for redaction)."""
    t = normalize_name_token(token)
    if not t:
        return ""
    t = re.sub(r"([kgcjtdpbs])h", r"\1", t)
    sk = t[:1] + re.sub(r"[aeiou]", "", t[1:])
    return re.sub(r"(.)\1+", r"\1", sk)


def name_tokens(name: str) -> list[str]:
    out: list[str] = []
    for raw in re.split(r"[\s_/,.-]+", name):
        stripped = raw.strip(".,:;()[]'\"").lower()
        if not stripped or stripped in _ALL_HONORIFICS or stripped in _NULL_NAME_TOKENS:
            continue
        normed = normalize_name_token(raw)
        if not normed or normed in _ALL_HONORIFICS or normed in _NULL_NAME_TOKENS:
            continue
        out.append(normed)
    return out


def name_key_tuple(name: str) -> tuple[str, ...]:
    toks = name_tokens(name)
    return tuple(sorted(set(toks)))


def detect_script(text: str) -> str:
    if _DEVA.search(text):
        return "deva"
    low = text.lower()
    if any(w in low.split() for w in ("ji", "sahab", "saab", "didi", "bhai", "bhaiya", "amma", "papa", "babu", "ka", "ki", "ke", "ko")):
        return "hinglish"
    return "latin"

"""Step C: Unicode/script normalization and Devanagari<->Latin transliteration with offset maps."""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

_ZERO_WIDTH = {"\u200b", "\u200c", "\u200d", "\ufeff", "\u00ad"}

# Cyrillic / Greek lookalike homoglyphs folded to ASCII Latin
_HOMOGLYPHS = str.maketrans({
    "А": "A", "а": "a", "В": "B", "Е": "E", "е": "e", "З": "3",
    "І": "I", "і": "i", "К": "K", "к": "k", "М": "M", "Н": "H",
    "О": "O", "о": "o", "Р": "P", "р": "p", "С": "C", "с": "c",
    "Т": "T", "т": "t", "Х": "X", "х": "x", "У": "Y", "у": "y",
    "Α": "A", "α": "a", "Β": "B", "Ε": "E", "ε": "e", "Ζ": "Z",
    "Η": "H", "Ι": "I", "ι": "i", "Κ": "K", "Μ": "M", "Ν": "N",
    "Ο": "O", "ο": "o", "Ρ": "P", "ρ": "p", "Τ": "T", "Χ": "X",
})

_DEVA_RUN_RE = re.compile(r"[\u0900-\u097F]+")

# Shared Unicode Latin & Devanagari letter classes (covers composed NFC Latin-1 & Latin Extended-A/B/Additional)
LATIN_LETTER_CLASS = r"A-Za-z\u00C0-\u00D6\u00D8-\u00F6\u00F8-\u024F\u1E00-\u1EFF"
LATIN_UPPER_CLASS = (
    r"A-Z\u00C0-\u00D6\u00D8-\u00DE"
    r"\u0100\u0102\u0104\u0106\u0108\u010A\u010C\u010E\u0110\u0112\u0114\u0116\u0118\u011A"
    r"\u011C\u011E\u0120\u0122\u0124\u0126\u0128\u012A\u012C\u012E\u0130\u0132\u0134\u0136"
    r"\u0139\u013B\u013D\u013F\u0141\u0143\u0145\u0147\u014A\u014C\u014E\u0150\u0152\u0154"
    r"\u0156\u0158\u015A\u015C\u015E\u0160\u0162\u0164\u0166\u0168\u016A\u016C\u016E\u0170"
    r"\u0172\u0174\u0176\u0178\u0179\u017B\u017D"
)
DEVA_LETTER_CLASS = r"\u0900-\u0963\u0970-\u097F"


def is_deva_char(ch: str) -> bool:
    return "\u0900" <= ch <= "\u0963" or "\u0970" <= ch <= "\u097F"


def fold_latin_accents(text: str) -> str:
    """Fold composed or decomposed Latin accents to ASCII letters while preserving Devanagari."""
    if text.isascii():
        return text
    out: list[str] = []
    for ch in unicodedata.normalize("NFKC", text):
        if ord(ch) < 128 or "\u0900" <= ch <= "\u097F":
            out.append(ch)
        else:
            for sub in unicodedata.normalize("NFKD", ch):
                if not unicodedata.combining(sub):
                    out.append(sub)
    return "".join(out)


def is_upper_or_deva_start(tok: str) -> bool:
    """True if token starts with any Unicode uppercase Latin letter (incl. accented) or Devanagari letter."""
    if not tok:
        return False
    ch = unicodedata.normalize("NFC", tok)[0]
    return ch.isupper() or is_deva_char(ch)

# Devanagari vowel signs/vowels that end in an explicit vowel (do NOT drop trailing 'a' from ITRANS)
_DEVA_EXPLICIT_END = set("अआइईउऊएऐओऔअंअःािीुूेैोौंः")


def deva_word_to_latin(word: str) -> str:
    """Transliterate a Devanagari word to natural lowercase Roman/Hinglish form."""
    if not word:
        return ""
    raw = transliterate(word, sanscript.DEVANAGARI, sanscript.ITRANS)
    # Clean ITRANS special capitals/digraphs
    s = raw
    s = s.replace("K", "kh").replace("q", "q").replace("G", "gh").replace("J", "z")
    s = s.replace(".D", "d").replace(".Dh", "dh").replace("Sh", "sh").replace("Ch", "chh")
    s = s.replace("GY", "gy").replace("~N", "n").replace("~n", "n").replace(".N", "n").replace(".n", "n")
    s = s.replace("M", "n").replace("H", "h")
    s = s.lower()
    # Hindi schwa deletion: bare-consonant ending words in ITRANS gain a phantom trailing 'a'
    if (
        len(s) >= 4
        and s.endswith("a")
        and not s.endswith(("aa", "ia", "ua"))
        and word[-1] not in _DEVA_EXPLICIT_END
    ):
        s = s[:-1]
    return s


def latin_word_to_deva(word: str) -> str:
    """Transliterate a Latin word into approximate Devanagari using indic-transliteration."""
    if not word:
        return ""
    # Heuristic ITRANS vowel lengthening for common Hindi names when given lowercase ASCII
    s = re.sub(r"ee", "I", word.lower())
    s = re.sub(r"oo", "U", s)
    s = re.sub(r"a$", "A", s)
    s = re.sub(r"i$", "I", s)
    out = transliterate(s, sanscript.ITRANS, sanscript.DEVANAGARI)
    # Strip trailing virama (halant) typical of ITRANS consonant-ending conversions
    if out.endswith("\u094d"):
        out = out[:-1]
    return out


def phonetic_key(text: str) -> str:
    """Script-agnostic canonical phonetic key for cross-script matching."""
    s = unicodedata.normalize("NFC", text)
    s = "".join(ch for ch in s if ch not in _ZERO_WIDTH)
    s = s.translate(_HOMOGLYPHS)
    # Convert Devanagari words to Latin
    s = _DEVA_RUN_RE.sub(lambda m: deva_word_to_latin(m.group(0)), s)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.casefold()
    # Common Indian transliteration equivalences
    toks = re.findall(r"[a-z0-9]+", s)
    norm_toks: list[str] = []
    for t in toks:
        if t in ("md", "mohd", "muhammad", "mohammad", "mohammed", "muhammed"):
            norm_toks.append("mohammad")
            continue
        t = t.replace("ee", "i").replace("oo", "u").replace("ou", "au")
        t = re.sub(r"(.)\1+", r"\1", t)  # collapse double letters
        t = t.replace("ph", "f").replace("v", "w").replace("q", "k").replace("z", "j").replace("sh", "s")
        t = t.replace("dh", "d").replace("th", "t").replace("bh", "b").replace("kh", "k").replace("gh", "g")
        # Medial and final schwa tolerance for Devanagari->ITRANS vs Romanized Hindi spellings
        t = re.sub(r"(?<=[bcdfghjklmnpqrstvwxyz])a(?=[bcdfghjklmnpqrstvwxyz])", "", t)
        t = re.sub(r"([bcdfghjklmnpqrstvwxyz])a$", r"\1", t)
        norm_toks.append(t)
    return " ".join(norm_toks)


@dataclass
class NormalizedView:
    """Dual representation: original text + normalized/transliterated analysis forms with offset maps."""

    original: str
    clean: str
    clean_to_orig: list[tuple[int, int]]
    latin_folded: str
    folded_to_orig: list[tuple[int, int]]
    _is_identity: bool = False

    def map_clean_span(self, start: int, end: int) -> tuple[int, int]:
        if start >= end:
            return (0, 0)
        if self._is_identity:
            n = len(self.original)
            s = max(0, min(start, n))
            e = max(s, min(end, n))
            return (s, e)
        if not self.clean_to_orig:
            return (0, 0)
        s = max(0, min(start, len(self.clean_to_orig) - 1))
        e = max(s + 1, min(end, len(self.clean_to_orig)))
        return (self.clean_to_orig[s][0], self.clean_to_orig[e - 1][1])

    def map_folded_span(self, start: int, end: int) -> tuple[int, int]:
        if start >= end:
            return (0, 0)
        if self._is_identity:
            n = len(self.original)
            s = max(0, min(start, n))
            e = max(s, min(end, n))
            return (s, e)
        if not self.folded_to_orig:
            return (0, 0)
        s = max(0, min(start, len(self.folded_to_orig) - 1))
        e = max(s + 1, min(end, len(self.folded_to_orig)))
        return (self.folded_to_orig[s][0], self.folded_to_orig[e - 1][1])


def build_normalized_view(text: str) -> NormalizedView:
    """Build NFC, zero-width-free, digit-ASCII, homoglyph-folded, and Devanagari->Latin representations with exact offset maps."""
    if text.isascii() and len(text) > 10000:
        return NormalizedView(
            original=text,
            clean=text,
            clean_to_orig=[],
            latin_folded=text,
            folded_to_orig=[],
            _is_identity=True,
        )
    # Step 1: group original chars into combining clusters, stripping zero-width chars
    clusters: list[tuple[str, int, int]] = []
    for idx, ch in enumerate(text):
        if ch in _ZERO_WIDTH:
            continue
        if clusters and ord(ch) >= 0x0300 and unicodedata.combining(ch) > 0:
            prev_txt, p_start, _ = clusters[-1]
            clusters[-1] = (prev_txt + ch, p_start, idx + 1)
        else:
            clusters.append((ch, idx, idx + 1))

    clean_chars: list[str] = []
    clean_to_orig: list[tuple[int, int]] = []
    for raw_seg, o_start, o_end in clusters:
        if len(raw_seg) == 1 and ord(raw_seg[0]) < 128:
            clean_chars.append(raw_seg)
            clean_to_orig.append((o_start, o_end))
            continue
        seg = unicodedata.normalize("NFC", raw_seg)
        # Fold NFKC compatibility forms (fullwidth digits/ASCII, ligatures) without breaking Devanagari nukta
        if not ("\u0900" <= seg[0] <= "\u097F"):
            seg = unicodedata.normalize("NFKC", seg)
        seg = seg.translate(_HOMOGLYPHS)
        for c in seg:
            code = ord(c)
            if 0x0966 <= code <= 0x096F:
                c = chr(ord("0") + (code - 0x0966))
            clean_chars.append(c)
            clean_to_orig.append((o_start, o_end))

    clean = "".join(clean_chars)

    # Step 2: build Devanagari->Latin folded representation with exact span map back to original
    folded_chars: list[str] = []
    folded_to_orig: list[tuple[int, int]] = []
    pos = 0
    for m in _DEVA_RUN_RE.finditer(clean):
        if m.start() > pos:
            for i in range(pos, m.start()):
                folded_chars.append(clean[i])
                folded_to_orig.append(clean_to_orig[i])
        deva_tok = m.group(0)
        lat_tok = deva_word_to_latin(deva_tok)
        orig_span = (clean_to_orig[m.start()][0], clean_to_orig[m.end() - 1][1])
        for ch in lat_tok:
            folded_chars.append(ch)
            folded_to_orig.append(orig_span)
        pos = m.end()
    if pos < len(clean):
        for i in range(pos, len(clean)):
            folded_chars.append(clean[i])
            folded_to_orig.append(clean_to_orig[i])

    return NormalizedView(
        original=text,
        clean=clean,
        clean_to_orig=clean_to_orig,
        latin_folded="".join(folded_chars),
        folded_to_orig=folded_to_orig,
    )

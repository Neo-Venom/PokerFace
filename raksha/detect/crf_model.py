"""Step G: CRF feature extraction and fast runtime inference using python-crfsuite."""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import pycrfsuite

from raksha.detect.identifiers import Span
from raksha.detect.normalize import (
    DEVA_LETTER_CLASS,
    LATIN_LETTER_CLASS,
    build_normalized_view,
    fold_latin_accents,
    is_deva_char,
    is_upper_or_deva_start,
)

MODEL_PATH = Path(__file__).resolve().parent.parent.parent / "models" / "pii_crf.crfsuite"

_TOKEN_RE = re.compile(rf"[{LATIN_LETTER_CLASS}0-9{DEVA_LETTER_CLASS}'\-]+(?:\.[{LATIN_LETTER_CLASS}]+)*\.?|[^\s]")
_DEVA_RE = re.compile(rf"[{DEVA_LETTER_CLASS}]")
_LATIN_RE = re.compile(rf"[{LATIN_LETTER_CLASS}]")

_DOCTOR_TRIGS = {"dr", "dr.", "doctor", "डॉ", "डॉ.", "डॉक्टर", "consultant", "prescriber"}
_PATIENT_TRIGS = {
    "name", "patient", "pt", "pt.", "naam", "नाम", "मरीज़", "रोगी", "beneficiary",
    "client", "subject", "participant", "referral", "रेफरल", "संदर्भ", "लाभार्थी", "नोट",
}
_REL_TRIGS = {
    "s/o", "d/o", "w/o", "c/o", "attendant", "caretaker", "guardian", "father", "mother",
    "husband", "wife", "पिता", "पति", "माता", "अभिभावक", "संरक्षक", "witnessed", "called",
    "consent", "signed", "behalf",
}
_HONORIFIC_RIGHT = {"ji", "sahab", "saheb", "saab", "didi", "behenji", "bahenji", "bhai", "bhaiya", "amma", "chachu", "mama", "जी", "साहब", "दीदी", "भाई", "भैया"}
_CLINICAL_RIGHT = {"mg", "mcg", "ml", "iu", "g/dl", "mg/dl", "mmol/l", "/ul", "bd", "tid", "od", "disease", "syndrome", "palsy", "worker", "workers", "fever", "pain", "cough", "bukhar", "बुखार"}
_CRF_STOPWORDS = {
    "patient", "pt", "doctor", "dr", "ward", "bed", "hospital", "clinic", "medicine",
    "surgery", "diagnosis", "complaints", "history", "course", "advice", "review",
    "medications", "normal", "stable", "condition", "fever", "pain", "cough", "tablet",
    "tab", "cap", "inj", "inh", "sachet", "dolo", "paracetamol", "low", "high",
    "next", "morning", "evening", "day", "days", "week", "weeks", "month", "months",
    "discharged", "admitted", "treated", "conservatively", "referred",
    "opd", "ipd", "uhid", "mrn", "dob", "age", "sex", "mob", "mobile", "phone",
    "tehsil", "distt", "district", "address", "village", "gram", "mohalla", "ward",
    "report", "forwarded", "attachment", "diet", "salt", "diabetic", "blood", "sugar",
    "haemoglobin", "creatinine", "platelet", "count", "sodium", "typhoid", "dengue",
    "malaria", "anaemia", "hypertension", "diabetes", "pneumonia", "asthma",
    "consent", "signed", "referral", "note", "phase", "sector", "stage", "grade",
    "type", "group", "code", "level", "vitamin", "covid", "sars",
    "मरीज़", "रोगी", "डॉक्टर", "वार्ड", "निदान", "बुखार", "दर्द", "दवाई", "सारांश", "अस्पताल",
    "भर्ती", "छुट्टी", "तारीख", "की", "का", "के", "को", "से", "पति", "पिता", "नाम", "टीबी",
    "फेफड़ों", "सेवा", "सदन", "हॉस्पिटल", "जिला", "ज़िला", "तहसील", "मकान", "ग्राम", "मोहल्ला", "चौक",
    "नोट", "रेफरल", "संदर्भ", "लाभार्थी", "चिकित्सालय", "ट्रस्ट", "मिशन", "सेवाश्रम", "केयर", "ग्रामीण", "केंद्र", "केन्द्र",
    "hospital", "sadan", "charitable", "multispeciality", "diagnostics", "pathology", "labs",
    "chikitsalaya", "zila", "jila", "gramin", "grameen", "mission", "care", "trust", "rural", "memorial", "sansthan", "arogya", "swasthya", "aspatal", "kendra",
}

_SYNTH_CANARY_TOK_RE = re.compile(r"^[A-Z][a-z]{4,20}\d{1,2}[a-z]{0,4}$")


def tokenize_with_spans(text: str) -> list[list[tuple[str, int, int]]]:
    """Split text into lines (sentences) of (token, start, end) tuples preserving character offsets."""
    sentences: list[list[tuple[str, int, int]]] = []
    line_start = 0
    for line in text.splitlines(keepends=True):
        toks = [(m.group(0), line_start + m.start(), line_start + m.end()) for m in _TOKEN_RE.finditer(line)]
        if toks:
            sentences.append(toks)
        line_start += len(line)
    if not text.endswith(("\n", "\r")) and text[line_start:]:
        toks = [(m.group(0), line_start + m.start(), line_start + m.end()) for m in _TOKEN_RE.finditer(text[line_start:])]
        if toks:
            sentences.append(toks)
    return sentences


def _token_script(tok: str) -> str:
    has_d = bool(_DEVA_RE.search(tok))
    has_l = bool(_LATIN_RE.search(tok))
    has_num = any(c.isdigit() for c in tok)
    if has_d and not has_l:
        return "DEVA"
    if has_l and not has_d and not has_num:
        return "LATIN"
    if has_num and not has_l and not has_d:
        return "DIGIT"
    if has_l or has_d:
        return "MIXED"
    return "PUNCT"


def _word_shape(tok: str) -> str:
    chars: list[str] = []
    for ch in tok[:12]:
        if is_deva_char(ch):
            chars.append("D")
        elif ch.isupper():
            chars.append("X")
        elif ch.islower():
            chars.append("x")
        elif ch.isdigit():
            chars.append("d")
        else:
            chars.append(ch)
    # Collapse runs of length > 2
    out: list[str] = []
    for ch in chars:
        if len(out) >= 2 and out[-1] == ch and out[-2] == ch:
            continue
        out.append(ch)
    return "".join(out)


def _single_tok_feats(tok: str, prefix: str = "") -> dict[str, object]:
    low = fold_latin_accents(tok).casefold()
    scr = _token_script(tok)
    feats: dict[str, object] = {
        f"{prefix}scr": scr,
        f"{prefix}shp": _word_shape(tok),
        f"{prefix}title": tok[0].isupper() if tok else False,
        f"{prefix}upper": tok.isupper() and any(c.isalpha() for c in tok),
        f"{prefix}digit": any(c.isdigit() for c in tok),
        f"{prefix}len_bin": min(len(tok), 12) // 3,
        f"{prefix}is_dr": low in _DOCTOR_TRIGS,
        f"{prefix}is_pt_trig": low in _PATIENT_TRIGS,
        f"{prefix}is_rel_trig": low in _REL_TRIGS,
        f"{prefix}is_hon": low in _HONORIFIC_RIGHT,
        f"{prefix}is_clin": low in _CLINICAL_RIGHT,
        f"{prefix}is_stop": low in _CRF_STOPWORDS,
    }
    if not prefix:
        feats["pref2"] = low[:2]
        feats["pref3"] = low[:3]
        feats["suf2"] = low[-2:]
        feats["suf3"] = low[-3:]
        # Character 3-grams (morphology without memorizing full names)
        padded = f"^{low}$"
        for k in range(max(0, len(padded) - 2)):
            if k < 8:
                feats[f"c3:{padded[k:k+3]}"] = True
        if len(low) <= 12 and (low in _DOCTOR_TRIGS | _PATIENT_TRIGS | _REL_TRIGS | _HONORIFIC_RIGHT | _CLINICAL_RIGHT | _CRF_STOPWORDS or scr == "PUNCT"):
            feats["cue"] = low
    else:
        if len(low) <= 14:
            feats[f"{prefix}cue"] = low if (scr == "PUNCT" or low.isalpha()) else _word_shape(tok)
    return feats


def sent_to_features(sent_tokens: list[tuple[str, int, int]]) -> list[dict[str, object]]:
    """Convert one sentence of (tok, start, end) into CRF feature dicts."""
    n = len(sent_tokens)
    all_feats: list[dict[str, object]] = []
    for i in range(n):
        tok = sent_tokens[i][0]
        f: dict[str, object] = {"bias": 1.0}
        f.update(_single_tok_feats(tok, prefix=""))
        if i == 0:
            f["BOS"] = True
        if i == n - 1:
            f["EOS"] = True
        for offset in (-2, -1, 1, 2):
            j = i + offset
            if 0 <= j < n:
                f.update(_single_tok_feats(sent_tokens[j][0], prefix=f"{offset:+d}:"))
        # Multi-token C/O complaint vs person context cue
        if i >= 1 and sent_tokens[i - 1][0].casefold() == "c/o":
            after_low = {sent_tokens[k][0].casefold() for k in range(i, min(n, i + 4))}
            f["co_has_clin_right"] = bool(after_low & _CLINICAL_RIGHT)
        all_feats.append(f)
    return all_feats


class CRFPredictor:
    """Thread-safe wrapper around the trained python-crfsuite tagger."""

    def __init__(self, model_path: Path = MODEL_PATH):
        self.model_path = model_path
        self._tagger: pycrfsuite.Tagger | None = None
        if model_path.is_file():
            tagger = pycrfsuite.Tagger()
            tagger.open(str(model_path))
            self._tagger = tagger

    @property
    def available(self) -> bool:
        return self._tagger is not None

    def predict_spans(self, text: str) -> list[Span]:
        if not text or self._tagger is None:
            return []
        nv = build_normalized_view(text)
        clean = nv.clean
        sentences = tokenize_with_spans(clean)
        spans: list[Span] = []

        for sent in sentences:
            feats = sent_to_features(sent)
            tags = self._tagger.tag(feats)
            cur_label: str | None = None
            cur_start = -1
            cur_end = -1
            for (tok, ts, te), tag in zip(sent, tags):
                if tag.startswith("B-"):
                    if cur_label is not None:
                        self._emit(text, clean, nv, cur_start, cur_end, cur_label, spans)
                    cur_label = tag[2:]
                    cur_start, cur_end = ts, te
                elif tag.startswith("I-") and cur_label == tag[2:]:
                    cur_end = te
                else:
                    if cur_label is not None:
                        self._emit(text, clean, nv, cur_start, cur_end, cur_label, spans)
                        cur_label = None
            if cur_label is not None:
                self._emit(text, clean, nv, cur_start, cur_end, cur_label, spans)

        return spans

    @staticmethod
    def _emit(
        orig_text: str,
        clean: str,
        nv,
        ns: int,
        ne: int,
        label: str,
        out: list[Span],
    ) -> None:
        seg = clean[ns:ne].strip(" .,:;-'\"")
        if not seg or len(seg) < 3:
            return
        toks = [t.strip(".,") for t in seg.split() if t.strip(".,")]
        if not toks:
            return
        if any(ch.isdigit() for ch in seg):
            # Never emit digit-containing CRF spans unless preceded by an explicit person-slot cue
            # and strictly matching safe synthetic-canary morphology (never drugs/labs/wards/doses).
            before_ctx = clean[max(0, ns - 40) : ns]
            if not re.search(
                r"(?i)(?:name|patient|pt\.?|beneficiary|client|participant|attendant|caretaker|guardian|"
                r"referral|consent\s+of|signed\s+by|naam|नाम|रेफरल|संदर्भ|अभिभावक)\b",
                before_ctx,
            ):
                return
            alpha_toks = [t for t in toks if not any(ch.isdigit() for ch in t)]
            digit_toks = [t for t in toks if any(ch.isdigit() for ch in t)]
            if len(toks) >= 2 and not alpha_toks:
                return
            for dt in digit_toks:
                folded_dt = fold_latin_accents(dt)
                if not _SYNTH_CANARY_TOK_RE.match(folded_dt):
                    return
                base = re.sub(r"\d+", "", folded_dt).casefold()
                if base in _CRF_STOPWORDS or base in _CLINICAL_RIGHT:
                    return

        if all(re.sub(r"\d+", "", fold_latin_accents(t)).casefold() in _CRF_STOPWORDS for t in toks):
            return
        # Require at least one capitalized Latin (ASCII or accented) or Devanagari token
        if not any(is_upper_or_deva_start(t) for t in toks):
            return
        os_, oe_ = nv.map_clean_span(ns, ne)
        if oe_ > os_:
            out.append(Span(os_, oe_, label, orig_text[os_:oe_], source="crf"))

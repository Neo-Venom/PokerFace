"""Per-patient format-preserving secret date shifter (preserves intervals & LOS)."""
from __future__ import annotations

import datetime as dt
import regex as re

from src.nlp.indic_norm import shadow_normalize
from src.nlp.medical_guard import MedicalGuard

_MONTH_NAMES = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}
_MONTH_FULL = [
    "", "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]
_MONTH_ABBR = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

_ISO_DT_RE = re.compile(r"^((?:19|20)\d{2})-(\d{2})-(\d{2})(T.+)$")
_ISO_D_RE = re.compile(r"^((?:19|20)\d{2})-(\d{2})-(\d{2})$")
_NUM_D_RE = re.compile(r"^(\d{1,2})([./-])(\d{1,2})\2((?:19|20)\d{2})$")
_COMPACT_DT_RE = re.compile(r"^((?:19|20)\d{2})(\d{2})(\d{2})(.*)$")

_TEXT_DATE_RE = re.compile(
    r"\b((?:19|20)\d{2})-(\d{2})-(\d{2})(?=(?:T|[ ,;:\n]|$))"
    r"|\b(\d{1,2})([./-])(\d{1,2})\5((?:19|20)\d{2})\b"
    r"|\b(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.?(?:\s*,?\s*((?:19|20)\d{2}))?\b"
    r"|\b(January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.?\s+(\d{1,2})(?:\s*,?\s*((?:19|20)\d{2}))?\b"
    r"|(?<=\[)(\d{1,2})/(\d{1,2})/(\d{2})(?=,)",
    re.IGNORECASE,
)


def parse_any_date(raw: str, dmy: bool = True, anchor_year: int = 2026) -> dt.date | None:
    s = shadow_normalize(raw).strip()
    m = _ISO_DT_RE.match(s) or _ISO_D_RE.match(s)
    if m:
        try:
            return dt.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            return None
    m = _NUM_D_RE.match(s)
    if m:
        a, b, y = int(m.group(1)), int(m.group(3)), int(m.group(4))
        day, mon = (a, b) if dmy else (b, a)
        if not (1 <= mon <= 12):
            day, mon = mon, day
        try:
            return dt.date(y, mon, day)
        except ValueError:
            return None
    return None


def _shift_d(d: dt.date, delta_days: int) -> dt.date:
    return d + dt.timedelta(days=delta_days)


def shift_date_value(raw: str, delta_days: int, dmy: bool = True, anchor_year: int = 2026) -> str:
    """Shift a structured date/datetime string while preserving its exact original representation."""
    if not raw or not isinstance(raw, str):
        return raw
    norm = shadow_normalize(raw).strip()
    m = _ISO_DT_RE.match(norm)
    if m:
        try:
            d = dt.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            nd = _shift_d(d, delta_days)
            return f"{nd.strftime('%Y-%m-%d')}{m.group(4)}"
        except ValueError:
            return raw

    m = _ISO_D_RE.match(norm)
    if m:
        try:
            d = dt.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            return _shift_d(d, delta_days).strftime("%Y-%m-%d")
        except ValueError:
            return raw

    m = _NUM_D_RE.match(norm)
    if m:
        a_s, sep, b_s, y_s = m.group(1), m.group(2), m.group(3), m.group(4)
        a, b, y = int(a_s), int(b_s), int(y_s)
        used_dmy = dmy
        day, mon = (a, b) if used_dmy else (b, a)
        if not (1 <= mon <= 12) and (1 <= day <= 12):
            used_dmy = not used_dmy
            day, mon = (a, b) if used_dmy else (b, a)
        try:
            nd = _shift_d(dt.date(y, mon, day), delta_days)
        except ValueError:
            return raw
        pad = len(a_s) == 2 or len(b_s) == 2
        d_out = f"{nd.day:02d}" if pad else str(nd.day)
        m_out = f"{nd.month:02d}" if pad else str(nd.month)
        return f"{d_out}{sep}{m_out}{sep}{nd.year:04d}" if used_dmy else f"{m_out}{sep}{d_out}{sep}{nd.year:04d}"

    return shift_dates_in_text(raw, delta_days=delta_days, dmy=dmy, anchor_year=anchor_year)


def _format_named_month(nd: dt.date, orig_mon: str, day_first: bool, has_year: bool) -> str:
    mon_out = _MONTH_FULL[nd.month] if len(orig_mon.rstrip(".")) > 4 else _MONTH_ABBR[nd.month]
    if day_first:
        return f"{nd.day} {mon_out} {nd.year}" if has_year else f"{nd.day} {mon_out}"
    return f"{mon_out} {nd.day}, {nd.year}" if has_year else f"{mon_out} {nd.day}"


def shift_dates_in_text(
    text: str,
    delta_days: int,
    dmy: bool = True,
    anchor_year: int = 2026,
    guard: MedicalGuard | None = None,
) -> str:
    """Shift free-text dates (ISO, numeric DMY/MDY, named month, year-less, WhatsApp)."""
    if not text or not any(ch.isdigit() for ch in shadow_normalize(text)):
        return text
    norm = shadow_normalize(text)
    prot = guard.protected_spans(norm) if guard else []

    # Find anchor year in text if present (for year-less dates like "Visit: 5 September")
    ym = re.search(r"\b(202[0-9])\b", norm)
    eff_anchor = int(ym.group(1)) if ym else anchor_year

    parts: list[str] = []
    pos = 0
    for m in _TEXT_DATE_RE.finditer(norm):
        ms, me = m.start(), m.end()
        if any(ps < me and pe > ms for ps, pe, _tag in prot):
            continue
        # Never corrupt already-redacted placeholders or clinical tokens
        parts.append(text[pos:ms])
        orig = text[ms:me]
        repl = orig
        try:
            if m.group(1):
                d = dt.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
                repl = _shift_d(d, delta_days).strftime("%Y-%m-%d")
            elif m.group(4):
                a_s, sep, b_s, y_s = m.group(4), m.group(5), m.group(6), m.group(7)
                a, b, y = int(a_s), int(b_s), int(y_s)
                if 1900 <= y <= 2035:
                    used_dmy = dmy
                    day, mon = (a, b) if used_dmy else (b, a)
                    if not (1 <= mon <= 12) and (1 <= day <= 12):
                        used_dmy = not used_dmy
                        day, mon = (a, b) if used_dmy else (b, a)
                    nd = _shift_d(dt.date(y, mon, day), delta_days)
                    pad = len(a_s) == 2 or len(b_s) == 2
                    d_out = f"{nd.day:02d}" if pad else str(nd.day)
                    m_out = f"{nd.month:02d}" if pad else str(nd.month)
                    repl = f"{d_out}{sep}{m_out}{sep}{nd.year:04d}" if used_dmy else f"{m_out}{sep}{d_out}{sep}{nd.year:04d}"
            elif m.group(8):
                day = int(m.group(8))
                mon_tok = m.group(9)
                mon = _MONTH_NAMES.get(mon_tok.lower(), 0)
                has_year = bool(m.group(10))
                yr = int(m.group(10)) if has_year else eff_anchor
                if mon and 1 <= day <= 31:
                    nd = _shift_d(dt.date(yr, mon, day), delta_days)
                    repl = _format_named_month(nd, mon_tok, day_first=True, has_year=has_year)
            elif m.group(11):
                mon_tok = m.group(11)
                mon = _MONTH_NAMES.get(mon_tok.lower(), 0)
                day = int(m.group(12))
                has_year = bool(m.group(13))
                yr = int(m.group(13)) if has_year else eff_anchor
                if mon and 1 <= day <= 31:
                    nd = _shift_d(dt.date(yr, mon, day), delta_days)
                    repl = _format_named_month(nd, mon_tok, day_first=False, has_year=has_year)
            elif m.group(14):
                # WhatsApp [DD/MM/YY, HH:MM]
                a, b, yy = int(m.group(14)), int(m.group(15)), int(m.group(16))
                yr = 2000 + yy if yy <= 35 else 1900 + yy
                day, mon = (a, b) if dmy else (b, a)
                if not (1 <= mon <= 12) and (1 <= day <= 12):
                    day, mon = mon, day
                nd = _shift_d(dt.date(yr, mon, day), delta_days)
                repl = f"{nd.day:02d}/{nd.month:02d}/{nd.year % 100:02d}"
        except ValueError:
            repl = orig
        parts.append(repl)
        pos = me
    parts.append(text[pos:])
    return "".join(parts)


def shift_compact_hl7_date(val: str, delta_days: int) -> str:
    """Shift HL7 compact timestamps like 20260415 or 20260415103000+0530 if valid."""
    m = _COMPACT_DT_RE.match(val.strip())
    if not m:
        return val
    try:
        d = dt.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        if 1990 <= d.year <= 2035:
            return _shift_d(d, delta_days).strftime("%Y%m%d") + m.group(4)
    except ValueError:
        pass
    return val

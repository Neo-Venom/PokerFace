"""M3 Requirement 1: Deterministic secret-keyed per-patient / per-record date shifting & weekday handling."""
from __future__ import annotations

import datetime as dt
import hashlib
import hmac
import re
import unicodedata

_DEVA_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")

_ISO_DT_RE = re.compile(
    r"\b(19\d{2}|20\d{2})-(\d{2})-(\d{2})(?:(T\d{2}:\d{2}:\d{2}(?:\.\d+)?)(Z|[+\-]\d{2}:\d{2})?)?\b"
)
_DMY_DATE_RE = re.compile(r"(?<![A-Za-z0-9./\-])(\d{1,2})([./\-])(\d{1,2})\2(19\d{2}|20\d{2})(?!\d)")
_WA_SHORT_DATE_RE = re.compile(r"\[(\d{1,2})/(\d{1,2})/(\d{2})(?=,\s*\d{1,2}:\d{2})")
_MONTH_MAP = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12,
}
_EN_MONTHS_FULL = ("January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December")
_EN_MONTHS_SHORT = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
_DMY_MON_RE = re.compile(
    r"\b(\d{1,2})([ \t\-]+)(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\.?([ \t\-,]+)(19\d{2}|20\d{2})\b",
    re.I,
)
_MON_DY_RE = re.compile(
    r"\b(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\.?([ \t]+)(\d{1,2})(,[ \t]+|[ \t]+)(19\d{2}|20\d{2})\b",
    re.I,
)

_EN_WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
_EN_WEEKDAYS_SHORT = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
_HI_WEEKDAYS = ("सोमवार", "मंगलवार", "बुधवार", "गुरुवार", "शुक्रवार", "शनिवार", "रविवार")

_WEEKDAY_FULL_RE = re.compile(
    r"(?i)\b(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday|Mon|Tue|Wed|Thu|Fri|Sat|Sun)\b"
    r"|(?:सोमवार|मंगलवार|बुधवार|गुरुवार|शुक्रवार|शनिवार|रविवार)"
)
_WEEKDAY_ADJACENT_BEFORE_RE = re.compile(
    r"(?i)((?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday|Mon|Tue|Wed|Thu|Fri|Sat|Sun"
    r"|सोमवार|मंगलवार|बुधवार|गुरुवार|शुक्रवार|शनिवार|रविवार))(\s*[,(\-]?\s*)$"
)
_WEEKDAY_ADJACENT_AFTER_RE = re.compile(
    r"(?i)^(\s*[,(\-]?\s*)(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday|Mon|Tue|Wed|Thu|Fri|Sat|Sun"
    r"|सोमवार|मंगलवार|बुधवार|गुरुवार|शुक्रवार|शनिवार|रविवार)(\s*\)?)"
)


def keyed_shift_days(
    key: str,
    shift_key: bytes,
    max_days: int = 180,
    min_days: int = 15,
) -> int:
    """Derive deterministic signed shift in [-max_days, -min_days] U [+min_days, +max_days]."""
    hi = max(min_days + 1, int(max_days))
    lo = max(1, min(min_days, hi - 1))
    span = hi - lo + 1  # number of magnitudes per sign
    digest = hmac.new(shift_key, b"m3-shift|" + key.encode("utf-8"), hashlib.sha256).digest()
    raw = int.from_bytes(digest[:8], "big")
    sign = -1 if (raw & 1) else 1
    mag = lo + ((raw >> 1) % span)
    return sign * mag


def infer_month_first(samples: list[str], header_or_rel_hint: str = "") -> bool:
    """Infer MM/DD/YYYY vs DD/MM/YYYY dynamically from unambiguous dates in column/doc, then hints."""
    hl = header_or_rel_hint.casefold()
    if "mm/dd" in hl or "mm-dd" in hl:
        return True
    if "dd/mm" in hl or "dd-mm" in hl:
        return False

    p1_gt12 = 0
    p2_gt12 = 0
    cand_pairs: list[tuple[int, int, int]] = []
    for s in samples:
        ns = unicodedata.normalize("NFKC", s).translate(_DEVA_DIGITS)
        for m in _DMY_DATE_RE.finditer(ns):
            a, b, y = int(m.group(1)), int(m.group(3)), int(m.group(4))
            if b > 12 >= a >= 1:
                p2_gt12 += 1
            elif a > 12 >= b >= 1:
                p1_gt12 += 1
            cand_pairs.append((a, b, y))
    if p2_gt12 > p1_gt12:
        return True
    if p1_gt12 > p2_gt12:
        return False
    # Ambiguous <=12 pairs: check if doc looks like an admission+discharge pair (short stay favors one order)
    if len(cand_pairs) >= 2:
        # Compare pair spreads; if admission->discharge in MDY gives short stay (0..60d) while DMY gives >=90d
        dmy_dates, mdy_dates = [], []
        for a, b, y in cand_pairs:
            try:
                if 1 <= b <= 12 and 1 <= a <= 31:
                    dmy_dates.append(dt.date(y, b, a))
            except ValueError:
                pass
            try:
                if 1 <= a <= 12 and 1 <= b <= 31:
                    mdy_dates.append(dt.date(y, a, b))
            except ValueError:
                pass
        if len(dmy_dates) >= 2 and len(mdy_dates) >= 2:
            d_span = (max(dmy_dates) - min(dmy_dates)).days
            m_span = (max(mdy_dates) - min(mdy_dates)).days
            if 0 < m_span <= 60 and d_span > 90:
                return True
            if 0 < d_span <= 60 and m_span > 90:
                return False
    return False


def _format_weekday_like(sample_wd: str, d: dt.date) -> str:
    idx = d.weekday()
    if any("\u0900" <= ch <= "\u097F" for ch in sample_wd):
        return _HI_WEEKDAYS[idx]
    if len(sample_wd) <= 3:
        res = _EN_WEEKDAYS_SHORT[idx]
    else:
        res = _EN_WEEKDAYS[idx]
    return res.upper() if sample_wd.isupper() else (res.lower() if sample_wd.islower() else res)


def shift_date_str(val: str, shift_days: int, month_first: bool = False) -> tuple[str, bool]:
    """Shift a structured date or ISO timestamp string while preserving its exact format."""
    if not val or shift_days == 0:
        return val, False
    ns = unicodedata.normalize("NFKC", val).translate(_DEVA_DIGITS).strip()

    m_iso = _ISO_DT_RE.fullmatch(ns)
    if m_iso:
        try:
            d = dt.date(int(m_iso.group(1)), int(m_iso.group(2)), int(m_iso.group(3))) + dt.timedelta(days=shift_days)
            time_part = m_iso.group(4) or ""
            tz_part = m_iso.group(5) or ""
            return f"{d.isoformat()}{time_part}{tz_part}", True
        except ValueError:
            return val, False

    m_dmy = _DMY_DATE_RE.fullmatch(ns)
    if m_dmy:
        a, sep, b, y = int(m_dmy.group(1)), m_dmy.group(2), int(m_dmy.group(3)), int(m_dmy.group(4))
        d_ord = (y, b, a)
        m_ord = (y, a, b)
        choices = (m_ord, d_ord) if month_first else (d_ord, m_ord)
        for cand_idx, (yy, mm, dd) in enumerate(choices):
            try:
                shifted = dt.date(yy, mm, dd) + dt.timedelta(days=shift_days)
                used_mdy = (month_first and cand_idx == 0) or (not month_first and cand_idx == 1)
                if used_mdy:
                    return f"{shifted.month:02d}{sep}{shifted.day:02d}{sep}{shifted.year:04d}", True
                return f"{shifted.day:02d}{sep}{shifted.month:02d}{sep}{shifted.year:04d}", True
            except ValueError:
                continue

    m_dm = _DMY_MON_RE.fullmatch(ns)
    if m_dm:
        d, sep1, mon_s, sep2, y = int(m_dm.group(1)), m_dm.group(2), m_dm.group(3), m_dm.group(4), int(m_dm.group(5))
        mo = _MONTH_MAP.get(mon_s.casefold())
        if mo:
            try:
                sd = dt.date(y, mo, d) + dt.timedelta(days=shift_days)
                mname = _EN_MONTHS_SHORT[sd.month - 1] if len(mon_s) <= 4 else _EN_MONTHS_FULL[sd.month - 1]
                return f"{sd.day}{sep1}{mname}{sep2}{sd.year:04d}", True
            except ValueError:
                pass

    m_md = _MON_DY_RE.fullmatch(ns)
    if m_md:
        mon_s, sep1, d, sep2, y = m_md.group(1), m_md.group(2), int(m_md.group(3)), m_md.group(4), int(m_md.group(5))
        mo = _MONTH_MAP.get(mon_s.casefold())
        if mo:
            try:
                sd = dt.date(y, mo, d) + dt.timedelta(days=shift_days)
                mname = _EN_MONTHS_SHORT[sd.month - 1] if len(mon_s) <= 4 else _EN_MONTHS_FULL[sd.month - 1]
                return f"{mname}{sep1}{sd.day}{sep2}{sd.year:04d}", True
            except ValueError:
                pass
    return val, False


def shift_dates_and_weekdays_in_text(text: str, shift_days: int, month_first: bool = False) -> tuple[str, bool]:
    """Shift every date inside free text cleanly and update/remove adjacent weekday names."""
    if not text or shift_days == 0:
        return text, False

    changed = False
    # Collect non-overlapping date matches (start, end, shifted_str, shifted_date)
    matches: list[tuple[int, int, str, dt.date]] = []

    for m in _ISO_DT_RE.finditer(text):
        try:
            orig_d = dt.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            sd = orig_d + dt.timedelta(days=shift_days)
            rep = f"{sd.isoformat()}{m.group(4) or ''}{m.group(5) or ''}"
            matches.append((m.start(), m.end(), rep, sd))
        except ValueError:
            continue

    for m in _DMY_DATE_RE.finditer(text):
        s, e = m.start(), m.end()
        if any(not (e <= ms or s >= me) for ms, me, _, _ in matches):
            continue
        ns = m.group(0).translate(_DEVA_DIGITS)
        m2 = _DMY_DATE_RE.fullmatch(ns)
        if not m2:
            continue
        a, sep, b, y = int(m2.group(1)), m2.group(2), int(m2.group(3)), int(m2.group(4))
        choices = ((y, a, b), (y, b, a)) if month_first else ((y, b, a), (y, a, b))
        for cand_idx, (yy, mm, dd) in enumerate(choices):
            try:
                orig_d = dt.date(yy, mm, dd)
                sd = orig_d + dt.timedelta(days=shift_days)
                used_mdy = (month_first and cand_idx == 0) or (not month_first and cand_idx == 1)
                rep = (
                    f"{sd.month:02d}{sep}{sd.day:02d}{sep}{sd.year:04d}"
                    if used_mdy
                    else f"{sd.day:02d}{sep}{sd.month:02d}{sep}{sd.year:04d}"
                )
                matches.append((s, e, rep, sd))
                break
            except ValueError:
                continue

    for m in _WA_SHORT_DATE_RE.finditer(text):
        s, e = m.start() + 1, m.end()  # preserve leading '['
        if any(not (e <= ms or s >= me) for ms, me, _, _ in matches):
            continue
        d, mo, y2 = int(m.group(1)), int(m.group(2)), int(m.group(3))
        try:
            orig_d = dt.date(2000 + y2, mo, d)
            sd = orig_d + dt.timedelta(days=shift_days)
            rep = f"{sd.day:02d}/{sd.month:02d}/{sd.year % 100:02d}"
            matches.append((s, e, rep, sd))
        except ValueError:
            continue

    for m in _DMY_MON_RE.finditer(text):
        s, e = m.start(), m.end()
        if any(not (e <= ms or s >= me) for ms, me, _, _ in matches):
            continue
        d, sep1, mon_s, sep2, y = int(m.group(1)), m.group(2), m.group(3), m.group(4), int(m.group(5))
        mo = _MONTH_MAP.get(mon_s.casefold())
        if mo:
            try:
                sd = dt.date(y, mo, d) + dt.timedelta(days=shift_days)
                mname = _EN_MONTHS_SHORT[sd.month - 1] if len(mon_s) <= 4 else _EN_MONTHS_FULL[sd.month - 1]
                matches.append((s, e, f"{sd.day}{sep1}{mname}{sep2}{sd.year:04d}", sd))
            except ValueError:
                continue

    for m in _MON_DY_RE.finditer(text):
        s, e = m.start(), m.end()
        if any(not (e <= ms or s >= me) for ms, me, _, _ in matches):
            continue
        mon_s, sep1, d, sep2, y = m.group(1), m.group(2), int(m.group(3)), m.group(4), int(m.group(5))
        mo = _MONTH_MAP.get(mon_s.casefold())
        if mo:
            try:
                sd = dt.date(y, mo, d) + dt.timedelta(days=shift_days)
                mname = _EN_MONTHS_SHORT[sd.month - 1] if len(mon_s) <= 4 else _EN_MONTHS_FULL[sd.month - 1]
                matches.append((s, e, f"{mname}{sep1}{sd.day}{sep2}{sd.year:04d}", sd))
            except ValueError:
                continue

    if not matches:
        return text, False

    matches.sort(key=lambda x: x[0])
    out_parts: list[str] = []
    cur = 0
    for s, e, rep, sd in matches:
        if s < cur:
            continue
        prefix = text[cur:s]
        # Recompute weekday immediately preceding the date (e.g., "Monday, 12/05/2026")
        mb = _WEEKDAY_ADJACENT_BEFORE_RE.search(prefix)
        if mb:
            new_wd = _format_weekday_like(mb.group(1), sd)
            prefix = prefix[: mb.start(1)] + new_wd + prefix[mb.end(1) :]
        out_parts.append(prefix)
        out_parts.append(rep)
        cur = e
        changed = True
        # Also check weekday immediately following the date (e.g., "12/05/2026 (Monday)")
        tail = text[cur : min(len(text), cur + 24)]
        ma = _WEEKDAY_ADJACENT_AFTER_RE.match(tail)
        if ma:
            new_wd = _format_weekday_like(ma.group(2), sd)
            out_parts.append(ma.group(1) + new_wd + ma.group(3))
            cur += ma.end()

    out_parts.append(text[cur:])
    return "".join(out_parts), changed

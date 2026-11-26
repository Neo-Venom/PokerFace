"""M3 Requirements 1-3 & 5: Truthful age/location/date transformations across CSV/JSON/FHIR/TXT."""
from __future__ import annotations

import copy
import datetime as dt
import re
from typing import Any

from raksha.detect.structured import _CLINICAL_DATE_KEYS, _is_fhir_bundle, _norm_key
from raksha.export.filenames import sanitize_base64_string
from raksha.privacy.dates import (
    infer_month_first,
    keyed_shift_days,
    shift_date_str,
    shift_dates_and_weekdays_in_text,
)
from raksha.privacy.risk import PolicySettings, PrivacyPlan

_AGE_LABELED_RE = re.compile(
    r"(?i)(\b(?:Age(?:/Sex)?|आयु)\s*[:\-]\s*)(\d{1,3})(\s*(?:Yrs?|वर्ष|Y))?(?:\s*/\s*([MF]|पुरुष|महिला))?"
)
_AGE_COMPACT_RE = re.compile(
    r"\b(\d{1,3})(\s*)(?:(Yrs?\s*/\s*[MF])|(/\s*[MF]\b)|(Y\s*/\s*[MF]\b)|(वर्ष))"
)
_AGE_YRS_TEXT_RE = re.compile(r"(?i)(?<!x\s)(?<!back\s)\b(\d{1,3})(\s+yrs?\b)(?!\s*(?:back|ago))")
_TEHSIL_TXT_RE = re.compile(
    r"(?i)(\bTehsil\.?\s+|तहसील\s*[:\-]?\s*)([A-Za-z\u0900-\u097F]+(?:\s+[A-Za-z\u0900-\u097F]+){0,2})(?=\s*[,\.\-\n\r]|$)"
)
_DISTRICT_TXT_RE = re.compile(
    r"(?i)(\bDistt?\.?\s+|\bDistrict\s*[:\-]?\s*|जिला\s*[:\-]?\s*)([A-Za-z\u0900-\u097F]+(?:\s+[A-Za-z\u0900-\u097F]+){0,2})(?=\s*[-,\.\n\r]|$)"
)
_SEX_LINE_RE = re.compile(
    r"(?i)(\b(?:Sex|Gender|लिंग)\s*[:\-]\s*)(Female|Male|F|M|Other|महिला|पुरुष|अन्य)\b"
)


def format_truthful_age_band(age: int, width: int, top_cap: int | None = None) -> str:
    """Return a truthful interval guaranteed to contain `age` (or [AGE] if width == 0)."""
    if width <= 0:
        return "[AGE]"
    if top_cap is not None and age >= top_cap:
        return f"{top_cap}+"
    lo = (max(0, int(age)) // width) * width
    hi = lo + width - 1
    return f"{lo}-{hi}"


def _band_sex_suffix(sex_raw: str | None, keep_gender: bool) -> str:
    if not sex_raw:
        return ""
    return sex_raw if keep_gender else "[GENDER]"


def _generalize_age_in_text(text: str, plan: PrivacyPlan) -> tuple[str, bool]:
    changed = False

    def _repl_labeled(m: re.Match) -> str:
        nonlocal changed
        a = int(m.group(2))
        if not (0 <= a <= 115):
            return m.group(0)
        changed = True
        band = format_truthful_age_band(a, plan.age_band_width, plan.top_age_cap)
        unit = m.group(3) or ""
        if m.group(4):
            sx = _band_sex_suffix(m.group(4), plan.keep_gender)
            return f"{m.group(1)}{band}{unit}/{sx}"
        return f"{m.group(1)}{band}{unit}"

    out = _AGE_LABELED_RE.sub(_repl_labeled, text)

    def _repl_compact(m: re.Match) -> str:
        nonlocal changed
        a = int(m.group(1))
        if not (0 <= a <= 115):
            return m.group(0)
        # Skip if preceded right after band replace like "50-54 Yrs/F"
        if m.start() > 0 and out[m.start() - 1] == "-":
            return m.group(0)
        changed = True
        band = format_truthful_age_band(a, plan.age_band_width, plan.top_age_cap)
        suffix = m.group(3) or m.group(4) or m.group(5) or m.group(6) or ""
        if not plan.keep_gender:
            suffix = re.sub(r"([MF]|पुरुष|महिला)$", "[GENDER]", suffix)
        return f"{band}{m.group(2)}{suffix}"

    out = _AGE_COMPACT_RE.sub(_repl_compact, out)

    def _repl_yrs(m: re.Match) -> str:
        nonlocal changed
        a = int(m.group(1))
        if not (0 <= a <= 115):
            return m.group(0)
        if m.start() > 0 and out[m.start() - 1] == "-":
            return m.group(0)
        before = out[max(0, m.start() - 14) : m.start()].casefold()
        if re.search(r"\b(?:back|ago|since|for|x|history)\s*$", before):
            return m.group(0)
        changed = True
        band = format_truthful_age_band(a, plan.age_band_width, plan.top_age_cap)
        return f"{band}{m.group(2)}"

    out = _AGE_YRS_TEXT_RE.sub(_repl_yrs, out)

    if not plan.keep_gender:
        def _repl_sex(m: re.Match) -> str:
            nonlocal changed
            changed = True
            return f"{m.group(1)}[GENDER]"

        out = _SEX_LINE_RE.sub(_repl_sex, out)
    return out, changed


def generalize_text(
    text: str,
    plan: PrivacyPlan,
    *,
    month_first: bool = False,
) -> tuple[str, bool]:
    """Apply keyed date shift + weekday update, truthful age band, and location escalation to free text."""
    if not text:
        return text, False
    out = text
    changed = False

    out, c_dt = shift_dates_and_weekdays_in_text(out, plan.shift_days, month_first=month_first)
    changed = changed or c_dt

    out, c_age = _generalize_age_in_text(out, plan)
    changed = changed or c_age

    if not plan.keep_tehsil:
        out_t = _TEHSIL_TXT_RE.sub(r"\1[TEHSIL]", out)
        if out_t != out:
            changed = True
            out = out_t

    if not plan.keep_district:
        out_d = _DISTRICT_TXT_RE.sub(r"\1[DISTRICT]", out)
        if out_d != out:
            changed = True
            out = out_d

    return out, changed


def apply_privacy_to_tabular(
    orig_header: list[str],
    orig_body: list[list[str]],
    clean_body: list[list[str]],
    rel: str,
    plans_by_row: dict[int, PrivacyPlan],
) -> tuple[list[list[str]], dict[int, bool]]:
    """Apply M3 privacy transformations row-by-row to sanitized tabular cells."""
    n_cols = len(orig_header)
    norm_hdrs = [_norm_key(h) for h in orig_header]
    col_month_first: dict[int, bool] = {}
    for ci, h in enumerate(orig_header):
        hk = norm_hdrs[ci]
        if hk in _CLINICAL_DATE_KEYS or "date" in hk or "registered" in hk or "collected" in hk or "reported" in hk:
            col_month_first[ci] = infer_month_first(
                [str(r[ci]) if ci < len(r) else "" for r in orig_body],
                header_or_rel_hint=f"{h} {rel}",
            )

    new_body: list[list[str]] = []
    gen_flags: dict[int, bool] = {}

    for r_idx, clean_row in enumerate(clean_body):
        plan = plans_by_row.get(r_idx)
        if plan is None:
            new_body.append(list(clean_row))
            gen_flags[r_idx] = False
            continue

        row = list(clean_row)
        row_changed = False
        orig_row = orig_body[r_idx] if r_idx < len(orig_body) else []

        for ci, cell in enumerate(row):
            if ci >= n_cols or not str(cell).strip():
                continue
            hk = norm_hdrs[ci]
            orig_val = str(orig_row[ci]).strip() if ci < len(orig_row) else str(cell).strip()
            sval = str(cell)

            if hk in ("age", "umar", "age_yrs", "age_val"):
                m_int = re.match(r"^\s*(\d{1,3})(\s*(?:Yrs?|वर्ष|Y))?\s*$", orig_val, re.I)
                if m_int:
                    a = int(m_int.group(1))
                    if 0 <= a <= 115:
                        band = format_truthful_age_band(a, plan.age_band_width, plan.top_age_cap)
                        unit = m_int.group(2) or ""
                        row[ci] = f"{band}{unit}"
                        row_changed = True
            elif hk == "age_sex":
                m_as = re.match(r"^\s*(\d{1,3})\s*/\s*([MF])\s*$", orig_val, re.I)
                if m_as:
                    a = int(m_as.group(1))
                    band = format_truthful_age_band(a, plan.age_band_width, plan.top_age_cap)
                    sx = m_as.group(2).upper() if plan.keep_gender else "[GENDER]"
                    row[ci] = f"{band}/{sx}"
                    row_changed = True
            elif hk in ("gender", "sex", "ling"):
                if not plan.keep_gender:
                    row[ci] = "[GENDER]"
                    row_changed = True
            elif hk == "tehsil":
                if not plan.keep_tehsil and sval != "[TEHSIL]":
                    row[ci] = "[TEHSIL]"
                    row_changed = True
            elif hk in ("district", "jila", "distt"):
                if not plan.keep_district and sval != "[DISTRICT]":
                    row[ci] = "[DISTRICT]"
                    row_changed = True
            elif ci in col_month_first:
                shifted, c_dt = shift_date_str(sval, plan.shift_days, month_first=col_month_first[ci])
                if c_dt:
                    row[ci] = shifted
                    row_changed = True
            elif hk in ("notes", "remarks", "comment", "address", "pata", "street") or "[HOUSE]" in sval or "[VILLAGE]" in sval:
                g_txt, c_t = generalize_text(sval, plan, month_first=infer_month_first([sval], header_or_rel_hint=rel))
                if c_t:
                    row[ci] = g_txt
                    row_changed = True

        new_body.append(row)
        gen_flags[r_idx] = row_changed

    return new_body, gen_flags


def _apply_privacy_dict_obj(
    obj: Any,
    plan: PrivacyPlan,
    *,
    month_first: bool = False,
) -> tuple[Any, bool]:
    """Iteratively apply M3 privacy transforms to a sanitized JSON record dict."""
    if not isinstance(obj, (dict, list)):
        return obj, False

    changed = False
    stack: list[tuple[Any, Any, str]] = []
    if isinstance(obj, dict):
        for k in list(obj.keys()):
            stack.append((obj, k, str(k)))
    else:
        for i in range(len(obj)):
            stack.append((obj, i, ""))

    while stack:
        container, slot, k_str = stack.pop()
        val = container[slot]
        kl = _norm_key(k_str)

        if isinstance(val, dict):
            if kl == "_export_meta":
                continue
            for ck in list(val.keys()):
                stack.append((val, ck, str(ck)))
            continue
        if isinstance(val, list):
            for i in range(len(val)):
                stack.append((val, i, k_str))
            continue

        if kl in ("age", "umar", "age_yrs"):
            a: int | None = None
            unit = ""
            if isinstance(val, int) and not isinstance(val, bool) and 0 <= val <= 115:
                a = val
            elif isinstance(val, str):
                m_a = re.match(r"^\s*(\d{1,3})(\s*(?:Yrs?|वर्ष|Y))?\s*$", val, re.I)
                if m_a and 0 <= int(m_a.group(1)) <= 115:
                    a = int(m_a.group(1))
                    unit = m_a.group(2) or ""
            if a is not None:
                container[slot] = f"{format_truthful_age_band(a, plan.age_band_width, plan.top_age_cap)}{unit}"
                changed = True
        elif kl == "age_sex" and isinstance(val, str):
            m_as = re.match(r"^\s*(\d{1,3})\s*/\s*([MF])\s*$", val, re.I)
            if m_as:
                band = format_truthful_age_band(int(m_as.group(1)), plan.age_band_width, plan.top_age_cap)
                sx = m_as.group(2).upper() if plan.keep_gender else "[GENDER]"
                container[slot] = f"{band}/{sx}"
                changed = True
        elif kl in ("gender", "sex", "ling") and isinstance(val, str):
            if not plan.keep_gender and val.strip():
                container[slot] = "[GENDER]"
                changed = True
        elif kl == "tehsil" and isinstance(val, str):
            if not plan.keep_tehsil and val.strip() and val != "[TEHSIL]":
                container[slot] = "[TEHSIL]"
                changed = True
        elif kl in ("district", "jila", "distt") and isinstance(val, str):
            if not plan.keep_district and val.strip() and val != "[DISTRICT]":
                container[slot] = "[DISTRICT]"
                changed = True
        elif kl == "dob" and isinstance(val, str) and val != "[DOB]":
            container[slot] = "[DOB]"
            changed = True
        elif isinstance(val, str) and (
            kl in _CLINICAL_DATE_KEYS
            or "date" in kl
            or kl in ("collected", "reported_on", "registered_on", "exported_at", "generated", "generated_at")
        ):
            shifted, c_dt = shift_date_str(val, plan.shift_days, month_first=month_first)
            if c_dt:
                container[slot] = shifted
                changed = True
        elif isinstance(val, str):
            # Decode & privacy-transform base64 string attachment if valid b64, else check inline text QIs
            b64_transformed = sanitize_base64_string(val, lambda txt: generalize_text(txt, plan, month_first=month_first)[0])
            if b64_transformed != val:
                container[slot] = b64_transformed
                changed = True
            elif kl in ("notes", "note", "remarks", "line", "address_note", "complaints", "history") or "address" in kl:
                g_txt, c_t = generalize_text(val, plan, month_first=month_first)
                if c_t:
                    container[slot] = g_txt
                    changed = True

    return obj, changed


def _apply_privacy_fhir_bundle(bundle: dict, plan: PrivacyPlan) -> tuple[dict, bool]:
    """Shift all dates and generalize patient QIs inside a FHIR R4 Bundle while preserving valid R4 types."""
    changed = False

    if isinstance(bundle.get("meta"), dict) and isinstance(bundle["meta"].get("lastUpdated"), str):
        sd, c = shift_date_str(bundle["meta"]["lastUpdated"], plan.shift_days)
        if c:
            bundle["meta"]["lastUpdated"] = sd
            changed = True
    if isinstance(bundle.get("timestamp"), str):
        sd, c = shift_date_str(bundle["timestamp"], plan.shift_days)
        if c:
            bundle["timestamp"] = sd
            changed = True

    for ent in bundle.get("entry", []):
        res = ent.get("resource") if isinstance(ent, dict) else None
        if not isinstance(res, dict):
            continue
        rt = res.get("resourceType")

        if rt == "Patient":
            if isinstance(res.get("text"), dict) and isinstance(res["text"].get("div"), str):
                g_div, c_div = generalize_text(res["text"]["div"], plan, month_first=False)
                if not plan.keep_gender:
                    g_div = re.sub(r"\b(female|male)\b", "unknown", g_div, flags=re.I)
                if c_div or not plan.keep_gender:
                    res["text"]["div"] = g_div
                    changed = True
            if "birthDate" in res and not plan.keep_fhir_birthdate:
                res.pop("birthDate", None)
                changed = True
            elif isinstance(res.get("birthDate"), str) and plan.keep_fhir_birthdate:
                res["birthDate"] = res["birthDate"][:4]
            if not plan.keep_gender and isinstance(res.get("gender"), str):
                # FHIR R4 strictly requires one of: male | female | other | unknown
                res["gender"] = "unknown"
                changed = True
            if isinstance(res.get("address"), list):
                for addr in res["address"]:
                    if not isinstance(addr, dict):
                        continue
                    if not plan.keep_district and isinstance(addr.get("district"), str):
                        addr["district"] = "[DISTRICT]"
                        changed = True
                    if isinstance(addr.get("line"), list):
                        new_lines = []
                        for ln in addr["line"]:
                            g_ln, c_ln = generalize_text(str(ln), plan)
                            changed = changed or c_ln
                            new_lines.append(g_ln)
                        addr["line"] = new_lines

        elif rt == "Composition":
            if isinstance(res.get("date"), str):
                sd, c = shift_date_str(res["date"], plan.shift_days)
                if c:
                    res["date"] = sd
                    changed = True

        elif rt == "Organization":
            if not plan.keep_district and isinstance(res.get("address"), list):
                for addr in res["address"]:
                    if isinstance(addr, dict) and isinstance(addr.get("city"), str) and addr["city"].strip():
                        addr["city"] = "[DISTRICT]"
                        changed = True

        elif rt == "Encounter":
            if isinstance(res.get("period"), dict):
                for pk in ("start", "end"):
                    if isinstance(res["period"].get(pk), str):
                        sd, c = shift_date_str(res["period"][pk], plan.shift_days)
                        if c:
                            res["period"][pk] = sd
                            changed = True

        elif rt == "Condition":
            for dk in ("recordedDate", "onsetDateTime"):
                if isinstance(res.get(dk), str):
                    sd, c = shift_date_str(res[dk], plan.shift_days)
                    if c:
                        res[dk] = sd
                        changed = True

        elif rt == "Observation":
            for dk in ("effectiveDateTime", "issued"):
                if isinstance(res.get(dk), str):
                    sd, c = shift_date_str(res[dk], plan.shift_days)
                    if c:
                        res[dk] = sd
                        changed = True

        elif rt == "MedicationRequest":
            if isinstance(res.get("authoredOn"), str):
                sd, c = shift_date_str(res["authoredOn"], plan.shift_days)
                if c:
                    res["authoredOn"] = sd
                    changed = True

        elif rt == "DiagnosticReport":
            for dk in ("effectiveDateTime", "issued"):
                if isinstance(res.get(dk), str):
                    sd, c = shift_date_str(res[dk], plan.shift_days)
                    if c:
                        res[dk] = sd
                        changed = True

        elif rt == "DocumentReference":
            if isinstance(res.get("date"), str):
                sd, c = shift_date_str(res["date"], plan.shift_days)
                if c:
                    res["date"] = sd
                    changed = True
            if isinstance(res.get("content"), list):
                for item in res["content"]:
                    if isinstance(item, dict) and isinstance(item.get("attachment"), dict):
                        att = item["attachment"]
                        if isinstance(att.get("data"), str):
                            out_b64 = sanitize_base64_string(att["data"], lambda t: generalize_text(t, plan)[0])
                            if out_b64 != att["data"]:
                                att["data"] = out_b64
                                changed = True

    return bundle, changed


def apply_privacy_to_json(
    cleaned_data: Any,
    rel: str,
    units: list[tuple[str, Any]],
    plans_by_loc: dict[str, PrivacyPlan],
    doc_shift_days: int,
    *,
    month_first: bool = False,
    key_to_cleaned: dict[str, str] | None = None,
) -> tuple[Any, dict[str, bool]]:
    """Apply M3 privacy layer to sanitized JSON (handles FHIR, single-doc JSON, and multi-record JSON)."""
    loc_gen: dict[str, bool] = {}
    if _is_fhir_bundle(cleaned_data):
        plan = plans_by_loc.get("doc")
        if plan is None:
            return cleaned_data, {"doc": False}
        out, ch = _apply_privacy_fhir_bundle(cleaned_data, plan)
        return out, {"doc": ch}

    if len(units) == 1 and units[0][0] == "doc":
        plan = plans_by_loc.get("doc")
        if plan is None:
            # Shift top-level auxiliary doc timestamps (e.g. misc/patient_index.json generated date) cleanly
            aux_plan = PrivacyPlan(
                profile_key="aux",
                shift_days=doc_shift_days,
                age_band_width=5,
                top_age_cap=85,
                keep_district=False,
                keep_tehsil=False,
                keep_gender=True,
                keep_fhir_birthdate=False,
            )
            out, ch = _apply_privacy_dict_obj(cleaned_data, aux_plan, month_first=month_first)
            return out, {"doc": ch}
        out, ch = _apply_privacy_dict_obj(cleaned_data, plan, month_first=month_first)
        return out, {"doc": ch}

    # Multi-record JSON: shift top-level metadata timestamps first by doc_shift_days
    if isinstance(cleaned_data, dict):
        for mk in ("exported_at", "generated", "generated_at"):
            if isinstance(cleaned_data.get(mk), str):
                sd, _ = shift_date_str(cleaned_data[mk], doc_shift_days)
                cleaned_data[mk] = sd
        if isinstance(cleaned_data.get("_export_meta"), dict):
            em = cleaned_data["_export_meta"]
            if isinstance(em.get("generated_at"), str):
                sd, _ = shift_date_str(em["generated_at"], doc_shift_days)
                em["generated_at"] = sd

        for loc, _sub in units:
            plan = plans_by_loc.get(loc)
            if plan is None or not loc.startswith("/"):
                loc_gen[loc] = False
                continue
            parts = loc[1:].split("/", 1)
            if len(parts) == 2:
                orig_pk = parts[0].replace("~1", "/").replace("~0", "~")
                pk = key_to_cleaned.get(orig_pk, orig_pk) if key_to_cleaned else orig_pk
                idx_s = parts[1]
                if isinstance(cleaned_data.get(pk), list) and idx_s.isdigit():
                    idx = int(idx_s)
                    if 0 <= idx < len(cleaned_data[pk]):
                        rec_out, c_rec = _apply_privacy_dict_obj(cleaned_data[pk][idx], plan, month_first=month_first)
                        cleaned_data[pk][idx] = rec_out
                        loc_gen[loc] = c_rec
                        continue
            loc_gen[loc] = False

    return cleaned_data, loc_gen

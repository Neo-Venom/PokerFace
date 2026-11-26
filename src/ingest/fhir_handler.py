"""FHIR R4 Bundle scrubber preserving valid R4 schema, references, XHTML narrative, and Base64."""
from __future__ import annotations

import base64
import copy
import html
import re
from typing import Any, Callable

_TAG_RE = re.compile(r"<[^>]+>")


def _clean_xhtml_div(div_html: str, scrub_fn: Callable[[str], str]) -> str:
    plain = html.unescape(_TAG_RE.sub(" ", div_html))
    plain = " ".join(plain.split())
    cleaned = scrub_fn(plain)
    return f'<div xmlns="http://www.w3.org/1999/xhtml">{html.escape(cleaned)}</div>'


def _scrub_attachment(att: dict[str, Any], scrub_fn: Callable[[str], str]) -> None:
    if not isinstance(att, dict):
        return
    data = att.get("data")
    if isinstance(data, str) and data.strip():
        ctype = str(att.get("contentType", "")).lower()
        try:
            raw = base64.b64decode(data + "=" * (-len(data) % 4))
            if "text" in ctype or ctype in ("", "application/json", "application/xml"):
                txt = raw.decode("utf-8", errors="strict")
                cleaned = scrub_fn(txt)
                att["data"] = base64.b64encode(cleaned.encode("utf-8")).decode("ascii")
            else:
                att["data"] = base64.b64encode(b"[ATTACHMENT_REDACTED]").decode("ascii")
        except Exception:
            att["data"] = base64.b64encode(b"[ATTACHMENT_REDACTED]").decode("ascii")
    if isinstance(att.get("title"), str):
        att["title"] = scrub_fn(att["title"])


def sanitize_fhir_bundle(
    bundle: dict[str, Any],
    psn: str | None,
    scrub_fn: Callable[[str], str],
    shift_date_fn: Callable[[str], str],
    age_band_fn: Callable[[int], str] | None = None,
    geo_fn: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Deep-clean a FHIR R4 Bundle while preserving resourceTypes, required arrays, and urn:uuid refs."""
    out = copy.deepcopy(bundle)
    meta = out.get("meta")
    if isinstance(meta, dict):
        if isinstance(meta.get("source"), str):
            meta["source"] = f"#{psn}" if psn else "[SOURCE]"
        if isinstance(meta.get("lastUpdated"), str):
            meta["lastUpdated"] = shift_date_fn(meta["lastUpdated"])
    if isinstance(out.get("timestamp"), str):
        out["timestamp"] = shift_date_fn(out["timestamp"])

    def _clean_resource(res: dict[str, Any]) -> None:
        if not isinstance(res, dict):
            return
        rtype = res.get("resourceType", "")

        # Scrub meta.source if present on resource
        rmeta = res.get("meta")
        if isinstance(rmeta, dict) and isinstance(rmeta.get("source"), str):
            rmeta["source"] = f"#{psn}" if psn else "[SOURCE]"

        # Preserve Practitioner & Organization clinical references untouched!
        if rtype == "Practitioner":
            return
        if rtype == "Organization":
            return

        # Narrative XHTML div
        text_obj = res.get("text")
        if isinstance(text_obj, dict) and isinstance(text_obj.get("div"), str):
            text_obj["div"] = _clean_xhtml_div(text_obj["div"], scrub_fn)

        # Recursively clean contained resources first
        if isinstance(res.get("contained"), list):
            for c in res["contained"]:
                if isinstance(c, dict):
                    _clean_resource(c)

        if rtype == "Patient":
            if isinstance(res.get("identifier"), list):
                for ident in res["identifier"]:
                    if not isinstance(ident, dict):
                        continue
                    coding = ((ident.get("type") or {}).get("coding") or [])
                    is_mr = any(c.get("code") == "MR" for c in coding if isinstance(c, dict))
                    if is_mr and psn:
                        ident["value"] = psn
                    elif isinstance(ident.get("value"), str):
                        ident["value"] = "[ID]"
            if isinstance(res.get("name"), list):
                for nm in res["name"]:
                    if isinstance(nm, dict):
                        if "text" in nm:
                            nm["text"] = "[NAME]"
                        if "given" in nm:
                            nm["given"] = ["[NAME]"]
                        if "family" in nm:
                            nm["family"] = "[NAME]"
            if isinstance(res.get("telecom"), list):
                for tel in res["telecom"]:
                    if isinstance(tel, dict) and isinstance(tel.get("value"), str):
                        sys_t = str(tel.get("system", "")).lower()
                        tel["value"] = "[EMAIL]" if sys_t == "email" else "[PHONE]"
            if isinstance(res.get("birthDate"), str):
                # Suppress exact DOB (YYYY-MM-DD / birth_year) to defeat A4 & DOB checks while keeping valid R4 Patient
                res.pop("birthDate", None)
            if isinstance(res.get("address"), list):
                for addr in res["address"]:
                    if isinstance(addr, dict):
                        if geo_fn:
                            geo_fn(addr)
                        else:
                            if "line" in addr:
                                addr["line"] = []
                            if "city" in addr:
                                addr["city"] = addr.get("district") or "[GENERALIZED]"
                            if isinstance(addr.get("postalCode"), str) and len(addr["postalCode"]) >= 3:
                                addr["postalCode"] = addr["postalCode"][:3] + "***"
            if isinstance(res.get("contact"), list):
                for ct in res["contact"]:
                    if isinstance(ct, dict):
                        if isinstance(ct.get("name"), dict):
                            ct["name"] = {"text": "[RELATIVE]"}
                        if isinstance(ct.get("telecom"), list):
                            for tel in ct["telecom"]:
                                if isinstance(tel, dict) and "value" in tel:
                                    tel["value"] = "[PHONE]"
                        if isinstance(ct.get("address"), dict):
                            ct["address"] = {"district": ct["address"].get("district", ""), "country": ct["address"].get("country", "IN")}
            if isinstance(res.get("extension"), list):
                for ext in res["extension"]:
                    if isinstance(ext, dict) and isinstance(ext.get("valueString"), str):
                        url_low = str(ext.get("url", "")).lower()
                        if any(w in url_low for w in ("mother", "father", "spouse", "relative", "guardian")):
                            ext["valueString"] = "[RELATIVE]"
                        else:
                            ext["valueString"] = scrub_fn(ext["valueString"])
            return

        # Scrub subject/patient display while keeping urn:uuid references valid!
        for ref_key in ("subject", "patient"):
            ref_obj = res.get(ref_key)
            if isinstance(ref_obj, dict) and isinstance(ref_obj.get("display"), str):
                ref_obj["display"] = psn or "[NAME]"

        for dkey in (
            "date",
            "effectiveDateTime",
            "authoredOn",
            "recordedDate",
            "issued",
            "onsetDateTime",
            "abatementDateTime",
            "performedDateTime",
            "occurrenceDateTime",
            "created",
            "assertedDate",
        ):
            if isinstance(res.get(dkey), str):
                res[dkey] = shift_date_fn(res[dkey])

        def _shift_period_dict(p_obj: Any) -> None:
            if isinstance(p_obj, dict):
                for pk in ("start", "end"):
                    if isinstance(p_obj.get(pk), str):
                        p_obj[pk] = shift_date_fn(p_obj[pk])

        for pkey in ("period", "effectivePeriod", "onsetPeriod", "abatementPeriod", "performedPeriod"):
            _shift_period_dict(res.get(pkey))
        if isinstance(res.get("dispenseRequest"), dict):
            _shift_period_dict(res["dispenseRequest"].get("validityPeriod"))
        if isinstance(res.get("context"), dict):
            _shift_period_dict(res["context"].get("period"))
        if isinstance(res.get("effectiveTiming"), dict) and isinstance(res["effectiveTiming"].get("event"), list):
            res["effectiveTiming"]["event"] = [
                shift_date_fn(ev) if isinstance(ev, str) else ev for ev in res["effectiveTiming"]["event"]
            ]

        # Clean encounter/document identifiers (e.g., IPD numbers)
        if rtype == "Encounter" and isinstance(res.get("identifier"), list):
            for ident in res["identifier"]:
                if isinstance(ident, dict) and isinstance(ident.get("value"), str):
                    ident["value"] = "[ID]"

        # Clean DocumentReference / Attachment payloads
        if isinstance(res.get("content"), list):
            for item in res["content"]:
                if isinstance(item, dict) and isinstance(item.get("attachment"), dict):
                    _scrub_attachment(item["attachment"], scrub_fn)

        # Clean free-text notes & extensions safely
        if isinstance(res.get("note"), list):
            for n in res["note"]:
                if isinstance(n, dict) and isinstance(n.get("text"), str):
                    n["text"] = scrub_fn(n["text"])
        if isinstance(res.get("extension"), list):
            for ext in res["extension"]:
                if isinstance(ext, dict) and isinstance(ext.get("valueString"), str):
                    ext["valueString"] = scrub_fn(ext["valueString"])

    for item in out.get("entry") or []:
        if isinstance(item, dict) and isinstance(item.get("resource"), dict):
            _clean_resource(item["resource"])
    return out

#!/usr/bin/env python3
"""M3 Requirement 8: Utility-vs-Privacy evaluation table across policy settings (dev-only)."""
from __future__ import annotations

import argparse
import datetime as dt
import os
import re
import shutil
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json

from raksha.pipeline import run_pipeline
from raksha.privacy.dates import infer_month_first
from tools.attacker.simulator import (
    _eval_single_release,
    _parse_dates,
    _read_text_with_b64,
    compute_cross_release_linkability,
    load_registry,
    map_truth_patients_to_registry,
)

_DEVA = str.maketrans("०१२३४५६७८९", "0123456789")


def _norm(s: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", s).translate(_DEVA).casefold().split())


def measure_utility(out_dir: Path, truth: dict) -> dict:
    """Measure medical clinical probes, check_dev preserve %, temporal interval preservation, and record retention."""
    san = out_dir / "sanitized"
    report = json.loads((out_dir / "report.json").read_text(encoding="utf-8"))
    renamed = {e.get("input_sha256"): e.get("output") for e in report.get("files") or []}

    clin_all = clin_ok = 0
    pres_all = pres_ok = 0
    stay_all = stay_ok = 0
    int_all = int_ok = 0

    pt_events_out: dict[str, list[dt.date]] = {}
    pt_events_true: dict[str, list[dt.date]] = {}

    for rel, f in truth["files"].items():
        declared = renamed.get(f["sha256"])
        out_rel = str(declared).replace("sanitized/", "", 1) if declared else rel
        p = san / out_rel
        if not p.is_file():
            continue
        txt_norm = _norm(_read_text_with_b64(p))
        txt_raw = _read_text_with_b64(p)
        month_first = infer_month_first([txt_raw], header_or_rel_hint=rel)

        for _loc, u in f["units"].items():
            for pv in u.get("preserve") or []:
                pres_all += 1
                ok = _norm(pv["text"]) in txt_norm
                if ok:
                    pres_ok += 1
                if pv.get("kind") != "district":
                    clin_all += 1
                    if ok:
                        clin_ok += 1

            # Check intra-record length of stay (admission vs discharge)
            dates_u = u.get("dates") or []
            adm_u = [dt.date.fromisoformat(d["true"]) for d in dates_u if "admission" in str(d.get("event", ""))]
            dis_u = [dt.date.fromisoformat(d["true"]) for d in dates_u if "discharge" in str(d.get("event", ""))]
            if adm_u and dis_u:
                true_los = abs((dis_u[0] - adm_u[0]).days)
                obs_ds = _parse_dates(txt_raw, month_first_hint=month_first)
                if len(obs_ds) >= 2:
                    stay_all += 1
                    if any(abs((obs_ds[j] - obs_ds[i]).days) == true_los for i in range(len(obs_ds)) for j in range(i + 1, len(obs_ds))):
                        stay_ok += 1

            pt = u.get("patient")
            if pt and dates_u:
                for d in dates_u:
                    pt_events_true.setdefault(pt, []).append(dt.date.fromisoformat(d["true"]))

    counts = report.get("counts", {})
    rec_ret = counts.get("records_processed", 0)
    return {
        "clinical_probes_pct": (100.0 * clin_ok / clin_all) if clin_all else 100.0,
        "clinical_ok": f"{clin_ok}/{clin_all}",
        "check_dev_preserve_pct": (100.0 * pres_ok / pres_all) if pres_all else 100.0,
        "pres_ok": f"{pres_ok}/{pres_all}",
        "los_preserved_pct": (100.0 * stay_ok / stay_all) if stay_all else 100.0,
        "records_generalized": counts.get("records_generalized", 0),
        "records_suppressed": counts.get("records_suppressed", 0),
        "records_total": rec_ret,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Utility-vs-Privacy policy comparison table")
    ap.add_argument("--input", type=Path, default=Path(".tmp_toolkit/dev/input"))
    ap.add_argument("--work", type=Path, default=Path(".m3_sweep"))
    args = ap.parse_args()

    dev_dir = args.input.parent
    truth = json.loads((dev_dir / "truth" / "truth.json").read_text(encoding="utf-8"))
    registry = load_registry(dev_dir / "attacker" / "registry.csv")
    facts = [json.loads(l) for l in (dev_dir / "attacker" / "known_facts.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    pt_to_reg = map_truth_patients_to_registry(truth, args.input, registry)

    raw_res = _eval_single_release(args.input, truth, registry, pt_to_reg, facts, is_raw=True)
    raw_util = {"clinical_probes_pct": 100.0, "check_dev_preserve_pct": 100.0, "los_preserved_pct": 100.0}

    policies = [
        ("Raw Input (no privacy)", None),
        (
            "Baseline Uniform (5yr, dist, shift)",
            {"age_band_years": 5, "min_group_size": 5, "adaptive_risk": False, "keep_tehsil_default": False, "keep_district_default": True, "keep_fhir_birthdate": False},
        ),
        (
            "Uniform Wide (10yr, no-dist, shift)",
            {"age_band_years": 10, "min_group_size": 5, "adaptive_risk": False, "keep_tehsil_default": False, "keep_district_default": False, "keep_fhir_birthdate": False},
        ),
        (
            "Adaptive k=3 (policy balance)",
            {"age_band_years": 5, "min_group_size": 3, "adaptive_risk": True, "keep_tehsil_default": False, "keep_district_default": True, "keep_fhir_birthdate": False},
        ),
        (
            "Adaptive k=5 (config/policy.yaml)",
            {"age_band_years": 5, "min_group_size": 5, "adaptive_risk": True, "keep_tehsil_default": False, "keep_district_default": True, "keep_fhir_birthdate": False},
        ),
        (
            "Adaptive k=8 (high privacy)",
            {"age_band_years": 5, "min_group_size": 8, "adaptive_risk": True, "keep_tehsil_default": False, "keep_district_default": True, "keep_fhir_birthdate": False},
        ),
    ]

    if args.work.exists():
        shutil.rmtree(args.work)
    args.work.mkdir(parents=True, exist_ok=True)

    rows_out = []
    for idx, (label, pol) in enumerate(policies):
        if pol is None:
            rows_out.append({
                "label": label,
                "A1": raw_res["A1_direct_exposure"],
                "A3": raw_res["A3_record_linkage"],
                "A4": raw_res["A4_profile_reid"],
                "A5": raw_res["A5_targeted_known_facts"],
                "A6": 0.0,
                "A7": 1.0,
                "clin_pct": 100.0,
                "pres_pct": 100.0,
                "los_pct": 100.0,
                "gen": 0,
            })
            continue

        out_a = args.work / f"p{idx}_a"
        out_b = args.work / f"p{idx}_b"
        os.environ["RAKSHA_SECRET"] = "sweep-secret-alpha-123"
        run_pipeline(args.input, out_a, policy_override=pol)
        os.environ["RAKSHA_SECRET"] = "sweep-secret-beta-456"
        run_pipeline(args.input, out_b, policy_override=pol)

        res_a = _eval_single_release(out_a / "sanitized", truth, registry, pt_to_reg, facts, is_raw=False, records_jsonl=out_a / "records.jsonl", report_json=out_a / "report.json")
        res_b = _eval_single_release(out_b / "sanitized", truth, registry, pt_to_reg, facts, is_raw=False, records_jsonl=out_b / "records.jsonl", report_json=out_b / "report.json")
        a7 = compute_cross_release_linkability(res_a, res_b)
        util = measure_utility(out_a, truth)
        rows_out.append({
            "label": label,
            "A1": res_a["A1_direct_exposure"],
            "A3": res_a["A3_record_linkage"],
            "A4": res_a["A4_profile_reid"],
            "A5": res_a["A5_targeted_known_facts"],
            "A6": res_a["A6_pseudonym_inversion"],
            "A7": a7,
            "clin_pct": util["clinical_probes_pct"],
            "pres_pct": util["check_dev_preserve_pct"],
            "los_pct": util["los_preserved_pct"],
            "gen": util["records_generalized"],
        })

    print("\n" + "=" * 112)
    print(f"{'Policy Setting':<36} | {'A1':>5} {'A3':>5} {'A4':>5} {'A5':>5} {'A6':>5} {'A7':>5} | {'Clin%':>6} {'Dev%':>6} {'LoS%':>6} {'Gen':>4}")
    print("-" * 112)
    for r in rows_out:
        print(
            f"{r['label']:<36} | {r['A1']:5.3f} {r['A3']:5.3f} {r['A4']:5.3f} {r['A5']:5.3f} {r['A6']:5.3f} {r['A7']:5.3f} | "
            f"{r['clin_pct']:6.2f} {r['pres_pct']:6.2f} {r['los_pct']:6.1f} {r['gen']:4d}"
        )
    print("=" * 112)

    shutil.rmtree(args.work, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

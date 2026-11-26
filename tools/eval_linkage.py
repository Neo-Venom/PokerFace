#!/usr/bin/env python3
"""Dev-only identity linkage evaluator against dev/truth/truth.json (never imported by raksha/ or main.py)."""
from __future__ import annotations

import argparse
import itertools
import json
from collections import defaultdict
from pathlib import Path


def _record_for(loc: str, recs: dict[str, str | None]) -> str | None:
    if loc in recs:
        return recs[loc]
    if loc.startswith("/"):
        parts = loc.split("/")
        for i in range(len(parts) - 1, 1, -1):
            anc = "/".join(parts[:i])
            if anc in recs:
                return recs[anc]
    return recs.get("doc")


def evaluate_linkage(output_dir: Path, truth_path: Path, show_diag: int = 15) -> dict:
    """Compute pairwise P/R/F0.5, B-cubed P/R/F1, wrong merges, and pseudonym consistency."""
    truth = json.loads(truth_path.read_text(encoding="utf-8"))
    report = json.loads((output_dir / "report.json").read_text(encoding="utf-8"))
    renamed = {e.get("input_sha256"): e.get("output") for e in report.get("files") or []}

    recs_by_file: dict[str, dict[str, str | None]] = defaultdict(dict)
    rec_rows: list[dict] = []
    for line in (output_dir / "records.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            rec_rows.append(r)
            out_rel = str(r["file"]).replace("sanitized/", "", 1)
            recs_by_file[out_rel][r["loc"]] = r.get("patient")

    san_dir = output_dir / "sanitized"
    file_text_cache: dict[str, str] = {}
    psn_claims = 0
    psn_present = 0
    psn_missing_examples: list[tuple[str, str, str]] = []
    for r in rec_rows:
        psn = r.get("patient")
        if psn:
            psn_claims += 1
            f_rel = str(r["file"]).replace("sanitized/", "", 1)
            if f_rel not in file_text_cache:
                p = san_dir / f_rel
                file_text_cache[f_rel] = p.read_text(encoding="utf-8", errors="replace") if p.is_file() else ""
            if psn in file_text_cache[f_rel]:
                psn_present += 1
            else:
                psn_missing_examples.append((f_rel, r["loc"], psn))

    units: list[tuple[str, str, str, str | None, dict]] = []
    ambiguous_units: list[tuple[str, str, str | None, dict]] = []
    for rel, f in truth["files"].items():
        declared = renamed.get(f["sha256"])
        out_rel = str(declared).replace("sanitized/", "", 1) if declared else rel
        file_recs = recs_by_file.get(out_rel, {})
        for loc, u in f["units"].items():
            pred = _record_for(loc, file_recs)
            if u.get("link") == "required" and u.get("patient"):
                units.append((rel, loc, u["patient"], pred, u))
            elif u.get("link") == "ambiguous":
                ambiguous_units.append((rel, loc, pred, u))

    # Pairwise P / R / F0.5 (same definition as check_dev.py)
    by_true: dict[str, list[int]] = defaultdict(list)
    by_pred: dict[str, list[int]] = defaultdict(list)
    for i, (_rel, _loc, t, y, _u) in enumerate(units):
        by_true[t].append(i)
        if y:
            by_pred[y].append(i)

    true_pairs = sum(len(v) * (len(v) - 1) // 2 for v in by_true.values())
    tp = fp = 0
    fp_examples: list[tuple[tuple[str, str, str], tuple[str, str, str], str]] = []
    for psn, idxs in by_pred.items():
        for i, j in itertools.combinations(idxs, 2):
            if units[i][2] == units[j][2]:
                tp += 1
            else:
                fp += 1
                fp_examples.append((
                    (units[i][0], units[i][1], units[i][2]),
                    (units[j][0], units[j][1], units[j][2]),
                    psn,
                ))
    fn = true_pairs - tp
    pw_prec = tp / (tp + fp) if (tp + fp) else 0.0
    pw_rec = tp / (tp + fn) if (tp + fn) else 0.0
    pw_f05 = (1.25 * pw_prec * pw_rec / (0.25 * pw_prec + pw_rec)) if (pw_prec + pw_rec) else 0.0
    wrong_merges = sum(1 for idxs in by_pred.values() if len({units[i][2] for i in idxs}) > 1)

    # B-cubed Precision / Recall / F1 (unlinked records treated as singletons)
    bc_prec_sum = 0.0
    bc_rec_sum = 0.0
    for i, (_rel, _loc, t, y, _u) in enumerate(units):
        t_set = set(by_true[t])
        p_set = set(by_pred[y]) if y else {i}
        inter = len(t_set & p_set)
        bc_prec_sum += inter / len(p_set)
        bc_rec_sum += inter / len(t_set)
    n_u = max(1, len(units))
    bc_prec = bc_prec_sum / n_u
    bc_rec = bc_rec_sum / n_u
    bc_f1 = (2 * bc_prec * bc_rec / (bc_prec + bc_rec)) if (bc_prec + bc_rec) else 0.0

    # Ambiguous units (should mostly remain null or at least not link multiple people)
    amb_null = sum(1 for _rel, _loc, y, _u in ambiguous_units if y is None)

    print("=" * 68)
    print("LINKAGE EVALUATION (dev/truth)")
    print("=" * 68)
    print(f"Linkable units     : {len(units)} (true clusters: {len(by_true)}, pred clusters: {len(by_pred)})")
    print(f"Pairwise Precision : {pw_prec:.4f} ({tp} TP / {tp + fp} pred pairs, {fp} FP pairs)")
    print(f"Pairwise Recall    : {pw_rec:.4f} ({tp} / {true_pairs} true pairs, {fn} FN pairs)")
    print(f"Pairwise F0.5      : {pw_f05:.4f}")
    print(f"Wrong merges       : {wrong_merges}")
    print(f"B-cubed P / R / F1 : {bc_prec:.4f} / {bc_rec:.4f} / {bc_f1:.4f}")
    print(f"PSN in-record      : {psn_present} / {psn_claims} ({(100 * psn_present / psn_claims) if psn_claims else 0.0:.1f}%)")
    if ambiguous_units:
        print(f"Ambiguous unlinked : {amb_null} / {len(ambiguous_units)}")

    if fp_examples and show_diag > 0:
        print(f"\n--- Top {min(show_diag, len(fp_examples))} False-Positive Merge Pairs ---")
        for a, b, psn in fp_examples[:show_diag]:
            print(f"  [{psn}] {a[2]}@{a[0]}#{a[1]}  <--MERGED-->  {b[2]}@{b[0]}#{b[1]}")

    if psn_missing_examples and show_diag > 0:
        print(f"\n--- Top {min(show_diag, len(psn_missing_examples))} Missing In-Record PSNs ---")
        for f_rel, loc, psn in psn_missing_examples[:show_diag]:
            print(f"  {psn} missing inside {f_rel} ({loc})")

    if show_diag > 0:
        split_pts: list[tuple[str, list[tuple[str, str, str | None]]]] = []
        for pat, idxs in sorted(by_true.items()):
            preds = {units[i][3] for i in idxs}
            if len(preds) > 1 or None in preds:
                split_pts.append((pat, [(units[i][0], units[i][1], units[i][3]) for i in idxs]))
        print(f"\n--- Split / Unlinked True Patients ({len(split_pts)} of {len(by_true)}) ---")
        for pat, items in split_pts[:show_diag]:
            print(f"  Patient {pat} ({len(items)} records):")
            for r_file, r_loc, r_pred in items:
                print(f"    {r_pred or 'NULL':<20} {r_file}#{r_loc}")

    return {
        "pairwise_precision": pw_prec,
        "pairwise_recall": pw_rec,
        "pairwise_f05": pw_f05,
        "wrong_merges": wrong_merges,
        "bcubed_precision": bc_prec,
        "bcubed_recall": bc_rec,
        "bcubed_f1": bc_f1,
        "psn_consistency": psn_present / psn_claims if psn_claims else 1.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate M2 patient identity linkage against truth.json")
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--truth", type=Path, default=None)
    ap.add_argument("--show", type=int, default=15)
    args = ap.parse_args()

    truth = args.truth
    if truth is None:
        cand1 = Path(".tmp_toolkit/dev/truth/truth.json")
        cand2 = Path("../Secret-Wards-ToolKit/dev/truth/truth.json")
        truth = cand1 if cand1.is_file() else cand2
    res = evaluate_linkage(args.output, truth, show_diag=args.show)
    return 0 if (res["pairwise_precision"] >= 0.98 and res["wrong_merges"] == 0) else 1


if __name__ == "__main__":
    raise SystemExit(main())

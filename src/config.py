"""Policy loader for config/policy.yaml."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Policy:
    age_band_years: int = 5
    age_top_code: int = 85
    date_shift_min_days: int = 30
    date_shift_max_days: int = 180
    min_group_size: int = 5
    location_min_group: int = 3
    rare_dx_min_patients: int = 4
    l_diversity: int = 2
    psn_length: int = 8
    linkage_merge_threshold: float = 7.5
    encrypt_quarantine_archive: bool = False
    max_json_depth: int = 500


def load_policy(base_dir: Path | None = None) -> Policy:
    root = base_dir or Path(__file__).resolve().parents[1]
    env_path = os.environ.get("RAKSHA_POLICY") or os.environ.get("RAKSHAK_POLICY")
    policy_path = Path(env_path) if env_path else (root / "config" / "policy.yaml")
    raw: dict[str, Any] = {}
    if policy_path.is_file():
        data = yaml.safe_load(policy_path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            raw = data
    return Policy(
        age_band_years=int(raw.get("age_band_years", 5)),
        age_top_code=int(raw.get("age_top_code", 85)),
        date_shift_min_days=int(raw.get("date_shift_min_days", 30)),
        date_shift_max_days=int(raw.get("date_shift_max_days", 180)),
        min_group_size=int(raw.get("min_group_size", 5)),
        location_min_group=int(raw.get("location_min_group", 3)),
        rare_dx_min_patients=int(raw.get("rare_dx_min_patients", 4)),
        l_diversity=int(raw.get("l_diversity", 2)),
        psn_length=int(raw.get("psn_length", 8)),
        linkage_merge_threshold=float(raw.get("linkage_merge_threshold", 7.5)),
        encrypt_quarantine_archive=bool(raw.get("encrypt_quarantine_archive", False)),
        max_json_depth=int(raw.get("max_json_depth", 500)),
    )

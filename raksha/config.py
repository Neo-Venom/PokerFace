"""Policy configuration and cryptographic secret/pseudonym helpers."""
from __future__ import annotations

import datetime as dt
import hashlib
import hmac
import os
from pathlib import Path

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

ROOT_DIR = Path(__file__).resolve().parent.parent
_ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"  # Crockford base32 (no I, L, O, U)


def load_policy(root: Path = ROOT_DIR) -> dict:
    """Read key: value pairs from config/policy.yaml (falling back to policy.yaml)."""
    policy: dict = {
        "age_band_years": 5,
        "date_shift_max_days": 180,
        "min_group_size": 5,
    }
    for candidate in (root / "config" / "policy.yaml", root / "policy.yaml"):
        if candidate.is_file():
            for line in candidate.read_text(encoding="utf-8").splitlines():
                line = line.split("#", 1)[0].strip()
                if ":" in line:
                    k, v = (x.strip() for x in line.split(":", 1))
                    policy[k] = int(v) if v.lstrip("-").isdigit() else v
            break
    return policy


def require_secret() -> bytes:
    """Validate and return RAKSHA_SECRET (error only if missing or empty)."""
    raw = os.environ.get("RAKSHA_SECRET", "")
    if not raw:
        raise ValueError("missing_secret")
    return raw.encode("utf-8")


def derive_subkey(secret: bytes, info: bytes) -> bytes:
    """Derive a domain-separated 32-byte subkey via HKDF-SHA256."""
    hkdf = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=b"rakshaguard-v1",
        info=info,
    )
    return hkdf.derive(secret)


def pseudonym(patient_key: str, psn_key: bytes) -> str:
    """Issue a deterministic >=80-bit Crockford base32 HMAC pseudonym."""
    digest = hmac.new(psn_key, patient_key.encode("utf-8"), hashlib.sha256).digest()
    # 10 bytes = 80 bits -> 16 Crockford base32 characters
    n = int.from_bytes(digest[:10], "big")
    code = "".join(_ALPHABET[(n >> (5 * i)) & 31] for i in range(15, -1, -1))
    return f"PSN-{code}"


def date_shift_days(patient_key: str, shift_key: bytes, max_days: int = 180) -> int:
    """Deterministic per-patient shift in [-max_days, +max_days], never 0."""
    d = hmac.new(shift_key, b"shift|" + patient_key.encode("utf-8"), hashlib.sha256).digest()
    v = int.from_bytes(d[:4], "big") % (2 * max_days) - max_days
    return v if v != 0 else 1


def shift_date(value: dt.date, patient_key: str, shift_key: bytes, max_days: int = 180) -> dt.date:
    return value + dt.timedelta(days=date_shift_days(patient_key, shift_key, max_days))


def age_band(age: int, width: int = 5) -> str:
    """Return a width-year age band guaranteed to contain `age`."""
    lo = (age // width) * width
    return f"{lo}-{lo + width - 1}"

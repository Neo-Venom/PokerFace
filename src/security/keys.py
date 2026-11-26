"""HKDF-SHA256 subkey derivation, keyed HMAC pseudonyms, and deterministic date shifts."""
from __future__ import annotations

import hashlib
import hmac
import os

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

_CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
_SALT = b"secret-wards-raksha-v1"


def _hkdf_expand(ikm: bytes, info: bytes, length: int = 32) -> bytes:
    return HKDF(
        algorithm=hashes.SHA256(),
        length=length,
        salt=_SALT,
        info=info,
    ).derive(ikm)


class KeyRing:
    """Holds domain-separated subkeys in RAM and scrubs RAKSHA_SECRET from os.environ."""

    def __init__(self, secret_bytes: bytes) -> None:
        ikm = secret_bytes if secret_bytes else b"default-empty-secret-guard"
        self.k_psn = _hkdf_expand(ikm, b"pseudonym")
        self.k_shift = _hkdf_expand(ikm, b"date-shift")
        self.k_fname = _hkdf_expand(ikm, b"file-name")
        self.k_vault = _hkdf_expand(ikm, b"vault")

    @classmethod
    def from_env(cls, scrub_env: bool = True) -> "KeyRing":
        raw = os.environ.get("RAKSHA_SECRET", "")
        if scrub_env and "RAKSHA_SECRET" in os.environ:
            os.environ.pop("RAKSHA_SECRET", None)
        return cls(raw.encode("utf-8"))

    def pseudonym(self, cluster_key: str, length: int = 8, disambiguator: int = 0) -> str:
        msg = cluster_key if disambiguator == 0 else f"{cluster_key}#{disambiguator}"
        digest = hmac.new(self.k_psn, msg.encode("utf-8"), hashlib.sha256).digest()
        val = int.from_bytes(digest[:8], "big")
        code = "".join(_CROCKFORD[(val >> (5 * i)) & 31] for i in range(length))
        return f"PSN-{code}"

    def date_shift_days(self, cluster_key: str, lo: int = 30, hi: int = 180) -> int:
        digest = hmac.new(self.k_shift, cluster_key.encode("utf-8"), hashlib.sha256).digest()
        val = int.from_bytes(digest[:4], "big")
        span = max(1, hi - lo + 1)
        mag = lo + (val % span)
        return mag if ((val >> 31) & 1) else -mag

    def file_tag(self, input_sha256: str, length: int = 8) -> str:
        digest = hmac.new(self.k_fname, input_sha256.encode("utf-8"), hashlib.sha256).digest()
        val = int.from_bytes(digest[:8], "big")
        return "".join(_CROCKFORD[(val >> (5 * i)) & 31] for i in range(length))

    def clear(self) -> None:
        self.k_psn = b""
        self.k_shift = b""
        self.k_fname = b""
        self.k_vault = b""

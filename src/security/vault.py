"""AES-256-GCM authenticated encryption helper for optional quarantined data at rest."""
from __future__ import annotations

import os
from pathlib import Path

from cryptography.hazmat.primitives.ciphers.aead import AESGCM


class EncryptedVault:
    """Envelope-style AES-256-GCM encryption keyed from HKDF vault subkey."""

    def __init__(self, vault_key: bytes) -> None:
        if len(vault_key) not in (16, 24, 32):
            raise ValueError("invalid_vault_key")
        self._aes = AESGCM(vault_key)

    def seal_bytes(self, plaintext: bytes, aad_sha256: str) -> bytes:
        nonce = os.urandom(12)
        ct = self._aes.encrypt(nonce, plaintext, aad_sha256.encode("ascii"))
        return nonce + ct

    def unseal_bytes(self, blob: bytes, aad_sha256: str) -> bytes:
        if len(blob) < 28:
            raise ValueError("corrupt_vault_blob")
        nonce, ct = blob[:12], blob[12:]
        return self._aes.decrypt(nonce, ct, aad_sha256.encode("ascii"))

    def write_quarantined_blob(self, dest: Path, raw_bytes: bytes, input_sha256: str) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(self.seal_bytes(raw_bytes, input_sha256))

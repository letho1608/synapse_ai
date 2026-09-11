"""
Behavioral tests: TLS staging (T1.1-Inc3 / C4, H5).

- Default OFF: pass-through is byte-identical (same object back).
- Builders fail LOUD on missing files (never silent plaintext fallback).
- Fingerprint deterministic, 64 hex chars, input-sensitive.
- Invalid modes resolve to off; fingerprint rejects empty input.
- Enabling TLS later changes bytes: flag-flip needs interop run (not here).
"""
import pytest

from synapse.transport import tls_wrap
from synapse.transport.tls_wrap import (
    build_client_context,
    build_server_context,
    fingerprint_bytes,
    maybe_wrap_socket,
    tls_enabled,
)


def test_default_off_and_invalid_modes(monkeypatch):
    monkeypatch.delenv("SYNAPSE_TLS", raising=False)
    assert tls_enabled() == "off"
    monkeypatch.setenv("SYNAPSE_TLS", "require")
    assert tls_enabled() == "require"
    monkeypatch.setenv("SYNAPSE_TLS", "BOGUS")
    assert tls_enabled() == "off"


def test_passthrough_returns_same_object():
    sock = object()
    assert maybe_wrap_socket(sock, None) is sock


def test_builders_refuse_missing_files_loudly(tmp_path):
    with pytest.raises(FileNotFoundError):
        build_server_context(str(tmp_path / "nope.crt"), str(tmp_path / "nope.key"))
    with pytest.raises(FileNotFoundError):
        build_client_context(cafile=str(tmp_path / "nope-ca.crt"))
    # No-CA client context builds (pinning enforced at call site, not here).
    ctx = build_client_context()
    assert ctx.verify_mode == tls_wrap.ssl.CERT_NONE


def test_fingerprint_deterministic():
    fp1 = fingerprint_bytes(b"cert-bytes-1")
    assert len(fp1) == 64
    assert fingerprint_bytes(b"cert-bytes-1") == fp1
    assert fingerprint_bytes(b"cert-bytes-2") != fp1
    with pytest.raises(ValueError):
        fingerprint_bytes(b"")

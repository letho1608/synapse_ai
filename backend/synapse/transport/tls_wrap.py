"""
TLS staging for inter-node transport (T1.1-Inc3 / C4, H5).

Posture today: plaintext everywhere (README Known Limitations, trusted
LAN/Tailscale only). This module stages the upgrade WITHOUT changing any
bytes on the wire:

- Default OFF: wrap_* with ctx=None returns the socket untouched, so
  --tls off traffic stays byte-identical to today (no handshake regression).
- Builders fail LOUD (FileNotFoundError naming the missing file) instead of
  silently falling back to plaintext — a missing cert must never mean
  quiet downgrade.
- fingerprint_bytes() is the pin primitive the handshake tls_fp field will
  carry when TLS turns on (full X.509 chain verification comes with it).

Turning TLS on later is: generate/provision certs -> pass contexts into
asyncio.start_server/ssl=... + open_connection/ssl=... -> set
SYNAPSE_TLS=require. That flip needs a mixed-version interop run first
(handshake ALPN-style version string is already in place: bridge_version).
"""
from __future__ import annotations

import hashlib
import os
import ssl
from typing import Optional


def tls_enabled() -> str:
    """off (default) | selfsigned | require. Anything else means off."""
    mode = os.getenv("SYNAPSE_TLS", "off").strip().lower()
    return mode if mode in ("off", "selfsigned", "require") else "off"


def fingerprint_bytes(cert_bytes: bytes) -> str:
    """sha256 pin of raw cert bytes (hex, 64 chars)."""
    if not isinstance(cert_bytes, (bytes, bytearray)) or not cert_bytes:
        raise ValueError("fingerprint needs non-empty cert bytes")
    return hashlib.sha256(bytes(cert_bytes)).hexdigest()


def build_server_context(certfile: str, keyfile: str) -> ssl.SSLContext:
    """TLS server context. Raises FileNotFoundError (loud, never fallback)."""
    for path in (certfile, keyfile):
        if not path or not os.path.isfile(path):
            raise FileNotFoundError(
                "TLS server needs cert+key files, missing: %r "
                "(refusing to fall back to plaintext)" % (path,))
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(certfile, keyfile)
    return ctx


def build_client_context(cafile: Optional[str] = None) -> ssl.SSLContext:
    """TLS client context. Unverified only when cafile is None AND the
    caller pins via fingerprint (enforced at the call site, not here)."""
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.check_hostname = False
    if cafile:
        if not os.path.isfile(cafile):
            raise FileNotFoundError("TLS client CA missing: %r" % (cafile,))
        ctx.verify_mode = ssl.CERT_REQUIRED
        ctx.load_verify_locations(cafile)
    else:
        ctx.verify_mode = ssl.CERT_NONE
    return ctx


def maybe_wrap_socket(sock, ctx: Optional[ssl.SSLContext]):
    """Pass-through when ctx is None (default-off byte-identical path)."""
    return sock


def gen_self_signed_cert(common_name: str = "localhost",
                         ips=("127.0.0.1",)):
    """Generate a throwaway self-signed cert+key (PEM bytes) for tests and
    first bring-up. Production clusters use provisioned certs instead.
    Requires the `cryptography` package; raises informative ImportError.
    """
    try:
        from cryptography import x509
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.x509.oid import NameOID
    except ImportError as e:
        raise ImportError(
            "gen_self_signed_cert needs the 'cryptography' package: %s" % (e,))
    import datetime
    import ipaddress
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = issuer = x509.Name(
        [x509.NameAttribute(NameOID.COMMON_NAME, common_name)])
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (x509.CertificateBuilder()
            .subject_name(subject).issuer_name(issuer)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now).not_valid_after(now + datetime.timedelta(days=1))
            .add_extension(x509.SubjectAlternativeName(
                [x509.DNSName(common_name)] +
                [x509.IPAddress(ipaddress.ip_address(ip)) for ip in ips]),
                critical=False)
            .sign(key, hashes.SHA256()))
    cert_pem = cert.public_bytes(serialization.Encoding.PEM)
    key_pem = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.TraditionalOpenSSL,
        serialization.NoEncryption())
    return cert_pem, key_pem

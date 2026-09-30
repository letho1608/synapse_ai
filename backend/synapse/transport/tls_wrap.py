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
    """TLS client context. Without a CA the chain is not verified here; the
    connection sites authenticate the server with check_peer_pin()."""
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


# ── Wiring: pins, self-signed bring-up, startup config ───────────────────

def cert_fingerprint_pem(cert_pem: bytes) -> str:
    """Pin of a PEM cert, computed over its DER form: the same bytes a peer
    sees in the TLS handshake (getpeercert(binary_form=True))."""
    der = ssl.PEM_cert_to_DER_cert(cert_pem.decode("ascii"))
    return fingerprint_bytes(der)


def allowed_pins() -> set:
    """SYNAPSE_TLS_PINS: comma-separated sha256 cert pins (hex)."""
    raw = os.getenv("SYNAPSE_TLS_PINS", "")
    return {p.strip().lower() for p in raw.split(",") if p.strip()}


def check_peer_pin(writer, pins=None) -> str:
    """Verify the server cert on a client TLS connection against the pins.

    With no CA, a TLS client (CERT_NONE) encrypts but does not know WHO it
    talks to; the pin is what authenticates the peer. No-op for plaintext
    connections and when no pins are configured (CA mode). Raises
    ssl.SSLError on a mismatch. Returns the peer fingerprint ("" if none)."""
    pins = allowed_pins() if pins is None else pins
    obj = writer.get_extra_info("ssl_object") if writer is not None else None
    if obj is None:
        return ""
    der = obj.getpeercert(binary_form=True)
    fp = fingerprint_bytes(der) if der else ""
    if pins and fp not in pins:
        raise ssl.SSLError("peer certificate %s... is not pinned (SYNAPSE_TLS_PINS)" % fp[:16])
    return fp


def ensure_self_signed(node_id: str, directory: Optional[str] = None):
    """(cert_path, key_path) for SYNAPSE_TLS=selfsigned without provisioned
    files: generated once per node id under ~/.synapse/tls and reused."""
    directory = directory or os.path.join(os.path.expanduser("~"), ".synapse", "tls")
    os.makedirs(directory, exist_ok=True)
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in node_id) or "node"
    cert_path = os.path.join(directory, safe + ".cert.pem")
    key_path = os.path.join(directory, safe + ".key.pem")
    if not (os.path.isfile(cert_path) and os.path.isfile(key_path)):
        cert_pem, key_pem = gen_self_signed_cert(common_name=safe)
        with open(key_path, "wb") as f:
            f.write(key_pem)
        with open(cert_path, "wb") as f:
            f.write(cert_pem)
    return cert_path, key_path


def configure_from_env(node_id: str):
    """(server_ctx, client_ctx, own_pin) from SYNAPSE_TLS*, or Nones when off.

    - selfsigned: SYNAPSE_TLS_CERT/KEY if given, else a generated per-node
      cert; peers are authenticated by SYNAPSE_TLS_PINS (or a CA if set).
    - require: provisioned cert/key, and a CA or pins MUST be configured:
      TLS without either authenticates nobody, so it is refused loudly.
    Missing files always fail loud; nothing falls back to plaintext."""
    mode = tls_enabled()
    if mode == "off":
        return None, None, None
    cert = os.getenv("SYNAPSE_TLS_CERT", "")
    key = os.getenv("SYNAPSE_TLS_KEY", "")
    ca = os.getenv("SYNAPSE_TLS_CA", "") or None
    if mode == "selfsigned" and not (cert or key):
        cert, key = ensure_self_signed(node_id)
    if mode == "require" and not ca and not allowed_pins():
        raise ValueError("SYNAPSE_TLS=require needs SYNAPSE_TLS_CA or SYNAPSE_TLS_PINS: "
                         "without either no peer is authenticated")
    server_ctx = build_server_context(cert, key)
    client_ctx = build_client_context(cafile=ca)
    with open(cert, "rb") as f:
        own_pin = cert_fingerprint_pem(f.read())
    return server_ctx, client_ctx, own_pin

"""
TLS staging for inter-node transport (T1.1-Inc3 / C4, H5).

Posture: a machine paired with `synapse pair` runs mutual TLS against a
cluster CA derived from the pairing code (mode "cluster", the default once
a code exists). Without a code, or with SYNAPSE_TLS=off, traffic is
plaintext (trusted LAN/Tailscale only):

- OFF: wrap_* with ctx=None returns the socket untouched, so plaintext
  traffic stays byte-identical (no handshake regression).
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
    """off | selfsigned | require | cluster.

    Unset SYNAPSE_TLS means "cluster" when this machine has a pairing code
    (`synapse pair`), else off. An explicit SYNAPSE_TLS=off always wins;
    an unknown value means off."""
    raw = os.getenv("SYNAPSE_TLS")
    if raw is None or not raw.strip():
        return "cluster" if cluster_code() else "off"
    mode = raw.strip().lower()
    return mode if mode in ("off", "selfsigned", "require", "cluster") else "off"


# ── Pairing: one code per cluster instead of certs + pins per machine ────
#
# Every machine given the same code derives the same cluster CA key from it
# (scrypt -> P-256 private key), signs its own node certificate with it, and
# runs mutual TLS against that CA: a node without the code cannot complete
# a handshake in either direction. The code is the only secret; it carries
# 160 random bits, so guessing it from an observed certificate is out of
# reach even before the scrypt cost.

_CODE_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"  # no 0/O, 1/I
_CODE_SALT = b"synapse-cluster-ca-v1"


def _synapse_dir() -> str:
    return os.path.join(os.path.expanduser("~"), ".synapse")


def code_path() -> str:
    return os.path.join(_synapse_dir(), "cluster_code")


def new_cluster_code() -> str:
    """A fresh pairing code: 32 symbols (160 bits) in groups of four."""
    import secrets
    sym = "".join(secrets.choice(_CODE_ALPHABET) for _ in range(32))
    return "-".join(sym[i:i + 4] for i in range(0, 32, 4))


def normalize_code(code: str) -> str:
    """Canonical form: case, spaces and dashes do not matter when typing.
    Raises ValueError for a code that is not one `new_cluster_code` makes."""
    sym = "".join(c for c in (code or "").upper() if c.isalnum())
    if len(sym) != 32 or any(c not in _CODE_ALPHABET for c in sym):
        raise ValueError("pairing code must be 32 letters/digits (from `synapse pair`)")
    return "-".join(sym[i:i + 4] for i in range(0, 32, 4))


def cluster_code() -> Optional[str]:
    """This machine's pairing code: SYNAPSE_CLUSTER_CODE, else the saved file."""
    raw = os.getenv("SYNAPSE_CLUSTER_CODE", "").strip()
    if not raw:
        try:
            with open(code_path(), encoding="ascii") as f:
                raw = f.read().strip()
        except OSError:
            return None
    return normalize_code(raw) if raw else None


def save_cluster_code(code: str) -> str:
    code = normalize_code(code)
    os.makedirs(_synapse_dir(), exist_ok=True)
    path = code_path()
    with open(path, "w", encoding="ascii") as f:
        f.write(code + "\n")
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    return path


def _cluster_ca(code: str):
    """(ca_key, ca_cert) derived from the code: same key on every machine."""
    import datetime
    import hashlib as _h
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID
    seed = _h.scrypt(normalize_code(code).encode(), salt=_CODE_SALT, n=2 ** 14, r=8, p=1, dklen=48)
    order = 0xFFFFFFFF00000000FFFFFFFFFFFFFFFFBCE6FAADA7179E84F3B9CAC2FC632551  # P-256
    key = ec.derive_private_key(int.from_bytes(seed, "big") % (order - 1) + 1, ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "synapse-cluster-ca")])
    start = datetime.datetime(2024, 1, 1, tzinfo=datetime.timezone.utc)
    cert = (x509.CertificateBuilder()
            .subject_name(name).issuer_name(name).public_key(key.public_key())
            .serial_number(1)
            .not_valid_before(start).not_valid_after(start + datetime.timedelta(days=365 * 50))
            .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
            .add_extension(x509.KeyUsage(digital_signature=False, content_commitment=False,
                                         key_encipherment=False, data_encipherment=False,
                                         key_agreement=False, key_cert_sign=True, crl_sign=True,
                                         encipher_only=False, decipher_only=False), critical=True)
            .add_extension(x509.SubjectKeyIdentifier.from_public_key(key.public_key()), critical=False)
            .sign(key, hashes.SHA256()))
    return key, cert


def ensure_cluster_identity(node_id: str, code: str, directory: Optional[str] = None):
    """(cert_path, key_path, ca_path) for this node under the pairing code.
    The node cert is reissued when the code changes (different CA key)."""
    import datetime
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID
    directory = directory or os.path.join(_synapse_dir(), "tls")
    os.makedirs(directory, exist_ok=True)
    ca_key, ca_cert = _cluster_ca(code)
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in node_id) or "node"
    cert_path = os.path.join(directory, safe + ".cluster.cert.pem")
    key_path = os.path.join(directory, safe + ".cluster.key.pem")
    ca_path = os.path.join(directory, "cluster-ca.pem")
    pem = serialization.Encoding.PEM
    valid = False
    if os.path.isfile(cert_path) and os.path.isfile(key_path):
        try:
            with open(cert_path, "rb") as f:
                cur = x509.load_pem_x509_certificate(f.read())
            ca_key.public_key().verify(cur.signature, cur.tbs_certificate_bytes,
                                       ec.ECDSA(cur.signature_hash_algorithm))
            valid = cur.not_valid_after_utc > datetime.datetime.now(datetime.timezone.utc)
        except Exception:
            valid = False
    if not valid:
        key = ec.generate_private_key(ec.SECP256R1())
        now = datetime.datetime.now(datetime.timezone.utc)
        cert = (x509.CertificateBuilder()
                .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, safe)]))
                .issuer_name(ca_cert.subject).public_key(key.public_key())
                .serial_number(x509.random_serial_number())
                .not_valid_before(now - datetime.timedelta(days=1))
                .not_valid_after(now + datetime.timedelta(days=3650))
                .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
                .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH,
                                                      ExtendedKeyUsageOID.CLIENT_AUTH]), critical=False)
                .add_extension(x509.AuthorityKeyIdentifier.from_issuer_public_key(ca_key.public_key()),
                               critical=False)
                .sign(ca_key, hashes.SHA256()))
        with open(key_path, "wb") as f:
            f.write(key.private_bytes(pem, serialization.PrivateFormat.TraditionalOpenSSL,
                                      serialization.NoEncryption()))
        with open(cert_path, "wb") as f:
            f.write(cert.public_bytes(pem))
    with open(ca_path, "wb") as f:
        f.write(ca_cert.public_bytes(pem))
    return cert_path, key_path, ca_path


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


def enable_mutual(server_ctx: ssl.SSLContext, client_ctx: ssl.SSLContext,
                  certfile: str, keyfile: str, cafile: str) -> None:
    """Mutual TLS (CA mode): servers require a client certificate signed by
    the cluster CA, and clients present theirs. Pins-only clusters cannot do
    this with the stdlib (a server cannot demand a client cert while
    skipping chain verification), so there the client side is authenticated
    by the server pin only."""
    server_ctx.verify_mode = ssl.CERT_REQUIRED
    server_ctx.load_verify_locations(cafile)
    client_ctx.load_cert_chain(certfile, keyfile)


def configure_from_env(node_id: str):
    """(server_ctx, client_ctx, own_pin) from SYNAPSE_TLS*, or Nones when off.

    - selfsigned: SYNAPSE_TLS_CERT/KEY if given, else a generated per-node
      cert; peers are authenticated by SYNAPSE_TLS_PINS (or a CA if set).
    - require: provisioned cert/key, and a CA or pins MUST be configured:
      TLS without either authenticates nobody, so it is refused loudly.
    - With a CA, TLS is mutual: both ends prove a CA-signed certificate.
    Missing files always fail loud; nothing falls back to plaintext."""
    mode = tls_enabled()
    if mode == "off":
        return None, None, None
    cert = os.getenv("SYNAPSE_TLS_CERT", "")
    key = os.getenv("SYNAPSE_TLS_KEY", "")
    ca = os.getenv("SYNAPSE_TLS_CA", "") or None
    if mode == "cluster":
        code = cluster_code()
        if not code:
            raise ValueError("SYNAPSE_TLS=cluster needs a pairing code: run `synapse pair` "
                             "(or set SYNAPSE_CLUSTER_CODE)")
        cert, key, ca = ensure_cluster_identity(node_id, code)
    if mode == "selfsigned" and not (cert or key):
        cert, key = ensure_self_signed(node_id)
    if mode == "require" and not ca and not allowed_pins():
        raise ValueError("SYNAPSE_TLS=require needs SYNAPSE_TLS_CA or SYNAPSE_TLS_PINS: "
                         "without either no peer is authenticated")
    server_ctx = build_server_context(cert, key)
    client_ctx = build_client_context(cafile=ca)
    if ca:
        enable_mutual(server_ctx, client_ctx, cert, key, ca)
    with open(cert, "rb") as f:
        own_pin = cert_fingerprint_pem(f.read())
    return server_ctx, client_ctx, own_pin

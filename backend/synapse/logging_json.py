"""
Canonical JSONL logging for scripts/CI (T2.1 / M4 stdout discipline).

Contract:
- emit() writes EXACTLY one JSON object per call to stdout, nothing else.
  Human hints go to stderr (caller's job). Timestamps are epoch floats.
- redact() drops secret-looking keys (dashed/underscored/camel alike).
- parse_stream() validates JSONL (one object per non-blank line) for tests
  and for scripts asserting stdout discipline.

Backend print sites migrate to this incrementally (M4); new code uses it
from day one. Nothing here touches the network or the event mesh.
"""
from __future__ import annotations

import json
import sys
import time
from typing import Any, Dict, List, Optional

LEVELS = ("DEBUG", "INFO", "WARN", "ERROR")


def _is_secret_key(key: str) -> bool:
    norm = key.lower().replace("-", "").replace("_", "")
    return any(s in norm for s in ("apikey", "secret", "token", "password"))


def redact(obj):
    if isinstance(obj, dict):
        return {k: ("***" if _is_secret_key(k) else redact(v))
                for k, v in obj.items()}
    if isinstance(obj, list):
        return [redact(v) for v in obj]
    return obj


def emit(level: str, cmd: str, msg: str,
         fields: Optional[Dict[str, Any]] = None,
         node_id: Optional[str] = None,
         ts: Optional[float] = None) -> Dict[str, Any]:
    """Emit one JSONL record to stdout. Returns the record."""
    if level not in LEVELS:
        level = "INFO"
    record = {"ts": ts if ts is not None else time.time(),
              "level": level, "cmd": cmd, "node_id": node_id,
              "msg": msg, "fields": redact(fields or {})}
    line = json.dumps(record) + "\n"
    write_raw = getattr(sys.stdout, "write_raw", None)
    if callable(write_raw):
        write_raw(line)
    else:
        sys.stdout.write(line)
        sys.stdout.flush()
    return record


def parse_stream(text: str) -> List[Dict[str, Any]]:
    """Parse JSONL strictly: every non-blank line must be one JSON object."""
    records = []
    for lineno, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        obj = json.loads(line)
        if not isinstance(obj, dict):
            raise ValueError("line %d is not a JSON object" % lineno)
        records.append(obj)
    return records


class _JsonLinesProxy:
    """Stdout wrapper making legacy print() output JSONL-parseable (M4).

    Buffers until newline; each complete line becomes one record. Partial
    lines flush through on flush(). Delegates everything else (isatty,
    fileno, encoding) to the wrapped stream. Install once via
    install_stdout_json_redirect(); human output belongs on stderr.
    """

    def __init__(self, wrapped):
        self._wrapped = wrapped
        self._buf = ""

    def write(self, s):
        if not isinstance(s, str):
            s = str(s)
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if line.strip():
                self.write_raw(json.dumps({
                    "ts": time.time(), "level": "INFO", "cmd": "backend",
                    "node_id": None, "msg": line,
                    "fields": {"stream": "stdout"}}) + "\n")
        return len(s)

    def write_raw(self, s):
        self._wrapped.write(s)
        try:
            self._wrapped.flush()
        except Exception:
            pass
        return len(s)

    def flush(self):
        if self._buf.strip():
            rest, self._buf = self._buf, ""
            self.write_raw(json.dumps({
                "ts": time.time(), "level": "INFO", "cmd": "backend",
                "node_id": None, "msg": rest,
                "fields": {"stream": "stdout", "partial": True}}) + "\n")
        try:
            self._wrapped.flush()
        except Exception:
            pass

    def __getattr__(self, name):
        return getattr(self.__dict__["_wrapped"], name)


def install_stdout_json_redirect() -> bool:
    """Replace sys.stdout with the JSONL proxy. Idempotent. Returns True
    when (newly) installed."""
    if isinstance(sys.stdout, _JsonLinesProxy):
        return False
    sys.stdout = _JsonLinesProxy(sys.stdout)
    return True

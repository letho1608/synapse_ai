"""
Per-machine CLI defaults (zero-flag boot + change-later inside console).

File: ~/.synapse/cli-defaults.json (override via SYNAPSE_DEFAULTS_PATH).
Only whitelisted keys are stored; anything else is ignored (never executes).
Precedence: explicit CLI flag > defaults file > builtin default (None =
"let the backend decide", e.g. auto node-id).

Changing a default never touches the running node — it applies to the next
serve/console boot. /set says so explicitly.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

BUILTIN_DEFAULTS: Dict[str, Any] = {
    "model": "HuggingFaceTB/SmolLM2-135M-Instruct",
    "node-id": None,
    "node-port": None,
    "discovery-module": None,
    "config": None,
    "wait-for-peers": None,
    "partition-method": None,
    "temperature": None,
    "chat-port": None,
}

INT_KEYS = ("node-port", "wait-for-peers", "chat-port")
FLOAT_KEYS = ("temperature", "ilp-time-limit-seconds")
CHOICE_KEYS = {
    "discovery-module": ("udp", "tailscale", "manual"),
    "partition-method": ("lrm", "ilp"),
}
VALID_KEYS = set(BUILTIN_DEFAULTS) | {"ilp-time-limit-seconds"}


def defaults_path(path: Optional[str] = None) -> Path:
    if path:
        return Path(path)
    env = os.getenv("SYNAPSE_DEFAULTS_PATH")
    if env:
        return Path(env)
    return Path.home() / ".synapse" / "cli-defaults.json"


def load(path: Optional[str] = None) -> Dict[str, Any]:
    """Read the file; missing/corrupt/non-dict -> {} (never raises)."""
    try:
        p = defaults_path(path)
        if not p.is_file():
            return {}
        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return {}
        return {k: v for k, v in data.items() if k in VALID_KEYS}
    except Exception:
        return {}


def validate(key: str, value: str):
    """Validate + coerce one value. Returns (ok, coerced_or_error)."""
    if key not in VALID_KEYS:
        return False, "unknown key '%s' (see /config)" % key
    if key in INT_KEYS:
        try:
            iv = int(value)
        except ValueError:
            return False, "'%s' needs an integer, got '%s'" % (key, value)
        if iv <= 0 or iv > 65535:
            return False, "'%s' out of range 1-65535" % key
        return True, iv
    if key in FLOAT_KEYS:
        try:
            return True, float(value)
        except ValueError:
            return False, "'%s' needs a number, got '%s'" % (key, value)
    if key in CHOICE_KEYS:
        if value not in CHOICE_KEYS[key]:
            return False, "'%s' must be one of %s" % (key, "/".join(CHOICE_KEYS[key]))
        return True, value
    if key in ("model", "node-id", "config"):
        return (True, value) if str(value).strip() else (False, "'%s' must not be empty" % key)
    return True, value


def save(values: Dict[str, Any], path: Optional[str] = None) -> Path:
    """Persist whitelisted keys (creates ~/.synapse as needed)."""
    p = defaults_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    clean = {k: v for k, v in values.items() if k in VALID_KEYS and v is not None}
    p.write_text(json.dumps(clean, indent=2, ensure_ascii=False), encoding="utf-8")
    return p


def set_key(key: str, value: str, path: Optional[str] = None):
    """Validate + persist one key. Returns (ok, message)."""
    ok, out = validate(key, value)
    if not ok:
        return False, out
    current = load(path)
    current[key] = out
    save(current, path)
    return True, "'%s' = %r (applies to the next serve/console boot)" % (key, out)


def unset_key(key: str, path: Optional[str] = None):
    """Remove one key from the file. Returns (ok, message)."""
    if key not in VALID_KEYS:
        return False, "unknown key '%s' (see /config)" % key
    current = load(path)
    if key not in current:
        return True, "'%s' was not set" % key
    del current[key]
    save(current, path)
    return True, "'%s' cleared (builtin default applies next boot)" % key


def effective(explicit: Dict[str, Any], path: Optional[str] = None) -> Dict[str, Any]:
    """Merge: explicit non-None CLI values > file > builtins."""
    merged = dict(BUILTIN_DEFAULTS)
    merged.update(load(path))
    for k, v in explicit.items():
        if k in VALID_KEYS and v is not None:
            merged[k] = v
    return merged


def describe_source(key: str, explicit: Dict[str, Any],
                    path: Optional[str] = None) -> str:
    """Where does the effective value come from: cli/file/builtin."""
    if explicit.get(key) is not None:
        return "cli"
    if key in load(path):
        return "file"
    return "builtin"


def apply_to_namespace(args, path: Optional[str] = None):
    """Fill None argparse fields: explicit flags stay, else file, else the
    non-None builtins (today: model). Unknown/hand-edited junk never crashes.
    """
    try:
        merged = effective({}, path)
    except Exception:
        return args
    for key, val in merged.items():
        try:
            dest = key.replace("-", "_")
            if getattr(args, dest, None) is None and val is not None:
                setattr(args, dest, val)
        except Exception:
            pass
    return args

"""
Theme system for the synapse console. Schemas mirror the reference repos:

- OpenCode: theme JSON with `defs` (reusable colors) + `theme` (token map),
  values accept #rrggbb hex, ANSI 0-255 ints, references to other tokens,
  {"dark":..,"light":..} variants, and "none" (terminal default).
  `/theme` picks, leader is ctrl+x, palette is ctrl+p.
- Claude Code: custom files {name, base, overrides} in a themes dir,
  value forms #rrggbb / #rgb / rgb() / ansi256(n) / ansi names.
  Brand orange rgb(215,119,87); dark success #4EBA65 / error #FF6B80;
  light success #2C7A39 / error #AB2B3F.

Builtins: `synapse` (default dark, Tokyonight-based),
`claude-dark`, `claude-light` (Claude brand tokens above).
Custom files live in ~/.synapse/themes/*.json; the selection lives in
~/.synapse/theme.json as {"theme": name}. Applies next console boot
(TUI live-applies when it can, else says so).
"""
from __future__ import annotations

import json
import os
import re

THEME_DIR = os.path.join(os.path.expanduser("~"), ".synapse", "themes")
SELECTION_FILE = os.path.join(os.path.expanduser("~"), ".synapse", "theme.json")

# Our UI token set (subset of both repos' tokens, only what we render).
TOKENS = ("accent", "text", "textMuted", "success", "error", "warning",
          "permission", "panel", "border", "borderActive", "selection",
          "inverseText")

BUILTINS = {
    "synapse": {
        "accent": "#7aa2f7", "text": "#c0caf5", "textMuted": "#565f89",
        "success": "#9ece6a", "error": "#f7768e", "warning": "#e0af68",
        "permission": "#bb9af7", "panel": "#24283b", "border": "#3b4261",
        "borderActive": "#7aa2f7", "selection": "#283457",
        "inverseText": "#1a1b26",
    },
    "ember-dark": {
        "accent": "#d77757", "text": "#ffffff", "textMuted": "#8a8a8a",
        "success": "#4eba65", "error": "#ff6b80", "warning": "#e0af68",
        "permission": "#a78bfa", "panel": "#1e1e1e", "border": "#3a3a3a",
        "borderActive": "#d77757", "selection": "#3d2b23",
        "inverseText": "#000000",
    },
    "ember-light": {
        "accent": "#c15f3c", "text": "#000000", "textMuted": "#6e6e6e",
        "success": "#2c7a39", "error": "#ab2b3f", "warning": "#9a6a1a",
        "permission": "#7c3aed", "panel": "#f2f2f2", "border": "#d0d0d0",
        "borderActive": "#c15f3c", "selection": "#f5d9c8",
        "inverseText": "#ffffff",
    },
}

_ANSI_NAMES = {"black": "ansiblack", "red": "ansired", "green": "ansigreen",
               "yellow": "ansiyellow", "blue": "ansiblue",
               "magenta": "ansimagenta", "cyan": "ansicyan",
               "white": "ansigray",
               "brightblack": "ansibrightblack", "brightred": "ansibrightred",
               "brightgreen": "ansibrightgreen", "brightyellow": "ansibrightyellow",
               "brightblue": "ansibrightblue", "brightmagenta": "ansibrightmagenta",
               "brightcyan": "ansibrightcyan", "brightwhite": "ansiwhite"}


def _norm_color(value):
    """Normalize one theme value to a prompt_toolkit color string.

    Returns "" for "none" (terminal default). Unknown/invalid -> "" too
    (a typo must never break the UI — same rule as Claude Code).
    """
    if value is None:
        return ""
    if isinstance(value, int):
        return ("ansi%d" % value) if 0 <= value <= 255 else ""
    s = str(value).strip()
    if not s or s.lower() == "none":
        return ""
    if s.startswith("#"):
        hexpart = s[1:]
        if re.fullmatch(r"[0-9a-fA-F]{6}", hexpart):
            return "#" + hexpart.lower()
        if re.fullmatch(r"[0-9a-fA-F]{3}", hexpart):
            return "#" + "".join(c * 2 for c in hexpart.lower())
        return ""
    m = re.fullmatch(r"rgb\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)", s)
    if m:
        r, g, b = (max(0, min(255, int(x))) for x in m.groups())
        return "#%02x%02x%02x" % (r, g, b)
    m = re.fullmatch(r"ansi256\(\s*(\d+)\s*\)", s)
    if m:
        n = int(m.group(1))
        return ("ansi%d" % n) if 0 <= n <= 255 else ""
    low = s.lower()
    if low.startswith("bright") and len(low) > 6:
        key = "bright" + re.sub(r"[^a-z]", "", low[6:])
    elif low.endswith("bright") and len(low) > 6:
        key = "bright" + re.sub(r"[^a-z]", "", low[:-6])
    else:
        key = re.sub(r"[^a-z]", "", low)
    return _ANSI_NAMES.get(key, "")


def _resolve_refs(theme, defs):
    """Resolve token values that reference defs or other tokens (one pass)."""
    defs = defs or {}
    out = {}
    for tok, val in (theme or {}).items():
        if isinstance(val, dict):
            val = val.get("dark", val.get("light"))
        if isinstance(val, str) and val in defs:
            val = defs[val]
        if isinstance(val, str) and val in theme and val != tok:
            ref = theme[val]
            if isinstance(ref, dict):
                ref = ref.get("dark", ref.get("light"))
            val = ref
        out[tok] = val
    return out


def load_custom(name):
    """Load ~/.synapse/themes/<name>.json -> resolved token dict or None."""
    path = os.path.join(THEME_DIR, name + ".json")
    try:
        with open(path, encoding="utf-8") as f:
            doc = json.load(f)
    except Exception:
        return None
    if not isinstance(doc, dict):
        return None
    base = doc.get("base")
    merged = dict(BUILTINS.get(base, {})) if base else {}
    body = doc.get("overrides", doc.get("theme", {}))
    if not isinstance(body, dict):
        return None
    merged.update(_resolve_refs(body, doc.get("defs")))
    return {t: merged[t] for t in TOKENS if t in merged} or None


def list_themes():
    """Builtin names + custom files found on disk."""
    names = list(BUILTINS)
    try:
        files = sorted(os.listdir(THEME_DIR))
    except Exception:
        files = []
    for fn in files:
        if fn.endswith(".json"):
            stem = fn[:-5]
            if stem not in names:
                names.append(stem)
    return names


def current_name():
    try:
        with open(SELECTION_FILE, encoding="utf-8") as f:
            doc = json.load(f)
        name = (doc.get("theme") or "").strip()
        if name in list_themes():
            return name
    except Exception:
        pass
    return "synapse"


def set_theme(name):
    """Persist selection. Returns (ok, message)."""
    if name not in list_themes():
        return False, "unknown theme '%s' (themes: %s)" % (name, ", ".join(list_themes()))
    try:
        os.makedirs(os.path.dirname(SELECTION_FILE), exist_ok=True)
        with open(SELECTION_FILE, "w", encoding="utf-8") as f:
            json.dump({"theme": name}, f)
    except Exception as e:
        return False, "cannot save theme: %s" % e
    return True, "theme '%s' saved — applies now in TUI, next boot in line mode" % name


def load_theme(name=None):
    """Resolved token->value map (raw values; normalized at style build)."""
    name = name or current_name()
    if name in BUILTINS:
        return dict(BUILTINS[name])
    custom = load_custom(name)
    if custom:
        base = dict(BUILTINS["synapse"])
        base.update(custom)
        return base
    return dict(BUILTINS["synapse"])


def build_style(theme=None):
    """prompt_toolkit Style dict for our UI classes."""
    theme = theme or load_theme()
    c = {t: _norm_color(theme.get(t, "")) for t in TOKENS}

    def fg(key, extra=""):
        return ((c[key] + " " + extra).strip() if c[key] else extra).strip()

    def bg(key, extra=""):
        return (("bg:" + c[key] + " " + extra).strip() if c[key] else extra).strip()

    return {
        "frame.border": fg("border"),
        "frame.title": fg("textMuted"),
        "prompt": ("bold " + c["accent"]).strip() if c["accent"] else "bold",
        "input": fg("text"),
        "bottom-toolbar": bg("panel", fg("text")),
        "leader-hint": bg("panel", fg("textMuted")),
        "completion-menu": bg("panel", fg("text")),
        "completion-menu.completion": bg("panel", fg("text")),
        "completion-menu.completion.current": bg("accent", fg("inverseText", "bold")) if c["accent"] else "reverse",
        "completion-menu.meta": bg("panel", fg("textMuted")),
        "completion-menu.meta.current": bg("accent", fg("inverseText")) if c["accent"] else "reverse",
        "dialog": bg("panel", fg("text")),
        "dialog.body": bg("panel", fg("text")),
        "dialog.frame.border": fg("permission", "bold"),
        "dialog.frame.title": fg("permission", "bold"),
        "button.focused": bg("accent", fg("inverseText", "bold")) if c["accent"] else "reverse",
        "ok": fg("success", "bold"),
        "err": fg("error", "bold"),
        "warn": fg("warning"),
        "echo": fg("accent", "bold"),
        "muted": fg("textMuted"),
        "node": fg("textMuted"),
        "accent": fg("accent", "bold"),
    }

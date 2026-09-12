"""
Behavioral tests for synapse.theme (schemas mirror OpenCode + Claude Code).
"""
import json
import os

import pytest

from synapse import theme as T


def _iso(monkeypatch, tmp_path):
    d = tmp_path / "themes"
    d.mkdir()
    monkeypatch.setattr(T, "THEME_DIR", str(d))
    monkeypatch.setattr(T, "SELECTION_FILE", str(tmp_path / "theme.json"))
    return d


def test_norm_color_forms():
    assert T._norm_color("#D77757") == "#d77757"  # claude brand orange
    assert T._norm_color("#abc") == "#aabbcc"
    assert T._norm_color("rgb(215,119,87)") == "#d77757"
    assert T._norm_color("ansi256(200)") == "ansi200"
    assert T._norm_color("redBright") == "ansibrightred"
    assert T._norm_color("cyan") == "ansicyan"
    assert T._norm_color("none") == ""
    assert T._norm_color("bogus-color") == ""  # typo never breaks UI
    assert T._norm_color("#zzzzzz") == ""
    assert T._norm_color(7) == "ansi7"
    assert T._norm_color(999) == ""


def test_builtins_have_all_tokens():
    for name, th in T.BUILTINS.items():
        assert set(T.TOKENS) <= set(th), name
    assert T.BUILTINS["ember-dark"]["accent"] == "#d77757"
    assert T.BUILTINS["ember-dark"]["success"] == "#4eba65"
    assert T.BUILTINS["ember-dark"]["error"] == "#ff6b80"


def test_set_list_current_roundtrip(monkeypatch, tmp_path):
    _iso(monkeypatch, tmp_path)
    assert T.current_name() == "synapse"  # default, nothing saved
    ok, _ = T.set_theme("ember-dark")
    assert ok and T.current_name() == "ember-dark"
    ok, msg = T.set_theme("nope")
    assert not ok and "synapse" in msg


def test_custom_file_base_overrides_defs(monkeypatch, tmp_path):
    d = _iso(monkeypatch, tmp_path)
    doc = {"name": "mine", "base": "ember-dark",
           "defs": {"myred": "#ff0000"},
           "overrides": {"error": "myred", "accent": "#123456"}}
    with open(os.path.join(str(d), "mine.json"), "w", encoding="utf-8") as f:
        json.dump(doc, f)
    assert "mine" in T.list_themes()
    th = T.load_theme("mine")
    assert th["error"] == "#ff0000" and th["accent"] == "#123456"
    assert th["success"] == "#4eba65"  # falls through to base


def test_broken_custom_file_ignored(monkeypatch, tmp_path):
    d = _iso(monkeypatch, tmp_path)
    with open(os.path.join(str(d), "bad.json"), "w", encoding="utf-8") as f:
        f.write("{not json")
    assert T.load_theme("bad") == T.BUILTINS["synapse"]
    assert T.load_theme("ghost") == T.BUILTINS["synapse"]


def test_build_style_has_ui_classes():
    style = T.build_style(T.BUILTINS["synapse"])
    for cls in ("prompt", "bottom-toolbar", "leader-hint", "echo",
                "completion-menu.completion.current", "dialog.frame.border",
                "button.focused", "ok", "err", "muted", "frame.border"):
        assert cls in style and style[cls], cls

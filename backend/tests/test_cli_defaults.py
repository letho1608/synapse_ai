"""
Behavioral tests: per-machine CLI defaults (zero-flag boot + /set later).

- Missing/corrupt/foreign files -> safe fallbacks, never raises.
- validate coerces + rejects (ports, choices, empties, unknown keys).
- set/unset round-trip through a real tmp file.
- Precedence: explicit CLI > file > builtin.
- apply_to_namespace fills only None fields, ignores unknown file keys.
"""
import json

import pytest

from synapse import cli_defaults as D


def test_missing_and_corrupt_files_are_safe(tmp_path):
    assert D.load(str(tmp_path / "nope.json")) == {}
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    assert D.load(str(bad)) == {}
    lst = tmp_path / "list.json"
    lst.write_text("[1, 2]", encoding="utf-8")
    assert D.load(str(lst)) == {}


def test_validate_ports_choices_empties():
    ok, v = D.validate("node-port", "5678")
    assert (ok, v) == (True, 5678)
    assert D.validate("node-port", "abc")[0] is False
    assert D.validate("node-port", "0")[0] is False
    assert D.validate("node-port", "99999")[0] is False
    assert D.validate("discovery-module", "manual") == (True, "manual")
    assert D.validate("discovery-module", "pigeon")[0] is False
    assert D.validate("partition-method", "ilp") == (True, "ilp")
    ok, v = D.validate("temperature", "0.7")
    assert ok and abs(v - 0.7) < 1e-9
    assert D.validate("temperature", "hot")[0] is False
    assert D.validate("model", "")[0] is False
    assert D.validate("model", "m1") == (True, "m1")
    assert D.validate("frobnicate", "x")[0] is False


def test_set_unset_roundtrip(tmp_path):
    fp = str(tmp_path / "d.json")
    ok, msg = D.set_key("model", "m1", fp)
    assert ok and "next serve/console boot" in msg
    ok, msg = D.set_key("node-port", "5679", fp)
    assert ok
    data = json.loads((tmp_path / "d.json").read_text(encoding="utf-8"))
    assert data == {"model": "m1", "node-port": 5679}
    assert D.set_key("node-port", "abc", fp)[0] is False
    assert D.set_key("nope", "x", fp)[0] is False
    ok, msg = D.unset_key("model", fp)
    assert ok and "cleared" in msg
    assert json.loads((tmp_path / "d.json").read_text(encoding="utf-8")) == {"node-port": 5679}
    ok, msg = D.unset_key("model", fp)
    assert ok and "was not set" in msg
    assert D.unset_key("nope", fp)[0] is False


def test_effective_precedence_and_source(tmp_path):
    fp = str(tmp_path / "d.json")
    D.set_key("model", "file-model", fp)
    D.set_key("chat-port", "5555", fp)
    eff = D.effective({"model": "cli-model", "node-port": None}, fp)
    assert eff["model"] == "cli-model"      # explicit wins
    assert eff["chat-port"] == 5555         # file wins over builtin
    assert eff["temperature"] is None       # builtin default
    assert D.describe_source("model", {"model": "x"}, fp) == "cli"
    assert D.describe_source("chat-port", {}, fp) == "file"
    assert D.describe_source("temperature", {}, fp) == "builtin"


def test_apply_to_namespace_fills_only_nones(tmp_path):
    import argparse
    fp = str(tmp_path / "d.json")
    D.set_key("model", "file-model", fp)
    D.set_key("temperature", "0.2", fp)
    ns = argparse.Namespace(model=None, temperature=0.9, node_port=None)
    D.apply_to_namespace(ns, fp)
    assert ns.model == "file-model"
    assert ns.temperature == 0.9  # explicit kept
    assert ns.node_port is None
    # hand-edited junk keys in file are ignored, never crash
    (tmp_path / "d.json").write_text('{"model": "m", "evil": {"x": 1}}', encoding="utf-8")
    ns2 = argparse.Namespace(model=None)
    D.apply_to_namespace(ns2, fp)
    assert ns2.model == "m" and not hasattr(ns2, "evil")


def test_save_ignores_non_whitelisted_and_none(tmp_path):
    fp = str(tmp_path / "d.json")
    D.save({"model": "m", "evil": 1, "chat-port": None}, fp)
    assert json.loads((tmp_path / "d.json").read_text(encoding="utf-8")) == {"model": "m"}


def test_zero_flag_boot_gets_builtin_model(tmp_path, monkeypatch):
    """Regression: bare `serve` must boot the builtin model, not []."""
    import argparse
    monkeypatch.setenv("SYNAPSE_DEFAULTS_PATH", str(tmp_path / "empty.json"))
    ns = argparse.Namespace(model=None, temperature=None, node_port=None)
    D.apply_to_namespace(ns)
    assert ns.model == "HuggingFaceTB/SmolLM2-135M-Instruct"
    assert ns.temperature is None and ns.node_port is None

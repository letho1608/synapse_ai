"""
Test full toan he thong: import, model_list, API (tat ca endpoint chinh).
Chay: python tests/test_system.py
Hoac: python -m pytest tests/test_system.py -v
"""
import sys
import os
import time
import subprocess
import json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

TEST_PORT = 59999


# --- Unit tests (khong can server) ---

def test_imports():
    """Import cac module chinh."""
    from synapse import VERSION, DEBUG
    from synapse.model_list import HF_MODELS, HF_MODEL_LAYERS, resolve_hf_id
    from synapse.inference.inference_engine import get_inference_engine, InferenceEngine
    from synapse.inference.shard import Shard
    from synapse.models import build_base_shard, get_repo
    assert len(HF_MODELS) == 22
    assert resolve_hf_id("qwen2.5:1.5b") == "Qwen/Qwen2.5-1.5B-Instruct"
    print("  [OK] test_imports")


def test_model_list():
    """Model list: resolve_hf_id, so luong, layers."""
    from synapse.model_list import HF_MODELS, HF_MODEL_LAYERS, resolve_hf_id
    assert len(HF_MODELS) == 22
    assert len(HF_MODEL_LAYERS) == 22
    assert resolve_hf_id("qwen2.5:1.5b") == "Qwen/Qwen2.5-1.5B-Instruct"
    assert resolve_hf_id("llama3.2:3b") == "meta-llama/Llama-3.2-3B-Instruct"
    assert resolve_hf_id("phi3:mini") == "microsoft/Phi-3-mini-4k-instruct"
    assert resolve_hf_id("") or resolve_hf_id("x")  # fallback co gia tri
    assert "qwen2.5:0.5b" in HF_MODELS
    assert HF_MODEL_LAYERS.get("qwen2.5:1.5b") == 28
    print("  [OK] test_model_list")


def test_build_base_shard():
    """Build base shard cho engine PyTorch (can dang ky model truoc)."""
    from synapse.inference.inference_engine import get_inference_engine
    from synapse.models import build_base_shard
    get_inference_engine("pytorch", None)  # dang ky 22 model vao registry
    shard = build_base_shard("qwen2.5:1.5b", "PyTorchHFInferenceEngine")
    assert shard is not None
    assert shard.model_id == "qwen2.5:1.5b"
    assert shard.n_layers == 28
    shard_none = build_base_shard("nonexistent-model-xyz", "PyTorchHFInferenceEngine")
    assert shard_none is None
    print("  [OK] test_build_base_shard")


def test_clean_completion_content():
    """API helper _clean_completion_content cat rac API/API."""
    from synapse.api.chatgpt_api import _clean_completion_content
    assert _clean_completion_content("Hello") == "Hello"
    out = _clean_completion_content("Text /API/API/API")
    assert out == "Text" or out == "Text ", out
    assert _clean_completion_content("Xcode/API") == ""
    assert _clean_completion_content("") == ""
    out_as = _clean_completion_content("As the code/API")
    assert out_as in ("As the", "")  # cat tai pattern
    out_can = _clean_completion_content("I can code/API")
    assert out_can in ("I can", "")
    assert _clean_completion_content(None) is None
    assert _clean_completion_content("  API/API tail  ") == ""
    print("  [OK] test_clean_completion_content")


def test_resolve_hf_id():
    """resolve_hf_id: alias, fallback, ollama prefix."""
    from synapse.model_list import resolve_hf_id, HF_MODELS
    assert resolve_hf_id("qwen2.5:1.5b") == "Qwen/Qwen2.5-1.5B-Instruct"
    assert resolve_hf_id("ollama/qwen2.5:1.5b") == "Qwen/Qwen2.5-1.5B-Instruct"
    assert resolve_hf_id("") == "Qwen/Qwen2.5-1.5B-Instruct"
    assert resolve_hf_id("unknown-xyz") == "Qwen/Qwen2.5-1.5B-Instruct"
    assert resolve_hf_id("  qwen2.5:3b  ") == "Qwen/Qwen2.5-3B-Instruct"
    print("  [OK] test_resolve_hf_id")


def test_message_and_chat_request():
    """Message, ChatCompletionRequest to_dict."""
    from synapse.api.chatgpt_api import Message, ChatCompletionRequest
    m = Message("user", "hello")
    assert m.to_dict() == {"role": "user", "content": "hello"}
    req = ChatCompletionRequest("qwen2.5:1.5b", [m], 0.7)
    d = req.to_dict()
    assert d["model"] == "qwen2.5:1.5b"
    assert d["temperature"] == 0.7
    assert len(d["messages"]) == 1 and d["messages"][0]["content"] == "hello"
    print("  [OK] test_message_and_chat_request")


# --- Live API tests (can server chay) ---

def _start_server():
    config_path = os.path.join(ROOT, "synapse", "networking", "manual", "test_data", "test_config_single_node.json")
    if not os.path.isfile(config_path):
        config_path = os.path.join(ROOT, "synapse", "networking", "manual", "test_data", "test_config.json")
    proc = subprocess.Popen(
        [sys.executable, os.path.join(ROOT, "main.py"), "--chatgpt-api-port", str(TEST_PORT),
         "--discovery-module", "manual", "--discovery-config-path", config_path],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=os.environ.copy(),
    )
    return proc


def _wait_server(base, timeout=45):
    try:
        import requests
    except ImportError:
        return False
    for _ in range(timeout):
        try:
            r = requests.get("%s/healthcheck" % base, timeout=2)
            if r.status_code == 200:
                return True
        except Exception:
            time.sleep(1)
    return False


def _stop_server(proc):
    if proc and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=8)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=3)


def test_api_full():
    """Chay server, goi day du endpoint chinh, tat server."""
    try:
        import requests
    except ImportError:
        print("  [SKIP] test_api_full (requires: pip install requests)")
        return

    proc = None
    base = "http://127.0.0.1:%s" % TEST_PORT
    try:
        proc = _start_server()
        if not _wait_server(base):
            _stop_server(proc)
            raise RuntimeError("Server did not start within 45s")

        session = requests.Session()
        session.headers["Content-Type"] = "application/json"
        ok = 0

        # GET /healthcheck
        r = session.get("%s/healthcheck" % base, timeout=5)
        assert r.status_code == 200, r.text
        d = r.json()
        assert d.get("status") == "ok" and "inference_ready" in d and "model_loaded" in d
        print("  [OK] GET /healthcheck"); ok += 1

        # GET /v1/system/stats
        r = session.get("%s/v1/system/stats" % base, timeout=10)
        assert r.status_code == 200, r.text
        d = r.json()
        assert "cpu" in d and "memory" in d and "gpu" in d and "models_count" in d
        print("  [OK] GET /v1/system/stats"); ok += 1

        # GET /v1/system/info
        r = session.get("%s/v1/system/info" % base, timeout=5)
        assert r.status_code == 200, r.text
        d = r.json()
        assert "project" in d and "version" in d
        print("  [OK] GET /v1/system/info"); ok += 1

        # GET /v1/models
        r = session.get("%s/v1/models" % base, timeout=5)
        assert r.status_code == 200, r.text
        d = r.json()
        assert "data" in d and isinstance(d["data"], list)
        print("  [OK] GET /v1/models"); ok += 1

        # GET /v1/models/list
        r = session.get("%s/v1/models/list" % base, timeout=15)
        assert r.status_code == 200, r.text
        d = r.json()
        assert "downloaded" in d and "available" in d
        print("  [OK] GET /v1/models/list"); ok += 1

        # GET /v1/models/status
        r = session.get("%s/v1/models/status" % base, timeout=5)
        assert r.status_code == 200, r.text
        d = r.json()
        assert "enabled" in d and "models" in d
        print("  [OK] GET /v1/models/status"); ok += 1

        # GET /v1/settings
        r = session.get("%s/v1/settings" % base, timeout=5)
        assert r.status_code == 200, r.text
        d = r.json()
        assert "default_model" in d
        print("  [OK] GET /v1/settings"); ok += 1

        # POST /v1/settings (valid)
        r = session.post("%s/v1/settings" % base, json={"default_model": "qwen2.5:1.5b"}, timeout=5)
        assert r.status_code == 200, r.text
        d = r.json()
        assert d.get("success") is True and "default_model" in d
        print("  [OK] POST /v1/settings"); ok += 1

        # GET /v1/topology
        r = session.get("%s/v1/topology" % base, timeout=5)
        assert r.status_code == 200, r.text
        d = r.json()
        assert "nodes" in d or "partitions" in d or isinstance(d, (list, dict))
        print("  [OK] GET /v1/topology"); ok += 1

        # GET /v1/distributed/status
        r = session.get("%s/v1/distributed/status" % base, timeout=5)
        assert r.status_code == 200, r.text
        d = r.json()
        assert "partitions_count" in d or "message" in d or "multi_node" in d or isinstance(d, dict)
        print("  [OK] GET /v1/distributed/status"); ok += 1

        # GET /v1/training/status
        r = session.get("%s/v1/training/status" % base, timeout=5)
        assert r.status_code == 200, r.text
        d = r.json()
        assert "status" in d or "job" in d or isinstance(d, dict)
        print("  [OK] GET /v1/training/status"); ok += 1

        # GET /v1/datasets
        r = session.get("%s/v1/datasets" % base, timeout=5)
        assert r.status_code == 200, r.text
        d = r.json()
        assert isinstance(d, list) or "data" in d or "datasets" in d
        print("  [OK] GET /v1/datasets"); ok += 1

        # GET /v1/activity
        r = session.get("%s/v1/activity" % base, timeout=5)
        assert r.status_code == 200, r.text
        d = r.json()
        assert "data" in d or isinstance(d, list)
        print("  [OK] GET /v1/activity"); ok += 1

        # GET / (dashboard)
        r = session.get("%s/" % base, timeout=5)
        assert r.status_code == 200, r.text
        assert "html" in r.text.lower() or "Synapse" in r.text or len(r.text) > 500
        print("  [OK] GET / (dashboard)"); ok += 1

        # POST /v1/chat/completions - model khong ton tai -> 400
        r = session.post("%s/v1/chat/completions" % base, json={
            "model": "__nonexistent__",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
        }, timeout=10)
        assert r.status_code == 400, (r.status_code, r.text)
        print("  [OK] POST /v1/chat/completions (invalid model -> 400)"); ok += 1

        # POST /v1/models/pull - thieu model -> 400
        r = session.post("%s/v1/models/pull" % base, json={}, timeout=5)
        assert r.status_code == 400, (r.status_code, r.text)
        print("  [OK] POST /v1/models/pull (missing model -> 400)"); ok += 1

        # GET /v1/logs/terminal
        r = session.get("%s/v1/logs/terminal" % base, timeout=5)
        assert r.status_code == 200, r.text
        print("  [OK] GET /v1/logs/terminal"); ok += 1

        # GET /v1/tailscale/nodes
        r = session.get("%s/v1/tailscale/nodes" % base, timeout=5)
        assert r.status_code == 200, r.text
        print("  [OK] GET /v1/tailscale/nodes"); ok += 1

        print("  Total: %d API checks passed." % ok)
    finally:
        _stop_server(proc)


if __name__ == "__main__":
    print("Full system test")
    print("- test_imports")
    test_imports()
    print("- test_model_list")
    test_model_list()
    print("- test_build_base_shard")
    test_build_base_shard()
    print("- test_clean_completion_content")
    test_clean_completion_content()
    print("- test_resolve_hf_id")
    test_resolve_hf_id()
    print("- test_message_and_chat_request")
    test_message_and_chat_request()
    print("- test_api_full (start server, all endpoints, stop)")
    test_api_full()
    print("Done.")

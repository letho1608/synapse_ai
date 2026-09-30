"""
SSE streaming must reproduce the exact answer text.

Found 2026-09-30 (live failover run): the stream decoded each token alone
and stripped every piece, so a streamed answer came out as
"Herearefivefruits:1.Banana2.Apple" while the non-stream answer was
"Here are five fruits:\\n\\n1. Banana\\n2. Apple". StreamDetokenizer emits
the new suffix of the whole decoded text instead.
"""
import os

import pytest

from synapse.api.chatgpt_api import StreamDetokenizer, _clean_completion_content

MODELS = ["peft-internal-testing/tiny-random-GPT2LMHeadModel", "HuggingFaceTB/SmolLM2-135M-Instruct"]


def _tok(model_id):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    except Exception:
        pytest.skip(f"{model_id} not cached")


@pytest.mark.parametrize("model_id", MODELS)
@pytest.mark.parametrize("text", [
    "Here are five fruits:\n\n1. Banana\n2. Apple\n3. Orange",
    "  leading spaces and trailing newline\n",
    "Thủ đô của Việt Nam là Hà Nội — 1000 năm văn hiến. 🍌",
])
def test_streamed_deltas_concatenate_to_the_answer(model_id, text):
    tok = _tok(model_id)
    ids = tok.encode(text, add_special_tokens=False)
    d = StreamDetokenizer(tok)
    streamed = "".join(d.push([t]) for t in ids)
    full = tok.decode(ids, skip_special_tokens=True)
    # Same text as the non-stream answer (which strips both ends once).
    assert streamed.strip() == _clean_completion_content(full)
    assert "�" not in streamed
    if "\n" in text.strip():
        assert "\n" in streamed and " " in streamed


def test_batches_of_tokens_work_too():
    tok = _tok(MODELS[1])
    ids = tok.encode("One two three four five six", add_special_tokens=False)
    d = StreamDetokenizer(tok)
    out = d.push(ids[:2]) + d.push(ids[2:5]) + d.push(ids[5:])
    assert out == "One two three four five six"

"""
Test Qwen 2.5 0.5B with local Node (no P2P).
Usage: python tests/test_qwen_inference.py --model Qwen/Qwen2.5-0.5B-Instruct --prompt "..." --tokens 20
"""
import asyncio
import time
import uuid
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import synapse
synapse.DEBUG = 1

from synapse.inference.inference_engine import get_inference_engine
from synapse.models import build_full_shard, get_repo
from synapse.inference.tokenizers import resolve_tokenizer
from synapse.loading import ShardDownloader
from synapse.topology.topology import Topology
from synapse.topology.device_capabilities import device_capabilities
from synapse.routing.event_router import EventRouter


class MinimalLibp2p:
    def __init__(self): self.id = "local-tester"
    async def start(self): pass
    async def stop(self): pass


class MinimalRouter:
    def __init__(self):
        self._handlers = {}

    def subscribe(self, topic, handler):
        self._handlers[topic] = handler

    def unsubscribe(self, topic):
        self._handlers.pop(topic, None)

    async def publish(self, topic, data, origin=None):
        pass

    async def start(self): pass
    async def stop(self): pass


async def run_inference(model_name: str, prompt: str, max_tokens: int):
    print(f"\n Model: {model_name}")
    print(f" Prompt: {prompt}")
    print("-" * 50)

    # 1. Register models + get engine (MUST call before build_full_shard)
    inference_engine = get_inference_engine("pytorch", None)
    shard_downloader = inference_engine.shard_downloader

    # 2. EventRouter
    er = EventRouter()
    await er.start()

    from synapse.orchestration.node import Node
    node = Node(
        "local-tester",
        er,
        MinimalLibp2p(),
        inference_engine,
        shard_downloader,
        max_generate_tokens=max_tokens
    )

    # Override device check - use dummy capabilities for testing
    node.device_capabilities = device_capabilities.__wrapped__() if hasattr(device_capabilities, '__wrapped__') else None
    if node.device_capabilities is None:
        from synapse.topology.device_capabilities import UNKNOWN_DEVICE_CAPABILITIES
        node.device_capabilities = UNKNOWN_DEVICE_CAPABILITIES

    node.topology = Topology()
    node.topology.update_node(node.id, node.device_capabilities)

    inference_class = node.inference_engine.__class__.__name__
    shard = build_full_shard(model_name, inference_class)
    if not shard:
        print(f"Unsupported model: {model_name}")
        return False

    print(f" Model: {shard.model_id}, layers={shard.n_layers}")

    tokenizer = await resolve_tokenizer(get_repo(shard.model_id, inference_class))

    # Load model
    print(f"\n[1/2] Loading model...")
    load_start = time.perf_counter()
    await node.inference_engine.ensure_shard(node.get_current_shard(shard))
    load_end = time.perf_counter()
    print(f" Loaded in {load_end - load_start:.2f}s")

    # Inference
    print(f"\n[2/2] Generating...")
    request_id = str(uuid.uuid4())
    tokens = []
    done_event = asyncio.Event()
    first_token_time = None

    def on_token(_req_id, _tokens, _is_finished):
        nonlocal first_token_time
        if _req_id != request_id:
            return
        if _tokens:
            if first_token_time is None:
                first_token_time = time.perf_counter()
            tokens.extend(_tokens)
            print(tokenizer.decode(_tokens), end="", flush=True)
        if _is_finished:
            done_event.set()

    node._on_token.register("qwen_test").on_next(on_token)

    try:
        start_time = time.perf_counter()
        await node.process_prompt(shard, prompt, request_id=request_id)
        await asyncio.wait_for(done_event.wait(), timeout=120)
        end_time = time.perf_counter()

        total_time = end_time - start_time
        ttft = (first_token_time - start_time) if first_token_time else 0
        gen_time = end_time - first_token_time if first_token_time and len(tokens) > 1 else 0
        tps = (len(tokens) - 1) / gen_time if gen_time > 0 else 0

        print(f"\n\n--- RESULTS ---")
        print(f" Load time:  {load_end - load_start:.2f}s")
        print(f" Total time: {total_time:.2f}s")
        print(f" TTFT:       {ttft:.3f}s")
        print(f" TPS:        {tps:.2f} tokens/s")
        print(f" Total tokens: {len(tokens)}")
        full_text = tokenizer.decode(tokens, skip_special_tokens=True) if tokens else ""
        if full_text:
            print(f" Output:\n{full_text}")

        return True
    except asyncio.TimeoutError:
        print("\n TIMEOUT")
        return False
    except Exception as e:
        print(f"\n ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        node._on_token.deregister("qwen_test")
        await er.stop()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="qwen2.5:0.5b")
    parser.add_argument("--prompt", type=str, default="Xin chào, bạn tên gì?")
    parser.add_argument("--tokens", type=int, default=20)
    args = parser.parse_args()

    from synapse.model_list import resolve_hf_id
    model_key = args.model
    # resolve short key like "qwen2.5:0.5b" -> full HuggingFace ID
    hf_id = resolve_hf_id(model_key)
    print(f"Using model: {hf_id}")

    ok = asyncio.run(run_inference(hf_id, args.prompt, args.tokens))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
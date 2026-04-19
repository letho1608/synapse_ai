
import asyncio
import time
import uuid
import argparse
import sys
import os
import json
import re
import numpy as np

# Thêm thư mục gốc vào path để import synapse khi chạy từ trong thư mục tests
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Bật debug hệ thống ngay từ đầu
import synapse
synapse.DEBUG = 1

from synapse.orchestration.node import Node
from synapse.inference.inference_engine import get_inference_engine
from synapse.inference.shard import Shard
from synapse.topology.lacp_partitioning import LACPPartitioningStrategy
from synapse.models import build_full_shard, get_repo
from synapse.inference.tokenizers import resolve_tokenizer
from synapse.loading import ShardDownloader
from synapse.networking.discovery import Discovery
from synapse.topology.topology import Topology
from synapse.topology.device_capabilities import device_capabilities

class MockDiscovery(Discovery):
    def __init__(self, node_id):
        self.node_id = node_id
    async def start(self): pass
    async def stop(self): pass
    async def discover_peers(self, wait_for_peers=0): return []


def _looks_vietnamese(text: str) -> bool:
    if not text:
        return False
    lowered = text.lower()
    # Ưu tiên nhận diện tiếng Việt có dấu.
    if re.search(r"[ăâđêôơưáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ]", lowered):
        return True
    common_vi_markers = [" toi ", " la ", " ban ", " tro ly ", " tieng viet ", " xin chao "]
    padded = f" {re.sub(r'[^a-z0-9]+', ' ', lowered).strip()} "
    score = sum(1 for marker in common_vi_markers if marker in padded)
    return score >= 2


def _build_vi_prompt(tokenizer, user_prompt: str, strict: bool = False) -> str:
    if strict:
        vi_system_instruction = (
            "Bạn là trợ lý AI. BẮT BUỘC chỉ trả lời bằng tiếng Việt có dấu. "
            "Không dùng tiếng Anh, không viết tắt, không ký tự rác. "
            "Trả lời 1-2 câu. Câu đầu phải bắt đầu bằng: 'Tôi là'."
        )
    else:
        vi_system_instruction = (
            "Bạn là trợ lý AI và luôn trả lời hoàn toàn bằng tiếng Việt tự nhiên. "
            "Không dùng tiếng Anh hoặc ký tự rác."
        )

    # Dùng prompt thuần để ổn định tiếng Việt và hạn chế echo nhãn hội thoại.
    return (
        f"{vi_system_instruction}\n"
        "Chỉ trả lời nội dung, không nhắc lại đề bài, không in lại 'Người dùng:' hoặc 'Trợ lý:'.\n\n"
        f"Người dùng: {user_prompt}\n"
        "Trợ lý: Tôi là một trợ lý AI"
    )

async def run_benchmark(model_name, prompt, max_tokens=50):
    print(f"\n🚀 Bắt đầu Benchmark cho model: {model_name}")
    print(f"📝 Prompt: \"{prompt}\"")
    print("-" * 50)

    # 1. Khởi tạo Engine & Node (Chế độ tối giản)
    inference_engine = get_inference_engine("pytorch", None)
    shard_downloader = inference_engine.shard_downloader
    
    # Discovery giả lập
    discovery = MockDiscovery("local-tester") 
    partitioning_strategy = LACPPartitioningStrategy()
    
    node = Node(
        "local-tester",
        None, 
        inference_engine,
        discovery,
        shard_downloader,
        partitioning_strategy=partitioning_strategy,
        max_generate_tokens=max_tokens
    )
    # Nhiệt độ vừa phải để giữ ổn định nhưng không bị cứng đầu ra.
    node.default_sample_temperature = 0.6

    # Lấy thông tin phần cứng
    node.device_capabilities = await device_capabilities()
    node.topology = Topology()
    node.topology.update_node(node.id, node.device_capabilities)

    inference_class = node.inference_engine.__class__.__name__
    shard = build_full_shard(model_name, inference_class)
    if not shard:
        print(f"❌ Không hỗ trợ hoặc không tìm thấy cấu hình cho model: {model_name}")
        return
    
    print(f" Nhận diện mô hình: {shard.model_id} với {shard.n_layers} tầng.")
    if not shard:
        print(f"❌ Không hỗ trợ model: {model_name}")
        return

    tokenizer = await resolve_tokenizer(get_repo(shard.model_id, inference_class))
    
    # 2. Chuẩn bị Prompt gốc
    user_prompt = prompt

    print(f"\n[1/2] Đang nạp mô hình (LACP logic call)...")
    load_start = time.perf_counter()
    await node.inference_engine.ensure_shard(node.get_current_shard(shard))
    load_end = time.perf_counter()
    print(f" Đã nạp xong trong: {load_end - load_start:.2f}s")

    print(f"\n[2/2] Đang sinh văn bản (Inference)...")
    print("-" * 20)
    
    final_text = ""
    final_tokens = []
    final_ttft = 0.0
    final_tps = 0.0
    final_total_time = 0.0

    for attempt in (1, 2):
        strict_mode = attempt == 2
        prompt_to_run = _build_vi_prompt(tokenizer, user_prompt, strict=strict_mode)
        request_id = str(uuid.uuid4())
        tokens = []
        start_time = time.perf_counter()
        first_token_time = None
        done_event = asyncio.Event()
        collector_key = f"bench_collector_{attempt}"

        def on_token_received(_req_id, _tokens, _is_finished):
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

        node._on_token.register(collector_key).on_next(on_token_received)

        try:
            await node.process_prompt(shard, prompt_to_run, request_id=request_id)
            try:
                await asyncio.wait_for(done_event.wait(), timeout=120)
            except asyncio.TimeoutError:
                print(f"\n⚠️ Timeout 120 giây chờ kết thúc sinh token")

            end_time = time.perf_counter()
            total_time = end_time - start_time
            ttft = (first_token_time - start_time) if first_token_time else 0
            generation_time = end_time - first_token_time if first_token_time and len(tokens) > 1 else 0
            tps = (len(tokens) - 1) / generation_time if generation_time > 0 else 0

            full_text = tokenizer.decode(tokens, skip_special_tokens=True) if tokens else ""
            final_text = full_text
            final_tokens = tokens
            final_ttft = ttft
            final_tps = tps
            final_total_time = total_time

            if full_text and _looks_vietnamese(full_text):
                break
            if attempt == 1:
                print("\n⚠️ Lượt đầu chưa ra tiếng Việt ổn định, thử lại với ràng buộc chặt hơn...")
                print("-" * 20)
        except Exception as e:
            print(f"\n❌ Lỗi trong quá trình benchmark: {e}")
            import traceback
            traceback.print_exc()
            break
        finally:
            node._on_token.deregister(collector_key)

    print(f"\n" + "-" * 20)
    print(f"📊 KẾT QUẢ BENCHMARK:")
    print(f"⏱️  Thời gian nạp mô hình: {load_end - load_start:.2f}s")
    print(f"⏱️  Thời gian phản hồi đầu (TTFT): {final_ttft:.4f}s")
    print(f"⚡ Tốc độ sinh chữ (TPS): {final_tps:.2f} tokens/s")
    print(f"🕒 Tổng thời gian: {final_total_time:.2f}s")
    print(f"🪙  Tổng số token: {len(final_tokens)}")
    if final_text:
        print(f"\n📝 Nội dung đầy đủ:\n{final_text}")
    print("-" * 50)

if __name__ == "__main__":
    from synapse.model_list import resolve_hf_id

    def _default_model_from_settings() -> str:
        settings_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "synapse",
            "config",
            "settings.json",
        )
        try:
            with open(settings_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            model = (data.get("default_model") or "").strip()
            if model:
                return model
        except Exception:
            pass
        return "Gensyn/Qwen2.5-0.5B-Instruct"

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None, help="Tên model (mặc định lấy từ settings.json)")
    parser.add_argument("--prompt", type=str, default="Hãy giới thiệu ngắn gọn về bạn bằng tiếng Việt.")
    parser.add_argument("--tokens", type=int, default=30)
    args = parser.parse_args()

    model_to_run = (args.model or _default_model_from_settings()).strip()
    print(f"Sử dụng model: {model_to_run}")

    final_model_id = resolve_hf_id(model_to_run)
    asyncio.run(run_benchmark(final_model_id, args.prompt, args.tokens))

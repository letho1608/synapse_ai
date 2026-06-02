"""
Benchmark: REAL distributed inference on 3 nodes.
Uses PyTorch Qwen2.5 (4 layers, 421M params) with actual forward passes.
Real TCP transport via P2P EventRouter.
"""
import asyncio, sys, os, time, json, torch
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from synapse.routing.event_router import EventRouter
from synapse.routing.libp2p_node import Libp2pNode
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding

NODE_IDS = ["NODE-A", "NODE-B", "NODE-C"]
PORTS = [5981, 5982, 5983]
N_LAYERS = 4
TOTAL_LAYERS = N_LAYERS
GEN_TOKENS = 3

LAYER_RANGES = [(0, 1), (2, 2), (3, 3)]

class BenchNode:
    def __init__(self, nid, port, layer_s, layer_e, model):
        self.id = nid; self.port = port
        self.layer_s = layer_s; self.layer_e = layer_e
        self.n_layers = layer_e - layer_s + 1
        self.model = model
        self.router = EventRouter()
        self.libp2p = Libp2pNode(nid, port, self.router)
        self.fwd_times = []; self.n_fwd = 0

    async def start(self):
        await self.router.start()
        asyncio.create_task(self.libp2p.start())

    async def connect_to(self, pid, addr):
        return await self.libp2p.connect_to_peer(pid, addr)

    async def stop(self):
        await self.libp2p.stop(); await self.router.stop()

def extract_layer(model, idx):
    return model.model.layers[idx]

async def bench():
    print("=" * 65)
    print("BENCHMARK THUC TE - 3 NODE PHAN TAN")
    print("=" * 65)

    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    total_layers = 4
    cfg = AutoConfig.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct",
        torch_dtype=torch.float32, num_hidden_layers=total_layers)
    model = AutoModelForCausalLM.from_config(cfg)
    model.eval()

    embed = model.model.embed_tokens
    norm = model.model.norm
    lm_head = model.lm_head
    rotary = Qwen2RotaryEmbedding(cfg)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Model: Qwen2.5 ({total_layers} layers, {n_params:.0f}M params)")
    print(f"  hidden={cfg.hidden_size}, heads={cfg.num_attention_heads}")
    print(f"  vocab={cfg.vocab_size}")
    print(f"  Device: CPU (fp32)")
    print(f"  Khoi tao: {time.perf_counter()-t0:.1f}s")

    nodes = []
    for i in range(3):
        sr, er = LAYER_RANGES[i]
        n = BenchNode(NODE_IDS[i], PORTS[i], sr, er, model)
        nodes.append(n)
        layers_str = ",".join(str(l) for l in range(sr, er+1))
        print(f"  {n.id:<8}: layers [{layers_str}] ({n.n_layers})")

    for n in nodes: await n.start()
    await asyncio.sleep(0.3)
    for n in nodes:
        for o in nodes:
            if n.id != o.id: await n.connect_to(o.id, f"127.0.0.1:{o.port}")
    await asyncio.sleep(0.3)
    for n in nodes:
        for o in nodes:
            if n.id != o.id and not n.libp2p.is_peer_connected(o.id):
                print(f"  FAIL: {n.id} not connected to {o.id}"); return
    print("  [MESH] 3-node full mesh: OK")

    result_queue = asyncio.Queue()
    log_path = os.path.join(os.path.dirname(__file__), "bench_log.txt")
    LOG = open(log_path, "w")

    async def handle_a(event):
        data = event.data; step = data["step"]
        hidden = torch.tensor(data["hidden"])
        sr, er = LAYER_RANGES[0]
        seq = hidden.shape[1]
        pos_ids = torch.arange(seq).unsqueeze(0)
        cos, sin = rotary(hidden, pos_ids)
        pos_emb = (cos, sin)
        t_fwd = time.perf_counter()
        with torch.no_grad():
            for li in range(sr, er + 1):
                hidden = extract_layer(model, li)(hidden, use_cache=False, position_embeddings=pos_emb)[0]
        t_fwd = time.perf_counter() - t_fwd
        nodes[0].fwd_times.append(t_fwd); nodes[0].n_fwd += 1
        LOG.write(f"[A] step={step} layers {sr}-{er} {t_fwd*1000:.0f}ms\n"); LOG.flush()
        await nodes[0].router.publish(f"synapse/inference/tensor/NODE-B",
            {"step": step, "t_start": data["t_start"], "hidden": hidden.tolist()})

    async def handle_b(event):
        data = event.data; step = data["step"]
        hidden = torch.tensor(data["hidden"])
        seq = hidden.shape[1]
        pos_ids = torch.arange(seq).unsqueeze(0)
        cos, sin = rotary(hidden, pos_ids)
        pos_emb = (cos, sin)
        t_fwd = time.perf_counter()
        with torch.no_grad():
            hidden = extract_layer(model, 2)(hidden, use_cache=False, position_embeddings=pos_emb)[0]
        t_fwd = time.perf_counter() - t_fwd
        nodes[1].fwd_times.append(t_fwd); nodes[1].n_fwd += 1
        LOG.write(f"[B] step={step} layer 2 {t_fwd*1000:.0f}ms\n"); LOG.flush()
        await nodes[1].router.publish(f"synapse/inference/tensor/NODE-C",
            {"step": step, "t_start": data["t_start"], "hidden": hidden.tolist()})

    async def handle_c(event):
        data = event.data; step = data["step"]
        hidden = torch.tensor(data["hidden"])
        seq = hidden.shape[1]
        pos_ids = torch.arange(seq).unsqueeze(0)
        cos, sin = rotary(hidden, pos_ids)
        pos_emb = (cos, sin)
        t_fwd = time.perf_counter()
        with torch.no_grad():
            hidden = extract_layer(model, 3)(hidden, use_cache=False, position_embeddings=pos_emb)[0]
            h = norm(hidden); logits = lm_head(h)
        token_id = logits[0, -1].argmax().item()
        t_fwd = time.perf_counter() - t_fwd
        t_end = time.perf_counter()
        nodes[2].fwd_times.append(t_fwd); nodes[2].n_fwd += 1
        LOG.write(f"[C] step={step} layer 3+head {t_fwd*1000:.0f}ms token={token_id}\n"); LOG.flush()
        await nodes[0].router.publish("synapse/inference/result/bench",
            {"step": step, "token": token_id, "t_start": data["t_start"], "t_end": t_end})

    nodes[0].router.subscribe("synapse/inference/tensor/NODE-A", handle_a)
    nodes[1].router.subscribe("synapse/inference/tensor/NODE-B", handle_b)
    nodes[2].router.subscribe("synapse/inference/tensor/NODE-C", handle_c)

    def on_result(event):
        result_queue.put_nowait(event.data)
    nodes[0].router.subscribe("synapse/inference/result/bench", on_result)

    await asyncio.sleep(0.2)

    prompt = "The future of AI is"
    inputs = tok(prompt, return_tensors="pt")
    prompt_ids = inputs.input_ids
    all_ids = prompt_ids.clone()
    print(f"\n  Prompt: '{prompt}' ({prompt_ids.shape[1]} tokens)")
    print(f"  Generating {GEN_TOKENS} tokens...\n")

    completed = []
    for step in range(GEN_TOKENS):
        with torch.no_grad():
            h = embed(all_ids)
        t_start = time.perf_counter()
        LOG.write(f"[MAIN] step={step} seq_len={all_ids.shape[1]} publishing\n"); LOG.flush()
        await nodes[0].router.publish("synapse/inference/tensor/NODE-A", {
            "step": step, "t_start": t_start, "hidden": h.tolist()})

        try:
            result = await asyncio.wait_for(result_queue.get(), timeout=120.0)
        except asyncio.TimeoutError:
            LOG.write(f"[MAIN] TIMEOUT step={step}\n"); LOG.flush()
            print(f"  [TIMEOUT] step={step}")
            break

        token_id = result["token"]
        latency = (result["t_end"] - result["t_start"]) * 1000
        print(f"  Token {step}: id={token_id} latency={latency:.0f}ms")
        all_ids = torch.cat([all_ids, torch.tensor([[token_id]])], dim=1)
        completed.append(result)

    total_t = time.perf_counter() - t0

    print(f"\n{'='*65}")
    print(f"KET QUA")
    print(f"{'='*65}")
    print(f"  Tong token sinh:  {len(completed)}")
    if completed:
        latencies = [(t["t_end"] - t["t_start"]) * 1000 for t in completed]
        wall_t = completed[-1]["t_end"] - completed[0]["t_start"]
        print(f"  Latency/token:    {np.mean(latencies):.0f}ms (avg)")
        print(f"  Min/Max:          {min(latencies):.0f}/{max(latencies):.0f}ms")
        print(f"  Generation time:  {wall_t:.2f}s")
        print(f"  Throughput:       {len(completed)/wall_t:.2f} tok/s")
    print()
    for i in range(3):
        n = nodes[i]
        avg_fwd = np.mean(n.fwd_times) * 1000 if n.fwd_times else 0
        lps = avg_fwd / n.n_layers if n.n_layers > 0 else 0
        print(f"  {n.id:<8}: {n.n_layers} layer(s) "
              f"forward_avg={avg_fwd:.0f}ms ({lps:.0f}ms/layer) "
              f"count={n.n_fwd}")

    if completed:
        total_compute = sum(sum(n.fwd_times) for n in nodes)
        wall_t = completed[-1]["t_end"] - completed[0]["t_start"]
        print(f"\n  Tong compute:     {total_compute:.2f}s")
        print(f"  Wall time:        {wall_t:.2f}s")
        print(f"  Overhead network: {wall_t - total_compute:.2f}s")
        print(f"  Hieu suat:        {total_compute/wall_t*100:.1f}%")

    for n in nodes: await n.stop()
    LOG.close()
    print("\n  [DONE]")
    print(f"\n  Log: bench_log.txt")

asyncio.run(bench())

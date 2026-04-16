import argparse
import asyncio
import atexit
import signal
import json
import os
import time
import traceback
import uuid
import numpy as np
from tqdm import tqdm
from synapse.train.dataset import load_dataset, iterate_batches
from synapse.networking.manual.manual_discovery import ManualDiscovery
from synapse.orchestration.node import Node
from synapse.networking.grpc.grpc_server import GRPCServer
from synapse.networking.udp.udp_discovery import UDPDiscovery
from synapse.networking.tailscale.tailscale_discovery import TailscaleDiscovery
from synapse.networking.grpc.grpc_peer_handle import GRPCPeerHandle
from synapse.topology.lacp_partitioning import LACPPartitioningStrategy
from synapse.api import ChatGPTAPI
# REFACTORED IMPORTS
from synapse.loading import ShardDownloader, RepoProgressEvent, create_local_model_loader, get_models_dir
from synapse.helpers import find_available_port, DEBUG, get_system_info, get_or_create_node_id, get_all_ip_addresses_and_interfaces, terminal_link, shutdown, check_model_hardware_fit
from synapse.inference.shard import Shard
from synapse.inference.inference_engine import get_inference_engine
from synapse.inference.tokenizers import resolve_tokenizer
from synapse.models import build_base_shard, get_repo
from synapse.viz.topology_viz import TopologyViz
import concurrent.futures
import psutil

os.environ["GRPC_VERBOSITY"] = "error"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# Configure event loop for Windows
def configure_event_loop():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.set_default_executor(concurrent.futures.ThreadPoolExecutor(max_workers=min(32, (os.cpu_count() or 1) * 4)))
    return loop

# parse args
parser = argparse.ArgumentParser(description="Initialize GRPC Discovery")
parser.add_argument("command", nargs="?", choices=["run", "eval", "train"], help="Command to run")
parser.add_argument("model_name", nargs="?", help="Model name to run")
parser.add_argument("--default-model", type=str, default=None, help="Default model")
parser.add_argument("--iters", type=int, default=100, help="Training iterations")
parser.add_argument("--save-every", type=int, default=5, help="Save the model every N iterations.")
parser.add_argument("--data", type=str, default="synapse/train/data/lora", help="Directory where training data lives")
parser.add_argument("--batch-size", type=int, default=1, help="Minibatch size.")
parser.add_argument("--resume-checkpoint", type=str, default=None, help="Path to a custom checkpoint to load")
parser.add_argument("--save-checkpoint-dir", type=str, default="checkpoints", help="Path to a folder where checkpoints are stored")
parser.add_argument("--node-id", type=str, default=None, help="Node ID")
parser.add_argument("--node-host", type=str, default="0.0.0.0", help="Node host")
parser.add_argument("--node-port", type=int, default=None, help="Node port")
parser.add_argument("--models-seed-dir", type=str, default=None, help="Model seed directory")
parser.add_argument("--listen-port", type=int, default=5678, help="Listening port for discovery")
parser.add_argument("--download-quick-check", action="store_true", help="Quick check local path for model shards download")
parser.add_argument("--max-parallel-downloads", type=int, default=8, help="Max parallel downloads for model shards download")
parser.add_argument("--broadcast-port", type=int, default=5678, help="Broadcast port for discovery")
parser.add_argument("--discovery-module", type=str, choices=["udp", "tailscale", "manual"], default="tailscale", help="Discovery module to use")
parser.add_argument("--discovery-timeout", type=int, default=30, help="Discovery timeout in seconds")
parser.add_argument("--discovery-config-path", type=str, default=None, help="Path to discovery config json file")
parser.add_argument("--wait-for-peers", type=int, default=0, help="Number of peers to wait to connect to before starting")
parser.add_argument("--chatgpt-api-port", type=int, default=52415, help="ChatGPT API port")
parser.add_argument("--chatgpt-api-response-timeout", type=int, default=900, help="ChatGPT API response timeout in seconds")
parser.add_argument("--max-generate-tokens", type=int, default=10000, help="Max tokens to generate in each request")
parser.add_argument("--inference-engine", type=str, default="pytorch", help="Inference engine (pytorch)")
parser.add_argument("--disable-tui", action=argparse.BooleanOptionalAction, default=True, help="Disable TUI (default: on)")
parser.add_argument("--run-model", type=str, help="Specify a model to run directly")
parser.add_argument("--prompt", type=str, help="Prompt for the model when using --run-model", default="")
parser.add_argument("--default-temp", type=float, help="Default token sampling temperature", default=0.0)
# Tailscale API key: Hardcode trực tiếp ở đây (thay giá trị bên dưới)
# Có thể override bằng command line argument --tailscale-api-key nếu cần
TAILSCALE_API_KEY_DEFAULT = "tskey-api-kZkPWrVyM311CNTRL-91psFey7AHYgSzXpLr7GJYKZm43RZkXVD"  # TODO: Thay bằng API key thật của bạn
parser.add_argument("--tailscale-api-key", type=str, default=TAILSCALE_API_KEY_DEFAULT, help="Tailscale API key (mặc định lấy từ hardcode trong code)")
# Tailnet name: Hardcode trực tiếp ở đây (thay giá trị bên dưới)
# Có thể override bằng command line argument --tailnet-name nếu cần
TAILNET_NAME_DEFAULT = "testdoki925@gmail.com"  # TODO: Thay bằng tailnet name thật của bạn
parser.add_argument("--tailnet-name", type=str, default=TAILNET_NAME_DEFAULT, help="Tailnet name (mặc định lấy từ hardcode trong code)")
parser.add_argument("--node-id-filter", type=str, default=None, help="Comma separated list of allowed node IDs (only for UDP and Tailscale discovery)")
parser.add_argument("--interface-type-filter", type=str, default=None, help="Comma separated list of allowed interface types (only for UDP discovery)")
parser.add_argument("--system-prompt", type=str, default="Bạn là AI thuộc hệ thống Synapse AI Server do đội ngũ NCKH 2026 của trường Đại học Mỏ-Địa chất phát triển.", help="System prompt for the ChatGPT API")
args = parser.parse_args()

system_info = get_system_info()

# Engine mặc định: PyTorch + Hugging Face (danh sách model trong model_list, download + web UI)
inference_engine_name = args.inference_engine if hasattr(args, "inference_engine") and getattr(args, "inference_engine", None) else "pytorch"
inference_engine = get_inference_engine(inference_engine_name, None)
shard_downloader: ShardDownloader = inference_engine.shard_downloader

if args.node_port is None:
  args.node_port = find_available_port(args.node_host)
  if DEBUG >= 1: print(f"Using available port: {args.node_port}")

args.node_id = args.node_id or get_or_create_node_id()
chatgpt_api_endpoints = [f"http://{ip}:{args.chatgpt_api_port}/v1/chat/completions" for ip, _ in get_all_ip_addresses_and_interfaces()]
web_chat_urls = [f"http://{ip}:{args.chatgpt_api_port}" for ip, _ in get_all_ip_addresses_and_interfaces()]

# Convert node-id-filter and interface-type-filter to lists if provided
allowed_node_ids = args.node_id_filter.split(',') if args.node_id_filter else None
allowed_interface_types = args.interface_type_filter.split(',') if args.interface_type_filter else None

if args.discovery_module == "udp":
  discovery = UDPDiscovery(
    args.node_id,
    args.node_port,
    args.listen_port,
    args.broadcast_port,
    lambda peer_id, address, description, device_capabilities: GRPCPeerHandle(peer_id, address, description, device_capabilities),
    discovery_timeout=args.discovery_timeout,
    allowed_node_ids=allowed_node_ids,
    allowed_interface_types=allowed_interface_types
  )
elif args.discovery_module == "tailscale":
  # Sử dụng giá trị từ command line argument (nếu có), nếu không thì dùng giá trị hardcode
  tailscale_api_key = args.tailscale_api_key
  tailnet_name = args.tailnet_name
  
  if not tailscale_api_key or tailscale_api_key == "tskey-api-xxxxx-xxxxx":
    raise ValueError("Tailscale API key is required. Vui lòng sửa giá trị TAILSCALE_API_KEY_DEFAULT trong file main.py hoặc dùng --tailscale-api-key")
  if not tailnet_name or tailnet_name == "your-org":
    raise ValueError("Tailnet name is required. Vui lòng sửa giá trị TAILNET_NAME_DEFAULT trong file main.py hoặc dùng --tailnet-name")
  
  discovery = TailscaleDiscovery(
    args.node_id,
    args.node_port,
    lambda peer_id, address, description, device_capabilities: GRPCPeerHandle(peer_id, address, description, device_capabilities),
    discovery_timeout=args.discovery_timeout,
    tailscale_api_key=tailscale_api_key,
    tailnet=tailnet_name,
    allowed_node_ids=allowed_node_ids
  )
elif args.discovery_module == "manual":
  if not args.discovery_config_path:
    raise ValueError(f"--discovery-config-path is required when using manual discovery. Please provide a path to a config json file.")
  discovery = ManualDiscovery(args.discovery_config_path, args.node_id, create_peer_handle=lambda peer_id, address, description, device_capabilities: GRPCPeerHandle(peer_id, address, description, device_capabilities))
topology_viz = TopologyViz(chatgpt_api_endpoints=chatgpt_api_endpoints, web_chat_urls=web_chat_urls) if not args.disable_tui else None
# Use LACP partitioning strategy
partitioning_strategy = LACPPartitioningStrategy()
if DEBUG >= 1: print("Using LACP partitioning strategy")

node = Node(
    args.node_id,
    None,
    inference_engine,
    discovery,
    shard_downloader,
    partitioning_strategy=partitioning_strategy,
    max_generate_tokens=args.max_generate_tokens,
    topology_viz=topology_viz,
    default_sample_temperature=args.default_temp
)
server = GRPCServer(node, args.node_host, args.node_port)
node.server = server
default_model = args.default_model or "qwen2.5:1.5b"
api = ChatGPTAPI(
  node,
  node.inference_engine.__class__.__name__,
  response_timeout=args.chatgpt_api_response_timeout,
  on_chat_completion_request=lambda req_id, __, prompt: topology_viz.update_prompt(req_id, prompt) if topology_viz else None,
  default_model=default_model,
  system_prompt=args.system_prompt,
  chatgpt_api_port=args.chatgpt_api_port,
)
buffered_token_output = {}
def update_topology_viz(req_id, tokens, is_finished):
  if not topology_viz: return
  if not node.inference_engine.shard: return
  if node.inference_engine.shard.model_id == 'stable-diffusion-2-1-base': return
  # CRITICAL FIX: Always update buffer, even with empty tokens
  if req_id in buffered_token_output: 
    buffered_token_output[req_id].extend(tokens)
  else: 
    buffered_token_output[req_id] = list(tokens) if tokens else []
  
  # CRITICAL FIX: Decode and update output, even if tokens are empty
  # This ensures terminal shows output even if generation finished with no tokens
  try:
    if buffered_token_output[req_id]:
      decoded = node.inference_engine.tokenizer.decode(buffered_token_output[req_id])
    else:
      # If no tokens, show empty string (or a message if finished)
      decoded = "" if not is_finished else "(no output generated)"
    topology_viz.update_prompt_output(req_id, decoded)
    if DEBUG >= 2:
      print(f"[update_topology_viz] req_id={req_id}, tokens={tokens}, is_finished={is_finished}, decoded={decoded[:50] if decoded else 'empty'}")
  except Exception as e:
    if DEBUG >= 1:
      print(f"[update_topology_viz] Error decoding tokens: {e}")
      import traceback
      traceback.print_exc()
    # Still update with empty string to show something
    topology_viz.update_prompt_output(req_id, "")
node.on_token.register("update_topology_viz").on_next(update_topology_viz)
def update_prompt_viz(request_id, opaque_status: str):
  if not topology_viz: return
  try:
    status = json.loads(opaque_status)
    if status.get("type") != "node_status" or status.get("status") != "start_process_prompt": return
    topology_viz.update_prompt(request_id, status.get("prompt", "corrupted prompt (this should never happen)"))
  except Exception as e:
    if DEBUG >= 2:
      print(f"Failed to update prompt viz: {e}")
      traceback.print_exc()
node.on_opaque_status.register("update_prompt_viz").on_next(update_prompt_viz)

def preemptively_load_shard(request_id: str, opaque_status: str):
  try:
    status = json.loads(opaque_status)
    if status.get("type") != "node_status" or status.get("status") != "start_process_prompt": return
    current_shard = node.get_current_shard(Shard.from_dict(status.get("shard")))
    if DEBUG >= 2: print(f"Preemptively starting download for {current_shard}")
    asyncio.create_task(node.inference_engine.ensure_shard(current_shard))
  except Exception as e:
    if DEBUG >= 2:
      print(f"Failed to preemptively start download: {e}")
      traceback.print_exc()
node.on_opaque_status.register("preemptively_load_shard").on_next(preemptively_load_shard)

last_events: dict[str, tuple[float, RepoProgressEvent]] = {}
def throttled_broadcast(shard: Shard, event: RepoProgressEvent):
  global last_events
  current_time = time.time()
  if event.status == "not_started": return
  last_event = last_events.get(shard.model_id)
  if last_event and last_event[1].status == "complete" and event.status == "complete": return
  if last_event and last_event[0] == event.status and current_time - last_event[0] < 0.2: return
  last_events[shard.model_id] = (current_time, event)
  asyncio.create_task(node.broadcast_opaque_status("", json.dumps({"type": "download_progress", "node_id": node.id, "progress": event.to_dict()})))
shard_downloader.on_progress.register("broadcast").on_next(throttled_broadcast)

async def run_model_cli(node: Node, model_name: str, prompt: str):
  inference_class = node.inference_engine.__class__.__name__
  shard = build_base_shard(model_name, inference_class)
  if not shard:
    print(f"Error: Unsupported model '{model_name}' for inference engine {inference_class}")
    return
  tokenizer = await resolve_tokenizer(get_repo(shard.model_id, inference_class))
  request_id = str(uuid.uuid4())
  callback_id = f"cli-wait-response-{request_id}"
  callback = node.on_token.register(callback_id)
  if topology_viz:
    topology_viz.update_prompt(request_id, prompt)
  try:
    if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template is not None:
      prompt = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
  except (ValueError, AttributeError):
    pass
  try:
    print(f"Processing prompt: {prompt}")
    await node.process_prompt(shard, prompt, request_id=request_id)
    tokens = []
    def on_token(_request_id, _tokens, _is_finished):
      tokens.extend(_tokens)
      return _request_id == request_id and _is_finished
    await callback.wait(on_token, timeout=300)
    print("\nGenerated response:")
    print(tokenizer.decode(tokens))
  except Exception as e:
    print(f"Error processing prompt: {str(e)}")
    traceback.print_exc()
  finally:
    node.on_token.deregister(callback_id)

def clean_path(path):
    """Clean and resolve path"""
    if path.startswith("Optional("):
        path = path.strip('Optional("').rstrip('")')
    return os.path.expanduser(path)

async def hold_outstanding(node: Node):
  while node.outstanding_requests:
    await asyncio.sleep(.5)
  return

async def run_iter(node: Node, shard: Shard, train: bool, data, batch_size=1):
  losses = []
  tokens = []
  for batch in tqdm(iterate_batches(data, batch_size), total=len(data) // batch_size):
    _, _, lengths = batch
    losses.append(np.sum(lengths * await node.enqueue_example(shard, *batch, train=train)))
    tokens.append(np.sum(lengths))
  total_tokens = np.sum(tokens)
  total_loss = np.sum(losses) / total_tokens

  return total_loss, total_tokens

async def eval_model_cli(node: Node, model_name, dataloader, batch_size, num_batches=-1):
  inference_class = node.inference_engine.__class__.__name__
  shard = build_base_shard(model_name, inference_class)
  if not shard:
    print(f"Error: Unsupported model '{model_name}' for inference engine {inference_class}")
    return
  tokenizer = await resolve_tokenizer(get_repo(shard.model_id, inference_class))
  train, val, test = dataloader(tokenizer.encode)
  print(f"Evaluating {len(test)} examples with batch_size {batch_size}")
  loss, tokens = await run_iter(node, shard, False, test, batch_size)
  print(f"total | {loss=}, {tokens=}")
  print("Waiting for outstanding tasks")
  await hold_outstanding(node)

async def train_model_cli(node: Node, model_name, dataloader, batch_size, iters, save_interval=0, checkpoint_dir=None):
  inference_class = node.inference_engine.__class__.__name__
  shard = build_base_shard(model_name, inference_class)
  if not shard:
    print(f"Error: Unsupported model '{model_name}' for inference engine {inference_class}")
    return
  tokenizer = await resolve_tokenizer(get_repo(shard.model_id, inference_class))
  train, val, test = dataloader(tokenizer.encode)
  print(f"Training on {len(train)} examples with batch_size {batch_size} for {iters} epochs")
  for i in tqdm(range(3)):
    await asyncio.sleep(1)
  for epoch in range(iters):
    loss, tokens = await run_iter(node, shard, True, train, batch_size)
    print(f"epoch {epoch + 1}/{iters}\t| loss: {loss}, tokens: {tokens}")
    if save_interval > 0 and epoch > 0 and (epoch % save_interval) == 0 and checkpoint_dir is not None:
      await node.coordinate_save(shard, epoch, checkpoint_dir)
      await hold_outstanding(node)
  await hold_outstanding(node)

# Global variable để lưu api_task cho cleanup
api_task_global = None

async def main():
  global api_task_global
  loop = asyncio.get_running_loop()

  # Print models directory info
  models_dir = get_models_dir()
  if DEBUG >= 1:
    print(f"📁 Models directory: {models_dir}")
    print(f"   Place your model files in subdirectories here")


  def restore_cursor():
    pass  # No cursor restoration needed on Windows

  # Restore the cursor when the program exits
  atexit.register(restore_cursor)

  # Use a more direct approach to handle signals
  def handle_exit():
    asyncio.ensure_future(shutdown(signal.SIGTERM, loop, node.server))

  await node.start(wait_for_peers=args.wait_for_peers)

  # Always start API server (unless disabled)
  if True:
    try:
      if DEBUG >= 1: print(f"Starting API server on port {args.chatgpt_api_port}...")
      api_task = asyncio.create_task(api.run(port=args.chatgpt_api_port))  # Start the API server as a non-blocking task
      api_task_global = api_task  # Lưu reference để cleanup sau
      
      def api_task_done(task):
        try:
          task.result()  # This will raise exception if task failed
        except asyncio.CancelledError:
          if DEBUG >= 2: print(f"[INFO] API server task cancelled (normal shutdown)")
        except Exception as e:
          print(f"[ERROR] API server task failed: {e}")
          if DEBUG >= 1:
            traceback.print_exc()
      api_task.add_done_callback(api_task_done)
      if DEBUG >= 1: print(f"API server started at http://127.0.0.1:{args.chatgpt_api_port}")
    except Exception as e:
      print(f"[ERROR] Error starting API server: {e}")
      if DEBUG >= 1:
        traceback.print_exc()
      raise

    if args.command == "run" or args.run_model:
      model_name = args.model_name or args.run_model
      if not model_name:
        print("Error: Model name is required when using 'run' command or --run-model")
        return

      # === Auto Hardware + Model Fit Check (ported from llmit) ===
      check_model_hardware_fit(model_name)
      # ===========================================================

      # Only run CLI inference if a prompt is provided
      if args.prompt:
        asyncio.create_task(run_model_cli(node, model_name, args.prompt))
      else:
        if DEBUG >= 1: print(f"Model '{model_name}' selected. Waiting for API requests...")
        # Optionally pre-load the model here if needed, but API will handle it on first request
      
      # Keep the program running so API server stays alive
      try:
        await asyncio.Event().wait()
      except Exception as e:
        if DEBUG >= 1:
          print(f"Error in main event loop: {e}")
          traceback.print_exc()
        raise
    elif args.command == "eval" or args.command == 'train':
      model_name = args.model_name
      dataloader = lambda tok: load_dataset(args.data, preprocess=lambda item: tok(item)
                                                   , loadline=lambda line: json.loads(line).get("text",""))
      if args.command == 'eval':
        if not model_name:
          print("Error: Much like a human, I can't evaluate anything without a model")
          return
        await eval_model_cli(node, model_name, dataloader, args.batch_size)
      else:
        if not model_name:
          print("Error: This train ain't leaving the station without a model")
          return
        await train_model_cli(node, model_name, dataloader, args.batch_size, args.iters, save_interval=args.save_every, checkpoint_dir=args.save_checkpoint_dir)

    else:
      # If no command specified, just wait (API server already started above)
      try:
        if DEBUG >= 1: print("Server running. Waiting for requests...")
        await asyncio.Event().wait()
      except Exception as e:
        if DEBUG >= 1:
          print(f"Error in main event loop: {e}")
          traceback.print_exc()
        raise

  if args.wait_for_peers > 0:
    print("Cooldown to allow peers to exit gracefully")
    for i in tqdm(range(50)):
      await asyncio.sleep(.1)

def run():
    loop = None
    try:
        loop = configure_event_loop()
        loop.run_until_complete(main())
    except KeyboardInterrupt:
        print("\nShutdown requested... exiting")
    except Exception as e:
        print(f"\n[ERROR] Fatal error: {e}")
        import traceback
        traceback.print_exc()
        print("\nProgram will exit. Please check the error above.")
    finally:
        if loop:
            try:
                # Cancel api_task first if it exists
                global api_task_global
                if api_task_global and not api_task_global.done():
                    if DEBUG >= 2: print("[INFO] Cancelling API server task...")
                    api_task_global.cancel()
                    try:
                        loop.run_until_complete(api_task_global)
                    except asyncio.CancelledError:
                        if DEBUG >= 2: print("[INFO] API server task cancelled successfully")
                    except Exception as e:
                        if DEBUG >= 2: print(f"[WARNING] Error cancelling API task: {e}")
                
                # Cancel all remaining pending tasks
                pending = asyncio.all_tasks(loop)
                if pending:
                    if DEBUG >= 2: print(f"[INFO] Cancelling {len(pending)} remaining tasks...")
                    for task in pending:
                        task.cancel()
                    # Wait for tasks to complete cancellation with timeout
                    try:
                        loop.run_until_complete(asyncio.wait_for(
                            asyncio.gather(*pending, return_exceptions=True),
                            timeout=5.0
                        ))
                    except asyncio.TimeoutError:
                        if DEBUG >= 1: print("[WARNING] Some tasks did not complete cancellation in time")
                    except Exception as e:
                        if DEBUG >= 2: print(f"[WARNING] Error during task cleanup: {e}")
            except Exception as e:
                if DEBUG >= 1: print(f"[WARNING] Error during cleanup: {e}")
            finally:
                try:
                    # Shutdown async generators
                    loop.run_until_complete(loop.shutdown_asyncgens())
                except Exception as e:
                    if DEBUG >= 2: print(f"[WARNING] Error shutting down async generators: {e}")
                try:
                    loop.close()
                    if DEBUG >= 2: print("[INFO] Event loop closed successfully")
                except Exception as e:
                    if DEBUG >= 2: print(f"[WARNING] Error closing loop: {e}")

if __name__ == "__main__":
  run()

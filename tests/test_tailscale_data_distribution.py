"""
Integration check for Synapse data distribution over Tailscale.

What this file verifies:
1. Tailscale has at least two online devices.
2. At least two devices expose the Synapse REST API.
3. Synapse topology and LACP resources show more than one node/layer shard.
4. A chat request can run on a cluster-capable node and still reports
   multi-node distribution afterward.
5. Optional: start a training job probe when explicitly enabled.

Run as a script:
    uv run python tests/test_tailscale_data_distribution.py

Run with pytest:
    uv run python -m pytest tests/test_tailscale_data_distribution.py -q

Required for live mode:
    TAILSCALE_API_KEY
    TAILNET_NAME

Optional:
    SYNAPSE_API_PORT=52415
    SYNAPSE_TEST_MODEL=qwen2.5:0.5b
    SYNAPSE_TEST_LIVE=1              # required for pytest live run
    SYNAPSE_TEST_RUN_CHAT=1          # default: 1
    SYNAPSE_TEST_RUN_TRAINING=0      # default: 0, avoids accidental heavy job
    SYNAPSE_TEST_TRAIN_DATASET=synapse/train/data/lora/train.jsonl
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import socket
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    import aiohttp
except ImportError as exc:
    aiohttp = None
    AIOHTTP_IMPORT_ERROR = exc
else:
    AIOHTTP_IMPORT_ERROR = None


TAILSCALE_API_KEY = os.getenv("TAILSCALE_API_KEY", "tskey-api-kFGrMwTg3L11CNTRL-zUKTZripuEfqx98GNpBtDfxoo2TdTpTph")
TAILNET_NAME = os.getenv("TAILNET_NAME", "testdoki925@gmail.com")
SYNAPSE_API_PORT = int(os.getenv("SYNAPSE_API_PORT") or os.getenv("CHATGPT_API_PORT") or "52415")
SYNAPSE_NODE_PORT = int(os.getenv("SYNAPSE_NODE_PORT") or os.getenv("NODE_PORT") or "5678")
TEST_MODEL = os.getenv("SYNAPSE_TEST_MODEL", "qwen2.5:0.5b")
RUN_CHAT_PROBE = os.getenv("SYNAPSE_TEST_RUN_CHAT", "1") not in {"0", "false", "False"}
RUN_TRAINING_PROBE = os.getenv("SYNAPSE_TEST_RUN_TRAINING", "0") in {"1", "true", "True"}
TRAINING_DATASET = os.getenv("SYNAPSE_TEST_TRAIN_DATASET", "synapse/train/data/lora/train.jsonl")
HTTP_TIMEOUT_S = float(os.getenv("SYNAPSE_TEST_HTTP_TIMEOUT", "20"))
CHAT_TIMEOUT_S = float(os.getenv("SYNAPSE_TEST_CHAT_TIMEOUT", "120"))


@dataclass
class PhaseResult:
    ok: bool
    reason: str = ""
    data: Any = None


@dataclass
class SynapseNode:
    name: str
    ip: str
    base_url: str
    device_id: str = ""
    addresses: List[str] = field(default_factory=list)
    is_self: bool = False
    health: Dict[str, Any] = field(default_factory=dict)
    topology: Dict[str, Any] = field(default_factory=dict)
    resources: Dict[str, Any] = field(default_factory=dict)
    distributed_status: Dict[str, Any] = field(default_factory=dict)
    resource_summary: Dict[str, Any] = field(default_factory=dict)
    tailscale_nodes: List[Dict[str, Any]] = field(default_factory=list)
    reported_node_id: str = ""


def log_header(phase: int, title: str) -> None:
    print(f"\n{'=' * 72}")
    print(f"PHASE {phase}: {title}")
    print(f"{'=' * 72}")


def log_ok(message: str) -> None:
    print(f"  PASS: {message}")


def log_fail(message: str) -> None:
    print(f"  FAIL: {message}")


def log_warn(message: str) -> None:
    print(f"  WARN: {message}")


def log_info(message: str) -> None:
    print(f"  INFO: {message}")


def require_aiohttp() -> None:
    if aiohttp is None:
        raise RuntimeError(f"aiohttp is required for this integration test: {AIOHTTP_IMPORT_ERROR}")


def parse_tailscale_time(value: str) -> Optional[float]:
    if not value:
        return None
    try:
        dt = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        return time.time() - dt.timestamp()
    except ValueError:
        return None


def first_ipv4(addresses: Iterable[Any]) -> Optional[str]:
    for raw in addresses or []:
        value = str(raw.get("addr") if isinstance(raw, dict) else raw).strip()
        if "/" in value:
            value = value.split("/", 1)[0].strip()
        if value.count(".") == 3 and all(part.isdigit() for part in value.split(".")):
            return value
    return None


def get_local_node_id(port: int = SYNAPSE_NODE_PORT) -> str:
    """Best-effort default node id, matching helpers.get_or_create_node_id style."""
    mac = "00:00:00:00:00:00"
    try:
        import psutil

        for _nic, addrs in psutil.net_if_addrs().items():
            for addr in addrs:
                if addr.family == psutil.AF_LINK and addr.address:
                    mac = addr.address
                    break
            if mac != "00:00:00:00:00:00":
                break
    except Exception:
        pass
    return hashlib.sha256(f"{mac}:{port}".encode()).hexdigest()[:32]


def get_local_ips() -> set[str]:
    ips: set[str] = {"127.0.0.1", "localhost"}
    try:
        import psutil

        for _nic, addrs in psutil.net_if_addrs().items():
            for addr in addrs:
                if getattr(socket, "AF_INET", None) and addr.family == socket.AF_INET and addr.address:
                    ips.add(addr.address.strip())
    except Exception:
        pass
    try:
        host_ip = socket.gethostbyname(socket.gethostname())
        if host_ip:
            ips.add(host_ip)
    except Exception:
        pass
    return ips


async def fetch_json(
    session: "aiohttp.ClientSession",
    method: str,
    url: str,
    *,
    json_body: Optional[Dict[str, Any]] = None,
    timeout_s: float = HTTP_TIMEOUT_S,
) -> Tuple[int, Any]:
    request_timeout = aiohttp.ClientTimeout(total=timeout_s)
    try:
        async with session.request(method, url, json=json_body, timeout=request_timeout) as resp:
            text = await resp.text()
            try:
                return resp.status, json.loads(text) if text else {}
            except json.JSONDecodeError:
                return resp.status, text
    except asyncio.TimeoutError:
        return 0, {"error": f"timeout after {timeout_s:g}s", "url": url}
    except (aiohttp.ClientError, OSError) as exc:
        return 0, {"error": f"{type(exc).__name__}: {exc}", "url": url}


async def phase1_tailscale_connectivity() -> PhaseResult:
    log_header(1, "Tailscale connectivity")
    require_aiohttp()

    if not TAILSCALE_API_KEY:
        return PhaseResult(False, "TAILSCALE_API_KEY is empty")
    if not TAILNET_NAME:
        return PhaseResult(False, "TAILNET_NAME is empty")

    url = f"https://api.tailscale.com/api/v2/tailnet/{TAILNET_NAME}/devices"
    headers = {"Authorization": f"Bearer {TAILSCALE_API_KEY}"}

    try:
        async with aiohttp.ClientSession(headers=headers) as session:
            status, payload = await fetch_json(session, "GET", url, timeout_s=45)
    except Exception as exc:
        return PhaseResult(False, f"cannot reach Tailscale API: {exc}")

    if status != 200:
        return PhaseResult(False, f"Tailscale API returned {status}: {str(payload)[:200]}")

    devices = payload.get("devices", []) if isinstance(payload, dict) else []
    online: List[Dict[str, Any]] = []

    for dev in devices:
        addresses = dev.get("addresses", [])
        ip = first_ipv4(addresses)
        last_seen_ago = parse_tailscale_time(dev.get("lastSeen", ""))
        is_online = dev.get("online") is True or (last_seen_ago is not None and last_seen_ago < 120)
        if ip and is_online:
            online.append(
                {
                    "name": dev.get("name", ""),
                    "device_id": dev.get("id", ""),
                    "addresses": addresses,
                    "ip": ip,
                    "last_seen_ago": last_seen_ago,
                }
            )

    for dev in online:
        ago = "unknown" if dev["last_seen_ago"] is None else f"{int(dev['last_seen_ago'])}s"
        log_info(f"{dev['name']} at {dev['ip']} online, lastSeen={ago}")

    if len(online) < 2:
        return PhaseResult(False, f"need >=2 online Tailscale devices, found {len(online)}", online)

    log_ok(f"found {len(online)} online Tailscale devices")
    return PhaseResult(True, data=online)


def normalize_tailscale_attributes(raw_attributes: Any) -> Dict[str, str]:
    if isinstance(raw_attributes, dict):
        return {str(key): "" if value is None else str(value) for key, value in raw_attributes.items()}
    if isinstance(raw_attributes, list):
        normalized: Dict[str, str] = {}
        for item in raw_attributes:
            if not isinstance(item, dict):
                continue
            key = item.get("key") or item.get("name") or item.get("id")
            if key is None:
                continue
            value = item.get("value")
            normalized[str(key)] = "" if value is None else str(value)
        return normalized
    return {}


async def fetch_device_attributes(device_id: str) -> Dict[str, str]:
    if not device_id:
        return {}
    url = f"https://api.tailscale.com/api/v2/device/{device_id}/attributes"
    headers = {"Authorization": f"Bearer {TAILSCALE_API_KEY}"}
    try:
        async with aiohttp.ClientSession(headers=headers) as session:
            status, payload = await fetch_json(session, "GET", url, timeout_s=20)
        if status != 200 or not isinstance(payload, dict):
            return {}
        return normalize_tailscale_attributes(payload.get("attributes"))
    except Exception:
        return {}


def get_synapse_node_id_attribute(attrs: Dict[str, str]) -> str:
    value = attrs.get("custom:synapse_node_id") or ""
    return value.replace("_", "-")


def extract_self_node_id(resources: Dict[str, Any], topology: Dict[str, Any]) -> str:
    nodes = resources.get("nodes", []) if isinstance(resources, dict) else []
    if isinstance(nodes, list):
        for item in nodes:
            if isinstance(item, dict) and item.get("is_self") and item.get("node_id"):
                return str(item["node_id"])

    topo_nodes = topology.get("nodes", {}) if isinstance(topology, dict) else {}
    if isinstance(topo_nodes, dict) and len(topo_nodes) == 1:
        return next(iter(topo_nodes.keys()))

    active = topology.get("active_node_id") if isinstance(topology, dict) else None
    return str(active or "")


def count_topology_nodes(topology: Dict[str, Any]) -> int:
    nodes = topology.get("nodes", {}) if isinstance(topology, dict) else {}
    return len(nodes) if isinstance(nodes, dict) else 0


def count_peer_edges(topology: Dict[str, Any]) -> int:
    peer_graph = topology.get("peer_graph", {}) if isinstance(topology, dict) else {}
    if not isinstance(peer_graph, dict):
        return 0
    return sum(len(edges) for edges in peer_graph.values() if isinstance(edges, list))


def is_generated_node_id(node_id: str) -> bool:
    return bool(re.fullmatch(r"[a-f0-9]{16,64}", node_id or ""))


def node_id_matches_tailscale_name(node_id: str, tailscale_name: str) -> bool:
    if not node_id or not tailscale_name:
        return False
    short_name = tailscale_name.split(".", 1)[0]
    return node_id in {tailscale_name, short_name}


def print_node_id_mismatch_guidance(nodes: List[SynapseNode]) -> None:
    log_warn("Tailscale Device Attributes are unavailable or unusable for node id sync.")
    log_warn("Tailscale discovery falls back to peer_id = Tailscale FQDN.")
    log_warn("Every Synapse process must therefore use --node-id equal to its Tailscale name.")
    print("")
    for node in nodes:
        if not node.name:
            continue
        print(
            "  "
            f"python main.py run --default-model {TEST_MODEL} "
            f"--discovery-module tailscale --node-id {node.name} "
            f"--chatgpt-api-port {SYNAPSE_API_PORT}"
        )
    print("")


def print_start_commands_for_devices(devices: List[Dict[str, Any]]) -> None:
    log_info("Expected startup command on each online Tailscale device:")
    for dev in devices:
        name = dev.get("name")
        if not name:
            continue
        print(
            "  "
            f"python main.py run --default-model {TEST_MODEL} "
            f"--discovery-module tailscale --node-id {name} "
            f"--chatgpt-api-port {SYNAPSE_API_PORT}"
        )
    print("")


async def phase2_synapse_discovery(devices: List[Dict[str, Any]]) -> PhaseResult:
    log_header(2, "Synapse API discovery and node id alignment")

    local_ips = get_local_ips()
    local_default_id = get_local_node_id()
    device_attributes: Dict[str, Dict[str, str]] = {}
    for dev in devices:
        device_id = dev.get("device_id", "")
        if device_id:
            device_attributes[device_id] = await fetch_device_attributes(device_id)
    attributes_readable = any(device_attributes.values())
    synced_attribute_count = sum(
        1 for attrs in device_attributes.values() if get_synapse_node_id_attribute(attrs)
    )
    log_info(f"Synapse API port: {SYNAPSE_API_PORT}")
    log_info(f"default local node id guess: {local_default_id[:16]}...")
    log_info(f"Tailscale Device Attributes readable: {attributes_readable}")
    log_info(f"Devices with custom:synapse_node_id: {synced_attribute_count}/{len(devices)}")
    if synced_attribute_count < len(devices):
        log_warn("Some devices do not expose custom:synapse_node_id; those peers will fall back to Tailscale FQDN.")

    nodes: List[SynapseNode] = []

    async with aiohttp.ClientSession() as session:
        for dev in devices:
            ip = dev.get("ip") or first_ipv4(dev.get("addresses", []))
            if not ip:
                continue

            base_url = f"http://{ip}:{SYNAPSE_API_PORT}"
            status, health = await fetch_json(session, "GET", f"{base_url}/healthcheck", timeout_s=10)
            if status != 200 or not isinstance(health, dict):
                detail = health.get("error") if isinstance(health, dict) else str(health)
                log_warn(f"{dev.get('name', ip)}: no Synapse REST API at {base_url} (status={status}, {detail})")
                continue

            _topo_status, topology = await fetch_json(session, "GET", f"{base_url}/v1/topology", timeout_s=10)
            _res_status, resources = await fetch_json(session, "GET", f"{base_url}/v1/cluster/resources", timeout_s=10)
            _ts_status, ts_nodes_payload = await fetch_json(session, "GET", f"{base_url}/v1/tailscale/nodes", timeout_s=10)
            ts_nodes = ts_nodes_payload.get("data", []) if isinstance(ts_nodes_payload, dict) else []

            node = SynapseNode(
                name=dev.get("name", ""),
                ip=ip,
                base_url=base_url,
                device_id=dev.get("device_id", ""),
                addresses=list(dev.get("addresses", [])),
                is_self=ip in local_ips,
                health=health,
                topology=topology if isinstance(topology, dict) else {},
                resources=resources if isinstance(resources, dict) else {},
                tailscale_nodes=ts_nodes if isinstance(ts_nodes, list) else [],
            )
            node.reported_node_id = extract_self_node_id(node.resources, node.topology)
            nodes.append(node)

            log_info(
                f"{node.name or node.ip}: API ok, model_loaded={health.get('model_loaded')}, "
                f"topology_nodes={count_topology_nodes(node.topology)}, peer_edges={count_peer_edges(node.topology)}, "
                f"node_id={node.reported_node_id or 'unknown'}"
            )

    if len(nodes) < 2:
        print_start_commands_for_devices(devices)
        return PhaseResult(False, f"need >=2 Synapse REST APIs, found {len(nodes)}", nodes)

    mismatches = []
    for node in nodes:
        node_id = node.reported_node_id
        attrs = device_attributes.get(node.device_id, {})
        attr_node_id = get_synapse_node_id_attribute(attrs)
        expected_peer_id = attr_node_id or node.name
        if not node_id:
            log_warn(f"{node.name or node.ip}: cannot verify node-id; /v1/topology did not expose one")
            continue
        if expected_peer_id and node_id != expected_peer_id:
            if attr_node_id or is_generated_node_id(node_id) or not node_id_matches_tailscale_name(node_id, node.name):
                mismatches.append(node)

    if mismatches:
        print_node_id_mismatch_guidance(nodes)
        for node in mismatches:
            attr_node_id = get_synapse_node_id_attribute(device_attributes.get(node.device_id, {}))
            expected_peer_id = attr_node_id or node.name
            log_warn(
                f"node-id mismatch: Synapse listens on {node.reported_node_id}, "
                f"but peers will call {expected_peer_id}"
            )
        return PhaseResult(False, "node id mismatch would make P2P RPC target the wrong topic", nodes)

    log_ok(f"found {len(nodes)} Synapse nodes")
    return PhaseResult(True, data=nodes)


def summarize_lacp_distribution(resources: Dict[str, Any]) -> Dict[str, Any]:
    rows = resources.get("nodes", []) if isinstance(resources, dict) else []
    if not isinstance(rows, list):
        rows = []
    raw_node_ids = [str(row.get("node_id")) for row in rows if isinstance(row, dict) and row.get("node_id")]

    layer_rows = [
        row
        for row in rows
        if isinstance(row, dict)
        and row.get("node_id")
        and int(row.get("layer_count") or 0) > 0
    ]
    unique_node_ids = {str(row.get("node_id")) for row in layer_rows}
    ranges = [
        (str(row.get("node_id")), int(row.get("layer_start") or 0), int(row.get("layer_end") or 0))
        for row in layer_rows
    ]
    total_layers = int(resources.get("total_layers") or 0) if isinstance(resources, dict) else 0
    return {
        "total_nodes": int(resources.get("total_nodes") or len(unique_node_ids)) if isinstance(resources, dict) else 0,
        "raw_node_count": len(rows),
        "raw_node_ids": raw_node_ids,
        "unique_node_count": len(unique_node_ids),
        "ranges": ranges,
        "total_layers": total_layers,
        "strategy_name": str(resources.get("strategy_name") or "") if isinstance(resources, dict) else "",
    }


async def phase3_partition_verification(nodes: List[SynapseNode]) -> PhaseResult:
    log_header(3, "Topology and LACP partition verification")

    evidence = []
    async with aiohttp.ClientSession() as session:
        for node in nodes:
            status_code, dist_status = await fetch_json(
                session, "GET", f"{node.base_url}/v1/distributed/status", timeout_s=10
            )
            _res_code, resources = await fetch_json(
                session, "GET", f"{node.base_url}/v1/cluster/resources", timeout_s=10
            )
            _topo_code, topology = await fetch_json(session, "GET", f"{node.base_url}/v1/topology", timeout_s=10)

            if isinstance(resources, dict):
                node.resources = resources
            if isinstance(topology, dict):
                node.topology = topology
            node.distributed_status = dist_status if isinstance(dist_status, dict) else {}

            summary = summarize_lacp_distribution(node.resources)
            node.resource_summary = summary
            topology_nodes = count_topology_nodes(node.topology)
            partitions_count = dist_status.get("partitions_count", 0) if isinstance(dist_status, dict) else 0
            multi_node_inference = bool(dist_status.get("multi_node_inference")) if isinstance(dist_status, dict) else False
            partition_error = ""
            if isinstance(resources, dict):
                partition_error = str(resources.get("partition_error") or "")
            if not partition_error and isinstance(dist_status, dict):
                partition_error = str(dist_status.get("partition_error") or "")

            log_info(
                f"{node.name or node.ip}: status={status_code}, partitions={partitions_count}, "
                f"topology_nodes={topology_nodes}, lacp_nodes={summary['unique_node_count']}, "
                f"resource_rows={summary['raw_node_count']}, resource_total_nodes={summary['total_nodes']}"
            )
            if partition_error:
                log_warn(f"{node.name or node.ip}: partition_error={partition_error}")
            if _res_code != 200:
                log_warn(f"{node.name or node.ip}: /v1/cluster/resources returned status={_res_code}")
            if partitions_count >= 2 and summary["unique_node_count"] < 2:
                log_warn(
                    f"{node.name or node.ip}: distributed/status suggests multi-node, "
                    "but cluster resources do not expose multi-node layer shards"
                )
                log_info(f"  resource node ids: {summary['raw_node_ids'] or 'none'}")
            for node_id, start, end in summary["ranges"]:
                log_info(f"  shard {node_id[:16]}... layers {start}-{end}")

            evidence.append(
                {
                    "node": node,
                    "partitions_count": partitions_count,
                    "multi_node_inference": multi_node_inference,
                    "topology_nodes": topology_nodes,
                    "lacp_unique_nodes": summary["unique_node_count"],
                    "ranges": summary["ranges"],
                    "resource_rows": summary["raw_node_count"],
                    "resource_total_nodes": summary["total_nodes"],
                    "partition_error": partition_error,
                }
            )

    has_topology_cluster = any(item["topology_nodes"] >= 2 for item in evidence)
    has_status_cluster = any(item["partitions_count"] >= 2 and item["multi_node_inference"] for item in evidence)
    has_layer_distribution = any(item["lacp_unique_nodes"] >= 2 and len(item["ranges"]) >= 2 for item in evidence)
    asymmetric_nodes = [
        item["node"].name or item["node"].ip
        for item in evidence
        if item["topology_nodes"] < 2 or item["partitions_count"] < 2 or item["lacp_unique_nodes"] < 2
    ]

    if has_topology_cluster and has_status_cluster and has_layer_distribution:
        if asymmetric_nodes:
            log_warn(
                "cluster view is asymmetric; these nodes still report single-node state: "
                + ", ".join(asymmetric_nodes)
            )
        log_ok("topology and LACP resources show multi-node layer distribution")
        return PhaseResult(True, data=evidence)

    reason = (
        "distribution evidence is incomplete: "
        f"topology={has_topology_cluster}, status={has_status_cluster}, lacp={has_layer_distribution}"
    )
    return PhaseResult(False, reason, evidence)


def distribution_score(node: SynapseNode) -> Tuple[int, int, int, int, int, int]:
    resources_summary = summarize_lacp_distribution(node.resources)
    partitions_count = int(node.distributed_status.get("partitions_count") or 0)
    return (
        1 if node.distributed_status.get("multi_node_inference") else 0,
        partitions_count,
        resources_summary["unique_node_count"],
        count_topology_nodes(node.topology),
        count_peer_edges(node.topology),
        1 if node.health.get("inference_ready") else 0,
    )


def select_chat_target(nodes: List[SynapseNode]) -> SynapseNode:
    return max(nodes, key=distribution_score)


async def phase4_e2e_inference(nodes: List[SynapseNode]) -> PhaseResult:
    log_header(4, "End-to-end chat probe")

    if not RUN_CHAT_PROBE:
        log_warn("chat probe disabled by SYNAPSE_TEST_RUN_CHAT=0")
        return PhaseResult(True, "skipped")

    target = select_chat_target(nodes)
    score = distribution_score(target)
    log_info(
        f"chat target: {target.name or target.ip} "
        f"(status_multi={score[0]}, partitions={score[1]}, lacp_nodes={score[2]}, "
        f"topology_nodes={score[3]}, peer_edges={score[4]}, ready={score[5]})"
    )
    payload = {
        "model": TEST_MODEL,
        "messages": [{"role": "user", "content": "Say OK in one short sentence."}],
        "temperature": 0.0,
        "max_tokens": 16,
        "stream": False,
    }

    async with aiohttp.ClientSession() as session:
        status, response = await fetch_json(
            session,
            "POST",
            f"{target.base_url}/v1/chat/completions",
            json_body=payload,
            timeout_s=CHAT_TIMEOUT_S,
        )
        if status != 200 or not isinstance(response, dict):
            return PhaseResult(False, f"chat request failed on {target.base_url}: status={status}, body={response}")

        choices = response.get("choices", [])
        content = ""
        first_choice = choices[0] if choices else {}
        if choices:
            first = first_choice
            message = first.get("message") or first.get("delta") or {}
            content = str(message.get("content") or first.get("text") or "")
        usage = response.get("usage") if isinstance(response.get("usage"), dict) else {}
        try:
            completion_tokens = int(usage.get("completion_tokens") or 0)
        except (TypeError, ValueError):
            completion_tokens = 0
        synapse_distribution = (
            response.get("synapse_distribution")
            if isinstance(response.get("synapse_distribution"), dict)
            else {}
        )
        try:
            stage_count = int(synapse_distribution.get("stage_count") or 0)
        except (TypeError, ValueError):
            stage_count = 0
        try:
            tokens_emitted = int(synapse_distribution.get("tokens_emitted") or 0)
        except (TypeError, ValueError):
            tokens_emitted = 0
        fallback_raw = synapse_distribution.get("fallback")
        fallback = fallback_raw is True or str(fallback_raw).lower() == "true"

        status2, dist_status = await fetch_json(
            session, "GET", f"{target.base_url}/v1/distributed/status", timeout_s=10
        )
        status3, resources = await fetch_json(
            session, "GET", f"{target.base_url}/v1/cluster/resources", timeout_s=10
        )

    if not content.strip():
        log_warn(f"empty chat response body: {json.dumps(response, ensure_ascii=False)[:500]}")
        log_warn(
            "empty diagnostics: "
            f"synapse_distribution={synapse_distribution}, finish_reason={first_choice.get('finish_reason')}, "
            f"usage={usage}, first_choice={json.dumps(first_choice, ensure_ascii=False)[:500]}"
        )
        return PhaseResult(False, "chat completed but returned empty assistant content", response)
    if completion_tokens <= 0:
        log_warn(
            "chat returned text but completion token count is not positive: "
            f"synapse_distribution={synapse_distribution}, finish_reason={first_choice.get('finish_reason')}, "
            f"usage={usage}, first_choice={json.dumps(first_choice, ensure_ascii=False)[:500]}"
        )
        return PhaseResult(False, "chat completed but usage.completion_tokens <= 0", response)
    if (
        synapse_distribution.get("mode") != "distributed"
        or stage_count < 2
        or fallback
        or tokens_emitted <= 0
    ):
        log_warn(
            "chat response does not prove distributed execution: "
            f"synapse_distribution={synapse_distribution}, finish_reason={first_choice.get('finish_reason')}, "
            f"usage={usage}, first_choice={json.dumps(first_choice, ensure_ascii=False)[:500]}"
        )
        return PhaseResult(False, "chat completed but synapse_distribution does not prove distributed execution", response)

    partitions_count = dist_status.get("partitions_count", 0) if isinstance(dist_status, dict) else 0
    resources_summary = summarize_lacp_distribution(resources if isinstance(resources, dict) else {})
    post_partition_error = resources.get("partition_error") if isinstance(resources, dict) else None
    log_info(
        f"post-chat evidence: distributed_status={status2}, partitions={partitions_count}, "
        f"resources_status={status3}, lacp_nodes={resources_summary['unique_node_count']}, "
        f"resource_rows={resources_summary['raw_node_count']}"
    )
    if post_partition_error:
        log_warn(f"post-chat partition_error={post_partition_error}")
    if status2 == 200 and partitions_count >= 2 and status3 == 200 and resources_summary["unique_node_count"] >= 2:
        log_ok(f"chat returned content and LACP resources still show {resources_summary['unique_node_count']} nodes")
        log_info(f"response: {content[:120]}")
        log_info(f"synapse_distribution={synapse_distribution}")
        return PhaseResult(
            True,
            data={"response": response, "distributed_status": dist_status, "resources": resources},
        )

    return PhaseResult(
        False,
        f"chat worked on {target.name or target.ip} but distribution is not proven after chat: "
        f"partitions={partitions_count}, lacp_nodes={resources_summary['unique_node_count']}",
        {"response": response, "distributed_status": dist_status, "resources": resources},
    )


async def phase5_training_probe(nodes: List[SynapseNode]) -> PhaseResult:
    log_header(5, "Training distribution probe")

    if not RUN_TRAINING_PROBE:
        log_warn("training start disabled; set SYNAPSE_TEST_RUN_TRAINING=1 to run it")
        return PhaseResult(True, "skipped")

    target = select_chat_target(nodes)
    score = distribution_score(target)
    log_info(
        f"training target: {target.name or target.ip} "
        f"(status_multi={score[0]}, partitions={score[1]}, lacp_nodes={score[2]}, "
        f"topology_nodes={score[3]}, peer_edges={score[4]}, ready={score[5]})"
    )
    payload = {
        "model": TEST_MODEL,
        "dataset": TRAINING_DATASET,
        "epochs": 1,
        "batch_size": 1,
        "max_steps": 1,
        "max_length": 64,
    }

    async with aiohttp.ClientSession() as session:
        status, response = await fetch_json(
            session,
            "POST",
            f"{target.base_url}/v1/training/distributed/start",
            json_body=payload,
            timeout_s=30,
        )
        if status != 200 or not isinstance(response, dict) or not response.get("success"):
            return PhaseResult(False, f"training start failed: status={status}, body={response}")

        await asyncio.sleep(1.0)
        status2, training_status = await fetch_json(
            session, "GET", f"{target.base_url}/v1/training/status", timeout_s=10
        )
        status3, resources = await fetch_json(
            session, "GET", f"{target.base_url}/v1/cluster/resources", timeout_s=10
        )

    summary = summarize_lacp_distribution(resources if isinstance(resources, dict) else {})
    if status2 == 200 and status3 == 200 and summary["unique_node_count"] >= 2:
        log_ok("training endpoint accepted job while cluster resources show multi-node LACP distribution")
        return PhaseResult(True, data={"start": response, "status": training_status, "resources": resources})

    return PhaseResult(False, f"training started but cluster evidence is weak: {training_status}", response)


def print_summary(results: Dict[str, PhaseResult]) -> None:
    print(f"\n{'=' * 72}")
    print("SUMMARY")
    print(f"{'=' * 72}")
    for name, result in results.items():
        marker = "PASS" if result.ok else "FAIL"
        reason = f" - {result.reason}" if result.reason else ""
        print(f"  {name:<28} {marker}{reason}")
    print(f"{'=' * 72}")


async def run_all_phases() -> Dict[str, PhaseResult]:
    print("Synapse AI Tailscale data distribution integration check")

    results: Dict[str, PhaseResult] = {}

    phase1 = await phase1_tailscale_connectivity()
    results["tailscale_connectivity"] = phase1
    if not phase1.ok:
        print_summary(results)
        return results

    phase2 = await phase2_synapse_discovery(phase1.data)
    results["synapse_discovery"] = phase2
    if not phase2.ok:
        print_summary(results)
        return results

    nodes: List[SynapseNode] = phase2.data

    phase3 = await phase3_partition_verification(nodes)
    results["partition_verification"] = phase3

    has_cluster_signal = any(
        item.get("topology_nodes", 0) >= 2 or item.get("partitions_count", 0) >= 2
        for item in (phase3.data or [])
        if isinstance(item, dict)
    )

    if phase3.ok or has_cluster_signal:
        if not phase3.ok:
            log_warn("partition verification failed, but cluster signals exist; continuing with diagnostic probes")
        results["chat_probe"] = await phase4_e2e_inference(nodes)
        if phase3.ok or RUN_TRAINING_PROBE:
            results["training_probe"] = await phase5_training_probe(nodes)
        else:
            results["training_probe"] = PhaseResult(
                True,
                "skipped; set SYNAPSE_TEST_RUN_TRAINING=1 to run diagnostic training despite weak LACP evidence",
            )
    else:
        results["chat_probe"] = PhaseResult(False, "skipped because partition verification failed")
        results["training_probe"] = PhaseResult(False, "skipped because partition verification failed")

    print_summary(results)
    return results


def assert_results_pass(results: Dict[str, PhaseResult]) -> None:
    failed = {name: result.reason for name, result in results.items() if not result.ok}
    assert not failed, f"distribution check failed: {failed}"


async def test_tailscale_data_distribution() -> None:
    try:
        import pytest
    except ImportError:
        pytest = None

    if os.getenv("SYNAPSE_TEST_LIVE", "0") not in {"1", "true", "True"}:
        if pytest is not None:
            pytest.skip("set SYNAPSE_TEST_LIVE=1 to run live Tailscale integration test")
        return

    results = await run_all_phases()
    assert_results_pass(results)


if __name__ == "__main__":
    try:
        live_results = asyncio.run(run_all_phases())
        assert_results_pass(live_results)
    except AssertionError as exc:
        print(f"\nFAILED: {exc}")
        sys.exit(1)

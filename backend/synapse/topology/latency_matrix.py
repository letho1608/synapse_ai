"""
LACP — Latency Matrix & Hierarchical Clustering

Builds an N×N latency matrix between all nodes in the cluster,
then performs agglomerative hierarchical clustering to group
nodes by proximity. This drives:

1. Pipeline stage assignment — low-latency nodes are placed in
   adjacent pipeline stages to minimize forwarding overhead.
2. Shard adjacency — nodes in the same cluster get contiguous
   layer ranges to minimize cross-cluster tensor transfers.
3. Bandwidth-aware routing — within-cluster communication uses
   direct TCP; cross-cluster uses compressed/quantized forwarding.
"""

from __future__ import annotations

import asyncio
import time
import struct
import random
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional

# ──────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────


@dataclass
class LatencyEntry:
    """A single N→M measurement."""
    from_node: str
    to_node: str
    latency_ms: float
    jitter_ms: float = 0.0
    bandwidth_mbps: float = 0.0
    packet_loss_pct: float = 0.0
    timestamp: float = 0.0

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


@dataclass
class LatencyMatrix:
    """N×N matrix of pairwise latency measurements.

    Structure: entries[from_node][to_node] = LatencyEntry
    All iteration code expects this nested dict structure.
    """
    nodes: List[str] = field(default_factory=list)
    entries: Dict[str, Dict[str, LatencyEntry]] = field(default_factory=dict)

    def get(self, from_node: str, to_node: str) -> Optional[LatencyEntry]:
        return self.entries.get(from_node, {}).get(to_node)

    def set(self, entry: LatencyEntry):
        if entry.from_node not in self.entries:
            self.entries[entry.from_node] = {}
        self.entries[entry.from_node][entry.to_node] = entry
        if entry.from_node not in self.nodes:
            self.nodes.append(entry.from_node)
        if entry.to_node not in self.nodes:
            self.nodes.append(entry.to_node)

    def average_latency_to(self, node_id: str) -> float:
        """Average latency from all other nodes to this node."""
        entries = []
        for f, to_entries in self.entries.items():
            for t, e in to_entries.items():
                if t == node_id and f != node_id:
                    entries.append(e.latency_ms)
        if not entries:
            return 0.0
        return sum(entries) / len(entries)

    def to_dict(self) -> dict:
        return {
            "nodes": self.nodes,
            "entries": {
                f"{f}->{t}": {
                    "latency_ms": e.latency_ms,
                    "jitter_ms": e.jitter_ms,
                    "bandwidth_mbps": e.bandwidth_mbps,
                    "packet_loss_pct": e.packet_loss_pct,
                }
                for f, to_entries in self.entries.items()
                for t, e in to_entries.items()
            }
        }

    @classmethod
    def from_dict(cls, data: dict) -> "LatencyMatrix":
        m = cls()
        m.nodes = list(data.get("nodes", []))
        for key, val in data.get("entries", {}).items():
            f, t = key.split("->")
            if f not in m.entries:
                m.entries[f] = {}
            m.entries[f][t] = LatencyEntry(
                from_node=f,
                to_node=t,
                latency_ms=val["latency_ms"],
                jitter_ms=val.get("jitter_ms", 0),
                bandwidth_mbps=val.get("bandwidth_mbps", 0),
                packet_loss_pct=val.get("packet_loss_pct", 0),
            )
        return m


@dataclass
class Cluster:
    """A group of nodes with low inter-node latency."""
    cluster_id: int
    nodes: List[str] = field(default_factory=list)
    centroid_latency_ms: float = 0.0
    intra_latency_ms: float = 0.0  # max latency within cluster


# ──────────────────────────────────────────────────────────
# Probing
# ──────────────────────────────────────────────────────────


class LatencyProber:
    """
    Measures latency and bandwidth between nodes using lightweight
    TCP ping-pong probes. Does NOT require root or special privileges.
    """

    # 64 KiB probe payload: large enough that the round-trip is dominated by
    # transfer time (serialization + echo) instead of the latency floor, so
    # the bandwidth estimate reflects real link throughput rather than a
    # single-1KB-RTT proxy. The P2P bridge reads this constant dynamically
    # and echoes whatever size the prober sends.
    PROBE_SIZE = 65536  # 64 KiB probe packets
    PROBE_COUNT = 5     # number of probes per measurement
    # EWMA smoothing for bandwidth/latency across probe rounds (P1): a single
    # slow sample used to swing the mean ~20%. Per-target state lives on the
    # prober instance; first sample initializes (no warmup bias).
    EWMA_ALPHA = 0.3
    EWMA_MAX_DEVIATION = 0.5  # ignore samples >50% away until 3 agree

    def __init__(self):
        self._ewma: dict = {}  # addr -> {"bw":float,"rtt":float,"rounds":int}

    async def _probe_once(self, address: str, timeout: float):
        """One 64 KiB echo round-trip. Returns (rtt_ms, bw_mbps) or None."""
        host, port_str = address.split(":")
        port = int(port_str)
        t0 = time.perf_counter()
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(host, port), timeout=timeout
        )
        try:
            probe_data = struct.pack("!dQ", time.time(), random.getrandbits(64))
            writer.write(probe_data + b"\x00" * (self.PROBE_SIZE - len(probe_data)))
            await writer.drain()
            t1 = time.perf_counter()
            await asyncio.wait_for(
                reader.readexactly(self.PROBE_SIZE), timeout=timeout
            )
            rtt = (time.perf_counter() - t1) * 1000
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass
        transfer_time_s = rtt / 1000.0
        bw = (self.PROBE_SIZE * 8 * 2) / (transfer_time_s * 1_000_000) if transfer_time_s > 0 else 0.0
        return rtt, bw

    def _smooth(self, address: str, rtt_ms: float, bw_mbps: float):
        """EWMA update with variance gate. Returns (rtt_ms, bw_mbps)."""
        st = self._ewma.get(address)
        if st is None:
            self._ewma[address] = {"bw": bw_mbps, "rtt": rtt_ms, "rounds": 1}
            return rtt_ms, bw_mbps
        st["rounds"] += 1
        if st["bw"] > 0 and abs(bw_mbps - st["bw"]) / st["bw"] > self.EWMA_MAX_DEVIATION \
                and st["rounds"] < 3:
            return st["rtt"], st["bw"]
        a = self.EWMA_ALPHA
        st["bw"] = a * bw_mbps + (1 - a) * st["bw"]
        st["rtt"] = a * rtt_ms + (1 - a) * st["rtt"]
        return st["rtt"], st["bw"]

    async def probe(self, node_id: str, address: str, timeout: float = 2.0) -> Optional[LatencyEntry]:
        """Probe a single target node. Returns LatencyEntry or None on failure.

        The 5 samples run CONCURRENTLY (was: sequential, ~5x wall time and
        connect-serialization bias). Size/count/protocol unchanged, so the
        bridge echo path is untouched. Returned latency/bandwidth are
        EWMA-smoothed across rounds; loss/jitter computed from this round.
        """
        try:
            results = await asyncio.gather(
                *(self._probe_once(address, timeout) for _ in range(self.PROBE_COUNT)),
                return_exceptions=True)
        except Exception:
            return None
        samples = [r for r in results if isinstance(r, tuple)]
        if not samples:
            return None
        latencies = [r[0] for r in samples]
        avg_lat = sum(latencies) / len(latencies)
        raw_bw_samples = [r[1] for r in samples]
        raw_bw = sum(raw_bw_samples) / len(raw_bw_samples) if raw_bw_samples else 0.0
        sm_lat, sm_bw = self._smooth(address, avg_lat, raw_bw)

        # Jitter = stddev of latency
        if len(latencies) > 1:
            variance = sum((l - avg_lat) ** 2 for l in latencies) / len(latencies)
            jitter = variance ** 0.5
        else:
            jitter = 0.0

        # Packet loss: probes that failed
        loss = (self.PROBE_COUNT - len(latencies)) / self.PROBE_COUNT

        return LatencyEntry(
            from_node=node_id,
            to_node=address,
            latency_ms=sm_lat,
            jitter_ms=jitter,
            bandwidth_mbps=sm_bw,
            packet_loss_pct=loss * 100,
        )

    async def probe_all(
        self, node_id: str, targets: Dict[str, str], timeout: float = 2.0
    ) -> LatencyMatrix:
        """Probe all targets concurrently and build a LatencyMatrix."""
        matrix = LatencyMatrix()

        async def probe_one(tid: str, addr: str):
            entry = await self.probe(node_id, addr, timeout)
            if entry:
                entry.from_node = node_id
                entry.to_node = tid
                return entry
            return None

        tasks = [probe_one(tid, addr) for tid, addr in targets.items()]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for entry in results:
            if isinstance(entry, LatencyEntry):
                matrix.set(entry)

        return matrix


# ──────────────────────────────────────────────────────────
# Hierarchical Agglomerative Clustering
# ──────────────────────────────────────────────────────────


class HierarchicalClusterer:
    """
    Groups nodes into clusters based on intra-cluster latency.
    Uses agglomerative (bottom-up) clustering with single-linkage
    distance metric (minimum inter-node latency).

    Algorithm:
    1. Start with each node in its own cluster
    2. Repeatedly merge the two clusters with the lowest max inter-cluster latency
    3. Stop when the next merge would exceed max_intra_latency_ms
    """

    def __init__(self, max_intra_latency_ms: float = 5.0):
        self.max_intra_latency_ms = max_intra_latency_ms

    def cluster(self, matrix: LatencyMatrix) -> List[Cluster]:
        """
        Perform hierarchical clustering on the latency matrix.
        Returns clusters ordered by centroid latency.
        """
        nodes = list(matrix.nodes)
        if not nodes:
            return []

        if len(nodes) == 1:
            return [Cluster(cluster_id=0, nodes=nodes, centroid_latency_ms=0.0)]

        # Build distance matrix from latency entries
        distances: Dict[Tuple[str, str], float] = {}
        for f, to_entries in matrix.entries.items():
            for t, entry in to_entries.items():
                distances[(f, t)] = entry.latency_ms

        # Initialize: each node is its own cluster
        clusters: Dict[str, int] = {nid: i for i, nid in enumerate(nodes)}
        cluster_members: Dict[int, List[str]] = {i: [nid] for i, nid in enumerate(nodes)}

        # Compute initial inter-cluster distances
        def cluster_distance(cid_a: int, cid_b: int) -> float:
            """Maximum latency between any two nodes in the two clusters (complete linkage)."""
            members_a = cluster_members[cid_a]
            members_b = cluster_members[cid_b]
            max_dist = 0.0
            for a in members_a:
                for b in members_b:
                    d = distances.get((a, b), distances.get((b, a), 1e9))
                    max_dist = max(max_dist, d)
            return max_dist

        next_cluster_id = len(nodes)
        while len(cluster_members) > 1:
            # Find two clusters with minimum inter-cluster distance
            cids = list(cluster_members.keys())
            best_dist = float("inf")
            best_pair = (cids[0], cids[1])

            for i in range(len(cids)):
                for j in range(i + 1, len(cids)):
                    d = cluster_distance(cids[i], cids[j])
                    if d < best_dist:
                        best_dist = d
                        best_pair = (cids[i], cids[j])

            if best_dist > self.max_intra_latency_ms:
                break

            # Merge the two clusters
            cid_a, cid_b = best_pair
            merged_id = next_cluster_id
            next_cluster_id += 1
            cluster_members[merged_id] = cluster_members.pop(cid_a) + cluster_members.pop(cid_b)

            for nid in cluster_members[merged_id]:
                clusters[nid] = merged_id  # used only for cluster weight in compute_partition, cluster_id values come from Cluster.nodes

        # Build Cluster objects with re-indexed cluster_ids (0,1,2,...)
        # so they can be safely used as list indices and dict keys consistently
        result = []
        for new_id, (cid, members) in enumerate(cluster_members.items()):
            intra_lat = 0.0
            if len(members) > 1:
                lats = []
                for i in range(len(members)):
                    for j in range(i + 1, len(members)):
                        d = distances.get((members[i], members[j]), distances.get((members[j], members[i]), 0))
                        lats.append(d)
                intra_lat = max(lats) if lats else 0.0

            centroid = matrix.average_latency_to(members[0]) if members else 0.0

            result.append(Cluster(
                cluster_id=new_id,
                nodes=members,
                centroid_latency_ms=centroid,
                intra_latency_ms=intra_lat,
            ))

        # Sort by centroid latency (closest first)
        result.sort(key=lambda c: c.centroid_latency_ms)
        return result
"""
P2P checkpoint pull over the existing event mesh (T1.1 / C2).

Design (no new transport, no format change):
- Requests ride `synapse/checkpoint/request/{holder_id}`; responses ride
  `synapse/checkpoint/response/{request_id}`. Both are small JSON events on
  the versioned bridge (additive keys, old peers ignore them).
- Only metadata JSON checkpoints (.ckpt text) travel this path, guarded by
  MAX_EVENT_BYTES (default 8 MiB). Anything bigger gets an explicit
  `too_large` error — bulk weight files wait for chunked TCP (T1.2).
- Receiver sha-verifies before acceptance and confines writes to its own
  checkpoint_dir (H5); the sender never executes anything.
- Wire into FaultToleranceManager via prefer_local_then_pull(..., fetch_fn)
  where fetch_fn = make_fetch_fn(event_router, holder_id). FT stays
  transport-agnostic (injected, never imported).

Flow: survivor -> request(holder) -> holder reads manifest+file ->
response(entry, text|error) -> survivor prefers local, else verifies + persists.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
import uuid
from typing import Any, Dict, Optional, Tuple

MAX_EVENT_BYTES = 8 * 1024 * 1024

REQ_TOPIC = "synapse/checkpoint/request/%s"
RESP_TOPIC = "synapse/checkpoint/response/%s"


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class CheckpointPullServer:
    """Answer pull requests from our local FT manager over the event mesh."""

    def __init__(self, ft_manager, event_router, node_id: str,
                 max_bytes: int = MAX_EVENT_BYTES):
        self.ft = ft_manager
        self.router = event_router
        self.node_id = node_id
        self.max_bytes = max_bytes
        self._subscribed = False

    async def start(self) -> None:
        if self._subscribed:
            return
        self.router.subscribe(REQ_TOPIC % self.node_id, self._on_request)
        self._subscribed = True

    async def stop(self) -> None:
        if self._subscribed:
            try:
                self.router.unsubscribe(REQ_TOPIC % self.node_id, self._on_request)
            except Exception:
                pass
            self._subscribed = False

    async def _on_request(self, event) -> None:
        data = event.data if isinstance(event.data, dict) else {}
        if not isinstance(data, dict):
            return
        request_id = data.get("request_id")
        target = data.get("target_node_id")
        model_id = data.get("model_id")
        step = data.get("step")
        if not request_id or not target:
            return
        resp_topic = RESP_TOPIC % request_id
        try:
            manifest = self.ft.get_checkpoint_manifest(target, model_id or "")
            entry = None
            if manifest:
                files = manifest.get("files", [])
                if step is None:
                    entry = max(files, key=lambda e: (e.get("step", 0), e.get("timestamp", 0)),
                                default=None)
                else:
                    entry = next((e for e in files if e.get("step") == step), None)
            if entry is None:
                await self.router.publish(resp_topic, {"ok": False, "reason": "not_found",
                                                       "request_id": request_id})
                return
            path = entry.get("path") or ""
            # H5: confine to our own dir; manifest stores basenames only.
            full = os.path.abspath(os.path.join(
                os.path.abspath(self.ft.checkpoint_dir), os.path.basename(path)))
            base = os.path.abspath(self.ft.checkpoint_dir)
            if os.path.commonpath([base, full]) != base or not os.path.isfile(full):
                await self.router.publish(resp_topic, {"ok": False, "reason": "not_found",
                                                       "request_id": request_id})
                return
            size = os.path.getsize(full)
            if size > self.max_bytes:
                await self.router.publish(resp_topic, {
                    "ok": False, "reason": "too_large", "size": size,
                    "max_bytes": self.max_bytes, "request_id": request_id,
                    "manifest_entry": entry})
                return
            with open(full, "r", encoding="utf-8") as f:
                text = f.read()
            if _sha256_text(text) != entry.get("sha256"):
                await self.router.publish(resp_topic, {"ok": False, "reason": "corrupt",
                                                       "request_id": request_id})
                return
            await self.router.publish(resp_topic, {"ok": True, "request_id": request_id,
                                                   "manifest_entry": entry, "text": text})
        except Exception as e:
            try:
                await self.router.publish(resp_topic, {"ok": False, "reason": "error",
                                                       "detail": str(e)[:200],
                                                       "request_id": request_id})
            except Exception:
                pass


def make_fetch_fn(event_router, holder_node_id: str, timeout: float = 30.0):
    """Build the fetch_fn injected into prefer_local_then_pull().

    Returns async (node_id, model_id, step) -> Optional[(entry, text)].
    Raises nothing: transport failures resolve to None (dual-read falls
    through); explicit server errors (not_found/too_large/corrupt) also
    resolve to None, with the reason logged by the caller if needed.
    """
    async def fetch(node_id: str, model_id: Optional[str],
                    step: Optional[int]) -> Optional[Tuple[Dict[str, Any], str]]:
        request_id = uuid.uuid4().hex
        resp_topic = RESP_TOPIC % request_id
        got: Dict[str, Any] = {}
        arrived = asyncio.Event()

        def _on_resp(event) -> None:
            data = event.data if isinstance(event.data, dict) else {}
            if isinstance(data, dict) and data.get("request_id") == request_id:
                got.update(data)
                arrived.set()

        event_router.subscribe(resp_topic, _on_resp)
        try:
            await event_router.publish(REQ_TOPIC % holder_node_id, {
                "request_id": request_id, "target_node_id": node_id,
                "model_id": model_id, "step": step,
                "from_node_id": getattr(event_router, "node_id", None),
                "ts": time.time(),
            })
            try:
                await asyncio.wait_for(arrived.wait(), timeout=timeout)
            except asyncio.TimeoutError:
                return None
            if not got.get("ok"):
                return None
            entry, text = got.get("manifest_entry"), got.get("text")
            if not isinstance(entry, dict) or not isinstance(text, str):
                return None
            if _sha256_text(text) != entry.get("sha256"):
                return None
            return entry, text
        finally:
            try:
                event_router.unsubscribe(resp_topic, _on_resp)
            except Exception:
                pass

    return fetch

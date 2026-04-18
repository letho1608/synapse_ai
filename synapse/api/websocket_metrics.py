"""
WebSocket handlers for real-time training metrics streaming.
Replaces HTTP polling with WebSocket push for instant dashboard updates.
"""

import asyncio
import json
import logging
from aiohttp import web
from typing import Optional, Set

logger = logging.getLogger(__name__)


class TrainingMetricsWebSocketHandler:
    """Handles WebSocket connections for real-time training metrics."""
    
    def __init__(self, stream_manager):
        self.stream_manager = stream_manager
        self.clients: Set[web.WebSocketResponse] = set()
    
    async def handle_ws_training_metrics(self, request):
        """
        WebSocket endpoint: /v1/ws/training/metrics?job_id=<job_id>
        
        Streams real-time training progress:
        - Metrics every step (loss, accuracy, timing)
        - Sync status updates (AllReduce progress)
        - Epoch boundaries
        
        Client can:
        - Connect: ws://localhost:8000/v1/ws/training/metrics?job_id=<id>
        - Receive: {"type": "training_metrics", "data": {...}}
        - Send: {"action": "cancel"} to stop training
        """
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        
        # Get job_id from query params
        job_id = request.rel_url.query.get("job_id", "")
        if not job_id:
            await ws.close(code=1008, message="Missing job_id parameter")
            return ws
        
        self.clients.add(ws)
        logger.info(f"WebSocket connected for job {job_id}")
        
        try:
            # Subscribe to metrics stream
            async def on_metrics(payload):
                """Called when new metrics are available."""
                try:
                    if not ws.closed:
                        await ws.send_json(payload)
                except Exception as e:
                    logger.error(f"Error sending metrics: {e}")
            
            await self.stream_manager.subscribe(job_id, on_metrics)
            
            # Wait for client messages (e.g., cancel request)
            async for msg in ws:
                if msg.type == web.WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                        action = data.get("action")
                        
                        if action == "cancel":
                            logger.info(f"Received cancel request for job {job_id}")
                            # TODO: Signal training cancellation
                            # await training_cancellation_handler(job_id)
                        elif action == "pause":
                            logger.info(f"Received pause request for job {job_id}")
                        elif action == "resume":
                            logger.info(f"Received resume request for job {job_id}")
                    except json.JSONDecodeError:
                        logger.error("Invalid JSON from WebSocket client")
                
                elif msg.type == web.WSMsgType.ERROR:
                    logger.error(f"WebSocket error: {ws.exception()}")
                    break
        
        finally:
            self.clients.discard(ws)
            await self.stream_manager.unsubscribe(job_id, on_metrics)
            logger.info(f"WebSocket disconnected for job {job_id}")
        
        return ws
    
    async def handle_ws_sync_status(self, request):
        """
        WebSocket endpoint: /v1/ws/training/sync?job_id=<job_id>
        
        Streams AllReduce (gradient sync) operations:
        - Sync start/phases/completion
        - Strategy selection (Tree/Ring/Mesh)
        - Network bandwidth utilization
        - Failure recovery info
        
        Useful for real-time cluster visualization.
        """
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        
        job_id = request.rel_url.query.get("job_id", "")
        if not job_id:
            await ws.close(code=1008, message="Missing job_id parameter")
            return ws
        
        self.clients.add(ws)
        logger.info(f"Sync WebSocket connected for job {job_id}")
        
        try:
            async def on_sync(payload):
                """Called when sync status changes."""
                try:
                    if not ws.closed:
                        await ws.send_json(payload)
                except Exception as e:
                    logger.error(f"Error sending sync status: {e}")
            
            await self.stream_manager.subscribe(job_id, on_sync)
            
            async for msg in ws:
                if msg.type == web.WSMsgType.ERROR:
                    logger.error(f"Sync WebSocket error: {ws.exception()}")
                    break
        
        finally:
            self.clients.discard(ws)
            await self.stream_manager.unsubscribe(job_id, on_sync)
            logger.info(f"Sync WebSocket disconnected for job {job_id}")
        
        return ws
    
    async def broadcast_to_all(self, payload: dict) -> None:
        """Broadcast a message to all connected clients."""
        disconnected = set()
        for ws in self.clients:
            try:
                if not ws.closed:
                    await ws.send_json(payload)
            except Exception as e:
                logger.error(f"Broadcast error: {e}")
                disconnected.add(ws)
        
        # Clean up disconnected clients
        self.clients -= disconnected


class MetricsHTTPHandler:
    """HTTP endpoints for retrieving metrics history."""
    
    def __init__(self, stream_manager):
        self.stream_manager = stream_manager
    
    async def handle_get_metrics_history(self, request):
        """
        GET /v1/training/metrics/history?job_id=<job_id>&limit=100
        
        Returns: List of recent metrics frames (for dashboard restore on reconnect)
        """
        job_id = request.rel_url.query.get("job_id", "")
        limit = int(request.rel_url.query.get("limit", "100"))
        
        if not job_id:
            return web.json_response(
                {"success": False, "message": "Missing job_id"},
                status=400
            )
        
        history = await self.stream_manager.get_metrics_history(job_id, limit)
        return web.json_response({
            "success": True,
            "job_id": job_id,
            "count": len(history),
            "data": history
        })
    
    async def handle_get_sync_history(self, request):
        """
        GET /v1/training/sync/history?job_id=<job_id>&limit=50
        
        Returns: List of recent sync (AllReduce) operations
        """
        job_id = request.rel_url.query.get("job_id", "")
        limit = int(request.rel_url.query.get("limit", "50"))
        
        if not job_id:
            return web.json_response(
                {"success": False, "message": "Missing job_id"},
                status=400
            )
        
        history = await self.stream_manager.get_sync_history(job_id, limit)
        return web.json_response({
            "success": True,
            "job_id": job_id,
            "count": len(history),
            "data": history
        })


def setup_websocket_routes(app, stream_manager):
    """Register WebSocket and metrics HTTP routes."""
    ws_handler = TrainingMetricsWebSocketHandler(stream_manager)
    http_handler = MetricsHTTPHandler(stream_manager)
    
    # WebSocket routes (no CORS needed for WebSocket)
    app.router.add_get("/v1/ws/training/metrics", ws_handler.handle_ws_training_metrics)
    app.router.add_get("/v1/ws/training/sync", ws_handler.handle_ws_sync_status)
    
    # HTTP history endpoints (with CORS)
    # These would be registered in the main API with CORS like other endpoints
    return ws_handler, http_handler

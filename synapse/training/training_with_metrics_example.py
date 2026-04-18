"""
Example integration of real-time metrics streaming with training.
Shows how to use MetricsStreamManager, TrainingMetricsCollector, and WebSocket.
"""

import asyncio
import time
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class TrainingJobWithMetrics:
    """
    Example training job that emits real-time metrics.
    This would replace the current run_finetune_sync() function.
    """
    
    def __init__(
        self,
        job_config: Dict[str, Any],
        stream_manager,
        metrics_collector,
        sync_tracker
    ):
        self.job_config = job_config
        self.job_id = job_config['job_id']
        self.stream_manager = stream_manager
        self.metrics_collector = metrics_collector
        self.sync_tracker = sync_tracker
        self.cancel_requested = False
    
    async def run(self) -> Dict[str, Any]:
        """
        Run training with real-time metrics streaming.
        
        Replaces blocking run_finetune_sync() with async version
        that emits metrics every step.
        """
        try:
            # Subscribe self as a metrics receiver
            await self.stream_manager.subscribe(
                self.job_id,
                self._on_metrics_received
            )
            
            # Train with metrics emission
            result = await self._train_loop()
            
            return {
                "success": True,
                "job_id": self.job_id,
                **result
            }
        
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return {
                "success": False,
                "job_id": self.job_id,
                "error": str(e)
            }
        
        finally:
            await self.stream_manager.unsubscribe(
                self.job_id,
                self._on_metrics_received
            )
    
    async def _train_loop(self) -> Dict[str, Any]:
        """Main training loop with metrics emission."""
        # Placeholder for actual training logic
        # This would be adapted from run_finetune_sync()
        
        epochs = self.job_config.get('epochs', 3)
        batch_size = self.job_config.get('batch_size', 4)
        max_steps = self.job_config.get('max_steps', 0)
        learning_rate = float(self.job_config.get('learning_rate', '1e-5'))
        
        total_steps = max_steps or (1000 // batch_size) * epochs
        step_count = 0
        
        start_time = time.time()
        
        for epoch in range(epochs):
            if self.cancel_requested:
                break
            
            epoch_loss = 0.0
            
            for step in range(100):  # Simulate batches
                if self.cancel_requested:
                    break
                
                # Simulate forward pass (120ms)
                await asyncio.sleep(0.12)
                loss = 2.5 - (step + epoch * 100) * 0.005  # Simulated loss decay
                
                # Simulate backward pass (95ms)
                await asyncio.sleep(0.095)
                
                # Simulate AllReduce sync (78ms)
                sync_id = await self.sync_tracker.start_sync('tree', 3, 256)
                await asyncio.sleep(0.078)
                await self.sync_tracker.complete_sync(sync_id, 78.0)
                
                # Record metrics
                step_count += 1
                progress_pct = min(100, int(100 * step_count / total_steps))
                
                await self.metrics_collector.emit_step_metrics(
                    epoch=epoch,
                    step=step_count,
                    loss=loss,
                    progress_pct=progress_pct,
                    lr=learning_rate,
                    sync_strategy='tree',
                    sync_bytes=256 * 1024 * 1024,
                    tokens_per_sec=15.3
                )
                
                epoch_loss = loss
                
                if max_steps > 0 and step_count >= max_steps:
                    break
            
            logger.info(f"Epoch {epoch+1} completed: loss={epoch_loss:.4f}")
        
        elapsed = time.time() - start_time
        return {
            "final_loss": epoch_loss,
            "total_steps": step_count,
            "elapsed_seconds": elapsed
        }
    
    async def _on_metrics_received(self, payload: Dict) -> None:
        """Called when metrics are received (for logging/monitoring)."""
        if payload['type'] == 'training_metrics':
            data = payload['data']
            logger.debug(
                f"Step {data['step']}: "
                f"loss={data['loss']:.4f}, "
                f"sync={data['sync_ms']:.1f}ms"
            )
    
    def request_cancel(self):
        """Request graceful training cancellation."""
        self.cancel_requested = True


# ============================================================================
# EXAMPLE USAGE IN API ENDPOINT
# ============================================================================

async def handle_post_training_with_metrics(
    request,
    api_instance
):
    """
    POST /v1/training/start endpoint with real-time metrics.
    
    Replaces the old blocking handler that couldn't emit metrics.
    """
    from synapse.api.metrics_streaming import (
        TrainingMetricsCollector,
        SyncStatusTracker,
        get_metrics_stream_manager
    )
    
    data = await request.json()
    job_id = data.get('job_id') or f"job_{int(time.time() * 1000)}"
    
    # Create job config
    job_config = {
        'job_id': job_id,
        'model': data.get('model'),
        'dataset': data.get('dataset'),
        'epochs': data.get('epochs', 3),
        'batch_size': data.get('batch_size', 4),
        'learning_rate': data.get('learning_rate', '1e-5'),
        'max_steps': data.get('max_steps', 0),
        # ... other params
    }
    
    # Get stream manager
    stream_manager = get_metrics_stream_manager()
    
    # Create collectors
    metrics_collector = TrainingMetricsCollector(job_id, stream_manager)
    sync_tracker = SyncStatusTracker(job_id, stream_manager)
    
    # Create training job
    training_job = TrainingJobWithMetrics(
        job_config,
        stream_manager,
        metrics_collector,
        sync_tracker
    )
    
    # Run in background (non-blocking)
    task = asyncio.create_task(training_job.run())
    
    # Store for later reference
    api_instance.training_jobs[job_id] = {
        'task': task,
        'job': training_job,
        'config': job_config
    }
    
    # Return immediately (don't wait for training to complete)
    return web.json_response({
        'success': True,
        'job_id': job_id,
        'message': 'Training started. Connect to WebSocket for real-time metrics.'
    })


# ============================================================================
# DASHBOARD JAVASCRIPT INTEGRATION EXAMPLE
# ============================================================================

DASHBOARD_INTEGRATION_JS = """
// Connect to real-time training metrics
async function startTrainingWithMetrics(trainingConfig) {
    try {
        // Start training job
        const startRes = await fetch('/v1/training/start', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(trainingConfig)
        });
        const startData = await startRes.json();
        const jobId = startData.job_id;
        
        // Connect to WebSocket for real-time metrics
        connectToTrainingMetrics(jobId);
        
    } catch (error) {
        console.error('Failed to start training:', error);
    }
}

function connectToTrainingMetrics(jobId) {
    const ws = new WebSocket(`ws://localhost:8000/v1/ws/training/metrics?job_id=${jobId}`);
    
    ws.onopen = () => {
        console.log('Connected to training metrics stream');
    };
    
    ws.onmessage = (event) => {
        const payload = JSON.parse(event.data);
        
        if (payload.type === 'training_metrics') {
            const metrics = payload.data;
            
            // Update dashboard in real-time (no 10-second delay!)
            updateTrainingProgress({
                epoch: metrics.epoch,
                step: metrics.step,
                loss: metrics.loss,
                progress_pct: metrics.progress_pct,
                forward_ms: metrics.forward_ms,
                backward_ms: metrics.backward_ms,
                sync_ms: metrics.sync_ms,
                sync_strategy: metrics.sync_strategy
            });
        }
    };
    
    ws.onerror = (error) => {
        console.error('WebSocket error:', error);
    };
    
    ws.onclose = () => {
        console.log('Disconnected from training metrics stream');
        // Could implement auto-reconnect here
    };
    
    // Store reference for later use (cancel, pause, etc.)
    window.trainingWebSocket = ws;
}

function updateTrainingProgress(metrics) {
    // Update progress bar
    document.querySelector('.progress-bar').style.width = metrics.progress_pct + '%';
    
    // Update metrics display
    document.querySelector('[data-epoch]').textContent = metrics.epoch;
    document.querySelector('[data-step]').textContent = metrics.step;
    document.querySelector('[data-loss]').textContent = metrics.loss.toFixed(4);
    
    // Update timing breakdown
    document.querySelector('[data-forward-ms]').textContent = metrics.forward_ms.toFixed(1);
    document.querySelector('[data-backward-ms]').textContent = metrics.backward_ms.toFixed(1);
    document.querySelector('[data-sync-ms]').textContent = metrics.sync_ms.toFixed(1);
    
    // Update sync strategy
    if (metrics.sync_strategy) {
        document.querySelector('[data-sync-strategy]').textContent = metrics.sync_strategy;
    }
}

// Cancel training
document.querySelector('.btn-cancel-training').addEventListener('click', () => {
    if (window.trainingWebSocket) {
        window.trainingWebSocket.send(JSON.stringify({ action: 'cancel' }));
    }
});
"""


if __name__ == '__main__':
    # Example test
    print("Training Job with Metrics Integration")
    print("=" * 50)
    print(DASHBOARD_INTEGRATION_JS)

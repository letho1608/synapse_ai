"""
Distributed Checkpoint Manager
Implements checkpointing with distributed backup for fault tolerance
"""

import asyncio
import json
import time
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import hashlib


@dataclass
class CheckpointMetadata:
    """Metadata for checkpoint"""
    step: int
    model_id: str
    timestamp: float
    node_rank: int
    world_size: int
    shard_mapping: Dict[int, Tuple[int, int]]  # rank -> (start_layer, end_layer)
    data_hash: str  # SHA256 of checkpoint data for integrity check
    
    def to_dict(self) -> Dict:
        return asdict(self)


class DistributedCheckpointManager:
    """
    Manages distributed checkpointing with cross-node replication.
    
    Replication Strategy:
    - Each node saves its shard locally
    - Replicates to next K nodes in ring (K = replication_factor)
    - Supports recovery from single node failure
    
    Example with 4 nodes, replication_factor=2:
        Node 0 checkpoint → backed up to Node 1, 2
        Node 1 checkpoint → backed up to Node 2, 3
        Node 2 checkpoint → backed up to Node 3, 0
        Node 3 checkpoint → backed up to Node 0, 1
    
    If Node 1 fails: Can recover from Node 0 or Node 2 backups
    """
    
    def __init__(
        self, 
        node_id: str,
        ring_rank: int,
        ring_world_size: int,
        replication_factor: int = 2,
        checkpoint_dir: str = "./checkpoints"
    ):
        self.node_id = node_id
        self.ring_rank = ring_rank
        self.ring_world_size = ring_world_size
        self.replication_factor = min(replication_factor, ring_world_size - 1)
        
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.metadata_dir = self.checkpoint_dir / "metadata"
        self.metadata_dir.mkdir(exist_ok=True)
        
        # Track which checkpoints are available locally
        self.available_checkpoints: Dict[str, list] = {}  # model_id -> [steps]
        self._load_checkpoint_inventory()
    
    def _load_checkpoint_inventory(self):
        """Load list of available checkpoints from disk"""
        for ckpt_path in self.checkpoint_dir.glob("*.ckpt"):
            try:
                # Parse filename: {model_id}_step{step}_rank{rank}.ckpt
                parts = ckpt_path.stem.split("_rank")
                if len(parts) == 2:
                    prefix = parts[0]
                    rank = int(parts[1])
                    
                    # Extract model_id and step from prefix
                    step_parts = prefix.rsplit("_step", 1)
                    if len(step_parts) == 2:
                        model_id = step_parts[0]
                        step = int(step_parts[1])
                        
                        if model_id not in self.available_checkpoints:
                            self.available_checkpoints[model_id] = []
                        
                        if step not in self.available_checkpoints[model_id]:
                            self.available_checkpoints[model_id].append(step)
            except Exception:
                pass  # Skip malformed filenames
        
        # Sort steps for each model
        for model_id in self.available_checkpoints:
            self.available_checkpoints[model_id].sort()
    
    @staticmethod
    def _compute_data_hash(data: np.ndarray) -> str:
        """Compute SHA256 hash of checkpoint data"""
        hash_obj = hashlib.sha256()
        hash_obj.update(data.tobytes())
        return hash_obj.hexdigest()
    
    def _get_checkpoint_path(
        self, 
        model_id: str, 
        step: int, 
        source_rank: int
    ) -> Path:
        """Get path for checkpoint file"""
        filename = f"{model_id}_step{step}_rank{source_rank}.ckpt"
        return self.checkpoint_dir / filename
    
    def _get_metadata_path(
        self, 
        model_id: str, 
        step: int, 
        source_rank: int
    ) -> Path:
        """Get path for checkpoint metadata"""
        filename = f"{model_id}_step{step}_rank{source_rank}.json"
        return self.metadata_dir / filename
    
    async def save_checkpoint(
        self,
        model_id: str,
        step: int,
        shard_data: np.ndarray,
        shard_mapping: Dict[int, Tuple[int, int]],
        backup_replica_callback=None
    ) -> bool:
        """
        Save checkpoint locally and asynchronously replicate to backup nodes.
        
        Args:
            model_id: Model identifier
            step: Training step
            shard_data: Numpy array containing this node's shard weights
            shard_mapping: Dict mapping ring rank to (start_layer, end_layer)
            backup_replica_callback: Async function to call for replication
        
        Returns:
            True if local save successful (backup is async)
        """
        try:
            # Compute hash for integrity
            data_hash = self._compute_data_hash(shard_data)
            
            # Create metadata
            metadata = CheckpointMetadata(
                step=step,
                model_id=model_id,
                timestamp=time.time(),
                node_rank=self.ring_rank,
                world_size=self.ring_world_size,
                shard_mapping=shard_mapping,
                data_hash=data_hash
            )
            
            # Phase 1: Save locally
            checkpoint_path = self._get_checkpoint_path(model_id, step, self.ring_rank)
            metadata_path = self._get_metadata_path(model_id, step, self.ring_rank)
            
            # Save data
            np.save(checkpoint_path, shard_data)
            
            # Save metadata
            with open(metadata_path, 'w') as f:
                json.dump(metadata.to_dict(), f, indent=2)
            
            # Update inventory
            if model_id not in self.available_checkpoints:
                self.available_checkpoints[model_id] = []
            if step not in self.available_checkpoints[model_id]:
                self.available_checkpoints[model_id].append(step)
            
            # Phase 2: Replicate to backup nodes (async, non-blocking)
            if backup_replica_callback and self.replication_factor > 0:
                asyncio.create_task(
                    self._replicate_to_backups(
                        model_id, step, shard_data, metadata,
                        backup_replica_callback
                    )
                )
            
            return True
        
        except Exception as e:
            print(f"❌ Checkpoint save failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def _replicate_to_backups(
        self,
        model_id: str,
        step: int,
        shard_data: np.ndarray,
        metadata: CheckpointMetadata,
        backup_replica_callback
    ):
        """Replicate checkpoint to backup nodes"""
        tasks = []
        
        for offset in range(1, self.replication_factor + 1):
            backup_rank = (self.ring_rank + offset) % self.ring_world_size
            
            # Schedule replication
            task = asyncio.create_task(
                backup_replica_callback(
                    backup_rank,
                    model_id,
                    step,
                    shard_data,
                    metadata
                )
            )
            tasks.append(task)
        
        # Wait for replications (with timeout)
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=30.0
            )
            
            success_count = sum(1 for r in results if not isinstance(r, Exception))
            if success_count > 0:
                print(f"✓ Replicated checkpoint to {success_count}/{self.replication_factor} backups")
            else:
                print(f"⚠ No successful replications (local save OK)")
        
        except asyncio.TimeoutError:
            print(f"⚠ Replication timeout after 30s (local save OK)")
        except Exception as e:
            print(f"⚠ Replication error: {e} (local save OK)")
    
    async def store_replica(
        self,
        source_rank: int,
        model_id: str,
        step: int,
        shard_data: np.ndarray,
        metadata: CheckpointMetadata
    ) -> bool:
        """
        Store a backup replica from another node.
        Called by remote nodes to store their backups.
        """
        try:
            # Store under source_rank to distinguish from primary
            checkpoint_path = self.checkpoint_dir / f"{model_id}_step{step}_rank{source_rank}_replica.ckpt"
            metadata_path = self.metadata_dir / f"{model_id}_step{step}_rank{source_rank}_replica.json"
            
            np.save(checkpoint_path, shard_data)
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata.to_dict(), f, indent=2)
            
            return True
        
        except Exception as e:
            print(f"❌ Failed to store replica: {e}")
            return False
    
    async def load_checkpoint(
        self,
        model_id: str,
        step: int,
        required_rank: Optional[int] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Load checkpoint from local storage or backups.
        
        Args:
            model_id: Model identifier
            step: Training step
            required_rank: If specified, only load from this specific rank
        
        Returns:
            Dict with keys: 'data' (np.ndarray), 'metadata' (CheckpointMetadata)
            Returns None if not found
        """
        # Try primary checkpoint first
        if required_rank is not None:
            ranks_to_try = [required_rank]
        else:
            # Try all nodes in order
            ranks_to_try = list(range(self.ring_world_size))
        
        for rank in ranks_to_try:
            # Try primary first
            checkpoint_path = self._get_checkpoint_path(model_id, step, rank)
            if checkpoint_path.exists():
                try:
                    data = np.load(checkpoint_path)
                    
                    # Load metadata
                    metadata_path = self._get_metadata_path(model_id, step, rank)
                    if metadata_path.exists():
                        with open(metadata_path, 'r') as f:
                            metadata_dict = json.load(f)
                            metadata = CheckpointMetadata(**metadata_dict)
                    else:
                        metadata = None
                    
                    # Verify integrity
                    if metadata and metadata.data_hash != self._compute_data_hash(data):
                        print(f"⚠ Hash mismatch for {checkpoint_path}")
                        continue
                    
                    print(f"✓ Loaded checkpoint from rank {rank}")
                    return {'data': data, 'metadata': metadata}
                
                except Exception as e:
                    print(f"⚠ Failed to load {checkpoint_path}: {e}")
                    continue
            
            # Try replica if primary not found
            replica_path = self.checkpoint_dir / f"{model_id}_step{step}_rank{rank}_replica.ckpt"
            if replica_path.exists():
                try:
                    data = np.load(replica_path)
                    print(f"✓ Loaded checkpoint replica from rank {rank}")
                    return {'data': data, 'metadata': None}
                except Exception as e:
                    print(f"⚠ Failed to load replica {replica_path}: {e}")
        
        print(f"❌ Checkpoint {model_id} step {step} not found")
        return None
    
    def list_checkpoints(self, model_id: str) -> list:
        """List all available checkpoint steps for a model"""
        return sorted(self.available_checkpoints.get(model_id, []))
    
    async def cleanup_old_checkpoints(self, model_id: str, keep_last_n: int = 5):
        """Remove old checkpoints, keeping only last N"""
        steps = self.list_checkpoints(model_id)
        
        if len(steps) <= keep_last_n:
            return
        
        steps_to_remove = steps[:-keep_last_n]
        
        for step in steps_to_remove:
            try:
                # Remove all rank variants
                for rank in range(self.ring_world_size):
                    checkpoint_path = self._get_checkpoint_path(model_id, step, rank)
                    if checkpoint_path.exists():
                        checkpoint_path.unlink()
                    
                    replica_path = self.checkpoint_dir / f"{model_id}_step{step}_rank{rank}_replica.ckpt"
                    if replica_path.exists():
                        replica_path.unlink()
                    
                    metadata_path = self._get_metadata_path(model_id, step, rank)
                    if metadata_path.exists():
                        metadata_path.unlink()
                    
                    replica_metadata_path = self.metadata_dir / f"{model_id}_step{step}_rank{rank}_replica.json"
                    if replica_metadata_path.exists():
                        replica_metadata_path.unlink()
                
                self.available_checkpoints[model_id].remove(step)
            
            except Exception as e:
                print(f"⚠ Failed to remove checkpoint step {step}: {e}")

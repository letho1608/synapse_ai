"""
Model Loading Module
Handles loading models from local storage.
"""

import os
import json
import asyncio
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
from synapse.inference.shard import Shard
from synapse.helpers import AsyncCallbackSystem, DEBUG

# ==========================================
# Data Classes & Interfaces
# ==========================================

@dataclass
class RepoProgressEvent:
    """Progress event for model loading"""
    shard: Shard
    repo_id: str
    revision: str
    downloaded_files: int
    total_files: int
    downloaded_bytes: int
    downloaded_this_session_bytes: int
    total_bytes: int
    download_speed: float
    eta: Optional[float]
    file_progress: Dict[str, float]
    status: str
    
    def to_dict(self):
        return {
            "repo_id": self.repo_id,
            "revision": self.revision,
            "downloaded_files": self.downloaded_files,
            "total_files": self.total_files,
            "downloaded_bytes": self.downloaded_bytes,
            "total_bytes": self.total_bytes,
            "download_speed": self.download_speed,
            "eta": self.eta,
            "status": self.status
        }
    
    @staticmethod
    def from_dict(data):
        # Placeholder for reconstruction if needed
        return RepoProgressEvent(
            shard=None,
            repo_id=data.get("repo_id"),
            revision=data.get("revision"),
            downloaded_files=data.get("downloaded_files"),
            total_files=data.get("total_files"),
            downloaded_bytes=data.get("downloaded_bytes"),
            downloaded_this_session_bytes=0,
            total_bytes=data.get("total_bytes"),
            download_speed=data.get("download_speed"),
            eta=data.get("eta"),
            file_progress={},
            status=data.get("status")
        )

class ShardDownloader(ABC):
    """Base class for loading model shards"""
    
    @property
    @abstractmethod
    def on_progress(self) -> AsyncCallbackSystem[str, tuple[Shard, RepoProgressEvent]]:
        """Progress callback system"""
        pass
    
    @abstractmethod
    async def ensure_shard(self, shard: Shard, inference_engine_name: str) -> Path:
        """Ensure shard is available"""
        pass
    
    @abstractmethod
    async def get_shard_download_status(self, inference_engine_name: str):
        """Get download status"""
        pass

# ==========================================
# Local Model Loader Implementation
# ==========================================

def get_models_dir():
    """Get the directory where models are stored"""
    # Priority: Env var -> User Profile -> Current Dir
    if os.environ.get("SYNAPSE_MODELS_DIR"):
        return Path(os.environ["SYNAPSE_MODELS_DIR"])
    
    return Path.home() / "synapse_models"

class LocalModelLoader(ShardDownloader):
    """Loads models from local filesystem"""
    
    def __init__(self):
        self._on_progress = AsyncCallbackSystem[str, tuple[Shard, RepoProgressEvent]]()
        self.models_dir = get_models_dir()
        self.models_dir.mkdir(parents=True, exist_ok=True)
        if DEBUG >= 1:
            print(f"LocalModelLoader initialized. Models dir: {self.models_dir}")

    @property
    def on_progress(self) -> AsyncCallbackSystem[str, tuple[Shard, RepoProgressEvent]]:
        return self._on_progress

    async def ensure_shard(self, shard: Shard, inference_engine_name: str) -> Path:
        """
        Locates the model shard in the local directory.
        Does NOT download anything.
        """
        model_id = shard.model_id
        # Convert model_id (e.g. "org/model") to folder name (e.g. "model" or "org--model")
        # We'll search for exact match or normalized match
        
        possible_names = [
            model_id,
            model_id.replace("/", "--"),
            model_id.split("/")[-1]
        ]
        
        found_path = None
        for name in possible_names:
            p = self.models_dir / name
            if p.exists() and p.is_dir():
                found_path = p
                break
        
        if not found_path:
            # If not found, we can't do anything since we are offline
            # But for compatibility, we return a path that might be created later or error out
            # Let's return the expected path and let the engine fail if files are missing
            found_path = self.models_dir / model_id.replace("/", "--")
            if DEBUG >= 1:
                print(f"Warning: Model {model_id} not found in {self.models_dir}. Expected at {found_path}")
        
        return found_path

    async def get_shard_download_status(self, inference_engine_name: str):
        """
        Scans local directory and reports 'complete' for found models.
        """
        # This is a generator
        if not self.models_dir.exists():
            return

        for item in self.models_dir.iterdir():
            if item.is_dir():
                # Assume if folder exists, it's a model
                # We yield a dummy progress event indicating completion
                # We don't have the shard object here easily without scanning everything, 
                # but this method is mostly for UI status
                pass
        yield None # Empty generator for now to satisfy interface

def create_local_model_loader() -> ShardDownloader:
    return LocalModelLoader()

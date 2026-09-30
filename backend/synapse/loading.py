"""
Model Loading Module
Handles loading models from HuggingFace cache.
"""

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict
from dataclasses import dataclass
from synapse.inference.shard import Shard
from synapse.helpers import AsyncCallbackSystem


@dataclass
class RepoProgressEvent:
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
            "repo_id": self.repo_id, "revision": self.revision,
            "downloaded_files": self.downloaded_files, "total_files": self.total_files,
            "downloaded_bytes": self.downloaded_bytes, "total_bytes": self.total_bytes,
            "download_speed": self.download_speed, "eta": self.eta, "status": self.status
        }

    @staticmethod
    def from_dict(data):
        return RepoProgressEvent(
            shard=None, repo_id=data.get("repo_id"), revision=data.get("revision"),
            downloaded_files=data.get("downloaded_files"), total_files=data.get("total_files"),
            downloaded_bytes=data.get("downloaded_bytes"),
            downloaded_this_session_bytes=0, total_bytes=data.get("total_bytes"),
            download_speed=data.get("download_speed"), eta=data.get("eta"),
            file_progress={}, status=data.get("status")
        )


class ShardDownloader(ABC):

    @property
    @abstractmethod
    def on_progress(self) -> AsyncCallbackSystem[str, tuple[Shard, RepoProgressEvent]]:
        pass

    @abstractmethod
    async def ensure_shard(self, shard: Shard, inference_engine_name: str) -> Path:
        pass

    @abstractmethod
    async def get_shard_download_status(self, inference_engine_name: str):
        pass


def get_models_dir():
    if os.environ.get("SYNAPSE_MODELS_DIR"):
        return Path(os.environ["SYNAPSE_MODELS_DIR"])
    return Path.home() / ".cache" / "huggingface" / "hub"

def get_checkpoints_base_dir():
    """Lấy thư mục gốc chứa các checkpoints (models/adapters)."""
    if os.environ.get("SYNAPSE_CHECKPOINTS_DIR"):
        return Path(os.environ["SYNAPSE_CHECKPOINTS_DIR"])
    p = Path("backend") / "checkpoints"
    if not p.exists():
        p = Path("checkpoints")
    p.mkdir(parents=True, exist_ok=True)
    return p

def get_checkpoints_dir(model_id: str = ""):
    base = Path.home() / ".cache" / "huggingface" / "hub"
    if model_id:
        return base / ("models--" + model_id.replace("/", "--")) / "checkpoints"
    return base
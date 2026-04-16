"""
Disk Embeddings Cache - Cache embeddings lên SSD/Disk
"""
import hashlib
import json
from pathlib import Path
from typing import Optional
import numpy as np
import diskcache
from collections import OrderedDict


class DiskEmbeddingsCache:
    """
    LRU Cache lưu embeddings trên disk.
    Dùng cho prompt embeddings hay dùng.
    """
    
    def __init__(self, cache_dir: str, max_size_gb: int = 50):
        """
        Args:
            cache_dir: Thư mục lưu cache
            max_size_gb: Dung lượng tối đa trên disk (GB)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size = max_size_gb * 1024 * 1024 * 1024  # bytes
        self.lru: OrderedDict[str, int] = OrderedDict()  # key -> size
        self._load_lru_index()
    
    def _get_cache_key(self, prompt: str) -> str:
        """Tạo cache key từ prompt"""
        return hashlib.sha256(prompt.encode()).hexdigest()
    
    def _get_cache_path(self, key: str) -> Path:
        """Get path cho cache file"""
        return self.cache_dir / f"{key}.npy"
    
    def _load_lru_index(self) -> None:
        """Load LRU index từ disk"""
        index_file = self.cache_dir / ".lru_index.json"
        if index_file.exists():
            try:
                with open(index_file, 'r') as f:
                    self.lru = OrderedDict(json.load(f))
            except (json.JSONDecodeError, IOError):
                self.lru = OrderedDict()
    
    def _save_lru_index(self) -> None:
        """Save LRU index ra disk"""
        index_file = self.cache_dir / ".lru_index.json"
        try:
            with open(index_file, 'w') as f:
                json.dump(list(self.lru.items()), f)
        except IOError:
            pass
    
    def _get_total_size(self) -> int:
        """Tính tổng size của cache"""
        return sum(self.lru.values())
    
    def _evict_lru_entries(self, needed_size: int) -> None:
        """Evict LRU entries cho đến khi có đủ space"""
        while self.lru and self._get_total_size() + needed_size > self.max_size:
            oldest_key = next(iter(self.lru))
            cache_path = self._get_cache_path(oldest_key)
            if cache_path.exists():
                cache_path.unlink()
            del self.lru[oldest_key]
        self._save_lru_index()
    
    def get(self, prompt: str) -> Optional[np.ndarray]:
        """
        Lấy embeddings từ cache.
        
        Args:
            prompt: Prompt text
            
        Returns:
            Embeddings array hoặc None nếu không có
        """
        key = self._get_cache_key(prompt)
        cache_path = self._get_cache_path(key)
        
        if cache_path.exists():
            # Update LRU
            if key in self.lru:
                self.lru.move_to_end(key)
            else:
                self.lru[key] = cache_path.stat().st_size
            self._save_lru_index()
            return np.load(cache_path)
        return None
    
    def set(self, prompt: str, embeddings: np.ndarray) -> None:
        """
        Lưu embeddings vào cache.
        
        Args:
            prompt: Prompt text
            embeddings: Embeddings array
        """
        key = self._get_cache_key(prompt)
        cache_path = self._get_cache_path(key)
        
        embeddings_size = embeddings.nbytes
        
        # Check size limit
        if self._get_total_size() + embeddings_size > self.max_size:
            self._evict_lru_entries(embeddings_size)
        
        # Save to disk
        np.save(cache_path, embeddings)
        self.lru[key] = embeddings_size
        self._save_lru_index()
    
    def has(self, prompt: str) -> bool:
        """Kiểm tra xem prompt có trong cache không"""
        key = self._get_cache_key(prompt)
        return self._get_cache_path(key).exists()
    
    def clear(self) -> None:
        """Clear tất cả cache"""
        for key in list(self.lru.keys()):
            cache_path = self._get_cache_path(key)
            if cache_path.exists():
                cache_path.unlink()
        self.lru.clear()
        self._save_lru_index()


class DiskEmbeddingsCacheManager:
    """
    Quản lý embeddings cache trên disk.
    """
    
    def __init__(self, cache_dir: str = "synapse/data/.embeddings_cache", max_size_gb: int = 50):
        """
        Args:
            cache_dir: Thư mục lưu cache
            max_size_gb: Dung lượng tối đa (GB)
        """
        self.cache = DiskEmbeddingsCache(cache_dir, max_size_gb)
    
    def get(self, prompt: str) -> Optional[np.ndarray]:
        """Lấy embeddings từ cache"""
        return self.cache.get(prompt)
    
    def set(self, prompt: str, embeddings: np.ndarray) -> None:
        """Lưu embeddings vào cache"""
        self.cache.set(prompt, embeddings)
    
    def has(self, prompt: str) -> bool:
        """Kiểm tra prompt có trong cache không"""
        return self.cache.has(prompt)
    
    def clear(self) -> None:
        """Clear cache"""
        self.cache.clear()
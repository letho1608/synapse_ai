"""
CPU RAM KV Cache - Lưu KV cache vào CPU RAM khi GPU VRAM đầy
"""
from typing import Optional, Dict, Any
import numpy as np
from collections import OrderedDict
import threading


class CPURAMCache:
    """
    LRU Cache lưu trên CPU RAM.
    Dùng khi GPU VRAM đầy để overflow KV cache.
    """
    
    def __init__(self, max_size_mb: int):
        """
        Args:
            max_size_mb: Dung lượng tối đa cache trên RAM (MB)
        """
        self.max_size = max_size_mb * 1024 * 1024  # Convert to bytes
        self.current_size = 0
        self.cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._lock = threading.Lock()
    
    def _get_size(self, arr: np.ndarray) -> int:
        """Tính size của array in bytes"""
        return arr.nbytes
    
    def has(self, key: str) -> bool:
        with self._lock:
            return key in self.cache
    
    def get(self, key: str) -> Optional[np.ndarray]:
        with self._lock:
            if key in self.cache:
                # Move to end (most recently used)
                self.cache.move_to_end(key)
                return self.cache[key]
            return None
    
    def set(self, key: str, value: np.ndarray) -> None:
        with self._lock:
            # Remove old entry if exists
            if key in self.cache:
                old_size = self._get_size(self.cache[key])
                self.current_size -= old_size
                del self.cache[key]
            
            value_size = self._get_size(value)
            
            # Evict oldest entries if needed
            while self.current_size + value_size > self.max_size and self.cache:
                oldest_key = next(iter(self.cache))
                oldest_size = self._get_size(self.cache[oldest_key])
                del self.cache[oldest_key]
                self.current_size -= oldest_size
            
            # Add new entry
            if value_size <= self.max_size:
                self.cache[key] = value
                self.current_size += value_size
    
    def evict_oldest(self) -> Optional[tuple]:
        """Evict và return oldest entry"""
        with self._lock:
            if self.cache:
                oldest_key = next(iter(self.cache))
                oldest_value = self.cache[oldest_key]
                oldest_size = self._get_size(oldest_value)
                del self.cache[oldest_key]
                self.current_size -= oldest_size
                return (oldest_key, oldest_value)
            return None
    
    def usage(self) -> float:
        """Trả về % cache đã sử dụng"""
        if self.max_size == 0:
            return 1.0
        return self.current_size / self.max_size
    
    def clear(self) -> None:
        with self._lock:
            self.cache.clear()
            self.current_size = 0


class CPURAMKVCacheManager:
    """
    Quản lý KV cache với CPU RAM overflow.
    
    Khi GPU VRAM đạt threshold (90%), chuyển KV cache cũ nhất sang CPU RAM.
    """
    
    def __init__(self, max_cpu_ram_gb: int = 64, gpu_threshold: float = 0.9):
        """
        Args:
            max_cpu_ram_gb: Dung lượng CPU RAM tối đa cho cache (GB)
            gpu_threshold: Ngưỡng GPU VRAM để bắt đầu overflow (0.0 - 1.0)
        """
        self.cpu_cache = CPURAMCache(max_cpu_ram_gb * 1024)  # MB
        self.gpu_threshold = gpu_threshold
        self._lock = threading.Lock()
    
    def set_with_overflow(
        self, 
        key: str, 
        kv: np.ndarray, 
        gpu_usage: float,
        gpu_set_fn=None
    ) -> None:
        """
        Lưu KV cache, tự động overflow sang CPU RAM nếu GPU đầy.
        
        Args:
            key: Cache key
            kv: KV tensor
            gpu_usage: GPU VRAM usage hiện tại (0.0 - 1.0)
            gpu_set_fn: Function để set trực tiếp vào GPU cache
        """
        with self._lock:
            if gpu_usage < self.gpu_threshold and gpu_set_fn is not None:
                # GPU còn space, lưu trực tiếp
                gpu_set_fn(key, kv)
            else:
                # GPU đầy hoặc không có GPU set function, lưu vào CPU RAM
                self.cpu_cache.set(key, kv)
    
    def get(self, key: str, gpu_get_fn=None) -> Optional[np.ndarray]:
        """
        Lấy KV cache, ưu tiên GPU trước.
        
        Args:
            key: Cache key
            gpu_get_fn: Function để get từ GPU cache
            
        Returns:
            KV tensor hoặc None nếu không tìm thấy
        """
        # Ưu tiên GPU cache
        if gpu_get_fn is not None:
            result = gpu_get_fn(key)
            if result is not None:
                return result
        
        # Fallback CPU RAM cache
        return self.cpu_cache.get(key)
    
    def has(self, key: str, gpu_has_fn=None) -> bool:
        """Kiểm tra xem key có trong cache không"""
        if gpu_has_fn is not None and gpu_has_fn(key):
            return True
        return self.cpu_cache.has(key)
    
    def cpu_cache_usage(self) -> float:
        """% CPU RAM cache đã sử dụng"""
        return self.cpu_cache.usage()
    
    def clear(self) -> None:
        """Clear tất cả cache"""
        self.cpu_cache.clear()
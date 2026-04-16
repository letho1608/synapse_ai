"""
Resource Manager Registry - Registry cho các resource managers
"""
from typing import Dict, Optional, Any
from abc import ABC, abstractmethod


class ResourceManager(ABC):
    """
    Base class cho resource managers.
    """
    
    @abstractmethod
    def get(self, key: str, default: Any = None) -> Any:
        """Lấy resource"""
        pass
    
    @abstractmethod
    def set(self, key: str, value: Any) -> None:
        """Set resource"""
        pass
    
    @abstractmethod
    def has(self, key: str) -> bool:
        """Kiểm tra resource tồn tại"""
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear tất cả resources"""
        pass


class ResourceManagerRegistry:
    """
    Registry cho các resource managers trong LACP.
    
    Cung cấp truy cập unified đến:
    - CPU RAM KV Cache
    - Disk Embeddings Cache
    - Multi-GPU Manager
    - Preload Manager
    """
    
    def __init__(self):
        self.managers: Dict[str, ResourceManager] = {}
        self._init_default_managers()
    
    def _init_default_managers(self) -> None:
        """Khởi tạo các managers mặc định"""
        # Import here để tránh circular imports
        try:
            from .cpu_ram_kv_cache import CPURAMKVCacheManager
            from .disk_embeddings_cache import DiskEmbeddingsCacheManager
            from .multi_gpu_manager import MultiGPUManager
            from .preload_strategy import PreloadManager
            
            self.managers["cpu_ram_kv"] = CPURAMKVCacheManager()
            self.managers["disk_cache"] = DiskEmbeddingsCacheManager()
            # Multi-GPU và Preload cần config, để None làm placeholder
            self.managers["multi_gpu"] = None
            self.managers["preload"] = PreloadManager()
        except ImportError as e:
            # Dependencies chưa được cài đặt
            pass
    
    def register(self, name: str, manager: ResourceManager) -> None:
        """
        Register một resource manager.
        
        Args:
            name: Tên manager
            manager: ResourceManager instance
        """
        self.managers[name] = manager
    
    def get_manager(self, name: str) -> Optional[ResourceManager]:
        """
        Get resource manager theo tên.
        
        Args:
            name: Tên manager
            
        Returns:
            ResourceManager hoặc None
        """
        return self.managers.get(name)
    
    def get(self, manager_name: str, key: str, default: Any = None) -> Any:
        """
        Get resource từ manager.
        
        Args:
            manager_name: Tên manager
            key: Resource key
            default: Default value nếu không tìm thấy
            
        Returns:
            Resource value hoặc default
        """
        manager = self.get_manager(manager_name)
        if manager:
            return manager.get(key, default)
        return default
    
    def set(self, manager_name: str, key: str, value: Any) -> None:
        """
        Set resource vào manager.
        
        Args:
            manager_name: Tên manager
            key: Resource key
            value: Resource value
        """
        manager = self.get_manager(manager_name)
        if manager:
            manager.set(key, value)
    
    def has(self, manager_name: str, key: str) -> bool:
        """
        Kiểm tra resource tồn tại.
        
        Args:
            manager_name: Tên manager
            key: Resource key
            
        Returns:
            True nếu tồn tại
        """
        manager = self.get_manager(manager_name)
        if manager:
            return manager.has(key)
        return False
    
    def list_managers(self) -> list:
        """List tất cả registered managers"""
        return list(self.managers.keys())
    
    def clear_all(self) -> None:
        """Clear tất cả managers"""
        for manager in self.managers.values():
            if manager:
                manager.clear()
    
    def setup_multi_gpu(self, gpu_ids: list) -> None:
        """
        Setup multi-GPU manager.
        
        Args:
            gpu_ids: List of GPU IDs
        """
        from .multi_gpu_manager import MultiGPUManager
        self.managers["multi_gpu"] = MultiGPUManager(gpu_ids)
    
    def get_multi_gpu(self) -> Optional["MultiGPUManager"]:
        """Get multi-GPU manager"""
        manager = self.get_manager("multi_gpu")
        return manager if manager else None
    
    def get_cpu_ram_kv(self) -> Optional["CPURAMKVCacheManager"]:
        """Get CPU RAM KV cache manager"""
        manager = self.get_manager("cpu_ram_kv")
        return manager if manager else None
    
    def get_disk_cache(self) -> Optional["DiskEmbeddingsCacheManager"]:
        """Get disk embeddings cache manager"""
        manager = self.get_manager("disk_cache")
        return manager if manager else None
    
    def get_preload(self) -> Optional["PreloadManager"]:
        """Get preload manager"""
        manager = self.get_manager("preload")
        return manager if manager else None
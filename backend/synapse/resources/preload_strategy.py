"""
Preload Strategy - Preload model vào RAM trước khi cần
"""
from typing import Dict, Optional, Set
import torch
import torch.nn as nn
from pathlib import Path


class PreloadStrategy:
    """
    Preload model vào RAM trước, chỉ chuyển sang GPU khi cần.
    
    Giảm thời gian load layer vì đã có sẵn trong RAM.
    """
    
    def __init__(self, model: nn.Module, device: str = "cuda"):
        """
        Args:
            model: Model cần preload
            device: Device để load layers lên
        """
        self.model = model
        self.device = torch.device(device) if isinstance(device, str) else device
        self.cpu_model = model.cpu()  # Load vào RAM
        self.loaded_layers: Dict[int, nn.Module] = {}  # layer_id -> GPU location
        self.num_layers = self._count_layers(model)
    
    def _count_layers(self, model: nn.Module) -> int:
        """Đếm số layers trong model"""
        count = 0
        for _ in model.modules():
            count += 1
        return count
    
    def ensure_layer(self, layer_id: int) -> nn.Module:
        """
        Đảm bảo layer được load lên GPU.
        
        Args:
            layer_id: Layer index
            
        Returns:
            Layer module đã load trên GPU
        """
        if layer_id not in self.loaded_layers:
            # Chuyển từ RAM -> GPU
            layer = self._get_layer_by_id(layer_id)
            if layer is not None:
                self.loaded_layers[layer_id] = layer.to(self.device)
            else:
                raise ValueError(f"Layer {layer_id} not found in model")
        return self.loaded_layers[layer_id]
    
    def _get_layer_by_id(self, layer_id: int) -> Optional[nn.Module]:
        """Get layer module by ID"""
        layers = list(self.model.modules())
        if 0 <= layer_id < len(layers):
            return layers[layer_id]
        return None
    
    def preload_all(self) -> None:
        """Load tất cả layers lên GPU (nếu đủ VRAM)"""
        for layer_id in range(self.num_layers):
            try:
                self.ensure_layer(layer_id)
            except RuntimeError as e:
                # Out of memory, stop preloading
                if "out of memory" in str(e).lower():
                    print(f"GPU out of memory at layer {layer_id}, stopping preload")
                    break
                raise
    
    def unload_layer(self, layer_id: int) -> None:
        """
        Unload layer khỏi GPU, chuyển về RAM.
        
        Args:
            layer_id: Layer index
        """
        if layer_id in self.loaded_layers:
            layer = self.loaded_layers[layer_id]
            layer = layer.cpu()
            self.loaded_layers[layer_id] = layer
    
    def get_loaded_layers(self) -> Set[int]:
        """Get set of loaded layer IDs"""
        return set(self.loaded_layers.keys())
    
    def clear(self) -> None:
        """Clear tất cả loaded layers"""
        self.loaded_layers.clear()
        torch.cuda.empty_cache()


class PreloadManager:
    """
    Quản lý preload cho nhiều models.
    """
    
    def __init__(self):
        self.preload_strategies: Dict[str, PreloadStrategy] = {}
    
    def register_model(self, model_id: str, model: nn.Module, device: str = "cuda") -> PreloadStrategy:
        """
        Register model để preload.
        
        Args:
            model_id: Model identifier
            model: Model instance
            device: Device để load lên
            
        Returns:
            PreloadStrategy cho model
        """
        strategy = PreloadStrategy(model, device)
        self.preload_strategies[model_id] = strategy
        return strategy
    
    def get_strategy(self, model_id: str) -> Optional[PreloadStrategy]:
        """Get preload strategy cho model"""
        return self.preload_strategies.get(model_id)
    
    def ensure_layer(self, model_id: str, layer_id: int) -> Optional[nn.Module]:
        """
        Ensure layer được load lên GPU.
        
        Args:
            model_id: Model identifier
            layer_id: Layer index
            
        Returns:
            Layer module hoặc None
        """
        strategy = self.get_strategy(model_id)
        if strategy:
            return strategy.ensure_layer(layer_id)
        return None
    
    def preload_model(self, model_id: str) -> None:
        """Preload tất cả layers của model"""
        strategy = self.get_strategy(model_id)
        if strategy:
            strategy.preload_all()
    
    def clear_model(self, model_id: str) -> None:
        """Clear preload cho model"""
        strategy = self.get_strategy(model_id)
        if strategy:
            strategy.clear()
            del self.preload_strategies[model_id]
    
    def clear_all(self) -> None:
        """Clear tất cả preload strategies"""
        for strategy in self.preload_strategies.values():
            strategy.clear()
        self.preload_strategies.clear()
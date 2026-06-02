"""
Multi-GPU Manager - Quản lý multi-GPU trong 1 máy
"""
from typing import List, Optional
import torch
import torch.nn as nn


class MultiGPUManager:
    """
    Quản lý multi-GPU với Pipeline Parallel và Tensor Parallel.
    
    Pipeline Parallel: Chia layers cho các GPU
    Tensor Parallel: Chia weights/tensors cho các GPU (cần NVLink)
    """
    
    def __init__(self, gpu_ids: List[int], tensor_parallel: bool = False):
        """
        Args:
            gpu_ids: Danh sách GPU IDs
            tensor_parallel: Sử dụng tensor parallel thay vì pipeline parallel
        """
        self.gpu_ids = gpu_ids
        self.num_gpus = len(gpu_ids)
        self.gpus = [torch.device(f"cuda:{i}") for i in gpu_ids]
        self.tensor_parallel = tensor_parallel
        self.tensor_dim = 0  # Dimension để split tensor
    
    def get_device_for_layer(self, layer_id: int, total_layers: int) -> torch.device:
        """
        Get GPU device cho layer với pipeline parallel.
        
        Args:
            layer_id: Layer index
            total_layers: Tổng số layers
            
        Returns:
            torch.device cho layer đó
        """
        if self.num_gpus == 1:
            return self.gpus[0]
        
        layers_per_gpu = total_layers / self.num_gpus
        gpu_index = min(int(layer_id // layers_per_gpu), self.num_gpus - 1)
        return self.gpus[gpu_index]
    
    def split_tensor(self, tensor: torch.Tensor) -> List[torch.Tensor]:
        """
        Chia tensor thành N phần theo tensor_dim.
        Dùng cho tensor parallel.
        
        Args:
            tensor: Tensor cần chia
            
        Returns:
            List of tensor chunks
        """
        if self.num_gpus == 1:
            return [tensor]
        
        chunk_size = tensor.shape[self.tensor_dim] // self.num_gpus
        return torch.split(tensor, chunk_size, dim=self.tensor_dim)
    
    def allgather_tensor(self, partial_tensors: List[torch.Tensor]) -> torch.Tensor:
        """
        Gom lại các tensor parts thành 1 tensor hoàn chỉnh.
        
        Args:
            partial_tensors: List of tensor parts
            
        Returns:
            Complete tensor
        """
        if self.num_gpus == 1:
            return partial_tensors[0]
        
        return torch.cat(partial_tensors, dim=self.tensor_dim)
    
    def forward_layers_pipeline(
        self, 
        layers: List[nn.Module], 
        input_data: torch.Tensor,
        layer_start: int,
        layer_end: int
    ) -> torch.Tensor:
        """
        Forward với pipeline parallel.
        
        Args:
            layers: List of layers
            input_data: Input tensor
            layer_start: Start layer index
            layer_end: End layer index (exclusive)
            
        Returns:
            Output tensor
        """
        current = input_data
        for i in range(layer_start, layer_end):
            device = self.get_device_for_layer(i, len(layers))
            with torch.cuda.device(device):
                current = current.to(device)
                current = layers[i](current)
        return current
    
    def forward_with_tensor_parallel(
        self, 
        layer: nn.Module, 
        input_data: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward với tensor parallel.
        
        Args:
            layer: Layer module
            input_data: Input tensor
            
        Returns:
            Output tensor
        """
        if self.num_gpus == 1:
            return layer(input_data)
        
        # Split input
        input_chunks = self.split_tensor(input_data)
        
        # Parallel compute
        outputs = []
        for i, gpu in enumerate(self.gpus):
            with torch.cuda.device(gpu):
                output_chunk = layer(input_chunks[i].to(gpu))
                outputs.append(output_chunk.cpu())
        
        # All-gather
        return self.allgather_tensor(outputs)
    
    def sync_between_gpus(self) -> None:
        """Đồng bộ các GPU (barrier)"""
        if self.num_gpus > 1:
            torch.cuda.synchronize()
            for i in range(self.num_gpus):
                torch.cuda.current_device()  # Ensure sync point
    
    def get_gpu_memory_info(self) -> List[dict]:
        """Get memory info cho từng GPU"""
        info = []
        for i, gpu_id in enumerate(self.gpu_ids):
            if torch.cuda.is_available():
                with torch.cuda.device(gpu_id):
                    allocated = torch.cuda.memory_allocated(gpu_id) / 1024**3  # GB
                    reserved = torch.cuda.memory_reserved(gpu_id) / 1024**3
                    total = torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3
                    info.append({
                        "gpu_id": gpu_id,
                        "allocated_gb": allocated,
                        "reserved_gb": reserved,
                        "total_gb": total,
                        "free_gb": total - reserved
                    })
        return info
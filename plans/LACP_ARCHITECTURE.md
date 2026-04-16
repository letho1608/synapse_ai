# LACP: Latency-Aware Collaborative Partitioning
## Tài liệu thiết kế kiến trúc phân tán cho Synapse AI

---

## Mục lục
1. [Phân tích phương thức hiện tại](#1-phân-tích-phương-thức-hiện-tại)
2. [LACP: Phương pháp đề xuất](#2-lacp-phương-pháp-đề-xuất)
3. [Tận dụng tài nguyên](#3-tận-dụng-tài-nguyên)
4. [Kiến trúc chi tiết](#4-kiến-trúc-chi-tiết)
5. [Implementation](#5-implementation)
6. [So sánh](#6-so-sánh)

---

## 1. Phân tích phương thức hiện tại

### 1.1 Code hiện tại

Xem [`synapse/topology/ring_memory_weighted_partitioning_strategy.py`](synapse/topology/ring_memory_weighted_partitioning_strategy.py:7):

```python
class RingMemoryWeightedPartitioningStrategy(PartitioningStrategy):
    def partition(self, topology: Topology) -> List[Partition]:
        nodes = list(topology.all_nodes())
        
        # Ưu tiên theo FLOPS fp16
        total_flops = sum(node[1].flops.fp16 for node in nodes)
        
        if total_flops > 0:
            nodes.sort(key=lambda x: (x[1].flops.fp16, x[0]), reverse=True)
            partitions = []
            start = 0
            for node in nodes:
                share = node[1].flops.fp16 / total_flops
                end = round(start + share, 5)
                partitions.append(Partition(node[0], start, end))
                start = end
            return partitions
        
        # Fallback: chỉ dùng memory khi FLOPS = 0
        nodes.sort(key=lambda x: (x[1].memory, x[0]), reverse=True)
        total_memory = sum(node[1].memory for node in nodes)
        # ... chia theo memory
```

### 1.2 Vấn đề cụ thể

| Vấn đề | Code | Hậu quả |
|--------|------|---------|
| **Chỉ dùng 1 metric** | `if total_flops > 0` → dùng FLOPS, KHÔNG dùng memory | Máy có FLOPS cao nhưng memory thấp → OOM |
| **FLOPS fp16 là ước lượng** | `node[1].flops.fp16` từ `device_capabilities()` | FLOPS lý thuyết ≠ thực tế (thermal throttling, driver) |
| **Giả định layer đều nhau** | `share = flops / total_flops` | Attention layer nặng = FFN layer nhẹ → imbalance |
| **Không tính latency** | Không có code nào đo latency | Máy A→B latency 200ms vẫn chia như latency 1ms |
| **Static partitioning** | Chia 1 lần lúc start | Workload thay đổi → không re-balance |

### 1.3 DeviceCapabilities hiện tại

Xem [`synapse/topology/device_capabilities.py`](synapse/topology/device_capabilities.py:60):

```python
class DeviceCapabilities(BaseModel):
    model: str                    # Tên GPU
    chip: str                     # Chip
    memory: int                   # VRAM (MB)
    flops: DeviceFlops            # FLOPS fp32, fp16, int8
    cpu_cores: int = 0
    disk_gb: int = 0
    system_ram_mb: int = 0
    gpu_backend: str = "Unknown"
    gpu_count: int = 0
    total_gpu_vram_mb: int = 0
```

**Vấn đề**: Có `system_ram_mb` nhưng KHÔNG dùng trong partitioning!

### 1.4 Ví dụ thực tế vấn đề

```
3 máy:
- Máy A: RTX 4090 (165 TFLOPS fp16, 24GB VRAM)
- Máy B: RTX 3060 (25 TFLOPS fp16, 6GB VRAM)
- Máy C: Tesla T4 (65 TFLOPS fp16, 16GB VRAM)

Model 32 layers, mỗi layer cần 4GB VRAM

Chia theo FLOPS:
- A: 165/255 = 64.7% → 21 layers → cần 84GB VRAM ❌ OOM!
- B: 25/255 = 9.8% → 3 layers → cần 12GB VRAM ✅
- C: 65/255 = 25.5% → 8 layers → cần 32GB VRAM ❌ OOM!

Thực tế:
- A chỉ chạy được ~6 layers (24GB / 4GB)
- C chỉ chạy được ~4 layers (16GB / 4GB)
→ Cần chia lại theo memory!
```

---

## 2. LACP: Phương pháp đề xuất

### 2.1 Tổng quan LACP

**LACP = Latency-Aware Collaborative Partitioning**

Ý tưởng cốt lõi:
1. **Đo thực tế** latency giữa các máy qua Tailscale
2. **Nhóm máy** theo latency (clustering)
3. **Chia layers** tối ưu dùng ILP, tính cả latency + memory
4. **Tận dụng** CPU RAM + Disk + Multi-GPU

### 2.2 3 Giai đoạn chi tiết

#### Giai đoạn 1: Latency Probing

**Mục đích**: Đo latency thực giữa mọi cặp máy

**Cách hoạt động**:
```python
# Mỗi máy gửi ping tới tất cả máy khác
# Gửi 10-20 packets, tính trung bình
# Loại bỏ outliers (max, min)

Latency Matrix (N x N):
              Máy A   Máy B   Máy C
Máy A         0ms     15ms    200ms
Máy B         15ms    0ms     180ms
Máy C         200ms   180ms   0ms
```

**Tại sao cần đo qua Tailscale**:
- Tailscale có DERP relay servers
- Latency phụ thuộc vào route (có thể đi qua relay)
- Đo trực tiếp sẽ chính xác hơn ước lượng

**Thời điểm đo**:
| Sự kiện | Hành động |
|---------|-----------|
| Server start | Ping tất cả máy (1 lần) |
| Máy mới tham gia | Ping máy đó tới tất cả |
| Định kỳ | Ping lại mỗi 5-10 phút (background) |
| Máy mất kết nối | Xóa khỏi matrix |

**Tài nguyên dùng**:
- CPU: < 1% trong 1-2 giây
- RAM: ~1KB cho matrix
- Network: Vài KB traffic
- GPU: 0 (không dùng)

**Cache**:
```python
# Lưu vào file JSON
{
    "version": "1.0",
    "timestamp": "2024-01-15T10:00:00Z",
    "matrix": {
        "machine_A": {"machine_B": 15.2, "machine_C": 200.1},
        "machine_B": {"machine_A": 15.1, "machine_C": 180.5},
        "machine_C": {"machine_A": 200.3, "machine_B": 180.2}
    }
}
```

#### Giai đoạn 2: Hierarchical Clustering

**Mục đích**: Nhóm máy gần nhau về network

**Thuật toán**: Agglomerative Clustering

```python
# Bước 1: Tính distance matrix từ latency matrix
# Bước 2: Bắt đầu với mỗi máy là 1 cluster
# Bước 3: Merge 2 clusters gần nhất nhất
# Bước 4: Lặp lại cho đến khi đạt k clusters

# Ví dụ kết quả:
Cluster 1 (VN datacenter):
    - Máy A (VN)
    - Máy B (VN)
    - Latency A↔B: 15ms

Cluster 2 (US datacenter):
    - Máy C (US)
    - Máy D (US)
    - Latency C↔D: 20ms

Cluster 3 (EU datacenter):
    - Máy E (EU)
    - Latency E: 30ms
```

**Lợi ích của clustering**:
1. Layer liền nhau ưu tiên đặt **trong cùng cluster**
2. Giảm cross-datacenter traffic
3. ILP solver chạy nhanh hơn (có constraints ràng buộc)

**Ngưỡng clustering**:
```python
# Latency < 50ms → cùng cluster
# Latency >= 50ms → cluster khác
CLUSTER_LATENCY_THRESHOLD = 50  # ms
```

#### Giai đoạn 3: ILP-Based Partitioning

**Mục đích**: Tìm cách chia layers tối ưu toàn cục

**Bài toán ILP**:

```
Variables:
x[i][m] = 1 nếu layer i được gán cho machine m, 0 otherwise
y[i][j] = 1 nếu layer i và j khác máy, 0 otherwise

Objective:
Minimize sum of weighted latency cost:
Minimize Σ(latency(m1, m2) * y[i][j]) cho tất cả i, j liền nhau
+ Σ(download_score[m] * weight) cho tất cả machines

Constraints:
1. Memory constraint:
Σ(layer_size[i] * x[i][m]) <= memory[m] cho tất cả m

2. Each layer assigned to exactly one machine:
Σ(x[i][m]) = 1 cho tất cả i

3. Consecutive layers constraint:
y[i][i+1] >= x[i][m1] + x[i+1][m2] - 1 (với m1 != m2)

4. Cluster constraint (soft):
Nếu i và i+1 khác cluster → cộng penalty vào objective

5. Leaf node preference (soft):
Ưu tiên gán layers cho leaf nodes (máy ít kết nối ra ngoài cluster)
```

**Leaf Node Preference (từ Exo)**:

Leaf node = máy có ít external connections (kết nối ra ngoài cluster/datacenter).
Leaf nodes ổn định hơn vì ít bị ảnh hưởng bởi network bên ngoài.

```python
def calculate_leaf_score(machine_id: str, topology: Topology) -> float:
    """
    Tính leaf score cho 1 machine.
    Leaf node = ít neighbors ngoài cluster.
    Score càng cao = càng được ưu tiên.
    """
    machine = topology.nodes[machine_id]
    external_connections = 0
    
    for peer_id in topology.peers:
        if peer_id == machine_id:
            continue
        # Đếm kết nối ra ngoài
        if not topology.in_same_cluster(machine_id, peer_id):
            external_connections += 1
    
    # Score cao nếu ít external connections
    return 1.0 / (1.0 + external_connections)

# Áp dụng trong ILP:
# Thêm soft constraint: ưu tiên machine có leaf_score cao
# Hoặc thêm vào objective: -leaf_score[m] * weight
```

**Download Score (từ Exo)**:

Machine đã có model downloaded → tiết kiệm thời gian, được ưu tiên hơn.

```python
def calculate_download_score(machine_id: str, model_id: str, cache_dir: str) -> float:
    """
    Tính download score cho 1 machine.
    Score cao = machine đã có model, không cần download.
    """
    cache_path = Path(cache_dir) / model_id / machine_id
    if cache_path.exists():
        # Đã download đầy đủ
        return 1.0
    elif _is_downloading(cache_path):
        # Đang download một phần
        return 0.5
    else:
        # Chưa download
        return 0.0

def _is_downloading(cache_path: Path) -> bool:
    """Kiểm tra xem có file .downloading không (Exo dùng marker file)"""
    return (cache_path / ".downloading").exists()

# Áp dụng trong ILP:
# Thêm vào objective: download_score[m] * download_weight
# Machine đã download → score cao → được ưu tiên gán layers
```

**Kết hợp Leaf + Download trong ILP**:

```python
# Trong _solve_ilp(), thêm vào objective:
leaf_weight = 0.1  # Trọng số cho leaf preference
download_weight = 0.2  # Trọng số cho download score

for m, machine_id in enumerate(machines):
    leaf_score = calculate_leaf_score(machine_id, topology)
    download_score = calculate_download_score(machine_id, model_id, cache_dir)
    
    # Thêm vào objective (minimize nên dùng -score)
    prob += LpSum([
        -leaf_score[m] * leaf_weight * x[i, m]
        for i in range(n_layers)
    ])
    prob += LpSum([
        download_score * download_weight * x[i, m]
        for i in range(n_layers)
    ])
```

**Input cho ILP**:
```python
{
    "layers": [
        {"id": 0, "memory_mb": 4000, "flops": 1e12},
        {"id": 1, "memory_mb": 4200, "flops": 1.1e12},
        # ... 32 layers
    ],
    "machines": [
        {"id": "A", "memory_mb": 24000, "cluster": 1},
        {"id": "B", "memory_mb": 6000, "cluster": 1},
        {"id": "C", "memory_mb": 16000, "cluster": 2},
    ],
    "latency_matrix": {
        "A": {"B": 15, "C": 200},
        "B": {"A": 15, "C": 180},
        "C": {"A": 200, "B": 180}
    },
    "cluster_penalty": 1000  # Penalty nếu cross-cluster
}
```

**Output từ ILP**:
```python
partitions = [
    Partition(node_id="A", start=0.0, end=0.5),    # layers 0-15
    Partition(node_id="B", start=0.5, end=0.625),  # layers 16-19
    Partition(node_id="C", start=0.625, end=1.0),  # layers 20-31
]
```

**ILP Solver**:
```python
# Dùng PuLP (MIT license)
from pulp import LpProblem, LpMinimize, LpVariable, LpBinary

# Timeout: 60 giây cho 100 máy
# Nếu timeout → dùng greedy heuristic fallback
```

---

## 3. Tận dụng tài nguyên

### 3.1 Tài nguyên hiện tại dùng vs chưa dùng

| Tài nguyên | Hiện tại | LACP |
|------------|----------|------|
| GPU | ✅ Dùng | ✅ Dùng |
| VRAM | ✅ Dùng | ✅ Dùng |
| CPU | ⚠️ Ít | ✅ Tối đa |
| CPU RAM | ❌ Không | ✅ Dùng |
| Disk/SSD | ❌ Không | ✅ Dùng |
| NVMe | ❌ Không | ✅ Dùng |
| Multi-GPU | ❌ Không | ✅ Dùng |

### 3.2 CPU RAM cho KV Cache Overflow

**Vấn đề**: GPU VRAM ít, không lưu được KV cache cho prompt dài

**Giải pháp**: Lưu KV cache vào CPU RAM khi GPU đầy

```python
class CPU RAMKVCache:
    def __init__(self, gpu_vram_mb: int, cpu_ram_mb: int):
        self.gpu_cache = GPUCache(gpu_vram_mb)
        self.cpu_cache = CPURAMCache(cpu_ram_mb)
        self.threshold = 0.9  # Khi 90% GPU full
    
    def set(self, key: str, kv: Tensor):
        if self.gpu_cache.usage() > self.threshold:
            # Chuyển phần cũ nhất sang CPU RAM
            oldest = self.gpu_cache.evict_oldest()
            self.cpu_cache.set(key, oldest)
        else:
            self.gpu_cache.set(key, kv)
    
    def get(self, key: str) -> Optional[Tensor]:
        # Ưu tiên GPU, fallback CPU RAM
        if self.gpu_cache.has(key):
            return self.gpu_cache.get(key)
        if self.cpu_cache.has(key):
            # Đọc từ CPU RAM (chậm hơn GPU)
            return self.cpu_cache.get(key)
        return None
```

**Khi nào dùng**:
- Prompt > 4096 tokens
- GPU VRAM < 16GB
- Batch size lớn

### 3.3 Disk/SSD cho Embeddings Cache

**Vấn đề**: Tính lại embeddings cho prompt hay dùng

**Giải pháp**: Cache embeddings vào SSD

```python
class DiskEmbeddingsCache:
    def __init__(self, cache_dir: str, max_size_gb: int = 50):
        self.cache_dir = Path(cache_dir)
        self.max_size = max_size_gb * 1024 * 1024 * 1024
        self.lru = LRUCache()
    
    def get_cache_key(self, prompt: str) -> str:
        # Hash prompt làm key
        return hashlib.sha256(prompt.encode()).hexdigest()
    
    def get(self, prompt: str) -> Optional[np.ndarray]:
        key = self.get_cache_key(prompt)
        cache_file = self.cache_dir / f"{key}.npy"
        
        if cache_file.exists():
            self.lru.access(key)  # Update LRU
            return np.load(cache_file)
        return None
    
    def set(self, prompt: str, embeddings: np.ndarray):
        key = self.get_cache_key(prompt)
        cache_file = self.cache_dir / f"{key}.npy"
        
        # Kiểm tra size limit
        if self.get_total_size() + embeddings.nbytes > self.max_size:
            self.evict_lru_entries()
        
        np.save(cache_file, embeddings)
        self.lru.add(key)
```

**Khi nào dùng**:
- Prompt hay dùng (frequency > threshold)
- Inference batch nhiều prompts giống nhau
- SSD có không gian trống

### 3.4 Multi-GPU và Tensor Parallel

**Vấn đề**: 1 GPU không đủ cho model lớn

**Giải pháp 1: Pipeline Parallel (chia layers)** - Mỗi GPU chạy 1 phần layers

```python
class MultiGPUManager:
    def __init__(self, gpu_ids: List[int]):
        self.gpus = [torch.device(f"cuda:{i}") for i in gpu_ids]
        self.num_gpus = len(gpu_ids)

    def get_device_for_layer(self, layer_id: int, total_layers: int) -> torch.device:
        # Chia layers đều cho các GPU
        layers_per_gpu = total_layers // self.num_gpus
        gpu_index = min(layer_id // layers_per_gpu, self.num_gpus - 1)
        return self.gpus[gpu_index]

    def forward_layers(self, layers: List[nn.Module], input_data: Tensor) -> Tensor:
        current = input_data
        for i, layer in enumerate(layers):
            device = self.get_device_for_layer(i, len(layers))
            current = current.to(device)
            current = layer(current)
        return current
```

**Giải pháp 2: Tensor Parallel (từ Exo)** - Mỗi GPU chạy TẤT CẢ layers nhưng chia tensor computation

```python
class TensorParallelManager:
    """
    Tensor Parallel: Mỗi machine chạy ALL layers nhưng chia tensor theo dimension.
    Ví dụ: Layer có weight [hidden, hidden] → chia thành N phần, mỗi GPU tính 1 phần.
    
    Ưu điểm:
    - Giảm memory mỗi GPU (thay vì N layers, mỗi GPU chỉ giữ 1/N weights)
    - Phù hợp khi model fit được trong 1 GPU nhưng cần batch lớn
    """
    
    def __init__(self, gpu_ids: List[int], tensor_dim: int = 0):
        self.gpus = [torch.device(f"cuda:{i}") for i in gpu_ids]
        self.num_gpus = len(gpu_ids)
        self.tensor_dim = tensor_dim  # Dimension để split (0=row, 1=col)
    
    def split_tensor(self, tensor: Tensor) -> List[Tensor]:
        """Chia tensor thành N phần theo tensor_dim"""
        return torch.split(tensor, tensor.shape[self.tensor_dim] // self.num_gpus, dim=self.tensor_dim)
    
    def allgather_tensor(self, partial_tensors: List[Tensor]) -> Tensor:
        """Gom lại từ N phần"""
        return torch.cat(partial_tensors, dim=self.tensor_dim)
    
    def forward_with_tensor_parallel(self, layer: nn.Module, input_data: Tensor) -> Tensor:
        """
        Forward với tensor parallel:
        1. Split input theo tensor_dim
        2. Mỗi GPU compute 1 phần
        3. All-gather kết quả
        """
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
```

**So sánh Pipeline vs Tensor Parallel**:

| Khía cạnh | Pipeline Parallel | Tensor Parallel |
|-----------|------------------|-----------------|
| **Chia gì** | Layers | Weights/Tensors |
| **Memory mỗi GPU** | `total_layers / num_gpus * layer_size` | `total_weights / num_gpus` |
| **Communication** | Giữa layers (activation) | Trong layer (all-reduce) |
| **Phù hợp** | Model > 1 GPU | Model fit 1 GPU, cần batch lớn |

**Yêu cầu**:
- Máy có 2+ GPU
- Dùng NVLink để truyền data nhanh (không thì PCIe)
- Tensor parallel yêu cầu GPU có high bandwidth (NVLink > PCIe)

### 3.5 Preload Strategy

**Vấn đề**: Load model vào GPU tốn thời gian lúc start

**Giải pháp**: Preload vào RAM trước, chỉ chuyển sang GPU khi cần

```python
class PreloadStrategy:
    def __init__(self, model: nn.Module, device: str = "cuda"):
        self.model = model
        self.device = device
        self.cpu_model = model.cpu()  # Load vào RAM
        self.loaded_layers = {}  # layer_id -> GPU location
    
    def ensure_layer(self, layer_id: int) -> nn.Module:
        if layer_id not in self.loaded_layers:
            # Chuyển từ RAM -> GPU
            layer = self.cpu_model.layers[layer_id]
            self.loaded_layers[layer_id] = layer.to(self.device)
        return self.loaded_layers[layer_id]
    
    def preload_all(self):
        # Load tất cả vào GPU (nếu đủ VRAM)
        for layer_id in range(self.model.num_layers):
            self.ensure_layer(layer_id)
```

### 3.6 CFG Parallel cho Image Generation (từ Exo)

**Vấn đề**: Image generation models (Stable Diffusion, FLUX, etc.) dùng Classifier-Free Guidance (CFG). CFG cần chạy model 2 lần: 1 lần có guidance, 1 lần không có → chậm gấp đôi.

**Giải pháp**: Chạy 2 lần song song trên 2 machines khác nhau.

```python
class CFGParallelPlanner:
    """
    CFG Parallel: Chia inference thành 2 phần chạy song song.
    
    - Machine A: Chạy model với guidance (CFG branch)
    - Machine B: Chạy model không có guidance (non-CFG branch)
    - Master: Kết hợp 2 kết quả theo công thức CFG
    
    Công thức CFG: output = non_cfg_output + scale * (cfg_output - non_cfg_output)
    """
    
    def __init__(self, cfg_scale: float = 7.5):
        self.cfg_scale = cfg_scale
    
    def get_shard_assignments_for_cfg_parallel(
        self,
        model_card: ModelCard,
        machines: List[Machine],
        total_layers: int
    ) -> Tuple[List[Shard], List[Shard]]:
        """
        Chia shards cho CFG parallel.
        
        Returns:
        - cfg_shards: Shards cho CFG branch (machine A)
        - non_cfg_shards: Shards cho non-CFG branch (machine B)
        """
        # Mỗi branch cần chạy toàn bộ model
        # Nên chia layers đều cho 2 machines
        
        half_layers = total_layers // 2
        
        # Machine A: layers 0 -> half
        cfg_shards = [
            Shard(
                model_id=model_card.id,
                start=i / total_layers,
                end=(i + 1) / total_layers
            )
            for i in range(0, half_layers)
        ]
        
        # Machine B: layers half -> end
        non_cfg_shards = [
            Shard(
                model_id=model_card.id,
                start=i / total_layers,
                end=(i + 1) / total_layers
            )
            for i in range(half_layers, total_layers)
        ]
        
        return cfg_shards, non_cfg_shards
    
    def combine_cfg_outputs(
        self,
        cfg_output: Tensor,
        non_cfg_output: Tensor
    ) -> Tensor:
        """
        Kết hợp 2 outputs theo CFG formula.
        
        output = non_cfg + scale * (cfg - non_cfg)
        """
        return non_cfg_output + self.cfg_scale * (cfg_output - non_cfg_output)
```

**Data Flow cho CFG Parallel**:

```
┌──────────┐         ┌──────────────┐         ┌──────────────┐
│  Client  │ prompt  │ Machine A    │         │ Machine B    │
│          │────────>│ (CFG branch) │         │(non-CFG)     │
└──────────┘         └──────┬───────┘         └──────┬───────┘
                            │                         │
                            │ full forward            │ full forward
                            │ (with guidance)         │ (no guidance)
                            │                         │
                            └──────────┬──────────────┘
                                       │
                               ┌───────▼───────┐
                               │ Master        │
                               │ combine:      │
                               │ non_cfg +     │
                               │ scale*(cfg-   │
                               │ non_cfg)      │
                               └───────┬───────┘
                                       │
                               ┌───────▼───────┐
                               │ final image   │
                               └───────────────┘
```

**Khi nào dùng CFG Parallel**:
- Image generation models (Stable Diffusion, FLUX, Qwen-Image, etc.)
- CFG scale > 1.0 (thường 7-12 cho SDXL, 1-4 cho FLUX)
- Cần 2 machines với đủ memory để chạy full model

**Lưu ý**:
- CFG Parallel KHÔNG giảm memory mỗi machine
- Mỗi machine vẫn cần chạy full model
- Chỉ giảm thời gian (chạy song song 2 lần)
- Cần đảm bảo 2 machines có latency thấp để đồng bộ

---

## 4. Kiến trúc chi tiết

### 4.1 Data Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                         LACP Architecture                            │
└─────────────────────────────────────────────────────────────────────┘

[1. Latency Probing]
┌──────────┐     ping      ┌──────────┐     ping      ┌──────────┐
│ Machine A│──────────────│ Machine B│──────────────│ Machine C│
└──────────┘              └──────────┘              └──────────┘
       │                        │                        │
       └────────────────────────┼────────────────────────┘
                                │
                    ┌───────────▼───────────┐
                    │   Latency Matrix      │
                    │   (cached to disk)    │
                    └───────────────────────┘

[2. Hierarchical Clustering]
                    ┌───────────────────────┐
                    │   Latency Matrix      │
                    └───────────┬───────────┘
                                │
                    ┌───────────▼───────────┐
                    │  Agglomerative         │
                    │  Clustering            │
                    └───────────┬───────────┘
                                │
              ┌─────────────────┼─────────────────┐
              │                 │                 │
     ┌────────▼────────┐ ┌──────▼──────┐ ┌───────▼───────┐
     │ Cluster 1 (VN)  │ │ Cluster 2   │ │ Cluster 3     │
     │ A, B            │ │ (US) C, D   │ │ (EU) E        │
     └─────────────────┘ └─────────────┘ └───────────────┘

[3. ILP Partitioning]
     ┌─────────────────────────────────────────────────────┐
     │ Input:                                             │
     │   - Latency Matrix                                 │
     │   - Memory per machine                             │
     │   - Memory per layer (profile)                     │
     │   - Cluster assignments                            │
     └─────────────────────────────────────────────────────┘
                                │
                    ┌───────────▼───────────┐
                    │   ILP Solver          │
                    │   (PuLP/OR-Tools)     │
                    └───────────┬───────────┘
                                │
     ┌──────────────────────────┼──────────────────────────┐
     │                          │                          │
┌────▼────┐              ┌──────▼──────┐              ┌─────▼─────┐
│Machine A│              │ Machine B   │              │Machine C  │
│L: 0-15  │              │ L: 16-20    │              │L: 21-31   │
└─────────┘              └─────────────┘              └───────────┘

[4. Resource Utilization]
┌─────────────────────────────────────────────────────────────┐
│ Machine A (example)                                         │
│ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────────────────┐│
│ │ GPU 0   │ │ GPU 1   │ │ CPU RAM │ │ SSD                 ││
│ │ L: 0-7  │ │ L: 8-15 │ │ KV Cache│ │ Embeddings Cache    ││
│ └─────────┘ └─────────┘ └─────────┘ └─────────────────────┘│
└─────────────────────────────────────────────────────────────┘
```

### 4.2 Class Diagram

```python
# LACP Main Classes

class LACPPartitioningStrategy(PartitioningStrategy):
    """Main entry point - thay thế RingMemoryWeightedPartitioningStrategy"""
    
    def __init__(self):
        self.latency_prober = LatencyProber()
        self.clusterer = HierarchicalClusterer()
        self.ilp_solver = ILPPartitioner()
        self.resource_managers = ResourceManagerRegistry()
    
    def partition(self, topology: Topology) -> List[Partition]:
        # 1. Get/update latency matrix
        latency_matrix = self.latency_prober.get_matrix()
        
        # 2. Cluster machines
        clusters = self.clusterer.cluster(latency_matrix, topology)
        
        # 3. ILP partitioning
        partitions = self.ilp_solver.find_optimal(
            topology=topology,
            latency_matrix=latency_matrix,
            clusters=clusters,
            layer_profiles=self.get_layer_profiles()
        )
        
        return partitions


class LatencyProber:
    """Đo latency giữa các máy"""
    
    def __init__(self, cache_file: str = "synapse/config/latency_cache.json"):
        self.cache_file = cache_file
        self.matrix = self._load_cache()
    
    def probe_all(self, peers: List[PeerHandle]) -> Dict[str, Dict[str, float]]:
        """Ping tất cả peers, trả về latency matrix"""
        # Gửi ping song song, đo RTT
        # Trả về matrix
    
    def probe_single(self, peer: PeerHandle) -> Dict[str, float]:
        """Ping 1 peer"""
    
    def get_matrix(self) -> Dict[str, Dict[str, float]]:
        """Lấy latency matrix (từ cache hoặc probe mới)"""


class HierarchicalClusterer:
    """Nhóm máy theo latency"""
    
    def cluster(
        self, 
        latency_matrix: Dict[str, Dict[str, float]],
        threshold: float = 50.0  # ms
    ) -> List[MachineCluster]:
        """Agglomerative clustering"""
    
    def get_cluster_for_machine(self, machine_id: str) -> MachineCluster:


class ILPPartitioner:
    """Tìm partition tối ưu dùng ILP"""
    
    def __init__(self, timeout: int = 60):
        self.timeout = timeout
    
    def find_optimal(
        self,
        topology: Topology,
        latency_matrix: Dict[str, Dict[str, float]],
        clusters: List[MachineCluster],
        layer_profiles: List[LayerProfile]
    ) -> List[Partition]:
        """Giải ILP, return partitions"""
    
    def _build_ilp_problem(self, ...) -> LpProblem:
        """Xây dựng bài toán ILP"""
    
    def _greedy_fallback(self, ...) -> List[Partition]:
        """Fallback greedy nếu ILP timeout"""


class ResourceManagerRegistry:
    """Registry cho các resource managers"""
    
    def __init__(self):
        self.managers = {
            "cpu_ram_kv": CPU RAMKVCacheManager(),
            "disk_cache": DiskEmbeddingsCacheManager(),
            "multi_gpu": MultiGPUManager(),
            "preload": PreloadManager(),
        }
    
    def get_manager(self, name: str) -> ResourceManager:


class CPU RAMKVCacheManager(ResourceManager):
    """Quản lý KV cache trên CPU RAM"""
    
    def __init__(self, max_cpu_ram_gb: int = 64):
        self.cache = CPU RAMCache(max_cpu_ram_gb * 1024)  # MB
    
    def set(self, key: str, kv: Tensor):
        """Lưu KV cache"""
    
    def get(self, key: str) -> Optional[Tensor]:


class DiskEmbeddingsCacheManager(ResourceManager):
    """Quản lý embeddings cache trên disk"""
    
    def __init__(self, cache_dir: str, max_size_gb: int = 50):
        self.cache = DiskEmbeddingsCache(cache_dir, max_size_gb)
    
    def get(self, prompt: str) -> Optional[np.ndarray]:
    
    def set(self, prompt: str, embeddings: np.ndarray):


class MultiGPUManager(ResourceManager):
    """Quản lý multi-GPU trong 1 máy"""
    
    def __init__(self, gpu_ids: List[int]):
        self.gpus = [torch.device(f"cuda:{i}") for i in gpu_ids]
    
    def get_device_for_layer(self, layer_id: int) -> torch.device:
    
    def sync_between_gpus(self):


class PreloadManager(ResourceManager):
    """Quản lý preload model vào RAM"""
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.loaded_layers = {}
    
    def ensure_layer(self, layer_id: int) -> nn.Module:
    
    def preload_all(self):
```

### 4.3 Sequence Diagram cho Inference

```
┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐
│  Client  │     │ Machine A│     │ Machine B│     │ Machine C│
│          │     │ (L: 0-15)│     │ (L:16-20)│     │ (L:21-31)│
└────┬─────┘     └────┬─────┘     └────┬─────┘     └────┬─────┘
     │                │                │                │
     │ prompt         │                │                │
     │───────────────>│                │                │
     │                │                │                │
     │                │ encode(prompt) │                │
     │                │──> GPU         │                │
     │                │<── tokens      │                │
     │                │                │                │
     │                │ infer L0-L15   │                │
     │                │──> GPU         │                │
     │                │                │                │
     │                │ send hidden    │                │
     │                │───────────────>│                │
     │                │ (latency: 15ms)│                │
     │                │                │                │
     │                │                │ infer L16-L20  │
     │                │                │──> GPU         │
     │                │                │                │
     │                │                │ send hidden    │
     │                │                │───────────────>│
     │                │                │ (latency: 180ms)
     │                │                │                │
     │                │                │                │ infer L21-L31
     │                │                │                │──> GPU
     │                │                │                │
     │                │                │                │ sample token
     │                │                │                │
     │                │                │ broadcast result
     │<───────────────────────────────│─────────────────│
     │                │                │                │
```

---

## 5. Implementation

### 5.1 Files cần tạo mới

```
synapse/
├── topology/
│   ├── lacp_partitioning.py          # LACP main class
│   ├── latency_clustering.py         # Hierarchical clustering
│   └── ilp_partitioner.py            # ILP solver
├── networking/
│   └── latency_probing.py            # Latency measurement
└── resources/
    ├── __init__.py
    ├── cpu_ram_kv_cache.py           # CPU RAM KV cache
    ├── disk_embeddings_cache.py      # SSD cache
    ├── multi_gpu_manager.py          # Multi-GPU support
    └── preload_strategy.py           # RAM preloading
```

### 5.2 Files cần sửa đổi

| File | Thay đổi |
|------|----------|
| [`synapse/topology/ring_memory_weighted_partitioning_strategy.py`](synapse/topology/ring_memory_weighted_partitioning_strategy.py) | Thêm LACP hoặc thay thế |
| [`synapse/orchestration/node.py`](synapse/orchestration/node.py) | Tích hợp LACP + resource managers |
| [`synapse/inference/inference_engine.py`](synapse/inference/inference_engine.py) | Thêm CPU RAM, disk cache methods |
| [`synapse/networking/tailscale/tailscale_discovery.py`](synapse/networking/tailscale/tailscale_discovery.py) | Gọi latency probing |

### 5.3 Dependencies mới

```txt
# requirements.txt - thêm

# ILP Solver
pulp>=2.7.0

# Clustering
scipy>=1.10.0
scikit-learn>=1.3.0

# Caching
diskcache>=5.6.0
```

### 5.4 Code skeleton cho LACP

```python
# synapse/topology/lacp_partitioning.py

from typing import List, Dict, Optional
from synapse.topology.partitioning_strategy import PartitioningStrategy, Partition
from synapse.topology.topology import Topology
from synapse.inference.shard import Shard

class LACPPartitioningStrategy(PartitioningStrategy):
    """
    Latency-Aware Collaborative Partitioning
    
    1. Probe latency giữa các máy
    2. Hierarchical clustering
    3. ILP-based partitioning tối ưu
    """
    
    def __init__(
        self,
        latency_cache_file: str = "synapse/config/latency_cache.json",
        cluster_threshold_ms: float = 50.0,
        ilp_timeout: int = 60,
        enable_resource_managers: bool = True
    ):
        self.latency_cache_file = latency_cache_file
        self.cluster_threshold_ms = cluster_threshold_ms
        self.ilp_timeout = ilp_timeout
        
        # Lazy initialization
        self._latency_prober = None
        self._clusterer = None
        self._ilp_solver = None
        self._resource_managers = None
        
        if enable_resource_managers:
            self._init_resource_managers()
    
    def partition(self, topology: Topology) -> List[Partition]:
        """
        Main entry point - chia model dựa trên LACP
        """
        # 1. Get latency matrix
        latency_matrix = self._get_latency_prober().get_matrix(topology)
        
        # 2. Cluster machines
        clusters = self._get_clusterer().cluster(
            latency_matrix, 
            threshold=self.cluster_threshold_ms
        )
        
        # 3. Get layer profiles (memory, compute per layer)
        layer_profiles = self._get_layer_profiles(topology)
        
        # 4. ILP partitioning
        partitions = self._get_ilp_solver().find_optimal(
            topology=topology,
            latency_matrix=latency_matrix,
            clusters=clusters,
            layer_profiles=layer_profiles,
            timeout=self.ilp_timeout
        )
        
        return partitions
    
    def _get_latency_prober(self):
        if self._latency_prober is None:
            from synapse.networking.latency_probing import LatencyProber
            self._latency_prober = LatencyProber(self.latency_cache_file)
        return self._latency_prober
    
    def _get_clusterer(self):
        if self._clusterer is None:
            from synapse.topology.latency_clustering import HierarchicalClusterer
            self._clusterer = HierarchicalClusterer()
        return self._clusterer
    
    def _get_ilp_solver(self):
        if self._ilp_solver is None:
            from synapse.topology.ilp_partitioner import ILPPartitioner
            self._ilp_solver = ILPPartitioner(timeout=self.ilp_timeout)
        return self._ilp_solver
    
    def _get_layer_profiles(self, topology: Topology) -> List[LayerProfile]:
        # Profile mỗi layer - memory, compute
        # Có thể profile trước hoặc ước lượng
        pass
    
    def _init_resource_managers(self):
        from synapse.resources import ResourceManagerRegistry
        self._resource_managers = ResourceManagerRegistry()
```

```python
# synapse/networking/latency_probing.py

import asyncio
import json
import time
from typing import Dict, List
from pathlib import Path

class LatencyProber:
    """
    Đo latency giữa các máy qua Tailscale
    """
    
    def __init__(self, cache_file: str = "synapse/config/latency_cache.json"):
        self.cache_file = Path(cache_file)
        self.matrix = self._load_cache()
        self.probe_count = 10  # Số ping mỗi lần đo
        self.timeout_ms = 5000
    
    def get_matrix(self, topology: Topology) -> Dict[str, Dict[str, float]]:
        """
        Lấy latency matrix, probe lại nếu cần
        """
        peers = list(topology.peers)
        machine_ids = [p.id() for p in peers]
        
        # Kiểm tra xem cần probe lại không
        if self._needs_update():
            return self.probe_all(peers)
        
        return self.matrix
    
    def probe_all(self, peers: List[PeerHandle]) -> Dict[str, Dict[str, float]]:
        """
        Ping tất cả peers song song
        """
        matrix = {}
        
        async def probe_pair(peer_a, peer_b):
            # Ping peer_a -> peer_b
            latencies = []
            for _ in range(self.probe_count):
                start = time.perf_counter()
                await peer_a.ping(peer_b)  # Gửi ping qua Tailscale
                end = time.perf_counter()
                latencies.append((end - start) * 1000)  # ms
            
            # Loại bỏ outliers (max, min)
            latencies.sort()
            latencies = latencies[1:-1]
            return sum(latencies) / len(latencies)
        
        # Probe tất cả cặp
        tasks = []
        for i, peer_a in enumerate(peers):
            for peer_b in peers[i+1:]:
                tasks.append((peer_a.id(), peer_b.id(), probe_pair(peer_a, peer_b)))
        
        # Chạy song song
        results = await asyncio.gather(*[t[2] for t in tasks], return_exceptions=True)
        
        # Xây dựng matrix
        for (id_a, id_b, _), latency in zip(tasks, results):
            if id_a not in matrix:
                matrix[id_a] = {}
            if id_b not in matrix:
                matrix[id_b] = {}
            matrix[id_a][id_b] = latency
            matrix[id_b][id_a] = latency
        
        # Cache lại
        self._save_cache(matrix)
        self.matrix = matrix
        
        return matrix
    
    def _needs_update(self) -> bool:
        """Kiểm tra xem cache có còn valid không"""
        if not self.cache_file.exists():
            return True
        
        cache = self._load_cache()
        if not cache:
            return True
        
        # Cache quá 5 phút → cần update
        age = time.time() - cache.get("timestamp", 0)
        return age > 300  # 5 phút
    
    def _load_cache(self) -> Dict:
        if self.cache_file.exists():
            with open(self.cache_file) as f:
                return json.load(f)
        return {}
    
    def _save_cache(self, matrix: Dict):
        cache = {
            "version": "1.0",
            "timestamp": time.time(),
            "matrix": matrix
        }
        with open(self.cache_file, 'w') as f:
            json.dump(cache, f)
```

```python
# synapse/topology/latency_clustering.py

import numpy as np
from typing import List, Dict, Tuple
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

@dataclass
class MachineCluster:
    id: int
    machine_ids: List[str]
    avg_latency: float

class HierarchicalClusterer:
    """
    Hierarchical clustering dựa trên latency matrix
    """
    
    def cluster(
        self,
        latency_matrix: Dict[str, Dict[str, float]],
        threshold_ms: float = 50.0
    ) -> List[MachineCluster]:
        """
        Agglomerative clustering
        
        Args:
            latency_matrix: Dict[machine_id][machine_id] = latency_ms
            threshold_ms: Ngưỡng latency để merge clusters
        
        Returns:
            List[MachineCluster]
        """
        # Chuyển sang distance matrix
        machine_ids = list(latency_matrix.keys())
        n = len(machine_ids)
        
        # Distance matrix (upper triangular)
        distances = []
        for i in range(n):
            for j in range(i+1, n):
                distances.append(latency_matrix[machine_ids[i]][machine_ids[j]])
        
        # Convert to condensed distance matrix
        dist_matrix = np.array(distances)
        
        # Hierarchical clustering
        Z = linkage(dist_matrix, method='average')
        
        # Cut dendrogram at threshold
        # threshold_ms = 50 → cắt ở độ cao 50
        clusters = fcluster(Z, t=threshold_ms, criterion='distance')
        
        # Build MachineCluster objects
        cluster_dict = {}
        for i, cluster_id in enumerate(clusters):
            if cluster_id not in cluster_dict:
                cluster_dict[cluster_id] = []
            cluster_dict[cluster_id].append(machine_ids[i])
        
        result = []
        for cluster_id, machine_list in cluster_dict.items():
            # Tính average latency trong cluster
            avg_lat = self._calc_avg_cluster_latency(
                machine_list, latency_matrix
            )
            result.append(MachineCluster(
                id=cluster_id,
                machine_ids=machine_list,
                avg_latency=avg_lat
            ))
        
        return result
    
    def _calc_avg_cluster_latency(
        self,
        machine_ids: List[str],
        latency_matrix: Dict[str, Dict[str, float]]
    ) -> float:
        """Tính average latency trong cluster"""
        total = 0
        count = 0
        for i, m1 in enumerate(machine_ids):
            for m2 in machine_ids[i+1:]:
                total += latency_matrix[m1][m2]
                count += 1
        return total / count if count > 0 else 0
```

```python
# synapse/topology/ilp_partitioner.py

from pulp import LpProblem, LpMinimize, LpVariable, LpBinary, LpSum
from typing import List, Dict, Optional
import numpy as np

@dataclass
class LayerProfile:
    layer_id: int
    memory_mb: float
    compute_flops: float

@dataclass 
class Partition:
    node_id: str
    start: float  # 0.0 - 1.0
    end: float    # 0.0 - 1.0

class ILPPartitioner:
    """
    ILP-based partitioner cho LACP
    """
    
    def __init__(self, timeout: int = 60):
        self.timeout = timeout
    
    def find_optimal(
        self,
        topology: Topology,
        latency_matrix: Dict[str, Dict[str, float]],
        clusters: List[MachineCluster],
        layer_profiles: List[LayerProfile],
        timeout: Optional[int] = None
    ) -> List[Partition]:
        """
        Tìm partition tối ưu dùng ILP
        """
        try:
            return self._solve_ilp(
                topology, latency_matrix, clusters, layer_profiles,
                timeout or self.timeout
            )
        except Exception as e:
            # Fallback to greedy
            return self._greedy_fallback(
                topology, latency_matrix, layer_profiles
            )
    
    def _solve_ilp(
        self,
        topology: Topology,
        latency_matrix: Dict[str, Dict[str, float]],
        clusters: List[MachineCluster],
        layer_profiles: List[LayerProfile],
        timeout: int
    ) -> List[Partition]:
        """
        Giải ILP
        """
        machines = list(topology.nodes.keys())
        n_layers = len(layer_profiles)
        n_machines = len(machines)
        
        # Create LP problem
        prob = LpProblem("LACP_Partitioning", LpMinimize)
        
        # Variables: x[i][m] = 1 if layer i assigned to machine m
        x = {}
        for i in range(n_layers):
            for m in range(n_machines):
                x[i, m] = LpVariable(
                    f"x_{i}_{m}",
                    cat=LpBinary
                )
        
        # Objective: Minimize weighted latency cost
        # Cost = sum over consecutive layers i,i+1 of:
        #   latency(m1,m2) * y[i,i+1]
        # where y = 1 if layers assigned to different machines
        latency_cost = []
        for i in range(n_layers - 1):
            for m1 in range(n_machines):
                for m2 in range(n_machines):
                    if m1 != m2:
                        machine_id_1 = machines[m1]
                        machine_id_2 = machines[m2]
                        lat = latency_matrix.get(machine_id_1, {}).get(
                            machine_id_2, 200  # default 200ms
                        )
                        # y[i,i+1] >= x[i][m1] + x[i+1][m2] - 1
                        y = LpVariable(f"y_{i}_{m1}_{m2}", cat=LpBinary)
                        prob += y >= x[i, m1] + x[i+1, m2] - 1
                        latency_cost.append(lat * y)
        
        prob += LpSum(latency_cost)
        
        # Constraint 1: Each layer assigned to exactly one machine
        for i in range(n_layers):
            prob += LpSum(x[i, m] for m in range(n_machines)) == 1
        
        # Constraint 2: Memory constraint per machine
        machine_memory = {
            m: topology.nodes[m].memory 
            for m in machines
        }
        for m in range(n_machines):
            mem_constraint = []
            for i in range(n_layers):
                mem_constraint.append(
                    layer_profiles[i].memory_mb * x[i, m]
                )
            prob += LpSum(mem_constraint) <= machine_memory[machines[m]]
        
        # Solve
        prob.solve(PULP_CBC_CMD(timeLimit=timeout))
        
        # Extract solution
        partitions = self._extract_partitions(
            x, machines, layer_profiles, n_layers
        )
        
        return partitions
    
    def _extract_partitions(
        self,
        x: Dict,
        machines: List[str],
        layer_profiles: List[LayerProfile],
        n_layers: int
    ) -> List[Partition]:
        """Extract partition boundaries từ ILP solution"""
        # Determine which machine each layer belongs to
        layer_to_machine = {}
        for i in range(n_layers):
            for m, machine_id in enumerate(machines):
                if x[i, m].value() > 0.5:
                    layer_to_machine[i] = machine_id
                    break
        
        # Group consecutive layers by machine
        partitions = []
        current_machine = None
        start_layer = 0
        
        for i in range(n_layers):
            machine = layer_to_machine[i]
            if machine != current_machine:
                if current_machine is not None:
                    # Save previous partition
                    partitions.append(Partition(
                        node_id=current_machine,
                        start=start_layer / n_layers,
                        end=i / n_layers
                    ))
                current_machine = machine
                start_layer = i
        
        # Last partition
        partitions.append(Partition(
            node_id=current_machine,
            start=start_layer / n_layers,
            end=1.0
        ))
        
        return partitions
    
    def _greedy_fallback(
        self,
        topology: Topology,
        latency_matrix: Dict[str, Dict[str, float]],
        layer_profiles: List[LayerProfile]
    ) -> List[Partition]:
        """
        Fallback greedy algorithm nếu ILP timeout
        """
        machines = list(topology.nodes.keys())
        n_layers = len(layer_profiles)
        
        # Sort machines by memory descending
        machines.sort(key=lambda m: topology.nodes[m].memory, reverse=True)
        
        # Simple greedy: assign layers to machines based on memory
        partitions = []
        current_start = 0
        current_machine_idx = 0
        current_memory = 0
        
        for i in range(n_layers):
            layer_mem = layer_profiles[i].memory_mb
            machine_mem = topology.nodes[machines[current_machine_idx]].memory
            
            if current_memory + layer_mem > machine_mem:
                # Save partition
                partitions.append(Partition(
                    node_id=machines[current_machine_idx],
                    start=current_start / n_layers,
                    end=i / n_layers
                ))
                current_start = i
                current_machine_idx += 1
                current_memory = 0
            
            current_memory += layer_mem
        
        # Last partition
        partitions.append(Partition(
            node_id=machines[current_machine_idx],
            start=current_start / n_layers,
            end=1.0
        ))
        
        return partitions
```

---

## 6. So sánh

### 6.1 Synapse cũ vs LACP

| Khía cạnh | Synapse cũ | LACP |
|-----------|------------|------|
| **Metric dùng** | Chỉ FLOPS fp16 (hoặc memory fallback) | FLOPS + Memory + Latency |
| **Đo latency** | ❌ Không | ✅ Có |
| **Clustering** | ❌ Không | ✅ Có |
| **ILP solver** | ❌ Không | ✅ Có |
| **CPU RAM** | ❌ Không | ✅ KV cache overflow |
| **Disk cache** | ❌ Không | ✅ Embeddings cache |
| **Multi-GPU** | ❌ Không | ✅ Tensor parallelism |
| **Tailscale-aware** | ❌ Không | ✅ Có |
| **Dynamic rebalance** | ❌ Không | ⚠️ Có thể thêm |

### 6.2 LACP vs các nghiên cứu khác

| Khía cạnh | PipeDream | ZeRO | Megatron | FlexFlow | LACP |
|-----------|-----------|------|----------|----------|------|
| **Đo latency thực** | ❌ | ❌ | ❌ | ❌ | ✅ |
| **Clustering** | ❌ | ❌ | ❌ | ❌ | ✅ |
| **ILP solver** | ❌ | ❌ | ❌ | ✅ | ✅ |
| **Tailscale-aware** | ❌ | ❌ | ❌ | ❌ | ✅ |
| **Full resources** | ❌ | ⚠️ | ⚠️ | ❌ | ✅ |
| **CPU RAM cache** | ❌ | ❌ | ❌ | ❌ | ✅ |
| **Disk cache** | ❌ | ❌ | ❌ | ❌ | ✅ |

### 6.3 Điểm novel của LACP

1. **Latency-aware partitioning cho Tailscale**
   - Chưa có nghiên cứu nào về distributed ML trên Tailscale VPN
   - Tailscale có relay nodes → latency không đơn giản

2. **Hierarchical ILP**
   - Kết hợp clustering + ILP để scale được với 100+ máy
   - ILP thuần túy không scale được

3. **Full resource utilization**
   - Không chỉ GPU mà còn CPU RAM, Disk, Multi-GPU
   - Tận dụng tất cả tài nguyên trong heterogeneous cluster

---

## 7. Benchmark metrics

| Metric | Cách đo |
|--------|---------|
| **Throughput** | Tokens/giây tổng |
| **Latency** | Thời gian end-to-end (prompt → token đầu tiên) |
| **Memory usage** | VRAM + CPU RAM + Disk cache |
| **Network traffic** | Bytes qua Tailscale |
| **GPU utilization** | % GPU sử dụng |
| **CPU utilization** | % CPU sử dụng |
| **Partition balance** | Variance của load giữa các máy |

---

## 8. Limitations

1. **ILP complexity**: NP-hard, timeout với 100+ máy
2. **Latency variability**: Tailscale route có thể thay đổi
3. **Layer profiling**: Cần profile từng layer cho chính xác
4. **Overhead**: Latency probing tốn vài giây lúc start

---

## 9. Future work

1. **Dynamic rebalancing**: Tự động re-partition khi workload thay đổi
2. **Model-specific profiling**: Profile cho từng model architecture
3. **RDMA support**: Truyền data qua RDMA nhanh hơn TCP
4. **Online ILP**: Cập nhật partition liên tục theo workload
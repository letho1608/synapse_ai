"""
Base Model Classes - Foundation cho hệ thống mô hình linh hoạt

File này định nghĩa các abstract base classes cho tất cả các mô hình AI
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import json


class ModelType(Enum):
    """Loại mô hình AI"""
    NLP = "nlp"
    VISION = "vision"
    MULTIMODAL = "multimodal"
    AUDIO = "audio"
    TIMESERIES = "timeseries"
    GRAPH = "graph"
    CUSTOM = "custom"


class TaskType(Enum):
    """Loại task mà mô hình thực hiện"""
    # NLP Tasks
    TEXT_GENERATION = "text_generation"
    TEXT_CLASSIFICATION = "text_classification"
    TOKEN_CLASSIFICATION = "token_classification"
    QUESTION_ANSWERING = "question_answering"
    SUMMARIZATION = "summarization"
    TRANSLATION = "translation"
    
    # Vision Tasks
    IMAGE_GENERATION = "image_generation"
    IMAGE_CLASSIFICATION = "image_classification"
    OBJECT_DETECTION = "object_detection"
    SEMANTIC_SEGMENTATION = "semantic_segmentation"
    IMAGE_TO_IMAGE = "image_to_image"
    
    # Multimodal Tasks
    IMAGE_TEXT_TO_TEXT = "image_text_to_text"
    TEXT_TO_IMAGE = "text_to_image"
    VISUAL_QUESTION_ANSWERING = "visual_question_answering"
    
    # Audio Tasks
    SPEECH_RECOGNITION = "speech_recognition"
    TEXT_TO_SPEECH = "text_to_speech"
    AUDIO_CLASSIFICATION = "audio_classification"
    
    # Other
    CUSTOM = "custom"


@dataclass
class ModelConfig:
    """
    Cấu hình chung cho tất cả các mô hình
    
    Attributes:
        model_name: Tên định danh của mô hình
        model_type: Loại mô hình (NLP, Vision, etc.)
        task_type: Task mà mô hình thực hiện
        hidden_size: Kích thước hidden dimension
        num_layers: Số lượng layers
        vocab_size: Kích thước vocabulary (cho NLP models)
        max_seq_length: Độ dài sequence tối đa
        dropout: Dropout rate
        supports_sharding: Có hỗ trợ distributed sharding không
        min_shard_layers: Số layers tối thiểu cho mỗi shard
        custom_params: Các tham số tùy chỉnh khác
    """
    model_name: str
    model_type: ModelType
    task_type: TaskType
    hidden_size: int
    num_layers: int
    
    # Optional params
    vocab_size: Optional[int] = None
    max_seq_length: int = 2048
    dropout: float = 0.1
    
    # Distributed params
    supports_sharding: bool = True
    min_shard_layers: int = 1
    
    # Custom params - dict để lưu các tham số đặc thù của từng mô hình
    custom_params: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        config_dict = asdict(self)
        config_dict['model_type'] = self.model_type.value
        config_dict['task_type'] = self.task_type.value
        return config_dict
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ModelConfig':
        """Create config from dictionary"""
        config_dict = config_dict.copy()
        config_dict['model_type'] = ModelType(config_dict['model_type'])
        config_dict['task_type'] = TaskType(config_dict['task_type'])
        return cls(**config_dict)
    
    def save(self, path: str):
        """Lưu config ra file JSON"""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load(cls, path: str) -> 'ModelConfig':
        """Load config từ file JSON"""
        with open(path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


class BaseModel(ABC):
    """
    Abstract base class cho tất cả các mô hình AI
    
    Mọi mô hình custom phải kế thừa class này và implement các methods bắt buộc
    """
    
    def __init__(self, config: ModelConfig):
        """
        Initialize model với config
        
        Args:
            config: ModelConfig object chứa cấu hình mô hình
        """
        self.config = config
        self.model_type = config.model_type
        self.task_type = config.task_type
        self._initialized = False
    
    @abstractmethod
    def forward(self, inputs: Any, **kwargs) -> Any:
        """
        Forward pass của mô hình
        
        Args:
            inputs: Input data (format tùy thuộc vào loại mô hình)
            **kwargs: Các tham số bổ sung
            
        Returns:
            Output của mô hình
        """
        pass
    
    @abstractmethod
    def get_num_params(self) -> int:
        """
        Tính tổng số parameters của mô hình
        
        Returns:
            Số lượng parameters
        """
        pass
    
    @abstractmethod
    def save_checkpoint(self, path: str) -> None:
        """
        Lưu checkpoint của mô hình
        
        Args:
            path: Đường dẫn để lưu checkpoint
        """
        pass
    
    @abstractmethod
    def load_checkpoint(self, path: str) -> None:
        """
        Load checkpoint vào mô hình
        
        Args:
            path: Đường dẫn đến checkpoint
        """
        pass
    
    def supports_distributed(self) -> bool:
        """
        Kiểm tra xem mô hình có hỗ trợ distributed inference không
        
        Returns:
            True nếu hỗ trợ, False nếu không
        """
        return self.config.supports_sharding
    
    def get_shard_config(self, shard_id: int, total_shards: int) -> Dict[str, Any]:
        """
        Tạo cấu hình cho một shard cụ thể trong distributed inference
        
        Args:
            shard_id: ID của shard (0-indexed)
            total_shards: Tổng số shards
            
        Returns:
            Dictionary chứa cấu hình shard
            
        Raises:
            NotImplementedError: Nếu mô hình không hỗ trợ sharding
        """
        if not self.supports_distributed():
            raise NotImplementedError(
                f"Model {self.config.model_name} doesn't support distributed sharding"
            )
        
        # Tính toán layer range cho shard này
        layers_per_shard = self.config.num_layers // total_shards
        start_layer = shard_id * layers_per_shard
        end_layer = start_layer + layers_per_shard - 1
        
        # Shard cuối cùng lấy tất cả layers còn lại
        if shard_id == total_shards - 1:
            end_layer = self.config.num_layers - 1
        
        return {
            "shard_id": shard_id,
            "total_shards": total_shards,
            "start_layer": start_layer,
            "end_layer": end_layer,
            "num_layers": end_layer - start_layer + 1,
            "is_first": shard_id == 0,
            "is_last": shard_id == total_shards - 1
        }
    
    def get_memory_footprint(self) -> Dict[str, int]:
        """
        Ước tính memory footprint của mô hình
        
        Returns:
            Dictionary với thông tin về memory usage
        """
        num_params = self.get_num_params()
        
        # Giả sử mỗi parameter là float32 (4 bytes)
        param_memory = num_params * 4
        
        # Ước tính gradient memory (bằng param memory khi training)
        gradient_memory = param_memory
        
        # Ước tính optimizer state memory (Adam: 2x param memory)
        optimizer_memory = param_memory * 2
        
        # Ước tính activation memory (phụ thuộc vào batch size và sequence length)
        # Đây là ước tính thô, thực tế phức tạp hơn
        activation_memory = self.config.hidden_size * self.config.max_seq_length * 4
        
        return {
            "parameters_bytes": param_memory,
            "parameters_mb": param_memory / (1024 * 1024),
            "parameters_gb": param_memory / (1024 * 1024 * 1024),
            "gradients_bytes": gradient_memory,
            "optimizer_bytes": optimizer_memory,
            "activations_bytes": activation_memory,
            "total_training_bytes": param_memory + gradient_memory + optimizer_memory + activation_memory,
            "total_training_gb": (param_memory + gradient_memory + optimizer_memory + activation_memory) / (1024 * 1024 * 1024),
            "total_inference_bytes": param_memory + activation_memory,
            "total_inference_gb": (param_memory + activation_memory) / (1024 * 1024 * 1024)
        }
    
    def summary(self) -> str:
        """
        In ra thông tin tóm tắt về mô hình
        
        Returns:
            String chứa thông tin mô hình
        """
        memory_info = self.get_memory_footprint()
        
        summary = f"""
╔══════════════════════════════════════════════════════════════╗
║                      MODEL SUMMARY                           ║
╠══════════════════════════════════════════════════════════════╣
║ Name:              {self.config.model_name:<40} ║
║ Type:              {self.config.model_type.value:<40} ║
║ Task:              {self.config.task_type.value:<40} ║
║ Hidden Size:       {self.config.hidden_size:<40} ║
║ Num Layers:        {self.config.num_layers:<40} ║
║ Vocab Size:        {str(self.config.vocab_size):<40} ║
║ Max Seq Length:    {self.config.max_seq_length:<40} ║
║ Dropout:           {self.config.dropout:<40} ║
║ Supports Sharding: {str(self.config.supports_sharding):<40} ║
╠══════════════════════════════════════════════════════════════╣
║ Total Parameters:  {self.get_num_params():>20,} params      ║
║ Memory (Inference):{memory_info['total_inference_gb']:>19.2f} GB        ║
║ Memory (Training): {memory_info['total_training_gb']:>19.2f} GB        ║
╚══════════════════════════════════════════════════════════════╝
"""
        return summary
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.config.model_name}, params={self.get_num_params():,})"


class BaseTokenizer(ABC):
    """
    Abstract base class cho tokenizers
    """
    
    @abstractmethod
    def encode(self, text: str, **kwargs) -> List[int]:
        """Encode text thành token IDs"""
        pass
    
    @abstractmethod
    def decode(self, token_ids: List[int], **kwargs) -> str:
        """Decode token IDs thành text"""
        pass
    
    @abstractmethod
    def get_vocab_size(self) -> int:
        """Lấy kích thước vocabulary"""
        pass
    
    @property
    @abstractmethod
    def pad_token_id(self) -> int:
        """ID của padding token"""
        pass
    
    @property
    @abstractmethod
    def eos_token_id(self) -> int:
        """ID của end-of-sequence token"""
        pass
    
    @property
    @abstractmethod
    def bos_token_id(self) -> int:
        """ID của beginning-of-sequence token"""
        pass

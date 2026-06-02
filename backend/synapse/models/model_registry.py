"""
Model Registry System - Quản lý tất cả các mô hình trong hệ thống

Registry này cho phép:
- Đăng ký mô hình mới dễ dàng
- Tạo instance của mô hình từ tên
- Liệt kê và tìm kiếm mô hình
- Quản lý cấu hình mô hình tập trung
"""

from typing import Dict, Type, Callable, Optional, List
from .base_model import BaseModel, ModelConfig, ModelType, TaskType


class ModelRegistry:
    """
    Singleton registry để quản lý tất cả các mô hình trong hệ thống
    
    Usage:
        # Đăng ký mô hình
        ModelRegistry.register("my-model", MyModelClass, config)
        
        # Tạo instance
        model = ModelRegistry.create_model("my-model")
        
        # Liệt kê mô hình
        models = ModelRegistry.list_models(model_type=ModelType.NLP)
    """
    
    # Class variables để lưu trữ registry
    _models: Dict[str, Type[BaseModel]] = {}
    _configs: Dict[str, ModelConfig] = {}
    _builders: Dict[str, Callable] = {}
    _metadata: Dict[str, Dict] = {}
    
    @classmethod
    def register(
        cls,
        name: str,
        model_class: Type[BaseModel],
        config: ModelConfig,
        builder: Optional[Callable] = None,
        metadata: Optional[Dict] = None
    ):
        """
        Đăng ký một mô hình mới vào registry
        
        Args:
            name: Tên định danh của mô hình (unique)
            model_class: Class của mô hình (phải kế thừa BaseModel)
            config: ModelConfig object
            builder: Optional custom builder function
            metadata: Optional metadata (author, description, etc.)
        
        Raises:
            ValueError: Nếu tên mô hình đã tồn tại
            TypeError: Nếu model_class không kế thừa BaseModel
        """
        # Validation
        if name in cls._models:
            raise ValueError(f"Model '{name}' đã được đăng ký rồi!")
        
        if not issubclass(model_class, BaseModel):
            raise TypeError(f"Model class phải kế thừa từ BaseModel")
        
        # Register
        cls._models[name] = model_class
        cls._configs[name] = config
        
        if builder:
            cls._builders[name] = builder
        
        if metadata:
            cls._metadata[name] = metadata
        else:
            cls._metadata[name] = {
                "author": "Unknown",
                "description": f"{name} model",
                "version": "1.0.0"
            }
        
        print(f"✅ Đã đăng ký mô hình: {name}")
        print(f"   - Type: {config.model_type.value}")
        print(f"   - Task: {config.task_type.value}")
        print(f"   - Layers: {config.num_layers}")
        print(f"   - Hidden Size: {config.hidden_size}")
    
    @classmethod
    def unregister(cls, name: str):
        """
        Hủy đăng ký một mô hình
        
        Args:
            name: Tên mô hình cần hủy đăng ký
        """
        if name in cls._models:
            del cls._models[name]
            del cls._configs[name]
            if name in cls._builders:
                del cls._builders[name]
            if name in cls._metadata:
                del cls._metadata[name]
            print(f"❌ Đã hủy đăng ký mô hình: {name}")
        else:
            print(f"⚠️  Mô hình '{name}' không tồn tại trong registry")
    
    @classmethod
    def get_model_class(cls, name: str) -> Type[BaseModel]:
        """
        Lấy model class từ tên
        
        Args:
            name: Tên mô hình
            
        Returns:
            Model class
            
        Raises:
            ValueError: Nếu mô hình không tồn tại
        """
        if name not in cls._models:
            available = ", ".join(cls._models.keys())
            raise ValueError(
                f"Mô hình '{name}' không tồn tại trong registry.\n"
                f"Các mô hình có sẵn: {available}"
            )
        return cls._models[name]
    
    @classmethod
    def get_config(cls, name: str) -> ModelConfig:
        """
        Lấy config của mô hình
        
        Args:
            name: Tên mô hình
            
        Returns:
            ModelConfig object
            
        Raises:
            ValueError: Nếu mô hình không tồn tại
        """
        if name not in cls._configs:
            raise ValueError(f"Config cho mô hình '{name}' không tồn tại")
        return cls._configs[name]
    
    @classmethod
    def get_metadata(cls, name: str) -> Dict:
        """
        Lấy metadata của mô hình
        
        Args:
            name: Tên mô hình
            
        Returns:
            Dictionary chứa metadata
        """
        return cls._metadata.get(name, {})
    
    @classmethod
    def create_model(cls, name: str, **kwargs) -> BaseModel:
        """
        Tạo instance của mô hình
        
        Args:
            name: Tên mô hình
            **kwargs: Các tham số bổ sung để truyền vào constructor
            
        Returns:
            Instance của mô hình
            
        Example:
            model = ModelRegistry.create_model("llama-7b", device="cuda")
        """
        # Nếu có custom builder, dùng nó
        if name in cls._builders:
            return cls._builders[name](**kwargs)
        
        # Nếu không, tạo từ class và config
        model_class = cls.get_model_class(name)
        config = cls.get_config(name)
        
        return model_class(config, **kwargs)
    
    @classmethod
    def list_models(
        cls,
        model_type: Optional[ModelType] = None,
        task_type: Optional[TaskType] = None
    ) -> List[str]:
        """
        Liệt kê các mô hình trong registry
        
        Args:
            model_type: Lọc theo loại mô hình (optional)
            task_type: Lọc theo loại task (optional)
            
        Returns:
            List tên các mô hình
            
        Example:
            # Lấy tất cả mô hình NLP
            nlp_models = ModelRegistry.list_models(model_type=ModelType.NLP)
            
            # Lấy tất cả mô hình text generation
            gen_models = ModelRegistry.list_models(task_type=TaskType.TEXT_GENERATION)
        """
        models = []
        for name, config in cls._configs.items():
            # Filter by model_type
            if model_type and config.model_type != model_type:
                continue
            
            # Filter by task_type
            if task_type and config.task_type != task_type:
                continue
            
            models.append(name)
        
        return sorted(models)
    
    @classmethod
    def get_models_by_type(cls, model_type: ModelType) -> Dict[str, ModelConfig]:
        """
        Lấy tất cả mô hình theo loại
        
        Args:
            model_type: Loại mô hình
            
        Returns:
            Dictionary {model_name: config}
        """
        return {
            name: config
            for name, config in cls._configs.items()
            if config.model_type == model_type
        }
    
    @classmethod
    def get_models_by_task(cls, task_type: TaskType) -> Dict[str, ModelConfig]:
        """
        Lấy tất cả mô hình theo task
        
        Args:
            task_type: Loại task
            
        Returns:
            Dictionary {model_name: config}
        """
        return {
            name: config
            for name, config in cls._configs.items()
            if config.task_type == task_type
        }
    
    @classmethod
    def search_models(cls, query: str) -> List[str]:
        """
        Tìm kiếm mô hình theo tên
        
        Args:
            query: Chuỗi tìm kiếm
            
        Returns:
            List tên các mô hình khớp với query
        """
        query_lower = query.lower()
        return [
            name for name in cls._models.keys()
            if query_lower in name.lower()
        ]
    
    @classmethod
    def print_registry(cls):
        """In ra tất cả mô hình trong registry"""
        if not cls._models:
            print("📋 Registry trống - chưa có mô hình nào được đăng ký")
            return
        
        print("\n" + "="*80)
        print("📋 MODEL REGISTRY")
        print("="*80)
        
        # Group by model type
        for model_type in ModelType:
            models = cls.get_models_by_type(model_type)
            if not models:
                continue
            
            print(f"\n🔹 {model_type.value.upper()} Models:")
            print("-" * 80)
            
            for name, config in models.items():
                metadata = cls._metadata.get(name, {})
                print(f"  • {name}")
                print(f"    Task: {config.task_type.value}")
                print(f"    Layers: {config.num_layers}, Hidden: {config.hidden_size}")
                print(f"    Author: {metadata.get('author', 'Unknown')}")
                print(f"    Description: {metadata.get('description', 'N/A')}")
                print()
        
        print("="*80)
        print(f"Tổng số mô hình: {len(cls._models)}")
        print("="*80 + "\n")
    
    @classmethod
    def clear(cls):
        """Xóa toàn bộ registry (dùng cho testing)"""
        cls._models.clear()
        cls._configs.clear()
        cls._builders.clear()
        cls._metadata.clear()
        print("🗑️  Đã xóa toàn bộ registry")


def register_model(
    name: str,
    config: ModelConfig,
    metadata: Optional[Dict] = None
):
    """
    Decorator để đăng ký mô hình dễ dàng
    
    Usage:
        @register_model("my-model", config, metadata={"author": "Me"})
        class MyModel(BaseModel):
            ...
    
    Args:
        name: Tên mô hình
        config: ModelConfig object
        metadata: Optional metadata
        
    Returns:
        Decorator function
    """
    def decorator(model_class: Type[BaseModel]):
        ModelRegistry.register(name, model_class, config, metadata=metadata)
        return model_class
    return decorator

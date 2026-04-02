"""
Training framework - PyTorch/HF (LoRA qua finetune_lora).
"""

from typing import Optional, Callable, Dict, Any
from pathlib import Path


class TrainerConfig:
    """Cấu hình training đơn giản (stub - dùng LoRA qua finetune_lora)."""
    def __init__(
        self,
        output_dir: str = "./checkpoints",
        num_epochs: int = 10,
        batch_size: int = 8,
        learning_rate: float = 1e-4,
        save_every: int = 5,
        **kwargs
    ):
        self.output_dir = Path(output_dir)
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.save_every = save_every
        self.custom_params = kwargs


class SimpleTrainer:
    """Stub: Dùng Training (LoRA) trong Dashboard hoặc finetune_lora."""

    def __init__(
        self,
        model: Any,
        train_dataset: Any,
        config: Optional[TrainerConfig] = None,
        loss_fn: Optional[Callable] = None,
        optimizer: Optional[Any] = None,
    ):
        raise NotImplementedError(
            "Dùng Training (LoRA) trong Dashboard hoặc API /v1/training/start."
        )

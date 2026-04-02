"""
Dataset Management - Quản lý dữ liệu training

Hỗ trợ JSON/JSONL format
"""

from pathlib import Path
import json
from typing import List, Dict, Any, Callable, Optional


class SimpleDataset:
    """
    Dataset đơn giản cho training
    
    Support JSON/JSONL files
    """
    
    def __init__(
        self,
        data_path: str,
        preprocess: Optional[Callable] = None
    ):
        self.data_path = Path(data_path)
        self.preprocess = preprocess or (lambda x: x)
        self.data = []
        
        # Load data
        self._load_data()
    
    def _load_data(self):
        """Load data from file"""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.data_path}")
        
        # Load JSONL
        if self.data_path.suffix == '.jsonl':
            with open(self.data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        self.data.append(json.loads(line))
        
        # Load JSON
        elif self.data_path.suffix == '.json':
            with open(self.data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    self.data = data
                else:
                    self.data = [data]
        
        print(f"📊 Loaded {len(self.data)} samples from {self.data_path.name}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.preprocess(self.data[idx])
    
    def get_batches(self, batch_size: int):
        """Get batches"""
        for i in range(0, len(self.data), batch_size):
            batch_data = self.data[i:i+batch_size]
            yield {
                "inputs": [self.preprocess(item) for item in batch_data],
                "targets": batch_data  # Placeholder
            }


def load_dataset(data_dir: str):
    """
    Load train/val/test datasets
    
    Args:
        data_dir: Directory containing train.jsonl, val.jsonl, test.jsonl
        
    Returns:
        Tuple of (train, val, test) datasets
    """
    data_dir = Path(data_dir)
    
    train_path = data_dir / "train.jsonl"
    val_path = data_dir / "val.jsonl"
    test_path = data_dir / "test.jsonl"
    
    train = SimpleDataset(train_path) if train_path.exists() else None
    val = SimpleDataset(val_path) if val_path.exists() else None
    test = SimpleDataset(test_path) if test_path.exists() else None
    
    return train, val, test

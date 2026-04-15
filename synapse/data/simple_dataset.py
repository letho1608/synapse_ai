"""
Dataset Management - Quản lý dữ liệu training

Hỗ trợ JSON/JSONL format
"""

from pathlib import Path
import json
import os
try:
    from PIL import Image
except ImportError:
    pass
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


class ImageDataset:
    """
    Dataset dành cho Hình ảnh (Vision/OCR & Classification).
    Support:
     1. Chuẩn metadata.jsonl / labels.txt.
     2. Chuẩn phân loại thư mục dựa trên Folder con.
    """
    def __init__(
        self,
        data_path: str,
        preprocess: Optional[Callable] = None,
        img_channel: str = 'RGB'
    ):
        self.data_path = Path(data_path)
        self.preprocess = preprocess or (lambda x: x)
        self.img_channel = img_channel
        self.data = []
        
        self._load_data()
        
    def _load_data(self):
        if not self.data_path.exists():
            raise FileNotFoundError(f"Image dataset not found: {self.data_path}")
            
        jsonl_path = self.data_path / "metadata.jsonl"
        labels_txt_path = self.data_path / "labels.txt"
        pk_path = self.data_path / "transcription.pk"
        
        # 1. Định dạng Pickle (Dictionary list: [{'path': 'text'}])
        if pk_path.exists():
            import pickle
            with open(pk_path, 'rb') as f:
                data_list = pickle.load(f)
            for entry in data_list:
                for rel_path, text in entry.items():
                    norm_path = rel_path.replace('/', os.sep).replace('\\', os.sep)
                    img_path = self.data_path / norm_path
                    if img_path.exists():
                        self.data.append({"image_path": str(img_path), "text": text, "label": text})
            print(f"🖼️ Loaded {len(self.data)} pairs from transcription.pk at {self.data_path.name}")
            
        # 2. HuggingFace Vision format - metadata.jsonl
        elif jsonl_path.exists():
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line)
                        file_name = item.get("file_name", "")
                        text = item.get("text", "")
                        img_path = self.data_path / file_name
                        if img_path.exists():
                            self.data.append({"image_path": str(img_path), "text": text, "label": text})
            print(f"🖼️ Loaded {len(self.data)} pairs from metadata.jsonl at {self.data_path.name}")
            
        # 3. OCR TXT format - labels.txt
        elif labels_txt_path.exists():
            with open(labels_txt_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split(maxsplit=1)
                    if len(parts) == 2:
                        file_name, text = parts[0], parts[1]
                        img_path = self.data_path / file_name
                        if img_path.exists():
                            self.data.append({"image_path": str(img_path), "text": text, "label": text})
            print(f"🖼️ Loaded {len(self.data)} pairs from labels.txt at {self.data_path.name}")
            

        # 4. ImageFolder / Classification Format
        elif self.data_path.is_dir():
            class_names = [d.name for d in self.data_path.iterdir() if d.is_dir()]
            if not class_names:
                for ext in ['.jpg', '.png', '.jpeg', '.webp']:
                    for img_path in self.data_path.glob(f"*{ext}"):
                        self.data.append({"image_path": str(img_path), "label": None})
            else:
                for idx, class_name in enumerate(class_names):
                    class_dir = self.data_path / class_name
                    for ext in ['.jpg', '.png', '.jpeg', '.webp', '.JPG', '.PNG']:
                        for img_path in class_dir.glob(f"*{ext}"):
                            self.data.append({
                                "image_path": str(img_path), 
                                "label": idx, 
                                "class_name": class_name
                            })
            print(f"🖼️ Loaded {len(self.data)} images from Directory Tree at {self.data_path.name}")

    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        item = self.data[idx].copy()
        try:
            img = Image.open(item["image_path"]).convert(self.img_channel)
            item["image"] = img
        except Exception as e:
            print(f"Lỗi mở ảnh {item['image_path']}: {e}")
            item["image"] = None
        return self.preprocess(item)

    def get_batches(self, batch_size: int):
        for i in range(0, len(self.data), batch_size):
            batch_data = self.data[i:i+batch_size]
            yield {
                "inputs": [self.preprocess(item) for item in batch_data],
                "targets": batch_data
            }


def load_dataset(data_dir: str, dataset_type: str = "text", img_channel: str = 'RGB'):
    """
    Load train/val/test datasets
    
    Args:
        data_dir: Directory containing train.jsonl, val.jsonl hoặc train/, val/, test/ folders
        dataset_type: "text" (JSON/JSONL) or "image"
        
    Returns:
        Tuple of (train, val, test) datasets
    """
    data_dir = Path(data_dir)
    
    if dataset_type == "image":
        train_path = data_dir / "train"
        val_path = data_dir / "val"
        test_path = data_dir / "test"
        
        train = ImageDataset(train_path, img_channel=img_channel) if train_path.exists() else (ImageDataset(data_dir, img_channel=img_channel) if data_dir.is_dir() else None)
        val = ImageDataset(val_path, img_channel=img_channel) if val_path.exists() else None
        test = ImageDataset(test_path, img_channel=img_channel) if test_path.exists() else None
        return train, val, test

    train_path = data_dir / "train.jsonl"
    val_path = data_dir / "val.jsonl"
    test_path = data_dir / "test.jsonl"
    
    train = SimpleDataset(train_path) if train_path.exists() else None
    val = SimpleDataset(val_path) if val_path.exists() else None
    test = SimpleDataset(test_path) if test_path.exists() else None
    
    return train, val, test

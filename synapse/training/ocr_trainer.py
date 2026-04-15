"""
CRNN (CNN + Bi-LSTM + CTC) Training for Vietnamese Handwriting.
Tích hợp vào Synapse AI framework.
"""

import os
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional

# Cấu hình kiến trúc (Tương tự code của người dùng cung cấp)
class CRNN(nn.Module):
    def __init__(self, num_chars, img_height=64):
        super(CRNN, self).__init__()
        # CNN layers (Feature extraction)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2), # 32
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2), # 16
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(), nn.MaxPool2d((2, 1)), # 8
            nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(), nn.MaxPool2d((2, 1)), # 4
            nn.Conv2d(512, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.MaxPool2d((2, 1)), # Giảm về 2px trước khi vào Conv cuối để đạt Height=1
            nn.Conv2d(512, 512, 2, 1, 0), nn.BatchNorm2d(512), nn.ReLU() # 1
        )
        # Bi-LSTM layers (Sequence modeling)
        self.rnn = nn.Sequential(
            nn.LSTM(512, 256, bidirectional=True, num_layers=2, batch_first=False)
        )
        # Output layer
        self.fc = nn.Linear(512, num_chars)

    def forward(self, x):
        # x: [B, 1, H, W]
        conv = self.cnn(x) # [B, 512, 1, W']
        b, c, h, w = conv.size()
        assert h == 1, "The height of conv features should be 1"
        conv = conv.squeeze(2) # [B, 512, W']
        conv = conv.permute(2, 0, 1) # [W', B, 512] (Sequence first for LSTM)
        
        output, _ = self.rnn(conv)
        T, B, H = output.size()
        output = self.fc(output.view(T * B, H))
        output = output.view(T, B, -1) # [T, B, num_chars]
        return output

class OCRDatasetAdapter(Dataset):
    """Adapter cho ImageDataset của Synapse để dùng với PyTorch DataLoader."""
    def __init__(self, data: List[Dict[str, Any]], vocab: Dict[str, int], img_height=64, img_width=448):
        self.data = data
        self.vocab = vocab
        self.img_height = img_height
        self.img_width = img_width
        self.char_list = sorted(list(vocab.keys()), key=lambda x: vocab[x])
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        item = self.data[idx]
        img_path = item["image_path"]
        text = item.get("text", "")
        
        try:
            img = Image.open(img_path).convert('L') # Grayscale
            img = img.resize((self.img_width, self.img_height), Image.BILINEAR)
            img = np.array(img, dtype=np.float32) / 255.0
            img = (img - 0.5) / 0.5 # Normalize to [-1, 1]
            img = torch.from_numpy(img).unsqueeze(0) # [1, H, W]
        except Exception:
            # Fallback
            img = torch.zeros((1, self.img_height, self.img_width))
            
        # Encode text
        labels = [self.vocab.get(c, 0) for c in text] # 0 is usually blank or unknown
        label_len = len(labels)
        
        return img, torch.IntTensor(labels), label_len

async def train_crnn_ocr(job: Dict[str, Any], raw_data: List[Dict[str, Any]], node: Optional[Any] = None):
    """
    Tiến trình training CRNN cho OCR Tiếng Việt.
    Hỗ trợ cả Local và Distributed (Master-Stream).
    """
    job_id = job.get("job_id")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Build Vocab
    chars = set()
    for item in raw_data:
        chars.update(list(item.get("text", "")))
    char_list = sorted(list(chars))
    vocab = {char: i + 1 for i, char in enumerate(char_list)}
    num_class = len(char_list) + 1
    
    checkpoint_dir = Path(job.get("checkpoint_dir", "checkpoints"))
    model_name = job.get("output_model_name", "vietnamese_crnn_latest")
    out_dir = checkpoint_dir / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    
    with open(out_dir / "vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    # 2. Setup Data
    dataset = OCRDatasetAdapter(raw_data, vocab)
    
    def collate_fn(batch):
        imgs, labels, label_lens = zip(*batch)
        imgs = torch.stack(imgs)
        max_label_len = max(label_lens)
        padded_labels = torch.zeros((len(labels), max_label_len), dtype=torch.int32)
        for i, l in enumerate(labels):
            padded_labels[i, :len(l)] = l
        return imgs, padded_labels, torch.IntTensor(label_lens)

    loader = DataLoader(dataset, batch_size=job.get("batch_size", 32), shuffle=True, collate_fn=collate_fn)
    
    # 3. Setup Model (Chỉ dùng Local nếu không có node)
    model = None
    optimizer = None
    criterion = None
    
    if node is None:
        model = CRNN(num_class).to(device)
        criterion = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.0005)
    else:
        print(f"[OCR-Stream] Chế độ Phân tán: Đang stream tới mạng lưới Synapse...")
        from synapse.inference.shard import Shard
        # Thử lấy model_id thực tế từ model_list
        from synapse.model_list import resolve_hf_id
        model_id = job.get("model", "vietnamese_crnn")
        n_layers = 1 # CRNN coi như 1 block lớn trong demo này
        base_shard = Shard(model_id=model_id, start_layer=0, end_layer=n_layers)

    epochs = job.get("epochs", 50)
    total_steps = len(loader) * epochs
    step_count = 0
    
    job["status"] = "training"
    
    for epoch in range(epochs):
        if model: model.train()
        
        if job.get("status") == "cancelled":
            break
            
        for i, (imgs, labels, label_lens) in enumerate(loader):
            if job.get("status") == "cancelled":
                break
            
            loss_item = 0.0
            
            if node:
                # --- CHẾ ĐỘ STREAM ---
                # Chuyển batch thành numpy để truyền qua gRPC/Networking
                imgs_np = imgs.numpy()
                labels_np = labels.numpy()
                lens_np = label_lens.numpy()
                
                # Gửi tới mạng lưới. Node sẽ tự động shard/forward/backward.
                loss_val = await node.enqueue_example(
                    base_shard, 
                    imgs_np, 
                    labels_np, 
                    lens_np, 
                    request_id=job_id, 
                    train=True
                )
                loss_item = float(loss_val) if isinstance(loss_val, (int, float)) else 0.0
            else:
                # --- CHẾ ĐỘ LOCAL ---
                imgs = imgs.to(device)
                batch_size = imgs.size(0)
                preds = model(imgs).log_softmax(2)
                T = preds.size(0)
                input_lens = torch.full(size=(batch_size,), fill_value=T, dtype=torch.int32).to(device)
                
                loss = criterion(preds, labels, input_lens, label_lens)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_item = loss.item()
            
            step_count += 1
            job["current_step"] = step_count
            job["current_epoch"] = epoch
            job["progress"] = min(100, int(100 * step_count / total_steps))
            job["loss"] = round(loss_item, 4) if loss_item else None
            
        # Lưu checkpoint (Trong chế độ phân tán, node master sẽ giữ trọng số)
        if (epoch + 1) % job.get("save_every", 5) == 0 or epoch == epochs - 1:
            if model:
                torch.save(model.state_dict(), out_dir / f"checkpoint_epoch_{epoch+1}.pth")
            elif node:
                # Yêu cầu node lưu checkpoint
                await node.coordinate_save(base_shard, epoch + 1, str(checkpoint_dir))

    if job.get("status") != "cancelled":
        job["status"] = "completed"
        job["progress"] = 100
        job["completed_at"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

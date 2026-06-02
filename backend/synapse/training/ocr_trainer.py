"""
CRNN (CNN + Bi-LSTM + CTC) Training for Vietnamese Handwriting.
Tich hop vao Synapse pipeline phan tan (LACP + Ring All-Reduce).
"""

from __future__ import annotations

import os
import time
import json
import asyncio
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional


class CRNN(nn.Module):
    def __init__(self, num_chars, img_height=64):
        super(CRNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(), nn.MaxPool2d((2, 1)),
            nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(), nn.MaxPool2d((2, 1)),
            nn.Conv2d(512, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(512, 512, 2, 1, 0), nn.BatchNorm2d(512), nn.ReLU()
        )
        self.rnn = nn.Sequential(
            nn.LSTM(512, 256, bidirectional=True, num_layers=2, batch_first=False)
        )
        self.fc = nn.Linear(512, num_chars)

    def forward(self, x):
        conv = self.cnn(x)
        b, c, h, w = conv.size()
        assert h == 1, "The height of conv features should be 1"
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)
        output, _ = self.rnn(conv)
        T, B, H = output.size()
        output = self.fc(output.view(T * B, H))
        output = output.view(T, B, -1)
        return output


class OCRDatasetAdapter(Dataset):
    def __init__(self, data: List[Dict[str, Any]], vocab: Dict[str, int], img_height=64, img_width=448):
        self.data = data
        self.vocab = vocab
        self.img_height = img_height
        self.img_width = img_width

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img_path = item["image_path"]
        text = item.get("text", "")
        try:
            img = Image.open(img_path).convert('L')
            img = img.resize((self.img_width, self.img_height), Image.BILINEAR)
            img = np.array(img, dtype=np.float32) / 255.0
            img = (img - 0.5) / 0.5
            img = torch.from_numpy(img).unsqueeze(0)
        except Exception:
            img = torch.zeros((1, self.img_height, self.img_width))
        labels = [self.vocab.get(c, 0) for c in text]
        label_len = len(labels)
        return img, torch.IntTensor(labels), label_len


from synapse.loading import get_checkpoints_dir


async def train_crnn_ocr(job: Dict[str, Any], raw_data: List[Dict[str, Any]], node: Optional[Any] = None):
    job_id = job.get("job_id")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    chars = set()
    for item in raw_data:
        chars.update(list(item.get("text", "")))
    char_list = sorted(list(chars))
    vocab = {char: i + 1 for i, char in enumerate(char_list)}
    num_class = len(char_list) + 1

    checkpoint_dir = Path(job.get("checkpoint_dir") or str(get_checkpoints_dir(job.get("model", "vietnamese_crnn"))))
    output_model_name = job.get("output_model_name", "vietnamese_crnn_latest")
    out_dir = checkpoint_dir / output_model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

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
    epochs = job.get("epochs", 50)
    total_steps = len(loader) * epochs
    step_count = 0
    save_every = int(job.get("save_every", 5))
    lr_val = float((job.get("learning_rate") or "0.0005").strip()) if job.get("learning_rate") else 0.0005
    world_size = 1
    use_distributed = False

    if node is not None:
        world_size = len(getattr(node, "peers", [])) + 1
        ext = getattr(node, "_pipeline_extension", None)
        use_distributed = ext is not None and world_size > 1

    if use_distributed:
        from synapse.pipeline.gradient_sync import DistributedTrainer, GradientShard
        trainer = DistributedTrainer(node_id=node.id, world_size=world_size, learning_rate=lr_val)
    else:
        model = CRNN(num_class).to(device)
        criterion = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr_val)
        trainer = None

    job["status"] = "training"

    for epoch in range(epochs):
        if node and job.get("status") == "cancelled":
            break
        if use_distributed:
            model_to_set = None
        elif hasattr(model, "train"):
            model.train()

        for i, (imgs, labels, label_lens) in enumerate(loader):
            if node and job.get("status") == "cancelled":
                break

            if use_distributed:
                loss_item = await _train_step_distributed_ocr(
                    node, job_id, imgs, labels, label_lens, device, epochs, step_count, total_steps
                )
            else:
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
            await asyncio.sleep(0)

        if (epoch + 1) % save_every == 0 or epoch == epochs - 1:
            if not use_distributed:
                torch.save(model.state_dict(), out_dir / f"checkpoint_epoch_{epoch+1}.pth")
            elif node:
                try:
                    from synapse.inference.shard import Shard
                    model_id = job.get("model", "vietnamese_crnn")
                    base_shard = Shard(model_id=model_id, start_layer=0, end_layer=1)
                    await node.coordinate_save(base_shard, epoch + 1, str(checkpoint_dir))
                except Exception:
                    pass

    if job.get("status") != "cancelled":
        job["status"] = "completed"
        job["progress"] = 100
        job["completed_at"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


async def _train_step_distributed_ocr(
    node: Any, job_id: str, imgs: torch.Tensor, labels: torch.Tensor,
    label_lens: torch.Tensor, device, epochs: int, step_count: int, total_steps: int
) -> float:
    from synapse.inference.shard import Shard
    from synapse.models import build_base_shard

    ext = getattr(node, "_pipeline_extension", None)
    if ext is None:
        return 0.0

    base_shard = Shard(model_id="vietnamese_crnn", start_layer=0, end_layer=1)
    imgs_np = imgs.numpy()
    labels_np = labels.numpy()

    world_size = len(getattr(node, "peers", [])) + 1
    if world_size > 1:
        loss_val = await ext.train_step_distributed(
            base_shard, imgs_np, labels_np,
            request_id=f"{job_id}_{step_count}",
            learning_rate=0.0005,
        )
    else:
        lens_np = label_lens.numpy()
        loss_val = await node.enqueue_example(
            base_shard, imgs_np, labels_np, lens_np,
            request_id=f"{job_id}_{step_count}",
            train=True,
        )
    return float(loss_val) if isinstance(loss_val, (int, float)) else 0.0
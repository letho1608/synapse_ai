"""
Finetuning LLM that voi Hugging Face Transformers + PEFT (LoRA).
Support ca Local va Distributed (Pipeline LACP + Ring All-Reduce).
"""

from __future__ import annotations

from pathlib import Path
import json
import time
import asyncio
from typing import Dict, List, Any, Optional, TYPE_CHECKING

from synapse.model_list import resolve_hf_id
from synapse.training.preprocessing import preprocess
from synapse.loading import get_checkpoints_dir

if TYPE_CHECKING:
    import numpy as np


def resolve_hf_model_id(model_name: str, base_model_id: Optional[str] = None) -> str:
    if (base_model_id or "").strip():
        return base_model_id.strip()
    return resolve_hf_id(model_name)


def load_raw_data(path: Path) -> List[Dict[str, Any]]:
    path = Path(path).resolve()
    if path.is_dir():
        from synapse.data.simple_dataset import ImageDataset
        try:
            ds = ImageDataset(str(path))
            if len(ds.data) > 0:
                return ds.data
        except Exception as e:
            print(f"Loi nap ImageDataset tu {path}: {e}")
    if not path.is_file():
        raise FileNotFoundError(f"Dataset khong ton tai hoac khong phai la file hop le: {path}")
    data = []
    if path.suffix == ".jsonl":
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
    else:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        data = raw if isinstance(raw, list) else [raw]
    return data


async def load_raw_data_async(path: Path) -> List[Dict[str, Any]]:
    """Async version - loads file I/O in thread executor."""
    import asyncio
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, load_raw_data, path)


def format_sample_to_text(sample: Dict[str, Any], tokenizer: Any) -> str:
    if "messages" in sample:
        msgs = sample["messages"]
        if hasattr(tokenizer, "apply_chat_template"):
            return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False,)
        parts = []
        for m in msgs:
            role = m.get("role", "user")
            content = m.get("content") or ""
            parts.append(f"<|{role}|>\n{content}")
        return "\n".join(parts) + "\n"
    if "instruction" in sample or "output" in sample:
        inst = sample.get("instruction", "")
        inp = sample.get("input", "")
        out = sample.get("output", "")
        if inp:
            text = f"Instruction: {inst}\nInput: {inp}\nResponse: {out}"
        else:
            text = f"Instruction: {inst}\nResponse: {out}"
        return text + "\n"
    if "text" in sample:
        return sample["text"] if isinstance(sample["text"], str) else str(sample["text"])
    return json.dumps(sample, ensure_ascii=False)


def _build_lora_config(job: Dict[str, Any]):
    try:
        from peft import LoraConfig, TaskType
    except ImportError:
        return None
    r = int(job.get("lora_r", 8))
    alpha = int(job.get("lora_alpha", 16))
    dropout = float(job.get("lora_dropout", 0.05))
    target_modules = job.get("lora_target_modules")
    if isinstance(target_modules, str):
        target_modules = [m.strip() for m in target_modules.split(",") if m.strip()]
    if not target_modules:
        target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
    return LoraConfig(r=r, lora_alpha=alpha, target_modules=target_modules, lora_dropout=dropout, bias="none", task_type=TaskType.CAUSAL_LM,)


def run_finetune_sync(job: Dict[str, Any]) -> None:
    """
    Chay finetuning LoRA local (blocking, HF Trainer).
    Co tien xu ly dataset: filter rac, dedup, validate.
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
        from peft import get_peft_model
        from datasets import Dataset as HFDataset
    except ImportError as e:
        job["status"] = "failed"
        job["error"] = f"Thieu thu vien: {e}. Cai: pip install torch transformers peft datasets accelerate"
        return

    job_id = job.get("job_id", "")
    model_name = job.get("model", "")
    dataset_path = (job.get("dataset") or "").strip()
    epochs = int(job.get("epochs", 3))
    batch_size = int(job.get("batch_size", 4))
    lr_str = (job.get("learning_rate") or "").strip() or "2e-5"
    learning_rate = float(lr_str) if lr_str.replace(".", "", 1).replace("e", "", 1).replace("-", "", 1).lstrip("-").isdigit() else 2e-5
    try:
        learning_rate = float(lr_str)
    except ValueError:
        learning_rate = 2e-5
    save_every = int(job.get("save_every", 1))
    max_steps = int(job.get("max_steps", 0))
    warmup_steps = int(job.get("warmup_steps", 0))
    output_model_name = (job.get("output_model_name") or job.get("output_name") or "").strip()
    checkpoint_dir = (job.get("checkpoint_dir") or "").strip() or str(get_checkpoints_dir(model_name))
    base_model_id = (job.get("base_model_id") or "").strip() or None
    min_chars = int(job.get("min_chars", 10))
    max_chars = int(job.get("max_chars", 100000))
    skip_dedup = job.get("skip_dedup", False)

    try:
        hf_model_id = resolve_hf_model_id(model_name, base_model_id)
    except Exception as e:
        job["status"] = "failed"
        job["error"] = f"Khong xac dinh duoc base model: {e}"
        return

    path = Path(dataset_path)
    if not path.is_absolute():
        base = Path("synapse/data")
        if (base / path.name).exists():
            path = base / path.name
        elif (base / path).exists():
            path = base / path
    if not path.exists():
        job["status"] = "failed"
        job["error"] = f"Dataset khong ton tai: {path}"
        return

    job["status"] = "started"
    job["progress"] = 0
    job["current_epoch"] = 0
    job["error"] = None
    job["loss"] = None

    try:
        raw_data = load_raw_data(path)
        if not raw_data:
            job["status"] = "failed"
            job["error"] = "Dataset rong"
            return

        pp = preprocess(raw_data, format_type="auto", min_chars=min_chars, max_chars=max_chars, dedup=not skip_dedup,)
        raw_data = pp["processed_samples"]
        job["preprocess"] = {
            "original_count": pp["original_count"],
            "empty_removed": pp["empty_removed"],
            "short_removed": pp["short_removed"],
            "duplicate_removed": pp["duplicate_removed"],
            "validation_errors": pp["validation_errors"],
            "valid_count": pp["valid_count"],
            "format_detected": pp["format_detected"],
        }
        if not raw_data:
            job["status"] = "failed"
            job["error"] = "Khong co mau hop le sau tien xu ly"
            return

        tokenizer = AutoTokenizer.from_pretrained(hf_model_id, trust_remote_code=True)
        if getattr(tokenizer, "pad_token", None) is None:
            tokenizer.pad_token = tokenizer.eos_token

        texts = []
        for sample in raw_data:
            try:
                t = format_sample_to_text(sample, tokenizer)
                if t and t.strip():
                    texts.append(t)
            except Exception:
                continue
        if not texts:
            job["status"] = "failed"
            job["error"] = "Khong co mau hop le sau khi format"
            return

        max_length = min(int(job.get("max_length", 512)), 2048)
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

        def tokenize_fn(examples):
            out = tokenizer(examples["text"], truncation=True, max_length=max_length, padding="max_length", return_tensors=None,)
            labels = []
            for ids in out["input_ids"]:
                lab = [id_ if id_ != pad_id else -100 for id_ in ids]
                labels.append(lab)
            out["labels"] = labels
            return out

        hf_dataset = HFDataset.from_dict({"text": texts})
        tokenized = hf_dataset.map(tokenize_fn, batched=True, remove_columns=["text"], desc="Tokenize")

        model = AutoModelForCausalLM.from_pretrained(
            hf_model_id, trust_remote_code=True,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        lora_config = _build_lora_config(job)
        if lora_config is None:
            job["status"] = "failed"
            job["error"] = "Thieu peft"
            return
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        out_dir = Path(checkpoint_dir) / (output_model_name or f"lora_{job_id[:8]}")
        out_dir.mkdir(parents=True, exist_ok=True)

        total_steps = max_steps if max_steps > 0 else (len(tokenized) // batch_size) * epochs
        if total_steps <= 0:
            total_steps = 100

        training_args = TrainingArguments(
            output_dir=str(out_dir),
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps if warmup_steps > 0 else min(50, total_steps // 10),
            max_steps=max_steps if max_steps > 0 else -1,
            save_strategy="epoch",
            save_total_limit=2,
            logging_steps=5,
            report_to="none",
            bf16=torch.cuda.is_available(),
            fp16=not torch.cuda.is_available(),
        )

        current_step = [0]

        def progress_callback(args):
            current_step[0] = args.get("global_step", 0)
            if job.get("job_id") != job_id:
                return
            job["current_step"] = current_step[0]
            total = args.get("max_steps") or total_steps
            job["progress"] = min(100, int(100 * current_step[0] / total)) if total else 0
            job["current_epoch"] = args.get("epoch", 0) or (current_step[0] * batch_size // len(tokenized))
            loss = args.get("loss")
            if loss is not None:
                job["loss"] = round(float(loss), 4)

        class ProgressTrainer(Trainer):
            def __init__(self, *a, **k):
                super().__init__(*a, **k)

            def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval):
                log_result = super()._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)
                if self.state.log_history:
                    last = self.state.log_history[-1]
                    progress_callback({"global_step": self.state.global_step, "max_steps": self.state.max_steps, "epoch": epoch, "loss": last.get("loss"),})
                return log_result

        from transformers import default_data_collator
        trainer = ProgressTrainer(model=model, args=training_args, train_dataset=tokenized, data_collator=default_data_collator,)
        trainer.train()
        trainer.save_model(str(out_dir))
        tokenizer.save_pretrained(str(out_dir))

        if job.get("job_id") == job_id:
            job["status"] = "completed"
            job["progress"] = 100
            job["current_epoch"] = epochs
            job["output_dir"] = str(out_dir)
            job["completed_at"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    except Exception as e:
        import traceback
        traceback.print_exc()
        if job.get("job_id") == job_id:
            job["status"] = "failed"
            job["error"] = str(e)
"""
Finetuning LLM thật với Hugging Face Transformers + PEFT (LoRA).
Chạy trong thread, cập nhật job dict (progress, current_epoch, loss, status).
"""

from pathlib import Path
import json
import time
from typing import Dict, List, Any, Optional

# Danh sách model: synapse.model_list (download + web UI)
from synapse.model_list import resolve_hf_id


def resolve_hf_model_id(model_name: str, base_model_id: Optional[str] = None) -> str:
    """Trả về HuggingFace model_id. Ưu tiên base_model_id, không thì tra model_list."""
    if (base_model_id or "").strip():
        return base_model_id.strip()
    return resolve_hf_id(model_name)


def load_raw_data(path: Path) -> List[Dict[str, Any]]:
    """Đọc file json/jsonl thành list dict."""
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Dataset không tồn tại: {path}")
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


def format_sample_to_text(sample: Dict[str, Any], tokenizer: Any) -> str:
    """
    Chuyển 1 mẫu thành 1 chuỗi text để train causal LM.
    Hỗ trợ: messages, instruction/input/output (Alpaca), text.
    """
    if "messages" in sample:
        msgs = sample["messages"]
        if hasattr(tokenizer, "apply_chat_template"):
            return tokenizer.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=False,
            )
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


def run_finetune_sync(job: Dict[str, Any]) -> None:
    """
    Chạy finetuning LoRA thật (blocking). Cập nhật job với progress, current_epoch, loss, status.
    job phải có: job_id, model, dataset (path), epochs, batch_size, learning_rate,
    save_every, output_model_name, checkpoint_dir, system_prompt, max_steps, warmup_steps.
    """
    try:
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            TrainingArguments,
            Trainer,
        )
        from peft import LoraConfig, get_peft_model, TaskType
        from datasets import Dataset as HFDataset
    except ImportError as e:
        job["status"] = "failed"
        job["error"] = (
            f"Thiếu thư viện finetuning: {e}. Cần cài: pip install torch transformers peft datasets accelerate"
        )
        return

    job_id = job.get("job_id", "")
    model_name = job.get("model", "")
    dataset_path = (job.get("dataset") or "").strip()
    epochs = int(job.get("epochs", 3))
    batch_size = int(job.get("batch_size", 4))
    lr_str = (job.get("learning_rate") or "").strip() or "2e-5"
    try:
        learning_rate = float(lr_str)
    except ValueError:
        learning_rate = 2e-5
    save_every = int(job.get("save_every", 1))
    max_steps = int(job.get("max_steps", 0))
    warmup_steps = int(job.get("warmup_steps", 0))
    output_model_name = (job.get("output_model_name") or job.get("output_name") or "").strip()
    checkpoint_dir = (job.get("checkpoint_dir") or "checkpoints").strip()
    base_model_id = (job.get("base_model_id") or "").strip() or None

    try:
        hf_model_id = resolve_hf_model_id(model_name, base_model_id)
    except Exception as e:
        job["status"] = "failed"
        job["error"] = f"Không xác định được base model: {e}"
        return

    path = Path(dataset_path)
    if not path.is_absolute():
        # Thử synapse/data nếu path là tên file
        base = Path("synapse/data")
        if (base / path.name).exists():
            path = base / path.name
        elif (base / path).exists():
            path = base / path
    if not path.exists():
        job["status"] = "failed"
        job["error"] = f"Dataset không tồn tại: {path}"
        return

    job["status"] = "started"
    job["progress"] = 0
    job["current_epoch"] = 0
    job["error"] = None
    job["loss"] = None

    try:
        # Load raw data
        raw_data = load_raw_data(path)
        if not raw_data:
            job["status"] = "failed"
            job["error"] = "Dataset rỗng"
            return

        tokenizer = AutoTokenizer.from_pretrained(hf_model_id, trust_remote_code=True)
        # Pad side: left cho causal LM thường dùng khi generate
        if getattr(tokenizer, "pad_token", None) is None:
            tokenizer.pad_token = tokenizer.eos_token

        texts = []
        for i, sample in enumerate(raw_data):
            try:
                t = format_sample_to_text(sample, tokenizer)
                if t and t.strip():
                    texts.append(t)
            except Exception:
                continue
        if not texts:
            job["status"] = "failed"
            job["error"] = "Không có mẫu hợp lệ sau khi format"
            return

        max_length = min(int(job.get("max_length", 512)), 2048)
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

        def tokenize_fn(examples):
            out = tokenizer(
                examples["text"],
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors=None,
            )
            # labels = bản copy input_ids, padding token thay bằng -100
            labels = []
            for ids in out["input_ids"]:
                lab = [id_ if id_ != pad_id else -100 for id_ in ids]
                labels.append(lab)
            out["labels"] = labels
            return out

        hf_dataset = HFDataset.from_dict({"text": texts})
        tokenized = hf_dataset.map(
            tokenize_fn,
            batched=True,
            remove_columns=["text"],
            desc="Tokenize",
        )

        model = AutoModelForCausalLM.from_pretrained(
            hf_model_id,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
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
                log_result = super()._maybe_log_save_evaluate(
                    tr_loss, model, trial, epoch, ignore_keys_for_eval
                )
                if self.state.log_history:
                    last = self.state.log_history[-1]
                    progress_callback({
                        "global_step": self.state.global_step,
                        "max_steps": self.state.max_steps,
                        "epoch": epoch,
                        "loss": last.get("loss"),
                    })
                return log_result

        from transformers import default_data_collator
        trainer = ProgressTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized,
            data_collator=default_data_collator,
        )
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

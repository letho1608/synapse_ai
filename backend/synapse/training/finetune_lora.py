"""
Dataset loading and sample formatting for fine-tuning (full or LoRA, see
synapse.training.stage_lora).
"""

from __future__ import annotations

from pathlib import Path
import json
from typing import Dict, List, Any


def load_raw_data(path: Path) -> List[Dict[str, Any]]:
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Dataset không tồn tại hoặc không phải là file hợp lệ: {path}")
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


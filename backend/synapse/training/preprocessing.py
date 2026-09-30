"""
Tiền xử lý dataset cho training pipeline.
Lọc rác, deduplicate, validate, format chuẩn hoá.
"""

from __future__ import annotations

import hashlib
import json
from typing import Dict, List, Any, Tuple


def filter_empty_samples(
    samples: List[Dict[str, Any]],
    format_type: str = "auto",
) -> Tuple[List[Dict[str, Any]], int]:
    """Loại bỏ mẫu rỗng hoặc chỉ chứa whitespace."""
    clean = []
    removed = 0
    for s in samples:
        if format_type == "messages":
            msgs = s.get("messages", [])
            if not msgs:
                removed += 1
                continue
            has_content = any((m.get("content") or "").strip() for m in msgs)
            if not has_content:
                removed += 1
                continue
        elif format_type == "alpaca":
            inst = (s.get("instruction") or "").strip()
            out = (s.get("output") or "").strip()
            if not inst and not out:
                removed += 1
                continue
        elif format_type == "text":
            txt = (s.get("text") or "").strip()
            if not txt:
                removed += 1
                continue
        else:
            txt = json.dumps(s, ensure_ascii=False).strip()
            if not txt or txt == "{}":
                removed += 1
                continue
        clean.append(s)
    return clean, removed


def filter_by_char_length(
    samples: List[Dict[str, Any]],
    format_type: str = "auto",
    min_chars: int = 10,
    max_chars: int = 100000,
) -> Tuple[List[Dict[str, Any]], int]:
    """Lọc mẫu theo độ dài ký tự."""
    clean = []
    removed = 0
    for s in samples:
        txt = _sample_to_text(s, format_type)
        length = len(txt)
        if length < min_chars or length > max_chars:
            removed += 1
            continue
        clean.append(s)
    return clean, removed


def deduplicate_samples(
    samples: List[Dict[str, Any]],
    format_type: str = "auto",
) -> Tuple[List[Dict[str, Any]], int]:
    """Loại bỏ mẫu trùng lặp (hash nội dung)."""
    seen = set()
    clean = []
    removed = 0
    for s in samples:
        txt = _sample_to_text(s, format_type)
        key = hashlib.md5(txt.encode("utf-8", errors="ignore")).hexdigest()
        if key in seen:
            removed += 1
            continue
        seen.add(key)
        clean.append(s)
    return clean, removed


def detect_format(samples: List[Dict[str, Any]]) -> str:
    """Tự động phát hiện format dataset."""
    formats = set()
    for s in samples:
        if not isinstance(s, dict):
            continue
        if "messages" in s:
            formats.add("messages")
        if "instruction" in s or "output" in s:
            formats.add("alpaca")
        if "text" in s:
            formats.add("text")
    if "messages" in formats:
        return "messages"
    if "alpaca" in formats:
        return "alpaca"
    if "text" in formats:
        return "text"
    return "json"


def validate_samples(
    samples: List[Dict[str, Any]],
    format_type: str = "auto",
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Validate & trả về (valid, errors)."""
    fmt = format_type if format_type != "auto" else detect_format(samples)
    valid = []
    errors = []
    for i, s in enumerate(samples):
        errs = []
        if fmt == "messages":
            msgs = s.get("messages")
            if not isinstance(msgs, list) or not msgs:
                errs.append("messages phải là list khác rỗng")
            else:
                for j, m in enumerate(msgs):
                    if not isinstance(m, dict):
                        errs.append(f"messages[{j}] không phải object")
                    elif "role" not in m or "content" not in m:
                        errs.append(f"messages[{j}] thiếu role hoặc content")
        elif fmt == "alpaca":
            if "instruction" not in s and "output" not in s:
                errs.append("thiếu instruction hoặc output")
        elif fmt == "text":
            if "text" not in s:
                errs.append("thiếu text")
        if errs:
            errors.append({"index": i, "errors": errs})
        else:
            valid.append(s)
    return valid, errors


def preprocess(
    samples: List[Dict[str, Any]],
    format_type: str = "auto",
    min_chars: int = 10,
    max_chars: int = 100000,
    dedup: bool = True,
    skip_validation: bool = False,
) -> Dict[str, Any]:
    """Tiền xử lý toàn bộ pipeline. Trả về dict summary."""
    fmt = format_type if format_type != "auto" else detect_format(samples)
    total_original = len(samples)
    summary = {
        "original_count": total_original,
        "format_detected": fmt,
        "empty_removed": 0,
        "short_removed": 0,
        "long_removed": 0,
        "duplicate_removed": 0,
        "validation_errors": 0,
        "valid_count": 0,
        "processed_samples": [],
    }

    if not skip_validation and total_original <= 10000:
        samples, validation_errors = validate_samples(samples, fmt)
        summary["validation_errors"] = len(validation_errors)
        if validation_errors:
            summary["error_details"] = validation_errors[:50]

    samples, removed = filter_empty_samples(samples, fmt)
    summary["empty_removed"] = removed

    if min_chars > 0:
        samples, removed = filter_by_char_length(samples, fmt, min_chars, max_chars)
        summary["short_removed"] = removed

    if dedup:
        samples, removed = deduplicate_samples(samples, fmt)
        summary["duplicate_removed"] = removed

    summary["valid_count"] = len(samples)
    summary["processed_samples"] = samples
    summary["total_removed"] = total_original - len(samples)

    return summary


async def preprocess_async(
    samples: List[Dict[str, Any]],
    format_type: str = "auto",
    min_chars: int = 10,
    max_chars: int = 100000,
    dedup: bool = True,
    skip_validation: bool = False,
) -> Dict[str, Any]:
    """Async version - runs CPU-bound preprocessing in thread executor."""
    import asyncio
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None, preprocess, samples, format_type, min_chars, max_chars, dedup, skip_validation
    )


def _sample_to_text(sample: Dict[str, Any], format_type: str = "auto") -> str:
    if format_type == "messages":
        msgs = sample.get("messages", [])
        parts = []
        for m in msgs:
            role = m.get("role", "")
            content = m.get("content") or ""
            parts.append(f"[{role}]{content}")
        return "\n".join(parts)
    if format_type == "alpaca":
        return f"{sample.get('instruction','')}\n{sample.get('output','')}"
    if format_type == "text":
        return sample.get("text", "")
    return json.dumps(sample, ensure_ascii=False, sort_keys=True)
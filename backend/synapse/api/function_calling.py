"""
Function Calling: parse tool call từ output của model, thực thi, trả về text kết quả để đưa lại vào context.
Format model cần output khi muốn gọi tool:
- Single: TOOL_CALL: {"name": "get_weather", "arguments": {"location": "Hanoi"}}
- Multiple: TOOL_CALLS: [{"name": "get_weather", "arguments": {...}}, {"name": "get_stock_price", "arguments": {...}}]
"""
import asyncio
import json
import re
from typing import List, Optional, Tuple

from synapse.api.tools.registry import execute_tool, get_tool_definitions


# Regex để bắt single TOOL_CALL: {...}
TOOL_CALL_PATTERN = re.compile(r"TOOL_CALL\s*:\s*(\{.*?\})", re.DOTALL)
# Regex để bắt multiple TOOL_CALLS: [...]
TOOL_CALLS_PATTERN = re.compile(r"TOOL_CALLS\s*:\s*(\[.*?\])", re.DOTALL | re.MULTILINE)


def build_tool_instructions() -> str:
    """Tạo đoạn text hướng dẫn model cách gọi tool (thêm vào system prompt)."""
    tools = get_tool_definitions()
    lines = [
        "Bạn có quyền gọi các công cụ (tools) để lấy thông tin thời gian thực. Khi cần thông tin hiện tại (thời gian, thời tiết, giá cổ phiếu, tỷ giá, tin tức, tìm kiếm web, tính toán, giá vàng, crypto, dịch, ngày lễ), hãy gọi tool bằng cách xuất ra đúng một dòng theo format sau:",
        "Nếu cần 1 tool: TOOL_CALL: {\"name\": \"tên_tool\", \"arguments\": {\"thamsố\": \"giá trị\"}}",
        "Nếu cần nhiều tools cùng lúc: TOOL_CALLS: [{\"name\": \"tool1\", \"arguments\": {...}}, {\"name\": \"tool2\", \"arguments\": {...}}]",
        "Ví dụ single: TOOL_CALL: {\"name\": \"get_weather\", \"arguments\": {\"location\": \"Hanoi\"}}",
        "Ví dụ multiple: TOOL_CALLS: [{\"name\": \"get_weather\", \"arguments\": {\"location\": \"Hanoi\"}}, {\"name\": \"get_weather\", \"arguments\": {\"location\": \"Ho Chi Minh City\"}}]",
        "Sau khi gọi tool(s), bạn sẽ nhận được kết quả và cần trả lời user dựa trên kết quả đó. Chỉ xuất TOOL_CALL/TOOL_CALLS khi thực sự cần dữ liệu thời gian thực; nếu không cần thì trả lời bình thường.",
        "Danh sách tools: " + ", ".join(t["function"]["name"] for t in tools) + ".",
    ]
    return "\n".join(lines)


def parse_tool_call(model_output: str) -> Optional[Tuple[str, dict]]:
    """
    Parse output của model, tìm TOOL_CALL (single). Trả về (name, arguments) hoặc None.
    """
    if not model_output or "TOOL_CALL" not in model_output:
        return None
    m = TOOL_CALL_PATTERN.search(model_output)
    if not m:
        return None
    try:
        raw = m.group(1).strip()
        obj = json.loads(raw)
        name = obj.get("name")
        args = obj.get("arguments")
        if isinstance(args, str):
            args = json.loads(args) if args.strip() else {}
        if not name or not isinstance(args, dict):
            return None
        return (name, args)
    except (json.JSONDecodeError, TypeError):
        return None


def parse_tool_calls(model_output: str) -> List[Tuple[str, dict]]:
    """
    Parse output của model, tìm TOOL_CALLS (multiple). Trả về list [(name, arguments), ...] hoặc [].
    Nếu không tìm thấy TOOL_CALLS, fallback về parse single TOOL_CALL.
    """
    if not model_output or "TOOL_CALL" not in model_output:
        return []
    
    # Thử parse multiple calls trước
    m = TOOL_CALLS_PATTERN.search(model_output)
    if m:
        try:
            raw = m.group(1).strip()
            calls = json.loads(raw)
            if isinstance(calls, list):
                result = []
                for call in calls:
                    name = call.get("name")
                    args = call.get("arguments", {})
                    if isinstance(args, str):
                        args = json.loads(args) if args.strip() else {}
                    if name and isinstance(args, dict):
                        result.append((name, args))
                if result:
                    return result
        except (json.JSONDecodeError, TypeError):
            pass
    
    # Fallback: parse single call
    single = parse_tool_call(model_output)
    return [single] if single else []


async def run_tool_and_format(name: str, arguments: dict) -> str:
    """Thực thi tool và trả về chuỗi mô tả kết quả để đưa vào context cho model."""
    result = await execute_tool(name, arguments)
    return json.dumps(result, ensure_ascii=False)


async def run_tools_parallel(tool_calls: List[Tuple[str, dict]]) -> str:
    """
    Thực thi nhiều tools song song và trả về JSON array kết quả.
    """
    if not tool_calls:
        return "[]"
    
    # Tạo tasks cho tất cả tools
    tasks = [execute_tool(name, args) for name, args in tool_calls]
    
    # Chạy song song với asyncio.gather
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Format kết quả: nếu exception thì trả về error dict
    formatted_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            formatted_results.append({
                "success": False,
                "error": str(result),
                "tool": tool_calls[i][0] if i < len(tool_calls) else "unknown"
            })
        else:
            formatted_results.append(result if isinstance(result, dict) else {"success": True, "result": result})
    
    return json.dumps(formatted_results, ensure_ascii=False)

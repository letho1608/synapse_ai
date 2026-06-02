"""
Tool Registry: định nghĩa tất cả tools (OpenAI-style) và thực thi.
Model sẽ được gửi tool definitions; khi model trả về tool call, executor gọi hàm tương ứng.
"""
import asyncio
import json
from typing import Any, Dict, List, Optional

from synapse.api.tools import time_tool, weather_tool, calculator_tool, stock_tool
from synapse.api.tools import currency_tool, news_tool, search_tool, gold_tool, crypto_tool, translate_tool, holidays_tool

# Map tool name -> async function
_TOOL_FUNCTIONS = {
    "get_current_time": time_tool.get_current_time,
    "get_weather": weather_tool.get_weather,
    "calculate": calculator_tool.calculate,
    "get_stock_price": stock_tool.get_stock_price,
    "get_exchange_rate": currency_tool.get_exchange_rate,
    "get_news": news_tool.get_news,
    "search_web": search_tool.search_web,
    "get_gold_price": gold_tool.get_gold_price,
    "get_crypto_price": crypto_tool.get_crypto_price,
    "translate": translate_tool.translate,
    "get_holidays": holidays_tool.get_holidays,
}

# Định nghĩa tools theo chuẩn OpenAI (để gửi cho model trong system prompt)
TOOL_DEFINITIONS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Lấy thời gian và ngày hiện tại theo múi giờ. Dùng khi user hỏi mấy giờ, hôm nay ngày mấy, thời gian ở đâu.",
            "parameters": {"type": "object", "properties": {"timezone": {"type": "string", "description": "Múi giờ, ví dụ Asia/Ho_Chi_Minh, UTC"}}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Lấy thời tiết hiện tại tại một địa điểm (thành phố).",
            "parameters": {"type": "object", "properties": {"location": {"type": "string", "description": "Tên thành phố, ví dụ Hanoi, Ho Chi Minh City"}, "units": {"type": "string", "enum": ["metric", "imperial"], "description": "metric = Celsius"}}, "required": ["location"]},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Tính toán biểu thức toán học (số, + - * /, sin, cos, sqrt, ...).",
            "parameters": {"type": "object", "properties": {"expression": {"type": "string", "description": "Biểu thức cần tính"}}, "required": ["expression"]},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": "Lấy giá cổ phiếu hiện tại (hoặc phiên gần nhất). Mã ví dụ: AAPL, VNM, VIC.",
            "parameters": {"type": "object", "properties": {"symbol": {"type": "string"}, "exchange": {"type": "string"}}, "required": ["symbol"]},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_exchange_rate",
            "description": "Lấy tỷ giá giữa hai loại tiền tệ (USD, VND, EUR, ...).",
            "parameters": {"type": "object", "properties": {"from_currency": {"type": "string"}, "to_currency": {"type": "string", "description": "Mặc định VND"}}, "required": ["from_currency"]},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_news",
            "description": "Lấy tin tức mới nhất (RSS). Có thể lọc theo chủ đề hoặc nguồn (vnexpress, dantri, ...).",
            "parameters": {"type": "object", "properties": {"topic": {"type": "string"}, "source": {"type": "string"}, "limit": {"type": "integer"}}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Tìm kiếm thông tin trên web. Dùng khi cần thông tin mới nhất hoặc không chắc chắn.",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}, "num_results": {"type": "integer"}}, "required": ["query"]},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_gold_price",
            "description": "Lấy giá vàng (Việt Nam, SJC, ...).",
            "parameters": {"type": "object", "properties": {"unit": {"type": "string"}}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_crypto_price",
            "description": "Lấy giá cryptocurrency (BTC, ETH, USDT, ...).",
            "parameters": {"type": "object", "properties": {"symbol": {"type": "string"}}, "required": ["symbol"]},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "translate",
            "description": "Dịch văn bản sang ngôn ngữ khác.",
            "parameters": {"type": "object", "properties": {"text": {"type": "string"}, "from_lang": {"type": "string"}, "to_lang": {"type": "string"}}, "required": ["text", "to_lang"]},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_holidays",
            "description": "Kiểm tra ngày lễ (VN) trong ngày chỉ định hoặc hôm nay.",
            "parameters": {"type": "object", "properties": {"country": {"type": "string"}, "date": {"type": "string", "description": "YYYY-MM-DD"}}, "required": []},
        },
    },
]


def get_tool_definitions() -> List[Dict[str, Any]]:
    return list(TOOL_DEFINITIONS)


async def execute_tool(name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Gọi tool theo tên với arguments (dict). Trả về dict kết quả."""
    if name not in _TOOL_FUNCTIONS:
        return {"success": False, "error": f"Tool không tồn tại: {name}"}
    fn = _TOOL_FUNCTIONS[name]
    import inspect
    sig = inspect.signature(fn)
    valid = {k: v for k, v in (arguments or {}).items() if k in sig.parameters}
    try:
        result = await asyncio.wait_for(fn(**valid), timeout=30.0)
        return result if isinstance(result, dict) else {"success": True, "result": result}
    except asyncio.TimeoutError:
        return {"success": False, "error": "Timeout khi thực thi tool"}
    except TypeError as e:
        return {"success": False, "error": f"Tham số không hợp lệ: {e}"}
    except Exception as e:
        return {"success": False, "error": str(e)}

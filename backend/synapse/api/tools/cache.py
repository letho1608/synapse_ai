"""Caching system cho tools để giảm số lượng API calls."""
import json
import time
from typing import Any, Dict, Optional

# Cache storage: {key: (timestamp, result)}
_cache: Dict[str, tuple[float, Any]] = {}

# Cache durations (seconds) cho từng tool
CACHE_DURATIONS = {
    "get_weather": 600,          # 10 phút
    "get_stock_price": 120,      # 2 phút
    "get_exchange_rate": 3600,   # 1 giờ
    "get_crypto_price": 120,      # 2 phút
    "get_news": 300,             # 5 phút
    "search_web": 1800,          # 30 phút
    "get_gold_price": 600,       # 10 phút
    "translate": 86400,          # 24 giờ (dịch không đổi)
    "get_holidays": 86400,       # 24 giờ (ngày lễ không đổi)
    "get_current_time": 0,       # Không cache (luôn cần thời gian mới nhất)
    "calculate": 0,              # Không cache (tính toán động)
}


def get_cache_key(tool_name: str, **kwargs) -> str:
    """Tạo cache key từ tool name và arguments."""
    # Sort keys để đảm bảo thứ tự nhất quán
    sorted_kwargs = json.dumps(kwargs, sort_keys=True, ensure_ascii=False)
    return f"{tool_name}:{sorted_kwargs}"


def get_cached(tool_name: str, **kwargs) -> Optional[Any]:
    """Lấy kết quả từ cache nếu còn hợp lệ."""
    cache_duration = CACHE_DURATIONS.get(tool_name, 300)  # Default 5 phút
    if cache_duration == 0:
        return None  # Tool này không cache
    
    key = get_cache_key(tool_name, **kwargs)
    if key in _cache:
        cached_time, result = _cache[key]
        age = time.time() - cached_time
        if age < cache_duration:
            return result
        else:
            # Cache hết hạn, xóa
            del _cache[key]
    return None


def set_cache(tool_name: str, result: Any, **kwargs) -> None:
    """Lưu kết quả vào cache."""
    cache_duration = CACHE_DURATIONS.get(tool_name, 300)
    if cache_duration == 0:
        return  # Tool này không cache
    
    key = get_cache_key(tool_name, **kwargs)
    _cache[key] = (time.time(), result)
    
    # Cleanup cache cũ (giữ tối đa 1000 entries)
    if len(_cache) > 1000:
        # Xóa 20% entries cũ nhất
        sorted_items = sorted(_cache.items(), key=lambda x: x[1][0])
        for old_key, _ in sorted_items[:200]:
            del _cache[old_key]


def clear_cache(tool_name: Optional[str] = None) -> None:
    """Xóa cache. Nếu tool_name=None thì xóa tất cả."""
    if tool_name is None:
        _cache.clear()
    else:
        keys_to_delete = [k for k in _cache.keys() if k.startswith(f"{tool_name}:")]
        for key in keys_to_delete:
            del _cache[key]

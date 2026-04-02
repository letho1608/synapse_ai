"""Rate limiting cho các APIs nhạy cảm."""
import asyncio
import time
from collections import deque
from typing import Dict

# Rate limiters cho từng tool
_rate_limiters: Dict[str, deque] = {}


def get_rate_limiter(tool_name: str, max_calls: int = 10, window_seconds: int = 60) -> deque:
    """Lấy hoặc tạo rate limiter cho tool."""
    if tool_name not in _rate_limiters:
        _rate_limiters[tool_name] = deque(maxlen=max_calls)
    return _rate_limiters[tool_name]


async def wait_if_needed(tool_name: str, max_calls: int = 10, window_seconds: int = 60, min_interval: float = 1.0) -> None:
    """
    Chờ nếu cần để tuân thủ rate limit.
    max_calls: số calls tối đa trong window_seconds
    min_interval: khoảng thời gian tối thiểu giữa các calls (seconds)
    """
    limiter = get_rate_limiter(tool_name, max_calls, window_seconds)
    now = time.time()
    
    # Xóa các calls cũ hơn window_seconds
    while limiter and now - limiter[0] > window_seconds:
        limiter.popleft()
    
    # Nếu đã đạt max_calls, chờ đến khi có slot
    if len(limiter) >= max_calls:
        oldest_call = limiter[0]
        wait_time = window_seconds - (now - oldest_call) + 0.1  # Thêm 0.1s buffer
        if wait_time > 0:
            await asyncio.sleep(wait_time)
            now = time.time()
    
    # Đảm bảo khoảng cách tối thiểu giữa các calls
    if limiter and min_interval > 0:
        last_call = limiter[-1] if limiter else 0
        time_since_last = now - last_call
        if time_since_last < min_interval:
            await asyncio.sleep(min_interval - time_since_last)
        now = time.time()
    
    # Ghi lại call này
    limiter.append(now)

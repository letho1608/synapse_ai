"""
Trigger tool theo từ khóa khi model nhỏ không tuân thủ TOOL_CALL.
Nếu tin nhắn user cuối có ý hỏi thời tiết, thời gian, ... thì gọi tool trước và inject kết quả vào context.
"""
import re
from typing import List, Tuple

# Các rule: (pattern regex, tool_name, dict để build arguments từ match)
# location/place có thể lấy từ group hoặc mặc định
def _weather_args(user_text: str) -> dict:
    return _extract_location(user_text)


INTENT_RULES = [
    # Thời tiết: thời tiết, weather, nhiệt độ, có mưa, ...
    (r"thời\s*tiết|weather|nhiệt\s*độ|mưa|nắng|độ\s*ẩm", "get_weather", _weather_args),
    # Thời gian: mấy giờ, thời điểm hiện tại, hôm nay ngày mấy, ...
    (r"mấy\s*giờ|thời\s*điểm\s*hiện\s*tại|bây\s*giờ|hôm\s*nay\s*ngày|giờ\s*hiện\s*tại|ngày\s*mấy|mấy\s*giờ\s*rồi", "get_current_time", lambda t: {}),
]

# Tên địa điểm thường gặp (chuẩn hóa cho API)
LOCATION_ALIASES = {
    "hà nội": "Hanoi", "hanoi": "Hanoi", "hn": "Hanoi",
    "sài gòn": "Ho Chi Minh City", "sài gọn": "Ho Chi Minh City",
    "hồ chí minh": "Ho Chi Minh City", "hcm": "Ho Chi Minh City", "sg": "Ho Chi Minh City",
    "đà nẵng": "Da Nang", "da nang": "Da Nang",
    "hải phòng": "Hai Phong", "hai phong": "Hai Phong",
    "cần thơ": "Can Tho", "can tho": "Can Tho",
}


def _extract_location(user_text: str) -> dict:
    """Lấy location từ câu user. Nếu không thấy thì mặc định Hanoi."""
    t = user_text.lower().strip()
    for alias, canonical in LOCATION_ALIASES.items():
        if alias in t:
            return {"location": canonical, "units": "metric"}
    # Mặc định
    return {"location": "Hanoi", "units": "metric"}


def detect_tool_intent(last_user_message: str) -> List[Tuple[str, dict]]:
    """
    Kiểm tra tin nhắn user cuối. Nếu có ý hỏi thời tiết/thời gian thì trả về [(tool_name, args), ...].
    """
    if not last_user_message or not isinstance(last_user_message, str):
        return []
    text = last_user_message.strip()
    if len(text) < 3:
        return []
    results = []
    seen = set()
    for pattern, tool_name, args_builder in INTENT_RULES:
        if tool_name in seen:
            continue
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            seen.add(tool_name)
            try:
                args = args_builder(text) if callable(args_builder) else {}
                if isinstance(args, dict):
                    results.append((tool_name, args))
            except Exception:
                pass
    return results

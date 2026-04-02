"""Thời gian hiện tại theo múi giờ. Không cần API."""
from datetime import datetime
from typing import Any, Dict

try:
    import pytz
    HAS_PYTZ = True
except ImportError:
    HAS_PYTZ = False


async def get_current_time(timezone: str = "Asia/Ho_Chi_Minh") -> Dict[str, Any]:
    """
    Lấy thời gian và ngày hiện tại theo múi giờ.
    timezone: ví dụ Asia/Ho_Chi_Minh, UTC, America/New_York
    """
    try:
        if HAS_PYTZ:
            tz = pytz.timezone(timezone)
            now = datetime.now(tz)
        else:
            now = datetime.utcnow()
            timezone = "UTC (pytz chưa cài, dùng UTC)"
        return {
            "success": True,
            "timezone": timezone,
            "datetime": now.strftime("%Y-%m-%d %H:%M:%S"),
            "date": now.strftime("%Y-%m-%d"),
            "time": now.strftime("%H:%M:%S"),
            "weekday": now.strftime("%A"),
            "weekday_vi": _weekday_vi(now.weekday()),
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def _weekday_vi(weekday: int) -> str:
    days = ["Thứ Hai", "Thứ Ba", "Thứ Tư", "Thứ Năm", "Thứ Sáu", "Thứ Bảy", "Chủ Nhật"]
    return days[weekday] if 0 <= weekday < 7 else ""

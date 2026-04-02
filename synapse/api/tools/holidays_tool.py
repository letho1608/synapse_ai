"""Ngày lễ (VN). Dữ liệu tĩnh."""
from datetime import datetime
from typing import Any, Dict, List

# Một số ngày lễ VN (có thể bổ sung)
VN_HOLIDAYS = [
    ("01-01", "Tết Dương lịch"),
    ("30-04", "Giải phóng miền Nam"),
    ("01-05", "Quốc tế Lao động"),
    ("02-09", "Quốc khánh"),
    ("01-01", "Tết Nguyên đán"),  # Âm lịch - cần tính theo năm
]


async def get_holidays(country: str = "VN", date: str = None) -> Dict[str, Any]:
    """
    Kiểm tra ngày lễ. date: YYYY-MM-DD hoặc None (hôm nay).
    country: VN (mặc định).
    """
    try:
        if date:
            d = datetime.strptime(date, "%Y-%m-%d")
        else:
            d = datetime.now()
        md = d.strftime("%m-%d")
        found = [name for (day, name) in VN_HOLIDAYS if day == md]
        return {
            "success": True,
            "date": d.strftime("%Y-%m-%d"),
            "country": country,
            "is_holiday": len(found) > 0,
            "holidays": found,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

"""Thời tiết hiện tại. Dùng wttr.in (free, không cần key)."""
import asyncio
import aiohttp
from typing import Any, Dict

from synapse.api.tools.cache import get_cached, set_cache


async def get_weather(location: str, units: str = "metric") -> Dict[str, Any]:
    """
    Lấy thời tiết hiện tại cho địa điểm.
    location: tên thành phố (ví dụ Hanoi, Ho Chi Minh City).
    units: metric (Celsius) hoặc imperial (Fahrenheit).
    """
    # Kiểm tra cache trước
    cached = get_cached("get_weather", location=location, units=units)
    if cached is not None:
        return cached
    
    try:
        # wttr.in trả JSON với ?format=j1
        url = f"https://wttr.in/{location}?format=j1"
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status != 200:
                    return {"success": False, "error": f"HTTP {resp.status}"}
                data = await resp.json()
        current = data.get("current_condition", [{}])[0]
        temp = current.get("temp_C" if units == "metric" else "temp_F", "N/A")
        feels = current.get("FeelsLikeC" if units == "metric" else "FeelsLikeF", "N/A")
        desc = current.get("weatherDesc", [{}])[0].get("value", "N/A")
        humidity = current.get("humidity", "N/A")
        wind = current.get("windspeedKmph", "N/A")
        result = {
            "success": True,
            "location": location,
            "temperature": temp,
            "feels_like": feels,
            "description": desc,
            "humidity": humidity,
            "wind_kmph": wind,
            "unit": "°C" if units == "metric" else "°F",
        }
        # Lưu vào cache
        set_cache("get_weather", result, location=location, units=units)
        return result
    except asyncio.TimeoutError:
        return {"success": False, "error": "Timeout khi lấy thời tiết"}
    except Exception as e:
        return {"success": False, "error": str(e)}

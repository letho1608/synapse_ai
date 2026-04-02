"""Tỷ giá ngoại tệ. Dùng API free không cần key."""
import aiohttp
from typing import Any, Dict

from synapse.api.tools.cache import get_cached, set_cache


async def get_exchange_rate(from_currency: str, to_currency: str = "VND") -> Dict[str, Any]:
    """
    Lấy tỷ giá giữa hai loại tiền tệ.
    from_currency: USD, EUR, JPY, ...
    to_currency: VND, USD, ...
    """
    # Kiểm tra cache trước
    cached = get_cached("get_exchange_rate", from_currency=from_currency, to_currency=to_currency)
    if cached is not None:
        return cached
    
    try:
        from_currency = from_currency.upper().strip()
        to_currency = to_currency.upper().strip()
        url = f"https://api.exchangerate-api.com/v4/latest/{from_currency}"
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status != 200:
                    return {"success": False, "error": f"HTTP {resp.status}"}
                data = await resp.json()
        rates = data.get("rates", {})
        if to_currency not in rates:
            return {"success": False, "error": f"Không hỗ trợ mã {to_currency}"}
        rate = rates[to_currency]
        result = {
            "success": True,
            "from_currency": from_currency,
            "to_currency": to_currency,
            "rate": rate,
            "message": f"1 {from_currency} = {rate} {to_currency}",
        }
        # Lưu vào cache
        set_cache("get_exchange_rate", result, from_currency=from_currency, to_currency=to_currency)
        return result
    except Exception as e:
        return {"success": False, "error": str(e)}
